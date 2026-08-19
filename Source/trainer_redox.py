"""Cross-validation training of multi-fidelity redox potential models.

ModelTrainer runs K-fold training of a Lightning model, restores the best
checkpoint of every fold, computes per-fold and ensemble metrics, and writes
predictions, metrics and loss curves to disk. ModelShell wraps the per-fold
checkpoints into an ensemble whose prediction is the mean over folds.

Early stopping and checkpoint selection monitor the validation loss computed on
the experimental records only whenever such records are present, so that model
selection is driven by high-fidelity data rather than by the calculated part of
the multi-fidelity set.

Output layout under the folder passed as output_folder:
    model_structure.json, model_config.torch
    models/fold_<n>/best_model.pt, losses.json
    metrics/fold_<n>_metrics.json, crossval_summary.json, metrics_table.csv,
        ensemble_test_metrics.json, solvent_metrics.csv
    predictions/fold_<n>/{train,val,val_exp}_predictions.csv,
        predictions/test_predictions.csv
"""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning
import torch
import torch_geometric
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

# Row order of the exported prediction tables: oxidation, reduction, then the rest
REDOX_ORDER = {"ox": 0, "red": 1}
UNKNOWN_SOLVENT = "unknown"
PREDICTION_BATCH_SIZE = 64


def _target_values(batch, target_name: str) -> np.ndarray:
    """Return the ground truth of one target as a NumPy array."""
    y = batch.y
    values = y[target_name] if isinstance(y, dict) else y
    return values.detach().cpu().numpy()


def _batch_solvents(batch, n_graphs: int) -> list[str]:
    """Return the solvent of every graph in a batch, or a placeholder if absent."""
    if not hasattr(batch, "solvent_smiles"):
        return [UNKNOWN_SOLVENT] * n_graphs
    solvents = batch.solvent_smiles
    if isinstance(solvents, list):
        return list(solvents)
    return [str(s) for s in solvents.cpu().numpy()]


def write_predictions_csv(data_list, model, path, device="cpu", target_name="E"):
    """Run inference over data_list and save a per-record prediction table.

    Columns: SMILES, solvent, redox type, data type, ground truth, prediction
    and error. Rows are ordered by redox type and then by target value.
    """
    data_list = list(data_list)
    model.eval()
    model.to(device)
    rows = []

    with torch.no_grad():
        for start in range(0, len(data_list), PREDICTION_BATCH_SIZE):
            chunk = data_list[start:start + PREDICTION_BATCH_SIZE]
            batch = Batch.from_data_list(chunk).to(device)
            predictions = model(batch)[target_name].cpu().numpy().reshape(len(chunk), -1).squeeze(-1)
            for record, prediction in zip(chunk, np.atleast_1d(predictions)):
                y_true = record.y[target_name].item() if isinstance(record.y, dict) else record.y.item()
                rows.append({
                    "smiles": getattr(record, "smiles", "NA"),
                    "solvent_smiles": getattr(record, "solvent_smiles", "NA"),
                    "redox_type": getattr(record, "redox_type", "NA"),
                    "data_type": getattr(record, "data_type", "NA"),
                    "target": y_true,
                    "prediction": float(prediction),
                    "error": float(prediction) - y_true,
                })

    rows.sort(key=lambda r: (REDOX_ORDER.get(r["redox_type"], 2), r["target"]))
    pd.DataFrame(rows).to_csv(path, index=False)


class ModelShell(nn.Module):
    """Ensemble of per-fold models; outputs the mean prediction across all folds."""

    def __init__(self, model_class, train_folder, device=torch.device("cpu")):
        super().__init__()
        self.models = []
        self.device = device

        train_folder = Path(train_folder)
        config = torch.load(
            train_folder / "model_config.torch", map_location="cpu", weights_only=False
        )
        base_model = model_class(**config)

        # Sorted numerically so that fold_10 follows fold_9 rather than fold_1
        fold_folders = [p for p in (train_folder / "models").iterdir()
                        if p.name.startswith("fold_")]
        for folder in sorted(fold_folders, key=lambda p: int(p.name.split("_")[-1])):
            state_dict = torch.load(
                folder / "best_model.pt", map_location=device, weights_only=False
            )
            model = copy.deepcopy(base_model)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            self.models.append(model)

        logger.info("Loaded ensemble of %d fold models from %s", len(self.models), train_folder)

    def forward(self, *args, **kwargs):
        all_pred = [model(*args, **kwargs) for model in self.models]
        return {
            name: torch.stack([pred[name] for pred in all_pred], dim=-1).mean(dim=-1)
            for name in all_pred[0]
        }


class ModelTrainer:
    """K-fold trainer producing per-fold models, metrics and prediction tables."""

    def __init__(self, model, train_valid_data, test_data=None, output_folder=None,
                 es_patience=20, epochs=1000, save_to_folder=True, seed=42,
                 targets=(), experiment_logger=None):
        """
        Args:
            model: Lightning model instance, deep-copied for every fold.
            train_valid_data: list of (train_loader, val_loader) tuples.
            test_data: DataLoader over the held-out test set, or None.
            output_folder: root folder for all artefacts.
            es_patience: early stopping patience in epochs.
            epochs: maximum number of epochs per fold.
            save_to_folder: write artefacts to disk.
            seed: random seed applied to Lightning and PyTorch Geometric.
            targets: target configuration providing name and metrics.
            experiment_logger: optional ExperimentLogger for human-readable logs.
        """
        pytorch_lightning.seed_everything(seed)
        torch_geometric.seed_everything(seed)

        self.initial_model = model
        self.models = []
        self.train_valid_data = train_valid_data
        self.test_data = test_data
        self.es_patience = es_patience
        self.epochs = epochs
        self.targets = targets
        self.save_to_folder = save_to_folder
        self.results_dict = {}
        self.seed = seed
        self.logger = experiment_logger

        self.main_folder = Path(output_folder) if output_folder else Path("output/default")
        self.models_folder = self.main_folder / "models"
        self.metrics_folder = self.main_folder / "metrics"
        self.predictions_folder = self.main_folder / "predictions"

    # ------------------------------------------------------------------
    # Folder and config setup
    # ------------------------------------------------------------------

    def prepare_out_folder(self):
        """Create the output tree and store the model structure and config."""
        for folder in (self.models_folder, self.metrics_folder, self.predictions_folder):
            folder.mkdir(parents=True, exist_ok=True)
        for fold in range(len(self.train_valid_data)):
            (self.models_folder / f"fold_{fold + 1}").mkdir(exist_ok=True)
            (self.predictions_folder / f"fold_{fold + 1}").mkdir(exist_ok=True)

        with open(self.main_folder / "model_structure.json", "w") as f:
            json.dump(self.initial_model.get_model_structure(), f, indent=2)
        torch.save(self.initial_model.config, self.main_folder / "model_config.torch")

    # ------------------------------------------------------------------
    # Prediction collection and metrics
    # ------------------------------------------------------------------

    def _collect_from_loader(self, model, loader, device="cpu"):
        """Run a model over a loader and return ground truth, predictions and solvents."""
        model.eval()
        model.to(device)
        target_names = [target["name"] for target in self.targets]
        true_lists = defaultdict(list)
        pred_lists = defaultdict(list)
        solvents: list[str] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                n_graphs = batch.ptr.size(0) - 1
                for name in target_names:
                    true_lists[name].append(
                        _target_values(batch, name).reshape(n_graphs, -1).squeeze(-1)
                    )
                    pred_lists[name].append(
                        out[name].cpu().numpy().reshape(n_graphs, -1).squeeze(-1)
                    )
                solvents.extend(_batch_solvents(batch, n_graphs))

        return (
            {k: np.concatenate(v) for k, v in true_lists.items()},
            {k: np.concatenate(v) for k, v in pred_lists.items()},
            solvents,
        )

    def _compute_metrics(self, true_dict, pred_dict):
        """Evaluate every configured metric of every target."""
        result = {}
        for target in self.targets:
            name = target["name"]
            for metric_name, (metric_fn, params) in target["metrics"].items():
                result[f"{name}_{metric_name}"] = float(
                    metric_fn(true_dict[name], pred_dict[name], **params)
                )
        return result

    def calculate_fold_metrics(self, model, train_loader, val_loader,
                               exp_val_loader=None, device="cpu"):
        """Metrics of one fold on train, validation and experimental validation."""
        metrics = {}
        subsets = [("train", train_loader), ("val", val_loader)]
        if exp_val_loader is not None and len(exp_val_loader.dataset) > 0:
            subsets.append(("val_exp", exp_val_loader))

        for suffix, loader in subsets:
            true_dict, pred_dict, _ = self._collect_from_loader(model, loader, device)
            for key, value in self._compute_metrics(true_dict, pred_dict).items():
                metrics[f"{key}_{suffix}"] = value

        return metrics

    def calculate_test_metrics(self, model, device="cpu"):
        """Metrics of a model on the held-out test set."""
        if self.test_data is None:
            return {}
        true_dict, pred_dict, _ = self._collect_from_loader(model, self.test_data, device)
        return {f"{k}_test": v for k, v in self._compute_metrics(true_dict, pred_dict).items()}

    def calculate_metrics_by_solvent(self, model_shell, test_loader, targets, device="cpu"):
        """Break the test metrics of the first target down by solvent."""
        target_name = targets[0]["name"]
        true_dict, pred_dict, solvents = self._collect_from_loader(model_shell, test_loader, device)
        trues = true_dict[target_name]
        preds = pred_dict[target_name]

        if len(solvents) != len(preds):
            logger.warning(
                "Solvent list length %d does not match %d predictions, "
                "per-solvent metrics skipped", len(solvents), len(preds)
            )
            return {}

        groups = defaultdict(lambda: {"trues": [], "preds": []})
        for true, pred, solvent in zip(trues, preds, solvents):
            if solvent != UNKNOWN_SOLVENT:
                groups[solvent]["trues"].append(true)
                groups[solvent]["preds"].append(pred)

        results = {}
        for solvent, values in groups.items():
            solvent_true = np.array(values["trues"]).flatten()
            solvent_pred = np.array(values["preds"]).flatten()
            if len(solvent_pred) > 0:
                metrics = {"count": len(solvent_pred)}
                for metric_name, (metric_fn, params) in targets[0]["metrics"].items():
                    metrics[metric_name] = float(metric_fn(solvent_true, solvent_pred, **params))
                results[solvent] = metrics

        return results

    # ------------------------------------------------------------------
    # Cross-validation loop
    # ------------------------------------------------------------------

    def train_cv_models(self):
        """Train one model per fold, then evaluate the fold ensemble on the test set."""
        if self.save_to_folder:
            self.prepare_out_folder()

        if self.logger:
            self.logger.log_experiment_start({
                "n_folds": len(self.train_valid_data),
                "epochs": self.epochs,
                "es_patience": self.es_patience,
                "seed": self.seed,
                "output_folder": str(self.main_folder),
            })

        for fold_ind, (train_dataloader, valid_dataloader) in enumerate(self.train_valid_data):
            model = copy.deepcopy(self.initial_model)
            model.metadata["fold_ind"] = fold_ind

            if self.logger:
                self.logger.log_fold_start(
                    fold_num=fold_ind + 1,
                    train_size=len(train_dataloader.dataset),
                    val_size=len(valid_dataloader.dataset),
                )

            self.train_model(model, train_dataloader, valid_dataloader, fold_ind, self.epochs)

        if self.test_data is not None and self.save_to_folder:
            self._evaluate_ensemble()

        self._summarise_cross_validation()

    def _evaluate_ensemble(self):
        """Evaluate the ensemble of fold models on the test set and export its outputs."""
        model_shell = ModelShell(
            model_class=self.initial_model.__class__,
            train_folder=str(self.main_folder),
            device=torch.device("cpu"),
        )

        test_metrics = self.calculate_test_metrics(model_shell)
        self.results_dict["ensemble_test"] = test_metrics
        with open(self.metrics_folder / "ensemble_test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        write_predictions_csv(
            self.test_data.dataset, model_shell,
            str(self.predictions_folder / "test_predictions.csv"),
            target_name=self.targets[0]["name"],
        )

        solvent_metrics = self.calculate_metrics_by_solvent(
            model_shell, self.test_data, self.targets
        )
        if solvent_metrics:
            df = pd.DataFrame.from_dict(solvent_metrics, orient="index")
            df.index.name = "solvent_smiles"
            df.reset_index(inplace=True)
            df.to_csv(self.metrics_folder / "solvent_metrics.csv", index=False)

        logger.info("Ensemble test metrics: %s", test_metrics)
        if self.logger:
            self.logger.main_logger.info("Ensemble test metrics: %s", test_metrics)

    def _summarise_cross_validation(self):
        """Aggregate per-fold metrics into mean and standard deviation."""
        aggregate = defaultdict(list)
        for fold_ind in range(len(self.train_valid_data)):
            fold_metrics = self.results_dict.get(f"fold_{fold_ind + 1}", {})
            for key, value in fold_metrics.items():
                aggregate[key].append(value)

        crossval_summary = {
            key: {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
            for key, values in aggregate.items()
        }
        self.results_dict["crossval_summary"] = crossval_summary

        if self.save_to_folder:
            with open(self.metrics_folder / "crossval_summary.json", "w") as f:
                json.dump(crossval_summary, f, indent=2)
            pd.DataFrame([
                {"Metric": key, "Mean": value["mean"], "Std": value["std"],
                 "Values": value["values"]}
                for key, value in crossval_summary.items()
            ]).to_csv(self.metrics_folder / "metrics_table.csv", index=False)

        if self.logger:
            self.logger.log_experiment_summary({
                "n_folds": len(self.train_valid_data),
                "crossval_metrics": {
                    key: f"{value['mean']:.4f} +/- {value['std']:.4f}"
                    for key, value in crossval_summary.items()
                },
            })

    # ------------------------------------------------------------------
    # Single fold training
    # ------------------------------------------------------------------

    def train_model(self, model, train_dataloader, valid_dataloader,
                    current_fold_num, epochs=1000):
        """Train one fold, restore its best checkpoint and export its outputs."""
        fold_models_folder = self.models_folder / f"fold_{current_fold_num + 1}"
        fold_predictions_folder = self.predictions_folder / f"fold_{current_fold_num + 1}"

        # Records without data_type are treated as experimental; when experimental
        # records are present, model selection is driven by their loss alone
        exp_val_data = [d for d in valid_dataloader.dataset
                        if getattr(d, "data_type", "exp") == "exp"]
        exp_val_loader = (
            DataLoader(exp_val_data, batch_size=valid_dataloader.batch_size, shuffle=False)
            if exp_val_data else None
        )

        if exp_val_loader is None:
            monitor_metric = "val_loss"
            val_loaders = valid_dataloader
        else:
            monitor_metric = "val_loss_exp"
            val_loaders = [valid_dataloader, exp_val_loader]

        es_callback = EarlyStopping(
            patience=self.es_patience, monitor=monitor_metric, mode="min", verbose=True
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(fold_models_folder),
            filename="best_checkpoint",
            monitor=monitor_metric,
            save_top_k=1,
            mode="min",
            save_weights_only=True,
            verbose=False,
        )

        pl_trainer = Trainer(
            callbacks=[es_callback, checkpoint_callback],
            log_every_n_steps=1,
            max_epochs=epochs,
            logger=False,
            accelerator="auto",
            deterministic="warn",
            enable_progress_bar=True,
        )
        pl_trainer.fit(model, train_dataloader, val_loaders)

        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt and Path(best_ckpt).exists():
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt.get("state_dict", ckpt))
        else:
            logger.warning(
                "Fold %d: no checkpoint written, keeping the last-epoch weights",
                current_fold_num + 1
            )

        model.eval()
        self.models.append(model)

        if self.save_to_folder:
            self._export_fold(model, train_dataloader, valid_dataloader, exp_val_data,
                              exp_val_loader, current_fold_num,
                              fold_models_folder, fold_predictions_folder)

    def _export_fold(self, model, train_dataloader, valid_dataloader, exp_val_data,
                     exp_val_loader, current_fold_num,
                     fold_models_folder, fold_predictions_folder):
        """Write weights, losses, metrics and prediction tables of one fold."""
        torch.save(model.state_dict(), fold_models_folder / "best_model.pt")
        with open(fold_models_folder / "losses.json", "w") as f:
            json.dump({"train_loss": model.train_losses,
                       "valid_loss": model.valid_losses}, f, indent=2)

        fold_metrics = self.calculate_fold_metrics(
            model, train_dataloader, valid_dataloader, exp_val_loader
        )
        self.results_dict[f"fold_{current_fold_num + 1}"] = fold_metrics
        with open(self.metrics_folder / f"fold_{current_fold_num + 1}_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

        logger.info("Fold %d metrics: %s", current_fold_num + 1, fold_metrics)
        if self.logger:
            self.logger.log_fold_metrics(
                fold_num=current_fold_num + 1, metrics=fold_metrics, phase="val"
            )

        target_name = self.targets[0]["name"]
        exports = [
            (train_dataloader.dataset, "train_predictions.csv"),
            (valid_dataloader.dataset, "val_predictions.csv"),
        ]
        if exp_val_data:
            exports.append((exp_val_data, "val_exp_predictions.csv"))
        for data_list, filename in exports:
            write_predictions_csv(
                data_list, model, str(fold_predictions_folder / filename),
                target_name=target_name,
            )