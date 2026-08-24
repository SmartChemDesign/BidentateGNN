import numpy as np
import torch
import copy
import json
import os
import pytorch_lightning
import torch_geometric

from torch_geometric.data import Batch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn

class GCNNTrainer:
    def __init__(self, model, train_valid_data, test_data=None, output_folder=None,
                 es_patience=20, epochs=1000, save_to_folder=True, seed=42,
                 targets=(), logger=None):
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

        self.main_folder = output_folder
        self.logger = logger

    def prepare_out_folder(self):
        def create(path):
            if os.path.exists(path) or path == "": return
            head, tail = os.path.split(path)
            create(head)
            os.mkdir(path)

        for fold in range(len(self.train_valid_data)):
            create(os.path.join(self.main_folder, f"fold_{fold + 1}"))
        self.write_model_structure()
        self.save_model_config()

    def write_model_structure(self):
        with open(os.path.join(self.main_folder, "model_structure.json"), "w") as jf:
            structure = self.initial_model.get_model_structure()
            json.dump(structure, jf)

    def save_model_config(self):
        torch.save(
            self.initial_model.config,
            os.path.join(self.main_folder, "model_config.torch")
        )

    def get_true_pred(self):
        def get_mean_pred(batch):
            all_pred = [model(batch) for model in self.models]
            output = {
                target["name"]: torch.cat([
                    pred[target["name"]].unsqueeze(-1) for pred in all_pred
                ], dim=-1).mean(dim=-1)
                for target in self.targets
            }
            return output

        def prepare(pred):
            for target in self.targets:
                key = target["name"]
                if target["mode"] == "binary_classification":
                    pred[key] = (pred[key] > 0.5).to(torch.int32)
                elif target["mode"] == "multiclass_classification":
                    pred[key] = torch.argmax(pred[key], dim=-1)
                pred[key] = pred[key].detach().numpy()
            return pred

        train, val = self.train_valid_data[0]
        train_batch = Batch.from_data_list(train.dataset + val.dataset)
        test_batch = Batch.from_data_list(self.test_data.dataset) if self.test_data else None

        train_pred = prepare(get_mean_pred(train_batch))
        train_true = {k: v.numpy() for k, v in train_batch.y.items()}

        test_pred = prepare(get_mean_pred(test_batch)) if self.test_data else None
        test_true = {k: v.numpy() for k, v in test_batch.y.items()} if self.test_data else None

        valid_batches = [Batch.from_data_list(self.train_valid_data[i][1].dataset) for i in range(len(self.models))]
        valid_preds = [model(batch) for model, batch in zip(self.models, valid_batches)]
        valid_pred = {target["name"]: torch.cat([pred[target["name"]] for pred in valid_preds], dim=0)
                      for target in self.targets}
        valid_pred = prepare(valid_pred)
        valid_true = {
            target["name"]: np.concatenate([batch.y[target["name"]].numpy() for batch in valid_batches])
            for target in self.targets
        }

        return (train_true, valid_true, test_true,
                train_pred, valid_pred, test_pred)

    def calculate_metrics(self):
        train_true, valid_true, test_true, train_pred, valid_pred, test_pred = self.get_true_pred()

        phase_names = ("train", "valid", "test") if self.test_data else ("train", "valid")
        true_values = (train_true, valid_true, test_true) if self.test_data else (train_true, valid_true)
        pred_values = (train_pred, valid_pred, test_pred) if self.test_data else (train_pred, valid_pred)

        results_dict = {}

        for target in self.targets:
            for metric_name in target["metrics"]:
                for phase, true, pred in zip(phase_names, true_values, pred_values):
                    local_true = true[target["name"]]
                    local_pred = pred[target["name"]]

                    metric, params = target["metrics"][metric_name]
                    key = f"{target['name']}_{phase}_{metric_name}"
                    res = metric(local_true, local_pred, **params)
                    results_dict[key] = float(res)

        return results_dict

    def train_cv_models(self):
        """
        Create model, train it with KFold cross-validation and write down metrics
        """
        if self.save_to_folder:
            self.prepare_out_folder()

        for fold_ind, (train_dataloader, valid_dataloader) in enumerate(self.train_valid_data):
            model = copy.deepcopy(self.initial_model)
            model.metadata["fold_ind"] = fold_ind

            self.train_model(model, train_dataloader, valid_dataloader, fold_ind, self.epochs)

        self.results_dict["general"] = self.calculate_metrics()

        if self.logger is not None:
            self.logger.log_metrics(self.results_dict["general"])

        if self.save_to_folder:
            with open(os.path.join(self.main_folder, "metrics.json"), "w") as jf:
                json.dump(self.results_dict["general"], jf)

        with open(os.path.join(self.main_folder, "test_true"), "w") as txtfile:
            txtfile.write(str(self.get_true_pred()[2]))
        with open(os.path.join(self.main_folder, "test_pred"), "w") as txtfile:
            txtfile.write(str(self.get_true_pred()[-1]))

    def train_model(self, model, train_dataloader, valid_dataloader, current_fold_num, epochs=1000):
        """
        Train model on certain fold and write down metrics
        """
        model.train()
        es_callback = EarlyStopping(patience=self.es_patience, monitor="val_loss")
        trainer = Trainer(callbacks=[es_callback], log_every_n_steps=20, max_epochs=epochs, logger=self.logger,
                          accelerator="auto", deterministic="warn")
        trainer.fit(model, train_dataloader, valid_dataloader)

        model.eval()
        self.models.append(model)

        current_folder = os.path.join(self.main_folder, f"fold_{current_fold_num + 1}")
        if self.save_to_folder:
            torch.save(model.state_dict(), os.path.join(current_folder, "best_model"))
            with open(os.path.join(current_folder, "losses.json"), "w") as jf:
                json.dump({"train_loss": model.train_losses,
                           "valid_loss": model.valid_losses}, jf)

        self.results_dict[f"fold_{current_fold_num + 1}"] = self.calculate_metrics()

        if self.save_to_folder:
            with open(os.path.join(current_folder, "metrics.json"), "w") as jf:
                json.dump(self.results_dict[f"fold_{current_fold_num + 1}"], jf)