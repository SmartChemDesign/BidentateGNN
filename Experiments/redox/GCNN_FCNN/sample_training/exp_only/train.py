"""Train GCNN_FCNN models on experimental redox potential data with solvent features.

The dataset is featurized from an SDF file, split into solvent-stratified
cross-validation folds with a held-out test set, and used to train one model per
fold. Metrics, predictions and per-fold statistics are written to the output
folder defined in config.py.
"""

import copy
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath("."))

from Source.data import solvent_stratified_split
from Source.experiment_utils import (
    ExperimentLogger,
    collect_fold_statistics,
    log_dataset_composition,
    log_fold_statistics,
    save_fold_statistics,
)
from Source.GCNN_FCNN.featurizers import featurize_sdf_with_solvent_all
from Source.GCNN_FCNN.model_redox import GCNN_FCNN
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.trainer_redox import ModelTrainer

from config import (
    batch_size, cv_folds, epochs, es_patience, gcnn_params, log_level,
    logs_folder, optimizer, optimizer_parameters, output_folder, path_to_sdf,
    post_fc_params, save_models, scale_solvent, seed, solvent_db_path,
    solvent_fc_params, targets, test_size,
)

EXPERIMENT_NAME = "exp_only"


def build_model(sample):
    """Instantiate GCNN_FCNN with the feature dimensions taken from a sample batch."""
    return GCNN_FCNN(
        solvent_features=sample.solvent_x.shape[-1],
        node_features=sample.x.shape[-1],
        targets=targets,
        solvent_fc_params=copy.deepcopy(solvent_fc_params),
        gcnn_params=copy.deepcopy(gcnn_params),
        post_fc_params=copy.deepcopy(post_fc_params),
        optimizer=optimizer,
        optimizer_parameters=optimizer_parameters,
    )


def run_training(folds, test_loader, out_path: Path, statistics_path: Path):
    """Train one model per fold and export metrics, predictions and statistics."""
    exp_logger = ExperimentLogger(
        experiment_name=EXPERIMENT_NAME,
        log_dir=logs_folder,
        console_level=getattr(logging, log_level),
    )

    fold_stats = collect_fold_statistics(folds, test_loader)
    log_fold_statistics(exp_logger, fold_stats)

    model = build_model(next(iter(test_loader)))

    trainer = ModelTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_loader,
        output_folder=out_path,
        epochs=epochs,
        es_patience=es_patience,
        targets=targets,
        seed=seed,
        save_to_folder=save_models,
        experiment_logger=exp_logger,
    )
    trainer.train_cv_models()

    save_fold_statistics(
        fold_stats,
        statistics_path,
        meta={"run": EXPERIMENT_NAME, "seed": seed},
    )
    exp_logger.close()
    return trainer


def main() -> None:
    # Input and output paths
    out_path = Path(output_folder) / EXPERIMENT_NAME
    dataset_csv = out_path / "redox_dataset.csv"
    model_dataset_csv = out_path / "redox_model_dataset.csv"
    statistics_path = out_path / "solvent_statistics.txt"

    out_path.mkdir(parents=True, exist_ok=True)

    dataset = featurize_sdf_with_solvent_all(
        path_to_sdf=path_to_sdf,
        mol_featurizer=ConvMolFeaturizer(),
        seed=seed,
        shuffle=True,
        scale_solvent=scale_solvent,
        solvent_db_path=solvent_db_path,
        output_dataset_path=str(dataset_csv),
        output_model_dataset_path=str(model_dataset_csv),
    )
    if not dataset:
        raise RuntimeError("Empty dataset after featurization.")

    exp_logger = ExperimentLogger(EXPERIMENT_NAME, logs_folder, getattr(logging, log_level))
    log_dataset_composition(dataset, exp_logger)
    exp_logger.close()

    folds, test_loader = solvent_stratified_split(
        dataset=dataset,
        batch_size=batch_size,
        n_folds=cv_folds,
        test_size=test_size,
        seed=seed,
    )

    run_training(folds, test_loader, out_path, statistics_path)


if __name__ == "__main__":
    main()