import os
import sys
import logging
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn

sys.path.append(os.path.abspath("."))

from Source.GCNN.featurizers import featurize_sdf_exp_no_solvent
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.trainer_redox import ModelTrainer
from Source.GCNN.model import GCNN
from Source.data import root_mean_squared_error, train_test_valid_split
from Source.experiment_utils import ExperimentLogger

from config import (
    path_to_sdf,
    cv_folds,
    seed,
    batch_size,
    epochs,
    es_patience,
    model_parameters,
    output_folder
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_single_redox_type(data, redox_label, base_output_folder):
    experiment_name = f"redox_{redox_label}"
    exp_logger = ExperimentLogger(experiment_name=experiment_name, log_dir="logs")
    log = exp_logger.main_logger

    log.info(f"Training {redox_label.upper()} model — {len(data)} samples")

    log.info(f"Creating {cv_folds}-fold cross-validation split...")
    folds, test_loader = train_test_valid_split(
        dataset=data,
        n_folds=cv_folds,
        test_ratio=0.2,
        batch_size=batch_size,
        seed=seed
    )
    log.info(f"Test set: {len(test_loader.dataset)} samples")
    for i, (train_loader, val_loader) in enumerate(folds):
        log.info(f"  Fold {i+1}: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    targets = ({
        "name": "E",
        "mode": "regression",
        "dim": 1,
        "metrics": {
            "R2": (r2_score, {}),
            "RMSE": (root_mean_squared_error, {}),
            "MAE": (mean_absolute_error, {})
        },
        "loss": nn.MSELoss(),
    },)

    sample_batch = next(iter(test_loader))
    node_features_dim = sample_batch.x.shape[-1]
    log.info(f"Node feature dimension: {node_features_dim}")

    model = GCNN(
        node_features=node_features_dim,
        targets=targets,
        **model_parameters,
        optimizer_parameters=None
    )

    specific_output_folder = Path(base_output_folder) / experiment_name

    trainer = ModelTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_loader,
        output_folder=specific_output_folder,
        epochs=epochs,
        es_patience=es_patience,
        targets=targets,
        seed=seed,
    )

    log.info("Starting training...")
    trainer.train_cv_models()
    log.info(f"Training complete. Results saved to {specific_output_folder}")

    exp_logger.close()


def main():
    logger.info("Training separate redox models (oxidation and reduction)")

    logger.info("Loading and featurizing dataset...")
    ox_data, red_data = featurize_sdf_exp_no_solvent(
        path_to_sdf=path_to_sdf,
        mol_featurizer=ConvMolFeaturizer()
    )
    logger.info(f"Oxidation: {len(ox_data)} samples | Reduction: {len(red_data)} samples")

    for label, data in [("ox", ox_data), ("red", red_data)]:
        if len(data) < cv_folds:
            logger.warning(f"Not enough data for {label} ({len(data)} samples) — skipping.")
            continue
        train_single_redox_type(data=data, redox_label=label, base_output_folder=output_folder)

    logger.info(f"All models trained. Results saved to {output_folder}")


if __name__ == "__main__":
    main()