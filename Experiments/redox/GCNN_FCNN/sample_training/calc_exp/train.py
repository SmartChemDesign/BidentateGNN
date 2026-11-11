"""
Train GCNN-FCNN redox potential models with K-fold cross-validation and a held-out
test set. The training regime (experimental-only, calculated-only, joint or
multifidelity) is set in config.py; the split and the in-fold calibration of the
calculated potentials are shared by all regimes.
"""

import logging
import os
import sys

from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn

sys.path.append(os.path.abspath("."))

from Source.data import (
    assign_molecule_splits,
    assign_splits,
    cv_train_val_test_split,
    load_assignments,
    root_mean_squared_error,
    save_assignments,
)
from Source.GCNN_FCNN.featurizers import featurize_sdf_with_solvent_all
from Source.GCNN_FCNN.model_redox import GCNN_FCNN
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.trainer_redox import ModelTrainer

from config import (
    BATCH_SIZE, CALIBRATION, CV_FOLDS, EPOCHS, ES_PATIENCE,
    HF_LOSS_WEIGHT, LF_LOSS_WEIGHT, LOSS_MODE, OUTPUT_FOLDER,
    PATH_TO_SDF, SCALE_SOLVENT, SEED, SOLVENT_DB_PATH, SPLIT_ASSIGNMENTS_PATH, SPLIT_SCHEME,
    TEST_SIZE, TRAIN_DATA_TYPES,
    model_parameters, optimizer_class, optimizer_parameters,
)

TARGETS = ({
    "name": "E",
    "mode": "regression",
    "dim": 1,
    "metrics": {
        "R2": (r2_score, {}),
        "RMSE": (root_mean_squared_error, {}),
        "MAE": (mean_absolute_error, {}),
    },
    # element-wise loss: the model reduces it per data type
    "loss": nn.MSELoss(reduction="none"),
},)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_FOLDER / "train.log", mode="w")],
    )

    all_data = featurize_sdf_with_solvent_all(
        path_to_sdf=PATH_TO_SDF,
        mol_featurizer=ConvMolFeaturizer(),
        solvent_db_path=SOLVENT_DB_PATH,
        scale_solvent=SCALE_SOLVENT,
        seed=SEED,
        shuffle=True,
        output_dataset_path=str(OUTPUT_FOLDER / "dataset.csv"),
        output_model_dataset_path=str(OUTPUT_FOLDER / "model_dataset.csv"),
        scaler_save_path=str(OUTPUT_FOLDER / "scaler.joblib"),
    )

    # The split is always defined on the full dataset so that every regime
    # shares the same test records and folds
    if SPLIT_ASSIGNMENTS_PATH:
        assignments = load_assignments(SPLIT_ASSIGNMENTS_PATH)
    elif SPLIT_SCHEME == "molecule":
        assignments = assign_molecule_splits(all_data, CV_FOLDS, test_size=TEST_SIZE, seed=SEED)
    else:
        assignments = assign_splits(all_data, CV_FOLDS, test_size=TEST_SIZE, seed=SEED)
    save_assignments(assignments, OUTPUT_FOLDER / "split_assignments.csv")

    # Calculated potentials are calibrated on the training records of each fold;
    # train/val folds keep only TRAIN_DATA_TYPES, the test set is always experimental
    folds, test_loader = cv_train_val_test_split(
        all_data, CV_FOLDS, BATCH_SIZE,
        assignments=assignments,
        calibration=CALIBRATION,
        train_data_types=TRAIN_DATA_TYPES,
        output_dir=OUTPUT_FOLDER / "split_statistics",
    )

    sample = all_data[0]
    model = GCNN_FCNN(
        solvent_features=sample.solvent_x.shape[-1],
        node_features=sample.x.shape[-1],
        targets=TARGETS,
        hf_loss_weight=HF_LOSS_WEIGHT,
        lf_loss_weight=LF_LOSS_WEIGHT,
        loss_mode=LOSS_MODE,
        **model_parameters,
        optimizer=optimizer_class,
        optimizer_parameters=optimizer_parameters,
    )

    trainer = ModelTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_loader,
        output_folder=str(OUTPUT_FOLDER),
        epochs=EPOCHS,
        es_patience=ES_PATIENCE,
        targets=TARGETS,
        seed=SEED,
    )
    trainer.train_cv_models()


if __name__ == "__main__":
    main()