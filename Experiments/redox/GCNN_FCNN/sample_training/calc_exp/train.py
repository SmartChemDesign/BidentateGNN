# Multi-fidelity GCNN_FCNN training with data_type masking in the forward pass.
# solvent_x carries data_type as a feature; it is automatically removed before
# the solvent FCNN and used only for HF/LF loss weighting.

import os
import sys

from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn

sys.path.append(os.path.abspath("."))

from Source.data import cv_train_val_test_split, root_mean_squared_error
from Source.GCNN_FCNN.featurizers import featurize_sdf_with_solvent_all
from Source.GCNN_FCNN.model_redox import (
    GCNN_FCNN_MaskedDataType,
    verify_model_masking,
)
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.trainer_redox import ModelTrainer

from config import (
    BATCH_SIZE, CALIBRATION_COEFFICIENTS_PATH, CV_FOLDS, DATA_TYPE_INDEX,
    EPOCHS, ES_PATIENCE, EXCLUDED_LF_PAIRS,
    HF_LOSS_WEIGHT, LF_LOSS_WEIGHT, OUTPUT_FOLDER,
    PATH_TO_SDF, SCALE_SOLVENT, SEED, SOLVENT_DB_PATH, SOLVENT_MODE,
    model_parameters, optimizer_class, optimizer_parameters,
)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    all_data = featurize_sdf_with_solvent_all(
        path_to_sdf=str(PATH_TO_SDF),
        mol_featurizer=ConvMolFeaturizer(),
        solvent_mode=SOLVENT_MODE,
        solvent_db_path=str(SOLVENT_DB_PATH) if SOLVENT_MODE == "descriptors" else None,
        scale_solvent=SCALE_SOLVENT,
        seed=SEED,
        shuffle=True,
        log_file=str(OUTPUT_FOLDER / "featurization.log"),
        output_dataset_path=str(OUTPUT_FOLDER / "dataset.csv"),
        output_model_dataset_path=str(OUTPUT_FOLDER / "model_dataset.csv"),
        scaler_save_path=str(OUTPUT_FOLDER / "scaler.joblib"),
        calibration_coefficients_path=str(CALIBRATION_COEFFICIENTS_PATH),
        excluded_lf_pairs=EXCLUDED_LF_PAIRS,
    )

    sample = all_data[0]
    solvent_dim = sample.solvent_x.shape[-1]
    node_dim = sample.x.shape[-1]

    folds, test_loader = cv_train_val_test_split(
        full_dataset=all_data,
        cv_folds=CV_FOLDS,
        batch_size=BATCH_SIZE,
        test_size=0.2,
        seed=SEED,
        output_dir=OUTPUT_FOLDER / "split_statistics",
    )

    targets = ({
        "name": "E",
        "mode": "regression",
        "dim": 1,
        "metrics": {
            "R2": (r2_score, {}),
            "RMSE": (root_mean_squared_error, {}),
            "MAE": (mean_absolute_error, {}),
        },
        "loss": nn.MSELoss(reduction="none"),  # element-wise loss required for HF/LF weighting
    },)

    model = GCNN_FCNN_MaskedDataType(
        solvent_features=solvent_dim,
        node_features=node_dim,
        targets=targets,
        data_type_index=DATA_TYPE_INDEX,
        hf_loss_weight=HF_LOSS_WEIGHT,
        lf_loss_weight=LF_LOSS_WEIGHT,
        **model_parameters,
        optimizer=optimizer_class,
        optimizer_parameters=optimizer_parameters,
    )

    sample_batch = next(iter(folds[0][0]))
    if not verify_model_masking(model, sample_batch):
        raise RuntimeError("Model masking verification failed. Check configuration.")

    trainer = ModelTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_loader,
        output_folder=str(OUTPUT_FOLDER),
        epochs=EPOCHS,
        es_patience=ES_PATIENCE,
        targets=targets,
        seed=SEED,
    )
    trainer.train_cv_models()


if __name__ == "__main__":
    main()