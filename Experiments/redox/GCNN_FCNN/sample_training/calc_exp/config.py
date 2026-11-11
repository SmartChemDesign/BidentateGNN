"""
Configuration of the GCNN-FCNN redox potential experiments.

The same architecture and hyperparameters (best Optuna trial of the multifidelity
model) serve every training regime; a regime is set by TRAIN_DATA_TYPES and
LOSS_MODE:
  experimental only   TRAIN_DATA_TYPES = ("exp",)         LOSS_MODE = "pooled"
  calculated only     TRAIN_DATA_TYPES = ("calc",)        LOSS_MODE = "pooled"
  joint               TRAIN_DATA_TYPES = ("exp", "calc")  LOSS_MODE = "pooled"
  multifidelity       TRAIN_DATA_TYPES = ("exp", "calc")  LOSS_MODE = "multifidelity"
The split is computed once on the full dataset (or loaded from
SPLIT_ASSIGNMENTS_PATH), so all regimes share the test set and the folds; the
calculated potentials are calibrated inside each fold on its training records.
"""

import os
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool

from Source.global_poolings import MaxPooling

PATH_TO_SDF = "Data/redox/dataset/calc_exp/exp_calc_dataset.sdf"
SOLVENT_DB_PATH = "Data/redox/additional/solvent_properties.csv"

# Calibration E_exp = a * E_calc + b per (solvent, redox_type), fitted on the training
# records of each fold: own fit for groups with at least min_points pairs, global fit
# of the redox type otherwise. "exclusion" None keeps every calculated record; a dict
# drops the groups that fail its criteria: own fits with p >= max_p_value or
# R^2 < min_r2; global fits when the global R^2 < min_global_r2, when the group's own
# pairs deviate from the global line by more than max_fallback_residual (V, mean
# absolute) or, for groups without any pair, when allow_no_pairs is False. With
# exclude_across_folds a group excluded in any fold is excluded in every fold, so all
# fold models are trained on the same groups. CALIBRATION = None trains on the raw
# PM7/COSMO potentials (meaningful for the calculated-only regime only).
# CALIBRATION_OFF=1 in the environment trains on the raw PM7/COSMO potentials
CALIBRATION = None if os.environ.get("CALIBRATION_OFF") else {
    "min_points": 5,
    "exclude_across_folds": True,
    "exclusion": None,
    # {
    #     "max_p_value": 0.05,
    #     "min_r2": 0.5,
    #     "min_global_r2": 0.5,
    #     "max_fallback_residual": 0.3,
    #     "allow_no_pairs": True,
    # },
}

# The regime and the seed can be overridden from the environment, e.g.
# TRAIN_DATA_TYPES=exp LOSS_MODE=pooled SEED=42 python3 Experiments/redox/GCNN_FCNN/train.py,
# so that several runs can be queued without editing this file between launches
TRAIN_DATA_TYPES = tuple(os.environ.get("TRAIN_DATA_TYPES", "exp,calc").split(","))
LOSS_MODE = os.environ.get("LOSS_MODE", "multifidelity")
HF_LOSS_WEIGHT = float(os.environ.get("HF_LOSS_WEIGHT", 0.8))
LF_LOSS_WEIGHT = float(os.environ.get("LF_LOSS_WEIGHT", 1.0 - HF_LOSS_WEIGHT))

CV_FOLDS = 5
TEST_SIZE = 0.2
SEED = int(os.environ.get("SEED", 36))
SCALE_SOLVENT = True

# Path to a split_assignments.csv of a previous run; None computes the split anew
SPLIT_ASSIGNMENTS_PATH = os.environ.get("SPLIT_ASSIGNMENTS_PATH") or None
# Split scheme: "random" (assign_splits: systematic test selection per solvent, other
# records of the test molecules withheld) or "molecule" (assign_molecule_splits: a
# stratified 20 % of the molecules with all their experimental records). The test
# molecules of the molecule split are chosen deterministically, so they are common to
# every regime and seed; SEED sets the folds, the initialisation and the batch order.
SPLIT_SCHEME = "molecule"

# Best Optuna trial of the multifidelity model (experimental + calculated data)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1024))
EPOCHS = 300
ES_PATIENCE = 40

model_parameters = {
    "solvent_fc_params": {
        "hidden": (64, 32),
        "actf": nn.LeakyReLU(negative_slope=0.01),
        "dropout": 0.22468741076252172,
        "use_bn": False,
    },
    "gcnn_params": {
        "pre_fc_params": {
            "hidden": (),
            "dropout": 0.0,
            "use_bn": False,
            "actf": nn.ReLU(),
        },
        "hidden_conv": (512, 512, 512),
        "conv_dropout": 0.22380490394118516,
        "conv_actf": nn.ReLU(),
        "conv_layer": GATConv,
        "conv_parameters": None,
        "graph_pooling": global_mean_pool,
        "post_fc_params": {
            "hidden": (512,),
            "actf": nn.ReLU(),
            "dropout": 0.0,
            "use_bn": True,
        },
    },
    "post_fc_params": {
        "hidden": (32,),
        "actf": nn.PReLU(),
        "dropout": 0.03572808128707892,
        "use_bn": False,
    },
    "global_pooling": MaxPooling,
}

optimizer_class = torch.optim.Adam
optimizer_parameters = {
    "lr": 0.0015348866748097405,
    "weight_decay": 7.570460132658095e-06,
}

RUN_LABEL = "" if CALIBRATION is not None else "_uncalibrated"
if SPLIT_SCHEME == "molecule":
    RUN_LABEL += "_molsplit"
OUTPUT_FOLDER = Path(f"Output/redox/GCNN_FCNN/{'_'.join(TRAIN_DATA_TYPES)}_{LOSS_MODE}{RUN_LABEL}_seed{SEED}")
# RUN_SUFFIX in the environment distinguishes runs with changed hyperparameters, e.g. _batch64
OUTPUT_FOLDER = OUTPUT_FOLDER.with_name(OUTPUT_FOLDER.name + os.environ.get("RUN_SUFFIX", ""))