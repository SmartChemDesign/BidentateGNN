# Configuration for multi-fidelity GCNN_FCNN training with best Optuna hyperparameters.
# solvent_x contains data_type at index DATA_TYPE_INDEX; it is masked in the forward
# pass and used only for multi-fidelity loss weighting.

from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool

from Source.global_poolings import MaxPooling

PATH_TO_SDF = "Data/redox/dataset/calc_exp/exp_calc_dataset.sdf"
SOLVENT_DB_PATH = "Data/redox/additional/solvent_properties.csv"
SOLVENT_MODE = "descriptors"

# Path to per-solvent linear calibration coefficients (E_lf = a * E_calc + b).
# Columns: solvent, redox_type, a, b, r2, p_value, n_points, source
CALIBRATION_COEFFICIENTS_PATH = "Data/redox/additional/calibration_coefficients.csv"

# (canonical_solvent_smiles, redox_type) pairs excluded from LF (calc) training data.
# Exclusion criteria: p >= 0.05 OR R^2 < 0.5 OR global_fallback (no threshold on n).
# Experimental data for these solvents are retained without changes.
EXCLUDED_LF_PAIRS: set[tuple[str, str]] = {
    # dimethoxyethane ox: global_fallback
    ("COCCOC", "ox"),
    # water ox: R^2=0.129, p=0.143
    ("O", "ox"),
    # 2-propanol red: global_fallback
    ("CC(C)O", "red"),
    # acetone red: p=0.082
    ("CC(C)=O", "red"),
    # propylene carbonate red: R^2=0.491, p=0.187
    ("CC1COC(=O)O1", "red"),
    # water red: R^2=0.205
    ("O", "red"),
}

CV_FOLDS = 5
SEED = 36
BATCH_SIZE = 1024
EPOCHS = 300
ES_PATIENCE = 40

HF_LOSS_WEIGHT = 0.8
LF_LOSS_WEIGHT = 0.2

OUTPUT_FOLDER = Path(
    f"Output/redox/GCNN_FCNN/calc_exp"
)

SOLVENT_FEATURES = 11
NODE_FEATURES = 75
SCALE_SOLVENT = True
DATA_TYPE_INDEX = -2  # position of data_type in solvent_x; masked before FCNN

# Solvent branch FC — input dim = SOLVENT_FEATURES - 1 (data_type excluded)
solvent_fc_params = {
    "hidden": (64, 32),
    "actf": nn.LeakyReLU(negative_slope=0.01),
    "dropout": 0.22468741076252172,
    "use_bn": False,
}

gcnn_params = {
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
}

post_fc_params = {
    "hidden": (32,),
    "actf": nn.PReLU(),
    "dropout": 0.03572808128707892,
    "use_bn": False,
}

global_pooling = MaxPooling

model_parameters = {
    "solvent_fc_params": solvent_fc_params,
    "gcnn_params": gcnn_params,
    "post_fc_params": post_fc_params,
    "global_pooling": global_pooling,
}

optimizer_class = torch.optim.Adam
optimizer_parameters = {
    "lr": 0.0015348866748097405,
    "weight_decay": 7.570460132658095e-06,
}