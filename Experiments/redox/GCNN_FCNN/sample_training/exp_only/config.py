"""Configuration of the GCNN_FCNN experiment on experimental redox potentials.

Holds dataset paths, split and training hyperparameters, the architecture of the
molecular and solvent branches, and the target definition with its metrics.
"""

import torch
from torch import nn
from torch_geometric.nn import MFConv, global_mean_pool
from sklearn.metrics import mean_absolute_error, r2_score

from Source.data import root_mean_squared_error

# Input data
path_to_sdf = "Data/redox/dataset/exp_only/exp_dataset.sdf"
solvent_db_path = "Data/redox/additional/solvent_properties.csv"
scale_solvent = True

# Split and training
test_size = 0.2
cv_folds = 5
seed = 42
batch_size = 256
epochs = 400
es_patience = 20

# Graph pooling applied after the convolutional stack
graph_pooling = global_mean_pool

# Solvent branch (FCNN)
solvent_fc_params = {
    "hidden": (128, 64, 32),
    "dropout": 0.2,
    "use_bn": True,
    "actf": nn.ReLU(),
}

# Molecular branch (GCNN)
gcnn_params = {
    "pre_fc_params": {
        "hidden": (128,),
        "dropout": 0.1,
        "use_bn": False,
        "actf": nn.ReLU(),
    },
    "hidden_conv": (256, 128, 64),
    "conv_dropout": 0.1,
    "conv_actf": nn.ReLU(),
    "conv_layer": MFConv,
    "conv_parameters": None,
    "graph_pooling": graph_pooling,
    "post_fc_params": {
        "hidden": (128,),
        "dropout": 0.1,
        "use_bn": False,
        "actf": nn.ReLU(),
    },
}

# Final fully connected block after concatenating the molecular and solvent embeddings
post_fc_params = {
    "hidden": (128, 64),
    "dropout": 0.1,
    "use_bn": True,
    "actf": nn.ReLU(),
}

targets = ({
    "name": "E",
    "mode": "regression",
    "dim": 1,
    "metrics": {
        "R2": (r2_score, {}),
        "RMSE": (root_mean_squared_error, {}),
        "MAE": (mean_absolute_error, {}),
    },
    "loss": nn.MSELoss(),
},)

optimizer = torch.optim.AdamW
optimizer_parameters = {"lr": 1e-3, "weight_decay": 1e-4}

# Output
output_folder = "Output/redox/GCNN_FCNN"
logs_folder = "logs"
log_level = "INFO"
save_models = True
save_predictions = True