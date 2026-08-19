from pathlib import Path
from torch import nn
from torch_geometric.nn import global_mean_pool, MFConv

path_to_sdf = 'Data/redox/dataset/exp_only/S2_exp_dataset.sdf'
cv_folds = 5
seed = 42
batch_size = 64
epochs = 300
es_patience = 20

model_parameters = {
    "pre_fc_params": {
        "hidden": (128,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.ReLU(),
    },
    "hidden_conv": (256, 128, 64),
    "conv_dropout": 0,
    "conv_actf": nn.ReLU(),
    "conv_layer": MFConv,
    "conv_parameters": None,
    "graph_pooling": global_mean_pool,
    "post_fc_params": {
        "hidden": (128,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.ReLU(),
    }
}

output_folder = "Output/redox/GCNN"
logs_folder = "logs"
save_models = True
save_predictions = True
calculate_solvent_metrics = True
log_level = "INFO"


def validate_config():
    """Validate configuration values; raises ValueError on invalid settings."""
    errors = []

    if not Path(path_to_sdf).exists():
        errors.append(f"SDF file not found: {path_to_sdf}")
    if cv_folds < 2:
        errors.append(f"cv_folds must be >= 2, got {cv_folds}")
    if batch_size < 1:
        errors.append(f"batch_size must be >= 1, got {batch_size}")
    if epochs < 1:
        errors.append(f"epochs must be >= 1, got {epochs}")
    if es_patience < 1:
        errors.append(f"es_patience must be >= 1, got {es_patience}")
    if len(model_parameters["hidden_conv"]) < 1:
        errors.append("At least one convolutional layer is required")

    if errors:
        raise ValueError("\n".join(["Configuration validation failed:"] + errors))

    return True