import logging
import sys, os.path
from datetime import datetime
import json

import torch
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn
from torch_geometric.nn import global_mean_pool, MFConv
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from Source.data import balanced_train_valid_split, root_mean_squared_error, tanimoto_train_valid_split
from Source.trainer_logK import GCNNTrainer
from Source.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions
from Source.GCNN_FCNN.model_logK import GCNN_FCNN
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.GCNN_FCNN.global_poolings import MaxPooling
from config import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

# Here we gather all available metals to single list to later load corresponding ligands and stability constant values
Ln_metals = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', ]
Ac_metals = ['Am', 'Cm', 'Bk', 'Cf']
train_metals = list(set(["Y", "Sc"] + Ln_metals + Ac_metals))

# training parameters
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
mode = "regression"
train_sdf_folder = "Data/logK"
output_folder = f"Output/logK/fold_{mode}_{time_mark}"

# target description
targets = ({"name": "logK",
            "mode": "regression",
            "dim": 1,
            "metrics": {
                "R2": (r2_score, {}),
                "RMSE": (root_mean_squared_error, {}),
                "MAE": (mean_absolute_error, {})
            },
            "loss": nn.MSELoss(),
            },)

# model parameter optimized for stability constant task
model_parameters = {
    "metal_fc_params": {
        "hidden": (256, 128, 128, 64, 64,),
        "dropout": 0.25108912274809364,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "gcnn_params": {
        "pre_fc_params": {
            "hidden": (),
            "dropout": 0,
            "actf": nn.LeakyReLU(),
        },
        "post_fc_params": {
            "hidden": (),
            "dropout": 0,
            "use_bn": False,
            "actf": nn.LeakyReLU(),
        },
        "hidden_conv": (128, 128, 64,),
        "conv_dropout": 0.27936243337975536,
        "conv_actf": nn.LeakyReLU(),
        "conv_layer": MFConv,
        "conv_parameters": None,
        "graph_pooling": global_mean_pool
    },
    "post_fc_params": {
        "hidden": (256,),
        "dropout": 0.06698879155641034,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "global_pooling": MaxPooling,
}

# convert all data from .sdf files to data objects
train_datasets = [featurize_sdf_with_metal_and_conditions(path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
                                                        mol_featurizer=ConvMolFeaturizer(),
                                                        metal_featurizer=SkipatomFeaturizer())
                for metal in tqdm(train_metals, desc="Featurizig")]

smiles = [i[1] for i in train_datasets]
train_datasets = [i[0] for i in train_datasets]

# split dataset to train and valid
logging.info("Splitting...")
folds = balanced_train_valid_split(train_datasets, n_folds=5,
                                batch_size=batch_size,
                                shuffle_every_epoch=True,
                                seed=seed)


# init model object
model = GCNN_FCNN(
    metal_features=next(iter(folds[0][0])).metal_x.shape[-1],
    node_features=next(iter(folds[0][0])).x.shape[-1],
    targets=targets,
    **model_parameters,
    optimizer=torch.optim.Adam,
    optimizer_parameters=None,
)

# init trainer object
trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=None,
    output_folder=output_folder,
    epochs=epochs,
    es_patience=es_patience,
    targets=targets,
    seed=seed,
)

# train n-fold models
trainer.train_cv_models()
