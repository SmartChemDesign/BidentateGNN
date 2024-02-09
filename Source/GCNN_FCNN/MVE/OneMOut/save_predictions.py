import pandas as pd
from torch_geometric.loader import DataLoader
import logging
import os.path
import sys

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, MFConv
from tqdm import tqdm

sys.path.append(os.path.abspath("Source/GCNN_FCNN/"))
from Source.GCNN_FCNN.model import GCNN_FCNN
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.GCNN_FCNN.global_poolings import MaxPooling
from config import ROOT_DIR
from Source.trainer import ModelTrainer

sys.path.append(os.path.abspath(".."))

from Source.trainer import ModelShell
from Source.data import balanced_train_valid_split
from Source.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions
from Source.GCNN_FCNN.MVE.metrics import r2_score_MVE, root_mean_squared_error_MVE, \
    mean_absolute_error_MVE, negative_log_likelihood
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

other_metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Mo', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'Hf', 'Re',
                'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
Ln_metals = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', ]
Ac_metals = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf']

cv_folds = 1
test_size = 0.1
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
mode = "regression"
train_sdf_folder = ROOT_DIR / "Data/OneM_cond_adds"

targets = ({
               "name": "logK",
               "mode": "regression",
               "dim": 2,
               "metrics": {
                   "R2": (r2_score_MVE, {}),
                   "RMSE": (root_mean_squared_error_MVE, {}),
                   "MAE": (mean_absolute_error_MVE, {})
               },
               "loss": negative_log_likelihood,
           },)
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

featurized = {
    metal: featurize_sdf_with_metal_and_conditions(
        path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
        mol_featurizer=ConvMolFeaturizer(),
        metal_featurizer=SkipatomFeaturizer())
    for metal in tqdm(list(set(["Y", "Sc"] + Ln_metals + Ac_metals) - {"Ac", "Pa"}), desc="Featurizig")
}

for test_metal in list(set(Ac_metals) - {"Ac", "Pa"}):
    train_metals = list(set(["Y", "Sc"] + Ln_metals + Ac_metals) - {"Ac", "Pa", test_metal})

    output_dir = ROOT_DIR / f"Output/Uncertainty/MVE_OneM/{test_metal}"
    train_folder_name = [f for f in os.listdir(ROOT_DIR / "Output/Uncertainty_MVE_OneMOut") if f.startswith(test_metal)][0]
    train_folder = ROOT_DIR / f"Output/Uncertainty_MVE_OneMOut/{train_folder_name}"

    train_datasets = [featurized[metal] for metal in train_metals]
    logging.info("Splitting...")
    folds = balanced_train_valid_split(train_datasets, n_folds=cv_folds,
                                       batch_size=batch_size,
                                       shuffle_every_epoch=True,
                                       seed=seed)

    test_loader = DataLoader(featurize_sdf_with_metal_and_conditions(
        path_to_sdf=os.path.join(train_sdf_folder, f"{test_metal}.sdf"),
        mol_featurizer=ConvMolFeaturizer(),
        metal_featurizer=SkipatomFeaturizer()), batch_size=batch_size)

    train_loader, val_loader = folds[0]

    super_model = ModelShell(GCNN_FCNN, train_folder)
    model = super_model.models[0]

    #######################################

    trainer = ModelTrainer(
        model=None,
        train_valid_data=folds,
        test_data=test_loader,
        targets=targets,
        seed=seed,
    )
    trainer.models = super_model.models
    result = trainer.calculate_metrics()

    print(f"{test_metal}: {result}")

    #######################################

    os.makedirs(output_dir, exist_ok=True)
    for loader, name in zip((train_loader, val_loader, test_loader), ("train", "valid", "test")):
        data = []
        for sample in loader.dataset:
            logK, logK_log_var = model(sample)["logK"].squeeze()
            data += [{
                "true_logK": sample.y["logK"].item(),
                "pred_logK": logK.item(),
                "pred_logK_std": torch.sqrt(torch.exp(logK_log_var)).item()
            }]

        pd.DataFrame(data).to_json(output_dir / f"{name}.json")
        print(f"{test_metal} {name} done")
    print()

print("All done")
