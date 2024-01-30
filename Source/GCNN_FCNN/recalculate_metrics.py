import logging
import os.path
import sys

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from config import ROOT_DIR

sys.path.append(os.path.abspath("."))

from model import GCNN_FCNN
from Source.models.GCNN.trainer import GCNNTrainer
from Source.trainer import ModelShell
from Source.data import balanced_train_valid_split, root_mean_squared_error
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

cv_folds = 5
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
train_sdf_folder = ROOT_DIR / "Data/OneM_cond_adds"
train_folder = ROOT_DIR / "Output/WithCondAdd/5fold/Ac_5fold_regression_2023_05_30_21_09_41"

targets = ({
               "name": "logK",
               "mode": "regression",
               "dim": 1,
               "metrics": {
                   "R2": (r2_score, {}),
                   "RMSE": (root_mean_squared_error, {}),
                   "MAE": (mean_absolute_error, {})
               },
               "loss": nn.MSELoss(),
           },)

test_metal = "Ac"

all_metals = ['Co', 'In', 'Zn', 'Tm', 'Mo', 'La', 'Al', 'Cd', 'Lu', 'Tb', 'Pa', 'Cs', 'Ni', 'Ho', 'Ti', 'Zr', 'Pd',
              'Gd', 'Cr', 'Am', 'Y', 'Eu', 'Pu', 'Hg', 'Pr', 'Au', 'Hf', 'Rh', 'Np', 'Cf', 'Mn', 'Pt', 'Li', 'Sc', 'Nd',
              'Bk', 'Ca', 'Tl', 'Re', 'Na', 'Bi', 'Be', 'Er', 'Cu', 'Ac', 'Pb', 'Th', 'Pm', 'Sr', 'U', 'Sn', 'Ag', 'Rb',
              'Dy', 'Ce', 'V', 'Yb', 'Ga', 'Sm', 'Mg', 'Fe', 'Cm', 'Sb', 'K', 'Ba']

super_model = ModelShell(GCNN_FCNN, train_folder)

train_datasets = [featurize_sdf_with_metal_and_conditions(path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
                                                          mol_featurizer=ConvMolFeaturizer(),
                                                          metal_featurizer=SkipatomFeaturizer())
                  for metal in tqdm(all_metals, desc="Featurizig") if metal != test_metal]
logging.info("Splitting...")
folds = balanced_train_valid_split(train_datasets, n_folds=cv_folds,
                                   batch_size=batch_size,
                                   shuffle_every_epoch=True,
                                   seed=seed)

test_loader = DataLoader(featurize_sdf_with_metal_and_conditions(
    path_to_sdf=os.path.join(train_sdf_folder, f"{test_metal}.sdf"),
    mol_featurizer=ConvMolFeaturizer(),
    metal_featurizer=SkipatomFeaturizer()),
    batch_size=batch_size)

trainer = GCNNTrainer(
    model=None,
    train_valid_data=folds,
    test_data=test_loader,
    targets=targets,
    seed=seed,
)
trainer.models = super_model.models
result = trainer.calculate_metrics()

print(result)
