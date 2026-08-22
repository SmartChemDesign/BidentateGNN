import logging
import torch

from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MFConv, SAGEConv
from config import ROOT_DIR

from Source.data import balanced_train_valid_split, root_mean_squared_error
from Source.trainer_solubility import GCNNTrainer
from Source.GCNN_FCNN.featurizers import featurize_sdf_with_solvent_experimental_united
from Source.GCNN_FCNN.model_solubility import GCNN_FCNN
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.global_poolings import MaxPooling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

parser = ArgumentParser()
parser.add_argument('--sdf', type=str, help='Path to the data file')  # required=True,
parser.add_argument('--experiment-name', type=str, help='The name of the experiment')  # required=True,
parser.add_argument('--max-samples', type=int, default=None, help='Use only a several samples for training')
parser.add_argument('--folds', type=int, default=1, help='Number of folds for cross-validation')
parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=1000, help='Maximum number of epochs to train')
parser.add_argument('--es-patience', type=int, default=100, help='Number of epochs to wait before early stopping')
parser.add_argument('--mode', type=str, default="regression", help='Mode of the target - regression, binary_classification or multiclass_classification')
parser.add_argument('--learning-rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--include-descriptors', type=str, default='', help='A string with additional descriptors separated by a space')
parser.add_argument('--test-solvents', type=str, default='water', help='A string with the names of solvents on which the model will be tested')
parser.add_argument('--n-test-in-train', type=int, default=0, help='Number of test samples in train dataset')
parser.add_argument('--use-maccskeys', type=bool, default=False, help='To use MACCSKeys or not to use')

args = parser.parse_args()

# target parameters
targets = ({
               "name": "Solubility",
               "mode": "regression",
               "dim": 1,
               "metrics": {
                   "R2": (r2_score, {}),
                   "RMSE": (root_mean_squared_error, {}),
                   "MAE": (mean_absolute_error, {})
               },
               "loss": nn.MSELoss(),
           },)

# model parameters
model_parameters = {
    "metal_fc_params": {
        "hidden": (),
        "dropout": 0,
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
        "hidden_conv": (512, 128, 128,),
        "conv_dropout": 0.27936243337975536,
        "conv_actf": nn.LeakyReLU(),
        "conv_layer": SAGEConv,
        "conv_parameters": None,
        "graph_pooling": global_mean_pool
    },
    "post_fc_params": {
        "hidden": (128, 64, 16),
        "dropout": 0.06698879155641034,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "global_pooling": MaxPooling,
}

featurize = partial(
    featurize_sdf_with_solvent_experimental_united,
    mol_featurizer=ConvMolFeaturizer(), include_descriptors=args.include_descriptors, use_maccskeys=False
)

train_val_dataset, test_dataset = featurize(path_to_sdf=str(ROOT_DIR / args.sdf), test_solvents=args.test_solvents, n_test_in_train=args.n_test_in_train)
logging.info("Splitting...")
folds = balanced_train_valid_split(
    datasets=[train_val_dataset],
    n_folds=args.folds,
    batch_size=args.batch_size,
    shuffle_every_epoch=True,
    seed=args.seed
)

test_loader = DataLoader(test_dataset)

model = GCNN_FCNN(
    metal_features=train_val_dataset[0].x_fully_connected.shape[-1],
    node_features=train_val_dataset[0].x.shape[-1],
    targets=targets,
    **model_parameters,
    optimizer=torch.optim.Adam,
    optimizer_parameters=None,
)

trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_loader,
    output_folder=str(ROOT_DIR / "Models" / "solubility" / "experimental_descriptors" / "boxplots" / args.experiment_name),
    epochs=args.epochs,
    es_patience=args.es_patience,
    targets=targets,
    seed=args.seed,
)

trainer.train_cv_models()