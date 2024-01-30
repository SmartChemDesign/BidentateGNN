import json
import os
import sys
from datetime import datetime
from inspect import isfunction, isclass, signature

import numpy as np
from dgllife.utils import CanonicalAtomFeaturizer, PretrainAtomFeaturizer, \
    AttentiveFPAtomFeaturizer, PAGTNAtomFeaturizer

import optuna
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.optim import Adam, AdamW, RMSprop
from torch_geometric.nn import MFConv, global_mean_pool, global_max_pool, GATConv, TransformerConv, SAGEConv, \
    CuGraphSAGEConv, GCNConv, ARMAConv, CuGraphGATConv, GINConv, GINEConv, SSGConv, XConv, TopKPooling, SAGPooling, \
    EdgePooling, ASAPooling, PANPooling, MemPooling

sys.path.append(os.path.abspath("."))

from Source.data import get_num_node_features, get_num_targets, train_test_valid_split
from Source.featurizers import ConvMolFeaturizer, featurize_csv_to_graph, DGLFeaturizer
from Source.models.GCNN.GCNN_model import GCNN
from Source.models.GCNN.GCNN_trainer import GCNNTrainer

time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

n_trials = None
timeout = 60 * 60 * 3

cv_folds = 1
batch_size = 64
epochs = 500
es_patience = 50

mode = "regression"
general_output_path = f"Output/Optuna3GCNN_{time_mark}"
path_to_csv = "Data/mouse_LD50.csv"
max_data = None
targets = ("mouse_intraperitoneal_LD50",
           "mouse_intravenous_LD50",
           "log_mouse_intraperitoneal_LD50",
           "log_mouse_intravenous_LD50")
target_metrics = {
    "mouse_intraperitoneal_LD50": {
        "R2": (r2_score, {}),
        "RMSE": (lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)), {}),
        "MSE": (mean_squared_error, {}),
        "MAE": (mean_absolute_error, {})
    },
    "mouse_intravenous_LD50": {
        "R2": (r2_score, {}),
        "RMSE": (lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)), {}),
        "MSE": (mean_squared_error, {}),
        "MAE": (mean_absolute_error, {})
    },
    "log_mouse_intraperitoneal_LD50": {
        "R2": (r2_score, {}),
        "RMSE": (lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)), {}),
        "MSE": (mean_squared_error, {}),
        "MAE": (mean_absolute_error, {})
    },
    "log_mouse_intravenous_LD50": {
        "R2": (r2_score, {}),
        "RMSE": (lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)), {}),
        "MSE": (mean_squared_error, {}),
        "MAE": (mean_absolute_error, {})
    },
}


class FCNNParams:
    def __init__(self, trial: optuna.Trial,
                 n_conv_lims=(2, 10), n_pre_linear_lims=(1, 5), n_post_linear_lims=(1, 5),
                 conv_dropout_lims=(0, 0.8), pre_linear_dropout_lims=(0, 0.8), post_linear_dropout_lims=(0, 0.8),
                 conv_layer_variants=None, pooling_layer_variants=None,
                 conv_actf_variants=None, linear_actf_variants=None,
                 optimizer_variants=None, hidden_lims=(32, 128, 32), post_linear_bn_variants=(True, False)):

        self.trial = trial
        self.n_conv_lims = n_conv_lims
        self.n_pre_linear_lims = n_pre_linear_lims
        self.n_post_linear_lims = n_post_linear_lims
        self.pre_linear_dropout_lims = pre_linear_dropout_lims
        self.post_linear_dropout_lims = post_linear_dropout_lims
        self.post_linear_bn_variants = post_linear_bn_variants
        self.conv_dropout_lims = conv_dropout_lims
        self.hidden_lims = hidden_lims

        if conv_actf_variants is None:
            conv_actf_variants = {
                "nn.ReLU()": nn.ReLU(),
                "nn.LeakyReLU()": nn.LeakyReLU(),
            }
        self.conv_actf_variants = conv_actf_variants

        if linear_actf_variants is None:
            linear_actf_variants = {
                "nn.ReLU()": nn.ReLU(),
                "nn.LeakyReLU()": nn.LeakyReLU(),
            }
        self.linear_actf_variants = linear_actf_variants

        if optimizer_variants is None:
            optimizer_variants = {
                "Adam": Adam,
            }
        self.optimizer_variants = optimizer_variants

        if pooling_layer_variants is None:
            pooling_layer_variants = {
                "global_mean_pool": global_mean_pool,
            }
        self.pooling_layer_variants = pooling_layer_variants

        if conv_layer_variants is None:
            conv_layer_variants = {
                "MFConv": MFConv,
            }
        self.conv_layer_variants = conv_layer_variants

    def get_conv_layer(self):
        self.conv_layer_name = self.trial.suggest_categorical(
            "conv_layer_name",
            list(self.conv_layer_variants.keys())
        )
        return self.conv_layer_variants[self.conv_layer_name]

    def get_pooling_layer(self):
        self.pooling_layer_name = self.trial.suggest_categorical(
            "pooling_layer_name",
            list(self.pooling_layer_variants.keys())
        )
        pooling_layer = self.pooling_layer_variants[self.pooling_layer_name]
        if isfunction(pooling_layer):
            return pooling_layer
        if isclass(pooling_layer) and "in_channels" in signature(pooling_layer).parameters:
            params = {"in_channels": self.hidden_conv[-1]}
            return pooling_layer(**params)

    def get_conv_actf(self):
        self.conv_actf_name = self.trial.suggest_categorical(
            "conv_actf_name",
            list(self.conv_actf_variants.keys())
        )
        return self.conv_actf_variants[self.conv_actf_name]

    def get_linear_actf(self):
        self.linear_actf_name = self.trial.suggest_categorical(
            "linear_actf_name",
            list(self.linear_actf_variants.keys())
        )
        return self.linear_actf_variants[self.linear_actf_name]

    def get_optimizer(self):
        self.optimizer_name = self.trial.suggest_categorical(
            "optimizer_name",
            list(self.optimizer_variants.keys())
        )
        return self.optimizer_variants[self.optimizer_name]

    def get_optimizer_parameters(self):
        optimizer_parameters = {
            "lr": self.trial.suggest_float("lr", 1e-5, 1e-1),
        }
        return optimizer_parameters

    def get_conv_parameters(self):
        conv_parameters = {}
        if self.conv_layer.__name__ == "SSGConv":
            conv_parameters = {
                "alpha": 0.5,
            }
        return conv_parameters

    def get(self):
        self.n_pre_linear = self.trial.suggest_int("n_pre_linear", *self.n_pre_linear_lims)
        self.n_conv = self.trial.suggest_int("n_conv", *self.n_conv_lims)
        self.n_post_linear = self.trial.suggest_int("n_post_linear", *self.n_post_linear_lims)

        self.conv_dropout = self.trial.suggest_float("conv_dropout", *self.conv_dropout_lims)
        self.pre_linear_dropout = self.trial.suggest_float("pre_linear_dropout", *self.pre_linear_dropout_lims)
        self.post_linear_dropout = self.trial.suggest_float("post_linear_dropout", *self.post_linear_dropout_lims)
        self.post_linear_bn = self.trial.suggest_categorical("post_linear_bn", self.post_linear_bn_variants)
        self.conv_layer = self.get_conv_layer()
        self.conv_parameters = self.get_conv_parameters()
        self.conv_actf = self.get_conv_actf()
        self.pre_linear_actf = self.post_linear_actf = self.get_linear_actf()

        self.hidden_pre_linear = [self.trial.suggest_int(f"pre_linear_hidden_{i}", *self.hidden_lims)
                                  for i in range(self.n_conv)]
        self.hidden_conv = [self.trial.suggest_int(f"conv_hidden_{i}", *self.hidden_lims)
                            for i in range(self.n_pre_linear)]
        self.hidden_post_linear = [self.trial.suggest_int(f"post_linear_hidden_{i}", *self.hidden_lims)
                                   for i in range(self.n_post_linear)]
        self.pooling_layer = self.get_pooling_layer()

        self.optimizer = self.get_optimizer()
        self.optimizer_parameters = self.get_optimizer_parameters()

        model_parameters = {
            "conv_layer": self.conv_layer,
            "conv_parameters": self.conv_parameters,
            "pooling_layer": self.pooling_layer,
            "hidden_pre_linear": self.hidden_pre_linear,
            "hidden_conv": self.hidden_conv,
            "hidden_post_linear": self.hidden_post_linear,
            "conv_dropout": self.conv_dropout,
            "conv_actf": self.conv_actf,
            "pre_linear_dropout": self.pre_linear_dropout,
            "pre_linear_actf": self.pre_linear_actf,
            "post_linear_dropout": self.post_linear_dropout,
            "post_linear_bn": self.post_linear_bn,
            "post_linear_actf": self.post_linear_actf,
            "optimizer": self.optimizer,
            "optimizer_parameters": self.optimizer_parameters,
        }

        return model_parameters


def objective(trial: optuna.Trial):
    featurizer_variants = {
        "ConvMolFeaturizer": ConvMolFeaturizer(),
        "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
        "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
        "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
    }

    featurizer_name = trial.suggest_categorical("featurizer_name", list(featurizer_variants.keys()))
    featurizer = featurizer_variants[featurizer_name]

    if featurizer_name != "ConvMolFeaturizer":
        add_self_loop = trial.suggest_categorical("add_self_loop", (True, False))
        featurizer = DGLFeaturizer(add_self_loop=add_self_loop, node_featurizer=featurizer)

    data = featurize_csv_to_graph(path_to_csv=path_to_csv, targets=targets, featurizer=featurizer, max_data=max_data)
    folds, test_data = train_test_valid_split(data, n_folds=cv_folds, test_ratio=0.1, batch_size=batch_size)

    model_parameters = FCNNParams(
        trial,
        n_pre_linear_lims=(1, 2),
        n_conv_lims=(1, 3),
        n_post_linear_lims=(1, 5),
        conv_dropout_lims=(0, 0.8),
        pre_linear_dropout_lims=(0, 0.8),
        post_linear_dropout_lims=(0, 0),
        post_linear_bn_variants=(True, False),
        conv_layer_variants={
            "MFConv": MFConv,
            "GATConv": GATConv,
            "TransformerConv": TransformerConv,
            "SAGEConv": SAGEConv,
            "GCNConv": GCNConv,  # edge_weight
            "ARMAConv": ARMAConv,  # edge_weight
            "SSGConv": SSGConv,  # edge_weight
        },
        pooling_layer_variants={
            "global_mean_pool": global_mean_pool,
            "global_max_pool": global_max_pool,
            # "TopKPooling": TopKPooling, # edge_attrs
            # "SAGPooling": SAGPooling, # edge_attrs
            # "EdgePooling": EdgePooling,
            # "ASAPooling": ASAPooling, # edge_weight
        },
        conv_actf_variants={
            "nn.LeakyReLU()": nn.LeakyReLU(),
            "nn.PReLU()": nn.PReLU(),
            "nn.Tanhshrink()": nn.Tanhshrink(),
        },
        linear_actf_variants={
            "nn.LeakyReLU()": nn.LeakyReLU(),
            "nn.PReLU()": nn.PReLU(),
            "nn.Tanhshrink()": nn.Tanhshrink(),
        },
        optimizer_variants={
            "Adam": Adam,
            "AdamW": AdamW,
            "RMSprop": RMSprop,
        },
        hidden_lims=(32, 512, 64),
    ).get()
    model_parameters["node_features"] = get_num_node_features(folds[0][0])
    model_parameters["num_targets"] = get_num_targets(folds[0][0])
    model_parameters["batch_size"] = batch_size
    model_parameters["mode"] = "regression"

    model = GCNN(**model_parameters)

    time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    output_folder = os.path.join(general_output_path, f"GCNN_mouse_intraperitoneal_LD50_{mode}_{time_mark}")

    trainer = GCNNTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_data,
        output_folder=output_folder,
        epochs=epochs,
        es_patience=es_patience,
        target_metrics=target_metrics,
    )

    trainer.train_cv_models()

    trial.set_user_attr(key="metrics", value=trainer.results_dict["general"])
    trial.set_user_attr(key="model_parameters", value={key: str(model_parameters[key]) for key in model_parameters})

    errors = [trainer.results_dict["general"][key] for key in trainer.results_dict["general"] if "MSE" in key]
    return max(errors)


def callback(study: optuna.Study, trial):
    study.trials_dataframe().to_csv(path_or_buf=os.path.join(general_output_path, f"trials.csv"), index=False)


if __name__ == "__main__":
    start = datetime.now()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[callback])
    end = datetime.now()

    result = {
        "trials": len(study.trials),
        "started": str(start).split(".")[0],
        "finished": str(end).split(".")[0],
        "duration": str(end - start).split(".")[0],
        **study.best_trial.user_attrs["metrics"],
        "model_parameters": study.best_trial.user_attrs["model_parameters"],
    }

    time_mark = str(start).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(general_output_path, f"trials.csv"), index=False)

    with open(os.path.join(general_output_path, f"result.json"), "w") as jf:
        json.dump(result, jf)
