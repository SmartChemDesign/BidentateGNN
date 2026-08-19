from inspect import isfunction, isclass, signature

import optuna

from Source.models.FCNN.optimize_hparams import GeneralParams as FCNNGeneralParams, FCNNParams
from Source.models.GCNN.featurizers import DGLFeaturizer


class GeneralParams(FCNNGeneralParams):
    def __init__(self, trial: optuna.Trial, *args, **kwargs):
        super(GeneralParams, self).__init__(trial, *args, **kwargs)

    def get_featurizer(self):
        featurizer = super().get_featurizer()
        add_self_loop = self.trial.suggest_categorical("add_self_loop", (True, False))
        featurizer = DGLFeaturizer(add_self_loop=add_self_loop, node_featurizer=featurizer)

        return featurizer


class GCNNParams:
    def __init__(self, trial: optuna.Trial,
                 pre_fc_params, post_fc_params,
                 n_conv_lims, dropout_lims,
                 actf_variants, dim_lims,
                 conv_layer_variants, pooling_layer_variants, prefix=""):
        self.trial = trial
        self.prefix = prefix
        self.pre_fc_params = pre_fc_params
        self.post_fc_params = post_fc_params
        self.conv_layer_variants = conv_layer_variants
        self.pooling_layer_variants = pooling_layer_variants
        self.n_layers_lims = n_conv_lims
        self.dim_lims = dim_lims
        self.dropout_lims = dropout_lims
        self.actf_variants = actf_variants

        self.pre_fc_params = FCNNParams(self.trial, **self.pre_fc_params, prefix="pre_fc").get()
        self.post_fc_params = FCNNParams(self.trial, **self.post_fc_params, prefix="post_fc").get()

        self.conv_layer_name = None
        self.pooling_layer_name = None
        self.actf_name = None

    def get_conv_layer(self):
        self.conv_layer_name = self.trial.suggest_categorical(
            f"{self.prefix}_conv_layer_name",
            list(self.conv_layer_variants.keys())
        )
        self.conv_layer = self.conv_layer_variants[self.conv_layer_name]
        return self.conv_layer

    def get_pooling_layer(self):
        self.pooling_layer_name = self.trial.suggest_categorical(
            f"{self.prefix}_pooling_layer_name",
            list(self.pooling_layer_variants.keys())
        )
        pooling_layer = self.pooling_layer_variants[self.pooling_layer_name]
        if isfunction(pooling_layer):
            return pooling_layer
        if isclass(pooling_layer) and "in_channels" in signature(pooling_layer).parameters:
            params = {"in_channels": self.dim_lims[-1]}
            return pooling_layer(**params)

    def get_actf(self):
        self.actf_name = self.trial.suggest_categorical(
            f"{self.prefix}_actf_name",
            list(self.actf_variants.keys())
        )
        return self.actf_variants[self.actf_name]

    def get_conv_parameters(self):
        conv_parameters = {}
        if self.conv_layer.__name__ == "SSGConv":
            conv_parameters = {
                "alpha": 0.5,
            }
        return conv_parameters

    def get_hidden_conv(self):
        n_layers = self.trial.suggest_int(f"{self.prefix}_n_layers", *self.n_layers_lims)
        dims = [
            self.trial.suggest_int(f"{self.prefix}_hidden_{i}", *self.dim_lims, log=True)
            for i in range(n_layers)
        ]
        return dims

    def get(self):

        model_parameters = {
            "pre_fc_params": self.pre_fc_params,
            "post_fc_params": self.post_fc_params,
            "hidden_conv": self.get_hidden_conv(),
            "conv_dropout": self.trial.suggest_float(f"{self.prefix}_dropout", *self.dropout_lims),
            "conv_actf": self.get_actf(),
            "conv_layer": self.get_conv_layer(),
            "conv_parameters": self.get_conv_parameters(),
            "graph_pooling": self.get_pooling_layer()
        }

        return model_parameters
