import optuna
from torch.optim import Adam

from Source.models.FCNN.featurizers import ECFPMolFeaturizer


class GeneralParams:
    def __init__(self, trial: optuna.Trial, optimizer_variants, lr_lims, featurizer_variants, batch_size_lims):
        self.trial = trial
        self.featurizer_variants = featurizer_variants or {
            "ECFPMolFeaturizer": ECFPMolFeaturizer(),
        }
        self.optimizer_variants = optimizer_variants or {"Adam": Adam}
        self.lr_lims = lr_lims
        self.batch_size_lims = batch_size_lims

    def get_optimizer(self):
        self.optimizer_name = self.trial.suggest_categorical(
            "optimizer_name",
            list(self.optimizer_variants.keys())
        )
        return self.optimizer_variants[self.optimizer_name]

    def get_optimizer_parameters(self):
        optimizer_parameters = {
            "lr": self.trial.suggest_float("lr", *self.lr_lims),
        }
        return optimizer_parameters

    def get_featurizer(self):
        featurizer_name = self.trial.suggest_categorical("featurizer_name", list(self.featurizer_variants.keys()))
        featurizer = self.featurizer_variants[featurizer_name]
        return featurizer

    def get(self):
        return {
            "optimizer": self.get_optimizer(),
            "optimizer_parameters": self.get_optimizer_parameters(),
            "featurizer": self.get_featurizer(),
            "batch_size": self.trial.suggest_int("batch_size", *self.batch_size_lims),
        }


class FCNNParams:
    def __init__(self, trial: optuna.Trial,
                 dim_lims, n_layers_lims, actf_variants, dropout_lims, bn_variants, prefix=""):
        self.trial = trial
        self.prefix = prefix
        self.dim_lims = dim_lims
        self.n_layers_lims = n_layers_lims
        self.dropout_lims = dropout_lims
        self.bn_variants = bn_variants
        self.actf_variants = actf_variants

    def get_activation(self):
        self.actf_name = self.trial.suggest_categorical(
            f"{self.prefix}_actf_name",
            list(self.actf_variants.keys())
        )
        return self.actf_variants[self.actf_name]

    def get(self):
        self.n_layers = self.trial.suggest_int(f"{self.prefix}_n_layers", *self.n_layers_lims)
        self.dropout = self.trial.suggest_float(f"{self.prefix}_dropout", *self.dropout_lims)
        self.use_batch_norm = self.trial.suggest_categorical(f"{self.prefix}_use_batch_norm", self.bn_variants)
        self.activation = self.get_activation()
        self.hidden = [
            self.trial.suggest_int(f"{self.prefix}_hidden_{i}", *self.dim_lims, log=True)
            for i in range(self.n_layers)
        ]

        model_parameters = {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "use_bn": self.use_batch_norm,
            "actf": self.activation,
        }

        return model_parameters
