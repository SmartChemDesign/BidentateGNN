from inspect import signature

import torch.nn as nn
import torch.optim.optimizer

from Source.base_model import BaseModel


class FCNN(BaseModel):
    def __init__(self, input_features, targets,
                 hidden=(64,), dropout=0, use_bn=False, actf=nn.LeakyReLU(),
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(FCNN, self).__init__(targets, use_out_sequential, optimizer, optimizer_parameters)
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        self.input_features = input_features
        self.hidden = hidden
        self.use_bn = use_bn
        self.dropout = dropout
        self.actf = actf

        self.fc_sequential = self.make_fc_blocks(hidden_dims=(input_features, *hidden),
                                                 actf=actf,
                                                 batch_norm=False,
                                                 dropout=dropout)

        self.last_common_dim = (input_features, *hidden)[-1]
        self.configure_out_layer()

    @staticmethod
    def make_fc_blocks(hidden_dims, actf, batch_norm=False, dropout=0.0):
        def fc_layer(in_f, out_f):
            layers = [nn.Linear(in_f, out_f), nn.Dropout(dropout), actf]
            if batch_norm: layers.insert(1, nn.BatchNorm1d(out_f))
            return nn.Sequential(*layers)

        lin_layers = [fc_layer(hidden_dims[i], hidden_dims[i + 1]) for i, val in enumerate(hidden_dims[:-1])]
        return nn.Sequential(*lin_layers)

    def forward(self, x):
        x = self.fc_sequential(x)
        if self.use_out_sequential:
            x = {target: sequential(x) for target, sequential in self.out_sequentials.items()}
        return x

    def validation_step(self, val_batch, *args, **kwargs):
        x, true = val_batch
        pred = self.forward(x)
        loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
        self.log('val_loss', loss)
        return loss
