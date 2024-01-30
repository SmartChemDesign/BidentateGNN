import warnings
import json

from inspect import signature

from pytorch_lightning import LightningModule
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim.optimizer
from torch_geometric.nn import global_mean_pool, Sequential
from torch_geometric.nn.conv import MFConv
from torch_geometric.utils import add_self_loops

from Source.GCNN_FCNN.global_poolings import ConcatPooling


class BaseModel(LightningModule):
    def __init__(self, targets, use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(BaseModel, self).__init__()

        self.targets = targets
        self.use_out_sequential = use_out_sequential

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}

        self.valid_losses = []
        self.train_losses = []

    def configure_out_layer(self):
        self.out_sequentials = ModuleDict()
        if self.use_out_sequential:
            self.output_dim = 0
            for target in self.targets:
                self.output_dim += target["dim"]
                if target["mode"] == "regression":
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device))
                elif target["mode"] == "binary_classification":
                    if target["dim"] != 1:
                        raise ValueError(
                            f"Target '{target['name']}': binary_classification requires dim=1, got {target['dim']} instead")
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device),
                        nn.Sigmoid())
                elif target["mode"] == "multiclass_classification":
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device),
                        nn.Softmax())
                else:
                    raise ValueError(
                        "Invalid mode value, only 'regression', 'binary_classification' or 'multiclass_classification' are allowed")
        else:
            self.output_dim = self.last_common_dim

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.2, patience=20, verbose=True),
                "monitor": "val_loss",
                "frequency": 1  # should be set to "trainer.check_val_every_n_epoch"
            },
        }

    def training_step(self, train_batch, *args, **kwargs):
        pred = self.forward(train_batch)
        true = train_batch.y
        loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
        self.log('train_loss', loss, batch_size=train_batch.batch.max() + 1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        pred = self.forward(val_batch)
        true = val_batch.y
        loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
        self.log('val_loss', loss, batch_size=val_batch.batch.max() + 1)
        return loss

    def get_model_structure(self):
        def make_jsonable(x):
            try:
                json.dumps(x)
                return x
            except (TypeError, OverflowError):
                if isinstance(x, dict):
                    return {key: make_jsonable(value) for key, value in x.items()}
                return str(x)

        return make_jsonable(self.config)




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
            x = self.out_sequential(x)
        return x

    def training_step(self, train_batch, *args, **kwargs):
        x, true = train_batch
        pred = self.forward(x)
        loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        x, true = val_batch
        pred = self.forward(x)
        loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
        self.log('val_loss', loss)
        return loss


class GCNN(BaseModel):
    def __init__(self, node_features, targets,
                 pre_fc_params=None, hidden_conv=(64,), conv_dropout=0, conv_actf=nn.LeakyReLU(), post_fc_params=None,
                 conv_layer=MFConv, conv_parameters=None, graph_pooling=global_mean_pool,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(GCNN, self).__init__(targets, use_out_sequential, optimizer, optimizer_parameters)
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        self.node_features = node_features
        self.conv_layer = conv_layer
        self.conv_parameters = conv_parameters or {}
        self.hidden_conv = hidden_conv
        self.conv_dropout = conv_dropout
        self.conv_actf = conv_actf

        # preparing params for fully connected blocks
        for p in [pre_fc_params, post_fc_params]:
            if "targets" in p and len(p["targets"]) > 0:
                warnings.warn(
                    "Not recommended to set 'targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["targets"] = ()
        if "use_bn" in pre_fc_params and pre_fc_params["use_bn"]:
            warnings.warn("Can't use batch normalization in FCNN on separate nodes")
            pre_fc_params["use_bn"] = False

        self.use_out_sequential = use_out_sequential

        pre_fc_params["input_features"] = node_features
        self.pre_fc_sequential = FCNN(**pre_fc_params)
        self.conv_sequential = self.make_conv_blocks(hidden_dims=(self.pre_fc_sequential.output_dim, *hidden_conv),
                                                     actf=conv_actf,
                                                     layer=conv_layer,
                                                     layer_parameters=conv_parameters,
                                                     dropout=conv_dropout)
        self.graph_pooling = graph_pooling
        post_fc_params["input_features"] = (self.pre_fc_sequential.output_dim, *hidden_conv)[-1]
        self.post_fc_sequential = FCNN(**post_fc_params)

        self.last_common_dim = self.post_fc_sequential.output_dim
        self.configure_out_layer()

    @staticmethod
    def make_conv_blocks(hidden_dims, actf, layer, layer_parameters=None, dropout=0.0):
        layer_parameters = layer_parameters or {}

        def conv_block(in_f, out_f):
            layers = [(layer(in_f, out_f, **layer_parameters), 'x, edge_index -> x'),
                      nn.Dropout(dropout),
                      actf]
            return Sequential("x, edge_index", layers)

        conv_layers = [(conv_block(hidden_dims[i], hidden_dims[i + 1]), 'x, edge_index -> x')
                       for i in range(len(hidden_dims) - 1)]
        return Sequential("x, edge_index", conv_layers)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.pre_fc_sequential(x)
        x = self.conv_sequential(x, edge_index)
        x = self.graph_pooling(x, batch=batch)
        x = self.post_fc_sequential(x)
        if self.use_out_sequential:
            x = {target: sequential(x) for target, sequential in self.out_sequentials.items()}
        return x

class GCNN_FCNN(BaseModel):
    def __init__(self, metal_features, node_features, targets,
                 metal_fc_params=None, gcnn_params=None, post_fc_params=None, global_pooling=ConcatPooling,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(GCNN_FCNN, self).__init__(targets, use_out_sequential, optimizer, optimizer_parameters)
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        # preparing params for model blocks
        for p in [metal_fc_params, gcnn_params, post_fc_params]:
            if "targets" in p and len(p["targets"]) > 0:
                warnings.warn(
                    "Not recommended to set 'targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["targets"] = ()

        metal_fc_params["input_features"] = metal_features
        gcnn_params["node_features"] = node_features
        self.graph_sequential = GCNN(**gcnn_params)
        self.metal_fc_sequential = FCNN(**metal_fc_params)

        self.global_pooling = global_pooling(input_dims=(self.graph_sequential.output_dim,
                                                         self.metal_fc_sequential.output_dim))

        post_fc_params["input_features"] = self.global_pooling.output_dim
        self.post_fc_sequential = FCNN(**post_fc_params)

        self.last_common_dim = (self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1]
        self.configure_out_layer()

    def forward(self, graph):
        x = self.graph_sequential(graph)
        metal_x = self.metal_fc_sequential(graph.metal_x)
        general = self.global_pooling(x, metal_x)
        general = self.post_fc_sequential(general)
        if self.use_out_sequential:
            general = {target: sequential(general) for target, sequential in self.out_sequentials.items()}
        return general
