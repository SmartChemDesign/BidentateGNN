import warnings
from inspect import signature

import torch.nn as nn
import torch.optim.optimizer
from torch_geometric.nn import global_mean_pool, Sequential
from torch_geometric.nn.conv import MFConv
from torch_geometric.utils import add_self_loops

from Source.FCNN.model import FCNN
from Source.base_model import BaseModel


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
            print(p)
            # if "targets" in p and len(p["targets"]) > 0:
            #     warnings.warn(
            #         "Not recommended to set 'targets' in model blocks as far as it doesn't affect anything",
            #         DeprecationWarning)
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

    def freeze_layers(self, number_of_layers: int) -> None:
        n_c = 2
        n_l = 1
        parameters = list(self.named_parameters())
        if number_of_layers <= n_c:
            for i in range(number_of_layers):
                for name, tensor in parameters:
                    if name.startswith(f'conv_sequential.module_{i}'):
                        tensor.requires_grad = False

        elif n_c + n_l >= number_of_layers > n_c:
            for i in range(n_c):
                for name, tensor in parameters:
                    if name.startswith(f"conv_sequential.module_{i}"):
                        tensor.requires_grad = False
                        print(tensor.requires_grad)
            for i in range(number_of_layers - n_c):
                for name, tensor in parameters:
                    if name.startswith(f"post_fc_sequential.fc_sequential.{i}"):
                        tensor.requires_grad = False
                        print(tensor.requires_grad)
        else:
            warnings.warn("Number of layers in model lower than requested freeze number")
            for name, tensor in parameters:
                tensor.requires_grad = False