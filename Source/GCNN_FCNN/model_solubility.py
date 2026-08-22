import warnings
import torch.optim.optimizer

from inspect import signature
from Source.FCNN.model import FCNN
from Source.GCNN.model import GCNN
from Source.base_model import BaseModel
from Source.global_poolings import ConcatPooling

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

    def forward(self, graph, return_latent=False):
        x_graph = self.graph_sequential(graph)
        x_fully_connected = self.metal_fc_sequential(graph.x_fully_connected)
        general = self.global_pooling(x_graph, x_fully_connected)
        last_latent = self.post_fc_sequential(general)
        if self.use_out_sequential:
            general = {target: sequential(last_latent) for target, sequential in self.out_sequentials.items()}

        return (general, last_latent) if return_latent else general
