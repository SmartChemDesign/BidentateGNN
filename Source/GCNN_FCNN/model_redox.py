"""
GCNN-FCNN model for redox potentials with a molecular graph branch and a solvent
descriptor branch.

One class covers all training regimes used in the paper. The regime is set by the
training data and by `loss_mode`:
  - "multifidelity": weighted sum of the experimental (HF) and calculated (LF) terms;
  - "pooled": plain mean over all samples (experimental-only, calculated-only and
    joint training without fidelity weighting).

solvent_x layout: [scaled descriptors..., redox_type, data_type]. The data_type flag
is removed before the solvent FCNN and used only for loss weighting, so the network
input is identical in every regime.
"""

from inspect import signature

import torch

from Source.FCNN.model import FCNN
from Source.GCNN.model import GCNN
from Source.base_model import BaseModel_MF
from Source.global_poolings import ConcatPooling

DATA_TYPE_INDEX = -1
LOSS_MODES = ("multifidelity", "pooled")


class GCNN_FCNN(BaseModel_MF):

    def __init__(self, solvent_features, node_features, targets,
                 solvent_fc_params=None, gcnn_params=None, post_fc_params=None,
                 global_pooling=ConcatPooling, use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None,
                 hf_loss_weight=0.8, lf_loss_weight=0.2, loss_mode="multifidelity"):
        """
        Args:
            solvent_features (int): full dimensionality of solvent_x, including the
                data_type flag.
            node_features (int): dimensionality of the node features.
            targets (tuple): targets configuration; the loss must be element-wise
                (e.g. nn.MSELoss(reduction="none")).
            solvent_fc_params (dict): parameters of the solvent FCNN block.
            gcnn_params (dict): parameters of the GCNN block.
            post_fc_params (dict): parameters of the post-pooling FCNN block.
            global_pooling (class): pooling class combining the two branches.
            use_out_sequential (bool): use per-target output layers.
            optimizer (class): optimizer class.
            optimizer_parameters (dict): optimizer parameters.
            hf_loss_weight (float): weight of the experimental term ("multifidelity").
            lf_loss_weight (float): weight of the calculated term ("multifidelity").
            loss_mode (str): "multifidelity" or "pooled".
        """
        super().__init__(
            targets=targets,
            use_out_sequential=use_out_sequential,
            optimizer=optimizer,
            optimizer_parameters=optimizer_parameters,
            hf_loss_weight=hf_loss_weight,
            lf_loss_weight=lf_loss_weight,
        )
        if loss_mode not in LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {LOSS_MODES}, got {loss_mode!r}")

        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters}
        self.loss_mode = loss_mode

        for p in (solvent_fc_params, gcnn_params, post_fc_params):
            p["use_out_sequential"] = False
            p["targets"] = ()

        # The data_type flag is dropped before the solvent FCNN, hence one input less
        solvent_fc_params["input_features"] = solvent_features - 1
        gcnn_params["node_features"] = node_features

        self.graph_sequential = GCNN(**gcnn_params)
        self.solvent_fc_sequential = FCNN(**solvent_fc_params)
        self.global_pooling = global_pooling(
            input_dims=(self.graph_sequential.output_dim, self.solvent_fc_sequential.output_dim)
        )
        post_fc_params["input_features"] = self.global_pooling.output_dim
        self.post_fc_sequential = FCNN(**post_fc_params)

        self.last_common_dim = self.post_fc_sequential.output_dim
        self.configure_out_layer()

    def forward(self, graph, return_latent=False):
        """
        Args:
            graph: PyG batch providing x, edge_index, batch and solvent_x.
            return_latent (bool): also return the last latent representation.

        Returns:
            dict of predictions per target, optionally paired with the latent tensor.
        """
        x = self.graph_sequential(graph)

        solvent_x = graph.solvent_x
        if solvent_x.dim() == 3:
            solvent_x = solvent_x.squeeze(1)
        solvent = self.solvent_fc_sequential(self._drop_data_type(solvent_x))

        last_latent = self.post_fc_sequential(self.global_pooling(x, solvent))

        if self.use_out_sequential:
            output = {target: sequential(last_latent)
                      for target, sequential in self.out_sequentials.items()}
        else:
            output = last_latent

        return (output, last_latent) if return_latent else output

    @staticmethod
    def _drop_data_type(solvent_x):
        """Return solvent_x without the trailing data_type flag."""
        return solvent_x[..., :DATA_TYPE_INDEX]

    def _calculate_loss(self, batch, pred, true):
        """
        Loss of one batch.

        "multifidelity": hf_loss_weight * mean(HF) + lf_loss_weight * mean(LF), where
        HF are experimental and LF calculated samples (batch.data_type == "exp" / "calc").
        "pooled": mean over all samples.
        """
        mask_hf = torch.tensor([t == "exp" for t in batch.data_type],
                               dtype=torch.bool, device=self.device)
        mask_lf = ~mask_hf

        total_loss = 0.0
        for target in self.targets:
            name = target["name"]
            loss = target["loss"](pred[name], true[name])
            while loss.ndim > 1:
                loss = loss.mean(dim=-1)

            loss_hf = loss[mask_hf].mean() if mask_hf.any() else loss.new_zeros(())
            loss_lf = loss[mask_lf].mean() if mask_lf.any() else loss.new_zeros(())

            if self.loss_mode == "multifidelity":
                total_loss = total_loss + self.hf_loss_weight * loss_hf + self.lf_loss_weight * loss_lf
            else:
                total_loss = total_loss + loss.mean()

            if self.training:
                self.log(f"{name}_loss_hf", loss_hf, on_step=False, on_epoch=True)
                self.log(f"{name}_loss_lf", loss_lf, on_step=False, on_epoch=True)
                self.log(f"{name}_num_hf", float(mask_hf.sum()), on_step=False, on_epoch=True)
                self.log(f"{name}_num_lf", float(mask_lf.sum()), on_step=False, on_epoch=True)

        return total_loss
