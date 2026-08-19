import warnings
from inspect import signature
import logging

import torch
import torch.optim.optimizer

from Source.FCNN.model import FCNN
from Source.GCNN.model import GCNN
from Source.base_model import BaseModel, BaseModel_MF  
from Source.global_poolings import ConcatPooling

logger = logging.getLogger(__name__)

class GCNN_FCNN(BaseModel):
    def __init__(self, solvent_features, node_features, targets,
                 solvent_fc_params=None, gcnn_params=None, post_fc_params=None, global_pooling=ConcatPooling,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None): 
        """
        GCNN_FCNN model with multi-fidelity support.
        
        Args:
            solvent_features (int): Number of solvent features
            node_features (int): Number of node features
            targets (tuple): Target configuration
            solvent_fc_params (dict): Parameters for solvent FC layers
            gcnn_params (dict): Parameters for GCNN layers
            post_fc_params (dict): Parameters for post-processing FC layers
            global_pooling (class): Pooling class for combining features
            use_out_sequential (bool): Use output sequential layers
            optimizer (class): Optimizer class
            optimizer_parameters (dict): Optimizer parameters
            hf_loss_weight (float): Weight for high-fidelity (experimental) data loss
            lf_loss_weight (float): Weight for low-fidelity (calculated) data loss
        """
   
        super(GCNN_FCNN, self).__init__(
            targets=targets, 
            use_out_sequential=use_out_sequential, 
            optimizer=optimizer, 
            optimizer_parameters=optimizer_parameters
        )
        
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        # preparing params for model blocks
        for p in [solvent_fc_params, gcnn_params, post_fc_params]:
            if p is not None and "targets" in p and len(p["targets"]) > 0:
                warnings.warn(
                    "Not recommended to set 'targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if p is not None and "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            if p is not None:
                p["use_out_sequential"] = False
                p["targets"] = ()

        if solvent_fc_params is not None:
            solvent_fc_params["input_features"] = solvent_features
        if gcnn_params is not None:
            gcnn_params["node_features"] = node_features
            
        self.graph_sequential = GCNN(**gcnn_params)
        self.solvent_fc_sequential = FCNN(**solvent_fc_params)

        self.global_pooling = global_pooling(input_dims=(self.graph_sequential.output_dim,
                                                         self.solvent_fc_sequential.output_dim))

        if post_fc_params is not None:
            post_fc_params["input_features"] = self.global_pooling.output_dim
        self.post_fc_sequential = FCNN(**post_fc_params)

        self.last_common_dim = (self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1]
        self.configure_out_layer()

    def forward(self, graph, return_latent=False):
        """
        Forward pass through the model.
        
        Args:
            graph: PyTorch Geometric Data object with:
                - x: Node features
                - edge_index: Graph connectivity
                - solvent_x: Solvent features
                - batch: Batch indices
            return_latent (bool): If True, return latent representation along with output
            
        Returns:
            dict or tuple: Model predictions (and latent representation if return_latent=True)
        """
        x = self.graph_sequential(graph)
        solvent = self.solvent_fc_sequential(graph.solvent_x)
        general = self.global_pooling(x, solvent)
        last_latent = self.post_fc_sequential(general)
        
        if self.use_out_sequential:
            general = {target: sequential(last_latent) for target, sequential in self.out_sequentials.items()}

        return (general, last_latent) if return_latent else general


class GCNN_FCNN_MaskedDataType(BaseModel_MF):
    """GCNN-FCNN model that masks the data-type flag during the forward pass."""

    def __init__(self, solvent_features, node_features, targets,
                 solvent_fc_params=None, gcnn_params=None, post_fc_params=None,
                 global_pooling=ConcatPooling,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None,
                 hf_loss_weight=0.8, lf_loss_weight=0.2,
                 data_type_index=-2):
        """
        Args:
            solvent_features (int): full dimensionality of solvent_x, including
                the data-type flag.
            node_features (int): dimensionality of the node features.
            targets (tuple): targets configuration.
            solvent_fc_params (dict): parameters of the solvent FCNN block.
            gcnn_params (dict): parameters of the GCNN block.
            post_fc_params (dict): parameters of the post-pooling FCNN block.
            global_pooling (class): pooling class.
            use_out_sequential (bool): use per-target output layers.
            optimizer (class): optimizer class.
            optimizer_parameters (dict): optimizer parameters.
            hf_loss_weight (float): weight of the high-fidelity (experimental) term.
            lf_loss_weight (float): weight of the low-fidelity (calculated) term.
            data_type_index (int): position of the data-type flag within solvent_x;
                -2 by default, -1 for the last element, None to disable masking.
        """
        super().__init__(
            targets=targets,
            use_out_sequential=use_out_sequential,
            optimizer=optimizer,
            optimizer_parameters=optimizer_parameters,
            hf_loss_weight=hf_loss_weight,
            lf_loss_weight=lf_loss_weight
        )

        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        self.data_type_index = data_type_index

        # The data-type flag is dropped before the FCNN, hence one input less
        if data_type_index is None:
            actual_solvent_features = solvent_features
        else:
            actual_solvent_features = solvent_features - 1
        logger.info(
            "Data type masking %s: solvent_x dimension %d, FCNN input dimension %d",
            "disabled" if data_type_index is None else f"enabled at index {data_type_index}",
            solvent_features,
            actual_solvent_features
        )

        for p in (solvent_fc_params, gcnn_params, post_fc_params):
            if p is None:
                continue
            if len(p.get("targets", ())) > 0:
                warnings.warn(
                    "Not recommended to set 'targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["targets"] = ()

        if solvent_fc_params is not None:
            solvent_fc_params["input_features"] = actual_solvent_features
        if gcnn_params is not None:
            gcnn_params["node_features"] = node_features

        self.graph_sequential = GCNN(**gcnn_params)
        self.solvent_fc_sequential = FCNN(**solvent_fc_params)

        self.global_pooling = global_pooling(
            input_dims=(self.graph_sequential.output_dim, self.solvent_fc_sequential.output_dim)
        )

        if post_fc_params is not None:
            post_fc_params["input_features"] = self.global_pooling.output_dim
        self.post_fc_sequential = FCNN(**post_fc_params)

        self.last_common_dim = (self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1]
        self.configure_out_layer()

    def forward(self, graph, return_latent=False):
        """Run the forward pass, masking the data-type flag of the solvent vector.

        Args:
            graph: PyG Data object providing x, edge_index, solvent_x and batch.
            return_latent (bool): also return the last latent representation.

        Returns:
            Model predictions, optionally paired with the latent representation.
        """
        x = self.graph_sequential(graph)

        solvent_x = graph.solvent_x
        if solvent_x.dim() == 3:
            solvent_x = solvent_x.squeeze(1)

        if self.data_type_index is None:
            solvent_x_masked = solvent_x
        else:
            solvent_x_masked = self._mask_data_type(solvent_x)

        solvent = self.solvent_fc_sequential(solvent_x_masked)

        general = self.global_pooling(x, solvent)
        last_latent = self.post_fc_sequential(general)

        if self.use_out_sequential:
            general = {target: sequential(last_latent)
                       for target, sequential in self.out_sequentials.items()}

        return (general, last_latent) if return_latent else general

    def _mask_data_type(self, solvent_x):
        """Return solvent_x with the data-type flag removed along the last axis."""
        if self.data_type_index == -2:
            # Layout [...descriptors..., data_type, redox_type]: keep the descriptors
            # and the trailing redox_type
            return torch.cat([solvent_x[..., :-2], solvent_x[..., -1:]], dim=-1)

        if self.data_type_index == -1:
            return solvent_x[..., :-1]

        indices = list(range(solvent_x.size(-1)))
        indices.pop(self.data_type_index)
        return solvent_x[..., indices]

    def _calculate_loss(self, batch, pred, true):
        """Compute the multifidelity loss as a weighted sum of the HF and LF terms.

        The data-type flag is taken from batch.data_type when present and recovered
        from batch.solvent_x otherwise.
        """
        total_loss = 0.0

        for target in self.targets:
            name = target["name"]
            y_pred = pred[name]
            y_true = true[name]

            if hasattr(batch, "data_type"):
                dt = batch.data_type
            else:
                solvent_x = batch.solvent_x
                if solvent_x.dim() == 3:
                    solvent_x = solvent_x.squeeze(1)
                data_type_values = solvent_x[:, self.data_type_index]
                dt = ["exp" if val > 0.5 else "calc" for val in data_type_values.cpu().tolist()]

            if isinstance(dt, list):
                mask_hf = torch.tensor([t == "exp" for t in dt], dtype=torch.bool, device=y_pred.device)
                mask_lf = torch.tensor([t == "calc" for t in dt], dtype=torch.bool, device=y_pred.device)
            elif isinstance(dt, torch.Tensor) and dt.dtype == torch.bool:
                mask_hf = dt
                mask_lf = ~dt
            elif isinstance(dt, torch.Tensor) and dt.dtype in (torch.long, torch.int):
                mask_hf = dt == 1
                mask_lf = dt == 0
            elif isinstance(dt, torch.Tensor):
                raise ValueError(f"Unsupported data_type tensor dtype: {dt.dtype}")
            else:
                # Unknown container: treat every record as high-fidelity
                mask_hf = torch.ones(y_pred.size(0), dtype=torch.bool, device=y_pred.device)
                mask_lf = torch.zeros_like(mask_hf)

            elementwise_loss = target["loss"](y_pred, y_true)
            while elementwise_loss.ndim > 1:
                elementwise_loss = elementwise_loss.mean(dim=-1)

            num_hf = mask_hf.sum().item()
            num_lf = mask_lf.sum().item()
            zero = torch.tensor(0.0, device=y_pred.device)
            loss_hf = elementwise_loss[mask_hf].mean() if num_hf > 0 else zero
            loss_lf = elementwise_loss[mask_lf].mean() if num_lf > 0 else zero

            total_loss += self.hf_loss_weight * loss_hf + self.lf_loss_weight * loss_lf

            if self.training:
                self.log(f"{name}_loss_hf", loss_hf, prog_bar=False)
                self.log(f"{name}_loss_lf", loss_lf, prog_bar=False)
                self.log(f"{name}_num_hf", float(num_hf), prog_bar=False)
                self.log(f"{name}_num_lf", float(num_lf), prog_bar=False)

        return total_loss


def verify_model_masking(model, sample_batch):
    """Check that the data-type flag is removed from the solvent vector.

    Args:
        model: GCNN_FCNN_MaskedDataType instance.
        sample_batch: batch of data used for the check.

    Returns:
        bool: True if the masked dimensionality and the forward pass are consistent.
    """
    solvent_x = sample_batch.solvent_x
    if solvent_x.dim() == 3:
        solvent_x = solvent_x.squeeze(1)
    logger.info("Input solvent_x dimension: %s", tuple(solvent_x.shape))

    if model.data_type_index is None:
        logger.info("Masking is disabled, dimensionality check skipped")
    else:
        solvent_x_masked = model._mask_data_type(solvent_x)
        expected_dim = solvent_x.size(-1) - 1
        logger.info(
            "Masked solvent_x dimension: %s, expected %d",
            tuple(solvent_x_masked.shape),
            expected_dim
        )
        if solvent_x_masked.size(-1) != expected_dim:
            logger.error("Masking failed: expected %d features, got %d",
                         expected_dim, solvent_x_masked.size(-1))
            return False

    model.eval()
    with torch.no_grad():
        output = model(sample_batch)
    output_shape = output["E"].shape if isinstance(output, dict) else output.shape
    logger.info("Forward pass succeeded, output shape %s", tuple(output_shape))

    if hasattr(sample_batch, "data_type"):
        logger.info("Data type taken from batch.data_type")
    else:
        logger.info("Data type recovered from solvent_x at index %d", model.data_type_index)

    return True