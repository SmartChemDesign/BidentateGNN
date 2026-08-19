# # import json
# # from collections import defaultdict

# # import mlflow
# # import torch.nn as nn
# # import torch.optim.optimizer
# # from pytorch_lightning import LightningModule
# # from torch.nn import ModuleDict
# # from torch.optim.lr_scheduler import ReduceLROnPlateau


# # class BaseModel(LightningModule):
# #     def __init__(self, targets, use_out_sequential=True,
# #                  optimizer=torch.optim.Adam, optimizer_parameters=None):
# #         super(BaseModel, self).__init__()

# #         self.targets = targets
# #         self.use_out_sequential = use_out_sequential

# #         self.optimizer = optimizer
# #         self.optimizer_parameters = optimizer_parameters or {}

# #         self.valid_losses = []
# #         self.train_losses = []
# #         self.metadata = defaultdict(None)

# #     def configure_out_layer(self):
# #         self.out_sequentials = ModuleDict()
# #         if self.use_out_sequential:
# #             self.output_dim = 0
# #             for target in self.targets:
# #                 self.output_dim += target["dim"]
# #                 if target["mode"] == "regression":
# #                     self.out_sequentials[target["name"]] = nn.Sequential(
# #                         nn.Linear(self.last_common_dim, target["dim"], device=self.device))
# #                 elif target["mode"] == "binary_classification":
# #                     if target["dim"] != 1:
# #                         raise ValueError(
# #                             f"Target '{target['name']}': binary_classification requires dim=1, got {target['dim']} instead")
# #                     self.out_sequentials[target["name"]] = nn.Sequential(
# #                         nn.Linear(self.last_common_dim, target["dim"], device=self.device),
# #                         target["activation"] if "activation" in target else nn.Sigmoid())
# #                 elif target["mode"] == "multiclass_classification":
# #                     self.out_sequentials[target["name"]] = nn.Sequential(
# #                         nn.Linear(self.last_common_dim, target["dim"], device=self.device),
# #                         target["activation"] if "activation" in target else nn.Softmax())
# #                 else:
# #                     raise ValueError(
# #                         "Invalid mode value, only 'regression', 'binary_classification' or 'multiclass_classification' are allowed")
# #         else:
# #             self.output_dim = self.last_common_dim

# #     def configure_optimizers(self):
# #         optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
# #         return {
# #             "optimizer": optimizer,
# #             "lr_scheduler": {
# #                 "scheduler": ReduceLROnPlateau(
# #                     optimizer,
# #                     factor    = 0.5,    
# #                     patience  = 20,     
# #                     threshold = 1e-5,   
# #                     cooldown  = 3,      
# #                     min_lr    = 1e-6,   
# #                     verbose   = False,  
# #                 ),
# #                 "monitor": "val_loss",
# #                 "frequency": 1  # should be set to "trainer.check_val_every_n_epoch"
# #             },
# #         }

# #     def training_step(self, train_batch, *args, **kwargs):
# #         pred = self.forward(train_batch)
# #         true = train_batch.y
# #         loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
# #         self.log('train_loss', loss, batch_size=train_batch.batch.max() + 1, prog_bar=True)
# #         self.train_losses.append(loss.item())  # ← ДОБАВИТЬ
# #         fold = self.metadata["fold_ind"]
# #         mlflow.log_metrics({f"train_loss_fold-{fold}": loss.item()}, step=self.global_step)
# #         return loss

# #     def validation_step(self, val_batch, *args, **kwargs):
# #         pred = self.forward(val_batch)
# #         true = val_batch.y
# #         loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
# #         self.log('val_loss', loss, batch_size=val_batch.batch.max() + 1, 
# #             on_step=False,   # ← НЕ логировать на каждом шаге
# #              on_epoch=True)
# #         self.valid_losses.append(loss.item())  # ← ДОБАВИТЬ
# #         fold = self.metadata["fold_ind"]
# #         mlflow.log_metrics({f"val_loss_fold-{fold}": loss.item()}, step=self.global_step)
# #         return loss


# #     def get_model_structure(self):
# #         def make_jsonable(x):
# #             try:
# #                 json.dumps(x)
# #                 return x
# #             except (TypeError, OverflowError):
# #                 if isinstance(x, dict):
# #                     return {key: make_jsonable(value) for key, value in x.items()}
# #                 return str(x)

# #         return make_jsonable(self.config)

# import json
# from collections import defaultdict

# import mlflow
# import torch.nn as nn
# import torch.optim.optimizer
# from pytorch_lightning import LightningModule
# from torch.nn import ModuleDict
# from torch.optim.lr_scheduler import ReduceLROnPlateau


# class BaseModel(LightningModule):
#     def __init__(self, targets, use_out_sequential=True,
#                  optimizer=torch.optim.Adam, optimizer_parameters=None):
#         super(BaseModel, self).__init__()

#         self.targets = targets
#         self.use_out_sequential = use_out_sequential

#         self.optimizer = optimizer
#         self.optimizer_parameters = optimizer_parameters or {}

#         self.valid_losses = []
#         self.train_losses = []
#         self.metadata = defaultdict(None)

#     def configure_out_layer(self):
#         self.out_sequentials = ModuleDict()
#         if self.use_out_sequential:
#             self.output_dim = 0
#             for target in self.targets:
#                 self.output_dim += target["dim"]
#                 if target["mode"] == "regression":
#                     self.out_sequentials[target["name"]] = nn.Sequential(
#                         nn.Linear(self.last_common_dim, target["dim"], device=self.device))
#                 elif target["mode"] == "binary_classification":
#                     if target["dim"] != 1:
#                         raise ValueError(
#                             f"Target '{target['name']}': binary_classification requires dim=1, got {target['dim']} instead")
#                     self.out_sequentials[target["name"]] = nn.Sequential(
#                         nn.Linear(self.last_common_dim, target["dim"], device=self.device),
#                         target["activation"] if "activation" in target else nn.Sigmoid())
#                 elif target["mode"] == "multiclass_classification":
#                     self.out_sequentials[target["name"]] = nn.Sequential(
#                         nn.Linear(self.last_common_dim, target["dim"], device=self.device),
#                         target["activation"] if "activation" in target else nn.Softmax())
#                 else:
#                     raise ValueError(
#                         "Invalid mode value, only 'regression', 'binary_classification' or 'multiclass_classification' are allowed")
#         else:
#             self.output_dim = self.last_common_dim

#     def configure_optimizers(self):
#         optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": ReduceLROnPlateau(optimizer, factor=0.2, patience=20, verbose=True),
#                 "monitor": "val_loss",
#                 "frequency": 1
#             },
#         }

#     def training_step(self, train_batch, *args, **kwargs):
#         pred = self.forward(train_batch)
#         true = train_batch.y
#         loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
#         # on_epoch=True — агрегируем по эпохе, чтобы EarlyStopping видел стабильное значение
#         self.log('train_loss', loss,
#                  batch_size=train_batch.batch.max() + 1,
#                  prog_bar=True,
#                  on_step=False,
#                  on_epoch=True)
#         self.train_losses.append(loss.item())
#         fold = self.metadata["fold_ind"]
#         mlflow.log_metrics({f"train_loss_fold-{fold}": loss.item()}, step=self.global_step)
#         return loss

#     def validation_step(self, val_batch, *args, **kwargs):
#         pred = self.forward(val_batch)
#         true = val_batch.y
#         loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
#         # on_epoch=True критично — EarlyStopping мониторит именно эпохальный val_loss
#         self.log('val_loss', loss,
#                  batch_size=val_batch.batch.max() + 1,
#                  on_step=False,
#                  on_epoch=True)
#         self.log("val_loss_exp", loss, batch_size=val_batch.batch.max() + 1,
#                  on_step=False,
#                  on_epoch=True)
#         self.valid_losses.append(loss.item())
#         fold = self.metadata["fold_ind"]
#         mlflow.log_metrics({f"val_loss_fold-{fold}": loss.item()}, step=self.global_step)
#         return loss

#     def get_model_structure(self):
#         def make_jsonable(x):
#             try:
#                 json.dumps(x)
#                 return x
#             except (TypeError, OverflowError):
#                 if isinstance(x, dict):
#                     return {key: make_jsonable(value) for key, value in x.items()}
#                 return str(x)

#         return make_jsonable(self.config)

# Base Lightning module for single-fidelity GNN models (e.g. GCNN).
# Supports single or dual validation dataloaders:
#   dataloader_idx=0 — primary val loader (all data); logs val_loss for ReduceLROnPlateau.
#   dataloader_idx=1 — exp-only loader; logs val_loss_exp for EarlyStopping / ModelCheckpoint.

import json
from collections import defaultdict

import mlflow
import torch
import torch.nn as nn
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(LightningModule):
    def __init__(self, targets, use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super().__init__()
        self.targets = targets
        self.use_out_sequential = use_out_sequential
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.valid_losses = []
        self.train_losses = []
        self.metadata = defaultdict(None)

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
                            f"Target '{target['name']}': binary_classification requires dim=1, "
                            f"got {target['dim']}")
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device),
                        target.get("activation", nn.Sigmoid()))
                elif target["mode"] == "multiclass_classification":
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device),
                        target.get("activation", nn.Softmax()))
                else:
                    raise ValueError(
                        "Invalid mode; allowed values: 'regression', "
                        "'binary_classification', 'multiclass_classification'")
        else:
            self.output_dim = self.last_common_dim

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.2, patience=20),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def training_step(self, train_batch, *args, **kwargs):
        pred = self.forward(train_batch)
        true = train_batch.y
        loss = sum(target["loss"](pred[target["name"]], true[target["name"]])
                   for target in self.targets)
        self.log("train_loss", loss,
                 batch_size=train_batch.batch.max().item() + 1,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.train_losses.append(loss.item())
        fold = self.metadata["fold_ind"]
        mlflow.log_metrics({f"train_loss_fold-{fold}": loss.item()}, step=self.global_step)
        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        pred = self.forward(val_batch)
        true = val_batch.y
        loss = sum(target["loss"](pred[target["name"]], true[target["name"]])
                   for target in self.targets)
        batch_size = val_batch.batch.max().item() + 1

        if dataloader_idx == 0:
            # Primary loader (all data). Used by ReduceLROnPlateau.
            self.log("val_loss", loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.valid_losses.append(loss.item())
            fold = self.metadata["fold_ind"]
            mlflow.log_metrics({f"val_loss_fold-{fold}": loss.item()}, step=self.global_step)
        elif dataloader_idx == 1:
            # Exp-only loader. Used by EarlyStopping and ModelCheckpoint.
            self.log("val_loss_exp", loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, add_dataloader_idx=False)

        return loss

    def get_model_structure(self):
        def make_jsonable(x):
            try:
                json.dumps(x)
                return x
            except (TypeError, OverflowError):
                if isinstance(x, dict):
                    return {k: make_jsonable(v) for k, v in x.items()}
                return str(x)

        return make_jsonable(self.config)

class BaseModel_MF(LightningModule):
    def __init__(self, targets, use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None,
                 hf_loss_weight=0.8, lf_loss_weight=0.2):
        super().__init__()
        self.targets = targets
        self.use_out_sequential = use_out_sequential
        self.hf_loss_weight = hf_loss_weight
        self.lf_loss_weight = lf_loss_weight
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.valid_losses = []
        self.train_losses = []
        self.metadata = defaultdict(lambda: None)

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
                            f"Target '{target['name']}': binary_classification requires dim=1, "
                            f"got {target['dim']}")
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device),
                        target.get("activation", nn.Sigmoid()))
                elif target["mode"] == "multiclass_classification":
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device),
                        target.get("activation", nn.Softmax()))
                else:
                    raise ValueError(
                        "Invalid mode; allowed values: 'regression', "
                        "'binary_classification', 'multiclass_classification'")
        else:
            self.output_dim = self.last_common_dim

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # Monitors mixed val_loss to track overall training progress.
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.2, patience=10),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def _get_dt_masks(self, batch):
        """
        Extract HF/LF masks from the last column of solvent_x.

        solvent_x layout (after featurization):
            [0 : -2]  physical descriptors (scaled)
            [-2]      redox_type binary
            [-1]      data_type binary: 1 = exp (HF), 0 = calc (LF)
        """
        solvent_x = batch.solvent_x
        if solvent_x.ndim == 3:
            solvent_x = solvent_x.squeeze(1)
        dt_binary = solvent_x[:, -1]
        return dt_binary.bool(), ~dt_binary.bool()

    def _calculate_loss(self, batch, pred, true):
        """
        Multi-fidelity weighted loss.
        Total exp contribution = hf_loss_weight; total calc contribution = lf_loss_weight,
        regardless of the ratio of exp to calc samples in the batch.
        """
        mask_hf, mask_lf = self._get_dt_masks(batch)
        num_hf = mask_hf.sum().clamp(min=1)
        num_lf = mask_lf.sum().clamp(min=1)

        ref = (batch.solvent_x[:, 0] if batch.solvent_x.ndim == 2
               else batch.solvent_x[:, 0, 0])
        sample_weights = torch.where(
            mask_hf,
            torch.full_like(ref, self.hf_loss_weight) / num_hf,
            torch.full_like(ref, self.lf_loss_weight) / num_lf,
        )

        total_loss = 0.0
        for target in self.targets:
            name = target["name"]
            elementwise_loss = target["loss"](pred[name], true[name])
            while elementwise_loss.ndim > 1:
                elementwise_loss = elementwise_loss.mean(dim=-1)
            total_loss += (elementwise_loss * sample_weights).sum()

            if self.training:
                loss_hf = (elementwise_loss[mask_hf].mean() if mask_hf.any()
                           else torch.tensor(0.0, device=pred[name].device))
                loss_lf = (elementwise_loss[mask_lf].mean() if mask_lf.any()
                           else torch.tensor(0.0, device=pred[name].device))
                self.log(f"{name}_loss_hf", loss_hf, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{name}_loss_lf", loss_lf, prog_bar=False, on_step=False, on_epoch=True)

        return total_loss

    def training_step(self, train_batch, *args, **kwargs):
        pred = self.forward(train_batch)
        loss = self._calculate_loss(train_batch, pred, train_batch.y)
        batch_size = (train_batch.batch.max().item() + 1
                      if hasattr(train_batch, "batch") else len(train_batch))
        self.log("train_loss", loss, batch_size=batch_size,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.train_losses.append(loss.item())
        if self.metadata["fold_ind"] is not None:
            mlflow.log_metrics(
                {f"train_loss_fold-{self.metadata['fold_ind']}": loss.item()},
                step=self.global_step,
            )
        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        pred = self.forward(val_batch)
        true = val_batch.y
        batch_size = (val_batch.batch.max().item() + 1
                      if hasattr(val_batch, "batch") else len(val_batch))

        if dataloader_idx == 0:
            # Mixed loader (calc + exp). Used by ReduceLROnPlateau.
            loss = self._calculate_loss(val_batch, pred, true)
            self.log("val_loss", loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.valid_losses.append(loss.item())
            if self.metadata["fold_ind"] is not None:
                mlflow.log_metrics(
                    {f"val_loss_fold-{self.metadata['fold_ind']}": loss.item()},
                    step=self.global_step,
                )
        elif dataloader_idx == 1:
            # Exp-only loader. Used by EarlyStopping and ModelCheckpoint.
            exp_loss = sum(
                target["loss"](pred[target["name"]], true[target["name"]]).mean()
                for target in self.targets
            )
            self.log("val_loss_exp", exp_loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, add_dataloader_idx=False)

        return pred

    def get_model_structure(self):
        def make_jsonable(x):
            try:
                json.dumps(x)
                return x
            except (TypeError, OverflowError):
                if isinstance(x, dict):
                    return {k: make_jsonable(v) for k, v in x.items()}
                return str(x)

        return make_jsonable(self.config) if hasattr(self, "config") else {}
