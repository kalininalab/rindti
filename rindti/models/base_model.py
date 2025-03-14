from argparse import ArgumentParser
from typing import Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import LongTensor, Tensor, nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import (
    accuracy,
    auroc,
    explained_variance,
    matthews_corrcoef,
    mean_absolute_error,
    pearson_corrcoef,
)

from ..data import TwoGraphData
from ..layers import (
    MLP,
    ChebConvNet,
    DiffPoolNet,
    FilmConvNet,
    GatConvNet,
    GINConvNet,
    GMTNet,
    MeanPool,
    TransformerNet,
)

node_embedders = {
    "ginconv": GINConvNet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "filmconv": FilmConvNet,
    "transformer": TransformerNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class BaseModel(LightningModule):
    """
    Base model, defines a lot of helper functions.
    """

    def __init__(self):
        super().__init__()

    def _get_label_embed(self, params: dict) -> nn.Embedding:
        return nn.Embedding(params["feat_dim"] + 1, params["hidden_dim"])

    def _get_onehot_embed(self, params: dict) -> nn.LazyLinear:
        return nn.Linear(params["feat_dim"], params["hidden_dim"], bias=False)

    def _get_feat_embed(self, params: dict) -> Union[nn.Embedding, nn.LazyLinear]:
        if params["feat_type"] == "onehot":
            return self._get_onehot_embed(params)
        elif params["feat_type"] == "label":
            return self._get_label_embed(params)
        else:
            raise ValueError("Unknown feature type!")

    def _get_node_embed(self, params: dict, out_dim=None) -> LightningModule:
        if params["edge_type"] == "none":
            params["edge_dim"] = None
        if out_dim:
            return node_embedders[params["node_embed"]](params["hidden_dim"], out_dim, **params)
        return node_embedders[params["node_embed"]](params["hidden_dim"], params["hidden_dim"], **params)

    def _get_pooler(self, params: dict) -> LightningModule:
        return poolers[params["pool"]](params["hidden_dim"], params["hidden_dim"], **params)

    def _get_mlp(self, params: dict) -> MLP:
        return MLP(**params, input_dim=self.embed_dim, out_dim=1)

    def _determine_feat_method(self, feat_method: str, drug_hidden_dim: int, prot_hidden_dim: int, **kwargs):
        """Which method to use for concatenating drug and protein representations"""
        if feat_method == "concat":
            self.merge_features = self._concat
            self.embed_dim = drug_hidden_dim + prot_hidden_dim
        elif feat_method == "element_l2":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._element_l2
            self.embed_dim = drug_hidden_dim
        elif feat_method == "element_l1":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._element_l1
            self.embed_dim = drug_hidden_dim
        elif feat_method == "mult":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._mult
            self.embed_dim = drug_hidden_dim
        else:
            raise ValueError("unsupported feature method")

    def _concat(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Concatenation"""
        return torch.cat((drug_embed, prot_embed), dim=1)

    def _element_l2(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L2 distance"""
        return torch.sqrt(((drug_embed - prot_embed) ** 2) + 1e-6).float()

    def _element_l1(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L1 distance"""
        return (drug_embed - prot_embed).abs()

    def _mult(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Multiplication"""
        return drug_embed * prot_embed

    def _get_reg_metrics(self, output: Tensor, labels: Tensor) -> dict:
        """Calculate metrics common for reg - corrcoef, MAE and explained variance
        Returns dict of all"""
        corr = pearson_corrcoef(output, labels)
        mae = mean_absolute_error(output, labels)
        expvar = explained_variance(output, labels)
        return {
            "corr": corr,
            "mae": mae,
            "expvar": expvar,
        }

    def _get_class_metrics(self, output: Tensor, labels: LongTensor):
        """Calculate metrics common for class - accuracy, auroc and Matthews coefficient
        Returns dict of all"""
        acc = accuracy(output, labels)
        _auroc = auroc(output, labels, pos_label=1)
        _mc = matthews_corrcoef(output, labels.squeeze(1), num_classes=2)
        return {
            "acc": acc,
            "auroc": _auroc,
            "matthews": _mc,
        }

    def training_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during training step"""
        ss = self.shared_step(data)
        for key, value in ss.items():
            self.log("train_" + key, value)
        return ss

    def validation_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        for key, value in ss.items():
            self.log("val_" + key, value)
        return ss

    def test_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during test step"""
        return self.shared_step(data)

    def log_histograms(self):
        """Logs the histograms of all the available parameters"""
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def shared_epoch_end(self, outputs: dict, prefix: str, log_hparams=False):
        """Things that are the same for train, test and val"""
        entries = outputs[0].keys()
        metrics = {}
        for i in entries:
            val = torch.stack([x[i] for x in outputs])
            val = val[~val.isnan()].mean()
            self.logger.experiment.add_scalar(prefix + i, val, self.current_epoch)
            metrics[i] = val
        if log_hparams:
            self.logger.log_hyperparams(self.hparams, metrics)

    def training_epoch_end(self, outputs: dict):
        """What to do at the end of a training epoch. Logs everything"""
        self.log_histograms()
        self.shared_epoch_end(outputs, "train_epoch_")

    def validation_epoch_end(self, outputs: dict):
        """What to do at the end of a validation epoch. Logs everything, saves hyperparameters"""
        self.shared_epoch_end(outputs, "val_epoch_", log_hparams=True)

    def test_epoch_end(self, outputs: dict):
        """What to do at the end of a test epoch. Logs everything, saves hyperparameters"""
        self.shared_epoch_end(outputs, "test_epoch_", log_hparams=True)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimiser and/or lr schedulers"""
        optimiser = {"adamw": AdamW, "adam": Adam, "sgd": SGD, "rmsprop": RMSprop}[self.hparams.optimiser]
        params = [{"params": self.parameters()}]
        if hasattr(self, "prot_encoder"):
            params.append({"params": self.prot_encoder.parameters(), "lr": self.hparams.prot_lr})
        if hasattr(self, "drug_encoder"):
            {"params": self.drug_encoder.parameters(), "lr": self.hparams.drug_lr}
        optimiser = optimiser(params=self.parameters(), lr=self.hparams.lr)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimiser,
                factor=self.hparams.reduce_lr_factor,
                patience=self.hparams.reduce_lr_patience,
                verbose=True,
            ),
            "monitor": self.hparams.monitor,
        }
        return [optimiser], [lr_scheduler]
