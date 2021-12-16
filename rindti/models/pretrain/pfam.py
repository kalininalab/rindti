from typing import List, Union

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor

from rindti.utils.cli import get_module

from ...data import DataCorruptor, TwoGraphData
from ...layers import MLP, BaseConv
from ...losses import GeneralisedLiftedStructureLoss, NodeLoss, PfamCrossEntropyLoss, SoftNearestNeighborLoss
from ..base_model import BasePretrainModel
from ..encoder import Encoder


class PfamModel(BasePretrainModel):
    """Pretraining with Pfam Model

    Args:
        node_pred (Union[MLP, BaseConv]): Module for predicting original node labels
        encoder (Encoder): Graph encoder
        loss (LightningModule): Loss function
        node_loss (NodeLoss): Loss function for node prediction
        masker (DataCorruptor): Module for masking the node labels
    """

    def __init__(
        self,
        node_pred: Union[MLP, BaseConv],
        encoder: Encoder,
        loss: LightningModule,
        node_loss: NodeLoss,
        masker: DataCorruptor,
        _optimizer_args: dict = None,
        _lr_scheduler_args: dict = None,
        _fam_list: List[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.node_pred = get_module(
            node_pred,
            input_dim=encoder["hidden_dim"],
            output_dim=encoder["feat_dim"] + 1,
        )
        self.encoder = Encoder(**encoder)
        self.loss = get_module(loss, _fam_list)
        self.node_loss = get_module(node_loss)
        self.masker = get_module(masker)
        self.optimizer_args = _optimizer_args
        self.lr_scheduler_args = _lr_scheduler_args

    @property
    def fam_idx(self) -> List[int]:
        """Using batch_size and prot_per_fam, get idx of each family

        Returns:
            List[List]: First list is families, second list is entries in the family
        """
        res = []
        for fam, _ in enumerate(range(0, self.hparams.batch_size, self.hparams.prot_per_fam)):
            res += [fam] * self.hparams.prot_per_fam
        return res

    def forward(self, data: dict) -> Tensor:
        """Forward pass of the model"""
        data = self.masker(data)
        embeds, node_embeds = self.encoder(data)
        node_preds = self.node_pred(node_embeds)
        return embeds, node_preds

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        embeds, node_preds = self.forward(data)
        fam_idx = torch.tensor(
            self.fam_idx,
            dtype=torch.long,
            device=self.device,
        ).view(-1, 1)
        node_metrics = self.node_loss(node_preds[data["x_idx"]], data["x_orig"] - 1)
        if self.hparams.loss == "crossentropy":
            metrics = self.loss(embeds, data.fam)
        else:
            metrics = self.loss(embeds, fam_idx).mean()
        metrics.update(node_metrics)
        metrics["loss"] = metrics["graph_loss"] + metrics["node_loss"] * self.hparams.alpha
        return {k: v.detach() if k != "loss" else v for k, v in metrics.items()}

    def log_node_confusionmatrix(self, confmatrix: Tensor):
        """Saves the confusion matrix of node prediction

        Args:
            confmatrix (Tensor): 20x20 matrix
        """
        fig = plt.figure()
        sns.heatmap(confmatrix.detach().cpu())
        self.logger.experiment.add_figure("confmatrix", fig, global_step=self.global_step)

    def log_distmap(self, data: TwoGraphData, embeds: Tensor):
        """Plot and save distance matrix of this batch"""
        fig = plt.figure()
        sns.heatmap(torch.cdist(embeds, embeds).detach().cpu())
        self.logger.experiment.add_figure("distmap", fig, global_step=self.global_step)
        self.logger.experiment.add_embedding(embeds, metadata=data.fam, global_step=self.global_step)
