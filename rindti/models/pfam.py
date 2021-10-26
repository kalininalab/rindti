from argparse import ArgumentParser
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.functional import Tensor

from ..data import TwoGraphData
from ..utils import MyArgParser
from .base_model import BaseModel, node_embedders, poolers
from .encoder import Encoder


class PfamModel(BaseModel):
    """Model for Pfam class comparison problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(return_nodes=False, **kwargs)
        self.losses = defaultdict(list)
        self.all_idx = set(range(kwargs["batch_size"]))
        self.fam_idx = self._get_fam_idx()
        self.loss = {
            "snnl": self.soft_nearest_neighbor_loss,
            "lifted": self.generalised_lifted_structure_loss,
        }[kwargs["loss"]]

    def generalised_lifted_structure_loss(self, embeds: Tensor) -> Tensor:
        """Hard mines for negatives

        Args:
            embeds (Tensor): Embeddings of all data points
            fam_idx (List[List]): List of lists of family indices

        Returns:
            Tensor: final loss
        """
        for idx in self.fam_idx:
            dist = torch.cdist(embeds, embeds)
            pos_idxt = torch.tensor(idx)
            neg_idxt = torch.tensor(list(self.all_idx.difference(idx)))
            pos = dist[pos_idxt[:, None], pos_idxt]
            neg = dist[neg_idxt[:, None], pos_idxt]
            pos_loss = torch.logsumexp(pos, dim=0)
            neg_loss = torch.logsumexp(self.hparams.margin - neg, dim=0)
        return torch.relu(pos_loss + neg_loss) ** 2

    def soft_nearest_neighbor_loss(self, embeds: Tensor) -> Tensor:
        temp, init_temp = (
            torch.tensor(1, dtype=torch.float32, device=self.device, requires_grad=True)
            if self.hparams.optim_temp
            else 1,
            self.hparams.temp,
        )
        norm_emb = torch.nn.functional.normalize(embeds)
        sim = 1 - torch.matmul(norm_emb, norm_emb.t())
        loss = self._get_loss(sim, init_temp / temp)
        if not self.hparams.optim_temp:
            return loss
        loss.mean().backward(inputs=[temp])
        with torch.no_grad():
            temp -= 0.2 * temp.grad
        return self._get_loss(sim, init_temp / temp)

    def _get_fam_loss(self, expsim: Tensor, idx: list) -> Tensor:
        """Calculate loss for one family

        Args:
            expsim (Tensor): Exponentiated similarity matrix
            idx (list): ids of target family
            all_idx (set): all_ids

        Returns:
            Tensor: 1D tensor of length len(idx) with losses
        """
        pos_idxt = torch.tensor(idx)
        pos = expsim[pos_idxt[:, None], pos_idxt]
        batch = expsim[:, pos_idxt]
        return -torch.log(pos.sum(dim=0) / batch.sum(dim=0))

    def _get_loss(self, sim: Tensor, tau: Tensor) -> Tensor:
        """Calculate SNNL

        Args:
            sim (Tensor): similarity matrix
            tau (Tensor): temperature

        Returns:
            Tensor: 1D Tensor of losses for each entry
        """
        expsim = torch.exp(-sim / tau) - torch.eye(self.hparams.batch_size, device=self.device)
        return torch.cat([self._get_fam_loss(expsim, idx) for idx in self.fam_idx])

    def _get_fam_idx(self) -> List[List]:
        """Using batch_size and prot_per_fam, get idx of each family

        Returns:
            List[List]: First list is families, second list is entries in the family
        """
        return [
            [*range(x, x + self.hparams.prot_per_fam)]
            for x in range(
                0,
                self.hparams.batch_size,
                self.hparams.prot_per_fam,
            )
        ]

    def forward(self, data: dict) -> Tensor:
        """Forward pass of the model"""
        return self.encoder(data)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        embeds = self.forward(data)
        loss = self.loss(embeds)
        return dict(loss=loss.mean())

    @staticmethod
    def add_arguments(parser: MyArgParser) -> MyArgParser:
        """Generate arguments for this module"""
        # Hack to find which embedding are used and add their arguments
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--node_embed", type=str, default="ginconv")
        tmp_parser.add_argument("--pool", type=str, default="gmt")
        args = tmp_parser.parse_known_args()[0]

        node_embed = node_embedders[args.node_embed]
        pool = poolers[args.pool]
        pooler_args = parser.add_argument_group("Pool", prefix="--")
        node_embed_args = parser.add_argument_group("Node embedding", prefix="--")
        node_embed.add_arguments(node_embed_args)
        pool.add_arguments(pooler_args)
        parser.add_argument("--margin", type=float, default=1)
        parser.add_argument("--prot_per_fam", type=int, default=8)
        parser.add_argument("--batch_per_epoch", type=int, default=1000)
        return parser
