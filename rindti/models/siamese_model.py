from argparse import ArgumentParser
from copy import deepcopy
from math import ceil
from typing import Union
import subprocess
import os.path as osp
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef
from torch.optim import Adam

from rindti.utils.data import TwoGraphData

from ..layers import (
    MLP,
    GINConvNet,
    GMTNet,
)
from .base_model import BaseModel


class SiameseModel(BaseModel):
    def corrupt_features(self, features: torch.Tensor, frac: float) -> torch.Tensor:
        num_feat = features.size(0)
        num_node_types = int(features.max())
        num_corrupt_nodes = ceil(num_feat * frac)
        corrupt_idx = np.random.choice(range(num_feat), num_corrupt_nodes, replace=False)
        corrupt_features = torch.tensor(
            np.random.choice(range(num_node_types), num_corrupt_nodes, replace=True),
            dtype=torch.long,
            device=self.device,
        )
        features[corrupt_idx] = corrupt_features
        return features, corrupt_idx

    def __init__(self, pdb_folder, corrupt_ratio, alpha, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.pdb_folder = pdb_folder
        self.corrupt_ratio = corrupt_ratio
        self.alpha = alpha
        self.feat_embed = Embedding(20, 32)
        self.node_embed = GINConvNet(32, 128, 64, 10)
        self.pool = GMTNet(128, 128, 128, ratio=0.25)
        self.node_pred = GINConvNet(128, 20, 64, 3)
        self.mlp = MLP(128, 128, 1, 1)

    def embed_graph(self, x, edge_index, batch, **kwargs):
        x, corrupt_idx = self.corrupt_features(x, self.corrupt_ratio)
        x = self.feat_embed(x)
        x = self.node_embed(x, edge_index)
        rep = self.pool(x, edge_index, batch)
        node_pred = self.node_pred(x, edge_index)
        return rep, node_pred, corrupt_idx

    def forward(self, x, edge_index, batch, ids, **kwargs):
        rep, node_pred, corrupt_idx = self.embed_graph(x, edge_index, batch)
        rep_b, ids_b = rep.clone(), deepcopy(ids)
        permutator = torch.randperm(rep.size(0))
        rep_b = rep_b[permutator]
        ids_b = ids_b[permutator]
        combined_rep = self._element_l1(rep, rep_b)
        pred_scores = self.mlp(combined_rep)
        return pred_scores, node_pred, corrupt_idx, zip(ids, ids_b)

    def shared_step(self, data):
        pred, node_pred, corrupt_idx, idzip = self.forward(data.x, data.edge_index, data.batch, np.asarray(data.id))
        scores = torch.tensor(self.tmalign_batch(idzip), device=self.device)
        loss = F.mse_loss(pred.squeeze(1), scores)
        node_loss = F.cross_entropy(node_pred[corrupt_idx], data.x[corrupt_idx])
        return {"loss": loss + node_loss * self.alpha, "tmscore_loss": loss, "node_loss": node_loss}

    def tmalign_process(self, id1, id2):
        filename1 = osp.join(self.pdb_folder, id1 + ".pdb")
        filename2 = osp.join(self.pdb_folder, id2 + ".pdb")
        return subprocess.Popen(
            ["/home/ilya/miniconda3/envs/rindti/bin/TMscore", filename1, filename2],
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )

    def parse_tmalign_output(self, text):
        match = re.search(r"TM\-score\s+=\s+(\d\.\d+)", text)
        return float(match.group(1))

    def tmalign_batch(self, idzip):
        procs = [self.tmalign_process(id1, id2) for id1, id2 in idzip]
        for proc in procs:
            proc.wait()
        return [self.parse_tmalign_output(x.stdout.read()) for x in procs]

    def configure_optimizers(self):
        """
        Configure the optimizer/s.
        Relies on initially saved hparams to contain learning rates, weight decays etc
        """
        return Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
