import random
from copy import deepcopy
from math import ceil
from typing import Tuple

import torch
from torch.nn import Embedding
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader

from ..layers import GINConvNet, GMTNet
from .base_model import BaseModel
import numpy as np
import torch.nn.functional as F

# NCE loss between graphs and prototypes


class GraphLogModel(BaseModel):
    def __init__(
        self,
        decay_ratio=0.5,
        mask_rate=0.3,
        alpha=1,
        beta=1,
        gamma=0.1,
        num_proto=8,
        hierarchy=3,
        batch_size=32,
        num_workers=4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.decay_ratio = decay_ratio
        self.mask_rate = mask_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_proto = num_proto
        self.hierarchy = hierarchy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embed_dim = 64
        self.feat_embed = Embedding(21, 64, padding_idx=20)
        self.node_embed = GINConvNet(64, 64, 64, num_layers=5)
        self.pool = GMTNet(64, 64, 64, ratio=0.15)
        self.proj = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64))

    def proto_NCE_loss(self, graph_reps, tau=0.04, epsilon=1e-6):
        # similarity for original and modified graphs
        graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
        exp_sim_list = []
        mask_list = []
        NCE_loss = 0

        for i in range(len(self.proto) - 1, -1, -1):
            tmp_proto = self.proto[i]
            proto_norm = torch.norm(tmp_proto, dim=1).unsqueeze(-1)

            sim = torch.mm(graph_reps, tmp_proto.t()) / (torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)
            exp_sim = torch.exp(sim / tau)

            if i != (len(self.proto) - 1):
                # apply the connection mask
                exp_sim_last = exp_sim_list[-1]
                idx_last = torch.argmax(exp_sim_last, dim=1).unsqueeze(-1)
                connection = self.proto_connection[i]
                connection_mask = (connection.unsqueeze(0) == idx_last.float()).float()
                exp_sim = exp_sim * connection_mask

                # define NCE loss between prototypes from consecutive layers
                upper_proto = self.proto[i + 1]
                upper_proto_norm = torch.norm(upper_proto, dim=1).unsqueeze(-1)
                proto_sim = torch.mm(tmp_proto, upper_proto.t()) / (
                    torch.mm(proto_norm, upper_proto_norm.t()) + epsilon
                )
                proto_exp_sim = torch.exp(proto_sim / tau)

                proto_positive_list = [proto_exp_sim[j, connection[j].long()] for j in range(proto_exp_sim.shape[0])]
                proto_positive = torch.stack(proto_positive_list, dim=0)
                proto_positive_ratio = proto_positive / (proto_exp_sim.sum(1) + epsilon)
                NCE_loss += -torch.log(proto_positive_ratio).mean()

            mask = (exp_sim == exp_sim.max(1)[0].unsqueeze(-1)).float()

            exp_sim_list.append(exp_sim)
            mask_list.append(mask)

        # define NCE loss between graph embedding and prototypes
        for i in range(len(self.proto)):
            exp_sim = exp_sim_list[i]
            mask = mask_list[i]

            positive = exp_sim * mask
            negative = exp_sim * (1 - mask)
            positive_ratio = positive.sum(1) / (positive.sum(1) + negative.sum(1) + epsilon)
            NCE_loss += -torch.log(positive_ratio).mean()

        return NCE_loss

    def update_proto_lowest(self, graph_reps, decay_ratio=0.7, epsilon=1e-6):
        graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
        proto_norm = torch.norm(self.proto[0], dim=1).unsqueeze(-1)
        sim = torch.mm(graph_reps, self.proto[0].t()) / (torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)

        # update states of prototypes
        mask = (sim == sim.max(1)[0].unsqueeze(-1)).float()
        cnt = mask.sum(0)
        self.proto_state[0] = (self.proto_state[0] + cnt).detach()

        # update prototypes
        batch_cnt = mask.t() / (cnt.unsqueeze(-1) + epsilon)
        batch_mean = torch.mm(batch_cnt, graph_reps)
        self.proto[0] = (
            self.proto[0] * (cnt == 0).float().unsqueeze(-1)
            + (self.proto[0] * decay_ratio + batch_mean * (1 - decay_ratio)) * (cnt != 0).float().unsqueeze(-1)
        ).detach()

    def init_proto_lowest(self, num_iter=5):
        self.eval()
        for _ in range(num_iter):
            for step, batch in enumerate(self.train_dataloader()):
                # get node and graph representations
                batch = batch.to(self.device)
                feat_reps = self.feat_embed(batch.x)
                node_reps = self.node_embed(feat_reps, batch.edge_index, batch.batch)
                graph_reps = self.pool(node_reps, batch.edge_index, batch.batch)

                # feature projection
                graph_reps_proj = self.proj(graph_reps)

                # update prototypes
                self.update_proto_lowest(graph_reps_proj)

        idx = torch.nonzero((self.proto_state[0] >= 2).float()).squeeze(-1)
        return torch.index_select(self.proto[0], 0, idx)

    def init_proto(self, index, num_iter=20):
        proto_connection = torch.zeros(self.proto[index - 1].shape[0], device=self.device)

        for iter in range(num_iter):
            for i in range(self.proto[index - 1].shape[0]):
                # update the closest prototype
                sim = torch.mm(self.proto[index], self.proto[index - 1][i, :].unsqueeze(-1)).squeeze(-1)
                idx = torch.argmax(sim)
                if iter == (num_iter - 1):
                    self.proto_state[index][idx] = 1
                proto_connection[i] = idx
                self.proto[index][idx, :] = self.proto[index][idx, :] * self.decay_ratio + self.proto[index - 1][
                    i, :
                ] * (1 - self.decay_ratio)

                # penalize rival
                sim[idx] = 0
                rival_idx = torch.argmax(sim)
                self.proto[index][rival_idx, :] = self.proto[index][rival_idx, :] * (
                    2 - self.decay_ratio
                ) - self.proto[index - 1][i, :] * (1 - self.decay_ratio)

        indices = torch.nonzero(self.proto_state[index]).squeeze(-1)
        proto_selected = torch.index_select(self.proto[index], 0, indices)
        for i in range(indices.shape[0]):
            idx = indices[i]
            idx_connection = torch.nonzero((proto_connection == idx.float()).float()).squeeze(-1)
            proto_connection[idx_connection] = i

        return proto_selected, proto_connection

    def mask_nodes(self, orig_x, frac):
        x = deepcopy(orig_x)
        num_nodes = x.size(0)
        num_masked_nodes = ceil(num_nodes * frac)
        masked_idx = np.random.choice(range(num_nodes), num_masked_nodes, replace=False)
        x[masked_idx] = 20
        return x, masked_idx

    def intra_NCE_loss(self, node_reps, node_mod_reps, masked_idx):
        return -torch.log(F.cosine_similarity(node_reps[masked_idx], node_mod_reps[masked_idx]) + 1e-6).sum()

    def inter_NCE_loss(self, graph_reps, graph_mod_reps):
        return -torch.log(F.cosine_similarity(graph_reps, graph_mod_reps) + 1e-6).sum()

    def embed_batch(self, x, edge_index, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_embed = self.feat_embed(x)
        node_reps = self.node_embed(feat_embed, edge_index, batch)
        graph_reps = self.pool(node_reps, edge_index, batch)
        node_reps = self.proj(node_reps)
        graph_reps = self.proj(graph_reps)
        return node_reps, graph_reps

    def forward(self, x, edge_index, batch):
        x_mod, masked_idx = self.mask_nodes(x, self.mask_rate)
        node_reps, graph_reps = self.embed_batch(x, edge_index, batch)
        node_mod_reps, graph_mod_reps = self.embed_batch(x_mod, edge_index, batch)
        return node_reps, node_mod_reps, graph_reps, graph_mod_reps, masked_idx

    def shared_step(self, data: Data, proto: bool = True) -> dict:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        (node_reps_proj, node_mod_reps_proj, graph_reps_proj, graph_mod_reps_proj, masked_idx) = self.forward(
            x, edge_index, batch
        )
        NCE_intra_loss = self.intra_NCE_loss(node_reps_proj, node_mod_reps_proj, masked_idx)
        NCE_inter_loss = self.inter_NCE_loss(graph_reps_proj, graph_mod_reps_proj)
        if proto:
            NCE_proto_loss = self.proto_NCE_loss(graph_reps_proj)
        else:
            NCE_proto_loss = torch.tensor(10, dtype=float, device=self.device)
        NCE_loss = self.alpha * NCE_intra_loss + self.beta * NCE_inter_loss + self.gamma * NCE_proto_loss
        return {
            "loss": NCE_loss,
            "NCE_intra_loss": NCE_intra_loss,
            "NCE_inter_loss": NCE_inter_loss,
            "NCE_proto_loss": NCE_proto_loss,
        }

    def training_step(self, batch, batch_idx):
        return self._do_step(batch, "train_")

    def validation_step(self, batch, batch_idx):
        return self._do_step(batch, "val_")

    def _do_step(self, data: Data, step_type: str):
        if self.trainer.current_epoch == 0:
            ss = self.shared_step(data, proto=False)
        else:
            ss = self.shared_step(data, proto=True)
        for key, value in ss.items():
            self.log(step_type + key, value)
        return ss

    def training_epoch_end(self, outputs):
        if self.trainer.current_epoch == 0:
            self.proto = [
                torch.rand((self.num_proto, self.embed_dim), device=self.device) for i in range(self.hierarchy)
            ]
            self.proto_state = [torch.zeros(self.num_proto, device=self.device) for i in range(self.hierarchy)]
            self.proto_connection = []
            tmp_proto = self.init_proto_lowest()
            self.proto[0] = tmp_proto
            for i in range(1, self.hierarchy):
                print("Initialize prototypes: layer ", i + 1)
                tmp_proto, tmp_proto_connection = self.init_proto(i)
                self.proto[i] = tmp_proto
                self.proto_connection.append(tmp_proto_connection)
        torch.cuda.empty_cache()
        return super().training_epoch_end(outputs)

    def configure_optimizers(self):
        """
        Configure the optimizer/s.
        Relies on initially saved hparams to contain learning rates, weight decays etc
        """
        optimiser = Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimiser,
                factor=self.hparams.reduce_lr_factor,
                patience=self.hparams.reduce_lr_patience,
                verbose=True,
            ),
            "monitor": "train_loss",
        }
        return [optimiser], [lr_scheduler]
