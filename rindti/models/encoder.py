from argparse import ArgumentParser
from typing import Tuple, Union

from torch import nn
from torch.functional import Tensor
from torch_geometric.data import Data

from .base_model import BaseModel


class Encoder(BaseModel):
    r"""Encoder for graphs

    Args:
        return_nodes (bool, optional): Return node embeddings as well. Defaults to False."""

    def __init__(self, return_nodes: bool = False, **kwargs):
        super().__init__()
        self.feat_embed = self._get_feat_embed(kwargs)
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)
        self.return_nodes = return_nodes

    def forward(
        self,
        data: Union[dict, Data],
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Encode a graph

        Args:
            data (Union[dict, Data]): Graph to encode. Must contain the following keys:
                - x: Node features
                - edge_index: Edge indices
                - batch: Batch indices
        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Either graph of graph+node embeddings
        """
        if not isinstance(data, dict):
            data = data.to_dict()
        x, edge_index, batch, edge_feats = (
            data["x"],
            data["edge_index"],
            data["batch"],
            data.get("edge_feats"),
        )
        feat_embed = self.feat_embed(x)
        node_embed = self.node_embed(
            x=feat_embed,
            edge_index=edge_index,
            edge_feats=edge_feats,
            batch=batch,
        )
        embed = self.pool(x=node_embed, edge_index=edge_index, batch=batch)
        if self.return_nodes:
            return embed, node_embed
        return embed

    def embed(self, data: Data, **kwargs):
        self.return_nodes = False
        embed = self.forward(data)
        return embed.detach()
