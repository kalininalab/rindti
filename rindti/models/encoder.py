from argparse import ArgumentParser
from pprint import pprint
from typing import Tuple, Union

from jsonargparse.typing import final
from torch import nn
from torch.functional import Tensor
from torch_geometric.data import Data

from ..layers import BaseConv, BasePool
from ..utils import get_module
from .base_model import BaseModel


@final
class Encoder(BaseModel):
    """Encoder for graphs"""

    def __init__(
        self,
        feat_type: str = None,
        edge_type: str = None,
        feat_dim: int = None,
        edge_dim: int = None,
        max_nodes: int = None,
        node_embed: BaseConv = None,
        pool: BasePool = None,
        hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.feat_type = feat_type
        self.edge_type = edge_type
        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.feat_embed = self._get_feat_embed()
        self.node_embed = get_module(node_embed, input_dim=hidden_dim, output_dim=hidden_dim)
        self.pool = get_module(pool, input_dim=hidden_dim, output_dim=hidden_dim)

    def forward(
        self,
        data: dict,
        return_nodes: bool = False,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Encode a graph

        Args:
            data (Data): torch_geometric - 'x', 'edge_index' etc

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
        if return_nodes:
            return embed, node_embed
        return embed

    def embed(self, data: Data, **kwargs):
        """Just encode and detach"""
        self.return_nodes = False
        embed = self.forward(data)
        return embed.detach()
