from argparse import ArgumentParser

import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch_geometric.nn import GraphMultisetTransformer
from torch_geometric.typing import Adj

from .base_pool import BasePool


class GMTNet(BasePool):
    """https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.GraphMultisetTransformer"""

    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = None,
        hidden_dim: int = 64,
        ratio: float = 0.25,
        max_nodes: int = 0,
        num_heads: int = 4,
        **kwargs,
    ):
        assert input_dim is not None and output_dim is not None, "input_dim and output_dim must be specified"
        assert max_nodes > 0, "max_nodes must be greater than 0"
        super().__init__()
        self.pool = GraphMultisetTransformer(
            input_dim,
            hidden_dim,
            output_dim,
            num_nodes=max_nodes,
            pooling_ratio=ratio,
            num_heads=num_heads,
            pool_sequences=["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"],
            layer_norm=True,
        )

    def forward(self, x: Tensor, edge_index: Adj, batch: LongTensor) -> Tensor:
        """Forward pass"""
        embeds = self.pool(x, batch, edge_index)
        return F.normalize(embeds, dim=1)
