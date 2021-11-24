from argparse import ArgumentParser

from torch.functional import Tensor
from torch.nn import Embedding
from torch_geometric.nn import PNAConv
from torch_geometric.typing import Adj

from .base_conv import BaseConv


class PNAConvNet(BaseConv):
    """Principal Neighbourhood Aggregation
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PNAConv
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, edge_dim=None, deg=None, **kwargs):
        super().__init__()
        self.edge_embedding = Embedding(edge_dim, edge_dim)
        self.conv1 = PNAConv(
            input_dim,
            hidden_dim,
            aggregators=["sum", "mean", "max"],
            scalers=["linear", "identity"],
            edge_dim=edge_dim,
            deg=deg,
        )
        self.conv2 = PNAConv(
            hidden_dim,
            output_dim,
            aggregators=["sum", "mean", "max"],
            scalers=["linear", "identity"],
            edge_dim=edge_dim,
            deg=deg,
        )

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """Forward pass of the module"""
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x
