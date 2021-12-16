from argparse import ArgumentParser

from torch import Tensor, nn
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj

from .base_conv import BaseConv


class TransformerNet(BaseConv):
    """https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv
    Args:
        input_dim (int, optional): Input dimension size. Defaults to None.
        output_dim (int, optional): Output dimension size. Defaults to None.
        hidden_dim (int, optional): Hidden layer(s) size. Defaults to 64.
        heads (int, optional): Number of heads. Defaults to 1.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        edge_dim (int, optional): Size of edge features. Defaults to None.
        edge_type (str, optional): Edge type. Defaults to "none".
        num_layers (int, optional): Number of layers. Defaults to 3.
    """

    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = None,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        edge_dim: int = None,
        edge_type: str = "none",
        heads: int = 1,
        num_layers: int = 3,
        **kwargs,
    ):
        assert input_dim is not None and output_dim is not None, "input_dim and output_dim must be specified"
        super().__init__()
        self.edge_type = edge_type
        if edge_type == "label":
            self.edge_embed = nn.Embedding(edge_dim + 1, edge_dim)
        elif edge_type == "none":
            edge_dim = None
        self.inp = TransformerConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False,
        )
        self.mid_layers = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=False,
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.out = TransformerConv(hidden_dim, output_dim, heads=1, dropout=dropout, edge_dim=edge_dim, concat=False)

    def forward(self, x: Tensor, edge_index: Adj, edge_feats: Tensor = None, **kwargs) -> Tensor:
        """Forward pass of the module"""
        if self.edge_type == "none":
            edge_feats = None
        elif self.edge_type == "label":
            edge_feats = self.edge_embed(edge_feats)
        x = self.inp(x, edge_index, edge_feats)
        for module in self.mid_layers:
            x = module(x, edge_index, edge_feats)
        x = self.out(x, edge_index, edge_feats)
        return x
