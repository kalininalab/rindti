from argparse import ArgumentParser

from torch import nn
from torch.functional import Tensor

from ..base_layer import BaseLayer


class MLP(BaseLayer):
    """Multi-layer perceptron

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        num_layers (int, optional): Total Number of layers. Defaults to 2.
        dropout (float, optional): Dropout ratio. Defaults to 0.2.
    """

    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        if input_dim is None:
            input_dim = hidden_dim
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        for i in range(num_layers - 2):
            self.mlp.add_module("hidden_linear{}".format(i), nn.Linear(hidden_dim, hidden_dim))
            self.mlp.add_module("hidden_relu{}".format(i), nn.ReLU())
            self.mlp.add_module("hidden_dropout{}".format(i), nn.Dropout(dropout))
        self.mlp.add_module("final_linear", nn.Linear(hidden_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module"""
        return self.mlp(x)
