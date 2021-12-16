from torch import LongTensor, Tensor
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class BasePool(BaseLayer):
    """Base class for graph convolutional layers

    Args:
        input_dim (int): Input dimension size.
        output_dim (int): Output dimension size."""

    def forward(self, x: Tensor, edge_index: Adj, batch: LongTensor, **kwargs) -> Tensor:
        """Forward pass of the module

        Args:
            x (Tensor): Node features tensor.
            edge_index (Adj): Edge index tensor.
            batch (LongTensor): Batch tensor, which node belongs to which graph.
        """
        raise NotImplementedError
