from typing import Optional

import torch
from pytorch_lightning import LightningModule
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torchmetrics.functional import accuracy

from ..layers import MLP


class PfamCrossEntropyLoss(LightningModule):
    """Simple cross=entropy loss with the added MLP to match dimensions

    Args:
        hidden_dim (int, optional): Size of hidden layer. Defaults to None.
        _fam_list (list, optional): list of all available protein families. Defaults to None.
    """

    def __init__(self, hidden_dim: Optional[int] = None, _fam_list: Optional[list] = None):
        super().__init__()
        self.mlp = MLP(hidden_dim, len(_fam_list))
        self.loss = torch.nn.CrossEntropyLoss()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(_fam_list)

    def forward(self, x: Tensor, y: list) -> Tensor:
        """Forward pass of the module"""
        x = self.mlp(x)
        y = torch.tensor(self.label_encoder.transform(y), device=self.device, dtype=torch.long)
        loss = self.loss(x, y)
        return dict(
            graph_loss=loss,
            graph_acc=accuracy(x, y),
        )
