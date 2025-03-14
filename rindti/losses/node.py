import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics.functional import accuracy


class NodeLoss(LightningModule):
    r"""Masked/corrupted node cross-entropy loss.

    Args:
        weighted (bool, optional): Whether to weigh based on popularity. Defaults to True.
    """

    def __init__(self, weighted: bool = True, **kwargs) -> None:
        super().__init__()
        self.weighted = weighted

    def forward(self, x: Tensor, target: Tensor) -> dict:
        """"""
        target = target if target.dtype == torch.long else target.argmax(dim=1)
        weight = (
            self.get_weights(x, target) if self.weighted else torch.ones(x.size(1), device=self.device) / x.size(1)
        )
        node_loss = F.cross_entropy(x, target, weight=weight)
        return dict(
            node_loss=node_loss,
            node_acc=accuracy(x, target),
        )

    def get_weights(self, x: Tensor, target: Tensor) -> Tensor:
        """Calculate the weights for the loss function.

        Args:
            target (Tensor): Target tensor of type torch.long.

        Returns:
            Tensor: Tensor of size self.feat_dim.
        """
        weights = []
        for i in range(x.size(1)):
            weights.append(target[target == i].size(0))
        weights = 1 / (torch.tensor(weights, dtype=torch.float32, device=self.device) + 1)
        return weights / weights.sum()
