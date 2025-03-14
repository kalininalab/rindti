import torch.nn.functional as F

from ...data import TwoGraphData
from ...utils import remove_arg_prefix
from .classification import ClassificationModel


class RegressionModel(ClassificationModel):
    """Model for DTI prediction as a reg problem"""

    def shared_step(self, data: TwoGraphData) -> dict:
        """"""
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        output = self.forward(prot, drug)["pred"]
        labels = data.label.unsqueeze(1).float()
        loss = F.mse_loss(output, labels)
        metrics = self._get_reg_metrics(output, labels)
        metrics.update(dict(loss=loss))
        return metrics
