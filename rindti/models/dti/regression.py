import torch.nn.functional as F

from ...data import TwoGraphData
from ...utils import remove_arg_prefix
from .classification import ClassificationModel


class RegressionModel(ClassificationModel):
    """DTI regression model

    Args:
        prot_encoder (Encoder): dictionary with `class_path` and `init_args` that describes the protein encoder
        drug_encoder (Encoder): dictionary with `class_path` and `init_args` that describes the drug encoder
        mlp (MLP): dictionary with `class_path` and `init_args` that describes the MLP
        feat_method (str, optional): How to merge the features. Defaults to "element_l1".
    """

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test
        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        output = self.forward(prot, drug)
        labels = data.label.unsqueeze(1).float()
        loss = F.mse_loss(output, labels)
        metrics = self._get_reg_metrics(output, labels)
        metrics.update(dict(loss=loss))
        return metrics
