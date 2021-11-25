from argparse import ArgumentParser
from typing import Iterable

import torch
import torch.nn.functional as F
from jsonargparse import lazy_instance
from jsonargparse.typing import final
from matplotlib.pyplot import get
from torch.functional import Tensor

from rindti.data.datamodules import DTIDataModule
from rindti.utils.cli import get_module

from ...data import TwoGraphData
from ...layers import MLP
from ...layers.base_layer import BaseLayer
from ...utils import remove_arg_prefix
from ..base_model import BaseModel
from ..encoder import Encoder
from ..pretrain import BGRLModel, GraphLogModel, InfoGraphModel, PfamModel


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a class problem"""

    def __init__(
        self,
        prot_encoder: Encoder = lazy_instance(Encoder),
        drug_encoder: Encoder = lazy_instance(Encoder),
        mlp: MLP = lazy_instance(MLP, output_dim=1),
        feat_method: str = "element_l1",
        **kwargs,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.prot_encoder = prot_encoder
        self.drug_encoder = drug_encoder
        self.mlp = mlp
        self._determine_feat_method(feat_method)

    def _load_pretrained(self, checkpoint_path: str) -> Iterable[BaseLayer]:
        """Load pretrained model

        Args:
            checkpoint_path (str): Path to checkpoint file.
            Has to contain 'infograph', 'graphlog', 'pfam' or 'bgrl', which will point to the type of model.

        Returns:
            Iterable[BaseLayer]: feat_embed, node_embed, pool of the pretrained model
        """
        if "infograph" in checkpoint_path:
            encoder = InfoGraphModel.load_from_checkpoint(checkpoint_path).encoder
        elif "graphlog" in checkpoint_path:
            encoder = GraphLogModel.load_from_checkpoint(checkpoint_path).encoder
        elif "pfam" in checkpoint_path:
            encoder = PfamModel.load_from_checkpoint(checkpoint_path).encoder
        elif "bgrl" in checkpoint_path:
            encoder = BGRLModel.load_from_checkpoint(checkpoint_path).student_encoder
        else:
            raise ValueError(
                """Unknown pretraining model type!
                Please ensure 'pfam', 'graphlog', 'bgrl' or 'infograph' are present in the model path"""
            )
        encoder.return_nodes = False
        return encoder

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """Forward pass of the model"""
        prot_embed = self.prot_encoder(prot)
        drug_embed = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        return self.mlp(joint_embedding)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test
        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        self.log("batch_size", len(data))
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        pred = torch.sigmoid(self.forward(prot, drug))
        labels = data.label.unsqueeze(1)
        bce_loss = F.binary_cross_entropy(pred, labels.float())
        metrics = self._get_class_metrics(pred, labels)
        metrics["loss"] = bce_loss
        return metrics
