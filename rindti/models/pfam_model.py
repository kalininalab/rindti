from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torch.nn import Embedding
from torch_geometric.typing import Adj
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef

from ..layers import MLP, ChebConvNet, DiffPoolNet, GatConvNet, GINConvNet, GMTNet, MeanPool, NoneNet
from ..utils import remove_arg_prefix
from ..utils.data import TwoGraphData
from .base_model import BaseModel

node_embedders = {
    "ginconv": GINConvNet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "none": NoneNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class PfamModel(BaseModel):
    """Model for Pfam class comparison problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._determine_feat_method(kwargs["feat_method"], kwargs["drug_hidden_dim"], kwargs["prot_hidden_dim"])
        # TODO fix hardcoded values
        self.feat_embed = Embedding(20, kwargs["prot_node_embed_dim"])
        self.node_embed = node_embedders[kwargs["node_embed"]](
            kwargs["node_embed_dim"], kwargs["hidden_dim"], **kwargs
        )
        self.pool = poolers[kwargs["pool"]](kwargs["hidden_dim"], kwargs["hidden_dim"], **kwargs)
        self.drug_pool = poolers[kwargs["pool"]](kwargs["hidden_dim"], kwargs["hidden_dim"], **kwargs)
        mlp_param = remove_arg_prefix("mlp_", kwargs)
        self.mlp = MLP(**mlp_param, input_dim=self.embed_dim, out_dim=1)

    def forward(
        self,
        a_x: Tensor,
        b_x: Tensor,
        a_edge_index: Adj,
        b_edge_index: Adj,
        a_batch: Tensor,
        b_batch: Tensor,
        *args,
    ) -> Tensor:
        """Forward pass of the model

        Args:
            prot_x (Tensor): Protein node features
            drug_x (Tensor): Drug node features
            prot_edge_index (Adj): Protein edge info
            drug_edge_index (Adj): Drug edge info
            prot_batch (Tensor): Protein batch
            drug_batch (Tensor): Drug batch

        Returns:
            (Tensor): Final prediction
        """
        a_x = self.feat_embed(a_x)
        b_x = self.feat_embed(b_x)
        a_x = self.node_embed(a_x, a_edge_index, a_batch)
        b_x = self.node_embed(b_x, b_edge_index, b_batch)
        a_embed = self.pool(a_x, a_edge_index, a_batch)
        b_embed = self.pool(b_x, b_edge_index, b_batch)
        joint_embedding = self.merge_features(a_embed, b_embed)
        logit = self.mlp(joint_embedding)
        return torch.sigmoid(logit)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        output = self.forward(
            data.a_x,
            data.b_x,
            data.a_edge_index,
            data.b_edge_index,
            data.a_x_batch,
        )
        labels = data.label.unsqueeze(1)
        loss = F.binary_cross_entropy(output, labels.float())
        t = (output > 0.5).float()
        acc = accuracy(t, labels)
        try:
            _auroc = auroc(t, labels)
        except Exception:
            _auroc = torch.tensor(np.nan, device=self.device)
        _mc = matthews_corrcoef(output, labels.squeeze(1), num_classes=2)
        return {
            "loss": loss,
            "acc": acc,
            "auroc": _auroc,
            "matthews": _mc,
        }

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        # Hack to find which embedding are used and add their arguments
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--drug_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_pool", type=str, default="gmt")
        tmp_parser.add_argument("--drug_pool", type=str, default="gmt")

        args = tmp_parser.parse_known_args()[0]
        prot_node_embed = node_embedders[args.prot_node_embed]
        drug_node_embed = node_embedders[args.drug_node_embed]
        prot_pool = poolers[args.prot_pool]
        drug_pool = poolers[args.drug_pool]
        prot = parser.add_argument_group("Prot", prefix="--prot_")
        drug = parser.add_argument_group("Drug", prefix="--drug_")
        prot.add_argument("node_embed", default="chebconv")
        prot.add_argument("node_embed_dim", default=16, type=int, help="Size of aminoacid embedding")
        drug.add_argument("node_embed", default="chebconv")
        drug.add_argument(
            "node_embed_dim",
            default=16,
            type=int,
            help="Size of atom element embedding",
        )

        prot_node_embed.add_arguments(prot)
        drug_node_embed.add_arguments(drug)
        prot_pool.add_arguments(prot)
        drug_pool.add_arguments(drug)
        return parser
