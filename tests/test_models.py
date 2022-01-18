from pprint import pprint

import pytest
import torch

from rindti.models import ClassificationModel, RegressionModel
from rindti.utils import get_module, read_config


@pytest.fixture
def default_config():
    return read_config("config/dti.yaml")


class BaseTestModel:
    @pytest.mark.parametrize(
        "prot_node_embed", ["GINConvNet", "GatConvNet", "ChebConvNet", "FilmConvNet", "TransformerNet"]
    )
    @pytest.mark.parametrize("prot_pool", ["GMTNet", "DiffPoolNet", "MeanPool"])
    def test_no_edge_shared_step(
        self,
        prot_node_embed,
        prot_pool,
        dti_datamodule,
        dti_batch,
        default_config,
    ):
        default_config["model"]["init_args"]["prot_encoder"]["node_embed"]["class_path"] = (
            "rindti.layers." + prot_node_embed
        )
        default_config["model"]["init_args"]["prot_encoder"]["pool"]["class_path"] = "rindti.layers." + prot_pool
        dti_datamodule.update_model_args(default_config["model"]["init_args"])
        default_config["model"]["init_args"]["prot_encoder"]["edge_type"] = "none"
        default_config["model"]["init_args"]["drug_encoder"]["edge_type"] = "none"
        pprint(default_config)
        model = get_module(
            default_config["model"],
            _optimizer_args=default_config["optimizer"],
            _lr_scheduler_args=default_config["lr_scheduler"],
        )

        model.shared_step(dti_batch)


class TestClassModel(BaseTestModel):

    model = ClassificationModel

    @pytest.mark.parametrize("feat_method", ["element_l1", "element_l2", "mult", "concat"])
    def test_feat_methods(self, feat_method, default_config):
        """Test feature concatenation"""
        default_config["feat_method"] = feat_method
        default_config["prot_feat_dim"] = 1
        default_config["drug_feat_dim"] = 1
        model = self.model(**default_config)
        prot = torch.rand((32, 64), dtype=torch.float32)
        drug = torch.rand((32, 64), dtype=torch.float32)
        combined = model.merge_features(drug, prot)
        assert combined.size(0) == 32
        if feat_method == "concat":
            assert combined.size(1) == 128
        else:
            assert combined.size(1) == 64


class TestRegressionModel(BaseTestModel):

    model = RegressionModel
