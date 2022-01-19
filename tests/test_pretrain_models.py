import pytest

from rindti.models import BGRLModel, GraphLogModel, InfoGraphModel, PfamModel
from rindti.models.base_model import node_embedders, poolers
from rindti.utils import read_config

from .conftest import PROT_EDGE_DIM, PROT_FEAT_DIM


class BaseTestModel:
    @pytest.mark.parametrize("node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("pool", list(poolers.keys()))
    def test_init(self, node_embed, pool):
        self.model()

    @pytest.mark.parametrize("node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("pool", list(poolers.keys()))
    def test_shared_step(self, node_embed, pool, pretrain_batch, default_config, pretrain_dataset):
        default_config["node_embed"] = node_embed
        default_config["pool"] = pool
        default_config["feat_dim"] = PROT_FEAT_DIM
        default_config["edge_dim"] = PROT_EDGE_DIM
        default_config.update(pretrain_dataset.config)
        model = self.model(**default_config)
        model.shared_step(pretrain_batch)


class TestGraphLogModel(BaseTestModel):

    model = GraphLogModel


class TestInfoGraphModel(BaseTestModel):

    model = InfoGraphModel


class TestBGRLModel(BaseTestModel):

    model = BGRLModel


class TestPfamModel(BaseTestModel):

    model = PfamModel
