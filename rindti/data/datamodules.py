from pytorch_lightning import LightningDataModule
from torch.utils.data.sampler import Sampler
from torch_geometric.loader import DataLoader

from ..utils import get_module, split_random
from .datasets import DTIDataset, PreTrainDataset
from .samplers import PfamSampler


class BaseDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test"""

    def __init__(
        self,
        filename: str,
        batch_size: int = 128,
        num_workers: int = 16,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.setup()

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
        )

    def get_config(self, prefix: str = "") -> dict:
        """Get the config for a single prefix"""
        return {k.strip(prefix): v for k, v in self.config.items() if k.startswith(prefix)}

    def train_dataloader(self):
        return DataLoader(self.train, **self._dl_kwargs(True))

    def val_dataloader(self):
        return DataLoader(self.val, **self._dl_kwargs(False))

    def test_dataloader(self):
        return DataLoader(self.test, **self._dl_kwargs(False))

    def __repr__(self):
        return "DTI DataModule\n" + "\n".join(
            [repr(getattr(self, x)) for x in ["train", "val", "test"] if hasattr(self, x)]
        )


class DTIDataModule(BaseDataModule):
    def setup(self, stage: str = None):
        """Load the individual datasets"""
        if stage == "fit" or stage is None:
            self.train = DTIDataset(self.filename, split="train").shuffle()
            self.val = DTIDataset(self.filename, split="val").shuffle()
        if stage == "test" or stage is None:
            self.test = DTIDataset(self.filename, split="test").shuffle()
        self.config = self.train.config
        self.prot_feat_dim = self.config["prot_feat_dim"]
        self.drug_feat_dim = self.config["drug_feat_dim"]
        self.prot_feat_type = self.config["prot_feat_type"]
        self.drug_feat_type = self.config["drug_feat_type"]
        self.prot_edge_dim = self.config["prot_edge_dim"]
        self.drug_edge_dim = self.config["drug_edge_dim"]
        self.prot_edge_type = self.config["prot_edge_type"]
        self.drug_edge_type = self.config["drug_edge_type"]
        self.prot_max_nodes = self.config["prot_max_nodes"]
        self.drug_max_nodes = self.config["drug_max_nodes"]

    def _dl_kwargs(self, shuffle: bool = False):
        kwargs = super()._dl_kwargs(shuffle)
        kwargs["follow_batch"] = ["prot_x", "drug_x"]
        return kwargs


class PreTrainDataModule(BaseDataModule):
    def setup(self, stage: str = None):
        """Load the individual datasets"""
        if stage == "fit" or stage is None:
            ds = PreTrainDataset(self.filename).shuffle()
            self.train, self.val = split_random(ds, 0.8)
        self.feat_dim = self.config["feat_dim"]
        self.feat_type = self.config["feat_type"]
        self.edge_dim = self.config["edge_dim"]
        self.edge_type = self.config["edge_type"]
        self.max_nodes = self.config["max_nodes"]
