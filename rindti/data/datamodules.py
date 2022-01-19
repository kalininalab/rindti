from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.sampler import Sampler
from torch_geometric.loader import DataLoader

from ..utils import get_module, split_random
from .datasets import DTIDataset, PreTrainDataset
from .samplers import PfamSampler


class BaseDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test"""

    def __init__(self, filename: str, batch_size: int = 128, num_workers: int = 16, shuffle: bool = True):
        """Base DataModule

        Args:
            filename (str): Pickle file containing the dataset
            batch_size (int, optional): Size of the batch. Defaults to 128.
            num_workers (int, optional): Workers for loading the data. Defaults to 16.
            shuffle (bool, optional): Whether to shuffle training set. Defaults to True.
        """
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

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
    """DataModule for the DTI class

    Args:
        filename (str): Pickle file containing the dataset
        batch_size (int, optional): Size of the batch. Defaults to 128.
        num_workers (int, optional): Workers for loading the data. Defaults to 16.
        shuffle (bool, optional): Whether to shuffle training set. Defaults to True.
    """

    def setup(self, stage: str = None):
        """Load the individual datasets"""
        if stage == "fit" or stage is None:
            self.train = DTIDataset(self.filename, split="train").shuffle()
            self.val = DTIDataset(self.filename, split="val").shuffle()
        if stage == "test" or stage is None:
            self.test = DTIDataset(self.filename, split="test").shuffle()
        self.config = self.train.config

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
            follow_batch=["prot_x", "drug_x"],
        )

    def update_model_args(self, model_init_args: dict):
        """Update the model arguments with the config"""
        for pref in ["prot_", "drug_"]:
            model_init_args[f"{pref}encoder"].update(self.get_config(pref))


class PreTrainDataModule(BaseDataModule):
    """DataModule for the protein pretraining class

    Args:
        filename (str): Pickle file containing the dataset
        batch_size (int, optional): Size of the batch. Defaults to 128.
        num_workers (int, optional): Workers for loading the data. Defaults to 16.
        shuffle (bool, optional): Whether to shuffle training set. Defaults to True.
        sampler (Optional[Sampler], optional): Sampler for the training set. Defaults to None.
        train_frac (float, optional): Fraction of the dataset to use for training. Defaults to 0.8.
    """

    def __init__(self, *args, sampler: Optional[Sampler] = None, train_frac: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        self.train_frac = train_frac

    def setup(self):
        """Load and split the dataset"""
        dataset = PreTrainDataset(self.filename)
        self.train, self.val = split_random(dataset, self.train_frac)
        self.config = dataset.config

    def update_model_args(self, model_init_args: dict):
        """Update the model arguments with the config"""
        model_init_args["encoder"].update(self.get_config())

    def train_dataloader(self):
        """Return the train dataloader"""
        sampler = get_module(self.sampler, self.train)
        return DataLoader(self.train, batch_sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        """Return the val dataloader"""
        sampler = get_module(self.sampler, self.val)
        return DataLoader(self.val, batch_sampler=sampler, num_workers=self.num_workers)
