from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from .datasets import DTIDataset


class DTIDataModule(LightningDataModule):
    """LightningDataModule for DTI, contains all the datasets for train, val and test"""

    def __init__(self, filename: str, batch_size: int = 128, num_workers: int = 16, shuffle: bool = True, **kwargs):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
            follow_batch=["prot_x", "drug_x"],
        )

    def setup(self, stage: str = None):
        """Load the individual datasets"""
        if stage == "fit" or stage is None:
            self.train = DTIDataset(self.filename, split="train").shuffle()
            self.val = DTIDataset(self.filename, split="val").shuffle()
        if stage == "test" or stage is None:
            self.test = DTIDataset(self.filename, split="test").shuffle()
        self.config = self.train.config

    def get_config(self, prefix: str) -> dict:
        """Get the config for a single prefix"""
        return {k.strip(prefix): v for k, v in self.config.items() if k.startswith(prefix)}

    def update_model_args(self, model_init_args: dict):
        """Update the model arguments with the config"""
        for pref in ["prot_", "drug_"]:
            model_init_args[f"{pref}encoder"].update(self.get_config(pref))

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
