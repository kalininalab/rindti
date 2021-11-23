from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from .datasets import DTIDataset


class DTIDataModule(LightningDataModule):
    def __init__(self, filename: str, batch_size: int = 64, num_workers: int = 0, shuffle: bool = True, **kwargs):
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
        if stage == "fit" or stage is None:
            self.train = DTIDataset(self.filename, split="train")
            self.val = DTIDataset(self.filename, split="val")
        if stage == "test" or stage is None:
            self.test = DTIDataset(self.filename, split="test")
        self.config = self.train.config

    def get_config(self, prefix: str) -> dict:
        return {k.strip(prefix): v for k, v in self.config.items() if k.startswith(prefix)}

    def train_dataloader(self):
        return DataLoader(self.train, **self._dl_kwargs(True))

    def val_dataloader(self):
        return DataLoader(self.val, **self._dl_kwargs(False))

    def test_dataloader(self):
        return DataLoader(self.test, **self._dl_kwargs(False))
