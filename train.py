from pprint import pprint

import torch
from pytorch_lightning import Trainer

from rindti.data import DTIDataModule
from rindti.models import ClassificationModel
from rindti.utils import MyCLI

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    cli = MyCLI(ClassificationModel, DTIDataModule, subclass_mode_model=True, run=False)
    cfg = cli.config
    pprint(cfg["trainer"])
    trainer = Trainer(**cfg["trainer"])
