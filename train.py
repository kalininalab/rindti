import torch

from rindti.data import DTIDataModule
from rindti.models import ClassificationModel
from rindti.utils import TrainCLI

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    cli = TrainCLI(ClassificationModel, DTIDataModule, subclass_mode_model=True)
