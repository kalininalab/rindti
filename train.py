from rindti.data import DTIDataModule
from rindti.models import ClassificationModel
from rindti.utils import TrainCLI

if __name__ == "__main__":
    cli = TrainCLI(ClassificationModel, DTIDataModule)
