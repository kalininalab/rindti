import torch

from rindti.data import PreTrainDataModule
from rindti.models.base_model import BaseModel
from rindti.models.pretrain import BGRLModel, GraphLogModel, InfoGraphModel, PfamModel
from rindti.utils import PreTrainCLI

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    cli = PreTrainCLI(BaseModel, PreTrainDataModule, subclass_mode_model=True)
