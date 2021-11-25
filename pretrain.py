from pprint import pprint

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import instantiate_class

from rindti.data import PreTrainDataModule
from rindti.models.base_model import BasePretrainModel
from rindti.utils import MyCLI
from rindti.utils.cli import get_module

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    cli = MyCLI(BasePretrainModel, PreTrainDataModule, subclass_mode_model=True, run=False)
    cfg = cli.config

    callbacks = (
        [get_module(callback_cfg) for callback_cfg in cfg["trainer"]["callbacks"]]
        if cfg["trainer"]["callbacks"]
        else []
    )
    if cfg["trainer"]["callbacks"]:
        del cfg["trainer"]["callbacks"]
    trainer = Trainer(**cfg["trainer"], callbacks=callbacks)
    data = PreTrainDataModule(**cfg["data"])
    data.setup()
    fam_list = {x.fam for x in data.train}
    cfg["model"]["init_args"]["_fam_list"] = fam_list
    data.update_model_args(cfg["model"]["init_args"])
    model = get_module(cfg["model"], _optimizer_args=cfg["optimizer"], _lr_scheduler_args=cfg["lr_scheduler"])
    trainer.fit(model, data)
