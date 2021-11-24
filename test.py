from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import LightningDataModule, Trainer

from rindti.data import DTIDataModule
from rindti.models.base_model import BaseModel
from rindti.utils import get_module


def train_dti(model: BaseModel, data_module: DTIDataModule, trainer: Trainer, **kwargs):
    data_module = get_module(data_module)
    data_module.setup(stage="fit")
    print(data_module)
    data_module.update_model_args(model["init_args"])
    model = get_module(model)
    trainer = get_module(trainer)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser(logger=True)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(train_dti)
    cfg = namespace_to_dict(parser.parse_args())
    train_dti(**cfg)
