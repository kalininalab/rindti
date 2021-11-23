from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import Trainer

from rindti.data import DTIDataModule
from rindti.models import ClassificationModel

parser = ArgumentParser(logger=True)
parser.add_argument("--config", action=ActionConfigFile)
parser.add_class_arguments(ClassificationModel, "model")
parser.add_class_arguments(DTIDataModule, "data")
parser.add_class_arguments(Trainer, "trainer")
cfg = parser.parse_args()
cfg = namespace_to_dict(cfg)
dm = DTIDataModule(**cfg["data"])
dm.setup()
for pref in ["prot_", "drug_"]:
    cfg["model"][f"{pref}encoder"].update(dm.get_config(pref))
model = ClassificationModel(**cfg["model"])
trainer = Trainer(**cfg["trainer"])
trainer.fit(model, dm)
