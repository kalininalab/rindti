import pickle
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from rindti.models import PfamModel
from rindti.utils.data import PreTrainDataset
from rindti.utils.transforms import PfamTransformer

with open(sys.argv[1], "rb") as file:
    merged_df = pickle.load(file)
transformer = PfamTransformer(merged_df)
dataset = PreTrainDataset(sys.argv[1], transform=transformer, pre_filter=transformer._filter)
print(len(dataset))
logger = TensorBoardLogger("tb_logs", name="pfam", default_hp_metric=False)
callbacks = [
    ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min"),
    EarlyStopping(monitor="train_loss", patience=20, mode="min"),
]
from torch_geometric.data import DataLoader

trainer = Trainer(
    gpus=1,
    callbacks=callbacks,
    logger=logger,
    gradient_clip_val=30,
    max_epochs=100,
    stochastic_weight_avg=True,
)
model = PfamModel(
    node_embed_dim=32,
    node_embed="ginconv",
    pool="gmt",
    feat_dim=dataset.info["feat_dim"],
    feat_method="concat",
    max_nodes=dataset.info["max_nodes"],
    embed_dim=32,
    optimiser="adam",
    lr=0.001,
    weight_decay=0.01,
    reduce_lr_patience=10,
    reduce_lr_factor=0.1,
    hidden_dim=32,
)
dl = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=True, follow_batch=["a_x", "b_x"])
trainer.fit(model, train_dataloader=dl)
