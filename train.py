import argparse
import os

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data.dataloader import DataLoader

from rindti.utils.utils import get_timestamp
from rindti.models import ClassificationModel, NoisyNodesClassModel, NoisyNodesRegModel, RegressionModel
from rindti.utils import MyArgParser
from rindti.utils.data import Dataset
from rindti.utils.transforms import GnomadTransformer, RandomTransformer

models = {
    "classification": ClassificationModel,
    "regression": RegressionModel,
    "noisyclass": NoisyNodesClassModel,
    "noisyreg": NoisyNodesRegModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    tmp = np.arange(100)
    np.random.shuffle(tmp)
    seeds = tmp[:kwargs["runs"]]

    if kwargs["transformer"] != "none":
        transform = {"gnomad": GnomadTransformer, "random": RandomTransformer}[kwargs["transformer"]].from_pickle(
            kwargs["transformer_pickle"], max_num_mut=kwargs["max_num_mut"]
        )
    else:
        transform = None

    train = Dataset(kwargs["data"], split="train", name=kwargs["name"], transform=transform)
    val = Dataset(kwargs["data"], split="val", name=kwargs["name"])
    test = Dataset(kwargs["data"], split="test", name=kwargs["name"])

    print("Train-Samples:", len(train))
    print("Validation Samples:", len(val))
    print("Test Samples:", len(test))
    if kwargs["debug"]:
        for i, batch in enumerate(train):
            print(batch)
            if i == 5:
                exit(0)

    kwargs.update(train.config)
    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", patience=kwargs["early_stop_patience"], mode="min"),
    ]
    
    dataloader_kwargs = {k: v for (k, v) in kwargs.items() if k in ["batch_size", "num_workers"]}
    dataloader_kwargs.update({"follow_batch": ["prot_x", "drug_x"]})
    
    sub_folder = os.path.join("tb_logs", kwargs["name"])
    folder = os.path.join(sub_folder, kwargs["model"] + ":" + kwargs["data"].split("/")[-1].split(".")[0])
    
    if not os.path.exists(sub_folder):
        os.mkdir(subfolder)

    if not os.path.exists(folder):
        os.mkdir(folder)

    if len(os.listdir(folder)) == 0:
        next_version = "0"
    else:
        next_version = str(int([d for d in os.listdir(folder) if "version" in d and os.path.isdir(os.path.join(folder, d))][-1].split("_")[1]) + 1)

    for seed in seeds:
        if kwargs["runs"] == 1:
            logger = TensorBoardLogger(
                save_dir=os.path.join("tb_logs", kwargs["name"]), default_hp_metric=False, 
                name=kwargs["model"] + ":" + kwargs["data"].split("/")[-1].split(".")[0],
            )
        else:
            logger = TensorBoardLogger(
                save_dir=os.path.join("tb_logs", kwargs["name"], "version_" + next_version), 
                name=kwargs["model"] + ":" + kwargs["data"].split("/")[-1].split(".")[0], 
                version=seed, default_hp_metric=False,
            )
        
        trainer = Trainer(
            gpus=kwargs["gpus"],
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=kwargs["gradient_clip_val"],
            deterministic=True,
            profiler=kwargs["profiler"],
            log_every_n_steps=30,
            max_epochs=kwargs["max_epochs"],
        )

        train_dataloader = DataLoader(train, **dataloader_kwargs, shuffle=True)
        val_dataloader = DataLoader(val, **dataloader_kwargs, shuffle=False)
        test_dataloader = DataLoader(test, **dataloader_kwargs, shuffle=False)
        checkpoint_dir = os.path.join("data", kwargs["data"].split("/")[-1].split(".")[0], "checkpoints")

        print("Run with seed", seed)
        seed_everything(seed)
        model = models[kwargs["model"]](**kwargs)
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)
        trainer.save_checkpoint(os.path.join(checkpoint_dir, kwargs["name"] + "_" + get_timestamp() + ".ckpt"))


def parse_args(predict=False):
    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--model", type=str, default="classification")
    if predict:
        tmp_parser.add_argument("-c", "--checkpoint", type=str, required=True)
    args = tmp_parser.parse_known_args()[0]
    model_type = args.model

    parser = MyArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("data", type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--early_stop_patience", type=int, default=60, help="epochs with no improvement before stop")
    parser.add_argument("--feat_method", type=str, default="element_l1", help="How to combine embeddings")
    parser.add_argument("--name", type=str, default=None, help="Subdirectory to store the graphs in")
    parser.add_argument("--debug", action='store_true', default=False, help="Flag to turn on the debug mode")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to perform to get more reliable results")

    trainer = parser.add_argument_group("Trainer")
    model = parser.add_argument_group("Model")
    optim = parser.add_argument_group("Optimiser")
    transformer = parser.add_argument_group("Transformer")

    trainer.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    trainer.add_argument("--max_epochs", type=int, default=1000, help="Max number of epochs")
    trainer.add_argument("--model", type=str, default="classification", help="Type of model")
    trainer.add_argument("--weighted", type=bool, default=1, help="Whether to weight the data points")
    trainer.add_argument("--gradient_clip_val", type=float, default=30, help="Gradient clipping")
    trainer.add_argument("--profiler", type=str, default=None)

    model.add_argument("--mlp_hidden_dim", default=64, type=int, help="MLP hidden dims")
    model.add_argument("--mlp_dropout", default=0.2, type=float, help="MLP dropout")

    optim.add_argument("--optimiser", type=str, default="adamw", help="Optimisation algorithm")
    optim.add_argument("--momentum", type=float, default=0.3)
    optim.add_argument("--lr", type=float, default=0.0005, help="mlp learning rate")
    optim.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    optim.add_argument("--reduce_lr_patience", type=int, default=20)
    optim.add_argument("--reduce_lr_factor", type=float, default=0.1)

    transformer.add_argument("--transformer", type=str, default="none", help="Type of transformer to apply")
    transformer.add_argument(
        "--transformer_pickle",
        type=str,
        default="../rins/results/prepare_transformer/onehot_simple_transformer.pkl",
    )
    transformer.add_argument("--max_num_mut", type=int, default=100)

    parser = models[model_type].add_arguments(parser)

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    argvars = parse_args()
    train(**argvars)
