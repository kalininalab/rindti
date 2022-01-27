from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader
from torch_geometric.data import DataLoader

from rindti.data import DTIDataset
from rindti.models import ClassificationModel, RegressionModel
from rindti.utils import read_config
from rindti.utils.utils import get_timestamp
from rindti.models import ClassificationModel, NoisyNodesClassModel, NoisyNodesRegModel, RegressionModel
from rindti.utils import MyArgParser
from rindti.utils.data import Dataset
from rindti.utils.transforms import GnomadTransformer, RandomTransformer

models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    train = DTIDataset(kwargs["data"], split="train").shuffle()
    val = DTIDataset(kwargs["data"], split="val").shuffle()
    test = DTIDataset(kwargs["data"], split="test").shuffle()
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
        exit(0)

    kwargs.update(train.config)
    dataloader_kwargs = {k: v for (k, v) in kwargs.items() if k in ["batch_size", "num_workers"]}
    dataloader_kwargs.update({"follow_batch": ["prot_x", "drug_x"]})

    sub_folder = os.path.join("tb_logs", kwargs["name"])
    model_name = kwargs["model"] + ":" + kwargs["data"].split("/")[-1].split(".")[0]
    folder = os.path.join(sub_folder, model_name)

    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)

    if not os.path.exists(folder):
        os.mkdir(folder)

    if len(os.listdir(folder)) == 0:
        next_version = "0"
    else:
        next_version = str(int([d for d in os.listdir(folder) if "version" in d and os.path.isdir(os.path.join(folder, d))][-1].split("_")[1]) + 1)

    for seed in seeds:
        callbacks = [
            ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
            EarlyStopping(monitor="val_loss", patience=kwargs["early_stop_patience"], mode="min"),
        ]
        if kwargs["predict"]:
            logger = None
        else:
            if kwargs["runs"] == 1:
                logger = TensorBoardLogger(
                    save_dir=os.path.join("tb_logs", kwargs["name"]), default_hp_metric=False, name=model_name,
                )
            else:
                logger = TensorBoardLogger(
                    save_dir=os.path.join("tb_logs", kwargs["name"], model_name),
                    name="version_" + next_version, version=seed, default_hp_metric=False,
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
        val_dataloader = DataLoader(val, **dataloader_kwargs, shuffle=True)
        test_dataloader = DataLoader(test, **dataloader_kwargs, shuffle=True)

        base_dir = os.path.join("data", kwargs["data"].split("/")[-1].split(".")[0])
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        prediction_dir = os.path.join(base_dir, "predictions")
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if not os.path.isdir(prediction_dir):
            os.mkdir(prediction_dir)

        seed_everything(seed)

        if not kwargs["predict"]:
            model = models[kwargs["model"]](**kwargs)
            trainer.fit(model, train_dataloader, val_dataloader)
            trainer.test(model, test_dataloader)
            trainer.save_checkpoint(os.path.join(checkpoint_dir, kwargs["name"] + "_" + get_timestamp() + ".ckpt"))
        else:
            model = models[kwargs["model"]].load_from_checkpoint(kwargs["checkpoint"])
            print("Start testing")

            protein_ids, drug_ids = set(), set()
            for dataset in [train, val, test]:
                for record in dataset:
                    protein_ids.add(record["prot_id"])
                    drug_ids.add(record["drug_id"])
    logger = TensorBoardLogger(
        "tb_logs", name=kwargs["model"] + ":" + kwargs["data"].split("/")[-1].split(".")[0], default_hp_metric=False
    )
    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", patience=kwargs["early_stop_patience"], mode="min"),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(
        gpus=kwargs["gpus"],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=kwargs["gradient_clip_val"],
        profiler=kwargs["profiler"],
        log_every_n_steps=25,
    )
    model = models[kwargs["model"]](**kwargs)
    dataloader_kwargs = {k: v for (k, v) in kwargs.items() if k in ["batch_size", "num_workers"]}
    dataloader_kwargs["follow_batch"] = ["prot_x", "drug_x"]
    train_dataloader = DataLoader(train, **dataloader_kwargs, shuffle=False)
    val_dataloader = DataLoader(val, **dataloader_kwargs, shuffle=False)
    test_dataloader = DataLoader(test, **dataloader_kwargs, shuffle=False)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    from rindti.utils import MyArgParser

    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--model", type=str, default="class")
    args = tmp_parser.parse_known_args()[0]
    model_type = args.model

    parser = MyArgParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = read_config(args.config)
    pprint(config)
    train(**config)
