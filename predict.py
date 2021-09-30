import argparse

from pytorch_lightning import seed_everything

from rindti.utils.data import Dataset
from rindti.utils.transforms import GnomadTransformer, RandomTransformer
from train import models


def predict(**kwargs):
    seed_everything(kwargs["seed"])

    if kwargs["transformer"] != "none":
        transform = {"gnomad": GnomadTransformer, "random": RandomTransformer}[kwargs["transformer"]].from_pickle(
            kwargs["transformer_pickle"], max_num_mut=kwargs["max_num_mut"]
        )
    else:
        transform = None

    model = models[kwargs["model"]](**kwargs)
    model.load_from_checkpoint(kwargs["checkpoint"])

    train = Dataset(kwargs["data"], split="train", name=kwargs["name"], transform=transform)
    val = Dataset(kwargs["data"], split="val", name=kwargs["name"])
    test = Dataset(kwargs["data"], split="test", name=kwargs["name"])

    train_dataloader = DataLoader(train)
    val_loader = DataLoader(val)
    test_loader = DataLoader(test)

    trainer = Trainer(
        gpus=kwargs["gpu"],
        deterministic=True,
    )

    train_result = trainer.test(model=model, dataloaders=train_loader)
    val_result = trainer.test(model=model, dataloaders=val_loader)
    test_result = trainer.test(model=model, dataloaders=test_loader)

    exit(0)
    for dataset in [test, train, val]:
        print(len(dataset))
        for index in range(len(dataset)):
            data = dataset.get(index)
            print(data)
            print(data.y)
            data = dataset.get(len(dataset) - index - 1)
            print(data)
            print(data.y)
            exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="classification")
    parser.add_argument("data", type=str)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, default=None, help="Subdirectory to store the graphs in")
    parser.add_argument("-d", "--debug", action='store_true', default=False, help="Flag to turn on the debug mode")
    parser.add_argument("-l", "--lectin")
    parser.add_argument("-g", "--glycan")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-t", "--transformer", type=str, default="none")
    # parser = models[model_type].add_arguments(parser)
    argvars = vars(parser.parse_args())

    predict(**argvars)
