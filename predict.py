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

    for dataset in [train, val, test]:
        for index in len(dataset):
            data = dataset.get(index)
            print(data)
            exit(0)


if __name__ == '__main__':
    tmp_parser = argparse.ArgumentParser()
    tmp_parser.add_argument("-m", "--model", type=str, default="classification")
    args = tmp_parser.parse_known_args()[0]
    model_type = args.model

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, default=None, help="Subdirectory to store the graphs in")
    parser.add_argument("-d", "--debug", action='store_true', default=False, help="Flag to turn on the debug mode")
    parser.add_argument("-l", "--lectin")
    parser.add_argument("-g", "--glycan")
    parser = models[model_type].add_arguments(parser)
    argvars = parser.parse_args()

    predict(**argvars)
