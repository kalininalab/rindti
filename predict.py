from pytorch_lightning import seed_everything, Trainer
from torch_geometric.loader import DataLoader

from rindti.utils.data import Dataset
from rindti.utils.transforms import GnomadTransformer, RandomTransformer
from train import models, parse_args
from rindti.utils.utils import remove_arg_prefix


def predict(**kwargs):
    """
    kwargs needs:
    seed

    """
    seed_everything(kwargs["seed"])

    if kwargs["transformer"] != "none":
        transform = {"gnomad": GnomadTransformer, "random": RandomTransformer}[kwargs["transformer"]].from_pickle(
            kwargs["transformer_pickle"], max_num_mut=kwargs["max_num_mut"]
        )
    else:
        transform = None

    train = Dataset(kwargs["data"], split="train", name=kwargs["name"], transform=transform)
    val = Dataset(kwargs["data"], split="val", name=kwargs["name"])
    test = Dataset(kwargs["data"], split="test", name=kwargs["name"])

    kwargs.update(train.config)

    model = models[kwargs["model"]](**kwargs)
    model.load_from_checkpoint(kwargs["checkpoint"])

    # print("Start testing")

    # train_result = trainer.test(model=model, dataloaders=train_loader)
    # val_result = trainer.test(model=model, dataloaders=val_loader)
    # test_result = trainer.test(model=model, dataloaders=test_loader)

    # print("Finished")

    # exit(0)
    for dataset in [test, train, val]:
        print(len(dataset))
        label_sum = 0
        for index in range(len(dataset)):
            record = dataset.get(index)
            prot = remove_arg_prefix("prot_", record)
            drug = remove_arg_prefix("drug_", record)
            pred = model.forward(prot, drug)
            print("Prediction:", pred, "<> Label:", record["label"])
        exit(0)
    print("Finished")


if __name__ == '__main__':
    predict(**parse_args(predict=True))
