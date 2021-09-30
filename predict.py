from pytorch_lightning import seed_everything, Trainer
from torch_geometric.data import DataLoader

from rindti.utils.data import Dataset
from rindti.utils.transforms import GnomadTransformer, RandomTransformer
from train import models, parse_args


def predict(**kwargs):
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

    train_loader = DataLoader(train)
    val_loader = DataLoader(val)
    test_loader = DataLoader(test)

    trainer = Trainer(
        gpus=kwargs["gpus"],
        deterministic=True,
    )

    model = models[kwargs["model"]](**kwargs)
    model.load_from_checkpoint(kwargs["checkpoint"])

    print("Start testing")
    
    train_result = trainer.test(model=model, dataloaders=train_loader)
    val_result = trainer.test(model=model, dataloaders=val_loader)
    test_result = trainer.test(model=model, dataloaders=test_loader)

    print("Finished")

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
    predict(**parse_args(predict=True))
