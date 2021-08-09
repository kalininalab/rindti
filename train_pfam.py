import pickle
from pprint import pprint

from rindti.models import PfamModel
from rindti.utils.data import PreTrainDataset
from rindti.utils.transforms import PfamTransformer


def train(**kwargs):
    with open(kwargs["data"], "rb") as file:
        merged_df = pickle.load(file)
    transformer = PfamTransformer(merged_df)
    dataset = PreTrainDataset(kwargs["data"], transform=transformer)
    pprint(dataset[0])
    print("\n\n\n")
    pprint(dataset[0])


train(**{"data": "../rindti_alphafold/data/merged_df.pkl"})
