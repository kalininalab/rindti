import os
import pickle
from typing import Callable, Iterable

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset

from ..utils import get_type
from .data import TwoGraphData


class DTIDataset(InMemoryDataset):
    """Dataset class for proteins and drugs

    Args:
        filename (str): Pickle file that stores the data
        split (str, optional): Split type ('train', 'val', 'test). Defaults to "train".
        transform (Callable, optional): transformer to apply on each access. Defaults to None.
        pre_transform (Callable, optional): pre-transformer to apply once before. Defaults to None.
    """

    splits = {"train": 0, "val": 1, "test": 2}

    def _set_filenames(self, filename: str) -> str:
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        self.filename = filename
        return os.path.join("data", basefilename)

    def _set_types(self, data: dict) -> dict:
        """Sets feat type in the self.config from snakemake self.config"""
        self.config["prot_feat_type"] = get_type(data, "prot_x")
        self.config["drug_feat_type"] = get_type(data, "drug_x")
        self.config["prot_edge_type"] = get_type(data, "prot_edge_feats")
        self.config["drug_edge_type"] = get_type(data, "drug_edge_feats")
        return self.config

    def process_(self, data_list: list, split: str):
        """Process the datalist

        Args:
            data_list (list): List of TwoGraphData entries
            s (int): index of train, val or test
        """
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, self.config), self.processed_paths[self.splits[split]])

    def _get_datum(self, all_data: dict, id: str, which: str) -> dict:
        """Get either prot or drug data"""
        graph = all_data[which].loc[id, "data"]
        graph["count"] = float(all_data[which].loc[id, "count"])
        graph["id"] = id
        return {which.rstrip("s") + "_" + k: v for k, v in graph.items()}

    @property
    def processed_file_names(self) -> Iterable[str]:
        """Files that are created"""
        return [k + ".pt" for k in self.splits.keys()]

    def __init__(
        self,
        filename: str,
        split: str = "train",
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ):
        root = self._set_filenames(filename)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.config = torch.load(self.processed_paths[self.splits[split]])

    def process(self):
        """If the dataset was not seen before, process everything"""
        with open(self.filename, "rb") as file:
            all_data = pickle.load(file)
            self.config = all_data["config"]
            self.config["prot_max_nodes"] = 0
            self.config["drug_max_nodes"] = 0
            for split in self.splits.keys():
                data_list = []
                for i in all_data["data"]:
                    if i["split"] != split:
                        continue
                    data = self._get_datum(all_data, i["prot_id"], "prots")
                    data.update(self._get_datum(all_data, i["drug_id"], "drugs"))
                    data["label"] = i["label"]
                    two_graph_data = TwoGraphData(**data)
                    two_graph_data.num_nodes = 1  # supresses the warning
                    self.config["prot_max_nodes"] = max(self.config["prot_max_nodes"], two_graph_data.n_nodes("prot_"))
                    self.config["drug_max_nodes"] = max(self.config["drug_max_nodes"], two_graph_data.n_nodes("drug_"))
                    data_list.append(two_graph_data)
                    self.config = self._set_types(data_list[0])
                if data_list:
                    self.process_(data_list, split)


class PreTrainDataset(InMemoryDataset):
    """Dataset class for pre-training

    Args:
        filename (str): Pickle file that stores the data
        split (str, optional): Split type ('train', 'val', 'test). Defaults to "train".
        transform (Callable, optional): transformer to apply on each access. Defaults to None.
        pre_transform (Callable, optional): pre-transformer to apply once before. Defaults to None.
    """

    def __init__(self, filename: str, transform: Callable = None, pre_transform: Callable = None):
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        root = os.path.join("data", basefilename)
        self.filename = filename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.config = torch.load(self.processed_paths[0])

    def index(self, id: str):
        """Find protein by id"""
        return self[self.data.id.index(id)]

    @property
    def processed_file_names(self) -> Iterable[str]:
        """Which files have to be in the dir to consider dataset processed

        Returns:
            Iterable[str]: list of files
        """
        return ["data.pt"]

    def process(self):
        """If the dataset was not seen before, process everything"""
        self.config = dict(max_nodes=0)
        df = pd.read_pickle(self.filename)
        data_list = []
        for id, x in df["data"].to_dict().items():
            if "index_mapping" in x:
                del x["index_mapping"]
            self.config["max_nodes"] = max(self.config["max_nodes"], x["x"].size(0))
            x["id"] = id
            data_list.append(Data(**x))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.config["feat_type"] = get_type(data_list[0], "x")
        self.config["edge_type"] = get_type(data_list[0], "edge_feats")
        data, slices = self.collate(data_list)
        torch.save((data, slices, self.config), self.processed_paths[0])


class LargePreTrainDataset(Dataset):
    """Dataset that doesn't fit in the memory"""

    def __init__(self, rawdir, transform=None, pre_transform=None):
        self.rawdir = rawdir
        dirname = rawdir.rstrip("/").split("/")[-1]
        root = os.path.join("data", dirname)
        super().__init__(root, transform, pre_transform)
        self.config = torch.load(self.config_file)

    @property
    def raw_file_names(self):
        """Sharded pickles"""
        return os.listdir(self.rawdir)

    @property
    def config_file(self):
        """Saved self.config file"""
        return os.path.join(self.processed_dir, "self.config.pt")

    @property
    def processed_file_names(self):
        """Saved graphs"""
        return [os.path.join(self.processed_dir, x) for x in ["self.config.pt", "data_0.pt"]]

    def process(self):
        """Save each graph as a file"""
        i = 0
        self.config = dict(max_nodes=0)
        for shard in os.listdir(self.rawdir):
            df = pd.read_pickle(os.path.join(self.rawdir, shard))
            for id, x in df["data"].to_dict().items():
                if "index_mapping" in x:
                    del x["index_mapping"]
                self.config["max_nodes"] = max(self.config["max_nodes"], x["x"].size(0))
                x["id"] = id
                data = Data(**x)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, "data_{}.pt".format(i)))
                i += 1
        self.config["count"] = i
        torch.save(self.config, self.config_file)

    def len(self):
        return self.config["count"]

    def get(self, idx):
        """Load graph"""
        return torch.load(os.path.join(self.processed_dir, "data_{}.pt".format(idx)))
