import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from prepare_drugs import edge_encoding as drug_edge_encoding
from prepare_drugs import node_encoding as drug_node_encoding
from prepare_proteins import edge_encoding as prot_edge_encoding
from prepare_proteins import node_encoding as prot_node_encoding


def process(row: pd.Series) -> dict:
    """Process each interaction, drugs encoded as graphs"""
    return {
        "label": row["Y"],
        "split": row["split"],
        "prot_id": row["Target_ID"],
        "drug_id": row["Drug_ID"],
    }


def process_df(df: DataFrame) -> Iterable[dict]:
    """Apply process() function to each row of the DataFrame"""
    return [process(row) for (_, row) in df.iterrows()]


def del_index_mapping(x: dict) -> dict:
    """Delete 'index_mapping' entry from the dict"""
    del x["index_mapping"]
    return x


def update_config(config: dict) -> dict:
    """Updates config with dims of everything"""
    config["prot_feat_dim"] = len(prot_node_encoding)
    config["drug_feat_dim"] = len(drug_node_encoding)
    if config["prepare_proteins"]["edge_feats"] == "label":
        config["prot_edge_dim"] = len(prot_edge_encoding)
    else:
        config["prot_edge_dim"] = 1
    config["drug_edge_dim"] = len(drug_edge_encoding)
    return config


def augment(df: DataFrame, config: dict):
    if config["parse_dataset"]["augment"] == 0:
        return
    proteins = list(df["Target_ID"].unique())
    drugs = list(df["Drug_ID"].unique())
    number = config["parse_dataset"]["augment"] * len(df)

    # assign splits aka "train", "test", "val"
    splitting = lambda x: "train" if x < config["split"]["train"] else \
        ("val" if x < config["split"]["train"] + config["split"]["val"] else "test")

    for _ in range(number):
        prot = proteins[np.random.randint(0, len(proteins))]
        drug = drugs[np.random.randint(0, len(drugs))]

        while len(df[df["Target_ID"] == prot & df["Drug_ID"] == drug]) > 0:
            prot = proteins[np.random.randint(0, len(proteins))]
            drug = drugs[np.random.randint(0, len(drugs))]

        df.append({"Target_ID": prot, "Drug_ID": drug,
                   "split": splitting(np.random.random()), "Y": 1}, ignore_index=True)


if __name__ == "__main__":

    interactions = pd.read_csv(snakemake.input.inter, sep="\t")

    with open(snakemake.input.drugs, "rb") as file:
        drugs = pickle.load(file)

    with open(snakemake.input.proteins, "rb") as file:
        prots = pickle.load(file)
    interactions = interactions[interactions["Target_ID"].isin(prots.index)]
    interactions = interactions[interactions["Drug_ID"].isin(drugs.index)]
    drug_count = interactions["Drug_ID"].value_counts()
    prot_count = interactions["Target_ID"].value_counts()
    prots["count"] = prot_count
    drugs["count"] = drug_count
    prots["data"] = prots["data"].apply(del_index_mapping)
    prots = prots[prots.index.isin(interactions["Target_ID"])]
    drugs = drugs[drugs.index.isin(interactions["Drug_ID"])]

    augment(interactions, snakemake.config)

    full_data = process_df(interactions)
    config = update_config(snakemake.config)

    final_data = {
        "data": full_data,
        "config": config,
        "prots": prots[["data", "count"]],
        "drugs": drugs[["data", "count"]],
    }
    with open(snakemake.output.combined_pickle, "wb") as file:
        pickle.dump(final_data, file, protocol=-1)
