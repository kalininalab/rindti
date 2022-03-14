import os.path
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning import seed_everything

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


def identify_set(df, names, col):
    return {(name, df[df[col] == name]["split"].data()[0]) for name in names}


def negative_sampling(df: DataFrame, config: dict):
    proteins = list(df["Target_ID"].unique())
    drugs = list(df["Drug_ID"].unique())
    number = config["parse_dataset"]["augment"] * len(df)

    if config["split"]["method"] not in ["random", "colddrug", "coldtarget"]:
        raise ValueError("Split method unknown!")

    # assign splits aka "train", "test", "val"
    if config["split"]["method"] == "colddrug":
        splitting = dict([(name, df[df["Drug_ID"] == name]["split"].values[0]) for name in drugs])
    elif config["split"]["method"] == "coldtarget":
        splitting = dict([(name, df[df["Target_ID"] == name]["split"].values[0]) for name in proteins])
    else:
        splitting = lambda x: "train" if x < config["split"]["train"] else \
            ("val" if x < config["split"]["train"] + config["split"]["val"] else "test")
    art = pd.DataFrame(columns=df.columns)

    for i in range(number):
        if config["split"]["method"] == "coldtarget":
            prot = proteins[i % len(proteins)]
        else:
            prot = proteins[np.random.randint(0, len(proteins))]

        if config["split"]["method"] == "colddrug":
            drug = drugs[i % len(drugs)]
        else:
            drug = drugs[np.random.randint(0, len(drugs))]

        while len(df[(df["Target_ID"] == prot) & (df["Drug_ID"] == drug)]) > 0:
            if config["split"]["method"] != "coldtarget":
                prot = proteins[np.random.randint(0, len(proteins))]
            if config["split"]["method"] != "colddrug":
                drug = drugs[np.random.randint(0, len(drugs))]

        if config["split"]["method"] == "colddrug":
            split = splitting[drug]
        elif config["split"]["method"] == "coldtarget":
            split = splitting[prot]
        else:
            split = splitting(np.random.random())

        # insert the artificial samples with value 1 as the negative class has labels above the threshold
        art.loc[len(art)] = [len(df) + i, drug, prot, 1, split]

    return art


if __name__ == "__main__":
    seed_everything(snakemake.config["seed"])
    interactions = pd.read_csv(snakemake.input.inter, sep="\t")

    with open(snakemake.input.drugs, "rb") as file:
        drugs = pickle.load(file)

    with open(snakemake.input.proteins, "rb") as file:
        prots = pickle.load(file)

    if os.path.exists(snakemake.input.protseqs):
        protseqs = pd.read_csv(snakemake.input.protseqs, sep='\t')
    else:
        protseqs = None

    interactions = interactions[interactions["Target_ID"].isin(prots.index)]
    interactions = interactions[interactions["Drug_ID"].isin(drugs.index)]
    drug_count = interactions["Drug_ID"].value_counts()
    prot_count = interactions["Target_ID"].value_counts()
    prots["count"] = prot_count
    drugs["count"] = drug_count
    prots["data"] = prots["data"].apply(del_index_mapping)
    prots = prots[prots.index.isin(interactions["Target_ID"])]
    drugs = drugs[drugs.index.isin(interactions["Drug_ID"])]

    if protseqs is not None:
        seqs = []
        for i, row in prots.iterrows():
            seqs.append(protseqs[protseqs["Target_ID"] == i]["AASeq"].values[0])
        prots["seqs"] = seqs

    if snakemake.config["parse_dataset"]["augment"] > 0:
        interactions = pd.concat([interactions, negative_sampling(interactions, snakemake.config)], ignore_index=True)

    full_data = process_df(interactions)
    config = update_config(snakemake.config)

    final_data = {
        "data": full_data,
        "config": config,
        "prots": prots[["data", "count"] if prots is None else ["data", "count", "seqs"]],
        "drugs": drugs[["data", "count"] if "IUPAC" not in drugs.columns else ["data", "count", "IUPAC"]],
    }
    
    with open(snakemake.output.combined_pickle, "wb") as file:
        pickle.dump(final_data, file, protocol=-1)
