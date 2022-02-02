import pickle
import pandas as pd
import os

from Bio.PDB import PDBParser
from glycowork.ml.models import prep_model
from glycowork.ml.inference import get_multi_pred
import torch
import esm
import argparse

from Bio import SeqIO


def get_esm_1b(proteins):
    """
    Args:
        proteins (List[Tuple[str, str]]): proteins as names and amino acid sequence
    """
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR505()
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(proteins)

    with torch.no_grad():
        results = model(batch_tokens, repr_layer=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, (_, seq) in enumerate(proteins):
        sequence_representations.append(token_representations[i, 1: len(seq) + 1].mean(0))

    return sequence_representations


def read_prot_sequence(name, path):
    chain = {record.id: record.seq for record in SeqIO.parse(os.path.join(path, "structures", name + ".pdb"), 'pdb-seqres')}
    pdbparser = PDBParser()

    structure = pdbparser.get_structure("ASDF", os.path.join(path, "structures", name + ".pdb"))
    return structure.get_chains()[0]


def main(**kwargs):
    print("=== Load data and representations ===")
    oracle = prep_model("LectinOracle", num_classes=1, trained=True)

    data = pickle.load(open(kwargs["data"], "rb"))["data"]
    glycan_data = pd.read_csv(os.path.join(kwargs["path"], "drugs", "lig.tsv"), sep="\t")

    lectins, glycans = {}, {}
    for entry in data:
        if entry["prot_id"] not in lectins:
            lectins[entry["prot_id"]] = read_prot_sequence(entry["prot_id"], kwargs["path"])
        if entry["drug_id"] not in glycans:
            glycans[entry["drug_id"]] = glycan_data[glycan_data.Drug_ID == data[0]["drug_id"]].IUPAC.values[0]
    lectin_names = list(lectins.values())
    protein_representations = get_esm_1b(lectin_names)

    for entry in data:
        get_multi_pred(
            lectins[entry["prot_id"]],
            glycans[entry["drug_id"]],
            oracle,
            {lectins[entry["prot_id"]]: protein_representations[lectin_names.index(lectins[entry["prot_id"]])]}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("-p", "--path", type=str, required=True, dest="path")
    parser.add_argument("-t", "--train", type=bool, default=False, dest="train")
    args = parser.parse_args()

    main(**vars(args))
