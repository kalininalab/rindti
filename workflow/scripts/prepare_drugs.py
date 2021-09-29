import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit.Chem.rdchem import ChiralType
from torch_geometric.utils import to_undirected

# node_encoding = {0: 'padding',
#                     1: 6,
#                     2: 8,
#                     3: 7,
#                     4: 16,
#                     5: 9,
#                     6: 17,
#                     7: 35,
#                     8: 15,
#                     9: 53,
#                     10: 11,
#                     11: 14,
#                     12: 5,
#                     13: 19,
#                     14: 34,
#                     15: 33,
#                     16: 30,
#                     17: 51,
#                     18: 3,
#                     19: 13,
#                     20: 20,
#                     21: 12,
#                     22: 52,
#                     23: 47,
#                     24: 56,
#                     25: 1,
#                     26: 38,
#                     27: 23,
#                     28: 'other'}

node_encoding = {
    "other": 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    16: 5,
    17: 6,
    35: 7,
    15: 8,
    53: 9,
    5: 10,
    11: 11,
    14: 12,
    34: 13,
}

glycan_encoding = {
    "other": [0, 0, 0],
    6: [1, 0, 0],  # carbon
    7: [0, 1, 0],  # nitrogen
    8: [0, 0, 1],  # oxygen
}

chirality_encoding = {
    ChiralType.CHI_OTHER: [0, 0, 0],
    ChiralType.CHI_TETRAHEDRAL_CCW: [1, 1, 0],  # counterclockwise rotation of polarized light -> rotate light to the left
    ChiralType.CHI_TETRAHEDRAL_CW: [1, 0, 1],  # clockwise rotation of polarized light -> rotate light to the right
    ChiralType.CHI_UNSPECIFIED: [0, 0, 0],
}

edge_encoding = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "AROMATIC": 2,
}


def featurize(smiles: str) -> dict:
    """Generate drug Data from smiles

    Args:
        smiles (str): SMILES

    Returns:
        dict: dict with x, edge_index etc
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:  # when rdkit fails to read a molecule it returns None
        return np.nan
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    edges = []
    edge_feats = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([start, end])
        btype = str(bond.GetBondType())
        # If bond type is unknown, remove molecule
        if btype not in edge_encoding.keys():
            return np.nan
        edge_feats.append(edge_encoding[btype])
    if not edges:  # If no edges (bonds) were found, remove molecule
        return np.nan
    atom_features = []
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if snakemake.config["mode"] == "gpcr":
            if atom_num not in node_encoding.keys():
                atom_features.append(node_encoding["other"])
            else:
                atom_features.append(node_encoding[atom_num])
        elif snakemake.config["mode"] in ["lectin", "one"]:
            if atom_num not in glycan_encoding.keys():
                cur_atom_features = list(glycan_encoding["other"])
            else:
                cur_atom_features = list(glycan_encoding[atom_num])
            
            if snakemake.config["mode"] == "lectin":
                cur_atom_features += chirality_encoding[atom.GetChiralTag()]
            
            atom_features.append(np.array(cur_atom_features))
    
    x = torch.tensor(atom_features, dtype=torch.long)
    edge_index = torch.tensor(edges).t().contiguous()
    edge_feats = torch.tensor(edge_feats, dtype=torch.long)
    edge_index, edge_feats = to_undirected(edge_index, edge_feats)
    return dict(x=x, edge_index=edge_index, edge_feats=edge_feats)


if __name__ == "__main__":
    ligs = pd.read_csv(snakemake.input.lig, sep="\t", dtype=str).drop_duplicates("Drug_ID").set_index("Drug_ID")
    ligs["data"] = ligs["Drug"].apply(featurize)
    ligs = ligs[ligs["data"].notna()]

    with open(snakemake.output.drug_pickle, "wb") as file:
        pickle.dump(ligs, file)
