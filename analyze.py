import pandas as pd
from rdkit import Chem


def main():
    df = pd.read_csv("./resources/drugs/lig.tsv", sep="\t")
    stats = {}
    for index, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["Drug"])
        for atom in mol.GetAtoms():
            print(atom.GetChiralTag())
            s = atom.GetSymbol()
            if s not in stats:
                stats[s] = 0
            stats[s] += 1
        break
    print(stats)


if __name__ == '__main__':
    main()

