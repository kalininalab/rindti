import pickle
from glycowork.ml.models import prep_model
from glycowork.ml.inference import get_multi_pred
import torch
import os
import argparse
import time

from lectinoracle import extract

cut = 1022


def run_esm_1b():
    esm_parser = extract.create_parser()
    esm_args = esm_parser.parse_args(["esm1b_t33_650M_UR50S", "lectinoracle/seqs.fasta", "lectinoracle/embeds",
                                      "--repr_layers", "33", "--include", "mean"])
    extract.main(esm_args)
    embeds = {}
    lines = list(open("lectinoracle/seqs.fasta").readlines())
    for i in range(0, len(lines), 2):
        lec, seq = lines[i:i + 2]
        repres = torch.load(f"./lectinoracle/embeds/{lec.strip()[1:]}.pt")["mean_representations"][33]
        embeds[seq.strip()] = repres
    return embeds


def extract_data(**kwargs):
    global cut

    print("============ Load data =============")
    start = time.time()

    data = pickle.load(open(kwargs["data"], "rb"))
    lectin_data = data["prots"]
    esm_embeds = pickle.load(open(kwargs["prot"], "rb")) if os.path.exists(kwargs["prot"]) else {}

    tmp = []
    output = open("lectinoracle/seqs.fasta", "w")
    for name, row in lectin_data.iterrows():
        if row.seqs[:cut] not in esm_embeds:
            tmp.append((name, row.seqs[:cut]))
            output.write(f">{name}\n{row.seqs[:cut]}\n")
    output.close()
    print(f"\t{len(tmp)} sequences not contained")
    print(f"\tFinished in {time.time() - start:0.5f} sec")

    print("===== generate representations =====")
    start = time.time()
    if len(tmp) > 0:
        representations = run_esm_1b()
        esm_embeds = {**esm_embeds, **representations}
        pickle.dump(esm_embeds, open(kwargs["prot"], "wb"))
    print(f"\tFinished in {time.time() - start:0.5f} sec")


def inference(kwargs):
    global cut

    extract_data(**kwargs)

    data = pickle.load(open(kwargs["data"], "rb"))
    lectin_data = data["prots"]
    glycan_data = data["drugs"]
    esm_embeds = pickle.load(open(kwargs["prot"], "rb")) if os.path.exists(kwargs["prot"]) else {}

    print("====== evaluate LectinOracle =======")
    oracle = prep_model("LectinOracle", num_classes=1, trained=True)

    for i, sample in enumerate(data["data"]):
        seq = lectin_data.loc[sample["prot_id"]].seqs[:cut]
        lec = esm_embeds[seq]
        gly = glycan_data.loc[sample["drug_id"]].IUPAC
        gly = gly.replace("NeuAc", "Neu5Ac")
        print(f"{sample['prot_id']} | {gly} ({sample['label']}): {get_multi_pred(seq, [gly], oracle, esm_embeds)}")


def training(kwargs):
    pass


def main(**kwargs):
    if kwargs["train"]:
        training(**kwargs)
    else:
        inference(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to pickled dataset")
    parser.add_argument("prot", type=str, help="Path to pickled dictionary mapping AA-sequences to ESM-1b embeddings")
    parser.add_argument("-t", "--train", type=bool, default=False, dest="train")
    args = parser.parse_args()

    main(**vars(args))
