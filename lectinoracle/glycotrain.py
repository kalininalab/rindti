import pickle

import pandas as pd
from glycowork.glycowork import lib
from glycowork.ml.models import prep_model, LectinOracle
from glycowork.ml.model_training import train_model as tm_lib, SAM
from glycowork.ml.inference import get_multi_pred
import torch
import os
import argparse
import time
import torch.nn as nn
import torch.optim as optim
from glycowork.motif.graph import glycan_to_graph
from glycowork.motif.processing import get_lib

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from lectinoracle import extract
from model_training_rwrkd import train_model as tm_own

from rindti.models import ClassificationModel, RegressionModel
from rindti.utils.data import TwoGraphData

models = {
    "classification": ClassificationModel,
    "regression": RegressionModel,
}

cut = 1022
lib += ["Gal3S", "Neu5Ac", "Gal4S", "Glc6S", "GlcNAc6S"]


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
    esm_embeds = pickle.load(open(kwargs["scnd"], "rb")) if os.path.exists(kwargs["scnd"]) else {}

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
        pickle.dump(esm_embeds, open(kwargs["scnd"], "wb"))
    print(f"\tFinished in {time.time() - start:0.5f} sec")


def read_data(pickle_data, split, **kwargs):
    file = open(pickle_data, "rb")
    all_data = pickle.load(file)
    lectin_data = all_data["prots"]
    glycan_data = all_data["drugs"]
    esm_embeds = pickle.load(open(kwargs["scnd"], "rb")) if os.path.exists(kwargs["scnd"]) else {}

    data = []
    for sample in all_data["data"]:
        if sample["split"] != split:
            continue
        seq = lectin_data.loc[sample["prot_id"]].seqs[:cut]
        lec = esm_embeds[seq]
        gly = glycan_data.loc[sample["drug_id"]].IUPAC
        gly = gly.replace("NeuAc", "Neu5Ac")
        data.append((lec, gly, sample["label"]))

    df = pd.DataFrame(data, columns=['seq', 'glycan', 'match'])
    x = list(
        zip(df.seq.values.tolist(), [glycan_to_graph(k, libr=kwargs["gly_lib"]) for k in df.glycan.values.tolist()]))
    y = df.match.values.tolist()
    dataset = [Data(x=torch.tensor(k[0][1][0], dtype=torch.long), y=torch.tensor(k[1], dtype=torch.float),
                    edge_index=torch.tensor([k[0][1][1][0], k[0][1][1][1]], dtype=torch.long)) for k in
               list(zip(x, y))]
    for k in range(len(dataset)):
        dataset[k].train_idx = torch.tensor(x[k][0], dtype=torch.float)

    return DataLoader(dataset, batch_size=128, shuffle=True)


def inference_oracle(**kwargs):
    extract_data(**kwargs)

    data = pickle.load(open(kwargs["data"], "rb"))
    lectin_data = data["prots"]
    glycan_data = data["drugs"]
    esm_embeds = pickle.load(open(kwargs["scnd"], "rb")) if os.path.exists(kwargs["scnd"]) else {}

    if kwargs["saved"] == "":
        oracle = prep_model("LectinOracle", num_classes=1, trained=True)
    else:
        oracle = LectinOracle(input_size_prot=1280, input_size_glyco=len(lib) + 1, hidden_size=128, num_classes=1)
        oracle.load_state_dict(kwargs["saved"])
        oracle = oracle.cuda()

    missing, mse, mae, seen = 0, 0, 0, 0
    for i, sample in enumerate(data["data"]):
        print(f"\r{i}/{len(data['data'])}", end="")
        seq = lectin_data.loc[sample["prot_id"]].seqs[:cut]
        lec = esm_embeds[seq]
        gly = glycan_data.loc[sample["drug_id"]].IUPAC
        gly = gly.replace("NeuAc", "Neu5Ac")
        try:
            pred = get_multi_pred(seq, [gly], oracle, esm_embeds)[0]
            mse += (pred - sample["label"]) ** 2
            mae += abs(pred - sample["label"])
            seen += 1
        except Exception:
            missing += 1
    print("\nMissing:", missing, "|", len(data["data"]))
    print("MSE:", mse / seen)
    print("MAE:", mae / seen)


def inference_rindti(**kwargs):
    data = pickle.load(open(kwargs["data"], "rb"))
    lectin_data = data["prots"]
    glycan_data = data["drugs"]

    for version in os.listdir(kwargs["scnd"]):
        model = models[kwargs["model"]].load_from_checkpoint(os.path.join(kwargs["scnd"], version))

        for i, sample in enumerate(data["data"]):
            prot_id = sample["prot_id"]
            drug_id = sample["drug_id"]
            prot_data = lectin_data.loc[prot_id, "data"]
            drug_data = glycan_data.loc[drug_id, "data"]
            new_i = {
                "prot_count": float(lectin_data.loc[prot_id, "count"]),
                "drug_count": float(glycan_data.loc[drug_id, "count"]),
                "prot_id": prot_id,
                "drug_id": drug_id,
                "label": sample["label"],
            }
            new_i.update({"prot_" + k: v for (k, v) in prot_data.items()})
            new_i.update({"drug_" + k: v for (k, v) in drug_data.items()})
            two_graph_data = TwoGraphData(**new_i)


def training(**kwargs):
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)

    extract_data(**kwargs)

    all_data = pickle.load(open(kwargs["data"], "rb"))
    glycan_data = all_data["drugs"]
    glycans = set()
    data_min, data_max = float("inf"), float("-inf")

    for sample in all_data["data"]:
        gly = glycan_data.loc[sample["drug_id"]].IUPAC
        gly = gly.replace("NeuAc", "Neu5Ac")
        glycans.add(gly)
        data_min = min(data_min, sample["label"])
        data_max = max(data_max, sample["label"])

    gly_lib = get_lib(list(glycans)) + ['GlcNAcOS', 'GalOS', 'HexNAc']
    kwargs["gly_lib"] = gly_lib

    print("===== Fill dataloader & model ======")
    start = time.time()
    train_loader = read_data(kwargs["data"], "train", **kwargs)
    val_loader = read_data(kwargs["data"], "val", **kwargs)

    model = LectinOracle(input_size_prot=1280, input_size_glyco=len(lib) + 1, hidden_size=128,
                         num_classes=1, data_min=data_min, data_max=data_max)
    model.apply(init_weights)
    model.cuda()

    optimizer = SAM(model.parameters(), optim.Adam, lr=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 80)
    criterion = nn.MSELoss().cuda()
    print(f"\tFinished in {time.time() - start:0.5f} sec")

    print("========== Start training ==========")
    start = time.time()
    model = tm_own(model=model, dataloaders={"train": train_loader, "val": val_loader}, criterion=criterion,
                   optimizer=optimizer, scheduler=scheduler, num_epochs=100, patience=20, mode="regression")
    print(f"\tFinished in {time.time() - start:0.5f} sec")
    print("========= Finished training ========")

    # save model
    torch.save(model.state_dict(), "./oracle.pth")

    # test
    test_loader = read_data(kwargs["data"], "train", **kwargs)


def main(**kwargs):
    if kwargs["train"]:
        training(**kwargs)
    else:
        if kwargs["lo"]:
            inference_oracle(**kwargs)
        else:
            kwargs["model"] = "classification" if "classification" in kwargs["scnd"] else "regression"
            inference_rindti(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to pickled dataset")
    parser.add_argument("scnd", type=str, help="Path to pickled dictionary mapping AA-sequences to ESM-1b embeddings "
                                               "or path to pretrained RINDTI model")
    parser.add_argument("-s", "--saved", type=str, default="", dest="saved", help="Path to oracle model weights")
    parser.add_argument("-t", "--train", action='store_true', default=False, dest="train")
    parser.add_argument("--lo", action='store_true', default=False, dest="lo")
    args = parser.parse_args()

    main(**vars(args))
