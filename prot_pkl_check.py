import pickle
import pandas as pd
 
prots = pickle.load(open("datasets/glass/results/prot_data/wdln_1479ac8c.pkl", "rb"))
inter = pd.read_csv("datasets/glass/results/split_data/rarn_7cbd3f55.tsv", sep="\t")
 
print(type(prots))
print("prots shape:", getattr(prots, "shape", None))
print("prots columns:", getattr(prots, "columns", None))
print("first few index values:", list(prots.index[:5]))
 
sub = prots[prots.index.isin(inter["Target_ID"].unique())]
print("filtered prots shape:", sub.shape)
print("filtered index sample:", list(sub.index[:5]))
print("is empty?", sub.empty)
