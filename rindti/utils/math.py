import pandas as pd
from torch.utils.data import random_split


def split_random(dataset, train_frac: float = 0.8):
    """Randomly split dataset"""
    tot = len(dataset)
    train = int(tot * train_frac)
    val = int(tot * (1 - train_frac))
    return random_split(dataset, [train, val])


def minmax_normalise(s: pd.Series) -> pd.Series:
    """MinMax normalisation of a pandas series"""
    return (s - s.min()) / (s.max() - s.min())


def to_prob(s: pd.Series) -> pd.Series:
    """Convert to probabilities"""
    return s / s.sum()
