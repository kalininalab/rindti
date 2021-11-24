from importlib import import_module

import yaml
from torch import FloatTensor, LongTensor

from ..layers.base_layer import BaseLayer


def get_module(init_args: dict, *args, **kwargs) -> BaseLayer:
    """Return a module from init_args path, instantied with init_args.

    Args:
        init_args (dict): A dictionary containing the path to the module and the init_args
        input_dim (int): The input dimension of the module
        output_dim (int): The output dimension of the module

    Returns:
        [BaseLayer]: The module instantiated with init_args
    """
    split_path = init_args.get("class_path", "").split(".")
    module = import_module(".".join(split_path[:-1]))
    mod_args = init_args.get("init_args", {})
    mod_args.update(kwargs)
    return getattr(module, split_path[-1])(*args, **mod_args)


def get_type(data: dict, key: str) -> str:
    """Check which type of data we have

    Args:
        data (dict): TwoGraphData or Data
        key (str): "x" or "prot_x" or "drug_x" usually

    Raises:
        ValueError: If not FloatTensor or LongTensor

    Returns:
        str: "label" for LongTensor, "onehot" for FloatTensor
    """
    feat = data.get(key)
    if isinstance(feat, LongTensor):
        return "label"
    if isinstance(feat, FloatTensor):
        return "onehot"
    if feat is None:
        return "none"
    raise ValueError("Unknown data type {}".format(type(data[key])))


def read_config(filename: str) -> dict:
    """Read in yaml config for training"""
    with open(filename, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def remove_arg_prefix(prefix: str, kwargs: dict) -> dict:
    """Removes the prefix from all the init_args
    Args:
        prefix (str): prefix to remove (`drug_`, `prot_` or `mlp_` usually)
        kwargs (dict): dict of arguments
    Returns:
        dict: Sub-dict of arguments
    """
    new_kwargs = {}
    prefix_len = len(prefix)
    for key, value in kwargs.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            if new_key == "x_batch":
                new_key = "batch"
            new_kwargs[new_key] = value
    return new_kwargs


def add_arg_prefix(prefix: str, kwargs: dict) -> dict:
    """Adds the prefix to all the init_args. Removes None values and "index_mapping"
    Args:
        prefix (str): prefix to add (`drug_`, `prot_` or `mlp_` usually)
        kwargs (dict): dict of arguments
    Returns:
        dict: Sub-dict of arguments
    """
    return {prefix + k: v for (k, v) in kwargs.items() if k != "index_mapping" and v is not None}
