from copy import deepcopy
from math import ceil

import pytest
import torch

from rindti.data import DataCorruptor, corrupt_features, mask_features

LOW = 1
HIGH = 10
N_NODES = 15

original_features = torch.randint(LOW, HIGH, (N_NODES,), dtype=torch.long)
original_data = {k: deepcopy(original_features) for k in ["x", "prot_x", "drug_x"]}


def _assert_corruption(orig_features, corrupt_features, frac):
    expected_number = ceil(N_NODES * frac)
    assert corrupt_features[corrupt_features != orig_features].size(0) <= expected_number


def _assert_mask(masked_features, frac):
    expected_number = ceil(N_NODES * frac)
    assert torch.count_nonzero(masked_features) == (N_NODES - expected_number)


@pytest.mark.parametrize("func", [mask_features, corrupt_features])
@pytest.mark.parametrize("frac", [-1, 3])
def test_assert_fail(func, frac):
    features = deepcopy(original_features)
    with pytest.raises(AssertionError):
        func(features, frac)


@pytest.mark.parametrize("d", [{"x": 0.2}, {"prot_x": 0.4, "drug_x": 0.7}])
@pytest.mark.parametrize("type", ["mask", "corrupt"])
def test_data_corruptor(d, type):
    dc = DataCorruptor(d, type)
    corrupted_data = deepcopy(original_data)
    corrupted_data = dc(corrupted_data)
    for k, v in d.items():
        if type == "corrupt":
            _assert_corruption(corrupted_data[k], original_features, v)
        elif type == "mask":
            _assert_mask(corrupted_data[k], v)
