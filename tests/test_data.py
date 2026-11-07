import numpy as np
import pytest

from src.amc.data import IQDataset, split_indices


def test_split_indices_are_reproducible_and_disjoint():
    labels = np.repeat(np.arange(4), 20)
    first = split_indices(labels, seed=7)
    second = split_indices(labels, seed=7)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert not (set(first[0]) & set(first[1]))
    assert not (set(first[0]) & set(first[2]))
    assert not (set(first[1]) & set(first[2]))


def test_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        IQDataset(np.zeros((2, 2, 8), dtype=np.float32), np.zeros(1, dtype=np.int64))
