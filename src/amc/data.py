"""Dataset utilities for RadioML-style automatic modulation classification data."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_radioml(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load a RadioML 2016.10a pickle file.

    The loader accepts the common ``{(modulation, snr): samples}`` format and
    returns tensors in ``(N, 2, 128)``-compatible NumPy form.
    """
    with Path(path).open("rb") as handle:
        raw: Any = pickle.load(handle, encoding="latin1")

    if not isinstance(raw, dict):
        raise ValueError("Expected a dictionary keyed by (modulation, SNR)")

    keys = sorted(raw, key=lambda item: (str(item[0]), int(item[1])))
    modulations = sorted({str(key[0]) for key in keys})
    class_to_id = {name: index for index, name in enumerate(modulations)}
    samples: list[np.ndarray] = []
    labels: list[int] = []
    snrs: list[int] = []

    for modulation, snr in keys:
        batch = np.asarray(raw[(modulation, snr)], dtype=np.float32)
        if batch.ndim != 3 or batch.shape[1] != 2:
            raise ValueError(
                f"Expected samples shaped (N, 2, T), got {batch.shape} for {(modulation, snr)}"
            )
        samples.append(batch)
        labels.extend([class_to_id[str(modulation)]] * len(batch))
        snrs.extend([int(snr)] * len(batch))

    x = np.concatenate(samples, axis=0)
    x /= np.sqrt(np.mean(np.square(x), axis=(1, 2), keepdims=True) + 1e-8)
    return x, np.asarray(labels, dtype=np.int64), modulations, np.asarray(snrs, dtype=np.int64)


def split_indices(
    labels: np.ndarray,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible stratified train/validation/test index splits."""
    indices = np.arange(len(labels))
    train_valid, test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    relative_validation = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_valid,
        test_size=relative_validation,
        random_state=seed,
        stratify=labels[train_valid],
    )
    return train, validation, test


class IQDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset for normalized I/Q samples and integer class labels."""

    def __init__(self, samples: np.ndarray, labels: np.ndarray) -> None:
        if len(samples) != len(labels):
            raise ValueError("Samples and labels must have equal lengths")
        self.samples = torch.from_numpy(samples).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index], self.labels[index]
