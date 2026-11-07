"""Evaluate a trained AMC checkpoint and save a confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amc.data import IQDataset, load_radioml, split_indices
from amc.model import AMCNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/confusion-matrix.png"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples, labels, class_names, snrs = load_radioml(args.data)
    _, _, test_idx = split_indices(labels, seed=args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = AMCNet(len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(IQDataset(samples[test_idx], labels[test_idx]), batch_size=args.batch_size)
    predictions: list[int] = []
    with torch.no_grad():
        for batch, _ in loader:
            predictions.extend(model(batch).argmax(dim=1).tolist())

    truth = labels[test_idx]
    print(f"test_accuracy={accuracy_score(truth, predictions):.4f}")
    matrix = confusion_matrix(truth, predictions, labels=np.arange(len(class_names)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        xticks_rotation="vertical", cmap="Blues", values_format="d"
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"confusion_matrix={args.output}")
    for snr in sorted(np.unique(snrs[test_idx])):
        mask = snrs[test_idx] == snr
        print(f"snr={snr:+d} accuracy={accuracy_score(truth[mask], np.asarray(predictions)[mask]):.4f}")


if __name__ == "__main__":
    main()
