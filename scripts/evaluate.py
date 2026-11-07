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
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2-results.png"))
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
    snr_values = sorted(np.unique(snrs[test_idx]))
    snr_accuracies = [
        accuracy_score(truth[snrs[test_idx] == snr], np.asarray(predictions)[snrs[test_idx] == snr])
        for snr in snr_values
    ]
    history = checkpoint.get("history", [])
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.32, wspace=0.2)
    ax_summary = figure.add_subplot(grid[0, :])
    ax_summary.axis("off")
    ax_summary.text(0.01, 0.72, "AUTOMATIC MODULATION CLASSIFICATION", fontsize=24, weight="bold")
    ax_summary.text(0.01, 0.46, "Phase 2 CNN evaluation on RadioML 2016.10a", fontsize=18, color="#334155")
    ax_summary.text(0.01, 0.18, f"Test accuracy: {accuracy_score(truth, predictions):.2%}", fontsize=28,
                    weight="bold", color="#1d4ed8")
    ax_summary.text(0.42, 0.18, f"Best validation accuracy: {checkpoint['validation_accuracy']:.2%}",
                    fontsize=22, color="#166534")
    ax_summary.text(0.78, 0.18, f"Test samples: {len(test_idx):,}", fontsize=20, color="#92400e")
    ax_cm = figure.add_subplot(grid[1, 0])
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=ax_cm, xticks_rotation="vertical", cmap="Blues", values_format="d", colorbar=False
    )
    ax_cm.set_title("Confusion matrix (held-out test set)")
    ax_snr = figure.add_subplot(grid[1, 1])
    ax_snr.plot(snr_values, snr_accuracies, marker="o", linewidth=2.5, color="#2563eb")
    ax_snr.set_title("Accuracy versus SNR")
    ax_snr.set_xlabel("SNR (dB)")
    ax_snr.set_ylabel("Accuracy")
    ax_snr.set_ylim(0, 1)
    ax_snr.grid(alpha=0.25)
    if history:
        ax_snr.text(0.02, 0.03, f"Training epochs: {len(history)}", transform=ax_snr.transAxes,
                    fontsize=11, color="#475569")
    figure.suptitle("SVNIT B.Tech ECE Project — Phase 2 Results", fontsize=30, weight="bold", y=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"results_figure={args.output}")
    for snr in sorted(np.unique(snrs[test_idx])):
        mask = snrs[test_idx] == snr
        print(f"snr={snr:+d} accuracy={accuracy_score(truth[mask], np.asarray(predictions)[mask]):.4f}")


if __name__ == "__main__":
    main()
