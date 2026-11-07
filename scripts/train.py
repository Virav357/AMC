"""Train the Phase 2 AMC CNN on a RadioML pickle dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amc.data import IQDataset, load_radioml, split_indices
from amc.model import AMCNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="RadioML pickle path")
    parser.add_argument("--output", type=Path, default=Path("artifacts/amc.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for samples, labels in loader:
            predictions = model(samples.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += len(labels)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples, labels, class_names, _ = load_radioml(args.data)
    train_idx, validation_idx, _ = split_indices(labels, seed=args.seed)
    train_set = IQDataset(samples[train_idx], labels[train_idx])
    validation_set = IQDataset(samples[validation_idx], labels[validation_idx])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)

    classes = np.arange(len(class_names))
    weights = compute_class_weight("balanced", classes=classes, y=labels[train_idx])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    model = AMCNet(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy = -1.0
    history: list[dict[str, float]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_samples, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_samples.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        validation_accuracy = evaluate(model, validation_loader, device)
        history.append({"epoch": float(epoch), "validation_accuracy": validation_accuracy})
        print(f"epoch={epoch:03d} validation_accuracy={validation_accuracy:.4f}")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_names": class_names,
                    "input_shape": list(samples.shape[1:]),
                    "validation_accuracy": best_accuracy,
                    "history": history,
                },
                args.output,
            )

    print(json.dumps({"checkpoint": str(args.output), "best_validation_accuracy": best_accuracy}))


if __name__ == "__main__":
    main()
