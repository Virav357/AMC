"""Generate panel-ready parameter, architecture, and training figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def table_figure(rows: list[tuple[str, str]], title: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, max(4.5, len(rows) * 0.43)))
    axis.axis("off")
    table = axis.table(
        cellText=[[key, value] for key, value in rows],
        colLabels=["Parameter", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.38, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#1d4ed8")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eff6ff")
    axis.set_title(title, fontsize=18, weight="bold", pad=20)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-history", type=Path, default=Path("artifacts/amc.pt"))
    parser.add_argument("--link-results", type=Path, default=Path("artifacts/phase2-link-results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    link = json.loads(args.link_results.read_text())
    history = []
    if args.phase1_history.exists():
        try:
            import torch

            checkpoint = torch.load(args.phase1_history, map_location="cpu", weights_only=False)
            history = checkpoint.get("history", [])
        except Exception:
            history = []

    phase1_rows = [
        ("Dataset", "RadioML 2016.10a"),
        ("Input", "2 channels (I/Q) × 128 samples"),
        ("Classes", "11 modulation classes"),
        ("Split", "70% train / 15% validation / 15% test"),
        ("Normalization", "Per-example power normalization"),
        ("Architecture", "3 Conv1D blocks + BN + ReLU + pooling + dropout"),
        ("Trainable parameters", "144,331"),
        ("Loss", "Class-weighted cross-entropy"),
        ("Optimizer", "AdamW, weight decay 1×10⁻⁴"),
        ("Scheduler", "Cosine annealing"),
        ("Batch size / epochs", "1024 / 8 (reported run)"),
        ("Learning rate / seed", "3×10⁻⁴ / 42"),
        ("Test accuracy", "53.94%"),
        ("SNR range", "-20 dB to +18 dB"),
    ]
    phase2_rows = [
        ("Transmitter", "Random bits → BPSK, QPSK, 8-PSK, 16-QAM"),
        ("Frame size", "256 symbols per frame"),
        ("Channel", "Block Rayleigh fading + AWGN"),
        ("Impairments", "Carrier phase and frequency offset"),
        ("Receiver", "Channel equalization + modulation selection + demodulation"),
        ("SNR sweep", "-4 to +16 dB in 2 dB steps"),
        ("Frames per SNR", str(link["frames_per_snr"])),
        ("Random seed", str(link["seed"])),
        ("Metrics", "BER, SER, decision accuracy, packet success, throughput"),
        ("At 10 dB", "BER 4.69%, decision accuracy 99.5%"),
        ("At 16 dB", "BER 0.095%, decision accuracy 100%"),
    ]
    manifest = {
        "phase1": {
            "dataset": "RadioML 2016.10a",
            "input_shape": [2, 128],
            "classes": 11,
            "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "model": "3 Conv1D blocks, batch normalization, ReLU, max-pooling, adaptive average pooling, dropout",
            "trainable_parameters": 144331,
            "loss": "class-weighted cross-entropy",
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "batch_size": 1024,
            "epochs": 8,
            "learning_rate": 3e-4,
            "seed": 42,
            "test_accuracy": 0.5394,
            "best_validation_accuracy": 0.5402,
        },
        "phase2": {
            "transmitter_modulations": ["BPSK", "QPSK", "8PSK", "16QAM"],
            "frame_symbols": 256,
            "channel": ["block Rayleigh fading", "AWGN"],
            "impairments": ["phase offset", "frequency offset"],
            "snr_db": link["snr_db"],
            "frames_per_snr": link["frames_per_snr"],
            "seed": link["seed"],
            "metrics": ["BER", "SER", "modulation decision accuracy", "packet success", "throughput"],
            "best_operating_point": {
                "snr_db": 16,
                "ber": link["adaptive_ber"][-1],
                "ser": link["adaptive_ser"][-1],
                "decision_accuracy": link["modulation_decision_accuracy"][-1],
                "packet_success": link["adaptive_packet_success"][-1],
            },
        },
    }
    (args.output_dir / "results-manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results-manifest.json").write_text(json.dumps(manifest, indent=2))
    table_figure(phase1_rows, "Phase 1 — CNN Baseline Parameters and Results",
                  args.output_dir / "phase1-parameters-results.png")
    table_figure(phase2_rows, "Phase 2 — Communication-Link Parameters and Results",
                  args.output_dir / "phase2-parameters-results.png")

    if history:
        figure, axis = plt.subplots(figsize=(10, 5))
        epochs = [item["epoch"] for item in history]
        accuracy = [item["validation_accuracy"] for item in history]
        axis.plot(epochs, accuracy, "o-", color="#2563eb", linewidth=2.5)
        axis.set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(0, 1),
                 title="Phase 1 CNN Training History")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "phase1-training-history.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(15, 7))
    axis.axis("off")
    boxes = [
        (0.03, 0.42, "PHASE 2\nTransmitter", "#dbeafe", "Random bits\nBPSK / QPSK /\n8-PSK / 16-QAM"),
        (0.29, 0.42, "Channel", "#fee2e2", "Rayleigh fading\nAWGN\nPhase/frequency offset"),
        (0.55, 0.42, "Receiver front-end", "#fef3c7", "Pilot-assisted\nsynchronization\nEqualization"),
        (0.81, 0.42, "Adaptive decision", "#dcfce7", "AMC decision\nSelect demodulator\nRecover bits"),
    ]
    for x, y, title, color, body in boxes:
        axis.text(x, y, title, transform=axis.transAxes, ha="left", va="center",
                  fontsize=16, weight="bold", bbox=dict(boxstyle="round,pad=0.8",
                  facecolor=color, edgecolor="#475569", linewidth=1.5))
        axis.text(x + 0.005, y - 0.17, body, transform=axis.transAxes,
                  ha="left", va="top", fontsize=13, color="#334155")
    for x in (0.255, 0.515, 0.775):
        axis.annotate("", xy=(x + 0.035, 0.51), xytext=(x, 0.51),
                      xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#64748b"})
    axis.text(0.5, 0.12, "Measured outputs: BER • SER • packet success • throughput • modulation decision accuracy",
              transform=axis.transAxes, ha="center", fontsize=17, weight="bold", color="#1e3a8a")
    axis.set_title("Adaptive Communication Receiver Mechanism", fontsize=21, weight="bold", pad=24)
    figure.savefig(args.output_dir / "phase2-system-architecture.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
