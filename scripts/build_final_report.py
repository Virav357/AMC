"""Build the final Phase 1 versus Phase 2 panel figure from experiment JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--link-results", type=Path, default=Path("artifacts/phase2-link-results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final-phase1-phase2-comparison.png"))
    args = parser.parse_args()
    data = json.loads(args.link_results.read_text())
    snr = np.asarray(data["snr_db"])
    phase1_accuracy = 0.5394
    phase1_low_snr = 0.0924
    phase1_high_snr = 0.8168

    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax = axes[0, 0]
    ax.bar(["Phase 1\nAMC baseline", "Phase 2\nadaptive receiver"],
           [phase1_accuracy, data["modulation_decision_accuracy"][-1]],
           color=["#64748b", "#2563eb"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy / rate")
    ax.set_title("Project-level contribution")
    ax.text(0, phase1_accuracy + .03, "53.94%", ha="center", weight="bold")
    ax.text(1, data["modulation_decision_accuracy"][-1] + .03,
            f"{data['modulation_decision_accuracy'][-1]:.1%}", ha="center", weight="bold")
    ax.grid(axis="y", alpha=.25)

    ax = axes[0, 1]
    ax.semilogy(snr, data["adaptive_ber"], "o-", label="Phase 2 adaptive")
    ax.semilogy(snr, data["fixed_qpsk_ber"], "s--", label="Fixed QPSK")
    ax.set(title="Communication reliability", xlabel="SNR (dB)", ylabel="BER")
    ax.grid(alpha=.25, which="both")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(snr, data["modulation_decision_accuracy"], "o-", label="Modulation decision")
    ax.plot(snr, data["adaptive_packet_success"], "s--", label="Packet success")
    ax.set(title="Adaptive receiver operation", xlabel="SNR (dB)", ylabel="Rate", ylim=(0, 1.05))
    ax.grid(alpha=.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(snr, data["adaptive_throughput_bits_per_symbol"], "D-", color="#16a34a")
    ax.set(title="Effective throughput", xlabel="SNR (dB)",
           ylabel="bits/symbol (packet-success weighted)")
    ax.grid(alpha=.25)

    figure.suptitle("SVNIT B.Tech ECE — Phase 1 vs Phase 2 Results", fontsize=24, weight="bold")
    figure.text(
        .5, .01,
        "Phase 1: classification accuracy on RadioML. Phase 2: complete "
        "transmitter/channel/receiver simulation with BER and packet metrics.",
        ha="center", fontsize=12,
    )
    figure.tight_layout(rect=(0, .04, 1, .95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
