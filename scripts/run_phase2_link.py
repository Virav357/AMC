"""Run the Phase 2 adaptive communication-link experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amc.communications import MODULATIONS, receive, estimate_modulation, transmit


def bit_errors(decoded: np.ndarray, truth: np.ndarray) -> int:
    """Count mismatches and account for a wrong demodulator bit rate."""
    length = max(len(decoded), len(truth))
    decoded_padded = np.pad(decoded, (0, length - len(decoded)), constant_values=-1)
    truth_padded = np.pad(truth, (0, length - len(truth)), constant_values=-2)
    return int(np.count_nonzero(decoded_padded != truth_padded))


def symbol_errors(decoded: np.ndarray, truth: np.ndarray, bits_per_symbol: int) -> int:
    """Count symbol errors after grouping recovered and transmitted bits."""
    truth_groups = truth[: len(truth) - len(truth) % bits_per_symbol].reshape(-1, bits_per_symbol)
    decoded_groups = decoded[: len(decoded) - len(decoded) % bits_per_symbol].reshape(
        -1, bits_per_symbol
    )
    count = min(len(truth_groups), len(decoded_groups))
    errors = int(np.count_nonzero(np.any(decoded_groups[:count] != truth_groups[:count], axis=1)))
    return errors + abs(len(truth_groups) - len(decoded_groups))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2-link-results.png"))
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    snrs = np.arange(-4, 17, 2)
    adaptive_ber: list[float] = []
    fixed_ber: list[float] = []
    adaptive_ser: list[float] = []
    fixed_ser: list[float] = []
    decision_accuracy: list[float] = []
    packet_success: list[float] = []
    adaptive_throughput: list[float] = []

    for snr in snrs:
        adaptive_errors = fixed_errors = total_bits = correct_decisions = successful_packets = 0
        adaptive_symbol_errors = fixed_symbol_errors = total_symbols = 0
        for _ in range(args.frames):
            modulation = str(rng.choice(MODULATIONS))
            bits_per_symbol = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4}[modulation]
            bits = rng.integers(0, 2, size=256 * bits_per_symbol, dtype=np.int8)
            frame = transmit(
                bits,
                modulation,
                float(snr),
                rng,
                rayleigh=True,
                frequency_offset=rng.uniform(-0.0005, 0.0005),
                phase_offset=rng.uniform(-np.pi, np.pi),
            )
            sample_index = np.arange(len(frame.received))
            synchronized = frame.received / (frame.channel_gain + 1e-12)
            synchronized *= np.exp(
                -1j * (frame.phase_offset + 2 * np.pi * frame.frequency_offset * sample_index)
            )
            predicted = estimate_modulation(synchronized)
            adaptive_bits, _ = receive(frame, predicted)
            fixed_bits, _ = receive(frame, "QPSK" if len(bits) % 2 == 0 else "BPSK")
            adaptive_errors += min(bit_errors(adaptive_bits, bits), len(bits))
            fixed_errors += min(bit_errors(fixed_bits, bits), len(bits))
            total_bits += len(bits)
            adaptive_symbol_errors += min(
                symbol_errors(adaptive_bits, bits, bits_per_symbol), len(frame.received)
            )
            fixed_symbol_errors += min(symbol_errors(fixed_bits, bits, 2), len(frame.received))
            total_symbols += len(frame.received)
            correct_decisions += int(predicted == modulation)
            successful_packets += int(np.array_equal(adaptive_bits, bits))
        adaptive_ber.append(adaptive_errors / total_bits)
        fixed_ber.append(fixed_errors / total_bits)
        adaptive_ser.append(adaptive_symbol_errors / max(total_symbols, 1))
        fixed_ser.append(fixed_symbol_errors / max(total_symbols, 1))
        decision_accuracy.append(correct_decisions / args.frames)
        packet_success.append(successful_packets / args.frames)
        adaptive_throughput.append(
            float(np.mean([{"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4}[m]
                          for m in MODULATIONS]) * packet_success[-1])
        )

    result = {
        "seed": args.seed,
        "frames_per_snr": args.frames,
        "snr_db": snrs.tolist(),
        "adaptive_ber": adaptive_ber,
        "fixed_qpsk_ber": fixed_ber,
        "adaptive_ser": adaptive_ser,
        "fixed_qpsk_ser": fixed_ser,
        "modulation_decision_accuracy": decision_accuracy,
        "adaptive_packet_success": packet_success,
        "adaptive_throughput_bits_per_symbol": adaptive_throughput,
        "modulations": list(MODULATIONS),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].semilogy(snrs, adaptive_ber, "o-", label="Adaptive receiver")
    axes[0, 0].semilogy(snrs, fixed_ber, "s--", label="Fixed QPSK receiver")
    axes[0, 0].set(xlabel="SNR (dB)", ylabel="BER", title="Bit error rate")
    axes[0, 0].grid(alpha=0.25, which="both")
    axes[0, 0].legend()
    axes[0, 1].semilogy(snrs, adaptive_ser, "o-", label="Adaptive receiver")
    axes[0, 1].semilogy(snrs, fixed_ser, "s--", label="Fixed QPSK receiver")
    axes[0, 1].set(xlabel="SNR (dB)", ylabel="SER", title="Symbol error rate")
    axes[0, 1].grid(alpha=0.25, which="both")
    axes[0, 1].legend()
    axes[1, 0].plot(snrs, decision_accuracy, "o-", label="Modulation decision accuracy")
    axes[1, 0].plot(snrs, packet_success, "s--", label="Adaptive packet success")
    axes[1, 0].set(xlabel="SNR (dB)", ylabel="Rate", title="Adaptive receiver operation", ylim=(0, 1.05))
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()
    axes[1, 1].plot(snrs, adaptive_throughput, "D-", color="#16a34a")
    axes[1, 1].set(xlabel="SNR (dB)", ylabel="bits/symbol", title="Packet-success weighted throughput")
    axes[1, 1].grid(alpha=0.25)
    figure.suptitle("Phase 2: Intelligent Adaptive Digital Communication Receiver", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(args.output, dpi=180)
    print(json.dumps(result, indent=2))
    print(f"results_figure={args.output}")


if __name__ == "__main__":
    main()
