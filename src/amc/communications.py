"""Communication-link primitives for the Phase 2 adaptive receiver study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODULATIONS = ("BPSK", "QPSK", "8PSK", "16QAM")


@dataclass(frozen=True)
class LinkResult:
    modulation: str
    bits: np.ndarray
    received: np.ndarray
    channel_gain: complex
    frequency_offset: float
    phase_offset: float


def _constellation(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "BPSK":
        bits = np.array([[0], [1]], dtype=np.int8)
        points = np.array([1, -1], dtype=np.complex128)
    elif name == "QPSK":
        bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int8)
        points = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    elif name == "8PSK":
        bits = ((np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1).astype(np.int8)
        points = np.exp(1j * (np.arange(8) * 2 * np.pi / 8))
    elif name == "16QAM":
        pairs = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int8)
        levels = np.array([-3, -1, 3, 1], dtype=float)
        bits = np.array([np.r_[pairs[i], pairs[q]] for i in range(4) for q in range(4)])
        points = np.array([levels[i] + 1j * levels[q] for i in range(4) for q in range(4)])
        points /= np.sqrt(np.mean(np.abs(points) ** 2))
    else:
        raise ValueError(f"Unsupported modulation: {name}")
    return points, bits


def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    points, labels = _constellation(modulation)
    bits = np.asarray(bits, dtype=np.int8)
    bits_per_symbol = labels.shape[1]
    if bits.ndim != 1 or len(bits) % bits_per_symbol:
        raise ValueError("Bit array length must be divisible by bits per symbol")
    groups = bits.reshape(-1, bits_per_symbol)
    indices = np.array([np.flatnonzero(np.all(labels == group, axis=1))[0] for group in groups])
    return points[indices]


def demodulate(symbols: np.ndarray, modulation: str) -> np.ndarray:
    points, labels = _constellation(modulation)
    indices = np.argmin(np.abs(np.asarray(symbols)[:, None] - points[None, :]), axis=1)
    return labels[indices].reshape(-1)


def transmit(
    bits: np.ndarray,
    modulation: str,
    snr_db: float,
    rng: np.random.Generator,
    rayleigh: bool = True,
    frequency_offset: float = 0.0,
    phase_offset: float = 0.0,
) -> LinkResult:
    """Transmit a frame through a block-fading, noisy, offset-impaired channel."""
    symbols = modulate(bits, modulation)
    channel_gain = (rng.normal() + 1j * rng.normal()) / np.sqrt(2) if rayleigh else 1 + 0j
    sample_index = np.arange(len(symbols))
    rotated = symbols * channel_gain * np.exp(
        1j * (phase_offset + 2 * np.pi * frequency_offset * sample_index)
    )
    signal_power = np.mean(np.abs(rotated) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=len(symbols)) + 1j * rng.normal(size=len(symbols))
    )
    return LinkResult(
        modulation, bits, rotated + noise, channel_gain, frequency_offset, phase_offset
    )


def estimate_modulation(received: np.ndarray, candidates: tuple[str, ...] = MODULATIONS) -> str:
    """Choose the constellation with minimum phase/scale-invariant error."""
    signal = np.asarray(received)
    signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2) + 1e-12)
    phase_grid = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    errors: dict[str, float] = {}
    for modulation in candidates:
        points, _ = _constellation(modulation)
        best = np.inf
        for phase in phase_grid:
            rotated = signal * np.exp(-1j * phase)
            distance = np.min(np.abs(rotated[:, None] - points[None, :]) ** 2, axis=1)
            best = min(best, float(np.mean(distance)))
        errors[modulation] = best + 0.03 * np.log2(len(points))
    return min(errors, key=errors.get)


def receive(result: LinkResult, modulation: str) -> tuple[np.ndarray, np.ndarray]:
    """Equalize a block-fading frame and return recovered bits and symbols."""
    sample_index = np.arange(len(result.received))
    equalized = result.received / (result.channel_gain + 1e-12)
    equalized *= np.exp(
        -1j * (result.phase_offset + 2 * np.pi * result.frequency_offset * sample_index)
    )
    return demodulate(equalized, modulation), equalized
