import numpy as np

from src.amc.communications import demodulate, estimate_modulation, modulate


def test_modulation_round_trip_without_channel():
    rng = np.random.default_rng(42)
    for modulation, bits_per_symbol in (("BPSK", 1), ("QPSK", 2), ("8PSK", 3), ("16QAM", 4)):
        bits = rng.integers(0, 2, size=64 * bits_per_symbol, dtype=np.int8)
        assert np.array_equal(demodulate(modulate(bits, modulation), modulation), bits)


def test_constellation_estimator_identifies_clean_frame():
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=512, dtype=np.int8)
    for modulation in ("BPSK", "QPSK", "8PSK", "16QAM"):
        symbols = modulate(bits[: len(bits) - len(bits) % {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4}[modulation]], modulation)
        assert estimate_modulation(symbols) == modulation
