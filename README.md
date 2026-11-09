# Intelligent Adaptive Communication Receiver — Phase 2

This repository contains the Phase-II adaptive digital communication
receiver developed for the B.Tech. ECE project. The system generates
information frames, applies channel impairments, estimates the modulation,
selects a demodulator, and measures recovered-link performance.

## Phase-II contribution

- BPSK, QPSK, 8-PSK, and 16-QAM modulation and demodulation.
- Block-Rayleigh fading, AWGN, phase offset, and frequency offset.
- Pilot-assisted simulated correction and channel equalization.
- Penalized constellation-distance modulation estimation.
- Adaptive demodulation with BER, SER, decision accuracy, packet success,
  and effective throughput.
- Fixed-QPSK comparison under the same experiment settings.

Run the communication experiment with:

```bash
python scripts/run_phase2_link.py --frames 200 \
  --output artifacts/phase2-link-results.png
```

The complete numerical record is stored in
`artifacts/phase2-link-results.json`. The report uses separate Phase-II
figures for BER, SER, decision accuracy, packet success, throughput, and
adaptive-versus-fixed comparisons.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The report package contains `MainFile.tex`, the SVNIT style dependency,
`MainFile.pdf`, all individual result figures, the JSON result record, the
communication module, experiment driver, and tests.
