# Intelligent Adaptive Communication Receiver — Phase 2

This repository contains the Phase 2 direction for the B.Tech ECE project:
an intelligent adaptive digital communication receiver. The earlier CNN-only
RadioML experiment is retained as the **Phase 1 baseline**. Phase 2 extends
that baseline into a transmitter-channel-receiver system and evaluates actual
communication quality.

Read [`PHASE2_SCOPE.md`](PHASE2_SCOPE.md) for the corrected project scope.

## Phase 1 baseline

- Load the RadioML 2016.10a pickle format and normalize each I/Q example.
- Create deterministic, stratified train/validation/test splits.
- Train a compact 1-D CNN with class-weighted cross-entropy, AdamW, and cosine
  learning-rate scheduling.
- Report overall accuracy, confusion matrices, and accuracy at each SNR.

## Phase 2 contribution

- Generate modulated bit streams at the transmitter.
- Pass signals through AWGN, fading, frequency-offset, and phase-offset
  channels.
- Use modulation recognition to select an adaptive demodulator.
- Measure BER, SER, packet success, throughput, spectral efficiency, and
  decision accuracy.
- Compare the adaptive receiver with a fixed-modulation receiver.

Run the new communication-domain experiment with:

```bash
python scripts/run_phase2_link.py --frames 200 \
  --output artifacts/phase2-link-results.png
```

This produces BER, packet-success, and modulation-decision results for the
adaptive receiver versus a fixed-QPSK receiver under block Rayleigh fading,
AWGN, carrier phase offset, and frequency offset with pilot-assisted
synchronization.

For the complete panel comparison, run:

```bash
python scripts/build_final_report.py
```

The final comparison figure is
[`artifacts/final-phase1-phase2-comparison.png`](artifacts/final-phase1-phase2-comparison.png),
with the written interpretation in [`FINAL_PHASE2_REPORT.md`](FINAL_PHASE2_REPORT.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download RadioML 2016.10a separately and place the pickle file outside the
repository if it is not already available locally. Then run:

```bash
python scripts/train.py --data /path/to/RML2016.10a_dict.pkl \
  --output artifacts/amc.pt
python scripts/evaluate.py --data /path/to/RML2016.10a_dict.pkl \
  --checkpoint artifacts/amc.pt --output artifacts/phase2-results.png
```

The default split is 70% training, 15% validation, and 15% testing. The
checkpoint stores the class-name ordering and input shape needed for
evaluation. Use `--epochs`, `--batch-size`, and `--learning-rate` to reproduce
experiments or compare model variants.

## Phase 1 deliverables retained for comparison

1. Dataset preparation and reproducible experiment configuration.
2. CNN baseline implementation for raw I/Q signals.
3. Test accuracy, confusion matrix, and SNR-wise analysis.
4. A baseline for the communication-system experiments in Phase 2.

The LaTeX report template remains in the repository and can be updated with
the results generated under `artifacts/`.

The Phase 2 workflow is summarized visually in
[`artifacts/phase2-results-overview.svg`](artifacts/phase2-results-overview.svg).

## Existing Phase 1 measured result

The completed baseline RadioML experiment produced 53.94% accuracy on the held-out
test set after 8 CNN training epochs. Accuracy increased from 9.24% at -20 dB
SNR to 81.68% at +12 dB SNR, demonstrating the expected effect of channel
noise. The complete figure is
[`artifacts/phase2-results.png`](artifacts/phase2-results.png). It is not the
final Phase 2 result; the communication-link evaluation is the new target.