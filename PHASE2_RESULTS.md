# Phase 1 Baseline Results — Automatic Modulation Classification

## Baseline objective

The Phase 2 objective was to design, train, and evaluate a deep-learning
receiver that identifies the modulation format of a received signal directly
from its complex baseband I/Q samples. This supports intelligent receivers,
spectrum monitoring, and cognitive-radio applications where the transmitter
protocol is not known in advance.

## Dataset and experiment

- **Dataset:** RadioML 2016.10a, containing labeled I/Q examples for 11
  modulation classes at SNR values from -20 dB to +18 dB.
- **Preprocessing:** per-example power normalization, integer label encoding,
  and deterministic stratified splitting.
- **Split:** 70% training, 15% validation, and 15% held-out testing.
- **Model:** three 1-D convolutional blocks with batch normalization, ReLU,
  max-pooling, adaptive average pooling, and dropout.
- **Optimization:** AdamW, class-weighted cross-entropy, cosine learning-rate
  scheduling, batch size 1024, and 8 training epochs.
- **Reproducibility:** random seed 42; the checkpoint records model weights,
  class ordering, input shape, and validation history.

## Final measured results

| Metric | Result |
|---|---:|
| Held-out test accuracy | **53.94%** |
| Best validation accuracy | **54.02%** |
| Test examples | **33,000** |
| Lowest-SNR accuracy (-20 dB) | **9.24%** |
| Highest measured SNR accuracy (+12 dB) | **81.68%** |

The result is technically meaningful because modulation classes overlap heavily
at low SNR. The strong increase in accuracy with SNR confirms that the model
has learned signal structure rather than producing a random class assignment.
The confusion matrix shows remaining ambiguity between modulation types with
similar constellation or waveform characteristics.

## Baseline deliverables

1. `src/amc/data.py`: RadioML loader, normalization, stratified split, and
   PyTorch dataset.
2. `src/amc/model.py`: compact CNN architecture for two-channel I/Q input.
3. `scripts/train.py`: reproducible training and checkpoint generation.
4. `scripts/evaluate.py`: test accuracy, confusion matrix, and SNR-wise
   analysis.
5. `artifacts/phase2-results.png`: final result figure for presentation.
6. `artifacts/amc.pt`: trained checkpoint generated locally; it is excluded
   from Git because of its binary size.

## Interpretation for evaluation

This is the Phase 1 classification baseline, not the final Phase 2
communication-system result. Its principal limitations are the synthetic
dataset, finite training budget, and absence of a transmitter/channel/
receiver BER experiment. Phase 2 addresses those limitations through the
adaptive communication receiver defined in `PHASE2_SCOPE.md`.

## Phase 2 communication-link result

The new Phase 2 experiment is implemented in `scripts/run_phase2_link.py`.
It generates mixed BPSK, QPSK, 8-PSK, and 16-QAM frames, applies block
Rayleigh fading, AWGN, carrier phase offset, and frequency offset, then uses
pilot-assisted synchronization and adaptive demodulation.

With 200 frames per SNR and seed 42, the adaptive receiver reached:

| SNR | BER | Modulation decision accuracy | Packet success |
|---:|---:|---:|---:|
| 10 dB | 4.69% | 99.5% | 49.0% |
| 12 dB | 1.85% | 100.0% | 54.5% |
| 16 dB | 0.095% | 100.0% | 75.5% |

At low SNR, the receiver correctly exposes the main communication challenge:
modulation ambiguity causes demodulator mismatch and packet failure. This is
why Phase 2 reports BER and packet success in addition to AMC accuracy. The
generated figure is `artifacts/phase2-link-results.png`, with machine-readable
values in `artifacts/phase2-link-results.json`.
