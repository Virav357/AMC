# Final Phase 2 Evaluation Report

## Project title

**Design and Performance Evaluation of an Intelligent Adaptive Digital
Communication Receiver using Automatic Modulation Classification**

## What changed from Phase 1?

| Phase 1: AMC baseline | Phase 2: communication-system contribution |
|---|---|
| Classifies an I/Q window | Completes a transmitter-channel-receiver link |
| Reports classification accuracy | Reports BER, SER, packet success, throughput, and decision accuracy |
| Uses a fixed offline dataset | Generates changing digital communication frames |
| No demodulator or bit recovery | Selects a demodulator and recovers transmitted bits |
| SNR-wise classification analysis | AWGN + Rayleigh fading + phase/frequency-offset robustness |

Phase 1 remains the reference baseline: 53.94% held-out RadioML classification
accuracy. Phase 2 is not merely a different neural-network architecture; it
uses modulation recognition as a receiver-control function in an adaptive
communication system.

## Phase 1 parameter and training record

The Phase 1 model receives a `(2, 128)` I/Q tensor and has **144,331
trainable parameters**. It contains Conv1D layers with 64, 128, and 256
filters, batch normalization, ReLU, max-pooling, adaptive average pooling,
and a dropout-regularized 11-class output layer. The reported training run
uses 70/15/15 stratified splitting, per-example power normalization,
class-weighted cross-entropy, AdamW, weight decay `1e-4`, cosine annealing,
learning rate `3e-4`, batch size `1024`, seed `42`, and `8` epochs.

The best validation accuracy is **54.02%** and the held-out test accuracy is
**53.94%**. The complete parameter table is
`artifacts/phase1-parameters-results.png`; the validation curve is
`artifacts/phase1-training-history.png`.

## Phase 2 processing chain

```text
Random bits
   -> BPSK / QPSK / 8-PSK / 16-QAM transmitter
   -> block Rayleigh fading + AWGN + phase/frequency offset
   -> pilot-assisted synchronization and channel equalization
   -> modulation decision
   -> adaptive demodulator
   -> recovered bits
   -> BER / SER / packet success / throughput
```

### Core mechanism

1. A random information frame is generated and mapped to one of four
   modulation constellations.
2. The complex symbols pass through a block-Rayleigh channel. AWGN is scaled
   from the selected SNR, while random carrier phase and frequency offsets
   model receiver synchronization stress.
3. The receiver uses the known channel/pilot estimate to equalize and
   synchronize the frame. This isolates the new adaptive-modulation problem
   from an unimplemented hardware timing loop.
4. A phase/scale-invariant constellation-error detector selects the most
   likely modulation. The selected demodulator converts symbols back to bits.
5. Recovered bits are compared with the transmitted bits. BER, SER, packet
   success, throughput, and modulation-decision accuracy are stored for every
   SNR.

The Phase 2 parameter table is
`artifacts/phase2-parameters-results.png`; the block diagram is
`artifacts/phase2-system-architecture.png`.

## Final measured experiment

- Random seed: 42
- Frames per SNR: 200
- SNR range: -4 to +16 dB in 2 dB steps
- Modulations: BPSK, QPSK, 8-PSK, 16-QAM
- Channel: block Rayleigh fading and AWGN
- Impairments: carrier phase and frequency offset
- Synchronization: pilot-assisted simulation estimate

| SNR | BER | SER | Decision accuracy | Packet success | Throughput |
|---:|---:|---:|---:|---:|---:|
| 10 dB | 4.69% | 6.94% | 99.5% | 49.0% | 1.225 bit/symbol |
| 12 dB | 1.85% | 3.05% | 100.0% | 54.5% | 1.363 bit/symbol |
| 16 dB | 0.095% | 0.172% | 100.0% | 75.5% | 1.888 bit/symbol |

## Interpretation

At high SNR, the receiver identifies the modulation reliably and the adaptive
demodulator produces low BER. At low SNR, modulation ambiguity causes
demodulator mismatch and packet failure. This failure is an important result:
it identifies the operating region where confidence-based fallback, coding,
or retransmission should be added.

The Phase 2 contribution is therefore a communication-system design and
evaluation, not only a classifier benchmark. The result is a software
simulation; SDR over-the-air testing is the appropriate next engineering
validation step.

## Final figures

- `artifacts/final-phase1-phase2-comparison.png`: panel comparison.
- `artifacts/phase2-link-results.png`: BER and adaptive operation plots.
- `artifacts/phase2-results.png`: Phase 1 AMC baseline confusion/SNR result.
- `artifacts/phase1-parameters-results.png`: Phase 1 parameters and measured result.
- `artifacts/phase1-training-history.png`: Phase 1 validation history.
- `artifacts/phase2-parameters-results.png`: Phase 2 link parameters and metrics.
- `artifacts/phase2-system-architecture.png`: Phase 2 communication mechanism.
- `artifacts/results-manifest.json`: machine-readable parameter/result record.
