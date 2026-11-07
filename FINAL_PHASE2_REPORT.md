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

## Final measured experiment

- Random seed: 42
- Frames per SNR: 200
- SNR range: -4 to +16 dB in 2 dB steps
- Modulations: BPSK, QPSK, 8-PSK, 16-QAM
- Channel: block Rayleigh fading and AWGN
- Impairments: carrier phase and frequency offset
- Synchronization: pilot-assisted simulation estimate

| SNR | BER | Decision accuracy | Packet success |
|---:|---:|---:|---:|
| 10 dB | 4.69% | 99.5% | 49.0% |
| 12 dB | 1.85% | 100.0% | 54.5% |
| 16 dB | 0.095% | 100.0% | 75.5% |

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
