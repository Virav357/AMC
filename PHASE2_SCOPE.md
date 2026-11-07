# Phase 2 Project Scope — Intelligent Adaptive Communication Receiver

## Why the existing result is Phase 1

The existing RadioML experiment is a **modulation-classification baseline**:
it loads labeled I/Q examples, trains one CNN, and reports classification
accuracy. It is valuable groundwork, but it does not yet demonstrate a
complete communication system. Therefore, its 53.94% test accuracy is retained
as the Phase 1 reference result.

## Proposed Phase 2 contribution

Phase 2 will develop and evaluate an intelligent adaptive digital receiver
that uses modulation recognition to select the correct demodulation and
receiver configuration under changing channel conditions.

**Proposed title:** *Design and Performance Evaluation of an Intelligent
Adaptive Digital Communication Receiver using Automatic Modulation
Classification.*

## New communication-domain work

1. **Transmitter chain:** random bit generation, pulse/symbol formation, and
   BPSK, QPSK, 8-PSK, and 16-QAM modulation.
2. **Channel model:** AWGN, block Rayleigh fading, frequency offset, phase
   offset, and selectable SNR.
3. **Receiver chain:** pilot-assisted synchronization/channel estimation,
   equalization, automatic modulation decision, adaptive demodulation, and
   bit recovery.
4. **System metrics:** BER, SER, packet success rate, throughput, spectral
   efficiency, modulation decision accuracy, and computational latency.
5. **Communication comparison:** fixed-modulation receiver versus the
   proposed adaptive receiver across SNR and fading conditions.
6. **Ablation/robustness study:** channel impairment enabled one at a time and
   then jointly, so the panel can see which part of the receiver affects
   performance.

## Scientific hypothesis

An adaptive receiver that identifies the modulation and selects a compatible
demodulator should maintain lower BER and higher packet success than a
receiver configured for one fixed modulation when the transmitter modulation
changes. Under severe noise and fading, modulation decisions may be
uncertain; the evaluation will quantify this failure mode rather than hide it.

## Phase 2 acceptance criteria

- Reproducible transmitter-channel-receiver simulation.
- Real BER/SER curves, not only an image of classification accuracy.
- Comparison against a fixed-modulation baseline.
- Confusion matrix and modulation decision accuracy for the new receiver.
- Results for AWGN and at least one fading/offset condition.
- Clear limitations: synthetic channel versus future SDR over-the-air testing.

The initial software experiment uses pilot-assisted synchronization so that
the contribution can be separated into two measurable questions: whether the
receiver can identify the modulation and whether the adaptive demodulator
improves communication reliability. A future hardware experiment will replace
the simulated channel estimate with an SDR synchronization loop.
