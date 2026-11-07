# Automatic Modulation Classification — Phase 2

This repository now contains the Phase 2 implementation for the B.Tech ECE
project. It extends the Phase 1 report/template work with a reproducible
training and evaluation pipeline for raw I/Q automatic modulation
classification (AMC).

## Scope

- Load the RadioML 2016.10a pickle format and normalize each I/Q example.
- Create deterministic, stratified train/validation/test splits.
- Train a compact 1-D CNN with class-weighted cross-entropy, AdamW, and cosine
  learning-rate scheduling.
- Report overall accuracy, confusion matrices, and accuracy at each SNR.

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
  --checkpoint artifacts/amc.pt
```

The default split is 70% training, 15% validation, and 15% testing. The
checkpoint stores the class-name ordering and input shape needed for
evaluation. Use `--epochs`, `--batch-size`, and `--learning-rate` to reproduce
experiments or compare model variants.

## Phase 2 deliverables

1. Dataset preparation and reproducible experiment configuration.
2. CNN baseline implementation for raw I/Q signals.
3. Test accuracy, confusion matrix, and SNR-wise analysis.
4. A foundation for Phase 3 improvements such as augmentation, residual
   networks, transformer baselines, and over-the-air validation.

The LaTeX report template remains in the repository and can be updated with
the results generated under `artifacts/`.

The Phase 2 workflow is summarized visually in
[`artifacts/phase2-results-overview.svg`](artifacts/phase2-results-overview.svg).