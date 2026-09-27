# AURA

**AURA: Risk-Aware Deferral for Banking Intent Classification**

AURA is a capstone research project that studies when a banking intent
classifier should accept its prediction and when it should defer the request
for human review.

## Status

The project is in initial development. No experimental results have been
produced yet.

## Research Question

Can a calibrated, pair-dependent risk-aware abstention policy reduce
high-impact accepted errors in BANKING77 intent classification compared with
confidence-based baselines at the same automation coverage?

## Planned Pipeline

1. Load and validate BANKING77.
2. Train a TF-IDF and Logistic Regression intent classifier.
3. Calibrate the classifier probabilities.
4. Estimate the harm of possible intent misclassifications.
5. Decide whether to auto-route or defer each request.
6. Compare policies at matched automation coverage.

## Dataset

The project uses the BANKING77 dataset, containing 13,083 banking-support
queries across 77 intents.

Dataset files are stored locally and are not committed to this repository.

## Project Structure

- `src/aura/data`: data loading, validation, and splitting
- `src/aura/models`: classifier training and prediction
- `src/aura/calibration`: probability calibration
- `src/aura/risk`: harm matrices and expected-harm calculations
- `src/aura/policies`: acceptance and deferral policies
- `src/aura/evaluation`: metrics, tables, and plots
- `tests`: unit and integration tests

## Responsible Use

AURA is a research prototype. It does not perform banking transactions,
provide financial advice, or replace professional human review.     