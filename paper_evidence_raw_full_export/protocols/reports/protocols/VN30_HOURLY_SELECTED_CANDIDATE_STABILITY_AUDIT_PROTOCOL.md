# VN30 Hourly Selected Candidate Stability Audit Protocol

## Scope

- Selected candidate: L2 Logistic h40, `feature_set_C_closest`, threshold 0.50.
- Setup: Track A canonical-like VN30 stock-only hourly.
- Final accuracy: 61.51%.
- Baseline Logistic h40: 60.43%.
- Historical RF h60 reference: 60.31%.
- Target62: not reached.
- Final65: not established.

## Purpose

This is a paper-ready stability audit for the selected L2 Logistic h40 result. The output is intended to support future paper tables, figures, and claim-boundary language.

## Controls

- No new model training.
- No new selection.
- No selected-candidate change.
- No confidence abstention.
- No ticker subset.
- No top-k/ranking substitution.
- No data fetch.
- No DOCX or paper draft generation.
- No trading, profitability, investment-recommendation, or live-deployment claim.

## Claim Rule

The audit may support an exploratory improved baseline60 claim if stability and baseline-lift evidence are adequate. It must not upgrade the result to target62 because final accuracy is below 62%. It must not make a final65 claim.

## Input Limitation

The audit uses existing selected-candidate and slice-level audit outputs. Row-level predictions were not saved in the target62 run, so row-level rolling windows are reported as unavailable unless row-level predictions are present in the existing artifacts. No row-level predictions are regenerated.
