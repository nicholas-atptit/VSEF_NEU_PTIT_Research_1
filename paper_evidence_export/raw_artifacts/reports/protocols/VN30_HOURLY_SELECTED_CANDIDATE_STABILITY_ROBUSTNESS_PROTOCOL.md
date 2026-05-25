# VN30 Hourly Selected Candidate Stability Robustness Protocol

## Scope

- Selected candidate: L2 Logistic h40, `feature_set_C_closest`, threshold 0.50.
- Final accuracy: 61.51%.
- Prior baseline60: Logistic h40 = 60.43%.
- Historical RF h60 reference: 60.31%.
- Target62: final accuracy >=62%.
- Result status before this audit: improved baseline60 candidate, not target62.

## Purpose

The purpose of this audit is to check average accuracy, ticker/month/quarter/regime stability, lift over majority/simple baselines, bootstrap robustness, simple significance evidence, validation-final mismatch, and whether the selected 61.51% result is stable enough to describe as an improved baseline60 result.

## Controls

- No model training.
- No new candidate selection.
- No final accuracy for selection.
- No selected-candidate change.
- No confidence abstention.
- No ticker subset.
- No top-k/ranking substitution.
- No data fetch.
- No paper/DOCX generation.
- No trading, profitability, investment-recommendation, or live-deployment claim.

## Inputs

The audit uses the existing selected-candidate summary and mandatory audit slice outputs:

- `selected_candidate_summary.csv`.
- `target62_by_ticker.csv`.
- `target62_by_time.csv`.
- `target62_by_regime.csv`.
- `target62_validation_mismatch.csv`.

No row-level predictions are regenerated.
