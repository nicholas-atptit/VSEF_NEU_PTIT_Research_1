# VN30 Hourly Selected Candidate Row-Level Rerun Protocol

## Purpose

The purpose of this rerun is to reproduce the already selected Track A canonical-like VN30 stock-only hourly candidate only to save row-level final-window predictions.

This is not new model selection. This is not new tuning. This is not a broad benchmark sweep.

## Fixed Candidate

The candidate is fixed before the rerun:

- `model=l2_logistic` (L2 Logistic).
- `horizon=h=40`.
- `feature_set=feature_set_C_closest`.
- `threshold=0.50`.
- Setup: Track A canonical-like VN30 stock-only hourly.

The selected candidate must not be changed by this task.

## Required Reproduction Rules

The rerun must use the same data split, same feature rules, same final scoring window, and same selected-candidate logic as the existing result:

- Data sources are existing local stock and index caches only.
- No new market data fetch is allowed.
- Feature matrix must be rebuilt with `feature_set_C_closest`.
- Labels must use the h=40 absolute forward direction rule from the existing code.
- Train split must use rows through 2023-12-31 23:59:59.
- Validation split must use rows from 2024-01-01 through 2024-12-31 23:59:59.
- Final split must use rows from 2025-01-01 onward.
- The model must be fit only for the fixed selected candidate.
- The final score must be computed on the same final window.
- Threshold 0.50 must be applied directly to predicted probability.
- No confidence abstention is allowed.
- No ticker subset is allowed.
- No top-k or ranking rule is allowed.

## Stop Rule

If exact reproduction cannot be guaranteed from existing code and local data, the rerun must stop and report the reason instead of inventing row-level predictions.

If reproduced final accuracy differs from the existing 61.51% result by more than 0.10 percentage point, the reproduction must be marked `failed_or_mismatch`.

If the reproduced final row count differs materially from the existing 4,074 final rows, the reproduction must be marked `failed_or_mismatch`.

No paper claim should be updated from a mismatched reproduction.

## Outputs

The outputs are used only for rolling stability checks and figure/table generation:

- `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/row_predictions.csv`
- `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/selected_candidate_reproduction_summary.csv`
- `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/selected_candidate_reproduction_manifest.json`
- `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/selected_candidate_reproduction_log.md`

These outputs are not new model-selection artifacts and do not change the selected candidate.

## Claim Boundary

The rerun does not support a target62 upgrade or a final65 claim.

The rerun does not make any trading, profitability, investment-recommendation, or live-deployment claim.
