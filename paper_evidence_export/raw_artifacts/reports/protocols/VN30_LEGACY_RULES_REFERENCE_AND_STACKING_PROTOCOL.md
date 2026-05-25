# VN30 Legacy Rules Reference and Stacking Protocol

## Purpose

This experiment fixes the comparability gap found in the previous full benchmark. The prior full benchmark used stricter train/validation label-window rules and did not final-score/export the exact L2 Logistic h40 reference candidate in the unified leaderboard. This protocol reruns the reference and newer model families under legacy-compatible row rules so the comparison is apples-to-apples.

## Reference Target

The old reference must be reproduced first:

- Model: L2 Logistic.
- Horizon: h=40.
- Feature family: `feature_set_C_closest` / `baseline_C_closest`.
- Threshold: fixed 0.50.
- Expected train rows: 9,600.
- Expected validation rows: 30,030.
- Expected final rows: 4,074.
- Expected final accuracy: 61.51%.

If the reference reproduction fails by more than 0.10 percentage point or the final row count differs materially, diagnostics continue, but all model comparisons are marked as not apples-to-apples.

## Legacy-Compatible Evaluation Rules

New models must be evaluated under the same legacy-compatible split and row rules used by the reference reproduction:

- Train rows: feature timestamp up to 2023-12-31 23:59:59 with non-null h-step label.
- Validation rows: feature timestamp from 2024-01-01 through 2024-12-31 23:59:59 with non-null h-step label.
- Final rows: feature timestamp from 2025-01-01 onward with non-null h-step label.
- Final window remains scoring-only.
- Model, feature, threshold, ensemble weight, and stacking selection use validation evidence only.

The headline result requires full 30-stock VN30 coverage. No ticker subset, confidence abstention, or top-k/ranking substitute is allowed for the main claim.

## Model and Stacking Rules

Single-model comparisons cover classical machine learning models and feasible deep sequence models. The primary horizon is h=40. Threshold 0.50 is primary; validation-selected thresholds are secondary and reported separately.

Stacking uses validation-only meta-learning:

- Base models are trained on train rows only.
- Base predictions are generated on validation and final rows.
- Ensemble weights are selected from validation metrics only.
- Meta-models are trained only on validation base predictions.
- Final labels are not used for ensemble weights, meta-model fitting, or method selection.

## Claim Boundary

No trading, profitability, investment recommendation, or live-deployment claim is made. Any improvement beyond 61.51% is a historical validation/final benchmark result only and must pass the audit checks before it can be classified as a single-model or stacking improvement.
