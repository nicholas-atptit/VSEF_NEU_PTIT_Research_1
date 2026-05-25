# VN30 Aggressive Model Tuning Result Summary

## Best Validation Candidate

- Candidate: `ensemble__logistic_xgboost__momentum_lag__h20__w000__t0p45`.
- Model family: soft_vote_ensemble.
- Feature group: momentum_lag.
- Horizon: 20.
- Threshold: 0.45.
- Validation accuracy: 52.38%.
- Strongest validation baseline: lag1_direction at 50.07%.
- Validation lift over strongest baseline: +2.30 pp.

## Locked Final Result

- Final accuracy: 51.09%.
- Strongest final baseline: always_up at 52.12%.
- Lift over strongest final baseline: -1.03 pp.
- Final rows: 4674.
- Validation-final gap: -1.29 pp.
- Rolling stability, min rolling250 accuracy: 14.40%.
- Ticker stability, min ticker accuracy: 36.54%.

## Claim Boundary

- Baseline60 defensible: false.
- Target62 defensible: false.
- Final65 defensible: false.
- Acceptance label: not_claimable.

Paper-safe wording:

> In a validation-locked VN30 stock hourly diagnostic benchmark, the selected candidate reached 51.09% final pooled directional accuracy over 4674 rows, -1.03 pp versus the strongest same-horizon simple baseline. This supports only the stated diagnostic benchmark scope and does not support trading, profitability, live-deployment, VN100, top-k-as-overall-accuracy, or final65 claims.

Artifacts:

- `reports/generated/vn30_aggressive_model_tuning/locked_candidate.json`
- `reports/generated/vn30_aggressive_model_tuning/final_once_result.csv`
- `reports/generated/vn30_aggressive_model_tuning/baseline_comparison.csv`
- `reports/generated/vn30_aggressive_model_tuning/rolling_stability.csv`
- `reports/generated/vn30_aggressive_model_tuning/ticker_stability.csv`
