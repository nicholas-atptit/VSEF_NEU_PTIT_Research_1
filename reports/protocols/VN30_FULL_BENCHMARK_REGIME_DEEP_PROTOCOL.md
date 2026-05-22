# VN30 Full Benchmark Regime Deep Protocol

## Scope

The main target is VN30 stock-only hourly overall directional accuracy. The headline claim requires full 30-stock coverage across the frozen VN30 universe and uses existing local data/artifacts only.

The reference result is the current selected candidate:

- L2 Logistic, h=40, `feature_set_C_closest`, threshold 0.50.
- Final accuracy: 61.51%.
- Final rows: 4,074.
- Majority baseline: 50.44%.
- Validation-final gap: +9.63 percentage points.
- Rolling stability: mixed.
- Claim level: exploratory improved baseline evidence.
- Final65: not established.

## Validation Rules

Walk-forward validation is mandatory. Candidate selection must use validation-window evidence only, and final scoring is performed only after the candidate is fixed.

The benchmark must not use:

- Target leakage, future features, same-row target leakage, or final-window derived features.
- Final-window score for model, feature, horizon, threshold, sequence length, early stopping, or regime selection.
- Ticker subsets for the main claim.
- Confidence abstention for the main claim.
- Top-k or ranking metrics as a substitute for overall directional accuracy.
- New market data fetches or provider behavior changes.

Train and validation labels must be strict: a row belongs to train or validation only if the feature timestamp and the future outcome timestamp both fall inside that split. Final rows are scoring-only rows from the final period with available future outcomes.

## Feature and Regime Rules

Index data may be used only as lagged market/context features and as a separate index benchmark reference. Index rows are not mixed into the stock-only headline target.

The regime-aware layer is used for ex-ante features and slice evaluation. Market direction regimes (`bull`, `bear`, `sideway`) and volatility regimes (`high_volatility`, `low_volatility`) must be computed from lagged rolling index returns/volatility only. Final-period regime labels may be used for evaluation only when they are computable ex ante.

## Model Comparison Rules

The benchmark includes simple ex-ante baselines, classical machine learning models, and deep learning models. Deep learning results are comparative benchmark results; they are not automatically stronger claims.

Validation-selected thresholds are reported separately from fixed 0.50 thresholds. Deep learning sequence lengths and early stopping decisions are validation-only.

## Claim Boundary

Any result above 61.51% must pass audit before being considered a stronger candidate. Any result above 65% remains exploratory unless future blind validation confirms it.

No trading, profitability, investment recommendation, or live-deployment claim is made by this benchmark.
## Required Implementation Outputs

The benchmark implementation must create the following scripts:

- `scripts/research/audit_vn30_full_benchmark_data_label_scope.py`
- `scripts/research/run_vn30_full_benchmark_regime_deep.py`
- `scripts/research/audit_vn30_full_benchmark_regime_deep.py`

The benchmark output directory is:

- `reports/generated/vn30_full_benchmark_regime_deep/`

Required outputs:

- `data_label_audit.md`
- `data_label_summary.csv`
- `label_distribution_by_split.csv`
- `baseline_results.csv`
- `baseline_row_predictions.csv`
- `classical_ml_results.csv`
- `classical_ml_selected_candidates.csv`
- `classical_ml_row_predictions.csv`
- `deep_learning_results.csv`
- `deep_learning_selected_candidates.csv`
- `deep_learning_row_predictions.csv`
- `deep_learning_skip_report.md`
- `regime_feature_manifest.json`
- `regime_distribution.csv`
- `regime_slice_results.csv`
- `walk_forward_config.json`
- `walk_forward_validation_results.csv`
- `walk_forward_final_results.csv`
- `unified_leaderboard.csv`
- `best_by_group.csv`
- `best_overall_validation_selected.json`
- `comparison_summary.md`
- `audit_result.md`
- `claim_register.md`