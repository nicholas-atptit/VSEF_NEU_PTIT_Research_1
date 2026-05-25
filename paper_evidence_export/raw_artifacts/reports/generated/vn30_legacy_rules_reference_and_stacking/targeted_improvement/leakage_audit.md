# VN30 Legacy Targeted Improvement Leakage Audit

## Checks

| Check | Status |
| --- | --- |
| validation_only_selection | pass |
| final_window_scoring_only | pass |
| full_30_ticker_coverage | pass |
| ticker_subset_main_claim | pass |
| confidence_abstention | pass |
| top_k_substitution | pass |
| stacking_main_candidate | pass |
| provider_behavior_changed | pass |
| market_data_fetch | pass |
| h40_legacy_split_used | pass |
| index_features_lagged_context_only | pass |

## Notes

- Candidate thresholds, ticker repairs, regime routers, regularization grids, and compact feature counts were selected from validation metrics only.
- Final accuracy appears in reports only after the selected per-track candidates are fixed.
- High overfit-risk selected candidates: none.
