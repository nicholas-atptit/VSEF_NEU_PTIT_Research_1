# VN30 Legacy Rules Reference and Stacking Audit

## Verdict

- Reference reproduced: yes.
- Leakage audit passed: yes.
- Apples-to-apples model comparison: yes.
- Stacking improves over 61.51%: no.
- Rolling stability not worse: no.
- Overfit risk classification: `low`.
- Acceptance classification: `single_model_improvement`.

## Checks

| Check | Status |
| --- | --- |
| reference_reproduced | pass |
| old_split_used_consistently | pass |
| full_30_ticker_coverage | pass |
| no_final_window_selection | pass |
| no_leakage | pass |
| no_ticker_subset | pass |
| no_confidence_abstention | pass |
| no_topk_substitution | pass |
| stacking_meta_validation_only | pass |
| ensemble_weights_validation_only | pass |
| final_score_scoring_only | pass |
| model_comparison_apples_to_apples | pass |
| stacking_improves_over_61_51 | fail |
| rolling_stability_not_worse | fail |

## Selected Validation-Only Results

- Reference final accuracy: 61.51%; final rows: 4074.
- Best single model: `legacy_single__logistic_l2__baseline_C_closest__h40__validation_selected_threshold__t0p550` validation 52.36%, final 61.63%.
- Best stacking method: `legacy_stack__meta_lightgbm_stacking` validation 61.02%, final 48.72%.

## Boundary

- No market data fetched.
- No provider behavior changed.
- No confidence abstention, ticker subset, or top-k substitution.
- No trading/profitability/live-deployment claim.
