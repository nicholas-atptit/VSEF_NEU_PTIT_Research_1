# VN30 Legacy Full-Horizon Model Comparison

## Scope

- Horizons: h20, h40, h60, h80.
- Data fetch: no.
- Split rule: legacy feature-timestamp split with non-null horizon labels.
- Selection rule for diagnostic horizon rows: validation accuracy, then candidate id; final rows are scoring-only.
- Main paper h40 result remains separate from full-horizon diagnostics.

## H40 Main Paper Result

- Fixed main result: Logistic L2 baseline_C_closest h40 threshold 0.55.
- Validation accuracy: 52.36%.
- Final accuracy: 61.63%.
- Final rows: 4074.
- Ticker coverage: 30/30.

## Horizon Row Counts

| Horizon | Train Rows | Validation Rows | Final Rows | Ticker Coverage | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| h20 | 9,600 | 30,030 | 4,674 | 30/30 | 36 |
| h40 | 9,600 | 30,030 | 4,074 | 30/30 | 36 |
| h60 | 9,600 | 30,030 | 3,474 | 30/30 | 36 |
| h80 | 9,600 | 30,030 | 2,874 | 30/30 | 36 |

## Diagnostic Best By Horizon

| Horizon | Candidate | Group | Validation | Final | Gap | Rows |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| h20 | `fullhorizon__classical_ml__xgboost__baseline_C_closest__h20__validation_selected_threshold__t0p450` | classical_ml | 52.05% | 51.71% | -0.34 pp | 4,674 |
| h40 | `fullhorizon__classical_ml__logistic_l2__baseline_C_closest__h40__validation_selected_threshold__t0p550` | classical_ml | 52.36% | 61.63% | +9.27 pp | 4,074 |
| h60 | `fullhorizon__classical_ml__logistic_elastic_net__baseline_C_closest__h60__validation_selected_threshold__t0p550` | classical_ml | 51.55% | 56.02% | +4.47 pp | 3,474 |
| h80 | `fullhorizon__classical_ml__logistic_elastic_net__baseline_C_closest__h80__validation_selected_threshold__t0p550` | classical_ml | 51.53% | 52.33% | +0.80 pp | 2,874 |

## Best By Model Group And Horizon

| Horizon | Group | Candidate | Validation | Final |
| --- | --- | --- | ---: | ---: |
| h20 | baseline | `fullhorizon__baseline__moving_average_rule__h20` | 50.00% | 48.82% |
| h20 | classical_ml | `fullhorizon__classical_ml__xgboost__baseline_C_closest__h20__validation_selected_threshold__t0p450` | 52.05% | 51.71% |
| h20 | deep_learning | `fullhorizon__deep_learning__gru__baseline_C_closest_sequence16__h20__seq16__fixed_0p50` | 49.78% | 49.72% |
| h20 | regime_aware | `fullhorizon__regime_aware__xgboost__regime_context__h20__validation_selected_threshold__t0p450` | 52.05% | 51.71% |
| h40 | baseline | `fullhorizon__baseline__random_walk_direction__h40` | 49.40% | 50.05% |
| h40 | classical_ml | `fullhorizon__classical_ml__logistic_l2__baseline_C_closest__h40__validation_selected_threshold__t0p550` | 52.36% | 61.63% |
| h40 | deep_learning | `fullhorizon__deep_learning__lstm__baseline_C_closest_sequence16__h40__seq16__fixed_0p50` | 49.55% | 49.73% |
| h40 | regime_aware | `fullhorizon__regime_aware__regime_threshold_router__baseline_C_closest__h40__validation_selected_regime_thresholds` | 52.47% | 61.68% |
| h60 | baseline | `fullhorizon__baseline__random_walk_direction__h60` | 49.07% | 50.49% |
| h60 | classical_ml | `fullhorizon__classical_ml__logistic_elastic_net__baseline_C_closest__h60__validation_selected_threshold__t0p550` | 51.55% | 56.02% |
| h60 | deep_learning | `fullhorizon__deep_learning__lstm__baseline_C_closest_sequence16__h60__seq16__fixed_0p50` | 48.56% | 52.91% |
| h60 | regime_aware | `fullhorizon__regime_aware__logistic_elastic_net__regime_context__h60__validation_selected_threshold__t0p550` | 51.55% | 56.02% |
| h80 | baseline | `fullhorizon__baseline__previous_direction__h80` | 50.79% | 41.72% |
| h80 | classical_ml | `fullhorizon__classical_ml__logistic_elastic_net__baseline_C_closest__h80__validation_selected_threshold__t0p550` | 51.53% | 52.33% |
| h80 | deep_learning | `fullhorizon__deep_learning__lstm__baseline_C_closest_sequence16__h80__seq16__fixed_0p50` | 47.24% | 54.91% |
| h80 | regime_aware | `fullhorizon__regime_aware__logistic_elastic_net__regime_context__h80__validation_selected_threshold__t0p550` | 51.53% | 52.33% |

## Audit Summary

- no_final_window_selection: pass.
- no_leakage: pass.
- full_30_stock_headline_coverage: pass.
- no_ticker_subset: pass.
- no_confidence_abstention: pass.
- no_topk_substitution: pass.
- horizon_specific_row_counts_reported: pass.
- h40_main_claim_kept_separate: pass.

## Interpretation

- The full-horizon tables are diagnostics for horizon robustness, not a replacement for the h40 paper claim.
- Regime-context and regime-threshold-router rows are reported in the regime-aware group; they are not used to replace the fixed h40 main claim.
- No ticker subset, confidence abstention, or top-k/ranking substitute is used.
