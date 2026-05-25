# VN30 Legacy Full-Horizon Claim Boundary

## Audit Checks

| Check | Status |
| --- | --- |
| no_final_window_selection | pass |
| no_leakage | pass |
| full_30_stock_headline_coverage | pass |
| no_ticker_subset | pass |
| no_confidence_abstention | pass |
| no_topk_substitution | pass |
| horizon_specific_row_counts_reported | pass |
| h40_main_claim_kept_separate | pass |

## Boundary

- Main paper claim remains `Logistic L2 baseline_C_closest h40 threshold 0.55` with final accuracy 61.63% unless a separate pre-registered validation-only horizon-selection objective is adopted before final-window scoring.
- This run does not use final-window score for model, feature, threshold, router, or horizon selection.
- Horizon-specific best rows are diagnostic and full-coverage only.
- Regime-aware rows include `regime_context` features and a validation-selected regime threshold router; both are scoring-only on final rows.
- No market data was fetched, no ticker subset was used, no confidence abstention was used, and no top-k metric is substituted for overall directional accuracy.
- No paper or DOCX artifact was generated.
