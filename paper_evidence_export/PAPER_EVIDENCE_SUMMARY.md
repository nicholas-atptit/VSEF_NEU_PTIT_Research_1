# Paper Evidence Summary

## Branch And Commit

- Branch: `research/vn30-legacy-rules-reference-and-stacking-v1`
- Commit: `35078143af530c18e0576d071c2471d2f923b186`
- Export timestamp: `2026-05-21T15:42:07.055171+00:00`

## Export Scope

- Created from current local repository artifacts only.
- Benchmarks rerun: no.
- Data fetch run: no.
- Model training run: no.
- Paper/DOCX generated: no.
- Provider behavior changed: no.
- Tags created or pushed: no.

## Main Claim-Eligible Result

- logistic_l2 / baseline_C_closest / h40 / validation_selected_threshold threshold 0.55: 61.63% final accuracy.
- Claim status: claim_eligible.

## Best Descriptive Result

- bull_bear_sideway_router / h40 / fixed_0.50: 63.33% final accuracy.
- Claim status: descriptive_only. This does not replace the main claim.

## Cooperation Diagnostic

- validation_lift_weighted_soft_vote: 62.00% final accuracy.
- Claim status: descriptive_only.

## Full-Horizon Results

- h20: 51.71% final accuracy
- h40: 61.63% final accuracy
- h60: 56.02% final accuracy
- h80: 52.33% final accuracy

## Model Universe Counts

- Model groups listed: 12.
- Fair-tuning groups tuned: 13.
- Planned variants: 75.
- Attempted variants: 75.
- Run variants: 74.
- CatBoost status: run.
- GARCH diagnostic status: not_recommended_with_reason.
- GARCH used as main direction classifier: no.
- Leakage audit passed: yes.

## Fair Tuning

- Model groups tuned: 13.
- Candidate/config rows evaluated successfully: 571.
- Best validation-selected fair-tuning model: stacking_xgboost_meta, final 47.05%, claim status failed_transfer.
- Best descriptive final remains separated from claim eligibility.

## KNN Support

- Standalone KNN comparator: 53.56%.
- Best descriptive KNN-support row: 57.46%.
- Validation-selected KNN-support row: 49.02%.

## Claim Boundary

- Claim-eligible leaderboard and descriptive final leaderboard are exported separately.
- No trading, profitability, investment recommendation, live-deployment, top-k, confidence-subset, or ticker-subset claim is made.
- Descriptive and diagnostic rows are retained as evidence context only.

## Missing Artifacts

- None

## Missing Key Rows

- None

## Tables Generated

- `table_01_dataset_split_summary.csv`
- `table_02_model_universe_coverage.csv`
- `table_03_claim_eligible_leaderboard.csv`
- `table_04_descriptive_final_leaderboard.csv`
- `table_05_full_horizon_summary.csv`
- `table_06_model_family_audit.csv`
- `table_07_fair_tuning_summary.csv`
- `table_08_knn_support_summary.csv`
- `table_09_model_cooperation_summary.csv`
- `table_10_overfit_risk_summary.csv`
- `table_11_statistical_diagnostics_summary.csv`
- `table_12_garch_diagnostic_summary.csv`
- `table_13_training_validation_final_key_results.csv`
- `table_14_references_to_artifacts.csv`

## Figures Generated

- `fig_01_model_universe_coverage.png`
- `fig_02_claim_eligible_vs_descriptive_accuracy.png`
- `fig_03_final_accuracy_by_model_family.png`
- `fig_04_validation_vs_final_accuracy.png`
- `fig_05_full_horizon_accuracy.png`
- `fig_06_overfit_risk_summary.png`
- `fig_07_knn_support_summary.png`
- `fig_08_model_cooperation_summary.png`
- `fig_09_current_main_vs_descriptive_best.png`
- `fig_10_claim_boundary_map.png`

## Recommended Paper Sections

- Data and protocol: tables 01, 02, 14.
- Main results: tables 03, 04, 05, 13 and figures 02, 05, 09.
- Model families and tuning: tables 06, 07 and figure 03.
- Diagnostics and claim boundary: tables 08, 09, 10, 11, 12 and figures 06, 07, 08, 10.
