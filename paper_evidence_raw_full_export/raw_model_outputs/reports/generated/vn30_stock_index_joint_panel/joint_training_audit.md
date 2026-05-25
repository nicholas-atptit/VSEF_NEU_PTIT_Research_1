# Joint Panel Training V1 Audit

- Training status: `run`.
- Result safety: `exploratory`.
- Combined reaches 60: false.
- Combined reaches 65: false.
- Stock-only reaches 60: false.
- Stock-only reaches 65: false.

## Audit Items

| audit_item | passed | detail |
| --- | --- | --- |
| metric_is_overall_directional_accuracy | True | candidate rows use pooled target_direction vs prediction accuracy |
| stock_30_present | True | 30 |
| index_6_present | True | 6 |
| total_36_present | True | 36 |
| combined_result_reported | True | 0.4920979020979021 |
| stock_only_result_reported | True | 0.4898937992463172 |
| index_only_result_reported | True | 0.4955800108244633 |
| no_confidence_abstention | True | False |
| no_instrument_subset | True | False |
| no_topk_ranking | True | False |
| no_daily_result_used_as_hourly | True | scripts read hourly cache paths only |
| final_labels_not_used_for_selection | True | True |
| final_evaluation_scoring_only | True | True |
| leakage_audit_passed | True | feature audit file exists |
| baseline_comparison_included | True | baseline_delta_summary.csv |
| combined_reaches_60 | False | 0.4920979020979021 |
| combined_reaches_65 | False | 0.4920979020979021 |
| stock_only_reaches_60 | False | 0.4898937992463172 |
| stock_only_reaches_65 | False | 0.4898937992463172 |
| result_safety | True | exploratory |
