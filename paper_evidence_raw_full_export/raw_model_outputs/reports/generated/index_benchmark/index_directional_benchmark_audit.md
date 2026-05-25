# Supported Index Directional Benchmark Audit

- Output directory: `outputs/index_directional_benchmark`.
- Audit passed: yes.
- Any exact index/frequency/model/horizon passed 60: yes.
- Index-only: yes.
- Stock claims: no.
- Trading/profitability/live-deployment claims: no.
- Daily-to-hourly resampling: no.
- Hourly-to-daily resampling: no.

## Best Results

| frequency | index_code | model | horizon | final_accuracy | baseline_accuracy | pass_60 |
| --- | --- | --- | --- | --- | --- | --- |
| 1D | UPCOMINDEX | xgboost | 1 | 97.60% | 97.60% | True |
| 1H | VNINDEX | xgboost | 40 | 66.67% | 63.52% | True |

## Checks

| check | status | severity | details |
| --- | --- | --- | --- |
| output_directory_exists | PASS | info | outputs/index_directional_benchmark |
| run_config_exists | PASS | info | outputs/index_directional_benchmark/run_config.json |
| manifest_exists | PASS | info | outputs/index_directional_benchmark/benchmark_manifest.json |
| accuracy_summary_exists_non_empty | PASS | info | outputs/index_directional_benchmark/accuracy_summary.csv |
| baseline_summary_exists_non_empty | PASS | info | outputs/index_directional_benchmark/baseline_summary.csv |
| predicted_vs_actual_exists_non_empty | PASS | info | outputs/index_directional_benchmark/predicted_vs_actual.csv |
| index_only | PASS | info | index_only |
| no_daily_hourly_resampling | PASS | info | no resampling flags |
| no_stock_claims | PASS | info | stock claims made false |
| no_trading_profitability_claims | PASS | info | trading/profitability claims made false |
| correct_frequency_values | PASS | info | 1D,1H |
| accuracy_required_columns | PASS | info | index_code,frequency,model,horizon,validation_accuracy,validation_rows,final_accuracy,final_rows,final_coverage,baseline_name,baseline_accuracy,delta_vs_baseline,pass_60,claim_level |
| any_index_passed_60 | PASS | info | pass_60_count=39 |
| validation_only_selection_if_used | PASS | info | no model selection policy used; all exact combinations reported |
| final_eval_scoring_only | PASS | info | final rows used only for scoring reported combinations |
