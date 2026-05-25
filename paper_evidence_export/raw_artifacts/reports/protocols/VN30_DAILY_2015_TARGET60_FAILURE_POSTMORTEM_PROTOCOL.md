# VN30 Daily 2015 Target60 Failure Postmortem Protocol

- Track: VN30 daily 2015 benchmark.
- Branch: `research/vn100-evidence-hardening-v1`.
- Starting tag: `nckh/vn30-daily-2015-target60-v2`.
- Status before postmortem: target60 not passed.
- Current best recorded daily result: LightGBM `daily_cross`, h=40, final accuracy 57.58%.
- Current best recorded v2 result: LightGBM `volatility_normalized`, h=50, threshold 0.525, final accuracy 57.45%.

## Scope Guardrails

- Use daily data only.
- Do not use hourly data.
- Do not resample daily data to hourly.
- Do not fetch data.
- Do not change the VN30 universe.
- Do not generate paper or DOCX artifacts.
- Do not make trading-readiness, profitability, or live-deployment claims.
- Do not use final labels for candidate or threshold selection.
- Do not run a new tuning sweep.
- Reproducing an already recorded candidate is allowed only for diagnostics that require row-level predictions.

## Inputs

- `outputs/vn30_daily_2015_benchmark/`
- `outputs/vn30_daily_2015_target60_optimization/`
- `outputs/vn30_daily_2015_target60_v2/`
- `data/market_cache/vnstock_data/vn30/daily_2015/`
- `configs/universes/vn30_constituents_frozen.csv`

## Questions

1. Why did v2 fail to improve over v1?
2. Is h=40 still the canonical best daily candidate?
3. Did h=50 `volatility_normalized` overfit validation, underperform final, or only underperform the h=40 record?
4. Which tickers are the largest accuracy drag?
5. Which months and quarters are the largest accuracy drag?
6. Is the problem mainly class imbalance?
7. Are simple baselines close to model performance?
8. Is validation selecting weak final candidates?
9. Is another daily v3 attempt justified?

## Required Outputs

The audit script must write:

- `reports/generated/vn30_daily_2015_target60_postmortem/daily_best_candidate_comparison.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_ticker_drag.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_time_drag.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_class_balance.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_baseline_comparison.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_validation_final_mismatch.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_target60_failure_postmortem.md`

## Method

1. Inventory saved aggregate results from benchmark, v1 target60 optimization, and v2 target60 optimization.
2. Identify these comparison anchors:
   - Benchmark best final candidate.
   - V1 best recorded final candidate.
   - V1 best validation candidate, when available.
   - V2 best recorded final candidate.
   - V2 best validation-stability candidate, when available.
3. Reproduce only the already recorded v1 h=40 `daily_cross` candidate and the already recorded v2 h=50 `volatility_normalized` candidate to create row-level diagnostic predictions.
4. Compute ticker drag, monthly drag, and quarterly drag from reproduced final-period row-level predictions.
5. Compute class balance and final-period majority-class baselines from the reproduced diagnostic rows.
6. Compare model accuracy against:
   - fixed 50% naive baseline;
   - final-period majority-class diagnostic baseline;
   - benchmark baseline files where available.
7. Measure validation-final mismatch using saved candidate tables:
   - validation score rank versus final accuracy rank;
   - Spearman and Pearson correlation where enough rows exist;
   - final accuracy of the validation-selected candidate versus the final-best candidate.
8. Answer whether v3 is justified only from diagnostics, not from hindsight final-label candidate selection.

## Decision Rules

Daily v3 is justified only if diagnostics show a narrow, validation-testable fix, such as:

- a repeatable validation-final mismatch pattern that can be corrected before final scoring;
- a class-imbalance treatment that is validated on rolling 2021-2024 and not chosen on final labels;
- a ticker or regime issue that can be addressed without changing the universe or excluding weak tickers;
- a baseline gap large enough to suggest unused daily signal remains.

Daily v3 is not justified if:

- v2 broadened features, horizons, and thresholds but did not improve over the h=40 record;
- the best model is only slightly above a simple majority-class diagnostic baseline;
- errors concentrate in specific final-period months or tickers without a validation-only fix;
- validation rankings are weak predictors of final accuracy;
- any proposed fix requires changing the target, changing the universe, using hourly data, or selecting from final labels.

## Claim Boundary

The postmortem may claim only that the daily 2015 target60 attempt failed under the current target and universe. It may report diagnostic accuracy, ticker drag, time drag, class balance, baseline deltas, and validation-final mismatch. It must not claim profitability, trading readiness, live deployment value, or hourly evidence from daily data.
