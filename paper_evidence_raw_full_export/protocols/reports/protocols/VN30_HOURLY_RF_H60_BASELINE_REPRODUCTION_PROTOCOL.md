# VN30 Hourly RF h60 Baseline Reproduction Protocol

## Purpose

This protocol checks whether the locked historical Random Forest h=60 VN30 hourly stock-only baseline is reproduced inside the current expanded-model pipeline, and whether the expanded model-pool failure is apples-to-apples with that locked baseline.

## Locked Reference

- Historical locked baseline: Random Forest h=60 around 60.31%.
- Canonical reference artifact: `reports/generated/vn30_hourly_2015_consistency/rf_h60_consistency_audit.csv`.
- Historical row count: 3,474 final rows.
- Historical target: absolute direction, `future_close > current_close`.
- Historical universe: 30/30 active VN30 January 2025 stocks.
- Historical feature set: stock lagged plus market-index context, recorded as feature set C in the horizon-relative target run.

## Current Pipeline Comparator

The current comparator is the expanded model-pool screening run:

- `outputs/vn30_hourly_expanded_model_pool_screening/final_candidate_results.csv`
- Model: Random Forest.
- Horizon: h=60.
- Feature sets checked:
  - `stock_lagged_rolling`
  - `stock_lagged_rolling_plus_index_context`

The closest available feature set to the historical feature set C is `stock_lagged_rolling_plus_index_context`.

## Required Checks

The reproduction audit must compare:

1. Historical RF h=60 result around 60.31%.
2. Current expanded-pipeline RF h=60 result.
3. Same universe check: 30/30.
4. Same horizon check: h=60.
5. Same final window check.
6. Same target definition check.
7. Same row count check.
8. Same feature set or closest available feature set.
9. Difference explanation if reproduced accuracy differs.

## Rules

- No new model family training.
- No broad tuning.
- No data fetch.
- No confidence abstention.
- No ticker subset.
- No final-label selection.
- No paper or DOCX generation.
- No trading, profitability, investment-recommendation, or live-deployment claim.

## Decision Rule

The historical baseline is reproduced inside the current expanded-model pipeline only if the closest current RF h=60 row uses the same universe, horizon, target, final window, row count, and comparable feature set, and its accuracy is materially equal to the canonical 60.31% reference.

If row count, split design, final window, or feature implementation differs, the historical lock may remain valid as its own locked benchmark, but it is not an apples-to-apples comparator for the expanded-model pipeline failure.

