# VN30 Hourly Track A Target62 Validation-Safe Protocol

## Scope

- Main target: VN30 stock-only hourly overall directional accuracy.
- Setup: Track A canonical-like setup.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage.
- Baseline60: Logistic h40 = 60.43%.
- Historical RF h60 reference: 60.31%.
- Target62: main target.
- Final65: stretch target only.

## Boundary

The diagnostic L2 Logistic h40 65.71% row is not retroactively claimable. This run uses a pre-registered validation-only rule. Any selected result remains subject to audit, and any pass65 result remains exploratory unless confirmed by future blind evidence.

## Eligible Candidates

- L2 Logistic Regression.
- Balanced L2 Logistic Regression.
- ElasticNet Logistic if stable.
- Ridge-style logistic variants if available.

Feature sets:

- `regime_feature_v2`.
- `feature_set_C_closest`.
- `stock_lagged_rolling_plus_index_context`, only as a Track A-compatible calendar-split feature set.

Horizons:

- h=40.
- h=60.
- h=80.

Thresholds:

- 0.45.
- 0.50.
- 0.55.

## Pre-Registered Selection Rule

1. Require positive validation lift over the train-majority baseline.
2. Require validation accuracy within 0.75 percentage points of the best validation candidate.
3. Require 30/30 ticker coverage.
4. Require full row coverage.
5. Prefer h=40 if within tolerance.
6. Prefer simpler model: L2 Logistic, then Balanced L2 Logistic, then ElasticNet, then other ridge-style logistic variants.
7. Prefer threshold closest to 0.50.
8. Prefer feature sets in this order: `regime_feature_v2`, `feature_set_C_closest`, `stock_lagged_rolling_plus_index_context`.
9. Final accuracy must not enter selection.

## Mandatory Audit Gates

Any result must pass:

1. Validation-only selection.
2. 30/30 ticker coverage.
3. Full coverage.
4. Leakage audit.
5. Overfit audit.
6. Validation-final mismatch audit.
7. Ticker/month/quarter/regime stability audit.
8. Lift over majority/simple baseline.
9. No confidence abstention.
10. No ticker subset.
11. No top-k substitution.

No data fetch, paper/DOCX generation, trading claim, profitability claim, investment-recommendation claim, or live-deployment claim is allowed.
