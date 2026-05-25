# VN30 Hourly Dual-Track Model Comparison Protocol

## Purpose

This protocol separates two VN30 stock-only hourly evaluation tracks so their claims do not get mixed:

- Track A: canonical historical setup for apples-to-apples comparison against the locked 60.31% RF h60 baseline.
- Track B: current broader expanded-pipeline setup for comparison against the current RF h60 baseline around 56.12%.

The two tracks are not interchangeable.

## Track A - Canonical Historical Setup

- Locked RF h60 baseline: 60.31%.
- Historical final rows: 3,474.
- Goal: beat 60.31 apples-to-apples.
- Scope: same VN30 stock universe, target, final window, row construction, and closest feature-set C implementation.
- Historical feature-set C: stock lagged/rolling/technical features plus lagged market-index context.
- Historical split: train through 2023-12-31, validation in calendar 2024, final from 2025-01-01 through the available historical cache end.
- Track A claims are valid only under this canonical setup.

## Track B - Current Broader Pipeline

- Current RF h60 baseline: 56.12%.
- Current final rows: 8,637.
- Goal: beat 56.12 under the broader current split.
- Scope: current expanded-model pipeline row construction, current final split, and current feature implementation.
- Track B must not be compared directly to 60.31 as an apples-to-apples claim.

## Shared Rules

- VN30 stock-only hourly.
- 30/30 January 2025 VN30 tickers.
- Full coverage.
- Main metric: pooled overall directional accuracy.
- No confidence abstention.
- No ticker subset.
- No top-k/ranking substitution.
- No index-only result as stock result.
- No daily result as hourly result.
- Validation-only selection.
- Final scoring-only.
- No final-label model selection.
- No data fetch.
- No paper or DOCX generation.
- No trading, profitability, investment-recommendation, or live-deployment claim.

## Candidate Families

Both tracks may compare:

- Random Forest.
- ExtraTrees.
- Decision Tree / CART.
- XGBoost.
- LightGBM.
- Logistic Regression if practical.
- Validation-weighted soft voting.
- Stacking only with validation or OOF predictions.

Horizons:

- h=40.
- h=60.
- h=80.
- h=100.
- h=120.

## Claim Boundary

Safe claims:

- Track A: only claims under the canonical historical setup.
- Track B: only claims under the current broader-pipeline setup.
- Both: must include baseline, row count, final coverage, and validation-selection status.

Unsafe claims:

- Treating Track B as an apples-to-apples improvement over 60.31.
- Treating confidence-filtered, ticker-subset, top-k, index-only, or daily results as the main stock-only hourly overall accuracy.
- Making trading, profitability, investment-recommendation, or live-deployment claims.

