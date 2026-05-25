# VN30 Hourly Expanded Model Pool + Stacking Protocol

## Objective

This protocol defines a controlled VN30 stock-only hourly benchmark extension for screening a broader model pool and then evaluating validation-selected ensembles or stacking.

## Locked Baseline And Targets

- Locked baseline: Random Forest h=60 approximately 60.31% stock-only hourly overall directional accuracy.
- Main metric: stock-only overall directional accuracy pooled across all eligible VN30 stock rows.
- Main target: beat 60.31%.
- Aspirational target: 65%.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage requirement: full coverage, with no confidence abstention and no ticker subset.

## Selection Rules

- Use all candidate models only for screening.
- Candidate screening selection must use validation metrics only.
- Final ensemble/stacking must use validation-selected models only.
- Final window is scoring-only.
- Final accuracy must not be used for model, horizon, feature-set, ensemble, or meta-model selection.
- Stacking must train the meta-model on out-of-sample or validation prediction rows only, never on final labels.

## Forbidden Substitutions

- No confidence abstention for the main target.
- No ticker subset as a full-universe result.
- No top-k/ranking substitution for overall directional accuracy.
- No index-only result as a stock result.
- No daily result as an hourly result.
- No trading, profitability, investment-recommendation, or live-deployment claim.

## Candidate Model Pool

Screening candidates:

- Random Forest.
- ExtraTrees if available.
- Decision Tree / CART if available.
- XGBoost.
- LightGBM.
- Logistic Regression if practical.
- SARIMAX / ETS only as diagnostic baselines if practical.
- LSTM / BiLSTM only if the existing pipeline is stable and not heavy.

Baselines:

- Majority class.
- Always-up.
- Previous direction.
- Moving average.
- Random seeded direction.

Ensemble and stacking candidates:

- Majority vote.
- Soft vote equal weight.
- Soft vote validation-weighted.
- Stacking with OOF or validation predictions.

## Horizons And Feature Families

Horizons:

- h=40.
- h=60.
- h=80.
- h=100.
- h=120.

Feature families:

- Existing stock features.
- Lagged and rolling stock features.
- Optional lagged index context features only if already available locally and leakage-safe.
- No future values.

## Claim Boundary

Safe claims require a validation-selected, full-coverage, stock-only result with baseline comparison. Because prior final windows have been inspected repeatedly, any improvement from this branch should remain exploratory unless later verified on future blind data.

