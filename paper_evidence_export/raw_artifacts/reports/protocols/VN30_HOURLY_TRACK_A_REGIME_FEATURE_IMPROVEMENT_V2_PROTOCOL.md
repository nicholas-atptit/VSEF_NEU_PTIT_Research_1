# VN30 Hourly Track A Regime Feature Improvement V2 Protocol

## Scope

- Main target: VN30 stock-only hourly overall directional accuracy.
- Setup: Track A canonical-like setup.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage.
- Locked best baseline: Logistic Regression h40 = 60.43%.
- Historical RF h60 reference: 60.31%.
- Aspirational target: 65%.

## Direction

Stacking all algorithms failed and should not be repeated in this protocol. Expanded model pooling, joint stock+index panels, and broad all-algorithm stacking are treated as negative/context evidence.

The new direction is regime-aware and feature improvement:

- Add leakage-safe lagged market and index trend features.
- Add stock-minus-market, rolling volatility, volatility regime, bull/bear/sideway proxy, volume shock, range shock, session/hour, ticker encoding, and existing static sector/group encoding if available.
- Test global and regime-aware models under Track A only.

## Selection And Scoring

- Selection must be validation-only.
- Final window is scoring-only.
- Final accuracy must not be used for model or threshold selection.
- No confidence abstention.
- No ticker subset.
- No top-k/ranking substitution.
- No data fetch.
- No paper/DOCX generation.
- No trading, profitability, investment-recommendation, or live-deployment claim.

## Leakage Controls

Forbidden features:

- Future return.
- Future regime.
- Final-label-derived feature.
- Final-period manual filters.

All regime proxies must be based on historical or lagged values available at prediction time.
