# VN30 Hourly True Stacking All Algorithms Protocol

## Scope

- Main target: VN30 stock-only hourly overall directional accuracy.
- Setup: Track A canonical-like setup.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage.
- Feature set: `feature_set_C_closest`, with only Track A-compatible leakage-safe features.
- Historical RF h60 reference: 60.31%.
- Prior Track A Logistic Regression h40 reference: 60.43%.
- Main targets: beat 60.31% and beat 60.43%.
- Aspirational target: 65%.

## Stacking Definition

This is true stacking across all available candidate algorithms. Base models must produce validation/OOF predictions and final predictions. Meta-models must train only on validation/OOF predictions. Final labels must not be used for training or model selection.

Accuracy averaging is not stacking. Stacking uses base-model predictions or probabilities as meta-model inputs. Final-window labels are scoring-only.

## Base Algorithms

- Random Forest.
- ExtraTrees.
- Decision Tree / CART.
- XGBoost.
- LightGBM.
- Logistic Regression.
- Baseline signals if useful: previous direction, moving average, majority/always-up.

## Meta-Model Candidates

- Logistic Regression meta-model.
- Ridge/L2 Logistic meta-model.
- LightGBM shallow meta-model.
- XGBoost shallow meta-model if safe.
- Validation-weighted soft voting is included as an ensemble comparator, not as a stacking substitute.

## Controls

- No confidence abstention.
- No ticker subset.
- No top-k/ranking substitution.
- No final-label selection.
- No data fetch.
- No paper/DOCX generation.
- No trading, profitability, investment-recommendation, or live-deployment claim.

## Selection Rule

Stacking methods are selected by validation accuracy only, with validation baseline delta as a tie-breaker. Final accuracy is reported only after selection.
