# VN30 Model Cooperation Transfer Audit Protocol

Paper title: Comparing Machine Learning Models for VN30 Equity Directional Forecasting: Evidence from Walk-Forward Testing and Vietnamese Market Index Benchmarks

## Purpose

This protocol defines a model-cooperation and transfer-audit experiment for the VN30 comprehensive benchmark. It is not a final-window tuning run and it is not an attempt to chase a prettier final accuracy number.

The goal is to test whether model families can support each other under validation-only selection through soft voting, model-as-feature meta-learning, validation-trained error correction, mixture-of-experts routing, calibration/threshold cooperation, and robust feature-selection cooperation.

No model family is privileged because of prior final-window accuracy. The prior 63.33% `bull_bear_sideway_router` descriptive row, the 61.63% current-main Logistic L2 row, the fair-tuning selected stacking result, and the KNN-support experiment are comparison context only. They must not determine model, feature, threshold, router, error-correction, calibration, or ensemble selection.

## Data And Split Rules

- Use existing local VN30 benchmark data and artifacts only.
- Do not fetch new market data.
- Do not change provider behavior.
- Require full VN30 30-stock coverage for headline rows.
- Do not use ticker subsets.
- Do not use confidence abstention.
- Do not use top-k/ranking as a substitute for overall directional accuracy.
- The final window is scoring-only.

## Cooperation Tracks

The experiment evaluates:

- validation-safe soft voting
- model-as-feature meta-models
- validation-only error correction
- mixture-of-experts routing
- calibration and threshold cooperation
- feature-selection cooperation

Each cooperation strategy must be fixed using validation-only evidence before final scoring. Base models are fit on train only. Meta-models and error-correction models may use validation predictions/diagnostics and validation labels, but never final labels. Ensemble weights, thresholds, calibration rules, routers, and feature sets must be selected without final-window scores.

## Selection Objectives

Selection is validation-only and uses:

1. `max_validation_accuracy`
2. `max_validation_balanced_accuracy`
3. `max_validation_lift_over_majority`
4. `max_validation_rolling_stability`
5. `max_validation_monthly_stability`
6. `max_validation_ticker_stability`
7. `min_validation_instability`
8. `balanced_transfer_score`

The balanced transfer score uses only validation metrics: validation accuracy, validation balanced accuracy, validation lift over majority, validation rolling 250/500/1000 means, validation monthly stability, validation ticker stability, and a validation instability penalty.

## Final Scoring And Claim Boundary

Final scoring is performed only after a cooperation strategy is fixed by validation-only rules. The final window must not be used for model, feature, threshold, horizon, ensemble, calibration, router, error-correction, or tuning selection.

The main result changes only if a cooperative candidate is validation-selected, full-coverage, leakage-audited, stability-audited, not high-overfit-risk, and claim-eligible. Descriptive final-window leaderboard rows may be reported separately, but they do not override validation-only selection.

No trading, profitability, investment recommendation, or live-deployment claim is made.
