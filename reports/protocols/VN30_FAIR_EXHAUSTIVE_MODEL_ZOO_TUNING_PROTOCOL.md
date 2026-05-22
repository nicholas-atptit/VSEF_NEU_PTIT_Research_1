# VN30 Fair Exhaustive Model-Zoo Tuning Protocol

Paper title: Comparing Machine Learning Models for VN30 Equity Directional Forecasting: Evidence from Walk-Forward Testing and Vietnamese Market Index Benchmarks

## Purpose

This protocol defines a fair full-model-zoo tuning run for the VN30 comprehensive benchmark. It is not a targeted tuning run, and it does not tune a single model family or any previously high-scoring descriptive final-window row in isolation.

The run compares all feasible model families under a common validation-only selection protocol and a documented tuning-budget design. No model family is privileged because of prior final-window performance. All feasible model families receive an explicit budget, and skipped or failed models must retain documented reasons.

## Prior Results Are Context Only

- The prior 63.33% `bull_bear_sideway_router` h40 fixed-threshold result is descriptive final-window context only. It must not influence tuning priority, model-family budget, feature choice, threshold choice, horizon choice, ensemble design, calibration choice, router choice, or claim selection.
- The prior 61.63% Logistic L2 `baseline_C_closest` h40 threshold 0.55 result is current-main context only. It must not prevent another model from being selected if that model is selected by validation-only rules, has full 30-stock coverage, and passes audit.
- No model is pre-claimed as the winner before the fair tuning run and audit are complete.

## Data And Split Rules

- Use existing local VN30 benchmark data and artifacts only.
- Do not fetch new market data.
- Do not change provider behavior.
- Require full VN30 30-stock coverage for headline rows.
- Do not use ticker subsets, confidence abstention, or top-k/ranking substitution for overall directional accuracy.
- The final window is scoring-only.

## Horizon Policy

- Primary full model-zoo tuning horizon: h40.
- Secondary h20, h60, and h80 diagnostics are run only for candidates selected by h40 validation objectives.
- Horizon selection must not use final-window scores.

## Feature Families

The fair tuning run may use:

- `baseline_C_closest`
- `volatility_normalized`
- `relative_strength`
- `regime_context`
- `combined_context`

Feature-family selection must be validation-only.

## Threshold Policy

Each feasible classifier-style candidate is evaluated with:

- fixed threshold 0.50
- validation-selected threshold from the grid: 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60

Threshold selection must use validation labels only. Final labels are never used for threshold selection.

## Selection Objectives

Candidates are selected by validation-only objectives:

- `max_validation_accuracy`
- `max_validation_balanced_accuracy`
- `max_validation_lift_over_majority`
- `max_validation_rolling_stability`
- `min_validation_instability`
- `max_validation_monthly_stability`
- `max_validation_ticker_stability`
- `balanced_robust_score`

The balanced robust score uses only validation metrics: validation accuracy, validation balanced accuracy, validation lift over majority, validation rolling 250/500/1000 stability, validation monthly stability, validation ticker stability, and a penalty for validation instability.

## Model Families

The fair tuning budget registry must cover:

1. `naive_baseline`
2. `technical_rules`
3. `linear_models`
4. `svm_and_kernel_models`
5. `distance_based_models`
6. `probabilistic_models`
7. `tree_models`
8. `boosting_models`
9. `neural_deep_models`
10. `ensemble_stacking_models`
11. `calibration_variants`
12. `regime_aware_models`
13. `statistical_models`

Naive baselines may use small or fixed budgets. Technical rules tune rule windows. Linear, SVM, tree, boosting, neural, ensemble, calibration, and regime-aware models receive non-trivial but controlled budgets. GARCH is volatility diagnostic only and is not a direct direction classifier.

## Audit Boundary

The audit must check:

- all model groups have documented budgets
- no model family is tuned only because of final-window performance
- no final-window selection
- no leakage
- no future regime labels
- scaler and imputer fitting occurs inside train-only pipelines or train-only transforms
- calibration is time-safe
- stacking meta-models use validation predictions only
- full 30-stock coverage for headline rows
- no ticker subset
- no confidence abstention
- no top-k substitution

Every selected candidate and every candidate beating 61.63% must receive overfit-risk diagnostics.

## Claim Boundary

A result may be claim eligible only if it is selected by validation-only rules, has full 30-stock coverage, is not diagnostic-only, and passes audit. Descriptive final-window leaderboard rows may be reported as descriptive only and must not override validation-only selection.

No trading, profitability, investment recommendation, or live-deployment claim is permitted.
