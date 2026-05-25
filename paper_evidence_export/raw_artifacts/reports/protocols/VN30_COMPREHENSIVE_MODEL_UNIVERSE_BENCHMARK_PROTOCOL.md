# VN30 Comprehensive Model Universe Benchmark Protocol

## Purpose

This benchmark is a comprehensive model-universe comparison for the paper:

Comparing Machine Learning Models for VN30 Equity Directional Forecasting: Evidence from Walk-Forward Testing and Vietnamese Market Index Benchmarks

The objective is broad model coverage, not final-window tuning. The run uses existing local data and artifacts only, keeps provider behavior unchanged, and evaluates feasible model families under the legacy-compatible VN30 walk-forward row rules.

## Model Universe

The benchmark covers naive baselines, technical rules, linear models, SVMs, distance-based models, probabilistic models, tree models, boosting models, neural/deep models, ensembles, calibration variants, regime-aware models, and traditional statistical models.

Heavy models should be run if technically feasible. SVM RBF, SVM Polynomial, KNN, Radius Neighbors, technical indicators, and feasible statistical direction models are mandatory attempts. Models may only be skipped for dependency, data-shape, implementation, or objective-compatibility reasons. Runtime risk alone is not a skip reason.

CatBoost is part of the boosting model family and is attempted when installed. The May 2026 exhaustive rerun installed the CatBoost and arch/GARCH dependencies before rerunning the benchmark. ARIMA, SARIMA, and ETS may be evaluated as direction models only by forecasting return or level and converting the forecast sign into up/down direction. GARCH is not a direct directional classifier; it is treated as a volatility diagnostic or future volatility feature source, not as a headline direction model.

## Evaluation Rules

The benchmark uses the legacy-compatible row rules:

- Train rows use feature timestamps up to 2023-12-31 23:59:59 with non-null horizon labels.
- Validation rows use feature timestamps from 2024-01-01 through 2024-12-31 23:59:59 with non-null horizon labels.
- Final rows use feature timestamps from 2025-01-01 onward with non-null horizon labels.
- Horizons are h20, h40, h60, and h80.
- Headline rows require full 30-stock VN30 coverage.

Feature families are baseline_C_closest, volatility_normalized, relative_strength, regime_context, and combined_context when feasible. Threshold policies are fixed 0.50 and validation-selected thresholds when classifier probabilities or scores are available. The threshold grid is 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, and 0.60.

Preprocessing must be fit on the train split only. StandardScaler is used for SVM, KNN, Radius Neighbors, Nearest Centroid, SGD, MLP, LDA/QDA, and similar scale-sensitive models where needed. Imputation is fit on train only when needed. No scaling or imputation leakage is allowed.

## Selection Boundary

Model and threshold selection must use validation only. The final window is scoring-only and must not be used for model, feature, threshold, horizon, ensemble, calibration, router, or dependency-related selection.

The current main h40 paper result remains:

- Logistic L2.
- baseline_C_closest.
- h40.
- Validation-selected threshold 0.55.
- Final accuracy 61.63%.
- Full 30-stock coverage.

The main h40 paper result changes only if a new model is validation-selected, has full 30-stock coverage, and passes the audit. Final-window accuracy alone is not sufficient.

## Deep Learning Rules

Deep learning models use time-safe sequence construction, no shuffle across time, and validation-only early stopping. Sequence lengths 16, 32, and 64 are attempted where feasible. A failed dependency, tensor shape, or implementation path must be recorded as failed_with_reason or skipped_with_reason, not silently omitted.

## Statistical Model Rules

ARIMA, SARIMA, and ETS direction rows may forecast return or level and convert the forecast sign to up/down direction. VAR direction rows are exploratory when feasible. GARCH is a volatility diagnostic only. It is written under statistical model diagnostics and is not claim eligible as a direct directional classifier unless a clearly defined ex-ante direction conversion rule is implemented before final scoring. The default benchmark role for GARCH is not claim eligible.

## Overfit-Risk Audit

Overfit cannot be ruled out absolutely. It is evaluated through validation-only model and threshold selection, validation-final gap, rolling 250/500/1000-row diagnostics, monthly and quarterly stability, ticker stability, regime-slice stability when available, and post-hoc final-leaderboard checks. Risk is classified as low, medium, or high.

## Claim Boundary

No trading, profitability, investment recommendation, or live-deployment claim is made. No ticker subset, confidence abstention, or top-k/ranking substitute is allowed for headline directional accuracy. Any skipped, failed, or not-recommended model must be listed with a concrete reason.
