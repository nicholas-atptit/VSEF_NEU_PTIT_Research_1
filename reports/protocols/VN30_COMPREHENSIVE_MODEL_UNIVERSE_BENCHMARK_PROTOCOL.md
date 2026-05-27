# VN30 Comprehensive Model Universe Benchmark Protocol

## Purpose

This benchmark is a comprehensive model-universe comparison for the paper:

Comparing Machine Learning Models for VN30 Equity Directional Forecasting: Evidence from Walk-Forward Testing and Vietnamese Market Index Benchmarks

The objective is broad model coverage, not final-window tuning. The run uses existing local data and artifacts only, keeps provider behavior unchanged, and evaluates feasible model families under the legacy-compatible VN30 walk-forward row rules.

## Research Gap and Evidence Positioning

Existing financial forecasting studies show that machine learning can be applied to equity prediction, but the resulting evidence is difficult to compare when studies differ in target definition, model scope, forecast horizon, validation design, and reporting granularity. A result from one algorithm, one index, one forecast horizon, or one final testing window does not establish general model-family behavior in stock-level VN30 directional forecasting. This paper therefore treats comparability, validation discipline, and stock-level diagnostic granularity as the central evidence gap rather than treating a single high final-window score as sufficient evidence.

The gap matters because narrow evidence can overstate model superiority. A model family may look strong at h40 but weaker at h20, h60, or h80; ensemble or cooperation mechanisms may improve one diagnostic slice while failing to transfer consistently; and index-level results can hide stock-level heterogeneity. The paper is therefore positioned as an evidence-based benchmark analysis with bounded interpretation, not as a leaderboard, trading system, profitability study, investment recommendation, or live-deployment claim.

## Benchmark Response

This paper addresses that gap with a controlled comprehensive VN30 stock-level model-universe benchmark across h20, h40, h60, and h80. The benchmark preserves full 30-stock headline coverage, uses a common directional target and horizon structure, selects models and thresholds from validation evidence only, and treats the final window as scoring-only. The model universe includes naive baselines, technical rules, linear and generalized linear models, SVM and kernel models, KNN and distance-based models, probabilistic classifiers, tree models, boosting models including CatBoost, neural and deep models, ensembles and stacking, calibration variants, regime-aware models, and statistical direction models; GARCH is reported only as a diagnostic, not as a headline direction classifier. Headline interpretation excludes ticker subsets, confidence abstention, and top-k ranking substitutions. Where Vietnamese market index evidence appears, it is used as market-context evidence only and does not replace stock-level VN30 results.

## Contribution Framing

The first contribution is an evidence contribution: broad stock-level VN30 model-universe evidence under a common directional target, horizon structure, and diagnostic framework. The second contribution is methodological: validation-only selection, final scoring-only evaluation, explicit claim boundaries, and model-family comparison rather than leaderboard selection. The third contribution is diagnostic: horizon dependence, validation-final transfer, model-family heterogeneity, and regime, cooperation, and KNN-support diagnostics. These contributions support bounded interpretation and do not establish trading readiness.

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

The fixed current h40 paper result remains Logistic L2 / baseline_C_closest / h40 / validation-selected threshold 0.55 / final accuracy 61.63% / full 30-stock coverage. The bull_bear_sideway_router h40 fixed 0.50 final accuracy 63.33% row is descriptive final-window context only and is not claim-eligible. The soft-voting final accuracy 62.00% cooperation row is descriptive context only and is not claim-eligible. Rows with high validation and poor final transfer, including stacking_xgboost_meta diagnostics, are interpreted as validation-final transfer or overfit failures rather than as main results. KNN-support rows are diagnostic support experiments and do not replace the main claim. GARCH is a volatility diagnostic only and not a direct headline direction classifier. Market-index evidence is market-context evidence and cannot substitute for stock-level VN30 evidence.
