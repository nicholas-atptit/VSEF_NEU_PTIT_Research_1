# Comparing Machine Learning Models for VN30 Equity Directional Forecasting: Evidence from Walk-Forward Testing and Vietnamese Market Index Benchmarks

## Abstract

This manuscript examines VN30 stock-level directional forecasting as a controlled benchmark problem rather than as a trading-system claim. Existing financial forecasting evidence shows that machine learning can be applied to equity prediction, but comparisons are difficult when studies differ in target definition, model scope, forecast horizon, validation design, and reporting granularity. The paper addresses this evidence comparability gap through a comprehensive VN30 stock-level model-universe benchmark with full 30-stock headline coverage, h20/h40/h60/h80 horizons, validation-only model and threshold selection, and a final window used only for scoring. The bounded h40 result is Logistic L2 using the C-closest reference feature set, with validation-selected threshold 0.55 and final accuracy 61.63% under full 30-stock coverage. Higher final-window descriptive rows, including the bull_bear_sideway_router h40 fixed-threshold result at 63.33% and soft-voting cooperation evidence near 62.00%, are not promoted because they are not claim-eligible under the validation-only boundary. The analysis supports bounded benchmark interpretation and does not establish trading readiness, profitability, investment advice, final65, or live deployment.

## Introduction

Financial forecasting studies often demonstrate that machine learning methods can be applied to equity prediction, but the interpretation of reported performance depends heavily on research design. A model result is shaped by the target definition, stock universe, forecast horizon, validation split, reporting granularity, and treatment of final-window evidence. For VN30 directional forecasting, these design choices are especially important because stock-level behavior can differ from index-level movement and because horizon-specific behavior can change the interpretation of model-family performance.

This manuscript therefore treats the VN30 forecasting problem as an evidence-positioning problem. The objective is not to identify a trading system or to elevate a final-window ranking. The objective is to compare model families under common stock-level rules and to preserve a clear boundary between validation-selected evidence and descriptive final-window diagnostics.

## Research Gap and Evidence Positioning

Existing financial forecasting evidence is difficult to compare when studies differ in target definition, model scope, forecast horizon, validation design, and reporting granularity. A result from one model, one index, one horizon, or one final testing window cannot establish model-family behavior in stock-level VN30 directional forecasting. This paper therefore frames the problem as one of evidence comparability, validation discipline, and diagnostic granularity.

The gap matters because narrow evidence can overstate model superiority. A model family may appear strong at h40 but weaker at h20, h60, or h80. Ensemble or cooperation mechanisms may improve one diagnostic slice but fail to transfer consistently. Index-level evidence can provide market context, but it cannot substitute for stock-level VN30 evidence.

## Contributions and Claim Boundary

The first contribution is an evidence contribution: broad stock-level VN30 model-universe evidence under a common directional target, horizon structure, and diagnostic framework. The second contribution is methodological: validation-only selection, final scoring-only evaluation, an explicit claim boundary, and model-family comparison rather than final-window ranking. The third contribution is diagnostic: horizon dependence, validation-final transfer, model-family heterogeneity, and regime, cooperation, and KNN-support diagnostics.

The bounded h40 result is Logistic L2 using the C-closest reference feature set, with validation-selected threshold 0.55 and final accuracy 61.63% under full 30-stock coverage. The bull_bear_sideway_router h40 fixed-threshold result at 63.33% is descriptive final-window context only and is not claim-eligible. The soft-voting final accuracy near 62.00% is descriptive cooperation context only and is not claim-eligible. Stacking_xgboost_meta rows with high validation and poor final transfer are interpreted as transfer-failure or overfit-risk evidence, not as main results. KNN-support is diagnostic and does not replace the main claim. GARCH is a volatility diagnostic only and is not used as a direct headline direction classifier.

## Data and Benchmark Design

The benchmark uses VN30 stock-level evidence as the main object of analysis. Headline rows require full 30-stock coverage. Market-index evidence is used only as market-context evidence and cannot replace stock-level VN30 results. Headline accuracy does not use ticker subsets, confidence abstention, or top-k/ranking substitution.

The comprehensive model-universe benchmark records 75 model variants planned and attempted, 74 variants run, 0 failed variants, 0 skipped variants, and 1 not-recommended variant. Candidate rows planned and attempted total 1,868, with 1,864 successful result rows. CatBoost status is run. GARCH diagnostic status is not_recommended_with_reason. GARCH used as a main directional classifier is no.

## Model Universe

The model universe covers naive reference rules, technical reference rules, linear and generalized linear models, SVM and kernel models, KNN and distance-based models, probabilistic classifiers, tree-based models, boosting models including CatBoost, neural and deep models, ensembles and stacking, calibration variants, regime-aware models, and statistical direction models. GARCH is retained only as a volatility diagnostic because it is not a direct headline directional classifier under the current benchmark definition.

## Evaluation Protocol

The evaluation uses h20, h40, h60, and h80 horizons under a common directional target and horizon structure. Model and threshold selection are validation-only. The final window is scoring-only and is not used for model, feature, threshold, horizon, ensemble, calibration, router, or claim selection. This separation is required because final-window performance alone can make descriptive rows appear stronger than validation-selected rows without providing claim-eligible evidence.

## Empirical Results

The bounded h40 result is Logistic L2 using the C-closest reference feature set, with validation-selected threshold 0.55 and final accuracy 61.63% under full 30-stock coverage. This result remains the main paper claim unless a new result is validation-selected, full-coverage, audit-passed, and consistent with the claim boundary.

The descriptive final-window context includes the bull_bear_sideway_router h40 fixed-threshold result with final accuracy 63.33%, but that row is not claim-eligible. It is reported as final-window context only and does not replace the bounded h40 Logistic L2 result. Stacking_xgboost_meta rows with high validation accuracy and poor final transfer are treated as validation-final transfer failures rather than evidence of superior model-family behavior.

## Full-Horizon Diagnostics

The full-horizon diagnostics report h20, h40, h60, and h80 under common row rules. Final row counts are 4,674 for h20, 4,074 for h40, 3,474 for h60, and 2,874 for h80. Ticker coverage is 30/30 for all four horizons.

The h40 main paper result remains separate from full-horizon diagnostics. Horizon-specific best rows are diagnostic evidence for horizon dependence and do not create a separate horizon-selection claim.

## Fair Exhaustive Model-Universe Diagnostics

The fair exhaustive model-universe diagnostics broaden model-family coverage under the same validation-only and final-scoring-only boundary. They are used to evaluate model-family heterogeneity, validation-final transfer, and robustness diagnostics. They do not create an independent route for selecting a final-window result.

Rows that exceed the main h40 result only in the final scoring window are descriptive. High-validation stacking_xgboost_meta rows with poor final accuracy are discussed as transfer failures or overfit-risk evidence, not as claim-eligible improvements.

## Cooperation and Transfer Diagnostics

The cooperation diagnostics evaluate mechanisms such as soft voting, model-as-feature designs, error correction, mixture routing, calibration cooperation, and feature-selection cooperation. These diagnostics test whether cooperation improves transfer behavior or primarily changes final-window rankings.

The soft-voting result near 62.00% final accuracy is descriptive cooperation context only. It is not promoted because it does not replace the validation-selected h40 main paper result under the claim boundary.

## KNN-Support Diagnostics

KNN-support evidence is diagnostic. It tests whether similarity-based information can support other model families while preserving validation-only selection and final scoring-only evaluation. KNN-support does not replace the h40 Logistic L2 main claim and is not treated as a standalone headline claim.

## Vietnamese Market Index Context

Vietnamese market-index evidence is used only to describe market context around the stock-level VN30 benchmark. It can help interpret broad market conditions, but it cannot substitute for stock-level VN30 evidence, cannot create a claim about individual-stock directional forecasting, and cannot override the validation-only claim boundary.

## Claim Boundary and Limitations

The paper supports bounded benchmark evidence for stock-level VN30 directional forecasting under the reported design. It does not prove general machine-learning forecasting ability for all Vietnamese equities, all forecast horizons, or all market regimes. It does not use VN100 evidence to support VN30 claims. It does not make trading readiness, profitability, investment advice, final65, live-deployment, or generalization claims beyond the reported VN30 evidence.

The final window is scoring-only. Descriptive final-window rows, including bull_bear_sideway_router at 63.33% and soft-voting evidence near 62.00%, remain contextual. Validation-selected rows with high validation accuracy but poor final transfer, including stacking_xgboost_meta examples, are treated as transfer-failure or overfit-risk evidence.

## Conclusion

This manuscript positions the VN30 model-universe benchmark as controlled evidence rather than as a trading system or final-window ranking exercise. The bounded h40 Logistic L2 result remains constrained by validation-only selection and full 30-stock coverage. The broader model-universe, horizon, cooperation, KNN-support, and Vietnamese market-index diagnostics strengthen the academic framing by showing where model-family evidence transfers, where it weakens, and why stock-level VN30 evidence cannot be replaced by narrower index-level or final-window-only results.

## Notes on figures and tables to insert later

- Insert a model-universe coverage table summarizing planned, attempted, run, failed, skipped, and not-recommended variants.
- Insert the validation-selected h40 claim table with the bounded Logistic L2 result.
- Insert descriptive final-window context tables separately from claim-eligible results.
- Insert full-horizon row-count and best-by-horizon diagnostic tables.
- Insert fair model-universe transfer-quality diagnostics.
- Insert cooperation and soft-voting diagnostics with claim-eligibility labels.
- Insert KNN-support diagnostic tables.
- Insert Vietnamese market-index context figures only as market-context evidence, not as stock-level substitutes.
