# Comparing Machine Learning Models for VN30 Equity Directional Forecasting: Evidence from Walk-Forward Testing and Vietnamese Market Index Benchmarks

## Abstract

Machine-learning evidence in financial forecasting is often difficult to compare because studies differ in target definition, model scope, forecast horizon, validation design, and reporting granularity. This paper addresses that comparability problem through a controlled benchmark for VN30 stock-level directional forecasting. The benchmark evaluates a comprehensive model universe under common h20, h40, h60, and h80 horizon definitions, full 30-stock headline coverage, validation-only model and threshold selection, and a final window used only for scoring. The model universe records 75 planned and attempted variants, 74 run variants, 0 failed variants, 0 skipped variants, and 1 not-recommended variant, with 1,868 candidate rows planned or attempted and 1,864 successful result rows. CatBoost was run, while GARCH is retained only as a volatility diagnostic and is not used as a main directional classifier. The bounded h40 result is Logistic L2 using the C-closest reference feature set, with validation-selected threshold 0.55 and final accuracy 61.63% under full 30-stock coverage. Higher final-window descriptive evidence, including the bull_bear_sideway_router h40 fixed-threshold result at 63.33% and soft-voting cooperation evidence near 62.00%, is not claim-eligible because it does not replace the validation-selected claim boundary. Stacking_xgboost_meta rows with high validation accuracy and poor final transfer are interpreted as transfer-failure or overfit-risk evidence. The paper provides bounded evidence about model-family behavior, not trading readiness, profitability, investment advice, final65, live deployment, or generalization beyond the reported VN30 evidence.

## Introduction

Machine learning has become a common tool for financial forecasting, including directional classification, return prediction, volatility modeling, and hybrid decision frameworks [citation needed]. Despite this broad use, empirical findings are often difficult to compare. Studies may use different target variables, different universes of stocks or indices, different forecast horizons, different validation rules, and different reporting levels. A reported result can therefore reflect the design of the experiment as much as the behavior of the model family.

This comparability problem is especially important for VN30 directional forecasting. VN30 index movement can provide useful information about the market environment, but index evidence cannot substitute for stock-level evidence. A model that performs well on an aggregate index, or in one final testing window, does not automatically establish that the same model family behaves consistently across individual VN30 constituents. Stock-level evidence must preserve ticker coverage, horizon definition, validation discipline, and final-window interpretation.

The final scoring window also requires careful treatment. A high final-window score can be descriptively useful, but it can become misleading if it is used to choose the model after observing final-window outcomes. This paper therefore separates validation-selected evidence from final-window diagnostic context. Model and threshold choices are made using validation evidence only; final-window rows are used for scoring and transfer interpretation only.

The objective of the paper is to provide bounded evidence about model-family behavior in VN30 stock-level directional forecasting. The paper compares a broad model universe under common design rules, evaluates horizon dependence across h20, h40, h60, and h80, and reports cooperation, transfer, KNN-support, and Vietnamese market-index diagnostics without using those diagnostics to replace the main claim. The remainder of the paper is organized as follows. Section 2 reviews related research areas requiring formal citation support. Section 3 states the research gap and contributions. Sections 4 through 8 describe the benchmark design, model universe, evaluation protocol, and empirical result boundary. Sections 9 through 13 report horizon, model-universe, cooperation, KNN-support, and market-context diagnostics. Sections 14 through 16 state the claim boundary, conclusion, citation TODO list, and insertion plan for figures and tables.

## Literature Review

### Machine learning for financial forecasting [citation needed]

Financial forecasting research has applied linear models, tree-based methods, boosting algorithms, support-vector methods, neural networks, and ensemble approaches to equity prediction problems [citation needed]. These studies show that machine learning can be applied to financial time series, but reported performance is sensitive to feature construction, target definition, horizon choice, sample period, market regime, and validation design [citation needed]. For that reason, this paper does not treat the existence of machine-learning applications as sufficient evidence of stock-level VN30 model-family behavior. Instead, it requires controlled comparison under common benchmark rules.

### Time-series validation and walk-forward testing [citation needed]

Time-series forecasting requires validation designs that respect temporal ordering [citation needed]. Walk-forward testing and related split designs are commonly used to reduce the risk of look-ahead bias and to distinguish model selection from out-of-sample scoring [citation needed]. This paper follows that principle by separating training, validation, and final scoring roles. The validation window is used for model and threshold selection, while the final window is used only for scoring and diagnostic interpretation.

### Model-family comparison and ensemble diagnostics [citation needed]

Model-family comparison is important because performance can vary across linear models, kernel methods, distance-based models, probabilistic classifiers, tree-based methods, boosting approaches, neural models, and ensembles [citation needed]. Ensemble and stacking methods can improve prediction under some conditions, but they may also introduce transfer risk when validation behavior does not persist into the final scoring window [citation needed]. This paper therefore reports ensemble and cooperation evidence as diagnostic evidence unless it satisfies the validation-only claim boundary.

### Regime dependence and market-context interpretation [citation needed]

Financial time series often exhibit regime dependence, including changes in trend, volatility, liquidity, and cross-sectional behavior [citation needed]. Regime-aware models and market-context diagnostics can help interpret performance variation, but they require careful claim boundaries. This paper treats regime-aware rows and Vietnamese market-index evidence as contextual and diagnostic unless they satisfy the same validation-only rules used for the main result.

### Vietnamese equity-market context [citation needed]

Vietnamese equity-market structure, index composition, trading constraints, liquidity variation, and market cycles may affect model behavior [citation needed]. VN30 evidence is therefore interpreted as bounded evidence for the reported VN30 stock-level design. The paper does not claim that the evidence generalizes to all Vietnamese equities, other market universes, other sampling frequencies, or future deployment settings.

## Research Gap and Contribution

The research gap is an evidence comparability gap. Existing forecasting studies can show that machine learning is applicable to equity prediction, but such studies are hard to compare when they differ in target definition, model scope, forecast horizon, validation design, and reporting granularity. A result from one model, one index, one horizon, or one final testing window cannot establish model-family behavior in stock-level VN30 directional forecasting.

This paper addresses that gap by constructing a controlled benchmark for stock-level VN30 evidence. The benchmark uses common directional targets, common horizon definitions, full 30-stock headline coverage, validation-only model and threshold selection, and a final scoring-only window. The design makes the comparison about model-family behavior rather than about post-hoc final-window selection.

The paper makes three contributions. First, it provides an evidence contribution: broad stock-level VN30 model-universe evidence under common h20, h40, h60, and h80 horizons. Second, it provides a methodological contribution: a validation-only selection boundary, final scoring-only interpretation, explicit claim-eligibility criteria, and model-family comparison rather than final-window ranking. Third, it provides a diagnostic contribution: horizon-dependence evidence, validation-final transfer analysis, model-family heterogeneity, regime-aware diagnostics, cooperation diagnostics, and KNN-support diagnostics.

These contributions are intentionally bounded. They support interpretation of the reported VN30 benchmark and do not establish trading readiness, profitability, investment advice, final65, live deployment, or generalization beyond the reported evidence.

## Data and Benchmark Design

The benchmark is organized around VN30 stock-level evidence. Headline evidence requires full 30-stock coverage so that a high result cannot be produced by a ticker subset. Market-index information is treated as market context only. It can help interpret broad conditions during the sample period, but it cannot replace stock-level VN30 evidence and cannot independently justify a claim about constituent-level directional forecasting.

The benchmark evaluates four forecast horizons: h20, h40, h60, and h80. These horizons allow the paper to examine whether model-family behavior is horizon-dependent rather than assuming that evidence at one horizon transfers automatically to others. The h40 result is the main bounded paper result, while the other horizons provide diagnostic evidence about robustness and transfer.

At a high level, the design separates training, validation, and final scoring roles. Training data are used to fit models and preprocessing. Validation data are used for model and threshold selection. The final window is used only for scoring. This distinction is central to the claim boundary because final-window evidence is informative as diagnostic context but is not used to choose the model.

The model-universe run records 75 model variants planned and attempted, 74 variants run, 0 failed variants, 0 skipped variants, and 1 not-recommended variant. It also records 1,868 candidate rows planned or attempted and 1,864 successful result rows. CatBoost status is run. GARCH diagnostic status is not_recommended_with_reason. GARCH used as a main directional classifier is no.

## Model Universe

The model universe is intentionally broad. It includes naive reference rules that provide simple directional comparison references, technical reference rules that encode common rule-based market signals, linear and generalized linear models, SVM and kernel models, KNN and distance-based models, probabilistic classifiers, tree-based models, boosting models including CatBoost, neural and deep models, ensembles and stacking, calibration variants, regime-aware models, and statistical direction models.

This range is important because a narrow model set could overstate or understate the behavior of a particular family. Linear models can offer stable and interpretable decision boundaries; kernel and SVM methods can capture nonlinear separation; KNN and distance-based methods can test similarity structure; probabilistic classifiers can provide simple distributional comparison points; tree-based and boosting models can capture nonlinear interactions; neural and deep models can test representation capacity; ensembles and stacking can test cooperation; calibration variants can assess score transformation; regime-aware models can test conditional behavior; and statistical direction models can connect the benchmark to traditional time-series forecasting.

GARCH is treated differently from the direct direction classifiers. It is retained as a volatility diagnostic only because the reported benchmark does not use it as a main directional classifier. This distinction prevents a volatility-model diagnostic from being interpreted as a headline stock-direction result.

## Evaluation Protocol

The evaluation protocol is built around validation-only selection. Model choice, threshold choice, and claim eligibility are evaluated using validation evidence, while the final window is scoring-only. This design reduces the risk that final-window performance becomes a hidden selection criterion.

Headline accuracy does not use ticker subsets, confidence abstention, or top-k/ranking substitution. A result must preserve full 30-stock coverage to be considered for the headline claim. This rule keeps the evidence aligned with stock-level VN30 directional forecasting rather than a narrower filtered task.

Claim eligibility also depends on transfer behavior. A validation-selected row with high validation accuracy can still fail the claim boundary if final performance collapses or if overfit-risk diagnostics are high. Such rows are not ignored; they are reported as diagnostic evidence about validation-final transfer. This treatment is especially important for stacking_xgboost_meta rows, where high validation accuracy and poor final transfer indicate transfer-failure or overfit-risk evidence rather than a stronger main result.

The final scoring window can still provide valuable information. It identifies descriptive rows, tests transfer, and helps reveal model-family heterogeneity. However, descriptive final-window rows cannot replace the validation-selected claim unless they satisfy the same claim boundary.

## Empirical Results

The bounded h40 result is Logistic L2 using the C-closest reference feature set, with validation-selected threshold 0.55 and final accuracy 61.63% under full 30-stock coverage. This result remains the main paper claim because it is the validation-selected h40 result under the stated claim boundary.

The bull_bear_sideway_router h40 fixed-threshold result reaches 63.33% final accuracy, but it is descriptive final-window context only and is not claim-eligible. It is useful for understanding how regime-aware routing behaves in the final scoring window, but it does not replace the bounded h40 Logistic L2 result because it does not satisfy the same validation-only claim boundary.

The cooperation diagnostics include soft-voting evidence near 62.00% final accuracy. This row is also descriptive context only. It indicates that cooperation mechanisms can be informative, but it does not replace the main result because the final score is not sufficient for claim promotion.

The stacking_xgboost_meta evidence illustrates the importance of validation-final transfer. Rows with high validation accuracy but poor final accuracy are interpreted as transfer-failure or overfit-risk evidence. This interpretation is central to the paper: a high validation result is not enough if the final scoring window does not support stable transfer under the claim boundary.

## Full-Horizon Diagnostics

The full-horizon diagnostics evaluate h20, h40, h60, and h80 under common row rules. Final row counts are 4,674 for h20, 4,074 for h40, 3,474 for h60, and 2,874 for h80. Ticker coverage is 30/30 for all four horizons.

These diagnostics show why a single horizon cannot establish general model-family behavior. A model family that appears strong at h40 may be weaker at h20, h60, or h80. Conversely, a model that is informative at another horizon may not support the h40 claim. The horizon evidence is therefore interpreted as diagnostic evidence about robustness and transfer, not as a horizon-selection mechanism.

The h40 main result remains separate from the full-horizon diagnostic rows. This separation prevents the diagnostic horizon analysis from changing the claim after observing final-window outcomes.

## Fair Exhaustive Model-Universe Diagnostics

The fair exhaustive model-universe diagnostics expand model-family coverage under the same validation-only and final-scoring-only boundary. They help assess whether performance differences reflect broad model-family behavior or isolated final-window outcomes.

Some rows can exceed 61.63% in the final scoring window, but such rows are not automatically promoted. Promotion would require satisfaction of the claim boundary, including validation-only selection, full stock coverage, and acceptable transfer diagnostics. Final-window evidence above the bounded h40 result is therefore descriptive unless it satisfies those criteria.

The fair diagnostics also show model-family heterogeneity. Different families can vary in validation accuracy, final accuracy, stability, and transfer behavior. This heterogeneity supports the paper's central framing: model-family comparison requires controlled design and diagnostic reporting, not a single final-window ranking.

## Cooperation and Transfer Diagnostics

The cooperation diagnostics evaluate soft voting, model-as-feature designs, error correction, mixture routing, calibration cooperation, and feature-selection cooperation. These mechanisms are analyzed as diagnostic tools for transfer behavior. They can show whether combining model information improves stability or whether it primarily changes final-window rankings.

The soft-voting result near 62.00% final accuracy is descriptive cooperation context only. It is informative because it suggests that cooperation can produce stronger final-window behavior in some settings, but it is not claim-eligible and does not replace the bounded h40 Logistic L2 result.

Model-as-feature evidence also illustrates transfer risk. High validation accuracy can fail to transfer into the final scoring window. In this paper, such behavior is interpreted as evidence of overfit risk or validation-final mismatch rather than as evidence of a stronger main result.

## KNN-Support Diagnostics

KNN-support evidence is treated as auxiliary diagnostic evidence. It evaluates whether similarity-based information can support other model families while preserving validation-only selection and final scoring-only interpretation.

The KNN-support diagnostics do not replace the main result. They are useful for studying whether distance-based structure contributes to transfer behavior or ticker-level robustness, but the reported evidence does not promote KNN-support to the headline claim. The interpretation remains bounded to the tested h40 support setting and does not assert cross-horizon robustness beyond the reported diagnostics.

## Vietnamese Market Index Context

Vietnamese market-index evidence is used as contextual evidence. It can help interpret broad market conditions around the VN30 stock-level benchmark, including market-wide movement and possible regime context. However, index evidence cannot substitute for stock-level VN30 evidence. The paper therefore does not use index-context results to replace ticker-level directional evidence.

This distinction is important because an index aggregates market behavior. Individual VN30 constituents can differ in response to market movement, sector conditions, liquidity, and idiosyncratic information. Stock-level directional forecasting therefore requires stock-level validation and final scoring evidence.

## Claim Boundary and Limitations

The paper makes a bounded benchmark claim only. It does not make a trading readiness claim, profitability claim, investment-advice claim, final65 claim, live-deployment claim, or generalization claim beyond the reported VN30 evidence. It also does not allow market-index evidence to substitute for stock-level VN30 evidence.

Several limitations follow from the design. First, final-window descriptive rows remain contextual even when their final accuracy exceeds the bounded h40 result. Second, high validation accuracy can be misleading when validation-final transfer is poor, as shown by stacking_xgboost_meta diagnostics. Third, horizon dependence means that h40 evidence does not automatically generalize to h20, h60, or h80. Fourth, regime-aware and cooperation diagnostics are informative but require the same validation discipline before they can support a headline claim. Fifth, future blind validation is needed before extending the interpretation beyond the current benchmark evidence.

These limitations are not incidental. They are part of the paper's contribution because they define what the evidence can and cannot support.

## Conclusion

This paper provides a controlled VN30 stock-level model-universe benchmark for directional forecasting. It addresses an evidence comparability gap by evaluating broad model families under common horizon definitions, full 30-stock headline coverage, validation-only model and threshold selection, and final scoring-only interpretation.

The bounded h40 result is Logistic L2 using the C-closest reference feature set, with validation-selected threshold 0.55 and final accuracy 61.63% under full 30-stock coverage. Descriptive final-window evidence, including the bull_bear_sideway_router h40 fixed-threshold result at 63.33% and soft-voting cooperation evidence near 62.00%, remains contextual and not claim-eligible. Stacking_xgboost_meta rows with high validation and poor final transfer are interpreted as transfer-failure or overfit-risk evidence rather than main results.

Future work should focus on future-blind validation, cost-aware evaluation, broader market periods, stronger regime analysis, and additional tests of validation-final transfer. Those extensions would help determine whether the bounded evidence reported here persists under stricter prospective conditions.

## Citation TODO List

- Add citations for machine learning applications in equity forecasting.
- Add citations for time-series validation, walk-forward testing, and prevention of look-ahead bias.
- Add citations for model-family comparison in financial forecasting.
- Add citations for ensemble, stacking, and cooperation methods in predictive modeling.
- Add citations for regime dependence and market-context interpretation in financial markets.
- Add citations for Vietnamese equity-market structure and VN30 market context.
- Add citations for GARCH as a volatility modeling framework and its distinction from direct direction classification.

## Figure and Table Insertion Plan

- Table 1: Model-universe coverage, including planned, attempted, run, failed, skipped, and not-recommended variants.
- Table 2: Bounded h40 result, including Logistic L2, the C-closest reference feature set, h40, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage.
- Table 3: Descriptive final-window context, including the bull_bear_sideway_router 63.33% row and soft-voting evidence near 62.00%, with claim-eligibility labels.
- Table 4: Full-horizon diagnostics, including h20, h40, h60, and h80 row counts and ticker coverage.
- Table 5: Fair model-universe transfer-quality diagnostics, including validation-final transfer and model-family heterogeneity.
- Table 6: Cooperation diagnostics, including soft voting, model-as-feature, error correction, mixture routing, calibration cooperation, and feature-selection cooperation.
- Table 7: KNN-support diagnostics, including auxiliary similarity-based evidence and claim-boundary status.
- Table 8: Claim-boundary matrix, listing what is claimed and what is not claimed.
- Figure 1: Benchmark design, showing training, validation, and final scoring roles.
- Figure 2: Model-universe coverage across model families.
- Figure 3: Validation-final transfer diagnostics.
- Figure 4: Horizon diagnostics for h20, h40, h60, and h80.
- Figure 5: Vietnamese market-index context, clearly labeled as market-context evidence only.
