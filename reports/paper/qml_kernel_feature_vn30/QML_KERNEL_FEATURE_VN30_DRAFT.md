# Drift-Aware Quantum Kernel Features for Market-Relative VN30 Stock Forecasting

## Abstract

This paper studies Quantum Machine Learning (QML) as a diagnostic representation layer for VN30 stock-level hourly forecasting. The target is market_relative_vn30 at horizon h40, not absolute direction, and the benchmark is governed by feature_timestamp and target_timestamp split discipline. A sequence of QML diagnostics first found a bounded-sample quantum-kernel signal, then showed that pure quantum-kernel classifiers weakened under larger samples because of class-balance drift, kernel concentration, and small-sample sensitivity. The final V8 diagnostic converts quantum kernels into features and combines them with relative-strength and market-context features in a validation-governed L2 Logistic meta-model with a robust validation-quantile threshold. The locked V8 candidate reached 60.56% validation accuracy and 64.44% final diagnostic accuracy, improving over the V7 QML-kernel-feature final result of 58.89% by +5.56 percentage points. It also beat same-target L2 Logistic, RBF SVM, and LightGBM small baselines in the recorded diagnostic comparison. These findings support QML-derived kernel features as a diagnostic representation method, not as a trading system, production forecasting system, investment recommendation, or replacement for the 61.61% absolute-direction classical champion.

## 1. Introduction

VN30 stock-level hourly forecasting is a difficult benchmark because the signal is time-varying, ticker-specific, and strongly affected by market context. A model that appears useful in one sample can fail when class balance, ticker mix, quarter composition, or market regime shifts. This paper therefore treats forecasting as a controlled diagnostic exercise rather than as a trading-system claim.

The target studied here is market_relative_vn30 at h40. This differs from absolute-direction forecasting. A market-relative target asks whether a stock outperforms or underperforms the VN30 market context over the forecast horizon. An absolute-direction target asks whether the stock return is positive. Because these targets answer different questions, the QML result in this paper is not treated as a replacement for the current absolute-direction classical champion.

QML was tested because quantum feature maps and quantum kernels may provide a nonlinear representation of compact relative-strength features. Early experiments found a bounded-sample quantum-kernel signal, but subsequent sample-scaling diagnostics showed that standalone quantum-kernel classifiers were unstable. The V8 experiment therefore focuses on quantum-kernel-derived features as a representation layer, followed by a classical meta-model selected under validation-only governance.

## 2. Related Work

Financial forecasting with classical machine learning has been widely studied, but reported results are difficult to compare when target definitions, validation schemes, stock universes, horizons, and reporting rules differ [citation needed]. Kernel methods such as support vector machines offer nonlinear decision boundaries and have been used in financial classification problems [citation needed]. Quantum kernel learning extends this idea by mapping classical inputs into quantum feature spaces and evaluating similarity through quantum kernels [citation needed].

The V8 design also connects to representation learning. Instead of relying on a pure QSVC decision rule, the quantum kernel is converted into derived features such as centroid similarity, kernel margin, local density, and eigenspace projections. These features are then used by a classical meta-model. This design is motivated by the observed instability of pure quantum-kernel classifiers under sample scaling and by the broader problem of drift and regime shift in financial time series [citation needed].

## 3. Data and Target Definition

The scope is VN30 stock hourly forecasting only. The target variant is market_relative_vn30 and the horizon is h40. VN30 index data may appear only as market-relative target context or lagged market-context information. This is not an index-as-stock experiment.

The split rule uses both feature_timestamp and target_timestamp. Training rows must remain in the training period, validation rows must have both timestamps inside the validation period, and final rows are scoring-only. All feature selection, scaling, kernel construction, and meta-model selection are train-only or validation-governed. No VN100 evidence is used.

## 4. Methodology

The QML track evolved through several diagnostic stages. Initial experiments used quantum-kernel classifiers with compact feature representations. Scaling experiments then showed that pure kernel classifiers weakened when sample sizes increased. V8 keeps the QML component but changes its role: it becomes a feature extractor.

The locked V8 candidate uses the relative_market_context_topk4 kernel-feature source with minmax_0_pi scaling. It constructs QML kernel-derived features and combines them with relative-strength and market-context features. The final meta-model is L2 Logistic. The decision threshold is selected by robust validation quantile, not by final performance.

The kernel-derived features include positive_centroid_similarity, negative_centroid_similarity, centroid_similarity_gap, kernel_margin_score, top_eigen_projection_1, top_eigen_projection_2, top_eigen_projection_3, kernel_local_density, and class_similarity_ratio. The most useful features identified in the V8 summary are positive_centroid_similarity, top_eigen_projection_1, and negative_centroid_similarity.

## 5. Experiment Evolution

### 5.1 QML v4 Bounded-Sample Kernel Discovery

QML v4 found a bounded-sample signal from a quantum_kernel_classifier on market_relative_vn30 at h40. The strong bounded-sample evidence motivated further testing but was not treated as claimable replacement evidence.

### 5.2 QML v5 Sample-Size Weakening

QML v5 replayed the promising design under larger sample sizes and same-target classical baselines. The signal weakened under medium and largest feasible samples, showing that the bounded-sample result did not transfer cleanly.

### 5.3 QML v6 Scaling Rescue Failure Diagnosis

QML v6 diagnosed scaling failure modes including class-balance drift, kernel concentration, small-sample overfit, and QSVC regularization sensitivity. Scaling variants, kernel variants, regularization changes, and QML-as-feature tests did not fully rescue the result.

### 5.4 QML v7 Hybrid and Kernel-Feature Partial Rescue

QML v7 introduced distribution matching, kernel health prefilters, hybrid kernels, normalization/shrinkage, regularization revisits, and kernel-feature meta-models. The best validation-governed QML-derived result used qml_kernel_features plus a calibrated logistic meta-model, reaching 61.11% validation accuracy and 58.89% final accuracy. This partly rescued representation quality but left validation-to-final drift unresolved.

### 5.5 QML v8 Drift-Aware Kernel-Feature Rescue

QML v8 focused on QML kernel features as a representation layer and added drift-aware classical meta-modeling. The locked candidate used an L2 Logistic meta-model with robust thresholding by validation quantile. It reached 60.56% validation accuracy and 64.44% final diagnostic accuracy.

## 6. Results

The locked V8 candidate is validation-governed and final-scored once. It reached 60.56% validation accuracy and 64.44% final diagnostic accuracy. Relative to the V7 final result of 58.89%, this is a +5.56 percentage-point improvement.

The drift audit indicates that drift remained present. Mean final PSI versus validation across kernel-feature audits was 1.5096 where finite. The validation positive ratio was 48.33%, while the final positive ratio was 35.56%. Ticker coverage was full 30-stock coverage in the selected distribution-matched sample.

The V8 candidate beat same-target baselines in the diagnostic comparison. Against L2 Logistic, it improved by +7.78 pp on validation and +3.33 pp on final. Against RBF SVM, it improved by +5.00 pp on validation and +25.00 pp on final. Against LightGBM small, it improved by +4.44 pp on validation and +1.67 pp on final. It also exceeded the best same-target classical final model by +0.56 pp.

The contextual absolute-direction classical champion remains L2 Logistic, feature_set_C_closest, h40, threshold 0.50, with 61.61% final accuracy and +10.90 pp lift. The V8 result is not a replacement for that champion because the target and scope differ.

## 7. Discussion

The strongest QML evidence now supports quantum kernels as a representation layer rather than as a standalone QSVC decision system. Pure quantum-kernel classifiers showed bounded-sample promise but weakened under scaling. V8 improved final transfer by extracting kernel-derived features and allowing a classical L2 Logistic meta-model to combine those features with relative-strength and market-context information.

The improvement does not remove governance constraints. The target is market_relative_vn30, not absolute_direction. The final window is scoring-only. Final-ranked rows remain exploratory_not_claimable. The locked candidate is best interpreted as a validation-governed diagnostic candidate requiring future-blind confirmation.

## 8. Claim Boundary and Governance

This paper is diagnostic-only. It makes no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or production readiness claim. It does not use VN100. It does not claim index-as-stock evidence. It does not claim that QML replaces the 61.61% absolute-direction classical champion.

Allowed interpretation is limited to this statement: under the recorded split-safe V8 diagnostic, QML-derived kernel features combined with relative-strength and market-context inputs and an L2 Logistic meta-model produced a validation-governed market_relative_vn30 h40 candidate with 60.56% validation accuracy and 64.44% final diagnostic accuracy. Stronger claims require future-blind confirmation.

## 9. Conclusion

The VN30 QML v8 diagnostic shows that quantum-kernel-derived features can provide a measurable market-relative forecasting representation improvement. Pure QML classifiers were unstable under scaling, but hybrid representation plus a classical decision layer produced the strongest observed QML path. The paper contribution is diagnostic and methodological: it documents how a QML signal weakened, how drift-aware representation rescue was tested, and why the result remains bounded by strict claim governance.
