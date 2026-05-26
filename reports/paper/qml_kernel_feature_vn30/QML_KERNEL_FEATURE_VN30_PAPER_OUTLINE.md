# Drift-Aware Quantum Kernel Features for Market-Relative VN30 Stock Forecasting

## Core Thesis

A split-safe QML diagnostic pipeline initially found a bounded-sample quantum-kernel signal for market-relative VN30 forecasting. Pure quantum-kernel classifiers weakened under sample scaling due to class-balance drift, kernel concentration, and small-sample sensitivity. A drift-aware rescue using quantum-kernel-derived features, relative-strength and market-context features, and an L2 Logistic meta-model improved final diagnostic accuracy from 58.89% in V7 to 64.44% in V8. The V8 model beat same-target L2 Logistic, RBF SVM, and LightGBM baselines, supporting QML as a diagnostic representation layer rather than a standalone trading or production forecasting system.

## Paper Structure

1. Title
   - Drift-Aware Quantum Kernel Features for Market-Relative VN30 Stock Forecasting

2. Abstract
   - VN30 hourly market-relative forecasting.
   - QML kernel-derived representation features.
   - Validation-governed drift-aware meta-model.
   - V8 validation accuracy: 60.56%.
   - V8 final diagnostic accuracy: 64.44%.
   - Improvement over V7 final diagnostic accuracy: +5.56 pp.
   - Same-target comparisons versus L2 Logistic, RBF SVM, and LightGBM small.
   - Diagnostic-only claim boundary.

3. Introduction
   - VN30 stock-level hourly forecasting is evaluated as a controlled diagnostic problem.
   - The target is market_relative_vn30, not absolute directional movement.
   - Absolute-direction classical champion evidence is contextual only because target and scope differ.
   - QML was tested as a representation mechanism because kernel geometry may expose nonlinear relative-strength structure.
   - Drift-aware rescue matters because pure quantum-kernel classifiers weakened under larger samples.

4. Related Work
   - Financial forecasting with classical machine learning [citation needed].
   - Kernel methods in financial classification [citation needed].
   - Quantum kernel learning and quantum feature maps [citation needed].
   - Representation learning for noisy financial targets [citation needed].
   - Drift, regime shift, and validation-final transfer in financial time series [citation needed].

5. Data and Target Definition
   - VN30 stock hourly scope only.
   - Target: market_relative_vn30.
   - Horizon: h40.
   - Split discipline uses feature_timestamp and target_timestamp.
   - No VN100 evidence.
   - No index-as-stock claim.

6. Methodology
   - QML v4-v7 diagnosed pure quantum-kernel behavior under bounded and scaled samples.
   - V8 converts quantum kernels into features:
     - positive_centroid_similarity
     - negative_centroid_similarity
     - centroid_similarity_gap
     - kernel_margin_score
     - top_eigen_projection_1
     - top_eigen_projection_2
     - top_eigen_projection_3
     - kernel_local_density
     - class_similarity_ratio
   - V8 combines QML kernel features with relative-strength and market-context features.
   - Locked V8 meta-model: L2 Logistic.
   - Locked V8 decision method: robust threshold by validation quantile.
   - All transforms are train-only or validation-governed.
   - Final rows are scoring-only.

7. Experiment Evolution
   - QML v4 bounded-sample kernel discovery.
   - QML v5 sample-size weakening.
   - QML v6 scaling rescue failure diagnosis.
   - QML v7 hybrid/kernel-feature partial rescue.
   - QML v8 drift-aware kernel-feature rescue.

8. Results
   - V8 locked candidate validation accuracy: 60.56%.
   - V8 locked candidate final diagnostic accuracy: 64.44%.
   - V8 final improvement over V7 final: +5.56 pp.
   - Drift audit:
     - mean final PSI versus validation: 1.5096.
     - validation positive ratio: 48.33%.
     - final positive ratio: 35.56%.
     - full 30-ticker coverage.
   - Most useful kernel features:
     - positive_centroid_similarity
     - top_eigen_projection_1
     - negative_centroid_similarity
   - Same-target classical comparison:
     - L2 Logistic: +7.78 pp validation, +3.33 pp final.
     - RBF SVM: +5.00 pp validation, +25.00 pp final.
     - LightGBM small: +4.44 pp validation, +1.67 pp final.
     - Best same-target classical final: +0.56 pp.

9. Discussion
   - QML appears more useful as a representation layer than as a standalone QSVC.
   - V8 partially corrected validation-final drift but did not remove it.
   - Final-window drift remains visible through PSI and label-ratio shift.
   - The result does not replace the 61.61% absolute-direction classical champion because target and scope differ.
   - Future-blind confirmation is required before stronger claims.

10. Claim Boundary and Governance
    - Diagnostic-only.
    - No trading, profitability, BUY/SELL, recommendation, investment advice, or live deployment claim.
    - No VN100.
    - No index-as-stock claim.
    - No direct replacement of the 61.61% classical champion.
    - No final-performance selection.

11. Conclusion
    - QML kernel features produced a measurable market-relative diagnostic improvement.
    - Pure QML classifier behavior was unstable under scaling.
    - Hybrid QML representation plus classical decision layer is the strongest observed path.
    - The contribution is methodological and diagnostic.
