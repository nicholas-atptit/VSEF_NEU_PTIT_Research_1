# QML Kernel Feature VN30 Claim Register

## Paper Title

Drift-Aware Quantum Kernel Features for Market-Relative VN30 Stock Forecasting

## Claim Scope

- Scope: VN30 stock hourly forecasting only.
- Target: market_relative_vn30.
- Horizon: h40.
- QML component: quantum-kernel-derived features.
- Decision layer: validation-governed classical meta-model.
- Main V8 locked candidate: relative_market_context_topk4, minmax_0_pi, QML kernel features plus relative-strength and market-context features, L2 Logistic meta-model, robust threshold by validation quantile.

## Allowed Claims

1. V8 produced a validation-governed QML kernel-feature diagnostic candidate.
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_locked_candidate.json`
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json`

2. V8 reached 60.56% validation accuracy.
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_validation_results.csv`
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_validation_leaderboard.csv`
   - Evidence: `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

3. V8 reached 64.44% final diagnostic accuracy.
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_final_result.csv`
   - Evidence: `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

4. V8 improved over V7 final diagnostic accuracy by +5.56 pp.
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_champion_context_comparison.csv`
   - Evidence: `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

5. V8 beat same-target L2 Logistic, RBF SVM, and LightGBM small baselines in the diagnostic comparison.
   - L2 Logistic: +7.78 pp validation and +3.33 pp final.
   - RBF SVM: +5.00 pp validation and +25.00 pp final.
   - LightGBM small: +4.44 pp validation and +1.67 pp final.
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_same_target_classical_comparison.csv`
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json`

6. QML kernel features were most useful through positive_centroid_similarity, top_eigen_projection_1, and negative_centroid_similarity.
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_kernel_feature_audit.csv`
   - Evidence: `reports/generated/vn30_qml_forecasting/qml_v8_kernel_feature_deciles.csv`
   - Evidence: `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

7. QML remains diagnostic-only.
   - Evidence: `reports/claims/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_CLAIM_BOUNDARY.md`

## Not Allowed Claims

1. QML replaces the 61.61% absolute-direction classical champion.
   - Reason: V8 target is market_relative_vn30, while the classical champion is absolute_direction.
   - Reason: future-blind confirmation is required.

2. QML proves trading profitability.
   - Reason: no trading simulation or profitability claim is part of the V8 paper scope.

3. QML gives BUY/SELL signals.
   - Reason: the output is a diagnostic forecasting benchmark, not a recommendation system.

4. QML is production-ready.
   - Reason: no live deployment, operations, monitoring, or production claim is made.

5. QML is future-blind confirmed.
   - Reason: the V8 result is validation-governed and final-scored, but stronger claims require future-blind confirmation.

6. QML works on VN100.
   - Reason: VN100 is out of scope.

7. QML is a live deployment system.
   - Reason: no live deployment claim is made.

8. QML uses index data as a stock substitute.
   - Reason: index data is market-relative context only; no index-as-stock claim is made.

## Required Paper-Safe Wording

The V8 result is a validation-governed, diagnostic market_relative_vn30 h40 candidate. It supports the interpretation that QML-derived kernel features can improve a same-target diagnostic benchmark when combined with relative-strength, market-context features, and a classical L2 Logistic meta-model. It does not establish trading profitability, investment advice, live deployment readiness, VN100 generalization, or replacement of the 61.61% absolute-direction classical champion.
