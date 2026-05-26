# QML Kernel Feature VN30 Table Plan

## Table 1. QML v4-v8 Evolution

Purpose: show why the paper centers on V8 rather than the original pure quantum-kernel classifier.

Columns:
- Stage
- Main design
- Target/horizon
- Validation result
- Final result
- Main interpretation
- Claim status

Rows:
- v4 bounded-sample quantum-kernel discovery
- v5 sample-size weakening
- v6 scaling rescue diagnosis
- v7 hybrid/kernel-feature partial rescue
- v8 drift-aware kernel-feature rescue

Evidence:
- `reports/results/VN30_QML_FORECASTING_V4_KERNEL_CONFIRMATION_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V5_FULL_CONFIRMATION_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V6_SCALING_RESCUE_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V7_HYBRID_KERNEL_RESCUE_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

## Table 2. V8 Locked Candidate Configuration

Purpose: document the exact validation-selected model.

Fields:
- candidate_id
- sample_id
- kernel_feature_source
- scaling
- feature_set
- meta_model
- drift_method
- target_variant
- horizon
- validation_rows
- final_rows
- threshold
- claim_label

Evidence:
- `reports/generated/vn30_qml_forecasting/qml_v8_locked_candidate.json`
- `reports/generated/vn30_qml_forecasting/qml_v8_final_result.csv`
- `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json`

## Table 3. V8 Main Result Versus V7

Purpose: show the measured drift-aware rescue improvement.

Rows:
- V7 best QML-kernel-feature meta-model
- V8 locked candidate

Columns:
- Validation accuracy
- Final diagnostic accuracy
- Delta final accuracy
- Interpretation

Evidence:
- `reports/generated/vn30_qml_forecasting/qml_v8_champion_context_comparison.csv`
- `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

## Table 4. V8 Same-Target Classical Comparison

Purpose: compare the V8 locked candidate against same-target baselines.

Rows:
- V8 locked candidate
- L2 Logistic
- RBF SVM
- LightGBM small
- Best same-target classical final model

Columns:
- Validation accuracy
- Final diagnostic accuracy
- V8 delta validation
- V8 delta final

Required reported deltas:
- L2 Logistic: +7.78 pp validation, +3.33 pp final
- RBF SVM: +5.00 pp validation, +25.00 pp final
- LightGBM small: +4.44 pp validation, +1.67 pp final
- Best same-target classical final: +0.56 pp

Evidence:
- `reports/generated/vn30_qml_forecasting/qml_v8_same_target_classical_comparison.csv`
- `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json`

## Table 5. Drift Audit Summary

Purpose: show why drift-aware modeling was needed.

Rows:
- V8 selected distribution-matched sample train split
- V8 selected distribution-matched sample validation split
- V8 selected distribution-matched sample final split

Columns:
- Rows
- Label positive ratio
- Ticker count
- Ticker entropy
- Mean final PSI versus validation
- Interpretation

Required reported values:
- Mean final PSI versus validation: 1.5096
- Validation positive ratio: 48.33%
- Final positive ratio: 35.56%
- Full 30-ticker coverage

Evidence:
- `reports/generated/vn30_qml_forecasting/qml_v8_drift_audit.csv`

## Table 6. Most Useful Kernel Features

Purpose: identify which QML-derived representation terms are most useful.

Rows:
- positive_centroid_similarity
- top_eigen_projection_1
- negative_centroid_similarity

Columns:
- Feature name
- Interpretation
- Evidence source
- Validation correlation or diagnostic indicator if reported

Evidence:
- `reports/generated/vn30_qml_forecasting/qml_v8_kernel_feature_audit.csv`
- `reports/generated/vn30_qml_forecasting/qml_v8_kernel_feature_deciles.csv`
- `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

## Table 7. Claim Boundary Table

Purpose: make the paper-safe claim boundary explicit.

Rows:
- Diagnostic-only QML kernel-feature candidate
- No trading/profitability claim
- No BUY/SELL or investment recommendation
- No live deployment claim
- No VN100
- No index-as-stock claim
- No replacement of the 61.61% absolute-direction classical champion
- Future-blind confirmation required

Evidence:
- `reports/claims/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_CLAIM_BOUNDARY.md`
- `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json`
