# QML Kernel Feature VN30 Figure Plan

## Figure 1. QML Drift-Aware Pipeline

Purpose: show the V8 architecture.

Elements:
- VN30 hourly stock rows
- feature_timestamp and target_timestamp split gate
- market_relative_vn30 h40 target
- relative-strength and market-context feature compression
- quantum kernel construction
- kernel-derived features
- L2 Logistic meta-model
- robust validation-quantile threshold
- validation-governed lock
- final scoring-only evaluation

Claim note:
- The diagram must state "diagnostic-only" and "no trading or live deployment claim."

Evidence:
- `reports/generated/vn30_qml_forecasting/qml_v8_locked_candidate.json`
- `reports/claims/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_CLAIM_BOUNDARY.md`

## Figure 2. QML v4-v8 Performance Evolution

Purpose: show transition from pure quantum-kernel discovery to drift-aware kernel-feature rescue.

Suggested chart:
- x-axis: v4, v5, v6, v7, v8
- y-axis: validation and final diagnostic accuracy
- visual marker for v8 locked candidate

Evidence:
- `reports/results/VN30_QML_FORECASTING_V4_KERNEL_CONFIRMATION_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V5_FULL_CONFIRMATION_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V6_SCALING_RESCUE_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V7_HYBRID_KERNEL_RESCUE_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

## Figure 3. V8 Versus Same-Target Classical Baselines

Purpose: compare V8 against same-target L2 Logistic, RBF SVM, and LightGBM small.

Suggested chart:
- grouped bars for validation accuracy and final diagnostic accuracy
- models:
  - V8 locked QML-kernel-feature meta-model
  - L2 Logistic
  - RBF SVM
  - LightGBM small
  - best same-target classical final model

Required annotations:
- +7.78 pp validation and +3.33 pp final versus L2 Logistic
- +5.00 pp validation and +25.00 pp final versus RBF SVM
- +4.44 pp validation and +1.67 pp final versus LightGBM small
- +0.56 pp versus best same-target classical final model

Evidence:
- `reports/generated/vn30_qml_forecasting/qml_v8_same_target_classical_comparison.csv`
- `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json`

## Figure 4. Drift Audit Chart

Purpose: show validation-final drift and why drift-aware rescue was tested.

Suggested chart:
- split-level bars for label positive ratio:
  - train: 47.50%
  - validation: 48.33%
  - final: 35.56%
- full 30-ticker coverage annotation
- mean final PSI versus validation: 1.5096

Evidence:
- `reports/generated/vn30_qml_forecasting/qml_v8_drift_audit.csv`

## Figure 5. Claim Boundary Diagram

Purpose: separate allowed diagnostic claims from blocked claims.

Left side:
- validation-governed QML kernel-feature candidate
- 60.56% validation accuracy
- 64.44% final diagnostic accuracy
- same-target diagnostic baseline wins
- future-blind required

Right side:
- no trading/profitability
- no BUY/SELL/recommendation
- no live deployment
- no VN100
- no index-as-stock
- no replacement of 61.61% absolute-direction classical champion

Evidence:
- `reports/claims/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_CLAIM_BOUNDARY.md`
- `reports/paper/qml_kernel_feature_vn30/QML_KERNEL_FEATURE_VN30_CLAIM_REGISTER.md`
