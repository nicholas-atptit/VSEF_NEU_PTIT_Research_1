# QML Kernel Feature VN30 Evidence Map

## Primary Evidence

| Claim or value | Evidence path | Notes |
| --- | --- | --- |
| V8 result summary | `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md` | Paper-safe summary of locked candidate, drift, comparisons, and claim boundary. |
| V8 claim boundary | `reports/claims/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_CLAIM_BOUNDARY.md` | Governance source for diagnostic-only boundaries. |
| V8 kernel feature audit | `reports/generated/vn30_qml_forecasting/qml_v8_kernel_feature_audit.csv` | Kernel feature source, selected features, kernel alignment, drift penalty, split-safety metadata. |
| V8 drift audit | `reports/generated/vn30_qml_forecasting/qml_v8_drift_audit.csv` | Label ratios, ticker coverage, PSI, distribution drift. |
| V8 kernel feature deciles | `reports/generated/vn30_qml_forecasting/qml_v8_kernel_feature_deciles.csv` | Kernel feature decile diagnostics by split. |
| V8 validation results | `reports/generated/vn30_qml_forecasting/qml_v8_validation_results.csv` | Full validation-governed candidate set. |
| V8 validation leaderboard | `reports/generated/vn30_qml_forecasting/qml_v8_validation_leaderboard.csv` | Validation-selected ranking. |
| V8 locked candidate | `reports/generated/vn30_qml_forecasting/qml_v8_locked_candidate.json` | Locked validation-selected model. |
| V8 final result | `reports/generated/vn30_qml_forecasting/qml_v8_final_result.csv` | Final scoring-only result. |
| V8 same-target classical comparison | `reports/generated/vn30_qml_forecasting/qml_v8_same_target_classical_comparison.csv` | Same-target baseline rows. |
| V8 champion context comparison | `reports/generated/vn30_qml_forecasting/qml_v8_champion_context_comparison.csv` | V7, V4, V8, and contextual classical champion comparison. |
| V8 rescue decision | `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json` | Decision labels and locked candidate deltas. |

## Supporting Evidence

| Evidence group | Evidence path | Purpose |
| --- | --- | --- |
| QML v4 bounded kernel confirmation | `reports/results/VN30_QML_FORECASTING_V4_KERNEL_CONFIRMATION_RESULT_SUMMARY.md` | Documents bounded-sample quantum-kernel discovery. |
| QML v5 full confirmation | `reports/results/VN30_QML_FORECASTING_V5_FULL_CONFIRMATION_RESULT_SUMMARY.md` | Documents sample-size weakening. |
| QML v6 scaling rescue diagnosis | `reports/results/VN30_QML_FORECASTING_V6_SCALING_RESCUE_RESULT_SUMMARY.md` | Documents class-balance drift, kernel concentration, small-sample sensitivity, and rescue failure modes. |
| QML v7 hybrid/kernel-feature rescue | `reports/results/VN30_QML_FORECASTING_V7_HYBRID_KERNEL_RESCUE_RESULT_SUMMARY.md` | Documents the V7 kernel-feature meta-model result at 61.11% validation and 58.89% final. |
| QML v4 generated artifacts | `reports/generated/vn30_qml_forecasting/qml_v4_*` | Candidate grids, validation/final evidence, runtime, and manifest. |
| QML v5 generated artifacts | `reports/generated/vn30_qml_forecasting/qml_v5_*` | Frozen replay, sample-size ladder, rolling-origin checks, same-target comparison. |
| QML v6 generated artifacts | `reports/generated/vn30_qml_forecasting/qml_v6_*` | Scaling method, kernel diagnostics, regularization, and diagnosis artifacts. |
| QML v7 generated artifacts | `reports/generated/vn30_qml_forecasting/qml_v7_*` | Distribution matching, kernel health, hybrid kernel, meta-feature, and decision artifacts. |

## Value-to-Source Map

| Reported value | Primary source | Cross-check source |
| --- | --- | --- |
| 60.56% validation accuracy | `reports/generated/vn30_qml_forecasting/qml_v8_validation_results.csv` | `reports/generated/vn30_qml_forecasting/qml_v8_final_result.csv`; `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md` |
| 64.44% final diagnostic accuracy | `reports/generated/vn30_qml_forecasting/qml_v8_final_result.csv` | `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json`; result summary |
| +5.56 pp over V7 final | `reports/generated/vn30_qml_forecasting/qml_v8_champion_context_comparison.csv` | result summary; `qml_v8_rescue_decision.json` |
| L2 Logistic comparison: +7.78 pp validation, +3.33 pp final | `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json` | `reports/generated/vn30_qml_forecasting/qml_v8_same_target_classical_comparison.csv` |
| RBF SVM comparison: +5.00 pp validation, +25.00 pp final | `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json` | `qml_v8_same_target_classical_comparison.csv` |
| LightGBM small comparison: +4.44 pp validation, +1.67 pp final | `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json` | `qml_v8_same_target_classical_comparison.csv` |
| Best same-target classical final comparison: +0.56 pp | `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json` | `qml_v8_same_target_classical_comparison.csv` |
| Mean final PSI versus validation: 1.5096 | `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md` | `reports/generated/vn30_qml_forecasting/qml_v8_drift_audit.csv` |
| Validation positive ratio: 48.33% | `reports/generated/vn30_qml_forecasting/qml_v8_drift_audit.csv` | result summary context |
| Final positive ratio: 35.56% | `reports/generated/vn30_qml_forecasting/qml_v8_drift_audit.csv` | result summary context |
| Full 30-ticker coverage | `reports/generated/vn30_qml_forecasting/qml_v8_drift_audit.csv` | `qml_v8_validation_results.csv` metadata |
| Most useful kernel features | `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md` | `qml_v8_kernel_feature_audit.csv`; `qml_v8_kernel_feature_deciles.csv` |
| Decision labels | `reports/generated/vn30_qml_forecasting/qml_v8_rescue_decision.json` | `reports/generated/vn30_qml_forecasting/qml_v8_manifest.json` |
| Classical 61.61% context | `reports/generated/vn30_qml_forecasting/qml_v8_champion_context_comparison.csv` | Current contextual benchmark: L2 Logistic, feature_set_C_closest, h40, threshold 0.50, 61.61% final accuracy, +10.90 pp lift. |

## Evidence Boundary

- Evidence is used only for a diagnostic QML kernel-feature paper.
- The paper does not edit or rely on the older classical paper draft as a source artifact.
- The classical 61.61% absolute-direction champion is contextual only.
- No final-ranked exploratory row is allowed to become a claimable result.
- No trading, profitability, BUY/SELL, live deployment, VN100, or index-as-stock claim is supported by this evidence map.
