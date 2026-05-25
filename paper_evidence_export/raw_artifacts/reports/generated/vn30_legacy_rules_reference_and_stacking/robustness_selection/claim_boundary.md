# VN30 Legacy Robustness Selection Claim Boundary

## Scope

- Inputs: existing legacy-compatible model and stacking prediction artifacts only.
- No model training, no market-data fetch, and no provider behavior change.
- Selection basis: validation-only metrics.
- Final-window accuracy is reported only after objective selection.
- The validation-final gap penalty is an ex-post reporting/audit field, not a model, feature, threshold, weight, or objective-selection input.
- Headline eligibility requires full 30-stock coverage on validation and final rows.

## Objective Selections

| Objective | Candidate | Validation Acc | Final Acc | Effect vs current 61.63% | Overfit Risk | Classification |
| --- | --- | ---: | ---: | --- | --- | --- |
| best_final_accuracy | `legacy_single__logistic_l2__baseline_C_closest__h40__validation_selected_threshold__t0p550` | 52.36% | 61.63% | preserves_final_accuracy | low | best_final_accuracy |
| best_validation_stability | `legacy_stack__meta_lightgbm_stacking` | 61.02% | 48.72% | reduces_final_accuracy | high | rejected_overfit_risk |
| best_rolling_stability | `legacy_stack__meta_lightgbm_stacking` | 61.02% | 48.72% | reduces_final_accuracy | high | rejected_overfit_risk |
| best_balanced | `legacy_stack__meta_lightgbm_stacking` | 61.02% | 48.72% | reduces_final_accuracy | high | rejected_overfit_risk |

## Result Interpretation

- Eligible full-coverage candidates evaluated: 18.
- Improving validation/stability objectives did not automatically improve final-window accuracy.
- The current Logistic L2 h40 threshold 0.55 candidate remains the fixed final-accuracy comparator at 61.63%.
- Best validation-stability selected final accuracy: 48.72%, effect: reduces_final_accuracy.
- Best rolling-stability selected final accuracy: 48.72%, effect: reduces_final_accuracy.
- Best balanced selected final accuracy: 48.72%, effect: reduces_final_accuracy.

## Claim Boundary

This robustness-selection experiment does not establish a stronger final-window candidate unless a validation-only objective also preserves or improves final accuracy without high overfit risk. Candidates with large negative validation-final gaps are classified as `rejected_overfit_risk` even if their validation metrics are strong.
