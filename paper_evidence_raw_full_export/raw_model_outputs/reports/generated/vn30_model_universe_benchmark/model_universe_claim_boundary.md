# Model Universe Claim Boundary

- Final-window scores are scoring-only and are not used for model, feature, threshold, horizon, ensemble, calibration, or router selection.
- The current h40 paper result remains Logistic L2 / baseline_C_closest / h40 / validation-selected threshold 0.55 / 61.63% unless a new model is validation-selected, full-coverage, and audit-passed.
- GARCH is diagnostic only and not a direct headline direction classifier.
- Total planned/run/failed/skipped/not recommended: 75/74/0/0/1.
- CatBoost status: run.
- GARCH diagnostic status: not_recommended_with_reason.
- No trading, profitability, investment recommendation, or live-deployment claim is made.

| candidate_id | model_id | validation_accuracy | final_accuracy | beats_61_63_yes_no | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- |
| universe__ensemble_stacking__stacking_xgboost_meta__validation_selected_base_models__h80__validation_selected_threshold__t0p525 | stacking_xgboost_meta | 0.6724941724941725 | 0.4968684759916493 | no | yes | high |
