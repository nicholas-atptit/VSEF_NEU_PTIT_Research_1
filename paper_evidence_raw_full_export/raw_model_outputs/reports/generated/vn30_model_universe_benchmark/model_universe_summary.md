# Model Universe Summary

- Exhaustive full run: True.
- Total model groups listed: 12.
- Total model variants planned: 75.
- Total model variants attempted: 75.
- Total model variants run: 74.
- Total model variants failed: 0.
- Total model variants skipped: 0.
- Total model variants not recommended: 1.
- Candidate rows planned/attempted: 1868.
- Successful result rows: 1864.
- CatBoost status: run.
- GARCH diagnostic status: not_recommended_with_reason.
- GARCH used as main directional classifier: no.
- Current main result: Logistic L2 baseline_C_closest h40 validation-selected threshold 0.55, 61.63%.

## Validation-Selected Row

| candidate_id | model_group | model_id | feature_family | horizon | threshold_policy | validation_accuracy | final_accuracy | ticker_coverage | claim_eligible_yes_no |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| universe__ensemble_stacking__stacking_xgboost_meta__validation_selected_base_models__h80__validation_selected_threshold__t0p525 | ensemble_stacking | stacking_xgboost_meta | validation_selected_base_models | 80 | validation_selected_threshold | 0.6724941724941725 | 0.4968684759916493 | 30 | yes |

## Top Final Accuracy Rows

| candidate_id | model_group | model_id | feature_family | horizon | threshold_policy | validation_accuracy | final_accuracy | ticker_coverage | selected_by_validation_yes_no | claim_eligible_yes_no |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| universe__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__fixed_0p50__t0p500 | regime_aware_models | bull_bear_sideway_router | baseline_C_closest | 40 | fixed_0.50 | 0.5178821178821179 | 0.6332842415316642 | 30 | no | no |
| universe__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__validation_selected_threshold__t0p525 | regime_aware_models | bull_bear_sideway_router | baseline_C_closest | 40 | validation_selected_threshold | 0.5212454212454213 | 0.6308296514482081 | 30 | no | no |
| universe__regime_aware_models__high_low_volatility_router__baseline_C_closest__h40__validation_selected_threshold__t0p550 | regime_aware_models | high_low_volatility_router | baseline_C_closest | 40 | validation_selected_threshold | 0.5206460206460206 | 0.6256750122729504 | 30 | no | no |
| universe__regime_aware_models__high_low_volatility_router__baseline_C_closest__h40__fixed_0p50__t0p500 | regime_aware_models | high_low_volatility_router | baseline_C_closest | 40 | fixed_0.50 | 0.518015318015318 | 0.62346588119784 | 30 | no | no |
| universe__calibration_variants__isotonic_logistic__volatility_normalized__h40__fixed_0p50__t0p500 | calibration_variants | isotonic_logistic | volatility_normalized | 40 | fixed_0.50 | 0.5229104229104229 | 0.6212567501227295 | 30 | no | no |
| universe__calibration_variants__isotonic_logistic__volatility_normalized__h40__validation_selected_threshold__t0p500 | calibration_variants | isotonic_logistic | volatility_normalized | 40 | validation_selected_threshold | 0.5229104229104229 | 0.6212567501227295 | 30 | no | no |
| universe__calibration_variants__isotonic_logistic__combined_context__h40__fixed_0p50__t0p500 | calibration_variants | isotonic_logistic | combined_context | 40 | fixed_0.50 | 0.5229104229104229 | 0.6212567501227295 | 30 | no | no |
| universe__calibration_variants__isotonic_logistic__combined_context__h40__validation_selected_threshold__t0p500 | calibration_variants | isotonic_logistic | combined_context | 40 | validation_selected_threshold | 0.5229104229104229 | 0.6212567501227295 | 30 | no | no |
| universe__linear_generalized_linear_models__logistic_l2__volatility_normalized__h40__fixed_0p50__t0p500 | linear_generalized_linear_models | logistic_l2 | volatility_normalized | 40 | fixed_0.50 | 0.5183483183483183 | 0.6190476190476191 | 30 | no | no |
| universe__linear_generalized_linear_models__logistic_l2__combined_context__h40__fixed_0p50__t0p500 | linear_generalized_linear_models | logistic_l2 | combined_context | 40 | fixed_0.50 | 0.5183483183483183 | 0.6190476190476191 | 30 | no | no |
