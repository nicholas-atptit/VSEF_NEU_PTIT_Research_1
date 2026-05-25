# Fair Tuning Claim Register

- Current main result context: Logistic L2, baseline_C_closest, h40, threshold 0.55, final accuracy 61.63%.
- Any model beats 61.63% descriptively: yes.
- Claim-eligible rows: 0.
- Final-window descriptive rows do not change the main result unless validation-selected and audit-passed.
- No trading, profitability, investment recommendation, or live-deployment claim is made.

## Claim-Eligible Leaderboard

_No rows._

## Descriptive Beating Rows

| candidate_id | model_group | model_id | feature_family | horizon | threshold_policy | validation_accuracy | final_accuracy | selected_by_validation_objective_yes_no | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fair__linear_models__logistic_l2__volatility_normalized__h40__C0p3_balanced__fixed_0p50__t0p500 | linear_models | logistic_l2 | volatility_normalized | 40 | fixed_0.50 | 0.5183483183483183 | 0.6190476190476191 | no | no | medium |
| fair__linear_models__logistic_l2__volatility_normalized__h40__C0p3_balanced__validation_selected_threshold__t0p600 | linear_models | logistic_l2 | volatility_normalized | 40 | validation_selected_threshold | 0.5235098235098236 | 0.6173294059891998 | no | no | medium |
| fair__linear_models__logistic_l2__combined_context__h40__C0p3_balanced__fixed_0p50__t0p500 | linear_models | logistic_l2 | combined_context | 40 | fixed_0.50 | 0.5183483183483183 | 0.6190476190476191 | no | no | medium |
| fair__linear_models__logistic_l2__combined_context__h40__C0p3_balanced__validation_selected_threshold__t0p600 | linear_models | logistic_l2 | combined_context | 40 | validation_selected_threshold | 0.5235098235098236 | 0.6173294059891998 | no | no | medium |
| fair__calibration_variants__platt_logistic__baseline_C_closest__h40__sigmoid__validation_selected_threshold__t0p600 | calibration_variants | platt_logistic | baseline_C_closest | 40 | validation_selected_threshold | 0.5208791208791209 | 0.616593028964163 | no | no | medium |
| fair__calibration_variants__platt_logistic__combined_context__h40__sigmoid__fixed_0p50__t0p500 | calibration_variants | platt_logistic | combined_context | 40 | fixed_0.50 | 0.512987012987013 | 0.6170839469808542 | no | no | medium |
| fair__calibration_variants__platt_logistic__combined_context__h40__sigmoid__validation_selected_threshold__t0p600 | calibration_variants | platt_logistic | combined_context | 40 | validation_selected_threshold | 0.5209790209790209 | 0.6170839469808542 | no | no | medium |
| fair__calibration_variants__isotonic_logistic__combined_context__h40__isotonic__fixed_0p50__t0p500 | calibration_variants | isotonic_logistic | combined_context | 40 | fixed_0.50 | 0.5210789210789211 | 0.6170839469808542 | no | no | medium |
| fair__calibration_variants__isotonic_logistic__combined_context__h40__isotonic__validation_selected_threshold__t0p500 | calibration_variants | isotonic_logistic | combined_context | 40 | validation_selected_threshold | 0.5210789210789211 | 0.6170839469808542 | no | no | medium |
| fair__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__router_by_market_direction_regime__fixed_0p50__t0p500 | regime_aware_models | bull_bear_sideway_router | baseline_C_closest | 40 | fixed_0.50 | 0.5178821178821179 | 0.6332842415316642 | no | no | medium |
| fair__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__router_by_market_direction_regime__validation_selected_threshold__t0p525 | regime_aware_models | bull_bear_sideway_router | baseline_C_closest | 40 | validation_selected_threshold | 0.5212454212454213 | 0.6308296514482081 | no | no | medium |
| fair__regime_aware_models__high_low_volatility_router__baseline_C_closest__h40__router_by_volatility_regime__fixed_0p50__t0p500 | regime_aware_models | high_low_volatility_router | baseline_C_closest | 40 | fixed_0.50 | 0.518015318015318 | 0.62346588119784 | no | no | medium |
| fair__regime_aware_models__high_low_volatility_router__baseline_C_closest__h40__router_by_volatility_regime__validation_selected_threshold__t0p550 | regime_aware_models | high_low_volatility_router | baseline_C_closest | 40 | validation_selected_threshold | 0.5206460206460206 | 0.6256750122729504 | no | no | medium |
