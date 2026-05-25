# Fair Exhaustive Model-Zoo Tuning Summary

- Primary horizon: h40.
- Secondary horizons: h20/h60/h80 only for h40 validation-selected candidates where rerun is feasible.
- Feature families: baseline_C_closest, volatility_normalized, relative_strength, regime_context, combined_context.
- Candidate rows planned/attempted: 572.
- Successful result rows: 571.
- Model groups tuned: 13.
- Current main context: Logistic L2, baseline_C_closest, h40, threshold 0.55, final accuracy 61.63%.
- Prior descriptive context only: bull_bear_sideway_router h40 fixed 0.50, final accuracy 63.33%.
- Final window role: scoring-only.
- Data fetch: no.
- Provider behavior changed: no.
- Paper/DOCX generation: no.

## Best Validation-Selected Candidate

| selection_objective | candidate_id | model_group | model_id | feature_family | horizon | threshold_policy | validation_accuracy | final_accuracy | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_validation_accuracy | fair__ensemble_stacking_models__stacking_xgboost_meta__validation_selected_base_models__h40__base_pool_validation_selected__fixed_0p50__t0p500 | ensemble_stacking_models | stacking_xgboost_meta | validation_selected_base_models | 40 | fixed_0.50 | 0.614052614052614 | 0.47054491899852724 | no | high |

## Best Balanced-Robust Candidate

| selection_objective | candidate_id | model_group | model_id | feature_family | horizon | threshold_policy | balanced_robust_score | validation_accuracy | final_accuracy | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_robust_score | fair__ensemble_stacking_models__stacking_xgboost_meta__validation_selected_base_models__h40__base_pool_validation_selected__fixed_0p50__t0p500 | ensemble_stacking_models | stacking_xgboost_meta | validation_selected_base_models | 40 | fixed_0.50 | 0.5044277049463934 | 0.614052614052614 | 0.47054491899852724 | no | high |

## Descriptive Final Leaderboard

| candidate_id | model_group | model_id | feature_family | horizon | threshold_policy | validation_accuracy | final_accuracy | selected_by_validation_objective_yes_no | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fair__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__router_by_market_direction_regime__fixed_0p50__t0p500 | regime_aware_models | bull_bear_sideway_router | baseline_C_closest | 40 | fixed_0.50 | 0.5178821178821179 | 0.6332842415316642 | no | no | medium |
| fair__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__router_by_market_direction_regime__validation_selected_threshold__t0p525 | regime_aware_models | bull_bear_sideway_router | baseline_C_closest | 40 | validation_selected_threshold | 0.5212454212454213 | 0.6308296514482081 | no | no | medium |
| fair__regime_aware_models__high_low_volatility_router__baseline_C_closest__h40__router_by_volatility_regime__validation_selected_threshold__t0p550 | regime_aware_models | high_low_volatility_router | baseline_C_closest | 40 | validation_selected_threshold | 0.5206460206460206 | 0.6256750122729504 | no | no | medium |
| fair__regime_aware_models__high_low_volatility_router__baseline_C_closest__h40__router_by_volatility_regime__fixed_0p50__t0p500 | regime_aware_models | high_low_volatility_router | baseline_C_closest | 40 | fixed_0.50 | 0.518015318015318 | 0.62346588119784 | no | no | medium |
| fair__linear_models__logistic_l2__volatility_normalized__h40__C0p3_balanced__fixed_0p50__t0p500 | linear_models | logistic_l2 | volatility_normalized | 40 | fixed_0.50 | 0.5183483183483183 | 0.6190476190476191 | no | no | medium |
| fair__linear_models__logistic_l2__combined_context__h40__C0p3_balanced__fixed_0p50__t0p500 | linear_models | logistic_l2 | combined_context | 40 | fixed_0.50 | 0.5183483183483183 | 0.6190476190476191 | no | no | medium |
| fair__linear_models__logistic_l2__volatility_normalized__h40__C0p3_balanced__validation_selected_threshold__t0p600 | linear_models | logistic_l2 | volatility_normalized | 40 | validation_selected_threshold | 0.5235098235098236 | 0.6173294059891998 | no | no | medium |
| fair__linear_models__logistic_l2__combined_context__h40__C0p3_balanced__validation_selected_threshold__t0p600 | linear_models | logistic_l2 | combined_context | 40 | validation_selected_threshold | 0.5235098235098236 | 0.6173294059891998 | no | no | medium |
| fair__calibration_variants__isotonic_logistic__combined_context__h40__isotonic__fixed_0p50__t0p500 | calibration_variants | isotonic_logistic | combined_context | 40 | fixed_0.50 | 0.5210789210789211 | 0.6170839469808542 | no | no | medium |
| fair__calibration_variants__isotonic_logistic__combined_context__h40__isotonic__validation_selected_threshold__t0p500 | calibration_variants | isotonic_logistic | combined_context | 40 | validation_selected_threshold | 0.5210789210789211 | 0.6170839469808542 | no | no | medium |

## Rows Beating 61.63%

| candidate_id | model_group | model_id | feature_family | horizon | threshold_policy | validation_accuracy | final_accuracy | selected_by_validation_objective_yes_no | claim_eligible_yes_no | overfit_risk | overfit_risk_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fair__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__router_by_market_direction_regime__fixed_0p50__t0p500 | regime_aware_models | bull_bear_sideway_router | baseline_C_closest | 40 | fixed_0.50 | 0.5178821178821179 | 0.6332842415316642 | no | no | medium | 1274 rolling 250 windows below 60%; monthly minimum 10.14%; quarterly minimum 37.46%; ticker minimum 32.82% |
| fair__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__router_by_market_direction_regime__validation_selected_threshold__t0p525 | regime_aware_models | bull_bear_sideway_router | baseline_C_closest | 40 | validation_selected_threshold | 0.5212454212454213 | 0.6308296514482081 | no | no | medium | 1295 rolling 250 windows below 60%; monthly minimum 10.14%; quarterly minimum 36.81%; ticker minimum 31.66% |
| fair__regime_aware_models__high_low_volatility_router__baseline_C_closest__h40__router_by_volatility_regime__validation_selected_threshold__t0p550 | regime_aware_models | high_low_volatility_router | baseline_C_closest | 40 | validation_selected_threshold | 0.5206460206460206 | 0.6256750122729504 | no | no | medium | 1199 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 45.44%; ticker minimum 33.33% |
| fair__regime_aware_models__high_low_volatility_router__baseline_C_closest__h40__router_by_volatility_regime__fixed_0p50__t0p500 | regime_aware_models | high_low_volatility_router | baseline_C_closest | 40 | fixed_0.50 | 0.518015318015318 | 0.62346588119784 | no | no | medium | 1165 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 44.30%; ticker minimum 32.43% |
| fair__linear_models__logistic_l2__volatility_normalized__h40__C0p3_balanced__fixed_0p50__t0p500 | linear_models | logistic_l2 | volatility_normalized | 40 | fixed_0.50 | 0.5183483183483183 | 0.6190476190476191 | no | no | medium | 1316 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 42.08%; ticker minimum 33.20% |
| fair__linear_models__logistic_l2__combined_context__h40__C0p3_balanced__fixed_0p50__t0p500 | linear_models | logistic_l2 | combined_context | 40 | fixed_0.50 | 0.5183483183483183 | 0.6190476190476191 | no | no | medium | 1316 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 42.08%; ticker minimum 33.20% |
| fair__linear_models__logistic_l2__volatility_normalized__h40__C0p3_balanced__validation_selected_threshold__t0p600 | linear_models | logistic_l2 | volatility_normalized | 40 | validation_selected_threshold | 0.5235098235098236 | 0.6173294059891998 | no | no | medium | 1570 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 41.26%; ticker minimum 33.33% |
| fair__linear_models__logistic_l2__combined_context__h40__C0p3_balanced__validation_selected_threshold__t0p600 | linear_models | logistic_l2 | combined_context | 40 | validation_selected_threshold | 0.5235098235098236 | 0.6173294059891998 | no | no | medium | 1570 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 41.26%; ticker minimum 33.33% |
| fair__calibration_variants__isotonic_logistic__combined_context__h40__isotonic__validation_selected_threshold__t0p500 | calibration_variants | isotonic_logistic | combined_context | 40 | validation_selected_threshold | 0.5210789210789211 | 0.6170839469808542 | no | no | medium | 1376 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 46.45%; ticker minimum 33.20% |
| fair__calibration_variants__platt_logistic__combined_context__h40__sigmoid__validation_selected_threshold__t0p600 | calibration_variants | platt_logistic | combined_context | 40 | validation_selected_threshold | 0.5209790209790209 | 0.6170839469808542 | no | no | medium | 1376 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 46.45%; ticker minimum 33.20% |
| fair__calibration_variants__isotonic_logistic__combined_context__h40__isotonic__fixed_0p50__t0p500 | calibration_variants | isotonic_logistic | combined_context | 40 | fixed_0.50 | 0.5210789210789211 | 0.6170839469808542 | no | no | medium | 1376 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 46.45%; ticker minimum 33.20% |
| fair__calibration_variants__platt_logistic__combined_context__h40__sigmoid__fixed_0p50__t0p500 | calibration_variants | platt_logistic | combined_context | 40 | fixed_0.50 | 0.512987012987013 | 0.6170839469808542 | no | no | medium | 1311 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 46.72%; ticker minimum 33.20% |
| fair__calibration_variants__platt_logistic__baseline_C_closest__h40__sigmoid__validation_selected_threshold__t0p600 | calibration_variants | platt_logistic | baseline_C_closest | 40 | validation_selected_threshold | 0.5208791208791209 | 0.616593028964163 | no | no | medium | 1401 rolling 250 windows below 60%; monthly minimum 6.86%; quarterly minimum 45.90%; ticker minimum 32.05% |

## Transfer Quality

| model_group | candidates | mean_validation_accuracy | mean_final_accuracy | best_final_accuracy | mean_validation_final_gap | mean_balanced_robust_score | mean_rolling_250 | mean_runtime_seconds | median_abs_validation_final_gap | transfer_quality_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_models | 200 | 0.515027639027639 | 0.5796269023073147 | 0.6190476190476191 | -0.06459926327967565 | 0.4165806621979933 | 0.5926214535947713 | 1.1642243780000718 | 0.06667662234672544 | 0.5129502799605893 |
| boosting_models | 68 | 0.5115154453389748 | 0.5639853879696208 | 0.5888561610211095 | -0.052469942630646096 | 0.4160684649973757 | 0.5758215763168012 | 2.0907108117663644 | 0.05111537946589495 | 0.5128700085037259 |
| distance_based_models | 32 | 0.5038669663669664 | 0.5433618679430535 | 0.5731467844869906 | -0.03949490157608712 | 0.409508661589813 | 0.5496893790849673 | 3.1591253750029864 | 0.03158010374505216 | 0.5117817641980014 |
| regime_aware_models | 14 | 0.5167570524713382 | 0.6060558243916123 | 0.6332842415316642 | -0.08929877192027416 | 0.41899782392794044 | 0.6204009710550887 | 0.26790946428705603 | 0.09821432862669982 | 0.5078414957649124 |
| probabilistic_models | 42 | 0.5104831676260247 | 0.5619959323935759 | 0.5827196858124694 | -0.051512764767551206 | 0.41990435416477634 | 0.5689064674758793 | 0.10181867618839965 | 0.05567250618797015 | 0.5063234262056058 |
| neural_deep_models | 32 | 0.5023767898767899 | 0.5369645925380462 | 0.5822287677957781 | -0.034587802661256266 | 0.4072596099869905 | 0.544977320261438 | 1.5049034624971682 | 0.03075137920498744 | 0.5062132133330588 |
| tree_models | 60 | 0.5058319458319459 | 0.5529495990836197 | 0.5861561119293078 | -0.04711765325167386 | 0.4106272317480159 | 0.5622868845315904 | 0.6007869066670537 | 0.04682430971090762 | 0.5061252893727122 |
| calibration_variants | 24 | 0.5120115995115995 | 0.5632363770250368 | 0.6170839469808542 | -0.051224777513437326 | 0.4138979514944174 | 0.5710447058823529 | 0.30811905000155093 | 0.06319968691102712 | 0.5000366901140096 |
| svm_and_kernel_models | 22 | 0.517783731420095 | 0.5468268844557505 | 0.593519882179676 | -0.02904315303565539 | 0.4207034410511134 | 0.5533236838978016 | 4.443758127274288 | 0.055146056692448375 | 0.4916808277633021 |
| statistical_models | 4 | 0.4968864468864469 | 0.49282032400589104 | 0.5061364752086401 | 0.004066122880555856 | nan | 0.49605986928104573 | nan | 0.01131274224057724 | 0.48150758176531383 |
| technical_rules | 54 | 0.49987543320876654 | 0.48815433007872866 | 0.553019145802651 | 0.011721103130037845 | 0.4200793939326355 | 0.48759581699346405 | 0.009944677778683533 | 0.023964351799403344 | 0.4641899782793253 |
| ensemble_stacking_models | 12 | 0.5513542013542013 | 0.5300073637702504 | 0.593519882179676 | 0.021346837583950968 | 0.4488031769853416 | 0.5314767755991285 | 0.025293583331707243 | 0.07461232925150457 | 0.4553950345187458 |
| naive_baseline | 7 | 0.4839208410636982 | 0.46549547654113194 | 0.5044182621502209 | 0.018425364522566268 | 0.4077067986471035 | 0.46908504201680673 | 0.0038693285688558327 | 0.024395192436429514 | 0.4411002841047024 |

## Failures And Skips

_No rows._
