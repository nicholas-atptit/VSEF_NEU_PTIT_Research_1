# VN30 Model Cooperation Transfer Audit Result

- Output directory: `reports/generated/vn30_model_universe_benchmark/model_cooperation_transfer_audit`.
- Current main result: Logistic L2, baseline_C_closest, h40, validation-selected threshold 0.55, final accuracy 61.63%.
- Prior descriptive context: bull_bear_sideway_router, h40, fixed 0.50, final accuracy 63.33%, not claim-eligible.
- Total candidates evaluated: 82.
- Tracks run: calibration_threshold_cooperation, error_correction, feature_selection_cooperation, mixture_of_experts, model_as_feature, reference_base, soft_voting.
- Full 30-stock coverage: yes.
- Leakage audit passed: yes.
- Current main result changes: no.

## Audit Checks

- required output files present: yes
- required figures present: yes
- no final-window selection: yes
- no leakage: yes
- no future regime labels: yes
- scaler/imputer fitted on train only: yes
- base model predictions used safely: yes
- meta-models trained only on validation predictions: yes
- error-correction trained only on validation diagnostics: yes
- ensemble weights selected only on validation: yes
- calibration time-safe: yes
- routers selected only on validation: yes
- feature-selection cooperation no final leakage: yes
- full 30-stock coverage: yes
- no ticker subset: yes
- no confidence abstention: yes
- no top-k substitution: yes
- selected by validation-only objective: yes
- claim/descriptive leaderboards separated: yes
- beating candidates have overfit diagnostics: yes
- leakage audit passed: yes

## Selected Candidates

| selection_objective | candidate_id | track | model_id | validation_accuracy | final_accuracy | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max_validation_accuracy | coop__model_as_feature__random_forest_meta__validation_base_predictions__meta_train_scopevalidation_predictions_only__fixed_0p50__t0p500 | model_as_feature | random_forest_meta | 0.6280386280386281 | 0.4864997545409916 | no | high |
| max_validation_balanced_accuracy | coop__model_as_feature__random_forest_meta__validation_base_predictions__meta_train_scopevalidation_predictions_only__fixed_0p50__t0p500 | model_as_feature | random_forest_meta | 0.6280386280386281 | 0.4864997545409916 | no | high |
| max_validation_lift_over_majority | coop__model_as_feature__random_forest_meta__validation_base_predictions__meta_train_scopevalidation_predictions_only__fixed_0p50__t0p500 | model_as_feature | random_forest_meta | 0.6280386280386281 | 0.4864997545409916 | no | high |
| max_validation_rolling_stability | coop__model_as_feature__random_forest_meta__validation_base_predictions__meta_train_scopevalidation_predictions_only__fixed_0p50__t0p500 | model_as_feature | random_forest_meta | 0.6280386280386281 | 0.4864997545409916 | no | high |
| max_validation_monthly_stability | coop__mixture_of_experts__ticker_group_router__ticker__mapping{'ACB'__'linear_svm',_'BCM'__'elastic_net',_'BID'__'knn_probability',_'BVH'__'xgboost',_'CTG'__validation_selected_threshold__t0p475 | mixture_of_experts | ticker_group_router | 0.5745920745920746 | 0.5346097201767305 | yes | medium |
| max_validation_ticker_stability | coop__model_as_feature__random_forest_meta__validation_base_predictions__meta_train_scopevalidation_predictions_only__fixed_0p50__t0p500 | model_as_feature | random_forest_meta | 0.6280386280386281 | 0.4864997545409916 | no | high |
| min_validation_instability | coop__mixture_of_experts__ticker_group_router__ticker__mapping{'ACB'__'linear_svm',_'BCM'__'elastic_net',_'BID'__'knn_probability',_'BVH'__'xgboost',_'CTG'__validation_selected_threshold__t0p475 | mixture_of_experts | ticker_group_router | 0.5745920745920746 | 0.5346097201767305 | yes | medium |
| balanced_transfer_score | coop__model_as_feature__random_forest_meta__validation_base_predictions__meta_train_scopevalidation_predictions_only__fixed_0p50__t0p500 | model_as_feature | random_forest_meta | 0.6280386280386281 | 0.4864997545409916 | no | high |

## Descriptive Final Leaderboard

| candidate_id | track | model_id | validation_accuracy | final_accuracy | selected_by_validation_yes_no | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coop__soft_voting__validation_lift_weighted_soft_vote__base_model_probabilities__weight_columnvalidation_lift_weight__fixed_0p50__t0p500 | soft_voting | validation_lift_weighted_soft_vote | 0.5198801198801198 | 0.6200294550810015 | no | no | medium |
| coop__error_correction__error_random_forest_fallback_to_validation_selected_model__validation_diagnostics__error_train_scopevalidation_only_rulefallback_to_validation_selected_model__fixed_0p50__t0p500 | error_correction | error_random_forest_fallback_to_validation_selected_model | 0.5190809190809191 | 0.6173294059891998 | no | no | medium |
| coop__mixture_of_experts__regime_router__market_direction_regime__mapping{'bear'__'knn_probability',_'bull'__'lightgbm',_'sideway'__'elastic_net'}_router_columnmarket__validation_selected_threshold__t0p475 | mixture_of_experts | regime_router | 0.5475524475524476 | 0.6131566028473245 | no | no | medium |
| coop__error_correction__error_lightgbm_fallback_to_validation_selected_model__validation_diagnostics__error_train_scopevalidation_only_rulefallback_to_validation_selected_model__fixed_0p50__t0p500 | error_correction | error_lightgbm_fallback_to_validation_selected_model | 0.5173160173160173 | 0.6131566028473245 | no | no | medium |
| coop__error_correction__error_random_forest_fallback_to_validation_selected_model__validation_diagnostics__error_train_scopevalidation_only_rulefallback_to_validation_selected_model__validation_selected_threshold__t0p600 | error_correction | error_random_forest_fallback_to_validation_selected_model | 0.5257742257742257 | 0.6119293078055965 | no | no | medium |
| coop__calibration_threshold_cooperation__regime_specific_threshold__logistic_l2_probability__threshold_scopemarket_direction_regime_thresholds{'bear'__0p575,_'bull'__0p6,_'sideway'__0p5}__fixed_0p50__t0p500 | calibration_threshold_cooperation | regime_specific_threshold | 0.5242091242091242 | 0.6116838487972509 | no | no | medium |
| coop__calibration_threshold_cooperation__regime_specific_threshold__logistic_l2_probability__threshold_scopemarket_direction_regime_thresholds{'bear'__0p575,_'bull'__0p6,_'sideway'__0p5}__validation_selected_threshold__t0p500 | calibration_threshold_cooperation | regime_specific_threshold | 0.5242091242091242 | 0.6116838487972509 | no | no | medium |
| coop__error_correction__error_xgboost_fallback_to_validation_selected_model__validation_diagnostics__error_train_scopevalidation_only_rulefallback_to_validation_selected_model__fixed_0p50__t0p500 | error_correction | error_xgboost_fallback_to_validation_selected_model | 0.517016317016317 | 0.6102110947471773 | no | no | medium |
| coop__error_correction__error_logistic_threshold_adjustment__validation_diagnostics__error_train_scopevalidation_only_rulethreshold_adjustment__fixed_0p50__t0p500 | error_correction | error_logistic_threshold_adjustment | 0.5203796203796204 | 0.6099656357388317 | no | no | medium |
| coop__reference_base__logistic_l2_reference_threshold_0p55__baseline_C_closest__threshold_reference0p55__fixed_0p50__t0p500 | reference_base | logistic_l2_reference_threshold_0p55 | 0.517016317016317 | 0.6099656357388317 | no | no | medium |
| coop__error_correction__error_logistic_probability_shrinkage__validation_diagnostics__error_train_scopevalidation_only_ruleprobability_shrinkage__fixed_0p50__t0p500 | error_correction | error_logistic_probability_shrinkage | 0.517016317016317 | 0.6099656357388317 | no | no | medium |
| coop__error_correction__error_random_forest_probability_shrinkage__validation_diagnostics__error_train_scopevalidation_only_ruleprobability_shrinkage__fixed_0p50__t0p500 | error_correction | error_random_forest_probability_shrinkage | 0.517016317016317 | 0.6099656357388317 | no | no | medium |
| coop__error_correction__error_lightgbm_probability_shrinkage__validation_diagnostics__error_train_scopevalidation_only_ruleprobability_shrinkage__fixed_0p50__t0p500 | error_correction | error_lightgbm_probability_shrinkage | 0.517016317016317 | 0.6099656357388317 | no | no | medium |
| coop__error_correction__error_xgboost_probability_shrinkage__validation_diagnostics__error_train_scopevalidation_only_ruleprobability_shrinkage__fixed_0p50__t0p500 | error_correction | error_xgboost_probability_shrinkage | 0.517016317016317 | 0.6099656357388317 | no | no | medium |
| coop__calibration_threshold_cooperation__validation_selected_global_threshold__logistic_l2_probability__threshold_scopeglobal_validation__fixed_0p50__t0p500 | calibration_threshold_cooperation | validation_selected_global_threshold | 0.517016317016317 | 0.6099656357388317 | no | no | medium |
