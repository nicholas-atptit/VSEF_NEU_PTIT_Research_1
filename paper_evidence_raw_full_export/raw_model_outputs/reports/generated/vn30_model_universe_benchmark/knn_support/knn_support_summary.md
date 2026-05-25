# KNN Support Experiment Summary

- Candidate rows: 48.
- Successful results: 48.
- Primary horizon: h40.
- Optional h20/h60/h80 diagnostics: not run because the requested primary KNN-support test is h40 and no main result was promoted.
- Data fetch: no.
- Final-window selection: no.
- Current main comparison: Logistic L2 h40 threshold 0.55 final 61.63%.
- Fair model-zoo comparison: fair tuning selected best final=50.86%; descriptive best=63.33%.
- Best descriptive comparison: bull_bear_sideway_router h40 fixed 0.50 final 63.33%.

## Validation-Selected KNN-Support Candidate

| candidate_id | model_group | model_id | feature_family | threshold_policy | validation_accuracy | final_accuracy | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn_support__ensemble_meta_knn_support__logistic_meta_with_knn_probability__base_probabilities_plus_knn__h40__base_inputs7inputs_meta_train_scopevalidation_predictions_only__fixed_0p50__t0p500 | ensemble_meta_knn_support | logistic_meta_with_knn_probability | base_probabilities_plus_knn | fixed_0.50 | 0.531968031968032 | 0.49018163966617573 | no | high |

## Standalone KNN Comparator

| candidate_id | validation_accuracy | final_accuracy | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- |
| knn_support__standalone_knn_comparator__knn_prediction_probability__knn_probability_only__h40__k11_rolestandalone_comparator_only__validation_selected_threshold__t0p575 | 0.5084915084915085 | 0.5355915562101129 | no | medium |

## Descriptive KNN-Support Final Leaderboard

| candidate_id | model_group | model_id | feature_family | threshold_policy | validation_accuracy | final_accuracy | selected_by_validation_yes_no | claim_eligible_yes_no | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn_support__random_forest_knn_features__random_forest_knn__baseline_plus_knn__h40__max_depth8_min_samples_leaf10_n_estimators140__validation_selected_threshold__t0p600 | random_forest_knn_features | random_forest_knn | baseline_plus_knn | validation_selected_threshold | 0.5048951048951049 | 0.5746195385370643 | no | no | medium |
| knn_support__random_forest_knn_features__random_forest_knn__combined_plus_knn__h40__max_depth8_min_samples_leaf10_n_estimators140__validation_selected_threshold__t0p600 | random_forest_knn_features | random_forest_knn | combined_plus_knn | validation_selected_threshold | 0.5055944055944056 | 0.5731467844869906 | no | no | medium |
| knn_support__random_forest_knn_features__random_forest_knn__baseline_plus_knn__h40__max_depthNone_min_samples_leaf25_n_estimators180__validation_selected_threshold__t0p600 | random_forest_knn_features | random_forest_knn | baseline_plus_knn | validation_selected_threshold | 0.505061605061605 | 0.5643102601865488 | no | no | medium |
| knn_support__random_forest_knn_features__random_forest_knn__combined_plus_knn__h40__max_depthNone_min_samples_leaf25_n_estimators180__validation_selected_threshold__t0p600 | random_forest_knn_features | random_forest_knn | combined_plus_knn | validation_selected_threshold | 0.5046620046620046 | 0.5625920471281296 | no | no | medium |
| knn_support__random_forest_knn_features__random_forest_knn__baseline_plus_knn__h40__max_depth8_min_samples_leaf10_n_estimators140__fixed_0p50__t0p500 | random_forest_knn_features | random_forest_knn | baseline_plus_knn | fixed_0.50 | 0.504029304029304 | 0.551791850760923 | no | no | medium |
| knn_support__xgboost_knn_features__xgboost_knn__baseline_plus_knn__h40__learning_rate0p05_max_depth3_n_estimators100__validation_selected_threshold__t0p575 | xgboost_knn_features | xgboost_knn | baseline_plus_knn | validation_selected_threshold | 0.5068265068265069 | 0.5510554737358861 | no | no | medium |
| knn_support__logistic_l2_knn_features__logistic_l2_knn__baseline_plus_knn__h40__C1p0__validation_selected_threshold__t0p600 | logistic_l2_knn_features | logistic_l2_knn | baseline_plus_knn | validation_selected_threshold | 0.511954711954712 | 0.5495827196858125 | no | no | medium |
| knn_support__lightgbm_knn_features__lightgbm_knn__baseline_plus_knn__h40__learning_rate0p05_max_depth3_n_estimators100_num_leaves15__validation_selected_threshold__t0p550 | lightgbm_knn_features | lightgbm_knn | baseline_plus_knn | validation_selected_threshold | 0.5066600066600067 | 0.5483554246440845 | no | no | medium |
| knn_support__random_forest_knn_features__random_forest_knn__combined_plus_knn__h40__max_depth8_min_samples_leaf10_n_estimators140__fixed_0p50__t0p500 | random_forest_knn_features | random_forest_knn | combined_plus_knn | fixed_0.50 | 0.5047952047952048 | 0.5478645066273933 | no | no | medium |
| knn_support__random_forest_knn_features__random_forest_knn__combined_plus_knn__h40__max_depthNone_min_samples_leaf25_n_estimators180__fixed_0p50__t0p500 | random_forest_knn_features | random_forest_knn | combined_plus_knn | fixed_0.50 | 0.5045288045288046 | 0.5478645066273933 | no | no | medium |
