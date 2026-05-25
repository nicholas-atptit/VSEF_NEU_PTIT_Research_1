# Claim Register

- Current main result: Logistic L2 baseline_C_closest h40 validation-selected threshold 0.55, 61.63%.
- Main result changes: no.
- Final-window score used for selection: no.
- Data fetch: no.
- GARCH used as main direction classifier: no.
- CatBoost run: yes.
- GARCH diagnostic run: yes.

| claim | candidate_id | model_id | feature_family | horizon | threshold_policy | validation_accuracy | final_accuracy | beats_61_63 | claim_eligible | overfit_risk | overfit_risk_reason | main_result_changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected_candidate | universe__ensemble_stacking__stacking_xgboost_meta__validation_selected_base_models__h80__validation_selected_threshold__t0p525 | stacking_xgboost_meta | validation_selected_base_models | 80 | validation_selected_threshold | 0.6724941724941725 | 0.4968684759916493 | no | yes | high | validation-final gap 17.56 pp; rolling 250 mean 50.65%; 1938 rolling 250 windows below 60%; monthly minimum 18.84%; quarterly minimum 33.33%; ticker minimum 0.00% | no |
| best_final_descriptive_only | universe__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__fixed_0p50__t0p500 | bull_bear_sideway_router | baseline_C_closest | 40 | fixed_0.50 | 0.5178821178821179 | 0.6332842415316642 | yes | no | high | post-hoc final leaderboard beating row, not validation-selected; 1274 rolling 250 windows below 60%; monthly minimum 10.14%; quarterly minimum 37.46%; ticker minimum 32.82% | no |
