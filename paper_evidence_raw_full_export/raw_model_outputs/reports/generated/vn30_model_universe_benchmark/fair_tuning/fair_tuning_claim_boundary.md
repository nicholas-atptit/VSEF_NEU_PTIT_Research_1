# Fair Tuning Claim Boundary

- Claim eligibility requires validation-only selection, full 30-stock coverage, non-diagnostic model role, audit pass, and no high overfit risk.
- Descriptive final-window leaderboard rows are descriptive only.
- The final window is not used for model, family, feature, threshold, horizon, calibration, ensemble, router, or claim selection.
- No trading, profitability, investment recommendation, or live-deployment claim is made.

## Validation-Selected Rows

| selection_objective | candidate_id | model_group | model_id | validation_accuracy | final_accuracy | claim_eligible_yes_no | reason_not_claim_eligible | overfit_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_validation_accuracy | fair__ensemble_stacking_models__stacking_xgboost_meta__validation_selected_base_models__h40__base_pool_validation_selected__fixed_0p50__t0p500 | ensemble_stacking_models | stacking_xgboost_meta | 0.614052614052614 | 0.47054491899852724 | no | selected by validation objective but high overfit risk | high |
| max_validation_balanced_accuracy | fair__ensemble_stacking_models__stacking_xgboost_meta__validation_selected_base_models__h40__base_pool_validation_selected__fixed_0p50__t0p500 | ensemble_stacking_models | stacking_xgboost_meta | 0.614052614052614 | 0.47054491899852724 | no | selected by validation objective but high overfit risk | high |
| max_validation_lift_over_majority | fair__ensemble_stacking_models__stacking_xgboost_meta__validation_selected_base_models__h40__base_pool_validation_selected__fixed_0p50__t0p500 | ensemble_stacking_models | stacking_xgboost_meta | 0.614052614052614 | 0.47054491899852724 | no | selected by validation objective but high overfit risk | high |
| max_validation_rolling_stability | fair__ensemble_stacking_models__stacking_xgboost_meta__validation_selected_base_models__h40__base_pool_validation_selected__fixed_0p50__t0p500 | ensemble_stacking_models | stacking_xgboost_meta | 0.614052614052614 | 0.47054491899852724 | no | selected by validation objective but high overfit risk | high |
| min_validation_instability | fair__naive_baseline__random_walk_direction__ex_ante_rule__h40__fixed_rule__fixed_0p50__t0p500 | naive_baseline | random_walk_direction | 0.494005994005994 | 0.5004909180166912 | no | selected by validation objective but high overfit risk | high |
| max_validation_monthly_stability | fair__technical_rules__rsi_rule__ex_ante_rule__h40__window10__fixed_0p50__t0p500 | technical_rules | rsi_rule | 0.5100566100566101 | 0.5085910652920962 | no | selected by validation objective but high overfit risk | high |
| max_validation_ticker_stability | fair__ensemble_stacking_models__stacking_xgboost_meta__validation_selected_base_models__h40__base_pool_validation_selected__fixed_0p50__t0p500 | ensemble_stacking_models | stacking_xgboost_meta | 0.614052614052614 | 0.47054491899852724 | no | selected by validation objective but high overfit risk | high |
| balanced_robust_score | fair__ensemble_stacking_models__stacking_xgboost_meta__validation_selected_base_models__h40__base_pool_validation_selected__fixed_0p50__t0p500 | ensemble_stacking_models | stacking_xgboost_meta | 0.614052614052614 | 0.47054491899852724 | no | selected by validation objective but high overfit risk | high |
