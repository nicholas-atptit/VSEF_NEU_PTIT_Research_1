# Model Universe Registry

- Total model groups listed: 12.
- Total model variants planned: 75.
- Total model variants attempted: 75.
- Total model variants run: 74.
- Total model variants failed: 0.
- Total model variants skipped: 0.
- Total model variants not recommended: 1.
- CatBoost status: run.
- GARCH diagnostic status: not_recommended_with_reason.
- GARCH used as main directional classifier: no.

| model_id | model_group | model_name | implementation_status | run_status | dependency_required | scaling_required | runtime_risk | data_requirement | reason_if_skipped | reason_if_failed | reason_if_not_recommended | paper_role | claim_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_class | naive_baselines | Majority Class | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| random_walk_direction | naive_baselines | Random Walk Direction | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| previous_direction | naive_baselines | Previous Direction | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| persistence_rule | naive_baselines | Persistence Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| moving_average_rule | naive_baselines | Moving Average Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| rolling_momentum_rule | naive_baselines | Rolling Momentum Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| volatility_adjusted_momentum_rule | naive_baselines | Volatility Adjusted Momentum Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| sma_crossover | technical_rule_baselines | Sma Crossover | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| ema_crossover | technical_rule_baselines | Ema Crossover | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| macd_rule | technical_rule_baselines | Macd Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| rsi_rule | technical_rule_baselines | Rsi Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| bollinger_band_rule | technical_rule_baselines | Bollinger Band Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| price_momentum_rule | technical_rule_baselines | Price Momentum Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| volume_momentum_rule | technical_rule_baselines | Volume Momentum Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| mean_reversion_rule | technical_rule_baselines | Mean Reversion Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| breakout_rule | technical_rule_baselines | Breakout Rule | implemented | run | none | no | low | OHLCV rows with non-null horizon labels |  |  |  | baseline_comparison | yes |
| logistic_l2 | linear_generalized_linear_models | Logistic L2 | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| logistic_l1 | linear_generalized_linear_models | Logistic L1 | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| logistic_elastic_net | linear_generalized_linear_models | Logistic Elastic Net | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| ridge_classifier | linear_generalized_linear_models | Ridge Classifier | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| lda | linear_generalized_linear_models | Lda | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| qda | linear_generalized_linear_models | Qda | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| passive_aggressive | linear_generalized_linear_models | Passive Aggressive | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| perceptron | linear_generalized_linear_models | Perceptron | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| sgd_hinge | linear_generalized_linear_models | Sgd Hinge | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| sgd_log_loss | linear_generalized_linear_models | Sgd Log Loss | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| linear_svm | linear_generalized_linear_models | Linear Svm | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| svm_rbf | kernel_distance_based_models | Svm Rbf | implemented | run | sklearn | yes | high | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| svm_poly | kernel_distance_based_models | Svm Poly | implemented | run | sklearn | yes | high | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| knn | kernel_distance_based_models | Knn | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| radius_neighbors | kernel_distance_based_models | Radius Neighbors | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| nearest_centroid | kernel_distance_based_models | Nearest Centroid | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| gaussian_naive_bayes | probabilistic_models | Gaussian Naive Bayes | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| bernoulli_naive_bayes | probabilistic_models | Bernoulli Naive Bayes | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| complement_naive_bayes | probabilistic_models | Complement Naive Bayes | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| decision_tree | tree_based_models | Decision Tree | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| random_forest | tree_based_models | Random Forest | implemented | run | sklearn | no | medium | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| extra_trees | tree_based_models | Extra Trees | implemented | run | sklearn | no | medium | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| adaboost | boosting_models | Adaboost | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| sklearn_gradient_boosting | boosting_models | Sklearn Gradient Boosting | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| hist_gradient_boosting | boosting_models | Hist Gradient Boosting | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| xgboost | boosting_models | Xgboost | implemented | run | xgboost | no | medium | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| lightgbm | boosting_models | Lightgbm | implemented | run | lightgbm | no | medium | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| catboost | boosting_models | Catboost | implemented | run | catboost | no | medium | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| mlp_classifier | neural_deep_models | Mlp Classifier | implemented | run | sklearn | yes | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| lstm | neural_deep_models | Lstm | implemented | run | torch | no | high | time-safe per-ticker sequence tensors with non-null labels |  |  |  | expanded_model_universe_candidate | yes |
| gru | neural_deep_models | Gru | implemented | run | torch | no | high | time-safe per-ticker sequence tensors with non-null labels |  |  |  | expanded_model_universe_candidate | yes |
| tcn | neural_deep_models | Tcn | implemented | run | torch | no | high | time-safe per-ticker sequence tensors with non-null labels |  |  |  | expanded_model_universe_candidate | yes |
| cnn_1d | neural_deep_models | Cnn 1D | implemented | run | torch | no | high | time-safe per-ticker sequence tensors with non-null labels |  |  |  | expanded_model_universe_candidate | yes |
| cnn_lstm | neural_deep_models | Cnn Lstm | implemented | run | torch | no | high | time-safe per-ticker sequence tensors with non-null labels |  |  |  | expanded_model_universe_candidate | yes |
| hard_voting | ensemble_stacking | Hard Voting | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_ensemble | yes |
| soft_voting | ensemble_stacking | Soft Voting | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_ensemble | yes |
| validation_weighted_soft_vote | ensemble_stacking | Validation Weighted Soft Vote | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_ensemble | yes |
| stacking_logistic_meta | ensemble_stacking | Stacking Logistic Meta | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_ensemble | yes |
| stacking_lightgbm_meta | ensemble_stacking | Stacking Lightgbm Meta | implemented | run | lightgbm | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_ensemble | yes |
| stacking_xgboost_meta | ensemble_stacking | Stacking Xgboost Meta | implemented | run | xgboost | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_ensemble | yes |
| blending | ensemble_stacking | Blending | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_ensemble | yes |
| platt_logistic | calibration_variants | Platt Logistic | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | probability_calibration_diagnostic | yes |
| isotonic_logistic | calibration_variants | Isotonic Logistic | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | probability_calibration_diagnostic | yes |
| calibrated_svm | calibration_variants | Calibrated Svm | implemented | run | sklearn | yes | high | numeric feature matrix with non-null horizon labels |  |  |  | probability_calibration_diagnostic | yes |
| calibrated_random_forest | calibration_variants | Calibrated Random Forest | implemented | run | sklearn | no | low | numeric feature matrix with non-null horizon labels |  |  |  | probability_calibration_diagnostic | yes |
| calibrated_xgboost | calibration_variants | Calibrated Xgboost | implemented | run | xgboost | no | medium | numeric feature matrix with non-null horizon labels |  |  |  | probability_calibration_diagnostic | yes |
| calibrated_lightgbm | calibration_variants | Calibrated Lightgbm | implemented | run | lightgbm | no | medium | numeric feature matrix with non-null horizon labels |  |  |  | probability_calibration_diagnostic | yes |
| regime_context_logistic | regime_aware_models | Regime Context Logistic | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| regime_context_xgboost | regime_aware_models | Regime Context Xgboost | implemented | run | xgboost | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| regime_context_lightgbm | regime_aware_models | Regime Context Lightgbm | implemented | run | lightgbm | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| bull_bear_sideway_router | regime_aware_models | Bull Bear Sideway Router | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| high_low_volatility_router | regime_aware_models | High Low Volatility Router | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| regime_threshold_router | regime_aware_models | Regime Threshold Router | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| regime_model_router | regime_aware_models | Regime Model Router | implemented | run | none | no | low | numeric feature matrix with non-null horizon labels |  |  |  | expanded_model_universe_candidate | yes |
| arima_direction | traditional_statistical_financial_models | Arima Direction | implemented | run | statsmodels | no | low | per-ticker close or return series from local historical rows |  |  |  | expanded_model_universe_candidate | yes |
| sarima_direction | traditional_statistical_financial_models | Sarima Direction | implemented | run | statsmodels | no | low | per-ticker close or return series from local historical rows |  |  |  | expanded_model_universe_candidate | yes |
| ets_direction | traditional_statistical_financial_models | Ets Direction | implemented | run | statsmodels | no | low | per-ticker close or return series from local historical rows |  |  |  | expanded_model_universe_candidate | yes |
| var_direction | traditional_statistical_financial_models | Var Direction | implemented | run | statsmodels | no | low | per-ticker close or return series from local historical rows |  |  |  | expanded_model_universe_candidate | yes |
| garch_volatility_diagnostic | traditional_statistical_financial_models | Garch Volatility Diagnostic | implemented | not_recommended_with_reason | arch | no | low | per-ticker close or return series from local historical rows |  |  | GARCH is a volatility diagnostic, not a direct direction classifier. | volatility_diagnostic_only | no |
