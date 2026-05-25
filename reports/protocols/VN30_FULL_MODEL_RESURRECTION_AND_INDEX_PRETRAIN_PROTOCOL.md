# VN30 Full Model Resurrection And Index-Pretrain Protocol

## Scope

- Final target: VN30 stock hourly directional benchmark only.
- Index usage: main market indices are used only to build point-in-time lagged market-context features.
- Index benchmark results are not stock benchmark claims.
- Top-k ranking is not used as overall directional accuracy.
- Out of scope: trading, profitability, BUY/SELL, recommendations, live deployment, DOCX, paper generation, git tags.

## Split Rules

- Train rows require feature_timestamp <= `2023-12-31 23:59:59` and target_timestamp <= `2023-12-31 23:59:59`.
- Validation rows require feature_timestamp and target_timestamp from `2024-01-01 00:00:00` through `2024-12-31 23:59:59`.
- Final rows require feature_timestamp and target_timestamp >= `2025-01-01 00:00:00`.
- Candidate, model, threshold, and lock selection use validation only.
- Final-ranked exploration is written separately as `exploratory_final_leaderboard.csv` and is not claimable.

## Index Context Layer

- Required context features: market_direction_lag1, market_direction_lag5, market_return_lag1, market_return_lag5, market_volatility_5, market_volatility_20, index_agreement_score, risk_on_risk_off_state, market_momentum_5, market_momentum_20, cross_index_breadth_proxy.
- All index features are lagged or computed from rolling windows shifted one bar before merging to stock rows.
- Stock labels are not used in the index layer.

## Tuning

- Horizons: [20, 30, 35, 40, 45, 50, 60].
- Thresholds: 0.450 to 0.600 step 0.005.
- Feature groups: old_baseline_C_closest, old_regime_feature_v2, feature_set_C_closest, feature_set_C_closest_plus_index_context, feature_set_C_closest_plus_relative_strength, feature_set_C_closest_plus_volatility_regime, feature_set_C_closest_plus_volume_shock, compact_stable_features, index_context_only_plus_stock_lags, full_market_context.
- Model families: logistic regression, elasticnet logistic, random forest, XGBoost, LightGBM, soft-vote ensemble, regime-gated ensemble.
- Full grid is enumerated in `candidate_grid.csv`; budgeted staged screening is recorded in `run_config.json`.

## Validation Composite Score

score = 0.35 * validation_lift + 0.25 * validation_accuracy + 0.15 * quarterly_stability_score + 0.10 * ticker_stability_score + 0.10 * prediction_balance_score + 0.05 * simplicity_score.
