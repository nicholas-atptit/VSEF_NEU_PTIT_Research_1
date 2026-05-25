# VN30 V8 Strategy Target Redesign Result Summary

## Locked Strategy

- Strategy: `v8strat_00425`.
- Target/feature/model: neutral_removed_direction_b0020 / combined_strategy_features / lightgbm.
- Horizon/template: h40 / top_n_rotation.
- Validation return/Sharpe/drawdown: 21.81% / 0.38642036688302206 / -56.57%.
- Validation trades/exposure/turnover: 125 / 0.7 / 39.7457122913484.
- Validation strongest baseline delta: equal_weight_vn30_stock_basket / +1.72 pp.

## Final Locked Result

- Final return after cost: 38.87%.
- Final Sharpe: 0.8450180947188821.
- Final max drawdown: -57.79%.
- Final trades/exposure/turnover: 40 / 0.7000000000000002 / 13.824713244641694.
- Final strongest baseline: buy_and_hold_vn30_index; baseline delta: -14.29 pp.

## Required Answers

1. Best validation target variant: neutral_removed_direction_b0020.
2. Best validation feature group: combined_strategy_features.
3. Best validation model family: lightgbm.
4. Locked strategy: `v8strat_00425`.
5. Final total return after cost: 38.87%.
6. Final Sharpe: 0.8450180947188821.
7. Final max drawdown: -57.79%.
8. Final trade count and exposure: 40 / 0.7000000000000002.
9. Beats buy-and-hold VN30: false (baseline=53.15%).
10. Beats equal-weight VN30: true (baseline=18.09%).
11. Beats random same-turnover: true (baseline=9.40%).
12. Best exploratory final strategy: [{'strategy_id': 'v8strat_00494', 'model_candidate_id': 'v8model_00320', 'model_family': 'lightgbm', 'target_variant': 'neutral_removed_direction_b0100', 'target_family': 'neutral_removed_direction', 'feature_group': 'combined_strategy_features', 'horizon': 40, 'strategy_template': 'regime_filtered_rotation', 'top_n': 3, 'top_quantile': '', 'w_model': '', 'w_rs': '', 'market_regime_filter': 'on', 'max_positions': 3, 'max_exposure': 0.7, 'cost_bps': 10, 'slippage_bps': 5, 'strategy_family_key': 'v8model_00320__regime_filtered_rotation__n3__q__wm__wrs__mp3__ex0p7', 'split': 'final', 'total_return': 0.799664501879372, 'annualized_return': 0.5677339393040515, 'sharpe': 1.0926749259007138, 'sortino': 0.2310814114780033, 'max_drawdown': -0.39563827160493836, 'calmar': 1.434982356486882, 'win_rate': 0.7083333333333334, 'profit_factor': 5.232192407273338, 'average_trade_return': 0.116064982932513, 'trade_count': 24, 'exposure_ratio': 0.6929054054054055, 'turnover': 15.659529727626827, 'cost_adjusted_return': 0.799664501879372, 'claim_label': 'exploratory_not_claimable', 'reason_not_claimable': 'final-ranked row is exploratory', 'strongest_baseline': 'buy_and_hold_vn30_index', 'strongest_baseline_return': 0.5315443716497913, 'baseline_delta': 0.2681201302295807}].
13. Target redesign improves over V7 locked final return: true.
14. Claimable results: none as a strategy claim; locked strategy is validation-governed diagnostic only.
15. Paper-safe wording: offline diagnostic only; no BUY/SELL, profitability, investment advice, deployment, or claimable strategy claim.

Paper-safe wording:

> VN30 V8 redesigned the strategy target from absolute-direction ranking toward market-relative, top-quantile, cost-adjusted, and neutral-removed stock selection labels. The locked strategy was selected by validation-only diagnostics and evaluated once on final. Final-ranked strategy rows are exploratory only, and future-blind confirmation is required before any stronger strategy claim.
