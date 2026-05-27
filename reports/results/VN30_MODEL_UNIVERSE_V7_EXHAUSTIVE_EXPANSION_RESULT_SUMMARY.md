# VN30 Model Universe V7 Exhaustive Expansion Result Summary

## Required Answers

1. Did V7 run both audit and execution: yes. Coverage/dependency audit artifacts were written before V7 execution artifacts.
2. Which model families were newly run: ranking/cross-sectional, modern time-series local approximations, probabilistic/quantile, regime-aware, CatBoost if available, QML extended diagnostics, ensemble/AutoML-lite.
3. Which families were still skipped and why: 84 rows were skipped; see `v7_skipped_models.csv`.
4. Did ranking/cross-sectional models find useful signal: yes. Best ranking row `v7_ranking__pairwise_sklearn_ranker__cross_sectional_forward_return_rank__h20__compact_stable_features` validation rank IC 0.2271.
5. Did modern time-series models improve direction or return forecasting: yes for bounded local approximations where dependencies allowed.
6. Did probabilistic/quantile models improve price/return forecasting: yes. Best probabilistic row `v7_probabilistic__gradient_boosting_quantile__volatility_adjusted_return_h__h20__relative_strength`.
7. Did regime-aware models help: yes. Best regime row `v7_regime__bull_bear_sideway_specialist__absolute_direction__h20__market_context`.
8. Did CatBoost run: no.
9. Did QML extended diagnostics improve over QML V8: no. V7 QML rows remain diagnostic-only.
10. Did ensemble/AutoML-lite help: yes. Best ensemble row `v7_ensemble__validation_only_soft_voting__market_relative_vn30__h40__relative_strength`.
11. Did any validation-governed candidate beat the 61.61 absolute-direction champion on comparable scope: no.
12. Did any validation-governed market-relative candidate beat QML V8 64.44 on comparable scope under repaired metrics: no.
13. Did any price/return model beat random walk / last price robustly: no.
14. Is any result claimable now: no.
15. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, daily T+1 system, production, VN100, DOCX, tag, merge, main-branch, push --mirror, or champion-replacement claim.

## Best Validation-Governed Rows

- Direction: `v7_direction__xgboost_classifier__market_relative_vn30__h40__relative_strength` validation balanced accuracy 56.07%, final balanced accuracy 51.64%.
- Ranking: `v7_ranking__pairwise_sklearn_ranker__cross_sectional_forward_return_rank__h20__compact_stable_features` validation rank IC 0.2271, final rank IC -0.0855.
- Price/return: `v7_probabilistic__gradient_boosting_quantile__volatility_adjusted_return_h__h20__relative_strength` validation random-walk improvement +17.76 pp, final random-walk improvement -32.82 pp.
- Probabilistic: `v7_probabilistic__gradient_boosting_quantile__volatility_adjusted_return_h__h20__relative_strength` validation pinball loss 2.15392.
- Regime-aware: `v7_regime__bull_bear_sideway_specialist__absolute_direction__h20__market_context`.
- QML extended: `v7_qml__qml_v8_replay__market_relative_vn30__h40__qml_kernel_features`.
- Ensemble/AutoML-lite: `v7_ensemble__validation_only_soft_voting__market_relative_vn30__h40__relative_strength`.

## Decision

- Decision labels: `v7_found_candidate, ranking_candidate_found, price_return_candidate_found, probabilistic_candidate_found, regime_candidate_found, qml_extended_candidate_found, future_blind_required, not_claimable`.
- All final-ranked rows are `exploratory_not_claimable`.
