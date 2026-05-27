# VN30 + Index Group Price-Range Forecast Result Summary

1. Which index codes were used? `VNINDEX, VN30, HNXINDEX, HNX30, UPCOMINDEX`.
2. Which index codes were skipped and why? `VNXALL`: index_cache_missing_or_empty
3. Which method best forecasts direction for the VN30 + index group? `lightgbm` on `market_relative_vn30` horizon `5`, validation balanced accuracy 53.79%, macro F1 51.92%, MCC 0.079731.
4. Which method best forecasts return/price? `xgboost_regressor` horizon `5`, validation RMSE 0.018095, MAE 0.013226, rank IC -0.028557.
5. Which method best forecasts price range / interval? `atr_rolling_volatility_band` horizon `5`, validation interval coverage 79.84%, average width 0.040533, Winkler score 0.061264.
6. Which method best ranks VN30 assets? `relative_strength_rank_baseline` horizon `40`, validation Spearman IC 0.161719, NDCG@10 0.893790, top-20 precision 47.50%.
7. Does any direction model beat the strongest baseline under repaired metrics? `True`. Market-relative rows are judged with balanced accuracy/lift, not raw accuracy alone.
8. Does any return/price model beat random walk / last close? `False`. Best validation random-walk RMSE improvement: -0.001322; last-close RMSE improvement: -0.001322.
9. Does any range model achieve useful interval coverage without excessive width? `True`. Useful here means validation coverage between 70% and 90% with average return-width <= 20%.
10. Are results stable across stocks and indices? `mixed-to-stable in validation diagnostics`.
11. Is any result claimable now? `False`.
12. Exact claim boundary: Offline research-lab diagnostics only for VN30 stocks plus configured Vietnamese index codes. Selection is validation-governed; final rows are scoring-only and not claimable. No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, daily T+1 production, VN100 assumption, main-branch, merge, tag, push --mirror, DOCX, or champion-replacement claim is made. A future-blind rerun with frozen code/data before the target period is required before any external claim.

## Scope Update

- Historical/local-data-only offline lab. The runner reads existing repository/cache files only; it does not fetch live data, call provider APIs, create real-time forecasts, or create production workflows.
- Missing required VN30 stock cache rows stop the run and report affected files/assets.
- Missing requested index cache rows are skipped and recorded with `skipped_reason`; no live data is fetched to fill gaps.

## Split Discipline

- Train: feature_timestamp <= `2023-12-31 23:59:59` and target_timestamp <= `2023-12-31 23:59:59`.
- Validation: feature_timestamp and target_timestamp inside calendar year `2024`.
- Final: feature_timestamp and target_timestamp >= `2025-01-01`; final rows are scoring-only.

## Output Artifacts

- Generated CSV/JSON artifacts: `reports/generated/vn30_index_group_range_forecast`.
- Configured horizons: `5,10,20,40,60`.
- Configured indices: `VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL`.

