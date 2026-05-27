# VN30 + Index Group Price-Range Forecast Claim Boundary

- This artifact is a research lab for direction, return/price, range/interval, and cross-sectional ranking diagnostics.
- Scope is VN30 stocks plus explicitly configured index codes only: `VNINDEX, VN30, HNXINDEX, HNX30, UPCOMINDEX, VNXALL`.
- Historical/local-data-only offline lab. The runner reads existing repository/cache files only; it does not fetch live data, call provider APIs, create real-time forecasts, or create production workflows.
- Used index codes: `VNINDEX, VN30, HNXINDEX, HNX30, UPCOMINDEX`.
- Skipped index codes are recorded in `reports/generated/vn30_index_group_range_forecast/index_coverage_audit.csv`.
- Feature rows and targets obey explicit feature_timestamp/target_timestamp split guards.
- Direction, return/price, range/interval, and ranking metrics are separate; stock and index row counts are reported separately where applicable.
- Raw accuracy alone is not used to judge market-relative direction targets.
- Final rows remain scoring-only and are not used for claimable selection.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, daily T+1 system, VN100 assumption, DOCX, tag, merge, push --mirror, or main-branch claim is made.
- Exact claim boundary: Offline research-lab diagnostics only for VN30 stocks plus configured Vietnamese index codes. Selection is validation-governed; final rows are scoring-only and not claimable. No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, daily T+1 production, VN100 assumption, main-branch, merge, tag, push --mirror, DOCX, or champion-replacement claim is made. A future-blind rerun with frozen code/data before the target period is required before any external claim.

