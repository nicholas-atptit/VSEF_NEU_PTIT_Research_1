# VN30 Stock + Index Joint Panel Readiness

- Stock instrument count: 30.
- Index instrument count: 6.
- Total instrument count: 36.
- Usable stock instruments: 29/30.
- Usable index instruments: 6/6.
- Joint panel can run with 36/36 instruments: false.
- Repaired stock cache checked: `data/market_cache/vnstock_data/vn30/hourly_2015`.
- Repaired index archive checked: `archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly`.
- Active index fallback checked: `data/market_cache/vnstock_data/indices/hourly_2015`.
- Benchmark/training run in this phase: no.

## Failed Or Risky Instruments

| instrument_code | instrument_type | row_count | first_timestamp | last_timestamp | frequency_status | source_path | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VPL | stock | 111 | 2026-02-11 14:00:00 | 2026-03-20 14:00:00 | ok | data/hourly_market_split_data/VPL.csv | insufficient_rows_for_h120:111<150; insufficient_split_rows_after_horizon |

## Decision

The joint 36-instrument hourly panel is not validation-ready from current cache.
