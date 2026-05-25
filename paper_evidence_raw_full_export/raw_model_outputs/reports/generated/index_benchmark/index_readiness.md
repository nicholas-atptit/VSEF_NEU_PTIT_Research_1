# Supported Index Benchmark Readiness

- Scope: index-only.
- Daily split: train 2015-01-01 to 2023-12-31; validation 2024; final 2025-01-01 to provider-current.
- Hourly split: actual available timestamp window only, split 60% train / 20% validation / 20% final by time order.
- Daily-to-hourly resampling used: no.
- Hourly-to-daily resampling used: no.
- Daily benchmark ready: yes (5/6 indices).
- Hourly benchmark ready: yes (6/6 indices).

## Readiness Rows

| index_code | frequency | cache_path | file_exists | row_count | first_timestamp | last_timestamp | train_rows | validation_rows | final_rows | train_start | train_end | validation_start | validation_end | final_start | final_end | duplicate_timestamps | valid_ohlcv | frequency_correct | usable | reason_unusable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VNINDEX | 1D | data/market_cache/vnstock_data/indices/daily_2015/VNINDEX.csv | yes | 2830 | 2015-01-05 07:00:00 | 2026-05-15 07:00:00 | 2244 | 250 | 336 | 2015-01-05 07:00:00 | 2023-12-29 07:00:00 | 2024-01-02 07:00:00 | 2024-12-31 07:00:00 | 2025-01-02 07:00:00 | 2026-05-15 07:00:00 | 0 | yes | yes | yes |  |
| HNXINDEX | 1D | data/market_cache/vnstock_data/indices/daily_2015/HNXINDEX.csv | yes | 2835 | 2015-01-05 00:00:00 | 2026-05-15 00:00:00 | 2249 | 250 | 336 | 2015-01-05 00:00:00 | 2023-12-29 00:00:00 | 2024-01-02 00:00:00 | 2024-12-31 00:00:00 | 2025-01-02 00:00:00 | 2026-05-15 00:00:00 | 0 | yes | yes | yes |  |
| UPCOMINDEX | 1D | data/market_cache/vnstock_data/indices/daily_2015/UPCOMINDEX.csv | yes | 2834 | 2015-01-05 07:00:00 | 2026-05-15 07:00:00 | 2249 | 250 | 335 | 2015-01-05 07:00:00 | 2023-12-29 07:00:00 | 2024-01-02 07:00:00 | 2024-12-31 07:00:00 | 2025-01-02 07:00:00 | 2026-05-15 07:00:00 | 0 | yes | yes | yes |  |
| VN30 | 1D | data/market_cache/vnstock_data/indices/daily_2015/VN30.csv | yes | 2819 | 2015-01-05 07:00:00 | 2026-05-15 07:00:00 | 2233 | 250 | 336 | 2015-01-05 07:00:00 | 2023-12-29 07:00:00 | 2024-01-02 07:00:00 | 2024-12-31 07:00:00 | 2025-01-02 07:00:00 | 2026-05-15 07:00:00 | 0 | yes | yes | yes |  |
| HNX30 | 1D | data/market_cache/vnstock_data/indices/daily_2015/HNX30.csv | yes | 2822 | 2015-01-05 07:00:00 | 2026-05-15 07:00:00 | 2237 | 250 | 335 | 2015-01-05 07:00:00 | 2023-12-29 07:00:00 | 2024-01-02 07:00:00 | 2024-12-31 07:00:00 | 2025-01-02 07:00:00 | 2026-05-15 07:00:00 | 0 | yes | yes | yes |  |
| VN100 | 1D | data/market_cache/vnstock_data/indices/daily_2015/VN100.csv | yes | 1 | 2026-05-15 00:00:00 | 2026-05-15 00:00:00 | 0 | 0 | 1 |  |  |  |  | 2026-05-15 00:00:00 | 2026-05-15 00:00:00 | 0 | yes | yes | no | train_rows<500; validation_rows<80; final_rows<80 |
| VNINDEX | 1H | data/market_cache/vnstock_data/indices/hourly_2015/VNINDEX.csv | yes | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | 597 | 199 | 199 | 2022-05-19 00:00:00 | 2024-10-04 00:00:00 | 2024-10-07 00:00:00 | 2025-07-24 00:00:00 | 2025-07-25 00:00:00 | 2026-05-15 00:00:00 | 0 | yes | yes | yes |  |
| HNXINDEX | 1H | data/market_cache/vnstock_data/indices/hourly_2015/HNXINDEX.csv | yes | 994 | 2022-05-19 00:00:00 | 2026-05-14 00:00:00 | 596 | 199 | 199 | 2022-05-19 00:00:00 | 2024-10-03 00:00:00 | 2024-10-04 00:00:00 | 2025-07-23 00:00:00 | 2025-07-24 00:00:00 | 2026-05-14 00:00:00 | 0 | yes | yes | yes |  |
| UPCOMINDEX | 1H | data/market_cache/vnstock_data/indices/hourly_2015/UPCOMINDEX.csv | yes | 994 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | 596 | 199 | 199 | 2022-05-19 00:00:00 | 2024-10-03 00:00:00 | 2024-10-04 00:00:00 | 2025-07-24 00:00:00 | 2025-07-25 00:00:00 | 2026-05-15 00:00:00 | 0 | yes | yes | yes |  |
| VN30 | 1H | data/market_cache/vnstock_data/indices/hourly_2015/VN30.csv | yes | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | 597 | 199 | 199 | 2022-05-19 00:00:00 | 2024-10-04 00:00:00 | 2024-10-07 00:00:00 | 2025-07-24 00:00:00 | 2025-07-25 00:00:00 | 2026-05-15 00:00:00 | 0 | yes | yes | yes |  |
| HNX30 | 1H | data/market_cache/vnstock_data/indices/hourly_2015/HNX30.csv | yes | 962 | 2022-07-04 00:00:00 | 2026-05-15 00:00:00 | 577 | 192 | 193 | 2022-07-04 00:00:00 | 2024-10-22 00:00:00 | 2024-10-23 00:00:00 | 2025-08-01 00:00:00 | 2025-08-04 00:00:00 | 2026-05-15 00:00:00 | 0 | yes | yes | yes |  |
| VN100 | 1H | data/market_cache/vnstock_data/indices/hourly_2015/VN100.csv | yes | 1570 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 942 | 314 | 314 | 2023-09-11 10:00:00 | 2024-08-19 11:00:00 | 2024-08-19 13:00:00 | 2024-12-09 14:00:00 | 2024-12-10 10:00:00 | 2026-05-15 00:00:00 | 0 | yes | yes | yes |  |
