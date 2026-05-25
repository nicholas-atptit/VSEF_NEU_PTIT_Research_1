# Supported Index Data Scope Audit

- Scope: index-only.
- Stock data used: no.
- Resampling used: no.
- Earliest daily index date: `2015-01-05 00:00:00`.
- Earliest hourly index timestamp: `2022-05-19 00:00:00`.
- 2015 daily index data exists: yes.
- 2015 hourly index data exists: no.
- Daily benchmark can run: yes.
- Hourly benchmark can run: yes.

## Best Local Daily Files

| index_code | frequency | path | file_exists | row_count | first_timestamp | last_timestamp | year_coverage | provider_source | usable_for_train_validation_final | reason_unusable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VNINDEX | 1D | data/market_cache/vnstock_data/indices/daily_2015/VNINDEX.csv | yes | 2830 | 2015-01-05 07:00:00 | 2026-05-15 07:00:00 | 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026 | vnstock_data;KBS | yes |  |
| HNXINDEX | 1D | data/market_cache/vnstock_data/indices/daily_2015/HNXINDEX.csv | yes | 2835 | 2015-01-05 00:00:00 | 2026-05-15 00:00:00 | 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026 | repo_adapter;KBS | yes |  |
| UPCOMINDEX | 1D | data/market_cache/vnstock_data/indices/daily_2015/UPCOMINDEX.csv | yes | 2834 | 2015-01-05 07:00:00 | 2026-05-15 07:00:00 | 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026 | legacy_vnstock;KBS | yes |  |
| VN30 | 1D | data/market_cache/vnstock_data/indices/daily_2015/VN30.csv | yes | 2819 | 2015-01-05 07:00:00 | 2026-05-15 07:00:00 | 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026 | vnstock_data;KBS | yes |  |
| HNX30 | 1D | data/market_cache/vnstock_data/indices/daily_2015/HNX30.csv | yes | 2822 | 2015-01-05 07:00:00 | 2026-05-15 07:00:00 | 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026 | vnstock_data;KBS | yes |  |
| VN100 | 1D | data/market_cache/vnstock_data/indices/daily_2015/VN100.csv | yes | 1 | 2026-05-15 00:00:00 | 2026-05-15 00:00:00 | 2026 | repo_adapter;KBS | no | insufficient_split_coverage |

## Best Local Hourly Files

| index_code | frequency | path | file_exists | row_count | first_timestamp | last_timestamp | year_coverage | provider_source | usable_for_train_validation_final | reason_unusable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VNINDEX | 1H | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/VNINDEX.csv | yes | 5143 | 2022-05-19 09:00:00 | 2026-05-13 15:00:00 | 2022,2023,2024,2025,2026 | vnstock_data;KBS | yes |  |
| HNXINDEX | 1H | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/HNXINDEX.csv | yes | 5147 | 2022-05-19 09:00:00 | 2026-05-13 15:00:00 | 2022,2023,2024,2025,2026 | vnstock_data;KBS | yes |  |
| UPCOMINDEX | 1H | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/UPCOMINDEX.csv | yes | 5955 | 2022-05-19 09:00:00 | 2026-05-13 15:00:00 | 2022,2023,2024,2025,2026 | vnstock_data;KBS | yes |  |
| VN30 | 1H | data/market_cache/vnstock_data/indices/hourly_2015/VN30.csv | yes | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | 2022,2023,2024,2025,2026 | repo_adapter;KBS | yes |  |
| HNX30 | 1H | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/HNX30.csv | yes | 4984 | 2022-07-04 09:00:00 | 2026-05-13 15:00:00 | 2022,2023,2024,2025,2026 | vnstock_data;KBS | yes |  |
| VN100 | 1H | data/market_cache/vnstock_data/indices/hourly_2015/VN100.csv | yes | 1570 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 2023,2024,2025,2026 | legacy_vnstock;repo_adapter;VCI;KBS | yes |  |
