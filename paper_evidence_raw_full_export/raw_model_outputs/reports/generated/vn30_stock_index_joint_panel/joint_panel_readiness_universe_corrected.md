# VN30 Stock + Index Joint Panel Readiness - Universe Corrected

- Universe config: `configs/universes/vn30_jan2025_joint_panel_universe.csv`.
- Stock instrument count: 30.
- Index instrument count: 6.
- Total instrument count: 36.
- BCM included and usable: true.
- BVH included and usable: true.
- DGC excluded: true.
- VPL excluded: true.
- Usable stock instruments: 30/30.
- Usable index instruments: 6/6.
- Joint panel can run with 36/36 instruments: true.
- Repaired stock cache checked: `data/market_cache/vnstock_data/vn30/hourly_2015`.
- Repaired index archive checked: `archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly`.
- Active index fallback checked: `data/market_cache/vnstock_data/indices/hourly_2015`.
- Benchmark/training run in this phase: no.

## Failed Or Risky Instruments

| instrument_code | instrument_type | row_count | first_timestamp | last_timestamp | frequency_status | source_path | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/ACB.csv | usable |
| BID | stock | 1455 | 2023-09-11 10:00:00 | 2026-05-14 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/BID.csv | usable |
| BCM | stock | 1651 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/BCM.csv | usable |
| BVH | stock | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/BVH.csv | usable |
| CTG | stock | 1440 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/CTG.csv | usable |
| FPT | stock | 1449 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/FPT.csv | usable |
| GAS | stock | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/GAS.csv | usable |
| GVR | stock | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/GVR.csv | usable |
| HDB | stock | 1439 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/HDB.csv | usable |
| HPG | stock | 1592 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/HPG.csv | usable |
| LPB | stock | 1590 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/LPB.csv | usable |
| MBB | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/MBB.csv | usable |
| MSN | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/MSN.csv | usable |
| MWG | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/MWG.csv | usable |
| PLX | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/PLX.csv | usable |
| SAB | stock | 1421 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/SAB.csv | usable |
| SHB | stock | 1620 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/SHB.csv | usable |
| SSB | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/SSB.csv | usable |
| SSI | stock | 1446 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/SSI.csv | usable |
| STB | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/STB.csv | usable |
| TCB | stock | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/TCB.csv | usable |
| TPB | stock | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/TPB.csv | usable |
| VCB | stock | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/VCB.csv | usable |
| VHM | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/VHM.csv | usable |
| VIB | stock | 1602 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/VIB.csv | usable |
| VIC | stock | 1448 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/VIC.csv | usable |
| VJC | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/VJC.csv | usable |
| VNM | stock | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/VNM.csv | usable |
| VPB | stock | 1589 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/VPB.csv | usable |
| VRE | stock | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/vn30/hourly_2015/VRE.csv | usable |
| VNINDEX | index | 5143 | 2022-05-19 09:00:00 | 2026-05-13 15:00:00 | ok | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/VNINDEX.csv | usable |
| VN30 | index | 5146 | 2022-05-19 09:00:00 | 2026-05-13 15:00:00 | ok | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/VN30.csv | usable |
| HNXINDEX | index | 5147 | 2022-05-19 09:00:00 | 2026-05-13 15:00:00 | ok | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/HNXINDEX.csv | usable |
| HNX30 | index | 4984 | 2022-07-04 09:00:00 | 2026-05-13 15:00:00 | ok | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/HNX30.csv | usable |
| UPCOMINDEX | index | 5955 | 2022-05-19 09:00:00 | 2026-05-13 15:00:00 | ok | archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/UPCOMINDEX.csv | usable |
| VN100 | index | 1570 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | ok | data/market_cache/vnstock_data/indices/hourly_2015/VN100.csv | usable |

## Decision

The joint 36-instrument hourly panel is ready.

## Remaining Blockers

None.
