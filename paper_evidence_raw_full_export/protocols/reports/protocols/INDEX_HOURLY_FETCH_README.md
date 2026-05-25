# Index Hourly Fetch README

This track is index-only.

- Benchmark run: no
- Paper generated: no
- DOCX generated: no
- Stock fetch: no
- Daily data: no
- Resampling: no

Supported index codes:

- `VNINDEX`
- `HNXINDEX`
- `UPCOMINDEX`
- `VN30`
- `HNX30`
- `VN100`

Provider/source priority:

1. `vnstock_data` package with `KBS`
2. `vnstock_data` package with `VCI`
3. legacy `vnstock` package with `KBS`
4. legacy `vnstock` package with `VCI`

Hourly interval only: `1H`.

Raw chunk path:

- `data/raw/vnstock_fetch/index_hourly/{INDEX_CODE}/`

Normalized cache path:

- `data/market_cache/vnstock_data/indices/hourly/{INDEX_CODE}.csv`

Fetch reports:

- `reports/generated/index_hourly_fetch/provider_probe/vnstock_supported_indices_probe.csv`
- `reports/generated/index_hourly_fetch/provider_probe/vnstock_supported_indices_probe.md`
- `reports/generated/index_hourly_fetch/fetch/index_hourly_fetch_summary.csv`
- `reports/generated/index_hourly_fetch/fetch/index_hourly_fetch_summary.md`
- `reports/generated/index_hourly_fetch/fetch/index_hourly_fetch_failures.csv`
- `reports/generated/index_hourly_fetch/fetch/index_hourly_chunk_log.csv`

Validation reports:

- `reports/generated/index_hourly_fetch/validation/index_hourly_validation.csv`
- `reports/generated/index_hourly_fetch/validation/index_hourly_validation.md`

## Full-Range Attempt Status

The required 2005-start command was run:

```text
<repo-approved-venv-python> scripts\research\fetch_vnstock_supported_indices_hourly.py --start 2005-01-01 --end auto --chunk-days 5 --resume
```

It stopped at the explicit progress cap with `stopped_reason=max_runtime_seconds=240` while scanning early VNINDEX chunks. The early chunks returned no rows before the cap, so full 2005-current index history is not yet complete.

To validate the provider path with real hourly rows, a supplemental index-only recent-window fetch was run from `2026-05-04` to provider-current. That run completed for all supported index codes using `vnstock_data` and `KBS`.
