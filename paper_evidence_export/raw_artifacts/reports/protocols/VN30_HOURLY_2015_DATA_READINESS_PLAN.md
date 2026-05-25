# VN30 Hourly 2015 Data Readiness Plan

## Design

The active VN30/index hourly data design starts at `2015-01-01 00:00:00`. Earlier 2005/2006 full-history designs are superseded for this track and must not be used as the required start.

- Frequency: hourly only, `1H`.
- Stock effective-start rule: `effective_start(ticker) = max(2015-01-01, first_trading_date)`.
- Stocks listed before 2015 start from `2015-01-01`.
- Stocks listed after 2015 start from their first trading date.
- Fetch direction: reverse, from provider-current/latest available timestamp backward to effective start.
- Training/history period: `2015-01-01 00:00:00` to `2024-12-31 23:59:59`, bounded per ticker by effective start.
- Evaluation/comparison start: `2025-01-01 00:00:00`.
- Evaluation end: provider-current/latest available timestamp, not a future date.
- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Daily data: not allowed.
- Daily-to-hourly resampling: not allowed.
- Synthetic missing bars: not allowed.
- Active universe: remains `configs/universes/vn30_constituents_frozen.csv` unless explicitly changed later.
- Extra metadata symbols: `BSR` is treated as `extra_user_provided_symbol` if it is not in the frozen universe.

## Supported Index Codes

Use only provider-supported names unless a later provider probe proves otherwise:

- `VNINDEX`
- `HNXINDEX`
- `UPCOMINDEX`
- `VN30`
- `HNX30`
- `VN100`

Do not use `VN30INDEX`; use `VN30`. Do not use `VNXALL` under the current provider evidence.

## Workflow

1. Reset old working artifacts with `scripts/research/reset_vn30_hourly_2015_workspace.py`.
2. Build listing-date reconciliation and effective starts with `scripts/research/vn30_hourly_2015_effective_start.py`.
3. Fetch index hourly data by reverse chunks with `scripts/research/fetch_supported_indices_hourly_gateway_2015.py`.
4. Validate index hourly data with `scripts/research/validate_supported_indices_hourly_gateway_2015.py`.
5. Fetch frozen VN30 stock hourly data by reverse chunks with `scripts/research/fetch_vn30_stocks_hourly_gateway_2015.py`.
6. Validate VN30 stock hourly data with `scripts/research/validate_vn30_stocks_hourly_gateway_2015.py`.
7. Build readiness only with `scripts/research/build_vn30_2015_benchmark_readiness_manifest.py`.

Benchmarking can proceed only after the readiness manifest says yes. This plan does not run benchmark, model training, confidence sweeps, regime diagnostics, cost/slippage diagnostics, paper generation, or DOCX generation.

## Efficient Reverse Fetch Strategy

The active fetch implementation uses provider-current/latest available date as the upper bound and walks backward to each symbol's effective start. For stocks, `effective_start(ticker) = max(2015-01-01, first_trading_date)`, so pre-2015 listings start at `2015-01-01` and later listings start at their first trading date.

Fetchers attempt large chunks first. The default chunk is yearly; if a yearly chunk fails, it is split into quarterly chunks, then monthly chunks, then 5-day chunks, with 1-day chunks only as the final fallback. Empty large chunks are recorded as completed instead of expanding into tiny scans.

Already usable symbols are skipped unless `--force` is passed. Resume state is checkpointed per symbol under `reports/generated/vn30_hourly_2015/fetch_state/`, so reruns continue from the next unfinished chunk rather than restarting the symbol. Benchmarking remains blocked until the readiness manifest reports benchmark readiness `yes`.
