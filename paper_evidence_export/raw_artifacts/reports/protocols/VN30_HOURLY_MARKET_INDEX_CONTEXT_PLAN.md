# VN30 Hourly Market Index Context Plan

## Purpose

Market-index data is used for market context features, benchmark comparison, and regime/market-state diagnostics. It is not used as a stock target label.

## Corrected Index Coverage

- `VNINDEX` covers the long market-history context from `2005-01-01 00:00:00` through `2026-05-31 23:59:59`.
- `VN30INDEX` is used from `2012-02-06 00:00:00` through `2026-05-31 23:59:59`.
- `VNXALL` is used from `2016-10-24 00:00:00` through `2026-05-31 23:59:59`.
- The aligned evaluation/comparison period remains `2025-01-01 00:00:00` through `2026-05-31 23:59:59` for stocks and all three indices.

## Pre-Start Handling

Before the official start dates for `VN30INDEX` and `VNXALL`, the model may use stock-level features and `VNINDEX` context only.

Missing pre-start `VN30INDEX` and `VNXALL` rows are not data failures. Do not reconstruct pre-start index history unless a vendor provides it and marks it clearly as `vendor_backfilled` or `vendor_reconstructed`.

## Available-Window Study

The available-window VN30 hourly study uses a selected stock train/eval period after 2025. For that available-window period, `VNINDEX`, `VN30INDEX`, and `VNXALL` hourly context should be included only if exact-code local hourly data overlaps the selected stock design.

If exact-code local index data is missing, report it as a limitation. Do not alias stock symbols such as `VNI` or `VNX`, do not use daily data, and do not fabricate hourly index context.
