# VN30 Stock + Index Joint Panel Readiness Repair Decision

## Decision

Recommended strategy: **B. Rebuild normalized joint cache from raw fetch data**, with a strict pre-training gate.

The recovery audit found that the prior joint-panel readiness failure came from a wrong stock source path and incomplete/mislabeled active index cache. Correcting local source selection improves readiness from 1/36 usable under the first joint-panel audit to 35/36 usable, but the joint hourly panel is still not validation-ready because `VPL` lacks sufficient true intraday-hourly rows.

No training should run until the remaining stock gap is resolved and a repaired readiness audit reports 36/36 usable instruments.

## Evidence Summary

- Baseline60 stock evidence source found: yes.
- Traced output directory: `outputs/vn30_hourly_2015_horizon_relative_target_experiments`.
- Locked candidate: Random Forest, h=60, feature set C, absolute direction, final accuracy `0.602188`, final rows `3474`, final coverage `1.0`.
- The historical h=60 script reads stocks from `data/market_cache/vnstock_data/vn30/hourly_2015`.
- Repaired stock readiness from this path: 29/30 usable.
- Remaining stock blocker: `VPL`.
- `VPL` active intraday fallback has only 111 rows, `2026-02-11 14:00:00` to `2026-03-20 14:00:00`.
- `VPL` files in `data/market_cache/vnstock_data/vn30/hourly_2015` and `data/raw/vnstock_fetch/vn30_hourly_2015/VPL/` are midnight-only daily-like rows despite hourly naming.
- True intraday-hourly index data found for all six supported indices.
- Repaired index readiness: 6/6 usable.
- Corrected joint 36/36 readiness: no, currently 35/36.

## Option Review

### A. Correct Cache Path Only

Partially works, but is insufficient.

Correcting the stock path to `data/market_cache/vnstock_data/vn30/hourly_2015` recovers 29 usable stock instruments. Correcting the index path to the archived true-intraday cache plus the active VN100 file recovers 6 usable index instruments. The panel still fails 36/36 because `VPL` does not have enough validation-safe true intraday-hourly rows.

### B. Rebuild Normalized Joint Cache From Raw Fetch Data

Recommended.

The repository contains raw and active local sources that can support most of the normalized joint cache without fetching. A rebuild/repair pass should normalize only existing local rows into a documented joint cache, preserve original source files, and explicitly fail if `VPL` cannot be recovered as true intraday hourly from local raw data.

This option keeps the work as data-source repair rather than model experimentation.

### C. Use Prediction-Output Traces Only For Audit, Not Training

Acceptable only for evidence tracing.

The baseline60 result is traceable from summary outputs, but no adjacent prediction file was found in `outputs/vn30_hourly_2015_horizon_relative_target_experiments`. Prediction/report artifacts cannot substitute for OHLCV training inputs in the joint panel.

### D. Fetch Missing True Intraday Hourly Index Data

Not currently needed for indices.

The audit found true intraday-hourly data for all six supported indices. Fetching would require a separate approved protocol and is not part of this pass.

### E. Change Joint-Panel Frequency To Daily

Possible fallback, not recommended for the current hourly objective.

`VPL` has daily-like rows, and index daily-like data exists, but changing frequency would create a different research question. Daily results cannot be used as hourly results.

### F. Keep Hourly Stock-Only As Main Track And Index Benchmark Separate

Fallback if `VPL` cannot be recovered locally and fetch is not approved.

This preserves current evidence boundaries: stock-hourly baseline60 remains stock-only evidence, and index benchmark results remain separate. It does not satisfy the requested combined 36-instrument hourly panel.

## Required Next Gate

Before any joint-panel training:

1. Create a documented normalized-cache repair protocol.
2. Rebuild or repair the joint cache from existing local raw/cache sources only, unless a separate fetch protocol is approved.
3. Verify `VPL` has true intraday-hourly rows with enough train/validation/final availability.
4. Rerun readiness and require 36/36 usable instruments.
5. Only then run joint stock+index training under validation-only selection and scoring-only final evaluation.

## Guardrails

- Benchmark training run: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.
- Tags created: no.
- Tags pushed: no.
