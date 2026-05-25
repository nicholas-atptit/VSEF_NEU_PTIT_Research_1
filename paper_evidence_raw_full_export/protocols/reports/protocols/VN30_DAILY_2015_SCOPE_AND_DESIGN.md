# VN30 Daily 2015 - Scope and Design

## Purpose
This document defines a **separate daily benchmark track** for VN30, starting from 2015.

## Why This Track Exists
- Hourly stock data for 2015–2022 is **unavailable** from the vendor (KBS/VCI).
- Actual hourly stock coverage starts in **2023-09-11**.
- The previous `hourly_2015` naming was a design target, not actual data availability.
- To enable a 2015-start benchmark, we use **daily** data instead.

## Scope
- **Universe**: VN30 January 2025 review (30 tickers).
- **Frequency**: Daily only.
- **Start date**: 2015-01-01 (or first trading date per ticker, whichever is later).
- **Final evaluation**: 2025-01-01 to latest available date.
- **Validation**: 2024 (or rolling 2021-2024 if coverage allows).
- **Metric**: Pooled overall directional accuracy.
- **Target**: Future daily return direction.
- **Horizons**: h=1, h=5, h=10, h=20, h=60 trading days.
- **Coverage**: Full universe where possible.
- **No confidence abstention** for main overall target.

## Design Constraints
1. **Daily results are NOT directly comparable to hourly results.** They measure different frequencies and market dynamics.
2. **No daily-to-hourly resampling is allowed.** Daily data must remain daily.
3. **Hourly available-window benchmark remains separate.** The hourly track continues with its 2023-2026 coverage.
4. **No mixing of daily and hourly outputs.** Separate directories, separate reports, separate claims.

## Directory Structure
- **Cache**: `data/market_cache/vnstock_data/vn30/daily_2015/{TICKER}.csv`
- **Index cache**: `data/market_cache/vnstock_data/indices/daily_2015/{INDEX_CODE}.csv`
- **Outputs**: `outputs/vn30_daily_2015_benchmark/`
- **Reports**: `reports/results/VN30_DAILY_2015_*.md`, `reports/claims/VN30_DAILY_2015_*.md`, and `reports/protocols/VN30_DAILY_2015_*.md`

## Data Schema
```
datetime,ticker,open,high,low,close,volume,provider,source,frequency
```

## Validation Design
- **Preferred**: Rolling validation (2021, 2022, 2023, 2024) if daily coverage allows.
- **Fallback**: 2024-only validation if earlier years have insufficient coverage.

## Success Target
- Overall directional accuracy >=65% on final evaluation.
- Full universe coverage (30 tickers).
- No confidence abstention.

## Claim Separation
- **Daily claims**: Apply only to this daily track.
- **Hourly claims**: Apply only to the hourly track (2023-2026).
- **Unsafe**: Mixing daily and hourly claims, or resampling between frequencies.
