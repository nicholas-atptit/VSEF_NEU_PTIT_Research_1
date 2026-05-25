# VNStock Agent Data Guide Summary

## Correct Venv

Run provider scripts with:

```powershell
C:\Users\luong\.venv\Scripts\python.exe
```

Do not rely on bare `python` unless `sys.executable` confirms this same venv.

## Provider Priority

1. Repository adapter: `src/data/adapters/vnstock_adapter.py`
2. Direct `vnstock_data`
3. Legacy `vnstock`

Every fallback must be logged. Do not silently bypass the adapter.

## Correct Index Codes

Use only provider-supported names unless a fresh provider probe proves more:

```text
VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100
```

Do not use `VN30INDEX`. Do not use `VNXALL`.

## Correct OHLCV Call

Use `Quote.history` for OHLCV with explicit dates and hourly interval:

```python
quote.history(start=start, end=end, interval="1H")
```

KBS and VCI are valid history/OHLCV sources. Prefer KBS for supported index
codes and use VCI fallback for stocks when KBS fails.

## Do Not Do

- Do not fetch daily data for hourly benchmark use.
- Do not resample daily prices into hourly bars.
- Do not use `intraday()` for hourly candles unless the work is tick-level research.
- Do not hardcode unsupported sources.
- Do not reuse VN100 evidence for a new benchmark.
- Do not assume sample support proves full-history support.
- Do not request provider dates beyond provider-current coverage without checking returned rows.
