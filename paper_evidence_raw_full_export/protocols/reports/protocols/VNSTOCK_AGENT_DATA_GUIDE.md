# VNStock Agent Data Guide

## Purpose

This guide gives future agents clear rules for fetching correct Vietnamese
stock and index prices in this repository. It is meant to prevent four common
errors: bypassing the repository provider path, using unsupported index codes,
fetching the wrong frequency, or creating fake hourly data by resampling daily
prices.

## Environment Rule

Always run provider scripts with the repository-approved virtual environment:

```powershell
C:\Users\luong\.venv\Scripts\python.exe
```

Do not rely on bare `python` unless `sys.executable` confirms it resolves to
`C:\Users\luong\.venv\Scripts\python.exe`. Bare `python` may resolve to
`C:\Python\python.exe`, where provider packages can differ.

## Provider Priority

Use this provider order:

1. Repository adapter first: `src/data/adapters/vnstock_adapter.py`
2. `vnstock_data` direct second
3. Legacy `vnstock` fallback third

Every fallback must be logged with the symbol, source, interval, start date,
end date, failing provider, and exception text. Do not silently bypass the
adapter in VN30/VN100 scripts.

## Correct Price API

Use `Quote.history` for OHLCV history. Use `interval="1H"` for hourly bars and
pass explicit `start` and `end` dates. Do not use `intraday()` for hourly
candles unless the task is explicitly tick-level research. Do not fetch daily
data and resample it to hourly data.

## Supported Sources

KBS and VCI are valid sources for history/OHLCV. Prefer KBS for index codes
where supported. Use VCI fallback for stocks if KBS fails. Do not hardcode
unsupported sources or treat one-symbol sample success as proof of full-history
coverage.

## Supported Index Codes

Use provider-supported index names:

- `VNINDEX`
- `HNXINDEX`
- `UPCOMINDEX`
- `VN30`
- `HNX30`
- `VN100`

Do not use `VN30INDEX` unless a provider probe proves it. Do not use `VNXALL`
unless a provider probe proves it; it is unsupported in the current provider
probe.

## Standard Normalized Schema

For stock OHLCV:

```text
datetime,ticker,open,high,low,close,volume,provider,source,interval
```

For index OHLCV:

```text
datetime,index_code,open,high,low,close,volume,provider,source,interval
```

The `provider` column should identify the path that returned rows, such as
`repo_adapter`, `vnstock_data`, or `legacy_vnstock`. The `source` column should
record the provider source, such as `KBS` or `VCI`.

## Validation Rules

Validate fetched OHLCV before benchmark or research use:

- Parse timestamps into a non-null `datetime` column.
- Sort by `datetime`.
- Reject duplicate `datetime` values per symbol.
- Convert OHLCV columns to numeric values.
- Require `open`, `high`, `low`, and `close` to be greater than zero.
- Require `volume` to be greater than or equal to zero.
- Require `high >= max(open, close, low)` and `low <= min(open, close, high)`.
- Do not forward fill prices or volume.
- Do not create synthetic missing bars.

## Minimal Code Pattern

This helper shows the intended provider order and validation shape. Prefer a
real adapter method when one exists; the fallback path is explicit and logged.

```python
from __future__ import annotations

import logging

import pandas as pd

from src.data.adapters.vnstock_adapter import VnstockAdapter

logger = logging.getLogger(__name__)


def _normalize_history_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
    provider: str,
    source: str,
    interval: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    time_column = next(
        (name for name in ("datetime", "time", "timestamp", "date") if name in out.columns),
        None,
    )
    if time_column is None:
        raise ValueError(f"{symbol}: provider returned no time column")

    out = out.rename(columns={time_column: "datetime"})
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"])

    required = ["open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in out.columns]
    if missing:
        raise ValueError(f"{symbol}: missing OHLCV columns: {missing}")

    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=required)

    if out.empty:
        return out
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive OHLC price returned")
    if (out["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume returned")
    if (out["high"] < out[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError(f"{symbol}: high is inconsistent with OHLC")
    if (out["low"] > out[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError(f"{symbol}: low is inconsistent with OHLC")
    if out["datetime"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate datetime rows returned")

    out = out.sort_values("datetime").reset_index(drop=True)
    out["ticker"] = symbol
    out["provider"] = provider
    out["source"] = source
    out["interval"] = interval
    return out[
        ["datetime", "ticker", "open", "high", "low", "close", "volume", "provider", "source", "interval"]
    ]


def fetch_ohlcv_vnstock(symbol, start, end, interval="1H", sources=("KBS", "VCI")):
    errors = []

    adapter = VnstockAdapter(symbol_list=[symbol])
    adapter_fetch = getattr(adapter, "fetch_ohlcv", None)
    if callable(adapter_fetch):
        for source in sources:
            try:
                frame = adapter_fetch(symbol=symbol, start=start, end=end, interval=interval, source=source)
                normalized = _normalize_history_frame(
                    frame,
                    symbol=symbol,
                    provider="repo_adapter",
                    source=source,
                    interval=interval,
                )
                if not normalized.empty:
                    return normalized
            except Exception as exc:
                errors.append(f"repo_adapter/{source}: {exc}")
                logger.warning("vnstock_repo_adapter_fallback", extra={"symbol": symbol, "source": source})

    try:
        from vnstock_data import Quote as VnstockDataQuote
    except Exception as exc:
        errors.append(f"vnstock_data/import: {exc}")
    else:
        for source in sources:
            try:
                quote = VnstockDataQuote(symbol=symbol, source=source)
                frame = quote.history(start=start, end=end, interval=interval)
                normalized = _normalize_history_frame(
                    frame,
                    symbol=symbol,
                    provider="vnstock_data",
                    source=source,
                    interval=interval,
                )
                if not normalized.empty:
                    return normalized
            except Exception as exc:
                errors.append(f"vnstock_data/{source}: {exc}")
                logger.warning("vnstock_data_direct_fallback", extra={"symbol": symbol, "source": source})

    # Legacy fallback only. Keep this import out of the primary provider path.
    from vnstock import Quote

    for source in sources:
        try:
            quote = Quote(symbol=symbol, source=source)
            frame = quote.history(start=start, end=end, interval=interval)
            normalized = _normalize_history_frame(
                frame,
                symbol=symbol,
                provider="legacy_vnstock",
                source=source,
                interval=interval,
            )
            if not normalized.empty:
                return normalized
        except Exception as exc:
            errors.append(f"legacy_vnstock/{source}: {exc}")
            logger.warning("legacy_vnstock_fallback_failed", extra={"symbol": symbol, "source": source})

    raise RuntimeError(f"{symbol}: no provider/source returned OHLCV rows; errors={errors}")
```

For index output, rename `ticker` to `index_code` after validation and preserve
the same OHLCV, provider, source, and interval fields.

## Agent Checklist Before Running Benchmark

- Venv verified as `C:\Users\luong\.venv\Scripts\python.exe`.
- Provider path verified: adapter first, direct `vnstock_data` second, legacy
  `vnstock` third.
- Data fetched from provider.
- Validation passed.
- All required symbols are usable.
- Benchmark output directory is empty or explicitly versioned.
- No VN100 evidence is reused for a new benchmark.
- No daily or resampled data is used.

## Known Pitfalls

- Bare `python` may use `C:\Python\python.exe`.
- `vnstock_data` may exist in the venv but not in system Python.
- The VN30 index code is `VN30`, not `VN30INDEX`.
- `VNXALL` is unsupported in the current provider probe.
- Sample support does not prove full-history support.
- Provider-current date may be earlier than a requested future end date.

## References

- [thinh-vu/vnstock](https://github.com/thinh-vu/vnstock)
- [Vnstock historical price documentation](https://vnstocks.com/docs/vnstock/thong-ke-gia-lich-su)
- [vnstock_data Python documentation](https://vnstock-data-python.readthedocs.io/en/latest/)
