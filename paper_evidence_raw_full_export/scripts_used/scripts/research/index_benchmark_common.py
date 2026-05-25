"""Shared helpers for supported Vietnamese index benchmark scripts."""
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_CODES = ("VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100")
OHLCV = ("open", "high", "low", "close", "volume")
INDEX_COLUMNS = ("datetime", "index_code", *OHLCV, "provider", "source", "frequency")
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "index_benchmark"
OUTPUT_DIR = REPO_ROOT / "outputs" / "index_directional_benchmark"
DAILY_CACHE = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "daily_2015"
HOURLY_CACHE = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
RAW_HOURLY = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "index_hourly"
ARCHIVE_ROOT = REPO_ROOT / "archive" / "generated_data_snapshots"
TARGET60 = 0.60


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_index_frame(path: Path, code: str | None = None, frequency: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(INDEX_COLUMNS))
    frame = pd.read_csv(path, low_memory=False)
    if "time" in frame.columns and "datetime" not in frame.columns:
        frame = frame.rename(columns={"time": "datetime"})
    if "ticker" in frame.columns and "index_code" not in frame.columns:
        frame = frame.rename(columns={"ticker": "index_code"})
    if "symbol" in frame.columns and "index_code" not in frame.columns:
        frame = frame.rename(columns={"symbol": "index_code"})
    if "index_code" not in frame.columns and code:
        frame["index_code"] = code
    if "provider" not in frame.columns:
        frame["provider"] = "unknown"
    if "source" not in frame.columns:
        frame["source"] = path.parent.name
    if "frequency" not in frame.columns and frequency:
        frame["frequency"] = frequency
    for column in INDEX_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[list(INDEX_COLUMNS)].copy()
    frame["index_code"] = frame["index_code"].astype(str).str.upper()
    if code:
        frame = frame[frame["index_code"].eq(code)].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
    return frame.reset_index(drop=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[dict[str, Any]], max_rows: int | None = None) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    display = rows if max_rows is None else rows[:max_rows]
    for row in display:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def fmt_pct(value: Any) -> str:
    try:
        number = float(value)
        if not math.isfinite(number):
            return ""
        return f"{number * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def validate_ohlcv(frame: pd.DataFrame, expected_frequency: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if frame.empty:
        reasons.append("no_rows")
        return False, reasons
    if frame["datetime"].isna().any():
        reasons.append("invalid_datetime")
    duplicates = int(frame.duplicated(["datetime"]).sum())
    if duplicates:
        reasons.append(f"duplicate_timestamps={duplicates}")
    if not frame["datetime"].is_monotonic_increasing:
        reasons.append("datetime_not_sorted")
    if frame[list(OHLCV)].isna().any().any():
        reasons.append("ohlcv_not_numeric")
    if not (frame[["open", "high", "low", "close"]] > 0).all().all():
        reasons.append("non_positive_price")
    if not (frame["volume"] >= 0).all():
        reasons.append("negative_volume")
    if not (
        (frame["high"] >= frame["low"]).all()
        and (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
        and (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
    ):
        reasons.append("ohlc_inconsistent")
    if not frame["frequency"].astype(str).eq(expected_frequency).all():
        reasons.append(f"frequency_not_{expected_frequency}")
    return not reasons, reasons


def year_coverage(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    years = sorted(frame["datetime"].dt.year.dropna().astype(int).unique().tolist())
    return ",".join(str(year) for year in years)


def split_frame(frame: pd.DataFrame, frequency: str) -> dict[str, pd.DataFrame]:
    frame = frame.sort_values("datetime").reset_index(drop=True)
    if frequency == "1D":
        return {
            "train": frame[(frame["datetime"] >= pd.Timestamp("2015-01-01")) & (frame["datetime"] <= pd.Timestamp("2023-12-31 23:59:59"))].copy(),
            "validation": frame[(frame["datetime"] >= pd.Timestamp("2024-01-01")) & (frame["datetime"] <= pd.Timestamp("2024-12-31 23:59:59"))].copy(),
            "final": frame[frame["datetime"] >= pd.Timestamp("2025-01-01")].copy(),
        }
    n = len(frame)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:val_end].copy(),
        "final": frame.iloc[val_end:].copy(),
    }


def readiness_thresholds(frequency: str) -> tuple[int, int, int]:
    if frequency == "1D":
        return 500, 80, 80
    return 400, 120, 120


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.sort_values("datetime").reset_index(drop=True).copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)
    out["return_1"] = close.pct_change(periods=1, fill_method=None)
    for lag in (2, 3, 5, 10, 20, 40, 60):
        out[f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
    for lag in (1, 2, 3, 5, 10):
        out[f"return_1_lag_{lag}"] = out["return_1"].shift(lag)
    for window in (5, 10, 20, 40, 60):
        min_periods = max(3, window // 2)
        out[f"rolling_return_mean_{window}"] = out["return_1"].rolling(window, min_periods=min_periods).mean()
        out[f"rolling_return_vol_{window}"] = out["return_1"].rolling(window, min_periods=min_periods).std()
        out[f"momentum_{window}"] = close / close.shift(window) - 1.0
        out[f"close_sma_ratio_{window}"] = close / close.rolling(window, min_periods=min_periods).mean() - 1.0
        vol_ma = volume.rolling(window, min_periods=min_periods).mean()
        out[f"volume_ma_ratio_{window}"] = volume / vol_ma - 1.0
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["volume_change_1"] = volume.pct_change(periods=1, fill_method=None)
    out["high_low_range"] = (out["high"] - out["low"]) / close
    out["open_close_spread"] = (out["close"] - out["open"]) / out["open"].replace(0.0, np.nan)
    out["close_position_in_range"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0.0, np.nan)
    feature_cols = [column for column in out.columns if column not in INDEX_COLUMNS]
    return out, feature_cols


def add_labels(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = frame.copy()
    out["future_close"] = out["close"].shift(-horizon)
    out["y_true"] = (out["future_close"] > out["close"]).astype(float)
    out.loc[out["future_close"].isna(), "y_true"] = np.nan
    out["previous_direction"] = (out["close"].diff() > 0).astype(int).shift(1)
    return out


def accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(true) & ~np.isnan(pred)
    total = int(mask.sum())
    if total == 0:
        return {"accuracy": math.nan, "correct": 0, "rows": 0}
    correct = int((true[mask] == pred[mask]).sum())
    return {"accuracy": correct / total, "correct": correct, "rows": total}


def baseline_predictions(frame: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = frame["y_true"].to_numpy(dtype=float)
    valid_y = y[~np.isnan(y)]
    majority = 1 if len(valid_y) and float(np.nanmean(valid_y)) >= 0.5 else 0
    ma_short = frame["close"].rolling(5, min_periods=3).mean()
    ma_long = frame["close"].rolling(20, min_periods=10).mean()
    moving_average = (ma_short > ma_long).astype(float).to_numpy()
    prev = frame["previous_direction"].fillna(majority).astype(float).to_numpy()
    return {
        "majority_class": np.full(len(frame), majority, dtype=float),
        "previous_direction": prev,
        "always_up": np.ones(len(frame), dtype=float),
        "random_seeded_direction": rng.integers(0, 2, size=len(frame)).astype(float),
        "moving_average_signal": moving_average,
    }


def timestamp_text(frame: pd.DataFrame, attr: str) -> str:
    if frame.empty:
        return ""
    value = getattr(frame["datetime"], attr)()
    return "" if pd.isna(value) else str(value)


def parse_date_text(value: str) -> datetime | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()
