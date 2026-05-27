"""VN30 + configured-index direction, return, range, and ranking research lab.

This runner is deliberately an offline research artifact generator. It reads the
local VN30 stock and supported-index caches, builds horizon-aware feature/target
rows, enforces feature_timestamp/target_timestamp calendar splits, and writes
scoring-only diagnostics. It is not a trading, recommendation, live deployment,
or daily T+1 production system.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.index_benchmark_common import read_index_frame
from scripts.research.vn30_hourly_common import REPO_ROOT, standardize_hourly_frame

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, QuantileRegressor, Ridge
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        ndcg_score,
        roc_auc_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC
except Exception as exc:  # pragma: no cover - the lab cannot run without sklearn.
    raise SystemExit(f"scikit-learn is required for this lab: {exc}") from exc


OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_index_group_range_forecast"
RESULT_SUMMARY_PATH = REPO_ROOT / "reports" / "results" / "VN30_INDEX_GROUP_PRICE_RANGE_FORECAST_RESULT_SUMMARY.md"
CLAIM_BOUNDARY_PATH = REPO_ROOT / "reports" / "claims" / "VN30_INDEX_GROUP_PRICE_RANGE_FORECAST_CLAIM_BOUNDARY.md"
LOCAL_HISTORICAL_ONLY_SCOPE = (
    "Historical/local-data-only offline lab. The runner reads existing repository/cache files only; "
    "it does not fetch live data, call provider APIs, create real-time forecasts, or create production workflows."
)

STOCK_CACHE_DIRS = (
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015",
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly",
    REPO_ROOT / "data" / "hourly_market_split_data",
)
INDEX_CACHE_DIRS = (
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015",
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly",
    REPO_ROOT
    / "archive"
    / "generated_data_snapshots"
    / "vn30_hourly_pre_benchmark_20260514_062528"
    / "data"
    / "market_cache"
    / "vnstock_data"
    / "indices"
    / "hourly",
)
JOINT_UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_jan2025_joint_panel_universe.csv"

TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VALIDATION_START = pd.Timestamp("2024-01-01 00:00:00")
VALIDATION_END = pd.Timestamp("2024-12-31 23:59:59")
FINAL_START = pd.Timestamp("2025-01-01 00:00:00")

OHLCV = ["open", "high", "low", "close", "volume"]
DEFAULT_INDEX_CODES = ["VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VNXALL"]
DEFAULT_HORIZONS = [5, 10, 20, 40, 60]
RANDOM_SEED = 42
MAX_MODEL_ROWS_PER_SPLIT = 12000
MAX_FORECAST_PANEL_ROWS = 60000

FEATURE_GROUPS: dict[str, list[str]] = {
    "stock_momentum": [
        "asset_return_1_lag1",
        "asset_return_2_lag1",
        "asset_return_3_lag1",
        "asset_return_5_lag1",
        "asset_return_10_lag1",
        "asset_return_20_lag1",
        "asset_momentum_5",
        "asset_momentum_10",
        "asset_momentum_20",
        "asset_momentum_40",
        "asset_sma_ratio_10",
        "asset_sma_ratio_20",
    ],
    "index_momentum": [
        "VNINDEX_return_1_lag1",
        "VNINDEX_return_5_lag1",
        "VNINDEX_momentum_20",
        "VN30_return_1_lag1",
        "VN30_return_5_lag1",
        "VN30_momentum_20",
        "HNXINDEX_return_1_lag1",
        "HNX30_return_1_lag1",
        "UPCOMINDEX_return_1_lag1",
    ],
    "relative_strength": [
        "asset_minus_vn30_lag1",
        "asset_minus_vnindex_lag1",
        "asset_minus_vn30_5",
        "asset_minus_vnindex_5",
        "asset_return_rank_pct",
        "asset_momentum_rank_pct",
        "asset_volatility_rank_pct",
        "prior_top20_strength",
        "prior_top30_strength",
    ],
    "market_context": [
        "VNINDEX_volatility_20",
        "VN30_volatility_20",
        "HNXINDEX_volatility_20",
        "HNX30_volatility_20",
        "UPCOMINDEX_volatility_20",
        "vn30_vs_vnindex_lag1",
        "hnx_vs_vnindex_lag1",
        "upcom_vs_vnindex_lag1",
        "market_breadth_proxy",
    ],
    "volume_volatility": [
        "asset_volume_change_lag1",
        "asset_volume_z20",
        "asset_volume_ratio_20",
        "asset_volatility_5",
        "asset_volatility_10",
        "asset_volatility_20",
        "asset_volatility_40",
    ],
    "range_volatility": [
        "asset_range_pct_lag1",
        "asset_range_pct_5",
        "asset_range_pct_10",
        "asset_range_pct_20",
        "asset_atr_pct_14",
        "asset_close_position_lag1",
        "asset_gap_return",
    ],
    "regime_features": [
        "regime_trend_vnindex_20",
        "regime_trend_vn30_20",
        "regime_volatility_vnindex_20",
        "regime_volatility_vn30_20",
        "regime_breadth",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "asset_type_flag",
    ],
    "compact_stable_features": [
        "asset_return_1_lag1",
        "asset_return_5_lag1",
        "asset_momentum_20",
        "asset_volatility_20",
        "asset_range_pct_10",
        "asset_minus_vn30_lag1",
        "asset_minus_vnindex_lag1",
        "VNINDEX_return_1_lag1",
        "VN30_return_1_lag1",
        "market_breadth_proxy",
        "asset_return_rank_pct",
    ],
}
FEATURE_GROUPS["combined_features"] = sorted({feature for features in FEATURE_GROUPS.values() for feature in features})


@dataclass(frozen=True)
class RunConfig:
    frequency: str
    horizons: list[int]
    index_codes: list[str]
    timeout_seconds: int
    started_at: float


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return math.nan
    return numerator / denominator


def timestamp_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def pct_text(value: Any) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.2f}%"


def number_text(value: Any, decimals: int = 6) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{decimals}f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return timestamp_text(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def markdown_table(headers: list[str], rows: list[dict[str, Any]], max_rows: int | None = None) -> str:
    displayed = rows if max_rows is None else rows[:max_rows]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in displayed:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def parse_horizons(raw: str) -> list[int]:
    horizons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be positive integers.")
    return sorted(dict.fromkeys(horizons))


def check_timeout(config: RunConfig, phase: str) -> None:
    elapsed = time.time() - config.started_at
    if elapsed > config.timeout_seconds:
        raise TimeoutError(f"Timeout exceeded during {phase}: elapsed={elapsed:.1f}s limit={config.timeout_seconds}s")


def read_joint_universe_stocks() -> list[str]:
    if not JOINT_UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Missing joint-panel universe: {rel(JOINT_UNIVERSE_PATH)}")
    with JOINT_UNIVERSE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stocks = [
        str(row.get("instrument_code", "")).strip().upper()
        for row in rows
        if str(row.get("instrument_type", "")).strip().lower() == "stock"
        and str(row.get("active_for_joint_panel", "")).strip().lower() in {"true", "1", "yes", "y"}
    ]
    if len(stocks) != 30:
        raise ValueError(f"Expected 30 active VN30 stocks from {rel(JOINT_UNIVERSE_PATH)}, got {len(stocks)}")
    return stocks


def is_intraday(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    time_column = "datetime" if "datetime" in frame.columns else "feature_timestamp"
    hours = pd.to_datetime(frame[time_column], errors="coerce").dt.hour.dropna().unique()
    return len(hours) > 1


def load_stock_frame(ticker: str) -> tuple[pd.DataFrame, str, str]:
    loaded: list[tuple[pd.DataFrame, bool, str]] = []
    for directory in STOCK_CACHE_DIRS:
        path = directory / f"{ticker}.csv"
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path, low_memory=False)
            frame = standardize_hourly_frame(raw, ticker)
        except Exception:
            continue
        if frame.empty:
            continue
        loaded.append((frame, is_intraday(frame), rel(path)))
    if not loaded:
        return pd.DataFrame(columns=["asset_code", "asset_type", "feature_timestamp", *OHLCV]), "", "stock_cache_missing"
    frame, _intraday, source = max(loaded, key=lambda item: (item[1], len(item[0])))
    out = frame.rename(columns={"ticker": "asset_code", "datetime": "feature_timestamp"}).copy()
    out["asset_code"] = ticker
    out["asset_type"] = "stock"
    return out[["asset_code", "asset_type", "feature_timestamp", *OHLCV]].reset_index(drop=True), source, ""


def load_index_frame_from_cache(code: str) -> tuple[pd.DataFrame, str, str]:
    candidates: list[tuple[pd.DataFrame, bool, str]] = []
    for directory in INDEX_CACHE_DIRS:
        path = directory / f"{code}.csv"
        if not path.exists():
            continue
        try:
            frame = read_index_frame(path, code=code, frequency="1H")
        except Exception:
            continue
        if frame.empty:
            continue
        frame = frame.rename(columns={"index_code": "asset_code", "datetime": "feature_timestamp"}).copy()
        frame["asset_code"] = code
        frame["asset_type"] = "index"
        candidates.append((frame[["asset_code", "asset_type", "feature_timestamp", *OHLCV]], is_intraday(frame), rel(path)))
    if not candidates:
        return pd.DataFrame(columns=["asset_code", "asset_type", "feature_timestamp", *OHLCV]), "", "index_cache_missing_or_empty"
    frame, _intraday, source = max(candidates, key=lambda item: (item[1], len(item[0])))
    return frame.reset_index(drop=True), source, ""


def normalize_ohlcv(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["feature_timestamp"] = pd.to_datetime(out["feature_timestamp"], errors="coerce")
    out["asset_code"] = out["asset_code"].astype(str).str.upper().str.strip()
    out["asset_type"] = out["asset_type"].astype(str).str.lower().str.strip()
    for column in OHLCV:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["asset_code", "asset_type", "feature_timestamp", "open", "high", "low", "close"])
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)].copy()
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.sort_values(["asset_code", "feature_timestamp"]).drop_duplicates(
        ["asset_code", "feature_timestamp"],
        keep="last",
    )
    return out.reset_index(drop=True)


def load_assets(config: RunConfig) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], list[str]]:
    stocks = read_joint_universe_stocks()
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    for ticker in stocks:
        frame, source, skipped_reason = load_stock_frame(ticker)
        if not frame.empty:
            frames.append(frame)
        timestamps = pd.to_datetime(frame["feature_timestamp"], errors="coerce") if not frame.empty else pd.Series(dtype="datetime64[ns]")
        coverage_rows.append(
            {
                "asset_code": ticker,
                "asset_type": "stock",
                "requested": True,
                "used": not frame.empty,
                "skipped_reason": skipped_reason,
                "source_path": source,
                "expected_cache_paths": ";".join(rel(directory / f"{ticker}.csv") for directory in STOCK_CACHE_DIRS),
                "should_skip_if_missing": False,
                "rows": int(len(frame)),
                "first_timestamp": timestamp_text(timestamps.min()) if not timestamps.empty else "",
                "last_timestamp": timestamp_text(timestamps.max()) if not timestamps.empty else "",
                "split_train_feature_rows": int((timestamps <= TRAIN_END).sum()) if not timestamps.empty else 0,
                "split_validation_feature_rows": int(((timestamps >= VALIDATION_START) & (timestamps <= VALIDATION_END)).sum()) if not timestamps.empty else 0,
                "split_final_feature_rows": int((timestamps >= FINAL_START).sum()) if not timestamps.empty else 0,
            }
        )

    used_indices: list[str] = []
    skipped_indices: list[str] = []
    for code in config.index_codes:
        frame, source, skipped_reason = load_index_frame_from_cache(code)
        if not frame.empty:
            frames.append(frame)
            used_indices.append(code)
        else:
            skipped_indices.append(code)
        timestamps = pd.to_datetime(frame["feature_timestamp"], errors="coerce") if not frame.empty else pd.Series(dtype="datetime64[ns]")
        coverage_rows.append(
            {
                "asset_code": code,
                "asset_type": "index",
                "requested": True,
                "used": not frame.empty,
                "skipped_reason": skipped_reason,
                "source_path": source,
                "expected_cache_paths": ";".join(rel(directory / f"{code}.csv") for directory in INDEX_CACHE_DIRS),
                "should_skip_if_missing": True,
                "rows": int(len(frame)),
                "first_timestamp": timestamp_text(timestamps.min()) if not timestamps.empty else "",
                "last_timestamp": timestamp_text(timestamps.max()) if not timestamps.empty else "",
                "split_train_feature_rows": int((timestamps <= TRAIN_END).sum()) if not timestamps.empty else 0,
                "split_validation_feature_rows": int(((timestamps >= VALIDATION_START) & (timestamps <= VALIDATION_END)).sum()) if not timestamps.empty else 0,
                "split_final_feature_rows": int((timestamps >= FINAL_START).sum()) if not timestamps.empty else 0,
            }
        )

    if not frames:
        raise RuntimeError("No stock or index cache rows were available.")
    missing_stocks = [row for row in coverage_rows if row["asset_type"] == "stock" and not row["used"]]
    if missing_stocks:
        details = "; ".join(
            f"{row['asset_code']} missing {row['expected_cache_paths']} should_skip_if_missing={row['should_skip_if_missing']}"
            for row in missing_stocks
        )
        raise FileNotFoundError(
            "Required VN30 stock historical cache data is missing. "
            "This lab is historical/local-data-only and will not fetch live/provider data. "
            f"Affected assets: {details}"
        )
    return normalize_ohlcv(pd.concat(frames, ignore_index=True)), coverage_rows, used_indices, skipped_indices


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    std = series.rolling(window, min_periods=max(3, window // 2)).std()
    return (series - mean) / std.replace(0, np.nan)


def add_asset_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["asset_code", "feature_timestamp"]).copy()
    grouped = out.groupby("asset_code", group_keys=False)
    close = out["close"]
    out["asset_return_1"] = grouped["close"].pct_change(fill_method=None)
    out["asset_log_return_1"] = np.log(out["close"] / grouped["close"].shift(1))
    out["asset_gap_return"] = out["open"] / grouped["close"].shift(1) - 1.0
    out["asset_range_pct_lag0"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    out["asset_close_position"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0.0, np.nan)

    for lag in (1, 2, 3, 5, 10, 20):
        out[f"asset_return_{lag}"] = grouped["close"].pct_change(lag, fill_method=None)
        out[f"asset_return_{lag}_lag1"] = out.groupby("asset_code")[f"asset_return_{lag}"].shift(1)
    for window in (5, 10, 20, 40):
        min_periods = max(3, window // 2)
        out[f"asset_momentum_{window}"] = close / grouped["close"].shift(window) - 1.0
        out[f"asset_volatility_{window}"] = grouped["asset_log_return_1"].rolling(window, min_periods=min_periods).std().reset_index(level=0, drop=True)
        out[f"asset_range_pct_{window}"] = grouped["asset_range_pct_lag0"].rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        out[f"asset_sma_ratio_{window}"] = close / grouped["close"].rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True) - 1.0
    out["asset_range_pct_lag1"] = grouped["asset_range_pct_lag0"].shift(1)
    out["asset_close_position_lag1"] = grouped["asset_close_position"].shift(1)
    true_range = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - grouped["close"].shift(1)).abs(),
            (out["low"] - grouped["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["asset_atr_pct_14"] = true_range.groupby(out["asset_code"]).rolling(14, min_periods=7).mean().reset_index(level=0, drop=True) / out[
        "close"
    ].replace(0.0, np.nan)
    out["asset_volume_change_lag1"] = grouped["volume"].pct_change(fill_method=None).shift(1)
    volume_mean = grouped["volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["asset_volume_ratio_20"] = out["volume"] / volume_mean.replace(0.0, np.nan) - 1.0
    out["asset_volume_z20"] = grouped["volume"].transform(lambda series: rolling_zscore(series.astype(float), 20))
    out["asset_type_flag"] = (out["asset_type"] == "index").astype(float)

    month = out["feature_timestamp"].dt.month.astype(float)
    hour = out["feature_timestamp"].dt.hour.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out


def asset_feature_slice(frame: pd.DataFrame, code: str, columns: Iterable[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    selected = frame[frame["asset_code"].eq(code)][["feature_timestamp", *available]].copy()
    return selected.sort_values("feature_timestamp").drop_duplicates("feature_timestamp", keep="last")


def merge_asof_context(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right.empty:
        return left
    merged = pd.merge_asof(
        left.sort_values("feature_timestamp"),
        right.sort_values("feature_timestamp"),
        on="feature_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values(["asset_code", "feature_timestamp"]).reset_index(drop=True)


def add_market_context(features: pd.DataFrame, used_indices: list[str]) -> pd.DataFrame:
    out = features.sort_values(["asset_code", "feature_timestamp"]).copy()
    context_columns = ["asset_return_1_lag1", "asset_return_5_lag1", "asset_momentum_20", "asset_volatility_20"]
    for code in used_indices:
        idx = asset_feature_slice(out, code, context_columns)
        if idx.empty:
            continue
        rename = {
            "asset_return_1_lag1": f"{code}_return_1_lag1",
            "asset_return_5_lag1": f"{code}_return_5_lag1",
            "asset_momentum_20": f"{code}_momentum_20",
            "asset_volatility_20": f"{code}_volatility_20",
        }
        idx = idx.rename(columns=rename)
        out = merge_asof_context(out, idx)

    for column in [
        "VNINDEX_return_1_lag1",
        "VN30_return_1_lag1",
        "HNXINDEX_return_1_lag1",
        "HNX30_return_1_lag1",
        "UPCOMINDEX_return_1_lag1",
    ]:
        if column not in out.columns:
            out[column] = np.nan
    out["asset_minus_vn30_lag1"] = out["asset_return_1_lag1"] - out["VN30_return_1_lag1"]
    out["asset_minus_vnindex_lag1"] = out["asset_return_1_lag1"] - out["VNINDEX_return_1_lag1"]
    out["asset_minus_vn30_5"] = out["asset_return_5_lag1"] - out.get("VN30_return_5_lag1", np.nan)
    out["asset_minus_vnindex_5"] = out["asset_return_5_lag1"] - out.get("VNINDEX_return_5_lag1", np.nan)
    out["vn30_vs_vnindex_lag1"] = out["VN30_return_1_lag1"] - out["VNINDEX_return_1_lag1"]
    out["hnx_vs_vnindex_lag1"] = out["HNXINDEX_return_1_lag1"] - out["VNINDEX_return_1_lag1"]
    out["upcom_vs_vnindex_lag1"] = out["UPCOMINDEX_return_1_lag1"] - out["VNINDEX_return_1_lag1"]

    stock_rows = out[out["asset_type"].eq("stock")].copy()
    ranks = stock_rows[["asset_code", "feature_timestamp", "asset_return_1_lag1", "asset_momentum_20", "asset_volatility_20"]].copy()
    ranks["asset_return_rank_pct"] = ranks.groupby("feature_timestamp")["asset_return_1_lag1"].rank(pct=True)
    ranks["asset_momentum_rank_pct"] = ranks.groupby("feature_timestamp")["asset_momentum_20"].rank(pct=True)
    ranks["asset_volatility_rank_pct"] = ranks.groupby("feature_timestamp")["asset_volatility_20"].rank(pct=True)
    ranks["prior_top20_strength"] = (ranks["asset_return_rank_pct"] >= 0.80).astype(float)
    ranks["prior_top30_strength"] = (ranks["asset_return_rank_pct"] >= 0.70).astype(float)
    out = out.merge(
        ranks[
            [
                "asset_code",
                "feature_timestamp",
                "asset_return_rank_pct",
                "asset_momentum_rank_pct",
                "asset_volatility_rank_pct",
                "prior_top20_strength",
                "prior_top30_strength",
            ]
        ],
        on=["asset_code", "feature_timestamp"],
        how="left",
    )
    out["market_breadth_proxy"] = out.groupby("feature_timestamp")["prior_top30_strength"].transform("mean")
    out["regime_breadth"] = out["market_breadth_proxy"]
    out["regime_trend_vnindex_20"] = (out.get("VNINDEX_momentum_20", np.nan) > 0).astype(float)
    out["regime_trend_vn30_20"] = (out.get("VN30_momentum_20", np.nan) > 0).astype(float)
    vnindex_vol = out.get("VNINDEX_volatility_20", pd.Series(np.nan, index=out.index))
    vn30_vol = out.get("VN30_volatility_20", pd.Series(np.nan, index=out.index))
    out["regime_volatility_vnindex_20"] = (vnindex_vol > vnindex_vol.median(skipna=True)).astype(float)
    out["regime_volatility_vn30_20"] = (vn30_vol > vn30_vol.median(skipna=True)).astype(float)
    return out


def split_for_row(feature_timestamp: pd.Timestamp, target_timestamp: pd.Timestamp) -> str:
    if pd.isna(feature_timestamp) or pd.isna(target_timestamp):
        return "dropped"
    if feature_timestamp <= TRAIN_END and target_timestamp <= TRAIN_END:
        return "train"
    if VALIDATION_START <= feature_timestamp <= VALIDATION_END and VALIDATION_START <= target_timestamp <= VALIDATION_END:
        return "validation"
    if feature_timestamp >= FINAL_START and target_timestamp >= FINAL_START:
        return "final"
    return "gap"


def future_window_value(group: pd.DataFrame, column: str, horizon: int, reducer: str) -> pd.Series:
    values: list[float] = []
    raw = group[column].to_numpy(dtype=float)
    for idx in range(len(group)):
        window = raw[idx + 1 : idx + horizon + 1]
        if len(window) < horizon:
            values.append(math.nan)
        elif reducer == "max":
            values.append(float(np.nanmax(window)))
        elif reducer == "min":
            values.append(float(np.nanmin(window)))
        else:
            raise ValueError(f"Unsupported future-window reducer: {reducer}")
    return pd.Series(values, index=group.index, dtype=float)


def shifted_group_rolling(frame: pd.DataFrame, source: str, window: int, reducer: str, min_periods: int) -> pd.Series:
    pieces: list[pd.Series] = []
    for _asset_code, group in frame.groupby("asset_code", sort=False):
        series = group[source]
        if reducer == "mean":
            rolled = series.rolling(window, min_periods=min_periods).mean()
        elif reducer == "median":
            rolled = series.rolling(window, min_periods=min_periods).median()
        else:
            raise ValueError(f"Unsupported rolling reducer: {reducer}")
        pieces.append(rolled.shift(1))
    return pd.concat(pieces).sort_index()


def shifted_group_expanding_mean(frame: pd.DataFrame, source: str, min_periods: int) -> pd.Series:
    pieces: list[pd.Series] = []
    for _asset_code, group in frame.groupby("asset_code", sort=False):
        pieces.append(group[source].expanding(min_periods=min_periods).mean().shift(1))
    return pd.concat(pieces).sort_index()


def build_horizon_dataset(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = features.sort_values(["asset_code", "feature_timestamp"]).copy()
    grouped = out.groupby("asset_code", group_keys=False)
    out["horizon"] = horizon
    out["target_timestamp"] = grouped["feature_timestamp"].shift(-horizon)
    out["future_close_h"] = grouped["close"].shift(-horizon)
    out["future_high_h"] = grouped.apply(lambda group: future_window_value(group, "high", horizon, "max")).reset_index(level=0, drop=True).sort_index()
    out["future_low_h"] = grouped.apply(lambda group: future_window_value(group, "low", horizon, "min")).reset_index(level=0, drop=True).sort_index()
    out["forward_simple_return_h"] = out["future_close_h"] / out["close"] - 1.0
    out["forward_log_return_h"] = np.log(out["future_close_h"] / out["close"])
    out["absolute_direction"] = (out["future_close_h"] > out["close"]).astype(float)
    out.loc[out["future_close_h"].isna(), "absolute_direction"] = np.nan
    out["future_high_return_h"] = np.log(out["future_high_h"] / out["close"])
    out["future_low_return_h"] = np.log(out["future_low_h"] / out["close"])
    out["future_range_pct_h"] = (out["future_high_h"] - out["future_low_h"]) / out["close"]
    out["lag1_direction"] = (grouped["close"].diff() > 0).astype(float)
    out["lag1_direction"] = out.groupby("asset_code")["lag1_direction"].shift(1)
    out["rolling_mean_return_baseline"] = shifted_group_rolling(out, "asset_return_1", 20, "mean", 5) * horizon
    out["rolling_median_return_baseline"] = shifted_group_rolling(out, "asset_return_1", 20, "median", 5) * horizon
    out["historical_mean_return_baseline"] = shifted_group_expanding_mean(out, "asset_return_1", 20) * horizon
    out["rolling_range_pct_baseline"] = shifted_group_rolling(out, "asset_range_pct_lag0", 20, "mean", 5)
    out["naive_previous_range_baseline"] = out.groupby("asset_code")["asset_range_pct_lag0"].shift(1)
    out["atr_band_return"] = out["asset_atr_pct_14"] * math.sqrt(horizon)

    out = out.dropna(subset=["target_timestamp", "future_close_h", "forward_simple_return_h"]).copy()
    out = add_market_forward_returns(out)
    out["market_relative_vn30"] = (out["forward_simple_return_h"] > out["vn30_forward_return_h"]).astype(float)
    out.loc[out["forward_simple_return_h"].isna() | out["vn30_forward_return_h"].isna(), "market_relative_vn30"] = np.nan
    out["market_relative_vnindex"] = (out["forward_simple_return_h"] > out["vnindex_forward_return_h"]).astype(float)
    out.loc[out["forward_simple_return_h"].isna() | out["vnindex_forward_return_h"].isna(), "market_relative_vnindex"] = np.nan
    out["market_excess_return_h"] = out["forward_simple_return_h"] - out["vn30_forward_return_h"]
    out["volatility_adjusted_return_h"] = out["forward_simple_return_h"] / (out["asset_volatility_20"] * math.sqrt(horizon)).replace(0, np.nan)

    stock_mask = out["asset_type"].eq("stock")
    out.loc[stock_mask, "cross_sectional_forward_return_rank"] = out[stock_mask].groupby("feature_timestamp")[
        "forward_simple_return_h"
    ].rank(pct=True)
    out.loc[stock_mask, "market_excess_return_rank"] = out[stock_mask].groupby("feature_timestamp")[
        "market_excess_return_h"
    ].rank(pct=True)
    out["top20_forward_return"] = np.where(out["cross_sectional_forward_return_rank"] >= 0.80, 1.0, 0.0)
    out["top30_forward_return"] = np.where(out["cross_sectional_forward_return_rank"] >= 0.70, 1.0, 0.0)
    out.loc[~stock_mask | out["cross_sectional_forward_return_rank"].isna(), ["top20_forward_return", "top30_forward_return"]] = np.nan
    out["top_quantile_forward_return"] = out["top20_forward_return"]

    out["split"] = [
        split_for_row(feature_ts, target_ts)
        for feature_ts, target_ts in zip(out["feature_timestamp"], out["target_timestamp"], strict=False)
    ]
    valid = out[out["split"].isin(["train", "validation", "final"])].copy()
    return valid.reset_index(drop=True)


def add_market_forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for code, output_col in [("VN30", "vn30_forward_return_h"), ("VNINDEX", "vnindex_forward_return_h")]:
        idx = out[out["asset_code"].eq(code)][["feature_timestamp", "close"]].sort_values("feature_timestamp").copy()
        if idx.empty:
            out[output_col] = np.nan
            continue
        start_context = idx.rename(columns={"feature_timestamp": "feature_timestamp", "close": f"{code}_start_close"})
        target_context = idx.rename(columns={"feature_timestamp": "target_timestamp", "close": f"{code}_target_close"})
        out = pd.merge_asof(
            out.sort_values("feature_timestamp"),
            start_context.sort_values("feature_timestamp"),
            on="feature_timestamp",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values(["asset_code", "target_timestamp"])
        out = pd.merge_asof(
            out.sort_values("target_timestamp"),
            target_context.sort_values("target_timestamp"),
            on="target_timestamp",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values(["asset_code", "feature_timestamp"])
        out[output_col] = out[f"{code}_target_close"] / out[f"{code}_start_close"] - 1.0
        out = out.drop(columns=[f"{code}_start_close", f"{code}_target_close"])
    return out


def build_full_dataset(features: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    frames = [build_horizon_dataset(features, horizon) for horizon in horizons]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_dataset_audit(panel: pd.DataFrame, dataset: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (asset_type, asset_code), group in panel.groupby(["asset_type", "asset_code"]):
        data_group = dataset[dataset["asset_code"].eq(asset_code)]
        timestamps = group["feature_timestamp"]
        rows.append(
            {
                "asset_code": asset_code,
                "asset_type": asset_type,
                "source_rows": int(len(group)),
                "dataset_rows_all_horizons": int(len(data_group)),
                "feature_timestamp_min": timestamp_text(timestamps.min()),
                "feature_timestamp_max": timestamp_text(timestamps.max()),
                "target_timestamp_min": timestamp_text(data_group["target_timestamp"].min()) if not data_group.empty else "",
                "target_timestamp_max": timestamp_text(data_group["target_timestamp"].max()) if not data_group.empty else "",
                "train_rows": int(data_group["split"].eq("train").sum()) if not data_group.empty else 0,
                "validation_rows": int(data_group["split"].eq("validation").sum()) if not data_group.empty else 0,
                "final_rows": int(data_group["split"].eq("final").sum()) if not data_group.empty else 0,
                "horizons": ",".join(str(item) for item in sorted(data_group["horizon"].dropna().astype(int).unique())) if not data_group.empty else "",
                "has_market_context": bool(any(column.startswith("VNINDEX_") for column in data_group.columns)),
            }
        )
    return rows


def build_split_guard_audit(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (horizon, split), group in dataset.groupby(["horizon", "split"]):
        feature_min = group["feature_timestamp"].min()
        feature_max = group["feature_timestamp"].max()
        target_min = group["target_timestamp"].min()
        target_max = group["target_timestamp"].max()
        if split == "train":
            passed = bool((group["feature_timestamp"] <= TRAIN_END).all() and (group["target_timestamp"] <= TRAIN_END).all())
            rule = "feature_timestamp <= 2023-12-31 23:59:59 and target_timestamp <= 2023-12-31 23:59:59"
        elif split == "validation":
            passed = bool(
                ((group["feature_timestamp"] >= VALIDATION_START) & (group["feature_timestamp"] <= VALIDATION_END)).all()
                and ((group["target_timestamp"] >= VALIDATION_START) & (group["target_timestamp"] <= VALIDATION_END)).all()
            )
            rule = "feature_timestamp and target_timestamp inside 2024"
        else:
            passed = bool((group["feature_timestamp"] >= FINAL_START).all() and (group["target_timestamp"] >= FINAL_START).all())
            rule = "feature_timestamp and target_timestamp >= 2025-01-01"
        rows.append(
            {
                "horizon": int(horizon),
                "split": split,
                "rows": int(len(group)),
                "feature_timestamp_min": timestamp_text(feature_min),
                "feature_timestamp_max": timestamp_text(feature_max),
                "target_timestamp_min": timestamp_text(target_min),
                "target_timestamp_max": timestamp_text(target_max),
                "guard_rule": rule,
                "guard_passed": passed,
            }
        )
    return rows


def build_target_audit(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    target_specs = {
        "direction": [
            "absolute_direction",
            "market_relative_vn30",
            "market_relative_vnindex",
            "top_quantile_forward_return",
            "top20_forward_return",
            "top30_forward_return",
        ],
        "return_price": [
            "forward_simple_return_h",
            "forward_log_return_h",
            "future_close_h",
            "market_excess_return_h",
            "volatility_adjusted_return_h",
        ],
        "range": [
            "future_high_h",
            "future_low_h",
            "future_high_return_h",
            "future_low_return_h",
            "future_range_pct_h",
        ],
        "ranking": [
            "cross_sectional_forward_return_rank",
            "market_excess_return_rank",
        ],
    }
    rows: list[dict[str, Any]] = []
    for task, targets in target_specs.items():
        for target in targets:
            if target not in dataset.columns:
                continue
            for horizon, group in dataset.groupby("horizon"):
                valid = group[target].dropna()
                rows.append(
                    {
                        "task": task,
                        "target_variant": target,
                        "horizon": int(horizon),
                        "valid_rows": int(len(valid)),
                        "stock_rows": int(group[group["asset_type"].eq("stock")][target].notna().sum()),
                        "index_rows": int(group[group["asset_type"].eq("index")][target].notna().sum()),
                        "positive_ratio": float(valid.mean()) if target in {"absolute_direction", "market_relative_vn30", "market_relative_vnindex", "top_quantile_forward_return", "top20_forward_return", "top30_forward_return"} and len(valid) else math.nan,
                        "mean": float(valid.mean()) if len(valid) else math.nan,
                        "std": float(valid.std()) if len(valid) > 1 else math.nan,
                        "min": float(valid.min()) if len(valid) else math.nan,
                        "max": float(valid.max()) if len(valid) else math.nan,
                        "target_timestamp_min": timestamp_text(group.loc[group[target].notna(), "target_timestamp"].min()) if len(valid) else "",
                        "target_timestamp_max": timestamp_text(group.loc[group[target].notna(), "target_timestamp"].max()) if len(valid) else "",
                    }
                )
    rows.extend(
        [
            {
                "task": "range",
                "target_variant": "close_interval_p10_p50_p90",
                "horizon": int(horizon),
                "valid_rows": int(dataset[dataset["horizon"].eq(horizon)]["future_close_h"].notna().sum()),
                "stock_rows": int(dataset[(dataset["horizon"].eq(horizon)) & dataset["asset_type"].eq("stock")]["future_close_h"].notna().sum()),
                "index_rows": int(dataset[(dataset["horizon"].eq(horizon)) & dataset["asset_type"].eq("index")]["future_close_h"].notna().sum()),
                "positive_ratio": math.nan,
                "mean": math.nan,
                "std": math.nan,
                "min": math.nan,
                "max": math.nan,
                "target_timestamp_min": timestamp_text(dataset[dataset["horizon"].eq(horizon)]["target_timestamp"].min()),
                "target_timestamp_max": timestamp_text(dataset[dataset["horizon"].eq(horizon)]["target_timestamp"].max()),
                "notes": "estimated by quantile/band interval models where available",
            }
            for horizon in sorted(dataset["horizon"].unique())
        ]
    )
    return rows


def detect_qml_artifacts() -> list[str]:
    candidates = [
        REPO_ROOT / "reports" / "generated" / "vn30_qml_forecasting" / "qml_feature_audit.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_model_universe_direction_price" / "v7_qml_extended_results.csv",
    ]
    return [rel(path) for path in candidates if path.exists()]


def build_feature_audit(dataset: pd.DataFrame, qml_artifacts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        available = [column for column in columns if column in dataset.columns]
        usable = [column for column in available if dataset[column].notna().any()]
        rows.append(
            {
                "feature_group": group_name,
                "configured_feature_count": len(columns),
                "available_feature_count": len(available),
                "usable_feature_count": len(usable),
                "usable_features": ",".join(usable),
                "uses_information_after_feature_timestamp": False,
                "uses_target_timestamp": False,
                "notes": "lagged, rolling, contemporaneous OHLCV metadata, and as-of index context only",
            }
        )
    rows.append(
        {
            "feature_group": "qml_kernel_features_optional",
            "configured_feature_count": 0,
            "available_feature_count": len(qml_artifacts),
            "usable_feature_count": 0,
            "usable_features": "",
            "uses_information_after_feature_timestamp": False,
            "uses_target_timestamp": False,
            "notes": "existing artifacts detected: " + ", ".join(qml_artifacts) if qml_artifacts else "no existing QML kernel artifacts detected",
        }
    )
    return rows


def selected_feature_columns(dataset: pd.DataFrame, group_name: str = "combined_features") -> list[str]:
    candidates = FEATURE_GROUPS[group_name]
    available = [column for column in candidates if column in dataset.columns]
    usable = [column for column in available if dataset[column].notna().sum() >= 20]
    return usable


def sample_for_model(frame: pd.DataFrame, max_rows: int = MAX_MODEL_ROWS_PER_SPLIT) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    sampled_indices: list[int] = []
    for _key, group in frame.groupby(["split", "horizon", "asset_type"], sort=False):
        group_quota = max(1, int(max_rows * len(group) / len(frame)))
        sampled_indices.extend(group.sample(n=min(len(group), group_quota), random_state=RANDOM_SEED).index.tolist())
    sampled = frame.loc[sorted(set(sampled_indices))].copy()
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=RANDOM_SEED).copy()
    return sampled.sort_values(["split", "horizon", "asset_type", "asset_code", "feature_timestamp"]).reset_index(drop=True)


def prepare_xy(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    task: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_valid = train.dropna(subset=[target]).copy()
    eval_valid = eval_frame.dropna(subset=[target]).copy()
    if task == "classification":
        train_valid = train_valid[train_valid[target].isin([0.0, 1.0])]
        eval_valid = eval_valid[eval_valid[target].isin([0.0, 1.0])]
    x_train = train_valid[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_valid[target]
    x_eval = eval_valid[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_eval = eval_valid[target]
    return x_train, y_train, x_eval, y_eval


def classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    y_prob: Iterable[Any] | None,
    eval_frame: pd.DataFrame,
    baseline_balanced_accuracy: float,
) -> dict[str, Any]:
    true = pd.Series(y_true).reset_index(drop=True).astype(float)
    pred = pd.Series(y_pred).reset_index(drop=True).astype(float)
    aligned_source = eval_frame.reset_index(drop=True)
    valid = true.notna() & pred.notna()
    if int(valid.sum()) == 0:
        return {
            "rows": 0,
            "raw_accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "macro_f1": math.nan,
            "mcc": math.nan,
            "auc": math.nan,
            "brier_score": math.nan,
            "lift_over_strongest_simple_baseline": math.nan,
            "prediction_balance": math.nan,
            "asset_level_stability": math.nan,
            "month_stability": math.nan,
            "quarter_stability": math.nan,
        }
    true_v = true[valid].astype(int)
    pred_v = pred[valid].astype(int)
    prob_v = None
    if y_prob is not None:
        prob = pd.Series(y_prob).reset_index(drop=True).astype(float)
        prob_v = prob[valid]
    metrics = {
        "rows": int(valid.sum()),
        "raw_accuracy": float(accuracy_score(true_v, pred_v)),
        "balanced_accuracy": float(balanced_accuracy_score(true_v, pred_v)) if true_v.nunique() > 1 else math.nan,
        "macro_f1": float(f1_score(true_v, pred_v, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true_v, pred_v)) if true_v.nunique() > 1 and pred_v.nunique() > 1 else 0.0,
        "auc": math.nan,
        "brier_score": math.nan,
        "lift_over_strongest_simple_baseline": math.nan,
        "prediction_balance": float(pred_v.mean()),
        "asset_level_stability": math.nan,
        "month_stability": math.nan,
        "quarter_stability": math.nan,
    }
    if prob_v is not None and prob_v.notna().all() and true_v.nunique() > 1:
        try:
            metrics["auc"] = float(roc_auc_score(true_v, prob_v))
        except ValueError:
            metrics["auc"] = math.nan
        try:
            metrics["brier_score"] = float(brier_score_loss(true_v, prob_v.clip(0, 1)))
        except ValueError:
            metrics["brier_score"] = math.nan
    if math.isfinite(baseline_balanced_accuracy) and math.isfinite(metrics["balanced_accuracy"]):
        metrics["lift_over_strongest_simple_baseline"] = metrics["balanced_accuracy"] - baseline_balanced_accuracy

    aligned = aligned_source.loc[valid[valid].index].copy()
    aligned["_correct"] = (true_v.to_numpy() == pred_v.to_numpy()).astype(float)
    by_asset = aligned.groupby("asset_code")["_correct"].mean()
    by_month = aligned.groupby(aligned["feature_timestamp"].dt.to_period("M"))["_correct"].mean()
    by_quarter = aligned.groupby(aligned["feature_timestamp"].dt.to_period("Q"))["_correct"].mean()
    metrics["asset_level_stability"] = float(by_asset.min()) if len(by_asset) else math.nan
    metrics["month_stability"] = float(by_month.min()) if len(by_month) else math.nan
    metrics["quarter_stability"] = float(by_quarter.min()) if len(by_quarter) else math.nan
    return metrics


def regression_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    eval_frame: pd.DataFrame,
    baseline_predictions: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    true = pd.Series(y_true).reset_index(drop=True).astype(float)
    pred = pd.Series(y_pred).reset_index(drop=True).astype(float)
    aligned_source = eval_frame.reset_index(drop=True)
    valid = true.notna() & pred.notna() & np.isfinite(true) & np.isfinite(pred)
    if int(valid.sum()) == 0:
        return {
            "rows": 0,
            "rmse": math.nan,
            "mae": math.nan,
            "smape": math.nan,
            "correlation_pred_actual": math.nan,
            "sign_accuracy": math.nan,
            "rank_ic": math.nan,
        }
    true_v = true[valid].astype(float)
    pred_v = pred[valid].astype(float)
    err = pred_v - true_v
    denom = (true_v.abs() + pred_v.abs()).replace(0, np.nan)
    metrics = {
        "rows": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "mae": float(np.mean(np.abs(err))),
        "smape": float(np.nanmean(2.0 * np.abs(err) / denom)),
        "correlation_pred_actual": float(pd.Series(pred_v).corr(pd.Series(true_v))) if len(true_v) > 2 else math.nan,
        "sign_accuracy": float((np.sign(true_v) == np.sign(pred_v)).mean()),
        "rank_ic": math.nan,
    }
    aligned = aligned_source.loc[valid[valid].index].copy()
    aligned["_true"] = true_v.to_numpy()
    aligned["_pred"] = pred_v.to_numpy()
    rank_ics = []
    for _timestamp, group in aligned.groupby("feature_timestamp"):
        if len(group) >= 3 and group["_true"].nunique() > 1 and group["_pred"].nunique() > 1:
            value = group["_true"].rank().corr(group["_pred"].rank())
            if math.isfinite(finite_float(value)):
                rank_ics.append(float(value))
    metrics["rank_ic"] = float(np.mean(rank_ics)) if rank_ics else math.nan
    if baseline_predictions:
        for name, base_pred in baseline_predictions.items():
            base_metrics = regression_metrics(true, base_pred, eval_frame, None)
            metrics[f"improvement_vs_{name}_rmse"] = base_metrics["rmse"] - metrics["rmse"] if math.isfinite(base_metrics["rmse"]) else math.nan
            metrics[f"improvement_vs_{name}_mae"] = base_metrics["mae"] - metrics["mae"] if math.isfinite(base_metrics["mae"]) else math.nan
    return metrics


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.nanmean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def interval_metrics(
    eval_frame: pd.DataFrame,
    low_return: Iterable[Any],
    mid_return: Iterable[Any],
    high_return: Iterable[Any],
) -> dict[str, Any]:
    actual = eval_frame["forward_simple_return_h"].astype(float).reset_index(drop=True)
    actual_high = eval_frame["future_high_return_h"].astype(float).reset_index(drop=True)
    actual_low = eval_frame["future_low_return_h"].astype(float).reset_index(drop=True)
    low = pd.Series(low_return).astype(float).reset_index(drop=True)
    mid = pd.Series(mid_return).astype(float).reset_index(drop=True)
    high = pd.Series(high_return).astype(float).reset_index(drop=True)
    valid = actual.notna() & low.notna() & mid.notna() & high.notna() & np.isfinite(actual) & np.isfinite(low) & np.isfinite(high)
    if int(valid.sum()) == 0:
        return {
            "rows": 0,
            "pinball_loss_p10": math.nan,
            "pinball_loss_p50": math.nan,
            "pinball_loss_p90": math.nan,
            "interval_coverage": math.nan,
            "average_interval_width": math.nan,
            "low_breach_rate": math.nan,
            "high_breach_rate": math.nan,
            "winkler_score": math.nan,
            "actual_close_inside_predicted_range": math.nan,
            "actual_high_low_coverage": math.nan,
        }
    actual_v = actual[valid]
    low_v = low[valid]
    mid_v = mid[valid]
    high_v = high[valid]
    coverage = ((actual_v >= low_v) & (actual_v <= high_v)).astype(float)
    alpha = 0.20
    width = high_v - low_v
    below = (low_v - actual_v).clip(lower=0)
    above = (actual_v - high_v).clip(lower=0)
    winkler = width + (2 / alpha) * below + (2 / alpha) * above
    high_low_valid = valid & actual_high.notna() & actual_low.notna()
    actual_high_low_coverage = math.nan
    if int(high_low_valid.sum()):
        actual_high_low_coverage = float(((actual_low[high_low_valid] >= low[high_low_valid]) & (actual_high[high_low_valid] <= high[high_low_valid])).mean())
    return {
        "rows": int(valid.sum()),
        "pinball_loss_p10": pinball_loss(actual_v, low_v, 0.10),
        "pinball_loss_p50": pinball_loss(actual_v, mid_v, 0.50),
        "pinball_loss_p90": pinball_loss(actual_v, high_v, 0.90),
        "interval_coverage": float(coverage.mean()),
        "average_interval_width": float(width.mean()),
        "low_breach_rate": float((actual_v < low_v).mean()),
        "high_breach_rate": float((actual_v > high_v).mean()),
        "winkler_score": float(winkler.mean()),
        "actual_close_inside_predicted_range": float(coverage.mean()),
        "actual_high_low_coverage": actual_high_low_coverage,
    }


def ranking_metrics(eval_frame: pd.DataFrame, score: Iterable[Any]) -> dict[str, Any]:
    frame = eval_frame.copy().reset_index(drop=True)
    frame["_score"] = pd.Series(score).astype(float)
    frame = frame[
        frame["asset_type"].eq("stock")
        & frame["_score"].notna()
        & frame["forward_simple_return_h"].notna()
        & frame["cross_sectional_forward_return_rank"].notna()
    ].copy()
    if frame.empty:
        return {
            "rows": 0,
            "spearman_ic": math.nan,
            "rank_ic": math.nan,
            "ndcg_at_5": math.nan,
            "ndcg_at_10": math.nan,
            "top20_precision": math.nan,
            "top30_precision": math.nan,
            "top_decile_realized_return_diagnostic": math.nan,
        }
    spearman_values: list[float] = []
    ndcg5_values: list[float] = []
    ndcg10_values: list[float] = []
    top20_hits: list[float] = []
    top30_hits: list[float] = []
    top_decile_returns: list[float] = []
    for _timestamp, group in frame.groupby("feature_timestamp"):
        if len(group) < 5 or group["_score"].nunique() <= 1 or group["forward_simple_return_h"].nunique() <= 1:
            continue
        corr = group["_score"].rank().corr(group["forward_simple_return_h"].rank())
        if math.isfinite(finite_float(corr)):
            spearman_values.append(float(corr))
        y_true = group["forward_simple_return_h"].rank(pct=True).to_numpy().reshape(1, -1)
        y_score = group["_score"].to_numpy().reshape(1, -1)
        try:
            ndcg5_values.append(float(ndcg_score(y_true, y_score, k=min(5, len(group)))))
            ndcg10_values.append(float(ndcg_score(y_true, y_score, k=min(10, len(group)))))
        except ValueError:
            pass
        selected20 = group.nlargest(max(1, math.ceil(len(group) * 0.20)), "_score")
        selected30 = group.nlargest(max(1, math.ceil(len(group) * 0.30)), "_score")
        top20_hits.append(float((selected20["cross_sectional_forward_return_rank"] >= 0.80).mean()))
        top30_hits.append(float((selected30["cross_sectional_forward_return_rank"] >= 0.70).mean()))
        top_decile = group.nlargest(max(1, math.ceil(len(group) * 0.10)), "_score")
        top_decile_returns.append(float(top_decile["forward_simple_return_h"].mean()))
    return {
        "rows": int(len(frame)),
        "spearman_ic": float(np.mean(spearman_values)) if spearman_values else math.nan,
        "rank_ic": float(np.mean(spearman_values)) if spearman_values else math.nan,
        "ndcg_at_5": float(np.mean(ndcg5_values)) if ndcg5_values else math.nan,
        "ndcg_at_10": float(np.mean(ndcg10_values)) if ndcg10_values else math.nan,
        "top20_precision": float(np.mean(top20_hits)) if top20_hits else math.nan,
        "top30_precision": float(np.mean(top30_hits)) if top30_hits else math.nan,
        "top_decile_realized_return_diagnostic": float(np.mean(top_decile_returns)) if top_decile_returns else math.nan,
    }


def make_preprocessor_model(model: Any, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def classifier_specs() -> list[tuple[str, Any, bool, bool]]:
    specs: list[tuple[str, Any, bool, bool]] = [
        ("logistic_regression", LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_SEED), True, True),
        ("calibrated_logistic", CalibratedClassifierCV(LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_SEED), cv=3), True, True),
        ("linear_svm", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=3000), True, False),
        ("rbf_svm", SVC(kernel="rbf", class_weight="balanced", probability=True, C=1.0, gamma="scale", random_state=RANDOM_SEED), True, True),
        ("random_forest", RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=10, n_jobs=-1, class_weight="balanced_subsample", random_state=RANDOM_SEED), False, True),
        ("extratrees", ExtraTreesClassifier(n_estimators=140, max_depth=8, min_samples_leaf=10, n_jobs=-1, class_weight="balanced", random_state=RANDOM_SEED), False, True),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, random_state=RANDOM_SEED), False, True),
        ("small_mlp", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.001, max_iter=120, random_state=RANDOM_SEED, early_stopping=True), True, True),
        ("regime_aware_classifier", RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=15, n_jobs=-1, class_weight="balanced_subsample", random_state=RANDOM_SEED + 1), False, True),
    ]
    if importlib.util.find_spec("xgboost") is not None:
        try:
            from xgboost import XGBClassifier

            specs.append(
                (
                    "xgboost",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        n_jobs=1,
                        random_state=RANDOM_SEED,
                    ),
                    False,
                    True,
                )
            )
        except Exception:
            pass
    if importlib.util.find_spec("lightgbm") is not None:
        try:
            from lightgbm import LGBMClassifier

            specs.append(
                (
                    "lightgbm",
                    LGBMClassifier(
                        n_estimators=140,
                        max_depth=4,
                        learning_rate=0.04,
                        num_leaves=15,
                        min_child_samples=25,
                        verbosity=-1,
                        random_state=RANDOM_SEED,
                    ),
                    False,
                    True,
                )
            )
        except Exception:
            pass
    if importlib.util.find_spec("catboost") is not None:
        try:
            from catboost import CatBoostClassifier

            specs.append(
                (
                    "catboost",
                    CatBoostClassifier(iterations=120, depth=4, learning_rate=0.05, verbose=False, random_seed=RANDOM_SEED),
                    False,
                    True,
                )
            )
        except Exception:
            pass
    return specs


def regressor_specs() -> list[tuple[str, Any, bool]]:
    specs: list[tuple[str, Any, bool]] = [
        ("ridge", Ridge(alpha=1.0), True),
        ("lasso", Lasso(alpha=0.0005, max_iter=2000, random_state=RANDOM_SEED), True),
        ("elasticnet", ElasticNet(alpha=0.0005, l1_ratio=0.30, max_iter=2000, random_state=RANDOM_SEED), True),
        ("hist_gradient_boosting_regressor", HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, random_state=RANDOM_SEED), False),
        ("random_forest_regressor", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_SEED), False),
        ("gradient_boosting_regressor", GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED), False),
        ("extratrees_regressor", ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_SEED), False),
    ]
    if importlib.util.find_spec("xgboost") is not None:
        try:
            from xgboost import XGBRegressor

            specs.append(
                (
                    "xgboost_regressor",
                    XGBRegressor(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        n_jobs=1,
                        random_state=RANDOM_SEED,
                    ),
                    False,
                )
            )
        except Exception:
            pass
    if importlib.util.find_spec("lightgbm") is not None:
        try:
            from lightgbm import LGBMRegressor

            specs.append(
                (
                    "lightgbm_regressor",
                    LGBMRegressor(
                        n_estimators=140,
                        max_depth=4,
                        learning_rate=0.04,
                        num_leaves=15,
                        min_child_samples=25,
                        verbosity=-1,
                        random_state=RANDOM_SEED,
                    ),
                    False,
                )
            )
        except Exception:
            pass
    if importlib.util.find_spec("catboost") is not None:
        try:
            from catboost import CatBoostRegressor

            specs.append(
                (
                    "catboost_regressor",
                    CatBoostRegressor(iterations=120, depth=4, learning_rate=0.05, verbose=False, random_seed=RANDOM_SEED),
                    False,
                )
            )
        except Exception:
            pass
    return specs


def get_probabilities(model: Pipeline, x_eval: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_eval)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
        except Exception:
            pass
    if hasattr(estimator, "decision_function"):
        try:
            scores = model.decision_function(x_eval)
            return 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))
        except Exception:
            pass
    return np.asarray(pred, dtype=float)


def direction_baseline_predictions(train: pd.DataFrame, eval_frame: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    valid_train = train.dropna(subset=[target])
    majority = 1.0 if len(valid_train) and valid_train[target].mean() >= 0.5 else 0.0
    positive_rate = float(valid_train[target].mean()) if len(valid_train) else 0.5
    rng = np.random.default_rng(RANDOM_SEED)
    return {
        "always_up": np.ones(len(eval_frame), dtype=float),
        "always_down": np.zeros(len(eval_frame), dtype=float),
        "majority_class": np.full(len(eval_frame), majority, dtype=float),
        "lag1_direction": eval_frame["lag1_direction"].fillna(majority).to_numpy(dtype=float),
        "random_same_class_balance": (rng.random(len(eval_frame)) < positive_rate).astype(float),
    }


def run_direction_baselines(dataset: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], float]]:
    rows: list[dict[str, Any]] = []
    strongest: dict[tuple[int, str, str], float] = {}
    targets = ["absolute_direction", "market_relative_vn30", "market_relative_vnindex", "top20_forward_return", "top30_forward_return"]
    for horizon in sorted(dataset["horizon"].unique()):
        hframe = dataset[dataset["horizon"].eq(horizon)]
        train = hframe[hframe["split"].eq("train")]
        for target in targets:
            if target not in hframe.columns:
                continue
            for split in ["validation", "final"]:
                eval_frame = hframe[hframe["split"].eq(split)].dropna(subset=[target]).copy()
                if eval_frame.empty:
                    continue
                predictions = direction_baseline_predictions(train, eval_frame, target)
                best_bal = -math.inf
                for name, pred in predictions.items():
                    metrics = classification_metrics(eval_frame[target], pred, pred, eval_frame, math.nan)
                    row = {
                        "task": "direction",
                        "baseline_id": name,
                        "target_variant": target,
                        "horizon": int(horizon),
                        "split": split,
                        "asset_scope": "stock_index_group",
                        **metrics,
                    }
                    rows.append(row)
                    if math.isfinite(metrics["balanced_accuracy"]):
                        best_bal = max(best_bal, metrics["balanced_accuracy"])
                strongest[(int(horizon), target, split)] = best_bal if math.isfinite(best_bal) else math.nan
    return rows, strongest


def return_baseline_predictions(train: pd.DataFrame, eval_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    train_returns = train["forward_simple_return_h"].dropna().astype(float)
    historical_mean = float(train_returns.mean()) if len(train_returns) else 0.0
    return {
        "random_walk_close": np.zeros(len(eval_frame), dtype=float),
        "last_close": np.zeros(len(eval_frame), dtype=float),
        "historical_mean_return": np.full(len(eval_frame), historical_mean, dtype=float),
        "rolling_mean_return": eval_frame["rolling_mean_return_baseline"].fillna(historical_mean).to_numpy(dtype=float),
        "rolling_median_return": eval_frame["rolling_median_return_baseline"].fillna(historical_mean).to_numpy(dtype=float),
    }


def run_return_baselines(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in sorted(dataset["horizon"].unique()):
        hframe = dataset[dataset["horizon"].eq(horizon)]
        train = hframe[hframe["split"].eq("train")]
        for split in ["validation", "final"]:
            eval_frame = hframe[hframe["split"].eq(split)].dropna(subset=["forward_simple_return_h"]).copy()
            if eval_frame.empty:
                continue
            predictions = return_baseline_predictions(train, eval_frame)
            for name, pred in predictions.items():
                metrics = regression_metrics(eval_frame["forward_simple_return_h"], pred, eval_frame)
                rows.append(
                    {
                        "task": "return_price",
                        "baseline_id": name,
                        "target_variant": "forward_simple_return_h",
                        "horizon": int(horizon),
                        "split": split,
                        "asset_scope": "stock_index_group",
                        **metrics,
                    }
                )
    return rows


def range_baseline_predictions(train: pd.DataFrame, eval_frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_returns = train["forward_simple_return_h"].dropna().astype(float)
    if len(train_returns):
        q10, q50, q90 = np.nanquantile(train_returns, [0.10, 0.50, 0.90])
    else:
        q10, q50, q90 = -0.01, 0.0, 0.01
    atr = eval_frame["atr_band_return"].fillna(eval_frame["rolling_range_pct_baseline"]).fillna(abs(q90 - q10) / 2.0).to_numpy(dtype=float)
    rolling_range = eval_frame["rolling_range_pct_baseline"].fillna(abs(q90 - q10)).to_numpy(dtype=float)
    prev_range = eval_frame["naive_previous_range_baseline"].fillna(abs(q90 - q10)).to_numpy(dtype=float)
    hist_low = np.full(len(eval_frame), q10, dtype=float)
    hist_mid = np.full(len(eval_frame), q50, dtype=float)
    hist_high = np.full(len(eval_frame), q90, dtype=float)
    zero = np.zeros(len(eval_frame), dtype=float)
    return {
        "atr_rolling_volatility_band": (zero - atr, zero, zero + atr),
        "historical_quantile_return_band": (hist_low, hist_mid, hist_high),
        "rolling_high_low_range_band": (zero - rolling_range / 2.0, zero, zero + rolling_range / 2.0),
        "naive_previous_range": (zero - prev_range / 2.0, zero, zero + prev_range / 2.0),
    }


def run_range_baselines(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in sorted(dataset["horizon"].unique()):
        hframe = dataset[dataset["horizon"].eq(horizon)]
        train = hframe[hframe["split"].eq("train")]
        for split in ["validation", "final"]:
            eval_frame = hframe[hframe["split"].eq(split)].dropna(subset=["forward_simple_return_h"]).copy()
            if eval_frame.empty:
                continue
            predictions = range_baseline_predictions(train, eval_frame)
            for name, (low, mid, high) in predictions.items():
                metrics = interval_metrics(eval_frame, low, mid, high)
                rows.append(
                    {
                        "task": "range_interval",
                        "baseline_id": name,
                        "target_variant": "return_p10_p50_p90",
                        "horizon": int(horizon),
                        "split": split,
                        "asset_scope": "stock_index_group",
                        **metrics,
                    }
                )
    return rows


def run_baselines(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[int, str, str], float]]:
    direction_rows, strongest = run_direction_baselines(dataset)
    rows = direction_rows + run_return_baselines(dataset) + run_range_baselines(dataset)
    baseline_frame = pd.DataFrame(rows)
    write_frame(OUTPUT_DIR / "baseline_results.csv", baseline_frame)
    return baseline_frame, strongest


def train_eval_classifier(
    model: Pipeline,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_train, y_train, x_eval, _y_eval = prepare_xy(train, eval_frame, feature_cols, target, "classification")
    aligned_eval = eval_frame.loc[x_eval.index].copy()
    if len(x_train) < 50 or len(x_eval) == 0 or y_train.nunique() < 2:
        raise ValueError("insufficient rows or class variation")
    model.fit(x_train, y_train.astype(int))
    pred = model.predict(x_eval)
    prob = get_probabilities(model, x_eval, pred)
    return np.asarray(pred, dtype=float), np.asarray(prob, dtype=float), aligned_eval


def run_direction_models(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    strongest_baseline: dict[tuple[int, str, str], float],
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    forecast_candidates: dict[str, Any] = {}
    targets = ["absolute_direction", "market_relative_vn30", "market_relative_vnindex", "top20_forward_return", "top30_forward_return"]
    sampled = sample_for_model(dataset)
    for horizon in sorted(sampled["horizon"].unique()):
        check_timeout(config, "direction_models")
        hframe = sampled[sampled["horizon"].eq(horizon)]
        train = hframe[hframe["split"].eq("train")]
        if train.empty:
            continue
        for target in targets:
            if target not in hframe.columns:
                continue
            for model_name, estimator, scale, _supports_probability in classifier_specs():
                model = make_preprocessor_model(estimator, scale=scale)
                split_predictions: dict[str, dict[str, Any]] = {}
                for split in ["validation", "final"]:
                    eval_frame = hframe[hframe["split"].eq(split)].dropna(subset=[target]).copy()
                    if eval_frame.empty:
                        continue
                    try:
                        pred, prob, aligned_eval = train_eval_classifier(model, train, eval_frame, feature_cols, target)
                        baseline_bal = strongest_baseline.get((int(horizon), target, split), math.nan)
                        metrics = classification_metrics(aligned_eval[target], pred, prob, aligned_eval, baseline_bal)
                        row = {
                            "model_id": model_name,
                            "task": "direction",
                            "target_variant": target,
                            "horizon": int(horizon),
                            "feature_group": "combined_features",
                            "split": split,
                            "asset_scope": "stock_index_group",
                            "stock_rows": int(aligned_eval["asset_type"].eq("stock").sum()),
                            "index_rows": int(aligned_eval["asset_type"].eq("index").sum()),
                            "status": "ok",
                            "skipped_reason": "",
                            "final_scoring_only": split == "final",
                            "claim_label": "not_claimable",
                            **metrics,
                        }
                        rows.append(row)
                        split_predictions[split] = {"frame": aligned_eval, "pred": pred, "prob": prob, "row": row}
                    except Exception as exc:
                        rows.append(
                            {
                                "model_id": model_name,
                                "task": "direction",
                                "target_variant": target,
                                "horizon": int(horizon),
                                "feature_group": "combined_features",
                                "split": split,
                                "asset_scope": "stock_index_group",
                                "status": "skipped",
                                "skipped_reason": str(exc)[:240],
                                "final_scoring_only": split == "final",
                                "claim_label": "not_claimable",
                            }
                        )
                if "validation" in split_predictions:
                    val_row = split_predictions["validation"]["row"]
                    has_stock_and_index = bool(
                        finite_float(val_row.get("stock_rows"), 0) > 0
                        and finite_float(val_row.get("index_rows"), 0) > 0
                    )
                    candidate_key = (
                        1 if has_stock_and_index else 0,
                        finite_float(val_row.get("balanced_accuracy"), -math.inf),
                        finite_float(val_row.get("macro_f1"), -math.inf),
                        finite_float(val_row.get("mcc"), -math.inf),
                    )
                    current = forecast_candidates.get("direction")
                    if current is None or candidate_key > current["key"]:
                        forecast_candidates["direction"] = {
                            "key": candidate_key,
                            "model_id": model_name,
                            "target_variant": target,
                            "horizon": int(horizon),
                            "validation": split_predictions.get("validation"),
                            "final": split_predictions.get("final"),
                        }
    results = pd.DataFrame(rows)
    leaderboard = build_direction_leaderboard(results)
    write_frame(OUTPUT_DIR / "direction_results.csv", results)
    write_frame(OUTPUT_DIR / "direction_leaderboard.csv", leaderboard)
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    return results, leaderboard, best, forecast_candidates.get("direction", {})


def build_direction_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        return ok
    validation = ok[ok["split"].eq("validation")].copy()
    final = ok[ok["split"].eq("final")].copy()
    keys = ["model_id", "target_variant", "horizon", "feature_group", "asset_scope"]
    merged = validation.merge(final, on=keys, how="left", suffixes=("_validation", "_final"))
    merged["has_stock_and_index_validation"] = (
        (pd.to_numeric(merged["stock_rows_validation"], errors="coerce") > 0)
        & (pd.to_numeric(merged["index_rows_validation"], errors="coerce") > 0)
    )
    merged = merged.sort_values(
        ["has_stock_and_index_validation", "balanced_accuracy_validation", "macro_f1_validation", "mcc_validation"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    merged.insert(0, "rank", range(1, len(merged) + 1))
    return merged


def train_eval_regressor(
    model: Pipeline,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    x_train, y_train, x_eval, _y_eval = prepare_xy(train, eval_frame, feature_cols, target, "regression")
    aligned_eval = eval_frame.loc[x_eval.index].copy()
    valid_train = y_train.notna() & np.isfinite(y_train.astype(float))
    x_train = x_train.loc[valid_train]
    y_train = y_train.loc[valid_train].astype(float)
    if len(x_train) < 50 or len(x_eval) == 0:
        raise ValueError("insufficient rows")
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    return np.asarray(pred, dtype=float), aligned_eval


def run_return_price_models(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    forecast_candidate: dict[str, Any] = {}
    sampled = sample_for_model(dataset)
    target = "forward_simple_return_h"
    for horizon in sorted(sampled["horizon"].unique()):
        check_timeout(config, "return_price_models")
        hframe = sampled[sampled["horizon"].eq(horizon)]
        train = hframe[hframe["split"].eq("train")]
        if train.empty:
            continue
        for model_name, estimator, scale in regressor_specs():
            model = make_preprocessor_model(estimator, scale=scale)
            split_predictions: dict[str, dict[str, Any]] = {}
            for split in ["validation", "final"]:
                eval_frame = hframe[hframe["split"].eq(split)].dropna(subset=[target]).copy()
                if eval_frame.empty:
                    continue
                try:
                    pred, aligned_eval = train_eval_regressor(model, train, eval_frame, feature_cols, target)
                    baselines = return_baseline_predictions(train, aligned_eval)
                    metrics = regression_metrics(aligned_eval[target], pred, aligned_eval, baselines)
                    row = {
                        "model_id": model_name,
                        "task": "return_price",
                        "target_variant": target,
                        "horizon": int(horizon),
                        "feature_group": "combined_features",
                        "split": split,
                        "asset_scope": "stock_index_group",
                        "stock_rows": int(aligned_eval["asset_type"].eq("stock").sum()),
                        "index_rows": int(aligned_eval["asset_type"].eq("index").sum()),
                        "status": "ok",
                        "skipped_reason": "",
                        "final_scoring_only": split == "final",
                        "claim_label": "not_claimable",
                        **metrics,
                    }
                    rows.append(row)
                    split_predictions[split] = {"frame": aligned_eval, "pred": pred, "row": row}
                except Exception as exc:
                    rows.append(
                        {
                            "model_id": model_name,
                            "task": "return_price",
                            "target_variant": target,
                            "horizon": int(horizon),
                            "feature_group": "combined_features",
                            "split": split,
                            "asset_scope": "stock_index_group",
                            "status": "skipped",
                            "skipped_reason": str(exc)[:240],
                            "final_scoring_only": split == "final",
                            "claim_label": "not_claimable",
                        }
                    )
            if "validation" in split_predictions:
                val_row = split_predictions["validation"]["row"]
                has_stock_and_index = bool(
                    finite_float(val_row.get("stock_rows"), 0) > 0
                    and finite_float(val_row.get("index_rows"), 0) > 0
                )
                key = (
                    1 if has_stock_and_index else 0,
                    finite_float(val_row.get("improvement_vs_random_walk_close_rmse"), -math.inf),
                    -finite_float(val_row.get("rmse"), math.inf),
                    finite_float(val_row.get("rank_ic"), -math.inf),
                )
                current = forecast_candidate.get("return_price")
                if current is None or key > current["key"]:
                    forecast_candidate["return_price"] = {
                        "key": key,
                        "model_id": model_name,
                        "target_variant": target,
                        "horizon": int(horizon),
                        "validation": split_predictions.get("validation"),
                        "final": split_predictions.get("final"),
                    }

    # GARCH/volatility is kept as an optional baseline row because arch may be unavailable
    # and per-asset GARCH fitting is intentionally bounded for this lab.
    rows.append(
        {
            "model_id": "garch_volatility_baseline_optional",
            "task": "return_price",
            "target_variant": target,
            "horizon": "",
            "feature_group": "range_volatility",
            "split": "validation",
            "asset_scope": "stock_index_group",
            "status": "skipped" if importlib.util.find_spec("arch") is None else "available_not_run_bounded_lab",
            "skipped_reason": "arch package unavailable" if importlib.util.find_spec("arch") is None else "kept as optional volatility baseline; bounded lab uses ATR/rolling volatility bands",
            "final_scoring_only": False,
            "claim_label": "not_claimable",
        }
    )
    results = pd.DataFrame(rows)
    leaderboard = build_return_leaderboard(results)
    write_frame(OUTPUT_DIR / "return_price_results.csv", results)
    write_frame(OUTPUT_DIR / "return_price_leaderboard.csv", leaderboard)
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    return results, leaderboard, best, forecast_candidate.get("return_price", {})


def build_return_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        return ok
    validation = ok[ok["split"].eq("validation")].copy()
    final = ok[ok["split"].eq("final")].copy()
    keys = ["model_id", "target_variant", "horizon", "feature_group", "asset_scope"]
    merged = validation.merge(final, on=keys, how="left", suffixes=("_validation", "_final"))
    merged["has_stock_and_index_validation"] = (
        (pd.to_numeric(merged["stock_rows_validation"], errors="coerce") > 0)
        & (pd.to_numeric(merged["index_rows_validation"], errors="coerce") > 0)
    )
    sort_cols = [
        "has_stock_and_index_validation",
        "improvement_vs_random_walk_close_rmse_validation",
        "improvement_vs_last_close_rmse_validation",
        "rank_ic_validation",
        "rmse_validation",
    ]
    for col in sort_cols:
        if col not in merged.columns:
            merged[col] = math.nan
    merged = merged.sort_values(sort_cols, ascending=[False, False, False, False, True], na_position="last")
    merged.insert(0, "rank", range(1, len(merged) + 1))
    return merged


def fit_quantile_model(
    model_name: str,
    quantile: float,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    if model_name == "gradient_boosting_quantile":
        model = make_preprocessor_model(
            GradientBoostingRegressor(loss="quantile", alpha=quantile, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED),
            scale=False,
        )
    elif model_name == "quantile_regression":
        model = make_preprocessor_model(QuantileRegressor(quantile=quantile, alpha=0.0001, solver="highs"), scale=True)
    elif model_name == "lightgbm_quantile":
        if importlib.util.find_spec("lightgbm") is None:
            raise ValueError("lightgbm unavailable")
        from lightgbm import LGBMRegressor

        model = make_preprocessor_model(
            LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=120,
                max_depth=4,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=25,
                verbosity=-1,
                random_state=RANDOM_SEED,
            ),
            scale=False,
        )
    else:
        raise ValueError(f"Unknown quantile model: {model_name}")
    return train_eval_regressor(model, train, eval_frame, feature_cols, "forward_simple_return_h")


def conformal_interval_predictions(train: pd.DataFrame, eval_frame: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train, x_eval, _ = prepare_xy(train, eval_frame, feature_cols, "forward_simple_return_h", "regression")
    aligned_train = train.loc[x_train.index].copy()
    train_sorted = aligned_train.sort_values("feature_timestamp")
    if len(train_sorted) < 100:
        raise ValueError("insufficient conformal rows")
    cut = max(50, int(len(train_sorted) * 0.75))
    proper = train_sorted.iloc[:cut]
    calibration = train_sorted.iloc[cut:]
    base = make_preprocessor_model(Ridge(alpha=1.0), scale=True)
    base.fit(proper[feature_cols].replace([np.inf, -np.inf], np.nan), proper["forward_simple_return_h"].astype(float))
    cal_pred = base.predict(calibration[feature_cols].replace([np.inf, -np.inf], np.nan))
    residual_q = float(np.nanquantile(np.abs(calibration["forward_simple_return_h"].to_numpy(dtype=float) - cal_pred), 0.80))
    pred = base.predict(x_eval)
    return pred - residual_q, pred, pred + residual_q


def run_range_interval_models(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    forecast_candidate: dict[str, Any] = {}
    sampled = sample_for_model(dataset)
    model_names = ["lightgbm_quantile", "gradient_boosting_quantile", "quantile_regression", "conformal_prediction_wrapper"]
    for horizon in sorted(sampled["horizon"].unique()):
        check_timeout(config, "range_interval_models")
        hframe = sampled[sampled["horizon"].eq(horizon)]
        train = hframe[hframe["split"].eq("train")]
        if train.empty:
            continue
        for model_name in model_names:
            split_predictions: dict[str, dict[str, Any]] = {}
            for split in ["validation", "final"]:
                eval_frame = hframe[hframe["split"].eq(split)].dropna(subset=["forward_simple_return_h"]).copy()
                if eval_frame.empty:
                    continue
                try:
                    if model_name == "conformal_prediction_wrapper":
                        low, mid, high = conformal_interval_predictions(train, eval_frame, feature_cols)
                        aligned_eval = eval_frame.loc[eval_frame.dropna(subset=["forward_simple_return_h"]).index].copy()
                    else:
                        low, aligned_eval = fit_quantile_model(model_name, 0.10, train, eval_frame, feature_cols)
                        mid, _ = fit_quantile_model(model_name, 0.50, train, eval_frame, feature_cols)
                        high, _ = fit_quantile_model(model_name, 0.90, train, eval_frame, feature_cols)
                    lower = np.minimum(low, high)
                    upper = np.maximum(low, high)
                    low, high = lower, upper
                    metrics = interval_metrics(aligned_eval, low, mid, high)
                    row = {
                        "model_id": model_name,
                        "task": "range_interval",
                        "target_variant": "return_p10_p50_p90",
                        "horizon": int(horizon),
                        "feature_group": "combined_features",
                        "split": split,
                        "asset_scope": "stock_index_group",
                        "stock_rows": int(aligned_eval["asset_type"].eq("stock").sum()),
                        "index_rows": int(aligned_eval["asset_type"].eq("index").sum()),
                        "status": "ok",
                        "skipped_reason": "",
                        "final_scoring_only": split == "final",
                        "claim_label": "not_claimable",
                        **metrics,
                    }
                    rows.append(row)
                    split_predictions[split] = {"frame": aligned_eval, "low": low, "mid": mid, "high": high, "row": row}
                except Exception as exc:
                    rows.append(
                        {
                            "model_id": model_name,
                            "task": "range_interval",
                            "target_variant": "return_p10_p50_p90",
                            "horizon": int(horizon),
                            "feature_group": "combined_features",
                            "split": split,
                            "asset_scope": "stock_index_group",
                            "status": "skipped",
                            "skipped_reason": str(exc)[:240],
                            "final_scoring_only": split == "final",
                            "claim_label": "not_claimable",
                        }
                    )
            if "validation" in split_predictions:
                val_row = split_predictions["validation"]["row"]
                coverage = finite_float(val_row.get("interval_coverage"))
                width = finite_float(val_row.get("average_interval_width"), math.inf)
                winkler = finite_float(val_row.get("winkler_score"), math.inf)
                coverage_penalty = abs(coverage - 0.80) if math.isfinite(coverage) else math.inf
                has_stock_and_index = bool(
                    finite_float(val_row.get("stock_rows"), 0) > 0
                    and finite_float(val_row.get("index_rows"), 0) > 0
                )
                key = (1 if has_stock_and_index else 0, -coverage_penalty, -width, -winkler)
                current = forecast_candidate.get("range_interval")
                if current is None or key > current["key"]:
                    forecast_candidate["range_interval"] = {
                        "key": key,
                        "model_id": model_name,
                        "target_variant": "return_p10_p50_p90",
                        "horizon": int(horizon),
                        "validation": split_predictions.get("validation"),
                        "final": split_predictions.get("final"),
                    }

        for baseline_name, predictions in range_baseline_predictions(train, hframe[hframe["split"].eq("validation")].dropna(subset=["forward_simple_return_h"])).items():
            eval_frame = hframe[hframe["split"].eq("validation")].dropna(subset=["forward_simple_return_h"]).copy()
            if eval_frame.empty:
                continue
            low, mid, high = predictions
            metrics = interval_metrics(eval_frame, low, mid, high)
            rows.append(
                {
                    "model_id": baseline_name,
                    "task": "range_interval",
                    "target_variant": "return_p10_p50_p90",
                    "horizon": int(horizon),
                    "feature_group": "baseline_range_features",
                    "split": "validation",
                    "asset_scope": "stock_index_group",
                    "stock_rows": int(eval_frame["asset_type"].eq("stock").sum()),
                    "index_rows": int(eval_frame["asset_type"].eq("index").sum()),
                    "status": "ok",
                    "skipped_reason": "",
                    "final_scoring_only": False,
                    "claim_label": "not_claimable",
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    leaderboard = build_range_leaderboard(results)
    write_frame(OUTPUT_DIR / "range_interval_results.csv", results)
    write_frame(OUTPUT_DIR / "range_interval_leaderboard.csv", leaderboard)
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    return results, leaderboard, best, forecast_candidate.get("range_interval", {})


def build_range_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        return ok
    validation = ok[ok["split"].eq("validation")].copy()
    validation["_coverage_gap"] = (validation["interval_coverage"].astype(float) - 0.80).abs()
    validation["has_stock_and_index_validation"] = (
        (pd.to_numeric(validation["stock_rows"], errors="coerce") > 0)
        & (pd.to_numeric(validation["index_rows"], errors="coerce") > 0)
    )
    validation = validation.sort_values(
        ["has_stock_and_index_validation", "_coverage_gap", "average_interval_width", "winkler_score"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    validation.insert(0, "rank", range(1, len(validation) + 1))
    return validation.drop(columns=["_coverage_gap"])


def run_ranking_models(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    forecast_candidate: dict[str, Any] = {}
    sampled = sample_for_model(dataset)
    model_specs: list[tuple[str, str]] = [
        ("relative_strength_rank_baseline", "baseline"),
        ("pairwise_sklearn_ranker_fallback", "extratrees"),
        ("rank_average_ensemble", "ensemble"),
    ]
    if importlib.util.find_spec("lightgbm") is not None:
        model_specs.append(("lightgbm_ranker", "lightgbm"))
    if importlib.util.find_spec("xgboost") is not None:
        model_specs.append(("xgboost_ranker", "xgboost"))

    for horizon in sorted(sampled["horizon"].unique()):
        check_timeout(config, "ranking_models")
        hframe = sampled[(sampled["horizon"].eq(horizon)) & sampled["asset_type"].eq("stock")].copy()
        train = hframe[hframe["split"].eq("train")].dropna(subset=["forward_simple_return_h"])
        if train.empty:
            continue
        for model_id, kind in model_specs:
            split_predictions: dict[str, dict[str, Any]] = {}
            for split in ["validation", "final"]:
                eval_frame = hframe[hframe["split"].eq(split)].dropna(subset=["forward_simple_return_h"]).copy()
                if eval_frame.empty:
                    continue
                try:
                    if kind == "baseline":
                        score = eval_frame["asset_return_rank_pct"].fillna(eval_frame["asset_momentum_rank_pct"]).fillna(0.5).to_numpy(dtype=float)
                    elif kind == "ensemble":
                        base_score = eval_frame["asset_return_rank_pct"].fillna(0.5).to_numpy(dtype=float)
                        model_score = fit_rank_regressor("extratrees", train, eval_frame, feature_cols)
                        score = 0.5 * pd.Series(base_score).rank(pct=True).to_numpy() + 0.5 * pd.Series(model_score).rank(pct=True).to_numpy()
                    else:
                        score = fit_rank_regressor(kind, train, eval_frame, feature_cols)
                    metrics = ranking_metrics(eval_frame, score)
                    row = {
                        "model_id": model_id,
                        "task": "ranking",
                        "target_variant": "cross_sectional_forward_return_rank",
                        "horizon": int(horizon),
                        "feature_group": "combined_features",
                        "split": split,
                        "asset_scope": "vn30_stocks_only",
                        "status": "ok",
                        "skipped_reason": "",
                        "final_scoring_only": split == "final",
                        "claim_label": "not_claimable",
                        **metrics,
                    }
                    rows.append(row)
                    split_predictions[split] = {"frame": eval_frame, "score": score, "row": row}
                except Exception as exc:
                    rows.append(
                        {
                            "model_id": model_id,
                            "task": "ranking",
                            "target_variant": "cross_sectional_forward_return_rank",
                            "horizon": int(horizon),
                            "feature_group": "combined_features",
                            "split": split,
                            "asset_scope": "vn30_stocks_only",
                            "status": "skipped",
                            "skipped_reason": str(exc)[:240],
                            "final_scoring_only": split == "final",
                            "claim_label": "not_claimable",
                        }
                    )
            if "validation" in split_predictions:
                val_row = split_predictions["validation"]["row"]
                key = (
                    finite_float(val_row.get("spearman_ic"), -math.inf),
                    finite_float(val_row.get("ndcg_at_10"), -math.inf),
                    finite_float(val_row.get("top20_precision"), -math.inf),
                )
                current = forecast_candidate.get("ranking")
                if current is None or key > current["key"]:
                    forecast_candidate["ranking"] = {
                        "key": key,
                        "model_id": model_id,
                        "target_variant": "cross_sectional_forward_return_rank",
                        "horizon": int(horizon),
                        "validation": split_predictions.get("validation"),
                        "final": split_predictions.get("final"),
                    }
    results = pd.DataFrame(rows)
    leaderboard = build_ranking_leaderboard(results)
    write_frame(OUTPUT_DIR / "ranking_results.csv", results)
    write_frame(OUTPUT_DIR / "ranking_leaderboard.csv", leaderboard)
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    return results, leaderboard, best, forecast_candidate.get("ranking", {})


def fit_rank_regressor(kind: str, train: pd.DataFrame, eval_frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if kind == "extratrees":
        estimator = ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_SEED)
        model = make_preprocessor_model(estimator, scale=False)
        target = "cross_sectional_forward_return_rank"
    elif kind == "lightgbm":
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(n_estimators=120, max_depth=4, learning_rate=0.04, num_leaves=15, verbosity=-1, random_state=RANDOM_SEED)
        model = make_preprocessor_model(estimator, scale=False)
        target = "cross_sectional_forward_return_rank"
    elif kind == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=RANDOM_SEED,
        )
        model = make_preprocessor_model(estimator, scale=False)
        target = "cross_sectional_forward_return_rank"
    else:
        raise ValueError(f"Unknown rank regressor kind: {kind}")
    x_train, y_train, x_eval, _ = prepare_xy(train, eval_frame, feature_cols, target, "regression")
    valid = y_train.notna() & np.isfinite(y_train.astype(float))
    if int(valid.sum()) < 50 or len(x_eval) == 0:
        raise ValueError("insufficient ranking rows")
    model.fit(x_train.loc[valid], y_train.loc[valid].astype(float))
    return np.asarray(model.predict(x_eval), dtype=float)


def build_ranking_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        return ok
    validation = ok[ok["split"].eq("validation")].copy()
    final = ok[ok["split"].eq("final")].copy()
    keys = ["model_id", "target_variant", "horizon", "feature_group", "asset_scope"]
    merged = validation.merge(final, on=keys, how="left", suffixes=("_validation", "_final"))
    merged = merged.sort_values(["spearman_ic_validation", "ndcg_at_10_validation", "top20_precision_validation"], ascending=False, na_position="last")
    merged.insert(0, "rank", range(1, len(merged) + 1))
    return merged


def confidence_label_from_direction(prob: Any) -> str:
    value = finite_float(prob)
    if not math.isfinite(value):
        return "unknown"
    distance = abs(value - 0.5)
    if distance >= 0.20:
        return "high"
    if distance >= 0.10:
        return "medium"
    return "low"


def build_forecast_panel(
    split: str,
    direction_candidate: dict[str, Any],
    return_candidate: dict[str, Any],
    range_candidate: dict[str, Any],
    ranking_candidate: dict[str, Any],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    base: pd.DataFrame | None = None
    if direction_candidate and split in direction_candidate:
        data = direction_candidate[split]
        frame = data["frame"].copy().reset_index(drop=True)
        frame["model_id"] = data["model_id"] if "model_id" in data else direction_candidate.get("model_id", "")
        frame["target_variant"] = direction_candidate.get("target_variant", "absolute_direction")
        frame["direction_probability"] = pd.Series(data["prob"]).astype(float)
        frame["predicted_direction"] = pd.Series(data["pred"]).astype(float)
        base = frame
    candidate_sources = [return_candidate, range_candidate, ranking_candidate]
    for candidate in candidate_sources:
        if candidate and split in candidate:
            frame = candidate[split]["frame"].copy().reset_index(drop=True)
            if base is None:
                base = frame
            break
    if base is None:
        return pd.DataFrame(columns=forecast_panel_columns())
    panel = base[
        [
            "asset_code",
            "asset_type",
            "feature_timestamp",
            "target_timestamp",
            "horizon",
            "open",
            "high",
            "low",
            "close",
            "forward_simple_return_h",
            "future_close_h",
            "future_high_h",
            "future_low_h",
            "absolute_direction",
        ]
    ].copy()
    panel["model_id"] = direction_candidate.get("model_id", "") if direction_candidate else ""
    panel["target_variant"] = "group_forecast_panel"
    if direction_candidate and split in direction_candidate:
        panel["direction_probability"] = pd.Series(direction_candidate[split]["prob"]).astype(float)
        panel["predicted_direction"] = pd.Series(direction_candidate[split]["pred"]).astype(float)
    else:
        panel["direction_probability"] = np.nan
        panel["predicted_direction"] = np.nan
    if return_candidate and split in return_candidate:
        ret_frame = return_candidate[split]["frame"].copy().reset_index(drop=True)
        ret = pd.DataFrame(
            {
                "asset_code": ret_frame["asset_code"],
                "feature_timestamp": ret_frame["feature_timestamp"],
                "predicted_return_p50": pd.Series(return_candidate[split]["pred"]).astype(float),
            }
        )
        panel = panel.merge(ret, on=["asset_code", "feature_timestamp"], how="left")
    else:
        panel["predicted_return_p50"] = np.nan
    if range_candidate and split in range_candidate:
        rng_frame = range_candidate[split]["frame"].copy().reset_index(drop=True)
        rng = pd.DataFrame(
            {
                "asset_code": rng_frame["asset_code"],
                "feature_timestamp": rng_frame["feature_timestamp"],
                "predicted_return_p10": pd.Series(range_candidate[split]["low"]).astype(float),
                "predicted_range_mid": pd.Series(range_candidate[split]["mid"]).astype(float),
                "predicted_return_p90": pd.Series(range_candidate[split]["high"]).astype(float),
            }
        )
        panel = panel.merge(rng, on=["asset_code", "feature_timestamp"], how="left")
    else:
        panel["predicted_return_p10"] = np.nan
        panel["predicted_range_mid"] = np.nan
        panel["predicted_return_p90"] = np.nan
    panel["predicted_return_p50"] = panel["predicted_return_p50"].fillna(panel["predicted_range_mid"])
    panel["predicted_close_low"] = panel["close"] * (1.0 + panel["predicted_return_p10"])
    panel["predicted_close_mid"] = panel["close"] * (1.0 + panel["predicted_return_p50"])
    panel["predicted_close_high"] = panel["close"] * (1.0 + panel["predicted_return_p90"])
    panel["predicted_low_price"] = panel["predicted_close_low"]
    panel["predicted_high_price"] = panel["predicted_close_high"]
    panel["predicted_range_pct"] = panel["predicted_return_p90"] - panel["predicted_return_p10"]
    if ranking_candidate and split in ranking_candidate:
        rank_frame = ranking_candidate[split]["frame"].copy().reset_index(drop=True)
        rank = pd.DataFrame(
            {
                "asset_code": rank_frame["asset_code"],
                "feature_timestamp": rank_frame["feature_timestamp"],
                "rank_score": pd.Series(ranking_candidate[split]["score"]).astype(float),
            }
        )
        panel = panel.merge(rank, on=["asset_code", "feature_timestamp"], how="left")
    else:
        panel["rank_score"] = np.nan
    panel["confidence_label"] = panel["direction_probability"].map(confidence_label_from_direction)
    panel["actual_return"] = panel["forward_simple_return_h"]
    panel["actual_close"] = panel["future_close_h"]
    panel["actual_high"] = panel["future_high_h"]
    panel["actual_low"] = panel["future_low_h"]
    panel["correct_direction"] = (panel["predicted_direction"] == panel["absolute_direction"]).astype(float)
    panel.loc[panel["predicted_direction"].isna() | panel["absolute_direction"].isna(), "correct_direction"] = np.nan
    panel["interval_hit"] = (
        (panel["actual_return"] >= panel["predicted_return_p10"]) & (panel["actual_return"] <= panel["predicted_return_p90"])
    ).astype(float)
    panel.loc[panel["predicted_return_p10"].isna() | panel["predicted_return_p90"].isna() | panel["actual_return"].isna(), "interval_hit"] = np.nan
    panel = panel[forecast_panel_columns()].copy()
    if len(panel) > MAX_FORECAST_PANEL_ROWS:
        panel = panel.sort_values(["horizon", "asset_type", "asset_code", "feature_timestamp"]).head(MAX_FORECAST_PANEL_ROWS)
    return panel


def forecast_panel_columns() -> list[str]:
    return [
        "asset_code",
        "asset_type",
        "feature_timestamp",
        "target_timestamp",
        "horizon",
        "model_id",
        "target_variant",
        "direction_probability",
        "predicted_direction",
        "predicted_return_p50",
        "predicted_return_p10",
        "predicted_return_p90",
        "predicted_close_low",
        "predicted_close_mid",
        "predicted_close_high",
        "predicted_low_price",
        "predicted_high_price",
        "predicted_range_pct",
        "rank_score",
        "confidence_label",
        "actual_return",
        "actual_close",
        "actual_high",
        "actual_low",
        "correct_direction",
        "interval_hit",
    ]


def decision_labels(
    best_direction: dict[str, Any],
    best_return: dict[str, Any],
    best_range: dict[str, Any],
    best_ranking: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    if best_direction:
        labels.append("direction_candidate_found")
    if best_return:
        labels.append("return_price_candidate_found")
    if best_range:
        labels.append("range_interval_candidate_found")
    if best_ranking:
        labels.append("ranking_candidate_found")
    if not labels:
        labels.append("no_candidate")
    labels.extend(["future_blind_required", "not_claimable"])
    return labels


def direction_beats_baseline(best_direction: dict[str, Any]) -> bool:
    lift = finite_float(best_direction.get("lift_over_strongest_simple_baseline_validation"))
    bal = finite_float(best_direction.get("balanced_accuracy_validation"))
    target = str(best_direction.get("target_variant", ""))
    if target.startswith("market_relative"):
        return bool(math.isfinite(lift) and lift > 0 and math.isfinite(bal) and bal > 0.5)
    return bool(math.isfinite(lift) and lift > 0)


def return_beats_random_walk(best_return: dict[str, Any]) -> bool:
    return bool(finite_float(best_return.get("improvement_vs_random_walk_close_rmse_validation")) > 0)


def useful_range_coverage(best_range: dict[str, Any]) -> bool:
    coverage = finite_float(best_range.get("interval_coverage"))
    width = finite_float(best_range.get("average_interval_width"))
    return bool(math.isfinite(coverage) and 0.70 <= coverage <= 0.90 and math.isfinite(width) and width <= 0.20)


def stability_summary(direction_results: pd.DataFrame) -> str:
    if direction_results.empty or "asset_level_stability" not in direction_results.columns:
        return "insufficient rows"
    ok = direction_results[(direction_results["status"].eq("ok")) & direction_results["split"].eq("validation")].copy()
    if ok.empty:
        return "insufficient rows"
    asset_min = finite_float(ok["asset_level_stability"].max())
    month_min = finite_float(ok["month_stability"].max())
    quarter_min = finite_float(ok["quarter_stability"].max())
    if asset_min >= 0.45 and month_min >= 0.45 and quarter_min >= 0.45:
        return "mixed-to-stable in validation diagnostics"
    return "not uniformly stable across assets/months/quarters"


def build_reports(
    config: RunConfig,
    coverage_rows: list[dict[str, Any]],
    used_indices: list[str],
    skipped_indices: list[str],
    best_direction: dict[str, Any],
    best_return: dict[str, Any],
    best_range: dict[str, Any],
    best_ranking: dict[str, Any],
    direction_results: pd.DataFrame,
) -> dict[str, Any]:
    skipped_rows = [row for row in coverage_rows if row["asset_type"] == "index" and not row["used"]]
    direction_beats = direction_beats_baseline(best_direction)
    return_beats = return_beats_random_walk(best_return)
    range_useful = useful_range_coverage(best_range)
    stable_text = stability_summary(direction_results)
    claim_boundary = (
        "Offline research-lab diagnostics only for VN30 stocks plus configured Vietnamese index codes. "
        "Selection is validation-governed; final rows are scoring-only and not claimable. "
        "No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, daily T+1 production, "
        "VN100 assumption, main-branch, merge, tag, push --mirror, DOCX, or champion-replacement claim is made. "
        "A future-blind rerun with frozen code/data before the target period is required before any external claim."
    )
    summary = f"""# VN30 + Index Group Price-Range Forecast Result Summary

1. Which index codes were used? `{", ".join(used_indices) if used_indices else "none"}`.
2. Which index codes were skipped and why? {", ".join(f"`{row['asset_code']}`: {row['skipped_reason']}" for row in skipped_rows) if skipped_rows else "None."}
3. Which method best forecasts direction for the VN30 + index group? `{best_direction.get("model_id", "")}` on `{best_direction.get("target_variant", "")}` horizon `{best_direction.get("horizon", "")}`, validation balanced accuracy {pct_text(best_direction.get("balanced_accuracy_validation"))}, macro F1 {pct_text(best_direction.get("macro_f1_validation"))}, MCC {number_text(best_direction.get("mcc_validation"))}.
4. Which method best forecasts return/price? `{best_return.get("model_id", "")}` horizon `{best_return.get("horizon", "")}`, validation RMSE {number_text(best_return.get("rmse_validation"))}, MAE {number_text(best_return.get("mae_validation"))}, rank IC {number_text(best_return.get("rank_ic_validation"))}.
5. Which method best forecasts price range / interval? `{best_range.get("model_id", "")}` horizon `{best_range.get("horizon", "")}`, validation interval coverage {pct_text(best_range.get("interval_coverage"))}, average width {number_text(best_range.get("average_interval_width"))}, Winkler score {number_text(best_range.get("winkler_score"))}.
6. Which method best ranks VN30 assets? `{best_ranking.get("model_id", "")}` horizon `{best_ranking.get("horizon", "")}`, validation Spearman IC {number_text(best_ranking.get("spearman_ic_validation"))}, NDCG@10 {number_text(best_ranking.get("ndcg_at_10_validation"))}, top-20 precision {pct_text(best_ranking.get("top20_precision_validation"))}.
7. Does any direction model beat the strongest baseline under repaired metrics? `{direction_beats}`. Market-relative rows are judged with balanced accuracy/lift, not raw accuracy alone.
8. Does any return/price model beat random walk / last close? `{return_beats}`. Best validation random-walk RMSE improvement: {number_text(best_return.get("improvement_vs_random_walk_close_rmse_validation"))}; last-close RMSE improvement: {number_text(best_return.get("improvement_vs_last_close_rmse_validation"))}.
9. Does any range model achieve useful interval coverage without excessive width? `{range_useful}`. Useful here means validation coverage between 70% and 90% with average return-width <= 20%.
10. Are results stable across stocks and indices? `{stable_text}`.
11. Is any result claimable now? `False`.
12. Exact claim boundary: {claim_boundary}

## Scope Update

- {LOCAL_HISTORICAL_ONLY_SCOPE}
- Missing required VN30 stock cache rows stop the run and report affected files/assets.
- Missing requested index cache rows are skipped and recorded with `skipped_reason`; no live data is fetched to fill gaps.

## Split Discipline

- Train: feature_timestamp <= `2023-12-31 23:59:59` and target_timestamp <= `2023-12-31 23:59:59`.
- Validation: feature_timestamp and target_timestamp inside calendar year `2024`.
- Final: feature_timestamp and target_timestamp >= `2025-01-01`; final rows are scoring-only.

## Output Artifacts

- Generated CSV/JSON artifacts: `{rel(OUTPUT_DIR)}`.
- Configured horizons: `{",".join(str(item) for item in config.horizons)}`.
- Configured indices: `{",".join(config.index_codes)}`.
"""
    RESULT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")

    claim = f"""# VN30 + Index Group Price-Range Forecast Claim Boundary

- This artifact is a research lab for direction, return/price, range/interval, and cross-sectional ranking diagnostics.
- Scope is VN30 stocks plus explicitly configured index codes only: `{", ".join(config.index_codes)}`.
- {LOCAL_HISTORICAL_ONLY_SCOPE}
- Used index codes: `{", ".join(used_indices) if used_indices else "none"}`.
- Skipped index codes are recorded in `reports/generated/vn30_index_group_range_forecast/index_coverage_audit.csv`.
- Feature rows and targets obey explicit feature_timestamp/target_timestamp split guards.
- Direction, return/price, range/interval, and ranking metrics are separate; stock and index row counts are reported separately where applicable.
- Raw accuracy alone is not used to judge market-relative direction targets.
- Final rows remain scoring-only and are not used for claimable selection.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, daily T+1 system, VN100 assumption, DOCX, tag, merge, push --mirror, or main-branch claim is made.
- Exact claim boundary: {claim_boundary}
"""
    CLAIM_BOUNDARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLAIM_BOUNDARY_PATH.write_text(claim + "\n", encoding="utf-8")

    return {
        "used_index_codes": used_indices,
        "skipped_index_codes": {row["asset_code"]: row["skipped_reason"] for row in skipped_rows},
        "best_direction_method": best_direction.get("model_id", ""),
        "best_return_price_method": best_return.get("model_id", ""),
        "best_range_interval_method": best_range.get("model_id", ""),
        "best_ranking_method": best_ranking.get("model_id", ""),
        "direction_beats_baseline": direction_beats,
        "return_price_beats_random_walk": return_beats,
        "range_interval_coverage_useful": range_useful,
        "any_result_claimable": False,
        "claim_boundary": claim_boundary,
    }


def run_lab(config: RunConfig) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, coverage_rows, used_indices, skipped_indices = load_assets(config)
    write_csv(OUTPUT_DIR / "index_coverage_audit.csv", coverage_rows)
    check_timeout(config, "dataset_construction")

    features = add_market_context(add_asset_features(panel), used_indices)
    dataset = build_full_dataset(features, config.horizons)
    if dataset.empty:
        raise RuntimeError("No valid split-guarded horizon rows were produced.")
    write_csv(OUTPUT_DIR / "dataset_audit.csv", build_dataset_audit(panel, dataset))
    write_csv(OUTPUT_DIR / "split_guard_audit.csv", build_split_guard_audit(dataset))
    write_csv(OUTPUT_DIR / "target_audit.csv", build_target_audit(dataset))
    write_csv(OUTPUT_DIR / "feature_audit.csv", build_feature_audit(dataset, detect_qml_artifacts()))
    check_timeout(config, "audits")

    feature_cols = selected_feature_columns(dataset, "combined_features")
    if not feature_cols:
        raise RuntimeError("No usable feature columns were available.")
    baseline_results, strongest_direction_baseline = run_baselines(dataset)
    check_timeout(config, "baselines")

    direction_results, direction_leaderboard, best_direction, direction_forecast = run_direction_models(
        dataset,
        feature_cols,
        strongest_direction_baseline,
        config,
    )
    return_results, return_leaderboard, best_return, return_forecast = run_return_price_models(dataset, feature_cols, config)
    range_results, range_leaderboard, best_range, range_forecast = run_range_interval_models(dataset, feature_cols, config)
    ranking_results, ranking_leaderboard, best_ranking, ranking_forecast = run_ranking_models(dataset, feature_cols, config)
    check_timeout(config, "models")

    validation_panel = build_forecast_panel("validation", direction_forecast, return_forecast, range_forecast, ranking_forecast)
    final_panel = build_forecast_panel("final", direction_forecast, return_forecast, range_forecast, ranking_forecast)
    write_frame(OUTPUT_DIR / "group_forecast_panel_validation.csv", validation_panel)
    write_frame(OUTPUT_DIR / "group_forecast_panel_final.csv", final_panel)

    report_decision = build_reports(
        config,
        coverage_rows,
        used_indices,
        skipped_indices,
        best_direction,
        best_return,
        best_range,
        best_ranking,
        direction_results,
    )
    decision = {
        "run_group_range_lab": True,
        "scope": "VN30 + configured index group historical forecast lab",
        "local_historical_data_only": True,
        "live_data_fetch": False,
        "real_time_forecast": False,
        "daily_t_plus_1_production_system": False,
        "provider_api_calls_for_new_data": False,
        "scheduler_dashboard_api_or_live_prediction_ledger": False,
        "frequency": config.frequency,
        "horizons": config.horizons,
        "configured_index_codes": config.index_codes,
        "used_index_codes": used_indices,
        "skipped_index_codes": report_decision["skipped_index_codes"],
        "decision_labels": decision_labels(best_direction, best_return, best_range, best_ranking),
        "best_direction": best_direction,
        "best_return_price": best_return,
        "best_range_interval": best_range,
        "best_ranking": best_ranking,
        "direction_beats_baseline": report_decision["direction_beats_baseline"],
        "return_price_beats_random_walk": report_decision["return_price_beats_random_walk"],
        "range_interval_coverage_useful": report_decision["range_interval_coverage_useful"],
        "any_result_claimable": False,
        "claim_boundary": report_decision["claim_boundary"],
        "final_rows_scoring_only": True,
        "raw_accuracy_alone_for_market_relative_targets": False,
        "artifacts": {
            "dataset_audit": rel(OUTPUT_DIR / "dataset_audit.csv"),
            "index_coverage_audit": rel(OUTPUT_DIR / "index_coverage_audit.csv"),
            "split_guard_audit": rel(OUTPUT_DIR / "split_guard_audit.csv"),
            "target_audit": rel(OUTPUT_DIR / "target_audit.csv"),
            "feature_audit": rel(OUTPUT_DIR / "feature_audit.csv"),
            "baseline_results": rel(OUTPUT_DIR / "baseline_results.csv"),
            "direction_results": rel(OUTPUT_DIR / "direction_results.csv"),
            "return_price_results": rel(OUTPUT_DIR / "return_price_results.csv"),
            "range_interval_results": rel(OUTPUT_DIR / "range_interval_results.csv"),
            "ranking_results": rel(OUTPUT_DIR / "ranking_results.csv"),
            "validation_forecast_panel": rel(OUTPUT_DIR / "group_forecast_panel_validation.csv"),
            "final_forecast_panel": rel(OUTPUT_DIR / "group_forecast_panel_final.csv"),
            "result_summary": rel(RESULT_SUMMARY_PATH),
            "claim_boundary": rel(CLAIM_BOUNDARY_PATH),
        },
    }
    write_json(OUTPUT_DIR / "group_forecast_decision.json", decision)
    return {
        "decision": decision,
        "coverage_rows": coverage_rows,
        "baseline_rows": int(len(baseline_results)),
        "direction_rows": int(len(direction_results)),
        "return_rows": int(len(return_results)),
        "range_rows": int(len(range_results)),
        "ranking_rows": int(len(ranking_results)),
        "validation_forecast_rows": int(len(validation_panel)),
        "final_forecast_rows": int(len(final_panel)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VN30 + configured-index direction/return/range/ranking research lab.")
    parser.add_argument("--run-group-range-lab", action="store_true", help="Execute the research lab.")
    parser.add_argument("--frequency", default="hourly", choices=["hourly"], help="Data frequency. Only hourly cache mode is supported.")
    parser.add_argument("--horizons", default=",".join(str(item) for item in DEFAULT_HORIZONS), help="Comma-separated horizon steps.")
    parser.add_argument("--index-codes", default=",".join(DEFAULT_INDEX_CODES), help="Comma-separated index codes to request.")
    parser.add_argument("--timeout-seconds", type=int, default=14400, help="Wall-clock timeout guard for the run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_group_range_lab:
        parser.error("--run-group-range-lab is required")
    config = RunConfig(
        frequency=args.frequency,
        horizons=parse_horizons(args.horizons),
        index_codes=parse_csv_list(args.index_codes),
        timeout_seconds=int(args.timeout_seconds),
        started_at=time.time(),
    )
    result = run_lab(config)
    decision = result["decision"]
    print(json.dumps(
        {
            "status": "ok",
            "used_index_codes": decision["used_index_codes"],
            "skipped_index_codes": decision["skipped_index_codes"],
            "best_direction_method": decision["best_direction"].get("model_id", ""),
            "best_return_price_method": decision["best_return_price"].get("model_id", ""),
            "best_range_interval_method": decision["best_range_interval"].get("model_id", ""),
            "best_ranking_method": decision["best_ranking"].get("model_id", ""),
            "any_result_claimable": decision["any_result_claimable"],
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
