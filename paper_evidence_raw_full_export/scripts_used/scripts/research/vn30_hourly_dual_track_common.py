"""Shared helpers for VN30 hourly dual-track model comparison."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_stock_index_joint_panel_features import read_joint_panel_universe  # noqa: E402

SEED = 42
LOCKED_RF_H60 = 0.6031
CURRENT_RF_H60 = 0.5611902280884566
HISTORICAL_FINAL_ROWS = 3474
CURRENT_FINAL_ROWS = 8637
HORIZONS = [40, 60, 80, 100, 120]
BASE_MODEL_NAMES = ["random_forest", "extra_trees", "decision_tree_cart", "xgboost", "lightgbm", "logistic_regression"]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
EXPANDED_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_expanded_model_pool_screening"
STACKING_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_stacking_ensemble_v1"

RESULT_COLUMNS_CANONICAL = [
    "track",
    "model",
    "horizon",
    "feature_set",
    "validation_accuracy",
    "final_accuracy",
    "final_rows",
    "final_coverage",
    "delta_vs_locked_60_31",
    "pass_60_31",
    "pass_65",
    "selected_on_validation",
    "claim_level",
]

RESULT_COLUMNS_CURRENT = [
    "track",
    "model",
    "horizon",
    "feature_set",
    "validation_accuracy",
    "final_accuracy",
    "final_rows",
    "final_coverage",
    "delta_vs_current_rf_h60_56_12",
    "pass_current_baseline",
    "pass_60",
    "pass_65",
    "selected_on_validation",
    "claim_level",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.2f}%"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


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


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    return "\n".join(lines)


def active_stock_tickers() -> list[str]:
    stocks, _indices = read_joint_panel_universe()
    return list(stocks)


def load_stock_data(tickers: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        path = STOCK_CACHE_DIR / f"{ticker}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["ticker"] = ticker
        for col in OHLCV_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["datetime", *OHLCV_COLUMNS])
        frame = frame[(frame["close"] > 0) & (frame["volume"] >= 0)]
        frames.append(frame[["datetime", "ticker", *OHLCV_COLUMNS]])
    if not frames:
        return pd.DataFrame(columns=["datetime", "ticker", *OHLCV_COLUMNS])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"]).reset_index(drop=True)


def load_index_data() -> dict[str, pd.DataFrame]:
    indices: dict[str, pd.DataFrame] = {}
    for code in ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]:
        path = INDEX_CACHE_DIR / f"{code}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["datetime", "close"])
        frame = frame[frame["close"] > 0].sort_values("datetime").reset_index(drop=True)
        indices[code] = frame
    return indices


def build_feature_set_c(stock_df: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    prepared = stock_df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    feat_cols: list[str] = []
    for _ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        prepared.loc[idx, "return_1"] = close.pct_change(periods=1, fill_method=None)
        for lag in (2, 3, 5, 10, 20):
            prepared.loc[idx, f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
        for lag in (1, 2, 3, 5, 10, 20):
            prepared.loc[idx, f"return_1_lag_{lag}"] = prepared.loc[idx, "return_1"].shift(lag)
        for window in (5, 10, 20, 60):
            min_p = max(3, min(window, window // 2))
            prepared.loc[idx, f"rolling_return_mean_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"rolling_return_vol_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).std()
            sma = close.rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"close_sma_ratio_{window}"] = close / sma - 1.0
            prepared.loc[idx, f"momentum_{window}"] = close / close.shift(window) - 1.0
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(14, min_periods=7).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi.loc[(loss == 0.0) & (gain > 0.0)] = 100.0
        rsi.loc[(loss == 0.0) & (gain == 0.0)] = 50.0
        prepared.loc[idx, "rsi_14"] = rsi
        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_12 - ema_26
        prepared.loc[idx, "macd"] = macd
        prepared.loc[idx, "macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        prepared.loc[idx, "macd_hist"] = prepared.loc[idx, "macd"] - prepared.loc[idx, "macd_signal"]
        prepared.loc[idx, "volume_change_1"] = volume.pct_change(periods=1, fill_method=None)
        vol_ma = volume.rolling(20, min_periods=5).mean()
        prepared.loc[idx, "volume_shock_20"] = volume / vol_ma - 1.0
        prepared.loc[idx, "high_low_range"] = (group["high"] - group["low"]) / close
        prepared.loc[idx, "open_close_spread"] = (close - group["open"]) / group["open"].replace(0.0, np.nan)
        prepared.loc[idx, "close_position_in_range"] = (close - group["low"]) / (group["high"] - group["low"]).replace(0.0, np.nan)
        for lag in (1, 2, 4, 8, 20):
            prepared.loc[idx, f"lag_ret_{lag}"] = close.pct_change(periods=lag, fill_method=None).shift(1)
        for window in (4, 8, 20, 40):
            prepared.loc[idx, f"roll_ret_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=max(2, window // 4)).mean()
        for window in (8, 20, 40):
            prepared.loc[idx, f"roll_vol_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=max(3, window // 4)).std()
            prepared.loc[idx, f"roll_vol_change_{window}"] = volume.pct_change(periods=1, fill_method=None).rolling(window, min_periods=max(3, window // 4)).mean()
    feat_cols = [
        "return_1", "return_2", "return_3", "return_5", "return_10", "return_20",
        "return_1_lag_1", "return_1_lag_2", "return_1_lag_3", "return_1_lag_5", "return_1_lag_10", "return_1_lag_20",
        "rolling_return_mean_5", "rolling_return_vol_5", "close_sma_ratio_5", "momentum_5",
        "rolling_return_mean_10", "rolling_return_vol_10", "close_sma_ratio_10", "momentum_10",
        "rolling_return_mean_20", "rolling_return_vol_20", "close_sma_ratio_20", "momentum_20",
        "rolling_return_mean_60", "rolling_return_vol_60", "close_sma_ratio_60", "momentum_60",
        "rsi_14", "macd", "macd_signal", "macd_hist", "volume_change_1", "volume_shock_20",
        "high_low_range", "open_close_spread", "close_position_in_range",
        "lag_ret_1", "lag_ret_2", "lag_ret_4", "lag_ret_8", "lag_ret_20",
        "roll_ret_4", "roll_ret_8", "roll_ret_20", "roll_ret_40",
        "roll_vol_8", "roll_vol_20", "roll_vol_40",
        "roll_vol_change_8", "roll_vol_change_20", "roll_vol_change_40",
    ]
    for code, idx_df in index_data.items():
        idx_df = idx_df.copy()
        idx_df["idx_ret"] = idx_df["close"].pct_change(periods=1, fill_method=None)
        idx_features: list[str] = []
        for lag in (1, 2, 3, 5, 10, 20):
            col = f"{code.lower()}_lag_{lag}"
            idx_df[col] = idx_df["idx_ret"].shift(lag)
            idx_features.append(col)
        for window in (20, 60):
            mean_col = f"{code.lower()}_roll_mean_{window}"
            vol_col = f"{code.lower()}_roll_vol_{window}"
            idx_df[mean_col] = idx_df["idx_ret"].rolling(window, min_periods=max(3, window // 4)).mean()
            idx_df[vol_col] = idx_df["idx_ret"].rolling(window, min_periods=max(3, window // 4)).std()
            idx_features.extend([mean_col, vol_col])
        prepared = prepared.merge(idx_df[["datetime", *idx_features]].drop_duplicates("datetime", keep="last"), on="datetime", how="left")
        feat_cols.extend(idx_features)
        if "vnindex_lag_1" in prepared.columns and "market_minus_stock_ret" not in feat_cols:
            prepared["market_minus_stock_ret"] = prepared["vnindex_lag_1"] - prepared["return_1"]
            feat_cols.append("market_minus_stock_ret")
    for col, values in {
        "day_of_week": prepared["datetime"].dt.dayofweek,
        "day_of_month": prepared["datetime"].dt.day,
        "month": prepared["datetime"].dt.month,
        "quarter": prepared["datetime"].dt.quarter,
        "hour": prepared["datetime"].dt.hour,
        "minute": prepared["datetime"].dt.minute,
    }.items():
        prepared[col] = values.astype(float)
    feat_cols = sorted({*feat_cols, "day_of_week", "day_of_month", "month", "quarter", "hour", "minute"})
    existing = [col for col in feat_cols if col in prepared.columns]
    prepared[existing] = prepared[existing].replace([np.inf, -np.inf], np.nan)
    return prepared, existing


def add_absolute_labels(frame: pd.DataFrame, horizon: int) -> pd.Series:
    labels: list[pd.Series] = []
    for _ticker, group in frame.groupby("ticker", sort=True):
        future_close = group["close"].shift(-horizon)
        direction = (future_close > group["close"]).astype(float)
        direction.loc[future_close.isna()] = np.nan
        labels.append(pd.Series(direction.values, index=group.index))
    return pd.concat(labels) if labels else pd.Series(dtype=float)


def make_model(model_name: str) -> Any | None:
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1)
    if model_name == "decision_tree_cart":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=SEED)
    if model_name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=SEED, eval_metric="logloss", n_jobs=2)
    if model_name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(num_leaves=31, max_depth=-1, learning_rate=0.05, n_estimators=300, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=2)
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=SEED)
    return None


def score_accuracy(y_true: pd.Series, pred: np.ndarray) -> tuple[float, int]:
    valid = y_true.dropna().astype(int)
    if valid.empty:
        return math.nan, 0
    pred_series = pd.Series(pred, index=valid.index).astype(int)
    return float((valid == pred_series).mean()), int(len(valid))


def selected_by_validation(rows: list[dict[str, Any]], accuracy_key: str = "validation_accuracy") -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("model") not in {"validation_weighted_soft_voting", "stacking_logistic_oof"} and math.isfinite(float(row.get(accuracy_key, math.nan)))]
    if not valid:
        return None
    selected = max(valid, key=lambda row: (float(row.get(accuracy_key, -1)), float(row.get("horizon", -1))))
    selected["selected_on_validation"] = True
    return selected

