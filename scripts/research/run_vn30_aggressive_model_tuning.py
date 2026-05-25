"""Aggressive validation-governed VN30 hourly model tuning.

The runner enumerates the requested model grids, fits a deterministic
validation-budgeted subset, locks one candidate using validation evidence only,
and evaluates the final split once for the locked candidate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    FINAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    assert_strict_target_boundaries,
    load_index_data,
    load_stock_data,
    rel,
    strict_target_split_indices,
    target_timestamp_from_labels,
)

REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_aggressive_model_tuning"
PROTOCOL_PATH = REPO_ROOT / "reports" / "protocols" / "VN30_AGGRESSIVE_MODEL_TUNING_PROTOCOL.md"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_AGGRESSIVE_MODEL_TUNING_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_AGGRESSIVE_MODEL_TUNING_CLAIM_BOUNDARY.md"

RANDOM_STATE = 42
HORIZONS = [20, 30, 40, 50, 60]
THRESHOLDS = [round(float(value), 2) for value in np.arange(0.45, 0.6001, 0.01)]
MARKET_CONTEXT_CODES = ["VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30"]
DEFAULT_FIT_BUDGET = {
    "logistic_regression": 30,
    "random_forest": 12,
    "xgboost": 12,
    "lightgbm": 12,
}
MODEL_COMPLEXITY = {
    "logistic_regression": 0,
    "random_forest": 1,
    "xgboost": 2,
    "lightgbm": 2,
    "soft_vote_ensemble": 3,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number * 100.0:.2f}%" if math.isfinite(number) else ""


def pp(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number * 100.0:+.2f} pp" if math.isfinite(number) else ""


def accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(y_pred, dtype=int)).mean())


def majority_value(y_true: pd.Series | np.ndarray) -> int:
    if len(y_true) == 0:
        return 1
    return int(float(np.asarray(y_true, dtype=int).mean()) >= 0.5)


def clean_feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return frame[feature_cols].replace([np.inf, -np.inf], np.nan)


def add_stock_feature_groups(stock_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = stock_df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    groups = {
        "momentum_lag": [],
        "volatility_lag": [],
        "volume_shock": [],
        "compact_stable_features": [],
    }
    for _ticker, group in out.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        open_ = group["open"].astype(float)
        volume = group["volume"].astype(float)
        ret = close.pct_change(fill_method=None)
        prev_close = close.shift(1)
        true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        for lag in (1, 2, 3, 5, 10):
            col = f"ret_lag_{lag}"
            out.loc[idx, col] = ret.shift(lag)
            groups["momentum_lag"].append(col)
        for window in (5, 10, 20):
            roll_ret = f"rolling_return_{window}_lag"
            roll_vol = f"rolling_volatility_{window}_lag"
            vol_z = f"volume_zscore_{window}_lag"
            vol_shock = f"volume_shock_{window}_lag"
            out.loc[idx, roll_ret] = (close / close.shift(window) - 1.0).shift(1)
            out.loc[idx, roll_vol] = ret.rolling(window, min_periods=max(3, window // 2)).std().shift(1)
            vol_mean = volume.rolling(window, min_periods=max(3, window // 2)).mean()
            vol_std = volume.rolling(window, min_periods=max(3, window // 2)).std()
            out.loc[idx, vol_z] = ((volume - vol_mean) / vol_std.replace(0.0, np.nan)).shift(1)
            out.loc[idx, vol_shock] = (volume / vol_mean.replace(0.0, np.nan) - 1.0).shift(1)
            groups["momentum_lag"].append(roll_ret)
            groups["volatility_lag"].append(roll_vol)
            groups["volume_shock"].extend([vol_z, vol_shock])
        out.loc[idx, "ticker_normalized_momentum_20_lag"] = (
            (ret.shift(1) - ret.rolling(20, min_periods=5).mean().shift(1))
            / ret.rolling(20, min_periods=5).std().shift(1).replace(0.0, np.nan)
        )
        out.loc[idx, "high_low_range_lag_1"] = ((high - low) / close.replace(0.0, np.nan)).shift(1)
        out.loc[idx, "open_close_spread_lag_1"] = ((close - open_) / open_.replace(0.0, np.nan)).shift(1)
        for window in (5, 10, 20):
            col = f"atr_proxy_{window}_lag"
            out.loc[idx, col] = (true_range / close.replace(0.0, np.nan)).rolling(window, min_periods=max(3, window // 2)).mean().shift(1)
            groups["volatility_lag"].append(col)
    groups["momentum_lag"].append("ticker_normalized_momentum_20_lag")
    groups["volatility_lag"].extend(["high_low_range_lag_1", "open_close_spread_lag_1"])
    for col, values in {
        "day_of_week": out["datetime"].dt.dayofweek.astype(float),
        "month": out["datetime"].dt.month.astype(float),
        "hour": out["datetime"].dt.hour.astype(float),
        "minute": out["datetime"].dt.minute.astype(float),
    }.items():
        out[col] = values
    groups = {name: sorted({col for col in cols if col in out.columns}) for name, cols in groups.items()}
    return out, groups


def build_market_context(index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    pieces: list[pd.DataFrame] = []
    market_cols: list[str] = []
    loaded_codes: list[str] = []
    for code in MARKET_CONTEXT_CODES:
        if code not in index_data:
            continue
        loaded_codes.append(code)
        frame = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
        close = pd.to_numeric(frame["close"], errors="coerce")
        ret = close.pct_change(fill_method=None)
        prefix = code.lower()
        local_cols: list[str] = []
        for lag in (1, 2, 3, 5, 10):
            col = f"{prefix}_ret_lag_{lag}"
            frame[col] = ret.shift(lag)
            local_cols.append(col)
        for window in (5, 10, 20):
            mean_col = f"{prefix}_rolling_return_{window}_lag"
            vol_col = f"{prefix}_rolling_volatility_{window}_lag"
            frame[mean_col] = (close / close.shift(window) - 1.0).shift(1)
            frame[vol_col] = ret.rolling(window, min_periods=max(3, window // 2)).std().shift(1)
            local_cols.extend([mean_col, vol_col])
        direction_col = f"{prefix}_market_direction_lag_1"
        vol_regime_col = f"{prefix}_market_volatility_regime_lag"
        vol20 = ret.rolling(20, min_periods=5).std().shift(1)
        vol60 = ret.rolling(60, min_periods=15).std().shift(1)
        frame[direction_col] = (ret.shift(1) > 0.0).astype(float)
        frame.loc[ret.shift(1).isna(), direction_col] = np.nan
        frame[vol_regime_col] = vol20 / vol60.replace(0.0, np.nan)
        local_cols.extend([direction_col, vol_regime_col])
        market_cols.extend(local_cols)
        pieces.append(frame[["datetime", *local_cols]])
    if not pieces:
        return pd.DataFrame(columns=["datetime"]), {"market_context": []}, loaded_codes
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="datetime", how="outer")
    return merged.sort_values("datetime").reset_index(drop=True), {"market_context": sorted(set(market_cols))}, loaded_codes


def add_relative_strength(frame: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = frame.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    relative_cols: list[str] = []
    for _ticker, group in out.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        ret = close.pct_change(fill_method=None).shift(1)
        for code in ("vnindex", "vn30"):
            market_col = f"{code}_ret_lag_1"
            if market_col not in out.columns:
                continue
            rel_col = f"relative_strength_vs_{code}_lag_1"
            market = pd.to_numeric(out.loc[idx, market_col], errors="coerce")
            relative = ret.to_numpy(dtype=float) - market.to_numpy(dtype=float)
            out.loc[idx, rel_col] = relative
            relative_cols.append(rel_col)
            relative_series = pd.Series(relative, index=idx)
            for window in (5, 10, 20):
                roll_col = f"relative_strength_vs_{code}_{window}_lag"
                out.loc[idx, roll_col] = relative_series.rolling(window, min_periods=max(3, window // 2)).mean()
                relative_cols.append(roll_col)
    groups["relative_strength"] = sorted({col for col in relative_cols if col in out.columns})
    return out, groups


def build_aggressive_features() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    if len(tickers) != 30:
        raise ValueError(f"expected 30 active VN30 tickers, got {len(tickers)}")
    if stock_df.empty:
        raise ValueError("stock data is empty")
    features, groups = add_stock_feature_groups(stock_df)
    market_context, market_groups, loaded_codes = build_market_context(index_data)
    if not market_context.empty:
        features = features.merge(market_context.drop_duplicates("datetime", keep="last"), on="datetime", how="left")
    groups.update(market_groups)
    features, groups = add_relative_strength(features, groups)
    compact_cols = [
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "rolling_return_5_lag",
        "rolling_return_10_lag",
        "rolling_volatility_10_lag",
        "rolling_volatility_20_lag",
        "volume_zscore_20_lag",
        "volume_shock_20_lag",
        "high_low_range_lag_1",
        "open_close_spread_lag_1",
        "atr_proxy_20_lag",
        "ticker_normalized_momentum_20_lag",
        "relative_strength_vs_vnindex_lag_1",
        "relative_strength_vs_vnindex_20_lag",
        "relative_strength_vs_vn30_lag_1",
        "relative_strength_vs_vn30_20_lag",
        "vnindex_market_direction_lag_1",
        "vnindex_market_volatility_regime_lag",
        "vn30_market_direction_lag_1",
        "vn30_market_volatility_regime_lag",
        "day_of_week",
        "month",
        "hour",
        "minute",
    ]
    groups["compact_stable_features"] = [col for col in compact_cols if col in features.columns]
    groups = {name: sorted({col for col in cols if col in features.columns}) for name, cols in groups.items()}
    all_feature_cols = sorted({col for cols in groups.values() for col in cols})
    features[all_feature_cols] = features[all_feature_cols].replace([np.inf, -np.inf], np.nan)
    manifest = {
        "stock_ticker_count": len(tickers),
        "stock_tickers": tickers,
        "stock_rows": int(len(stock_df)),
        "market_context_codes_requested": MARKET_CONTEXT_CODES,
        "market_context_codes_loaded": loaded_codes,
        "feature_groups": {name: {"feature_count": len(cols), "columns": cols} for name, cols in groups.items()},
        "all_features_are_lagged_or_same_bar_ohlcv_known_at_timestamp": True,
        "target_label_features": False,
        "final_window_feature_selection": False,
    }
    return features, groups, manifest


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    splits = strict_target_split_indices(features, labels, TRAIN_END, VAL_START, VAL_END, FINAL_START)
    assert_strict_target_boundaries(features, labels, splits, TRAIN_END, VAL_START, VAL_END, FINAL_START)
    return splits


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    probability: np.ndarray,
    threshold: float,
    candidate_id: str,
    split: str,
) -> pd.DataFrame:
    out = features.loc[idx, ["datetime", "ticker"]].copy()
    target_timestamp = target_timestamp_from_labels(labels).reindex(idx)
    out["target_timestamp"] = target_timestamp.to_numpy()
    out["y_true"] = labels.reindex(idx).astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(probability, dtype=float)
    out["threshold"] = float(threshold)
    out["y_pred"] = (out["y_score_or_probability"].to_numpy(dtype=float) >= float(threshold)).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    out["candidate_id"] = candidate_id
    out["split"] = split
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def group_accuracy_stats(frame: pd.DataFrame, group_col: str) -> dict[str, float]:
    if frame.empty:
        return {"count": 0.0, "min": math.nan, "mean": math.nan, "std": 0.0, "below_50": 0.0, "below_55": 0.0, "below_60": 0.0}
    grouped = frame.groupby(group_col)["correct"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] > 0]
    if grouped.empty:
        return {"count": 0.0, "min": math.nan, "mean": math.nan, "std": 0.0, "below_50": 0.0, "below_55": 0.0, "below_60": 0.0}
    acc = grouped["mean"]
    return {
        "count": float(len(grouped)),
        "min": float(acc.min()),
        "mean": float(acc.mean()),
        "std": float(acc.std(ddof=0) if len(acc) > 1 else 0.0),
        "below_50": float((acc < 0.50).sum()),
        "below_55": float((acc < 0.55).sum()),
        "below_60": float((acc < 0.60).sum()),
    }


def rolling_stats(frame: pd.DataFrame, window: int = 250) -> dict[str, float]:
    if len(frame) < window:
        return {"count": 0.0, "min": math.nan, "mean": math.nan, "std": 0.0, "below_60": 0.0}
    work = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    rolling = work["correct"].astype(float).rolling(window, min_periods=window).mean().dropna()
    return {
        "count": float(len(rolling)),
        "min": float(rolling.min()),
        "mean": float(rolling.mean()),
        "std": float(rolling.std(ddof=0) if len(rolling) > 1 else 0.0),
        "below_60": float((rolling < 0.60).sum()),
    }


def stability_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    work = frame.copy()
    work["month"] = pd.to_datetime(work["datetime"], errors="coerce").dt.to_period("M").astype(str)
    work["quarter"] = pd.to_datetime(work["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    month = group_accuracy_stats(work, "month")
    quarter = group_accuracy_stats(work, "quarter")
    ticker = group_accuracy_stats(work, "ticker")
    rolling = rolling_stats(work)
    return {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_unique_tickers": int(frame["ticker"].nunique()) if not frame.empty else 0,
        f"{prefix}_accuracy": float(frame["correct"].mean()) if not frame.empty else math.nan,
        f"{prefix}_monthly_min_accuracy": month["min"],
        f"{prefix}_monthly_mean_accuracy": month["mean"],
        f"{prefix}_monthly_std_accuracy": month["std"],
        f"{prefix}_months_below_60": int(month["below_60"]),
        f"{prefix}_quarterly_min_accuracy": quarter["min"],
        f"{prefix}_quarterly_mean_accuracy": quarter["mean"],
        f"{prefix}_quarters_below_60": int(quarter["below_60"]),
        f"{prefix}_ticker_min_accuracy": ticker["min"],
        f"{prefix}_ticker_mean_accuracy": ticker["mean"],
        f"{prefix}_ticker_std_accuracy": ticker["std"],
        f"{prefix}_tickers_below_50": int(ticker["below_50"]),
        f"{prefix}_tickers_below_55": int(ticker["below_55"]),
        f"{prefix}_tickers_below_60": int(ticker["below_60"]),
        f"{prefix}_rolling250_min_accuracy": rolling["min"],
        f"{prefix}_rolling250_mean_accuracy": rolling["mean"],
        f"{prefix}_rolling250_windows_below_60": int(rolling["below_60"]),
    }


def baseline_predictions(features: pd.DataFrame, labels: pd.Series, idx: pd.Index, train_y: pd.Series) -> dict[str, np.ndarray]:
    majority = majority_value(train_y)
    selected = features.loc[idx]
    raw: dict[str, pd.Series | np.ndarray] = {
        "majority_class": np.full(len(idx), majority, dtype=int),
        "always_up": np.ones(len(idx), dtype=int),
    }
    if "ret_lag_1" in selected.columns:
        raw["lag1_direction"] = (pd.to_numeric(selected["ret_lag_1"], errors="coerce") > 0.0).astype(float)
    if "rolling_return_20_lag" in selected.columns:
        raw["rolling_momentum_20"] = (pd.to_numeric(selected["rolling_return_20_lag"], errors="coerce") > 0.0).astype(float)
    if "vnindex_market_direction_lag_1" in selected.columns:
        raw["vnindex_direction_lag1"] = pd.to_numeric(selected["vnindex_market_direction_lag_1"], errors="coerce")
    out: dict[str, np.ndarray] = {}
    for name, pred in raw.items():
        series = pd.Series(pred, index=idx) if not isinstance(pred, pd.Series) else pred.reindex(idx)
        out[name] = pd.to_numeric(series, errors="coerce").fillna(float(majority)).round().clip(0, 1).astype(int).to_numpy()
    return out


def baseline_rows_for_split(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    split: str,
    horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idx = splits[split]
    train_y = labels.reindex(splits["train"]).dropna().astype(int)
    y_true = labels.reindex(idx).dropna().astype(int)
    rows: list[dict[str, Any]] = []
    for baseline_name, pred in baseline_predictions(features, labels, idx, train_y).items():
        acc = accuracy(y_true, pred)
        rows.append(
            {
                "split": split,
                "horizon": horizon,
                "baseline_name": baseline_name,
                "accuracy": acc,
                "rows": int(len(y_true)),
                "selection_role": "validation_baseline" if split == "validation" else "post_lock_final_baseline",
            }
        )
    strongest = max(rows, key=lambda row: float(row["accuracy"])) if rows else {
        "baseline_name": "",
        "accuracy": math.nan,
        "rows": 0,
    }
    return rows, strongest


def logistic_param_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    compatible_specs = [
        ("l2", "liblinear"),
        ("l1", "liblinear"),
        ("l2", "saga"),
        ("l1", "saga"),
        ("elasticnet", "saga"),
    ]
    c_values = [0.03, 0.1, 0.3, 1, 3, 10, 0.01, 0.003]
    for c_value, (penalty, solver), class_weight in itertools.product(c_values, compatible_specs, [None, "balanced"]):
        if penalty == "elasticnet" and solver != "saga":
            continue
        if penalty == "l1" and solver not in {"liblinear", "saga"}:
            continue
        rows.append(
            {
                "model_family": "logistic_regression",
                "C": c_value,
                "penalty": penalty,
                "solver": solver,
                "class_weight": class_weight,
                "l1_ratio": 0.5 if penalty == "elasticnet" else None,
            }
        )
    return rows


def rf_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "random_forest",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
        }
        for n_estimators, max_depth, min_samples_leaf, max_features, class_weight in itertools.product(
            [200, 500, 800],
            [3, 5, 8, None],
            [10, 20, 50, 100],
            ["sqrt", 0.5, "log2"],
            [None, "balanced"],
        )
    ]


def xgb_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "xgboost",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
        }
        for n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_lambda, min_child_weight in itertools.product(
            [100, 200, 400],
            [2, 3, 4],
            [0.01, 0.03, 0.05],
            [0.7, 0.85, 1.0],
            [0.7, 0.85, 1.0],
            [1, 5, 10, 20],
            [5, 10, 20],
        )
    ]


def lgbm_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "lightgbm",
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
        }
        for n_estimators, num_leaves, learning_rate, min_child_samples, subsample, colsample_bytree, reg_lambda in itertools.product(
            [100, 200, 400],
            [7, 15, 31],
            [0.01, 0.03, 0.05],
            [30, 50, 100],
            [0.7, 0.85, 1.0],
            [0.7, 0.85, 1.0],
            [1, 5, 10, 20],
        )
    ]


def enumerate_candidate_grid(feature_groups: dict[str, list[str]]) -> pd.DataFrame:
    model_params = logistic_param_grid() + rf_param_grid() + xgb_param_grid() + lgbm_param_grid()
    rows: list[dict[str, Any]] = []
    seq = 0
    for feature_group in feature_groups:
        for horizon in HORIZONS:
            for params in model_params:
                seq += 1
                rows.append(
                    {
                        "grid_id": f"grid_{seq:06d}",
                        "feature_group": feature_group,
                        "horizon": horizon,
                        "model_family": params["model_family"],
                        "params_json": json.dumps(params, sort_keys=True),
                        "threshold_grid": ",".join(f"{threshold:.2f}" for threshold in THRESHOLDS),
                        "selected_for_fit": False,
                    }
                )
    return pd.DataFrame(rows)


def select_budgeted_grid(candidate_grid: pd.DataFrame, budgets: dict[str, int]) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for model_family, budget in budgets.items():
        family = candidate_grid[candidate_grid["model_family"].eq(model_family)].copy()
        if budget <= 0 or family.empty:
            continue
        family["params_rank"] = family.groupby(["feature_group", "horizon"]).cumcount()
        groups = [group.sort_values("params_rank") for _key, group in family.groupby(["feature_group", "horizon"], sort=True)]
        chosen: list[pd.DataFrame] = []
        cursor = 0
        while sum(len(part) for part in chosen) < budget and groups:
            next_groups = []
            for group in groups:
                if cursor < len(group) and sum(len(part) for part in chosen) < budget:
                    chosen.append(group.iloc[[cursor]])
                if cursor + 1 < len(group):
                    next_groups.append(group)
            groups = next_groups
            cursor += 1
        if chosen:
            selected_parts.append(pd.concat(chosen, ignore_index=True))
    return pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=candidate_grid.columns)


def make_model(model_family: str, params: dict[str, Any]) -> Pipeline | None:
    if model_family == "logistic_regression":
        model_params = {
            "C": params["C"],
            "penalty": params["penalty"],
            "solver": params["solver"],
            "class_weight": params["class_weight"],
            "max_iter": 900 if params["solver"] == "saga" else 1000,
            "tol": 1e-3 if params["solver"] == "saga" else 1e-4,
            "random_state": RANDOM_STATE,
        }
        if params["penalty"] == "elasticnet":
            model_params["l1_ratio"] = params.get("l1_ratio", 0.5)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LogisticRegression(**model_params))])
    if model_family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_family == "xgboost" and XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=float(params["min_child_weight"]),
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=2,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_family == "lightgbm" and LGBMClassifier is not None:
        model = LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            num_leaves=int(params["num_leaves"]),
            learning_rate=float(params["learning_rate"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=2,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    return None


def predict_probability(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    return np.asarray(model.predict(x_data), dtype=float)


def validation_acceptance_label(validation_accuracy: float, lift: float) -> str:
    if not math.isfinite(validation_accuracy):
        return "failed"
    if lift <= 0:
        return "not_claimable"
    if validation_accuracy >= 0.62:
        return "target62_candidate"
    if validation_accuracy >= 0.60:
        return "baseline60_candidate"
    return "diagnostic_only"


def candidate_result_row(
    candidate_id: str,
    base_id: str,
    model_family: str,
    feature_group: str,
    horizon: int,
    threshold: float,
    val_frame: pd.DataFrame,
    train_accuracy: float,
    strongest_baseline: dict[str, Any],
    params_json: str,
    ensemble: bool = False,
    ensemble_weights_json: str = "",
    base_model_ids: str = "",
) -> dict[str, Any]:
    metrics = stability_metrics(val_frame, "validation")
    validation_accuracy = float(metrics["validation_accuracy"])
    baseline_accuracy = float(strongest_baseline.get("accuracy", math.nan))
    lift = validation_accuracy - baseline_accuracy
    train_validation_gap = train_accuracy - validation_accuracy if math.isfinite(train_accuracy) else math.nan
    return {
        "candidate_id": candidate_id,
        "base_id": base_id,
        "model_family": model_family,
        "feature_group": feature_group,
        "horizon": horizon,
        "threshold": threshold,
        "status": "ok",
        "selection_source": "validation_only",
        "final_window_role": "not_scored_until_lock",
        "final_accuracy_used_for_selection": False,
        "params_json": params_json,
        "ensemble": ensemble,
        "ensemble_weights_json": ensemble_weights_json,
        "base_model_ids": base_model_ids,
        "train_accuracy": train_accuracy,
        "train_validation_gap_abs": abs(train_validation_gap) if math.isfinite(train_validation_gap) else math.nan,
        "validation_strongest_baseline_name": strongest_baseline.get("baseline_name", ""),
        "validation_strongest_baseline_accuracy": baseline_accuracy,
        "validation_lift_over_strongest_baseline": lift,
        "acceptance_label": validation_acceptance_label(validation_accuracy, lift),
        "model_complexity_rank": MODEL_COMPLEXITY.get(model_family, 9),
        **metrics,
    }


def fit_validation_candidates(
    features: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    fit_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    baseline_rows: list[dict[str, Any]] = []
    label_cache: dict[int, tuple[pd.Series, dict[str, pd.Index], dict[str, Any]]] = {}
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        splits = split_indices(features, labels)
        validation_baseline_rows, validation_strongest = baseline_rows_for_split(features, labels, splits, "validation", horizon)
        baseline_rows.extend(validation_baseline_rows)
        label_cache[horizon] = (labels, splits, validation_strongest)

    for row in fit_grid.to_dict("records"):
        model_family = str(row["model_family"])
        if model_family == "xgboost" and XGBClassifier is None:
            continue
        if model_family == "lightgbm" and LGBMClassifier is None:
            continue
        feature_group = str(row["feature_group"])
        horizon = int(row["horizon"])
        feature_cols = [col for col in feature_groups[feature_group] if col in features.columns]
        labels, splits, strongest_baseline = label_cache[horizon]
        train_idx = splits["train"]
        validation_idx = splits["validation"]
        train_y = labels.reindex(train_idx).dropna().astype(int)
        validation_y = labels.reindex(validation_idx).dropna().astype(int)
        if len(train_y) < 100 or len(validation_y) < 100 or train_y.nunique() < 2 or not feature_cols:
            continue
        params = json.loads(str(row["params_json"]))
        model = make_model(model_family, params)
        if model is None:
            continue
        x_train = clean_feature_matrix(features.loc[train_idx], feature_cols)
        x_validation = clean_feature_matrix(features.loc[validation_idx], feature_cols)
        base_id = str(row["grid_id"])
        try:
            model.fit(x_train, train_y)
            train_prob = predict_probability(model, x_train)
            validation_prob = predict_probability(model, x_validation)
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": f"{base_id}__failed",
                    "base_id": base_id,
                    "model_family": model_family,
                    "feature_group": feature_group,
                    "horizon": horizon,
                    "threshold": math.nan,
                    "status": "failed",
                    "failure_reason": str(exc)[:300],
                    "selection_source": "validation_only",
                    "final_accuracy_used_for_selection": False,
                    "params_json": row["params_json"],
                }
            )
            continue
        train_accuracy = accuracy(train_y, (train_prob >= 0.50).astype(int))
        payloads[base_id] = {
            "base_id": base_id,
            "model": model,
            "model_family": model_family,
            "feature_group": feature_group,
            "horizon": horizon,
            "feature_cols": feature_cols,
            "labels": labels,
            "splits": splits,
            "validation_prob": validation_prob,
            "train_accuracy": train_accuracy,
            "params_json": str(row["params_json"]),
        }
        for threshold in THRESHOLDS:
            candidate_id = f"{base_id}__t{threshold:.2f}".replace(".", "p")
            val_frame = prediction_frame(features, validation_idx, labels, validation_prob, threshold, candidate_id, "validation")
            rows.append(
                candidate_result_row(
                    candidate_id,
                    base_id,
                    model_family,
                    feature_group,
                    horizon,
                    threshold,
                    val_frame,
                    train_accuracy,
                    strongest_baseline,
                    str(row["params_json"]),
                )
            )
    return pd.DataFrame(rows), payloads, baseline_rows


def top_base_per_family(validation_results: pd.DataFrame) -> pd.DataFrame:
    eligible = validation_results[(validation_results["status"].eq("ok")) & (~validation_results["ensemble"].astype(bool))].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(
        by=[
            "feature_group",
            "horizon",
            "model_family",
            "validation_accuracy",
            "validation_lift_over_strongest_baseline",
            "validation_rolling250_min_accuracy",
            "validation_ticker_min_accuracy",
            "model_complexity_rank",
        ],
        ascending=[True, True, True, False, False, False, False, True],
    )
    return eligible.groupby(["feature_group", "horizon", "model_family"], as_index=False).head(1)


def integer_weight_grid(n_models: int) -> list[np.ndarray]:
    weights: list[np.ndarray] = []
    for parts in itertools.product([1, 2, 3], repeat=n_models):
        arr = np.asarray(parts, dtype=float)
        weights.append(arr / arr.sum())
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for weight in weights:
        key = tuple(round(float(item), 4) for item in weight)
        unique[key] = weight
    return list(unique.values())


def add_ensemble_candidates(
    features: pd.DataFrame,
    validation_results: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    base_best = top_base_per_family(validation_results)
    rows: list[dict[str, Any]] = []
    ensemble_payloads: dict[str, dict[str, Any]] = {}
    combo_specs = [
        ("logistic_xgboost", ["logistic_regression", "xgboost"]),
        ("logistic_lightgbm", ["logistic_regression", "lightgbm"]),
        ("logistic_xgboost_lightgbm", ["logistic_regression", "xgboost", "lightgbm"]),
        ("logistic_xgboost_lightgbm_random_forest", ["logistic_regression", "xgboost", "lightgbm", "random_forest"]),
    ]
    baseline_lookup: dict[int, dict[str, Any]] = {}
    for row in baseline_rows:
        if row["split"] == "validation":
            current = baseline_lookup.get(int(row["horizon"]))
            if current is None or float(row["accuracy"]) > float(current["accuracy"]):
                baseline_lookup[int(row["horizon"])] = row
    for feature_group in sorted(base_best["feature_group"].unique()):
        for horizon in sorted(base_best["horizon"].astype(int).unique()):
            subset = base_best[(base_best["feature_group"].eq(feature_group)) & (base_best["horizon"].astype(int).eq(horizon))]
            by_family = {str(row["model_family"]): row.to_dict() for _, row in subset.iterrows()}
            for combo_name, families in combo_specs:
                if not all(family in by_family for family in families):
                    continue
                base_rows = [by_family[family] for family in families]
                base_payloads = [payloads[str(row["base_id"])] for row in base_rows]
                first = base_payloads[0]
                if not all(payload["splits"]["validation"].equals(first["splits"]["validation"]) for payload in base_payloads):
                    continue
                validation_idx = first["splits"]["validation"]
                labels = first["labels"]
                strongest_baseline = baseline_lookup[horizon]
                for weight_idx, weights in enumerate(integer_weight_grid(len(base_payloads))):
                    validation_prob = np.zeros(len(first["validation_prob"]), dtype=float)
                    train_accuracy = 0.0
                    for weight, payload in zip(weights, base_payloads):
                        validation_prob += float(weight) * np.asarray(payload["validation_prob"], dtype=float)
                        train_accuracy += float(weight) * float(payload["train_accuracy"])
                    base_ids = [str(row["base_id"]) for row in base_rows]
                    base_id = f"ensemble__{combo_name}__{feature_group}__h{horizon}__w{weight_idx:03d}"
                    weights_json = json.dumps({family: float(weight) for family, weight in zip(families, weights)}, sort_keys=True)
                    ensemble_payloads[base_id] = {
                        "base_id": base_id,
                        "model_family": "soft_vote_ensemble",
                        "feature_group": feature_group,
                        "horizon": horizon,
                        "labels": labels,
                        "splits": first["splits"],
                        "base_model_ids": base_ids,
                        "weights": weights,
                        "weights_json": weights_json,
                        "train_accuracy": train_accuracy,
                    }
                    for threshold in THRESHOLDS:
                        candidate_id = f"{base_id}__t{threshold:.2f}".replace(".", "p")
                        val_frame = prediction_frame(features, validation_idx, labels, validation_prob, threshold, candidate_id, "validation")
                        rows.append(
                            candidate_result_row(
                                candidate_id,
                                base_id,
                                "soft_vote_ensemble",
                                feature_group,
                                horizon,
                                threshold,
                                val_frame,
                                train_accuracy,
                                strongest_baseline,
                                "{}",
                                ensemble=True,
                                ensemble_weights_json=weights_json,
                                base_model_ids=";".join(base_ids),
                            )
                        )
    return pd.DataFrame(rows), ensemble_payloads


def select_locked_candidate(validation_results: pd.DataFrame) -> dict[str, Any]:
    valid = validation_results[validation_results["status"].eq("ok")].copy()
    if valid.empty:
        raise ValueError("no valid validation candidates")
    valid["positive_lift"] = valid["validation_lift_over_strongest_baseline"].astype(float) > 0.0
    eligible = valid[valid["positive_lift"]].copy()
    if eligible.empty:
        eligible = valid.copy()
    ranked = eligible.sort_values(
        by=[
            "validation_accuracy",
            "validation_lift_over_strongest_baseline",
            "validation_rows",
            "validation_rolling250_min_accuracy",
            "validation_ticker_min_accuracy",
            "train_validation_gap_abs",
            "model_complexity_rank",
        ],
        ascending=[False, False, False, False, False, True, True],
    )
    selected = ranked.iloc[0].to_dict()
    selected["locked_at_utc"] = now_utc()
    selected["selection_rule"] = (
        "validation accuracy, positive lift over strongest validation baseline, row count, rolling stability, "
        "ticker stability, lower train-validation gap as validation-final mismatch proxy, simpler model"
    )
    selected["final_accuracy_used_for_selection"] = False
    return selected


def final_acceptance_label(final_accuracy: float, final_lift: float) -> str:
    if not math.isfinite(final_accuracy):
        return "failed"
    if final_lift <= 0:
        return "not_claimable"
    if final_accuracy >= 0.62:
        return "target62_candidate"
    if final_accuracy >= 0.60:
        return "baseline60_candidate"
    return "diagnostic_only"


def evaluate_locked_final(
    features: pd.DataFrame,
    locked: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_id = str(locked["base_id"])
    payload = payloads[base_id]
    labels = payload["labels"]
    splits = payload["splits"]
    final_idx = splits["final"]
    threshold = float(locked["threshold"])
    if payload["model_family"] == "soft_vote_ensemble":
        final_prob = np.zeros(len(final_idx), dtype=float)
        for weight, base_model_id in zip(payload["weights"], payload["base_model_ids"]):
            base_payload = payloads[base_model_id]
            x_final = clean_feature_matrix(features.loc[final_idx], base_payload["feature_cols"])
            final_prob += float(weight) * predict_probability(base_payload["model"], x_final)
    else:
        x_final = clean_feature_matrix(features.loc[final_idx], payload["feature_cols"])
        final_prob = predict_probability(payload["model"], x_final)
    final_frame = prediction_frame(features, final_idx, labels, final_prob, threshold, str(locked["candidate_id"]), "final")
    final_baseline_rows, final_strongest = baseline_rows_for_split(features, labels, splits, "final", int(locked["horizon"]))
    validation_baseline_rows, _validation_strongest = baseline_rows_for_split(features, labels, splits, "validation", int(locked["horizon"]))
    final_metrics = stability_metrics(final_frame, "final")
    final_accuracy = float(final_metrics["final_accuracy"])
    final_baseline_accuracy = float(final_strongest["accuracy"])
    final_lift = final_accuracy - final_baseline_accuracy
    result = {
        "candidate_id": locked["candidate_id"],
        "base_id": locked["base_id"],
        "model_family": locked["model_family"],
        "feature_group": locked["feature_group"],
        "horizon": int(locked["horizon"]),
        "threshold": threshold,
        "validation_accuracy": float(locked["validation_accuracy"]),
        "validation_strongest_baseline_name": locked["validation_strongest_baseline_name"],
        "validation_strongest_baseline_accuracy": float(locked["validation_strongest_baseline_accuracy"]),
        "validation_lift_over_strongest_baseline": float(locked["validation_lift_over_strongest_baseline"]),
        "final_accuracy": final_accuracy,
        "final_strongest_baseline_name": final_strongest["baseline_name"],
        "final_strongest_baseline_accuracy": final_baseline_accuracy,
        "final_lift_over_strongest_baseline": final_lift,
        "validation_final_gap": final_accuracy - float(locked["validation_accuracy"]),
        "acceptance_label": final_acceptance_label(final_accuracy, final_lift),
        "baseline60_defensible": bool(final_accuracy >= 0.60 and final_lift > 0.0),
        "target62_defensible": bool(final_accuracy >= 0.62 and final_lift > 0.0),
        "final65_defensible": False,
        "final_evaluation_role": "locked_candidate_scoring_once",
        **final_metrics,
    }
    baseline_comparison = pd.DataFrame(validation_baseline_rows + final_baseline_rows)
    ticker_stability = (
        final_frame.groupby(["split", "ticker"], sort=True)
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"), target_positive_rate=("y_true", "mean"), prediction_positive_rate=("y_pred", "mean"))
        .reset_index()
    )
    rolling = final_frame.sort_values(["datetime", "ticker"]).reset_index(drop=True).copy()
    rolling["row_number"] = np.arange(1, len(rolling) + 1)
    rolling["rolling250_accuracy"] = rolling["correct"].rolling(250, min_periods=250).mean()
    rolling_stability = rolling.dropna(subset=["rolling250_accuracy"])[["split", "candidate_id", "row_number", "datetime", "ticker", "rolling250_accuracy"]]
    return pd.DataFrame([result]), baseline_comparison, rolling_stability, ticker_stability


def build_feature_group_summary(feature_manifest: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_group": name,
                "feature_count": payload["feature_count"],
                "columns": ";".join(payload["columns"]),
                "leakage_guard": "lagged_or_known_at_feature_timestamp",
            }
            for name, payload in feature_manifest["feature_groups"].items()
        ]
    )


def write_reports(
    run_config: dict[str, Any],
    locked: dict[str, Any],
    final_result: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    rolling_stability: pd.DataFrame,
    ticker_stability: pd.DataFrame,
) -> None:
    final_row = final_result.iloc[0].to_dict()
    strongest_final = final_row["final_strongest_baseline_name"]
    protocol = f"""# VN30 Aggressive Model Tuning Protocol

## Scope

- Scope: VN30 stock hourly selected-candidate diagnostic benchmark using local stock hourly rows.
- Market-context scope: {", ".join(MARKET_CONTEXT_CODES)} only where local rows exist.
- Out of scope: VN100, trading, profitability, BUY/SELL, live deployment, DOCX/paper artifacts, git tags.

## Split Discipline

- Train rows require feature timestamp <= `{TRAIN_END}` and target_timestamp <= `{TRAIN_END}`.
- Validation rows require feature timestamp and target_timestamp between `{VAL_START}` and `{VAL_END}`.
- Final rows require feature timestamp and target_timestamp >= `{FINAL_START}`.
- Model, hyperparameter, threshold, and ensemble selection use validation only.
- Final is evaluated once after writing `locked_candidate.json`.

## Tuning

- Horizons: {HORIZONS}.
- Thresholds: {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f} step 0.01.
- Requested model grids are enumerated in `candidate_grid.csv`.
- Fit budget per model family for this run: {run_config["fit_budget_per_model_family"]}.
- Baseline comparison uses the strongest simple baseline for the same split and horizon.

## Selection Rule

Validation-only rank order:

1. validation accuracy
2. positive lift over strongest validation baseline
3. row count
4. rolling stability
5. ticker stability
6. lower train-validation gap as a validation-final mismatch proxy
7. simpler model if tied
"""
    write_markdown(PROTOCOL_PATH, protocol)

    rolling_min = final_row.get("final_rolling250_min_accuracy", math.nan)
    ticker_min = final_row.get("final_ticker_min_accuracy", math.nan)
    summary = f"""# VN30 Aggressive Model Tuning Result Summary

## Best Validation Candidate

- Candidate: `{locked["candidate_id"]}`.
- Model family: {locked["model_family"]}.
- Feature group: {locked["feature_group"]}.
- Horizon: {int(locked["horizon"])}.
- Threshold: {float(locked["threshold"]):.2f}.
- Validation accuracy: {pct(locked["validation_accuracy"])}.
- Strongest validation baseline: {locked["validation_strongest_baseline_name"]} at {pct(locked["validation_strongest_baseline_accuracy"])}.
- Validation lift over strongest baseline: {pp(locked["validation_lift_over_strongest_baseline"])}.

## Locked Final Result

- Final accuracy: {pct(final_row["final_accuracy"])}.
- Strongest final baseline: {strongest_final} at {pct(final_row["final_strongest_baseline_accuracy"])}.
- Lift over strongest final baseline: {pp(final_row["final_lift_over_strongest_baseline"])}.
- Final rows: {int(final_row["final_rows"])}.
- Validation-final gap: {pp(final_row["validation_final_gap"])}.
- Rolling stability, min rolling250 accuracy: {pct(rolling_min)}.
- Ticker stability, min ticker accuracy: {pct(ticker_min)}.

## Claim Boundary

- Baseline60 defensible: {str(bool(final_row["baseline60_defensible"])).lower()}.
- Target62 defensible: {str(bool(final_row["target62_defensible"])).lower()}.
- Final65 defensible: false.
- Acceptance label: {final_row["acceptance_label"]}.

Paper-safe wording:

> In a validation-locked VN30 stock hourly diagnostic benchmark, the selected candidate reached {pct(final_row["final_accuracy"])} final pooled directional accuracy over {int(final_row["final_rows"])} rows, {pp(final_row["final_lift_over_strongest_baseline"])} versus the strongest same-horizon simple baseline. This supports only the stated diagnostic benchmark scope and does not support trading, profitability, live-deployment, VN100, top-k-as-overall-accuracy, or final65 claims.

Artifacts:

- `reports/generated/vn30_aggressive_model_tuning/locked_candidate.json`
- `reports/generated/vn30_aggressive_model_tuning/final_once_result.csv`
- `reports/generated/vn30_aggressive_model_tuning/baseline_comparison.csv`
- `reports/generated/vn30_aggressive_model_tuning/rolling_stability.csv`
- `reports/generated/vn30_aggressive_model_tuning/ticker_stability.csv`
"""
    write_markdown(RESULT_PATH, summary)

    claim = f"""# VN30 Aggressive Model Tuning Claim Boundary

- Claimable scope: VN30 stock hourly selected-candidate diagnostic benchmark only.
- Selection source: validation only.
- Final role: one-time locked-candidate scoring.
- Baseline requirement: report strongest same-horizon baseline and lift.
- Baseline60 defensible: {str(bool(final_row["baseline60_defensible"])).lower()}.
- Target62 defensible: {str(bool(final_row["target62_defensible"])).lower()}.
- Final65 remains not defensible.
- Index context is used only as lagged market-context features and not as stock-result evidence.
- No trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, or paper artifact claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 aggressive validation-governed model tuning.")
    parser.add_argument("--logistic-budget", type=int, default=DEFAULT_FIT_BUDGET["logistic_regression"])
    parser.add_argument("--rf-budget", type=int, default=DEFAULT_FIT_BUDGET["random_forest"])
    parser.add_argument("--xgb-budget", type=int, default=DEFAULT_FIT_BUDGET["xgboost"])
    parser.add_argument("--lgbm-budget", type=int, default=DEFAULT_FIT_BUDGET["lightgbm"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    budgets = {
        "logistic_regression": int(args.logistic_budget),
        "random_forest": int(args.rf_budget),
        "xgboost": int(args.xgb_budget),
        "lightgbm": int(args.lgbm_budget),
    }
    features, feature_groups, feature_manifest = build_aggressive_features()
    candidate_grid = enumerate_candidate_grid(feature_groups)
    fit_grid = select_budgeted_grid(candidate_grid, budgets)
    selected_ids = set(fit_grid["grid_id"].astype(str))
    candidate_grid.loc[candidate_grid["grid_id"].astype(str).isin(selected_ids), "selected_for_fit"] = True
    validation_results, payloads, baseline_rows = fit_validation_candidates(features, feature_groups, fit_grid)
    ensemble_results, ensemble_payloads = add_ensemble_candidates(features, validation_results, payloads, baseline_rows)
    if not ensemble_results.empty:
        validation_results = pd.concat([validation_results, ensemble_results], ignore_index=True)
        payloads.update(ensemble_payloads)

    locked = select_locked_candidate(validation_results)
    run_config = {
        "created_at_utc": now_utc(),
        "scope": "VN30 stock hourly selected-candidate diagnostic",
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
        "horizons": HORIZONS,
        "thresholds": THRESHOLDS,
        "fit_budget_per_model_family": budgets,
        "full_candidate_grid_rows": int(len(candidate_grid)),
        "fit_grid_rows": int(len(fit_grid)),
        "validation_result_rows": int(len(validation_results)),
        "selection_uses_final": False,
        "final_once_after_lock": True,
        "paper_docx_generated": False,
        "git_tags_created": False,
        "market_context_codes": feature_manifest["market_context_codes_loaded"],
    }
    tuning_manifest = {
        **run_config,
        "feature_manifest": feature_manifest,
        "optional_dependencies": {
            "xgboost_available": XGBClassifier is not None,
            "lightgbm_available": LGBMClassifier is not None,
        },
        "validation_only_selection_rule": locked["selection_rule"],
        "locked_candidate_id": locked["candidate_id"],
    }

    write_json(REPORT_DIR / "run_config.json", run_config)
    write_frame(REPORT_DIR / "candidate_grid.csv", candidate_grid)
    write_frame(REPORT_DIR / "validation_results.csv", validation_results)
    write_json(REPORT_DIR / "locked_candidate.json", locked)

    final_result, baseline_comparison, rolling_stability, ticker_stability = evaluate_locked_final(features, locked, payloads)
    write_frame(REPORT_DIR / "final_once_result.csv", final_result)
    write_frame(REPORT_DIR / "baseline_comparison.csv", baseline_comparison)
    write_frame(REPORT_DIR / "rolling_stability.csv", rolling_stability)
    write_frame(REPORT_DIR / "ticker_stability.csv", ticker_stability)
    write_frame(REPORT_DIR / "feature_group_summary.csv", build_feature_group_summary(feature_manifest))
    tuning_manifest["final_once_result"] = final_result.iloc[0].to_dict()
    write_json(REPORT_DIR / "tuning_manifest.json", tuning_manifest)
    write_reports(run_config, locked, final_result, baseline_comparison, rolling_stability, ticker_stability)
    print(f"VN30 aggressive model tuning complete: {rel(REPORT_DIR)}")
    print(f"Locked candidate: {locked['candidate_id']}")
    print(f"Final accuracy: {pct(final_result.iloc[0]['final_accuracy'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
