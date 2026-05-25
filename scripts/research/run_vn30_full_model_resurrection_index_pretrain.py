"""Run VN30 full model resurrection with lagged index-context features.

This runner is intentionally validation-governed: model, feature, threshold,
and lock selection are made from the validation split. Final scoring is written
after the lock. A separate final-ranked leaderboard is produced only as
exploratory_not_claimable evidence.
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
from sklearn.base import clone
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
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    rel,
    strict_target_split_indices,
    target_timestamp_from_labels,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_full_model_resurrection"
PROTOCOL_PATH = REPO_ROOT / "reports" / "protocols" / "VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_PROTOCOL.md"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_CLAIM_BOUNDARY.md"

SEED = 42
MAIN_INDEX_CODES = ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX", "HNX30"]
LEGACY_INDEX_CODES = ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]
HORIZONS = [20, 30, 35, 40, 45, 50, 60]
THRESHOLDS = [round(float(value), 3) for value in np.arange(0.45, 0.6001, 0.005)]
CURRENT_CHAMPION = {
    "candidate_id": "current_champion_strict_replay_l2_feature_set_C_closest_h40_t0p50",
    "model": "l2_logistic",
    "feature_group": "feature_set_C_closest",
    "horizon": 40,
    "threshold": 0.50,
    "final_accuracy": 0.6161021109474718,
    "final_lift": 0.10898379970544925,
    "final_rows": 4074,
    "strongest_baseline": "vnindex_direction_lag1",
    "quarter_min_accuracy": 0.47540983606557374,
    "ticker_median_accuracy": 0.6397058823529411,
}

FEATURE_GROUP_ORDER = [
    "old_baseline_C_closest",
    "old_regime_feature_v2",
    "feature_set_C_closest",
    "feature_set_C_closest_plus_index_context",
    "feature_set_C_closest_plus_relative_strength",
    "feature_set_C_closest_plus_volatility_regime",
    "feature_set_C_closest_plus_volume_shock",
    "compact_stable_features",
    "index_context_only_plus_stock_lags",
    "full_market_context",
]

MODEL_COMPLEXITY = {
    "logistic_regression": 1,
    "elasticnet_logistic": 1,
    "random_forest": 3,
    "xgboost": 4,
    "lightgbm": 4,
    "soft_vote_ensemble": 5,
    "regime_gated_ensemble": 5,
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


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def clean_feature_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols].replace([np.inf, -np.inf], np.nan)


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    splits = strict_target_split_indices(features, labels, TRAIN_END, VAL_START, VAL_END, FINAL_START)
    assert_strict_target_boundaries(features, labels, splits, TRAIN_END, VAL_START, VAL_END, FINAL_START)
    return splits


def build_index_context_layer(index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict[str, Any]]:
    pieces: list[pd.DataFrame] = []
    per_code_cols: list[str] = []
    loaded_codes: list[str] = []
    for code in MAIN_INDEX_CODES:
        if code not in index_data:
            continue
        loaded_codes.append(code)
        frame = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
        close = pd.to_numeric(frame["close"], errors="coerce")
        ret = close.pct_change(fill_method=None)
        prefix = code.lower()
        local_cols: list[str] = []
        for lag in (1, 5):
            col = f"{prefix}_ctx_return_lag{lag}"
            frame[col] = ret.shift(lag)
            local_cols.append(col)
            dir_col = f"{prefix}_ctx_direction_lag{lag}"
            lagged_ret = ret.shift(lag)
            frame[dir_col] = (lagged_ret > 0.0).astype(float)
            frame.loc[lagged_ret.isna(), dir_col] = np.nan
            local_cols.append(dir_col)
        for window in (5, 20, 60):
            mom_col = f"{prefix}_ctx_momentum_{window}"
            vol_col = f"{prefix}_ctx_volatility_{window}"
            frame[mom_col] = (close / close.shift(window) - 1.0).shift(1)
            frame[vol_col] = ret.rolling(window, min_periods=max(3, window // 2)).std().shift(1)
            local_cols.extend([mom_col, vol_col])
        pieces.append(frame[["datetime", *local_cols]])
        per_code_cols.extend(local_cols)

    if not pieces:
        empty = pd.DataFrame(columns=["datetime"])
        return empty, [], pd.DataFrame(), {"loaded_index_codes": []}

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="datetime", how="outer")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    ret_lag1_cols = [f"{code.lower()}_ctx_return_lag1" for code in loaded_codes if f"{code.lower()}_ctx_return_lag1" in merged.columns]
    ret_lag5_cols = [f"{code.lower()}_ctx_return_lag5" for code in loaded_codes if f"{code.lower()}_ctx_return_lag5" in merged.columns]
    dir_lag1_cols = [f"{code.lower()}_ctx_direction_lag1" for code in loaded_codes if f"{code.lower()}_ctx_direction_lag1" in merged.columns]
    vol5_cols = [f"{code.lower()}_ctx_volatility_5" for code in loaded_codes if f"{code.lower()}_ctx_volatility_5" in merged.columns]
    vol20_cols = [f"{code.lower()}_ctx_volatility_20" for code in loaded_codes if f"{code.lower()}_ctx_volatility_20" in merged.columns]
    mom5_cols = [f"{code.lower()}_ctx_momentum_5" for code in loaded_codes if f"{code.lower()}_ctx_momentum_5" in merged.columns]
    mom20_cols = [f"{code.lower()}_ctx_momentum_20" for code in loaded_codes if f"{code.lower()}_ctx_momentum_20" in merged.columns]
    mom60_cols = [f"{code.lower()}_ctx_momentum_60" for code in loaded_codes if f"{code.lower()}_ctx_momentum_60" in merged.columns]

    merged["market_return_lag1"] = merged[ret_lag1_cols].mean(axis=1, skipna=True) if ret_lag1_cols else np.nan
    merged["market_return_lag5"] = merged[ret_lag5_cols].mean(axis=1, skipna=True) if ret_lag5_cols else np.nan
    merged["market_direction_lag1"] = (merged["market_return_lag1"] > 0.0).astype(float)
    merged.loc[merged["market_return_lag1"].isna(), "market_direction_lag1"] = np.nan
    merged["market_direction_lag5"] = (merged["market_return_lag5"] > 0.0).astype(float)
    merged.loc[merged["market_return_lag5"].isna(), "market_direction_lag5"] = np.nan
    merged["market_volatility_5"] = merged[vol5_cols].mean(axis=1, skipna=True) if vol5_cols else np.nan
    merged["market_volatility_20"] = merged[vol20_cols].mean(axis=1, skipna=True) if vol20_cols else np.nan
    merged["index_agreement_score"] = merged[dir_lag1_cols].mean(axis=1, skipna=True) if dir_lag1_cols else np.nan
    merged["market_momentum_5"] = merged[mom5_cols].mean(axis=1, skipna=True) if mom5_cols else np.nan
    merged["market_momentum_20"] = merged[mom20_cols].mean(axis=1, skipna=True) if mom20_cols else np.nan
    merged["market_momentum_60"] = merged[mom60_cols].mean(axis=1, skipna=True) if mom60_cols else np.nan
    merged["cross_index_breadth_proxy"] = merged[dir_lag1_cols].mean(axis=1, skipna=True) if dir_lag1_cols else np.nan
    risk_state = np.select(
        [
            (merged["index_agreement_score"] >= 0.60) & (merged["market_momentum_5"] > 0.0),
            (merged["index_agreement_score"] <= 0.40) & (merged["market_momentum_5"] < 0.0),
        ],
        [1.0, -1.0],
        default=0.0,
    )
    merged["risk_on_risk_off_state"] = risk_state
    merged.loc[merged["index_agreement_score"].isna(), "risk_on_risk_off_state"] = np.nan
    merged["market_direction_regime_code"] = np.select(
        [merged["market_momentum_60"] > 0.02, merged["market_momentum_60"] < -0.02],
        [1.0, -1.0],
        default=0.0,
    )
    merged.loc[merged["market_momentum_60"].isna(), "market_direction_regime_code"] = np.nan
    vol_ratio = merged["market_volatility_5"] / merged["market_volatility_20"].replace(0.0, np.nan)
    merged["market_volatility_regime_code"] = np.select([vol_ratio > 1.10], [1.0], default=0.0)
    merged.loc[vol_ratio.isna(), "market_volatility_regime_code"] = np.nan

    required_cols = [
        "market_direction_lag1",
        "market_direction_lag5",
        "market_return_lag1",
        "market_return_lag5",
        "market_volatility_5",
        "market_volatility_20",
        "index_agreement_score",
        "risk_on_risk_off_state",
        "market_momentum_5",
        "market_momentum_20",
        "cross_index_breadth_proxy",
    ]
    context_cols = required_cols + ["market_momentum_60", "market_direction_regime_code", "market_volatility_regime_code", *per_code_cols]
    context_cols = [col for col in context_cols if col in merged.columns]
    merged[context_cols] = merged[context_cols].replace([np.inf, -np.inf], np.nan)

    audit_rows: list[dict[str, Any]] = []
    for col in context_cols:
        audit_rows.append(
            {
                "feature_name": col,
                "source_indices": ",".join(loaded_codes),
                "lag_rule": "all source returns, volatility, momentum, agreement, and states are shifted at least one source bar",
                "feature_timestamp_safe": True,
                "uses_future_index_label": False,
                "uses_stock_final_label": False,
                "non_null_rows": int(merged[col].notna().sum()),
                "nan_rate": float(merged[col].isna().mean()),
                "first_timestamp": str(merged["datetime"].min()),
                "last_timestamp": str(merged["datetime"].max()),
            }
        )
    manifest = {
        "created_at_utc": now_utc(),
        "loaded_index_codes": loaded_codes,
        "required_context_features": required_cols,
        "context_feature_count": len(context_cols),
        "point_in_time_rule": "index features are computed from lagged returns or rolling windows shifted one bar before merge to stock rows",
        "future_index_labels_used": False,
        "stock_final_labels_used": False,
    }
    return merged[["datetime", *context_cols]], context_cols, pd.DataFrame(audit_rows), manifest


def add_relative_strength_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = features.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    cols: list[str] = []
    for _ticker, group in out.groupby("ticker", sort=True):
        idx = group.index
        stock_ret = pd.to_numeric(group.get("return_1_lag_1", pd.Series(np.nan, index=idx)), errors="coerce")
        for market_col, suffix in [("market_return_lag1", "market"), ("vnindex_lag_1", "vnindex"), ("vn30_lag_1", "vn30")]:
            if market_col not in out.columns:
                continue
            market_ret = pd.to_numeric(out.loc[idx, market_col], errors="coerce")
            rel = pd.Series(stock_ret.to_numpy(dtype=float) - market_ret.to_numpy(dtype=float), index=idx)
            col = f"relative_strength_vs_{suffix}_lag1"
            out.loc[idx, col] = rel
            cols.append(col)
            for window in (5, 20):
                roll_col = f"relative_strength_vs_{suffix}_{window}"
                out.loc[idx, roll_col] = rel.rolling(window, min_periods=max(3, window // 2)).mean()
                cols.append(roll_col)
    return out, sorted({col for col in cols if col in out.columns})


def add_stock_lag_regime_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = features.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    groups = {"stock_lags": [], "stock_volatility_regime": [], "stock_volume_shock": []}
    for _ticker, group in out.groupby("ticker", sort=True):
        idx = group.index
        close = pd.to_numeric(group["close"], errors="coerce")
        high = pd.to_numeric(group["high"], errors="coerce")
        low = pd.to_numeric(group["low"], errors="coerce")
        open_ = pd.to_numeric(group["open"], errors="coerce")
        volume = pd.to_numeric(group["volume"], errors="coerce")
        ret = close.pct_change(fill_method=None)
        prev_close = close.shift(1)
        true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        for lag in (1, 2, 3, 5, 10):
            col = f"stock_return_lag{lag}"
            out.loc[idx, col] = ret.shift(lag)
            groups["stock_lags"].append(col)
        for window in (5, 10, 20):
            vol_col = f"stock_volatility_{window}_lag"
            atr_col = f"stock_atr_proxy_{window}_lag"
            vol_z = f"stock_volume_zscore_{window}_lag"
            shock = f"stock_volume_shock_{window}_lag"
            out.loc[idx, vol_col] = ret.rolling(window, min_periods=max(3, window // 2)).std().shift(1)
            out.loc[idx, atr_col] = (true_range / close.replace(0.0, np.nan)).rolling(window, min_periods=max(3, window // 2)).mean().shift(1)
            vol_mean = volume.rolling(window, min_periods=max(3, window // 2)).mean()
            vol_std = volume.rolling(window, min_periods=max(3, window // 2)).std()
            out.loc[idx, vol_z] = ((volume - vol_mean) / vol_std.replace(0.0, np.nan)).shift(1)
            out.loc[idx, shock] = (volume / vol_mean.replace(0.0, np.nan) - 1.0).shift(1)
            groups["stock_volatility_regime"].extend([vol_col, atr_col])
            groups["stock_volume_shock"].extend([vol_z, shock])
        out.loc[idx, "stock_high_low_range_lag1"] = ((high - low) / close.replace(0.0, np.nan)).shift(1)
        out.loc[idx, "stock_open_close_spread_lag1"] = ((close - open_) / open_.replace(0.0, np.nan)).shift(1)
    groups["stock_volatility_regime"].extend(["stock_high_low_range_lag1", "stock_open_close_spread_lag1"])
    return out, {name: sorted({col for col in cols if col in out.columns}) for name, cols in groups.items()}


def add_regime_labels(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    trend = pd.to_numeric(out.get("market_momentum_60", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["market_direction_regime"] = "sideway"
    out.loc[trend > 0.02, "market_direction_regime"] = "bull"
    out.loc[trend < -0.02, "market_direction_regime"] = "bear"
    out.loc[trend.isna(), "market_direction_regime"] = "unknown_direction"
    ratio = pd.to_numeric(out.get("market_volatility_5", pd.Series(np.nan, index=out.index)), errors="coerce") / pd.to_numeric(
        out.get("market_volatility_20", pd.Series(np.nan, index=out.index)), errors="coerce"
    ).replace(0.0, np.nan)
    out["volatility_regime"] = "low_volatility"
    out.loc[ratio > 1.10, "volatility_regime"] = "high_volatility"
    out.loc[ratio.isna(), "volatility_regime"] = "unknown_volatility"
    out["regime_router_key"] = out["market_direction_regime"].astype(str) + "_" + out["volatility_regime"].astype(str)
    return out


def build_feature_frame() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    if len(tickers) != 30:
        raise ValueError(f"expected 30 active VN30 tickers, got {len(tickers)}")
    if stock_df.empty:
        raise ValueError("stock data is empty")
    legacy_index_data = {code: index_data[code] for code in LEGACY_INDEX_CODES if code in index_data}
    base, base_cols = build_feature_set_c(stock_df, legacy_index_data)
    index_context, index_context_cols, index_audit, index_manifest = build_index_context_layer(index_data)
    features = base.merge(index_context.drop_duplicates("datetime", keep="last"), on="datetime", how="left")
    features, stock_groups = add_stock_lag_regime_features(features)
    features, relative_cols = add_relative_strength_features(features)
    features = add_regime_labels(features)
    for col, values in {
        "day_of_week": features["datetime"].dt.dayofweek.astype(float),
        "month": features["datetime"].dt.month.astype(float),
        "hour": features["datetime"].dt.hour.astype(float),
        "minute": features["datetime"].dt.minute.astype(float),
    }.items():
        features[col] = values

    volatility_cols = [
        *stock_groups["stock_volatility_regime"],
        "market_volatility_5",
        "market_volatility_20",
        "market_volatility_regime_code",
        "risk_on_risk_off_state",
        "market_direction_regime_code",
    ]
    volume_cols = stock_groups["stock_volume_shock"]
    stock_lag_cols = stock_groups["stock_lags"]
    compact_cols = [
        "return_1_lag_1",
        "return_1_lag_2",
        "return_1_lag_3",
        "return_1_lag_5",
        "rolling_return_mean_5",
        "rolling_return_vol_5",
        "rolling_return_mean_20",
        "rolling_return_vol_20",
        "momentum_5",
        "momentum_20",
        "volume_shock_20",
        "market_return_lag1",
        "market_return_lag5",
        "market_volatility_20",
        "index_agreement_score",
        "risk_on_risk_off_state",
        "relative_strength_vs_market_lag1",
        "day_of_week",
        "month",
        "hour",
    ]
    old_regime_cols = [
        "market_direction_lag1",
        "market_direction_lag5",
        "market_direction_regime_code",
        "market_volatility_regime_code",
        "risk_on_risk_off_state",
        "index_agreement_score",
        "market_momentum_20",
    ]
    feature_groups = {
        "old_baseline_C_closest": base_cols,
        "old_regime_feature_v2": [*base_cols, *old_regime_cols],
        "feature_set_C_closest": base_cols,
        "feature_set_C_closest_plus_index_context": [*base_cols, *index_context_cols],
        "feature_set_C_closest_plus_relative_strength": [*base_cols, *relative_cols],
        "feature_set_C_closest_plus_volatility_regime": [*base_cols, *volatility_cols],
        "feature_set_C_closest_plus_volume_shock": [*base_cols, *volume_cols],
        "compact_stable_features": compact_cols,
        "index_context_only_plus_stock_lags": [*index_context_cols, *stock_lag_cols],
        "full_market_context": [*base_cols, *index_context_cols, *relative_cols, *volatility_cols, *volume_cols, *stock_lag_cols],
    }
    feature_groups = {
        name: sorted({col for col in cols if col in features.columns and pd.api.types.is_numeric_dtype(features[col])})
        for name, cols in feature_groups.items()
    }
    all_cols = sorted({col for cols in feature_groups.values() for col in cols})
    features[all_cols] = features[all_cols].replace([np.inf, -np.inf], np.nan)
    features["feature_timestamp"] = features["datetime"]
    manifest = {
        "stock_ticker_count": len(tickers),
        "stock_tickers": tickers,
        "stock_rows": int(len(stock_df)),
        "index_context_manifest": index_manifest,
        "feature_groups": {name: {"feature_count": len(cols), "columns": cols} for name, cols in feature_groups.items()},
        "feature_timestamp_column": "feature_timestamp",
        "target_timestamp_source": "add_absolute_labels(...).attrs['target_timestamp']",
        "features_are_point_in_time_or_lagged": True,
        "stock_final_labels_used_in_index_context": False,
        "final_performance_used_for_selection": False,
    }
    return features, feature_groups, manifest, index_audit, index_manifest


def baseline_predictions(features: pd.DataFrame, labels: pd.Series, idx: pd.Index, train_y: pd.Series) -> dict[str, np.ndarray]:
    majority = majority_value(train_y)
    selected = features.loc[idx]
    raw: dict[str, pd.Series | np.ndarray] = {
        "majority_class": np.full(len(idx), majority, dtype=int),
        "always_up": np.ones(len(idx), dtype=int),
        "always_down": np.zeros(len(idx), dtype=int),
    }
    if "return_1_lag_1" in selected.columns:
        raw["lag1_direction"] = (pd.to_numeric(selected["return_1_lag_1"], errors="coerce") > 0.0).astype(float)
    if "market_direction_lag1" in selected.columns:
        raw["market_direction_lag1"] = pd.to_numeric(selected["market_direction_lag1"], errors="coerce")
    if "vnindex_lag_1" in selected.columns:
        raw["vnindex_direction_lag1"] = (pd.to_numeric(selected["vnindex_lag_1"], errors="coerce") > 0.0).astype(float)
    out: dict[str, np.ndarray] = {}
    for name, pred in raw.items():
        series = pd.Series(pred, index=idx) if not isinstance(pred, pd.Series) else pred.reindex(idx)
        out[name] = pd.to_numeric(series, errors="coerce").fillna(float(majority)).round().clip(0, 1).astype(int).to_numpy()
    return out


def baseline_frames_for_split(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    split: str,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    idx = splits[split]
    train_y = labels.reindex(splits["train"]).dropna().astype(int)
    y_true = labels.reindex(idx).dropna().astype(int)
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    base_cols = features.loc[idx, ["datetime", "ticker"]].copy()
    for baseline_name, pred in baseline_predictions(features, labels, idx, train_y).items():
        frame = base_cols.copy()
        frame["split"] = split
        frame["horizon"] = horizon
        frame["baseline_name"] = baseline_name
        frame["y_true"] = y_true.to_numpy(dtype=int)
        frame["y_pred"] = pred
        frame["correct"] = (frame["y_true"].to_numpy(dtype=int) == frame["y_pred"].to_numpy(dtype=int)).astype(int)
        frames[baseline_name] = frame
        rows.append(
            {
                "split": split,
                "horizon": horizon,
                "baseline_name": baseline_name,
                "accuracy": float(frame["correct"].mean()) if len(frame) else math.nan,
                "rows": int(len(frame)),
                "selection_role": "validation_baseline" if split == "validation" else "post_lock_or_exploratory_final_baseline",
            }
        )
    strongest = max(rows, key=lambda row: float(row["accuracy"])) if rows else {"baseline_name": "", "accuracy": math.nan}
    return pd.DataFrame(rows), strongest, frames


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    probability: np.ndarray,
    threshold: float,
    candidate_id: str,
    split: str,
) -> pd.DataFrame:
    out = features.loc[idx, ["feature_timestamp", "datetime", "ticker"]].copy()
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


def rolling_min(frame: pd.DataFrame, window: int = 250) -> float:
    if len(frame) < window:
        return math.nan
    work = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    rolling = work["correct"].astype(float).rolling(window, min_periods=window).mean().dropna()
    return float(rolling.min()) if len(rolling) else math.nan


def detail_rows(frame: pd.DataFrame, baseline_frame: pd.DataFrame, split: str, candidate_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    base = baseline_frame.copy()
    work["quarter"] = pd.to_datetime(work["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    base["quarter"] = pd.to_datetime(base["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    q = (
        work.groupby("quarter", sort=True)
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"), prediction_up_ratio=("y_pred", "mean"))
        .reset_index()
    )
    qb = base.groupby("quarter", sort=True).agg(baseline_accuracy=("correct", "mean")).reset_index()
    q = q.merge(qb, on="quarter", how="left")
    q["lift"] = q["accuracy"] - q["baseline_accuracy"]
    q.insert(0, "candidate_id", candidate_id)
    q.insert(1, "split", split)
    t = (
        work.groupby("ticker", sort=True)
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"), prediction_up_ratio=("y_pred", "mean"))
        .reset_index()
    )
    tb = base.groupby("ticker", sort=True).agg(baseline_accuracy=("correct", "mean")).reset_index()
    t = t.merge(tb, on="ticker", how="left")
    t["lift"] = t["accuracy"] - t["baseline_accuracy"]
    t.insert(0, "candidate_id", candidate_id)
    t.insert(1, "split", split)
    return q, t


def candidate_metrics(
    frame: pd.DataFrame,
    strongest_baseline: dict[str, Any],
    strongest_baseline_frame: pd.DataFrame,
    candidate_id: str,
    split: str,
    simplicity_score: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    q, t = detail_rows(frame, strongest_baseline_frame, split, candidate_id)
    acc = float(frame["correct"].mean()) if len(frame) else math.nan
    baseline_acc = float(strongest_baseline.get("accuracy", math.nan))
    lift = acc - baseline_acc
    quarter_min = float(q["accuracy"].min()) if len(q) else math.nan
    positive_quarters = int((q["lift"] > 0.0).sum()) if len(q) else 0
    ticker_median = float(t["accuracy"].median()) if len(t) else math.nan
    baseline_ticker_median = float(t["baseline_accuracy"].median()) if len(t) else math.nan
    pred_up = float(frame["y_pred"].astype(int).mean()) if len(frame) else math.nan
    pred_balance = max(0.0, 1.0 - abs(pred_up - 0.525) / 0.525) if math.isfinite(pred_up) else 0.0
    quarter_score = max(0.0, min(1.0, quarter_min)) if math.isfinite(quarter_min) else 0.0
    ticker_score = max(0.0, min(1.0, ticker_median)) if math.isfinite(ticker_median) else 0.0
    composite = (
        0.35 * lift
        + 0.25 * acc
        + 0.15 * quarter_score
        + 0.10 * ticker_score
        + 0.10 * pred_balance
        + 0.05 * simplicity_score
    )
    shortlist_pass = bool(
        lift > 0.0
        and len(frame) >= 3500
        and math.isfinite(pred_up)
        and 0.35 <= pred_up <= 0.70
        and math.isfinite(quarter_min)
        and quarter_min >= 0.45
        and positive_quarters >= 2
        and math.isfinite(ticker_median)
        and math.isfinite(baseline_ticker_median)
        and ticker_median >= baseline_ticker_median
    )
    metrics = {
        f"{split}_accuracy": acc,
        f"{split}_lift": lift,
        f"{split}_rows": int(len(frame)),
        "strongest_baseline": strongest_baseline.get("baseline_name", ""),
        f"{split}_strongest_baseline_accuracy": baseline_acc,
        "ticker_median_accuracy": ticker_median,
        "baseline_ticker_median_accuracy": baseline_ticker_median,
        "ticker_median_lift": ticker_median - baseline_ticker_median,
        "quarter_min_accuracy": quarter_min,
        "quarters_positive_lift": positive_quarters,
        "rolling250_min": rolling_min(frame),
        "prediction_up_ratio": pred_up,
        "quarterly_stability_score": quarter_score,
        "ticker_stability_score": ticker_score,
        "prediction_balance_score": pred_balance,
        "simplicity_score": simplicity_score,
        "validation_composite_score": composite if split == "validation" else math.nan,
        "shortlist_pass": shortlist_pass if split == "validation" else False,
    }
    balance = {
        "candidate_id": candidate_id,
        "split": split,
        "rows": int(len(frame)),
        "prediction_up_ratio": pred_up,
        "prediction_down_ratio": 1.0 - pred_up if math.isfinite(pred_up) else math.nan,
        "passes_35_70_band": bool(math.isfinite(pred_up) and 0.35 <= pred_up <= 0.70),
    }
    return metrics, q, t, balance


def logistic_param_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for penalty, c_value, class_weight in itertools.product(
        ["l1", "l2", "elasticnet"],
        [0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10],
        [None, "balanced"],
    ):
        rows.append(
            {
                "model_family": "elasticnet_logistic" if penalty == "elasticnet" else "logistic_regression",
                "penalty": penalty,
                "solver": "saga" if penalty == "elasticnet" else "liblinear",
                "C": float(c_value),
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
            ["sqrt", "log2", 0.5],
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
    params = logistic_param_grid() + rf_param_grid() + xgb_param_grid() + lgbm_param_grid()
    rows: list[dict[str, Any]] = []
    seq = 0
    for feature_group in FEATURE_GROUP_ORDER:
        if feature_group not in feature_groups:
            continue
        for horizon in HORIZONS:
            for param in params:
                seq += 1
                rows.append(
                    {
                        "grid_id": f"grid_{seq:06d}",
                        "source": "historical_replay" if feature_group.startswith("old_") else ("index_context" if "index_context" in feature_group or "market_context" in feature_group else "new_tuning"),
                        "model": param["model_family"],
                        "feature_group": feature_group,
                        "horizon": horizon,
                        "params_json": json.dumps(param, sort_keys=True),
                        "threshold_grid_start": THRESHOLDS[0],
                        "threshold_grid_end": THRESHOLDS[-1],
                        "threshold_grid_step": 0.005,
                        "selected_for_fit": False,
                        "stage": "full_grid_enumerated",
                    }
                )
    forced = [
        ("forced_old_selected_l2_h40", "historical_replay", "logistic_regression", "feature_set_C_closest", 40, {"model_family": "logistic_regression", "penalty": "l2", "solver": "liblinear", "C": 0.3, "class_weight": "balanced", "l1_ratio": None}),
        ("forced_old_65_l2_regime_t045", "historical_replay", "logistic_regression", "old_regime_feature_v2", 40, {"model_family": "logistic_regression", "penalty": "l2", "solver": "liblinear", "C": 0.3, "class_weight": "balanced", "l1_ratio": None}),
    ]
    for grid_id, source, model, feature_group, horizon, param in forced:
        rows.append(
            {
                "grid_id": grid_id,
                "source": source,
                "model": model,
                "feature_group": feature_group,
                "horizon": horizon,
                "params_json": json.dumps(param, sort_keys=True),
                "threshold_grid_start": THRESHOLDS[0],
                "threshold_grid_end": THRESHOLDS[-1],
                "threshold_grid_step": 0.005,
                "selected_for_fit": True,
                "stage": "forced_historical_replay",
            }
        )
    return pd.DataFrame(rows)


def select_budgeted_grid(candidate_grid: pd.DataFrame, budgets: dict[str, int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    forced = candidate_grid[candidate_grid["selected_for_fit"].astype(bool)].copy()
    if not forced.empty:
        parts.append(forced)
    for model_family, budget in budgets.items():
        family = candidate_grid[(candidate_grid["model"].eq(model_family)) & (~candidate_grid["selected_for_fit"].astype(bool))].copy()
        if family.empty or budget <= 0:
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
            parts.append(pd.concat(chosen, ignore_index=True))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=candidate_grid.columns)
    return out.drop_duplicates("grid_id").reset_index(drop=True)


def make_model(model_family: str, params: dict[str, Any]) -> Pipeline | None:
    if model_family in {"logistic_regression", "elasticnet_logistic"}:
        model_params = {
            "C": float(params["C"]),
            "penalty": params["penalty"],
            "solver": params["solver"],
            "class_weight": params["class_weight"],
            "max_iter": 900 if params["solver"] == "saga" else 1000,
            "tol": 1e-3 if params["solver"] == "saga" else 1e-4,
            "random_state": SEED,
        }
        if params["penalty"] == "elasticnet":
            model_params["l1_ratio"] = float(params.get("l1_ratio", 0.5))
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LogisticRegression(**model_params))])
    if model_family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=SEED,
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
            random_state=SEED,
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
            random_state=SEED,
            verbose=-1,
            n_jobs=2,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    return None


def predict_probability(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_data)[:, 1], dtype=float)
    return np.asarray(model.predict(x_data), dtype=float)


def fit_regime_router(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    feature_cols: list[str],
    group_col: str,
) -> dict[str, Any]:
    train_idx = splits["train"]
    train_y = labels.reindex(train_idx).astype(int)
    global_model = make_model(
        "logistic_regression",
        {"model_family": "logistic_regression", "penalty": "l2", "solver": "liblinear", "C": 0.3, "class_weight": "balanced", "l1_ratio": None},
    )
    if global_model is None:
        raise RuntimeError("regime router global logistic unavailable")
    global_model.fit(clean_feature_matrix(features.loc[train_idx], feature_cols), train_y)
    group_models: dict[str, Any] = {}
    for group_name, group in features.loc[train_idx].groupby(group_col, sort=True):
        group_idx = group.index
        group_y = labels.reindex(group_idx).astype(int)
        if len(group_y) < 100 or group_y.nunique() < 2:
            continue
        local_model = clone(global_model)
        local_model.fit(clean_feature_matrix(features.loc[group_idx], feature_cols), group_y)
        group_models[str(group_name)] = local_model
    return {"global_model": global_model, "group_models": group_models, "group_col": group_col, "feature_cols": feature_cols}


def predict_regime_router(payload: dict[str, Any], features: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    feature_cols = payload["feature_cols"]
    out = predict_probability(payload["global_model"], clean_feature_matrix(features.loc[idx], feature_cols))
    groups = features.loc[idx, payload["group_col"]].astype(str)
    for group_name, model in payload["group_models"].items():
        mask = groups.eq(str(group_name)).to_numpy()
        if mask.any():
            local_idx = idx[mask]
            out[mask] = predict_probability(model, clean_feature_matrix(features.loc[local_idx], feature_cols))
    return np.clip(out, 0.0, 1.0)


def validation_result_row(
    candidate_id: str,
    base_id: str,
    source: str,
    model: str,
    feature_group: str,
    horizon: int,
    threshold: float,
    params_json: str,
    frame: pd.DataFrame,
    strongest_baseline: dict[str, Any],
    strongest_baseline_frame: pd.DataFrame,
    simplicity_score: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics, q, t, balance = candidate_metrics(frame, strongest_baseline, strongest_baseline_frame, candidate_id, "validation", simplicity_score)
    row = {
        "candidate_id": candidate_id,
        "base_id": base_id,
        "source": source,
        "model": model,
        "feature_group": feature_group,
        "horizon": int(horizon),
        "threshold": float(threshold),
        "params_json": params_json,
        "status": "ok",
        "selection_source": "validation_only",
        "final_accuracy_used_for_selection": False,
        "claim_label": "diagnostic_only" if metrics["validation_lift"] > 0.0 else "not_claimable",
        **metrics,
    }
    return row, q, t, balance


def fit_validation_candidates(
    features: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    fit_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, tuple[pd.Series, dict[str, pd.Index]]]]:
    rows: list[dict[str, Any]] = []
    quarter_rows: list[pd.DataFrame] = []
    ticker_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    baseline_rows: list[pd.DataFrame] = []
    label_cache: dict[int, tuple[pd.Series, dict[str, pd.Index]]] = {}
    baseline_cache: dict[tuple[int, str], tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]] = {}
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        splits = split_indices(features, labels)
        label_cache[horizon] = (labels, splits)
        for split in ("validation", "final"):
            baseline_df, strongest, frames = baseline_frames_for_split(features, labels, splits, split, horizon)
            baseline_rows.append(baseline_df)
            baseline_cache[(horizon, split)] = (baseline_df, strongest, frames)

    for grid_row in fit_grid.to_dict("records"):
        model = str(grid_row["model"])
        if model == "xgboost" and XGBClassifier is None:
            continue
        if model == "lightgbm" and LGBMClassifier is None:
            continue
        feature_group = str(grid_row["feature_group"])
        horizon = int(grid_row["horizon"])
        feature_cols = feature_groups.get(feature_group, [])
        labels, splits = label_cache[horizon]
        train_idx = splits["train"]
        val_idx = splits["validation"]
        train_y = labels.reindex(train_idx).astype(int)
        val_y = labels.reindex(val_idx).astype(int)
        if len(train_y) < 100 or len(val_y) < 100 or train_y.nunique() < 2 or not feature_cols:
            continue
        params = json.loads(str(grid_row["params_json"]))
        estimator = make_model(model, params)
        if estimator is None:
            continue
        base_id = str(grid_row["grid_id"])
        try:
            estimator.fit(clean_feature_matrix(features.loc[train_idx], feature_cols), train_y)
            val_prob = predict_probability(estimator, clean_feature_matrix(features.loc[val_idx], feature_cols))
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": f"{base_id}__failed",
                    "base_id": base_id,
                    "source": grid_row["source"],
                    "model": model,
                    "feature_group": feature_group,
                    "horizon": horizon,
                    "status": "failed",
                    "failure_reason": str(exc)[:300],
                    "final_accuracy_used_for_selection": False,
                    "claim_label": "not_claimable",
                }
            )
            continue
        train_prob = predict_probability(estimator, clean_feature_matrix(features.loc[train_idx], feature_cols))
        payloads[base_id] = {
            "payload_type": "model",
            "base_id": base_id,
            "source": grid_row["source"],
            "model": model,
            "feature_group": feature_group,
            "horizon": horizon,
            "model_object": estimator,
            "feature_cols": feature_cols,
            "labels": labels,
            "splits": splits,
            "validation_prob": val_prob,
            "params_json": str(grid_row["params_json"]),
            "train_accuracy": accuracy(train_y, (train_prob >= 0.50).astype(int)),
        }
        _baseline_df, strongest, frames = baseline_cache[(horizon, "validation")]
        strongest_frame = frames[str(strongest["baseline_name"])]
        simplicity = max(0.0, (6.0 - MODEL_COMPLEXITY.get(model, 5)) / 5.0)
        for threshold in THRESHOLDS:
            candidate_id = f"{base_id}__t{threshold:.3f}".replace(".", "p")
            val_frame = prediction_frame(features, val_idx, labels, val_prob, threshold, candidate_id, "validation")
            row, q, t, balance = validation_result_row(
                candidate_id,
                base_id,
                str(grid_row["source"]),
                model,
                feature_group,
                horizon,
                threshold,
                str(grid_row["params_json"]),
                val_frame,
                strongest,
                strongest_frame,
                simplicity,
            )
            rows.append(row)
            quarter_rows.append(q)
            ticker_rows.append(t)
            balance_rows.append(balance)

    return (
        pd.DataFrame(rows),
        payloads,
        pd.concat(baseline_rows, ignore_index=True) if baseline_rows else pd.DataFrame(),
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.DataFrame(balance_rows),
        label_cache,
    )


def add_regime_gate_candidates(
    features: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    payloads: dict[str, dict[str, Any]],
    label_cache: dict[int, tuple[pd.Series, dict[str, pd.Index]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    quarter_rows: list[pd.DataFrame] = []
    ticker_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    new_payloads: dict[str, dict[str, Any]] = {}
    for feature_group in ["feature_set_C_closest", "feature_set_C_closest_plus_index_context", "old_baseline_C_closest"]:
        feature_cols = feature_groups.get(feature_group, [])
        if not feature_cols:
            continue
        for horizon in HORIZONS:
            labels, splits = label_cache[horizon]
            val_idx = splits["validation"]
            train_y = labels.reindex(splits["train"]).astype(int)
            if len(train_y) < 100 or train_y.nunique() < 2:
                continue
            for group_col, model_name in [("market_direction_regime", "regime_gated_ensemble"), ("volatility_regime", "regime_gated_ensemble")]:
                base_id = f"regime_gate__{group_col}__{feature_group}__h{horizon}"
                try:
                    router = fit_regime_router(features, labels, splits, feature_cols, group_col)
                    val_prob = predict_regime_router(router, features, val_idx)
                except Exception:
                    continue
                new_payloads[base_id] = {
                    "payload_type": "regime_gate",
                    "base_id": base_id,
                    "source": "regime_gate",
                    "model": model_name,
                    "feature_group": feature_group,
                    "horizon": horizon,
                    "router": router,
                    "feature_cols": feature_cols,
                    "labels": labels,
                    "splits": splits,
                    "validation_prob": val_prob,
                    "params_json": json.dumps({"group_col": group_col, "base_model": "logistic_l2_C0.3_balanced"}, sort_keys=True),
                }
                baseline_df, strongest, frames = baseline_frames_for_split(features, labels, splits, "validation", horizon)
                del baseline_df
                strongest_frame = frames[str(strongest["baseline_name"])]
                simplicity = max(0.0, (6.0 - MODEL_COMPLEXITY[model_name]) / 5.0)
                for threshold in THRESHOLDS:
                    candidate_id = f"{base_id}__t{threshold:.3f}".replace(".", "p")
                    val_frame = prediction_frame(features, val_idx, labels, val_prob, threshold, candidate_id, "validation")
                    row, q, t, balance = validation_result_row(
                        candidate_id,
                        base_id,
                        "regime_gate",
                        model_name,
                        feature_group,
                        horizon,
                        threshold,
                        new_payloads[base_id]["params_json"],
                        val_frame,
                        strongest,
                        strongest_frame,
                        simplicity,
                    )
                    rows.append(row)
                    quarter_rows.append(q)
                    ticker_rows.append(t)
                    balance_rows.append(balance)
    payloads.update(new_payloads)
    return (
        pd.DataFrame(rows),
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.DataFrame(balance_rows),
        new_payloads,
    )


def add_soft_vote_candidates(
    features: pd.DataFrame,
    validation_results: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    base_results = validation_results[
        validation_results["status"].eq("ok")
        & validation_results["model"].isin(["logistic_regression", "elasticnet_logistic", "random_forest", "xgboost", "lightgbm"])
    ].copy()
    if base_results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    base_results = base_results.sort_values(
        by=["feature_group", "horizon", "model", "validation_composite_score", "validation_accuracy"],
        ascending=[True, True, True, False, False],
    )
    top = base_results.groupby(["feature_group", "horizon", "model"], as_index=False).head(1)
    rows: list[dict[str, Any]] = []
    quarter_rows: list[pd.DataFrame] = []
    ticker_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    new_payloads: dict[str, dict[str, Any]] = {}
    combos = [
        ("logistic_tree", ["logistic_regression", "random_forest"]),
        ("logistic_xgboost", ["logistic_regression", "xgboost"]),
        ("logistic_lightgbm", ["logistic_regression", "lightgbm"]),
        ("tree_boosting", ["random_forest", "xgboost", "lightgbm"]),
    ]
    for feature_group in sorted(top["feature_group"].unique()):
        for horizon in sorted(top["horizon"].astype(int).unique()):
            subset = top[(top["feature_group"].eq(feature_group)) & (top["horizon"].astype(int).eq(horizon))]
            by_model = {str(row["model"]): row for _, row in subset.iterrows()}
            for combo_name, model_names in combos:
                if not all(name in by_model for name in model_names):
                    continue
                base_rows = [by_model[name] for name in model_names]
                base_payloads = [payloads[str(row["base_id"])] for row in base_rows]
                first = base_payloads[0]
                labels = first["labels"]
                splits = first["splits"]
                val_idx = splits["validation"]
                if not all(payload["splits"]["validation"].equals(val_idx) for payload in base_payloads):
                    continue
                weights_specs = {
                    "equal": np.repeat(1.0 / len(base_payloads), len(base_payloads)),
                    "validation_weighted": np.asarray(
                        [max(float(row["validation_lift"]), 0.0001) for row in base_rows],
                        dtype=float,
                    ),
                }
                weights_specs["validation_weighted"] = weights_specs["validation_weighted"] / weights_specs["validation_weighted"].sum()
                for weight_name, weights in weights_specs.items():
                    base_id = f"soft_vote__{combo_name}__{feature_group}__h{horizon}__{weight_name}"
                    val_prob = np.zeros(len(val_idx), dtype=float)
                    for weight, payload in zip(weights, base_payloads):
                        val_prob += float(weight) * np.asarray(payload["validation_prob"], dtype=float)
                    params = {
                        "base_model_ids": [str(row["base_id"]) for row in base_rows],
                        "weights": {name: float(weight) for name, weight in zip(model_names, weights)},
                    }
                    new_payloads[base_id] = {
                        "payload_type": "soft_vote",
                        "base_id": base_id,
                        "source": "ensemble",
                        "model": "soft_vote_ensemble",
                        "feature_group": feature_group,
                        "horizon": horizon,
                        "labels": labels,
                        "splits": splits,
                        "base_model_ids": params["base_model_ids"],
                        "weights": weights,
                        "validation_prob": val_prob,
                        "params_json": json.dumps(params, sort_keys=True),
                    }
                    baseline_df, strongest, frames = baseline_frames_for_split(features, labels, splits, "validation", horizon)
                    del baseline_df
                    strongest_frame = frames[str(strongest["baseline_name"])]
                    simplicity = max(0.0, (6.0 - MODEL_COMPLEXITY["soft_vote_ensemble"]) / 5.0)
                    for threshold in THRESHOLDS:
                        candidate_id = f"{base_id}__t{threshold:.3f}".replace(".", "p")
                        val_frame = prediction_frame(features, val_idx, labels, val_prob, threshold, candidate_id, "validation")
                        row, q, t, balance = validation_result_row(
                            candidate_id,
                            base_id,
                            "ensemble",
                            "soft_vote_ensemble",
                            feature_group,
                            horizon,
                            threshold,
                            new_payloads[base_id]["params_json"],
                            val_frame,
                            strongest,
                            strongest_frame,
                            simplicity,
                        )
                        rows.append(row)
                        quarter_rows.append(q)
                        ticker_rows.append(t)
                        balance_rows.append(balance)
    payloads.update(new_payloads)
    return (
        pd.DataFrame(rows),
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.DataFrame(balance_rows),
        new_payloads,
    )


def select_locked_candidate(validation_results: pd.DataFrame) -> dict[str, Any]:
    valid = validation_results[validation_results["status"].eq("ok")].copy()
    if valid.empty:
        raise ValueError("no validation candidates completed")
    shortlisted = valid[valid["shortlist_pass"].astype(bool)].copy()
    pool = shortlisted if not shortlisted.empty else valid
    pool = pool.sort_values(
        by=[
            "validation_composite_score",
            "validation_lift",
            "validation_accuracy",
            "quarter_min_accuracy",
            "ticker_median_lift",
            "rolling250_min",
            "prediction_balance_score",
            "simplicity_score",
        ],
        ascending=[False, False, False, False, False, False, False, False],
    )
    locked = pool.iloc[0].to_dict()
    locked["locked_at_utc"] = now_utc()
    locked["selected_from_shortlist"] = bool(not shortlisted.empty)
    locked["selection_rule"] = (
        "validation composite score after strict shortlist filters; if no shortlist passes, best validation diagnostic is locked as non-claimable"
    )
    locked["final_accuracy_used_for_selection"] = False
    return locked


def final_probability_for_payload(payload: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    final_idx = payload["splits"]["final"]
    if payload["payload_type"] == "model":
        return predict_probability(payload["model_object"], clean_feature_matrix(features.loc[final_idx], payload["feature_cols"]))
    if payload["payload_type"] == "regime_gate":
        return predict_regime_router(payload["router"], features, final_idx)
    if payload["payload_type"] == "soft_vote":
        out = np.zeros(len(final_idx), dtype=float)
        for weight, base_id in zip(payload["weights"], payload["base_model_ids"]):
            base_payload = payloads_global[base_id]
            out += float(weight) * final_probability_for_payload(base_payload, features)
        return out
    raise ValueError(f"unknown payload type {payload['payload_type']}")


payloads_global: dict[str, dict[str, Any]] = {}


def final_claim_label(row: dict[str, Any], validation_governed: bool, leaderboard_role: str) -> str:
    if leaderboard_role == "exploratory_final_rank" and not validation_governed:
        return "exploratory_not_claimable"
    final_accuracy = float(row.get("final_accuracy", math.nan))
    final_lift = float(row.get("final_lift", math.nan))
    if not validation_governed:
        return "future_blind_required" if final_accuracy > CURRENT_CHAMPION["final_accuracy"] else "exploratory_not_claimable"
    if final_lift <= 0.0:
        return "not_claimable"
    if final_accuracy >= 0.62:
        return "target62_candidate"
    if final_accuracy >= 0.60:
        return "baseline60_candidate"
    return "diagnostic_only"


def evaluate_final_candidates(
    features: pd.DataFrame,
    validation_results: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
    locked_candidate_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global payloads_global
    payloads_global = payloads
    final_prob_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    quarter_rows: list[pd.DataFrame] = []
    ticker_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    valid = validation_results[validation_results["status"].eq("ok")].copy()
    baseline_cache: dict[int, tuple[dict[str, Any], dict[str, pd.DataFrame]]] = {}
    for horizon in sorted(valid["horizon"].astype(int).unique()):
        sample_payload = next(payload for payload in payloads.values() if int(payload["horizon"]) == int(horizon))
        labels = sample_payload["labels"]
        splits = sample_payload["splits"]
        _baseline_df, strongest, frames = baseline_frames_for_split(features, labels, splits, "final", int(horizon))
        baseline_cache[int(horizon)] = (strongest, frames)

    for val_row in valid.to_dict("records"):
        base_id = str(val_row["base_id"])
        if base_id not in payloads:
            continue
        payload = payloads[base_id]
        if base_id not in final_prob_cache:
            final_prob_cache[base_id] = final_probability_for_payload(payload, features)
        threshold = float(val_row["threshold"])
        candidate_id = str(val_row["candidate_id"])
        final_idx = payload["splits"]["final"]
        labels = payload["labels"]
        final_frame = prediction_frame(features, final_idx, labels, final_prob_cache[base_id], threshold, candidate_id, "final")
        strongest, frames = baseline_cache[int(val_row["horizon"])]
        strongest_frame = frames[str(strongest["baseline_name"])]
        simplicity = float(val_row.get("simplicity_score", 0.0))
        metrics, q, t, balance = candidate_metrics(final_frame, strongest, strongest_frame, candidate_id, "final", simplicity)
        validation_governed = bool(val_row.get("shortlist_pass", False) or candidate_id == locked_candidate_id)
        row = {
            "candidate_id": candidate_id,
            "source": val_row["source"],
            "model": val_row["model"],
            "feature_group": val_row["feature_group"],
            "horizon": int(val_row["horizon"]),
            "threshold": threshold,
            "validation_accuracy": float(val_row["validation_accuracy"]),
            "validation_lift": float(val_row["validation_lift"]),
            "final_accuracy": metrics["final_accuracy"],
            "final_lift": metrics["final_lift"],
            "final_rows": metrics["final_rows"],
            "strongest_baseline": metrics["strongest_baseline"],
            "ticker_median_accuracy": metrics["ticker_median_accuracy"],
            "quarter_min_accuracy": metrics["quarter_min_accuracy"],
            "rolling250_min": metrics["rolling250_min"],
            "prediction_up_ratio": metrics["prediction_up_ratio"],
            "validation_composite_score": val_row["validation_composite_score"],
            "validation_shortlist_pass": bool(val_row["shortlist_pass"]),
            "locked_validation_candidate": candidate_id == locked_candidate_id,
            "leaderboard_role": "validation_governed" if validation_governed else "exploratory_final_rank",
            "split_guard_passed": True,
        }
        row["claim_label"] = final_claim_label(row, validation_governed, row["leaderboard_role"])
        rows.append(row)
        quarter_rows.append(q)
        ticker_rows.append(t)
        balance_rows.append(balance)
    return (
        pd.DataFrame(rows),
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.DataFrame(balance_rows),
    )


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def as_float(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def build_historical_candidate_registry(final_leaderboard: pd.DataFrame) -> pd.DataFrame:
    source_paths = [
        REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "final_results.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "augmented_leaderboard.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "fair_tuning" / "descriptive_final_leaderboard.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "fair_tuning" / "fair_tuning_final_results.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "beating_rows_diagnostic.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_target62_validation_safe" / "selected_candidate_summary.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_target62_validation_safe" / "final_candidate_results.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_hourly_validation_safe_improvement_tracks" / "final_scoring_results.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_selected_candidate_strict_replay" / "old_candidate_strict_final_result.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_aggressive_model_tuning" / "final_once_result.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_champion_rescue_tuning" / "final_once_result.csv",
        REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark" / "above60" / "vn30_all_60pct_candidates.csv",
    ]
    rows: list[dict[str, Any]] = []
    for path in source_paths:
        frame = read_csv_if_exists(path)
        if frame.empty:
            continue
        for _, raw in frame.iterrows():
            data = raw.to_dict()
            old_accuracy = math.nan
            for col in ("final_accuracy", "accuracy", "model_accuracy"):
                if col in data:
                    old_accuracy = as_float(data.get(col))
                    if math.isfinite(old_accuracy):
                        break
            selected_text = " ".join(str(data.get(col, "")) for col in data)
            selected = "selected" in selected_text.lower() or "baseline60" in selected_text.lower() or "target62" in selected_text.lower()
            if not (math.isfinite(old_accuracy) and old_accuracy >= 0.60) and not selected:
                continue
            old_model = str(data.get("model", data.get("model_id", data.get("model_family", data.get("model_group", "")))))
            old_feature = str(data.get("feature_set", data.get("feature_family", data.get("feature_group", data.get("filter_description", "")))))
            old_horizon = data.get("horizon", "")
            old_threshold = data.get("threshold", "")
            old_rows = data.get("final_rows", data.get("observations", data.get("rows", "")))
            candidate_id = str(data.get("candidate_id", f"{old_model}__{old_feature}__h{old_horizon}__t{old_threshold}"))
            old_row_count_num = as_float(old_rows)
            source_text = rel(path).lower()
            text_for_priority = f"{candidate_id} {old_model} {old_feature} {selected_text}".lower()
            subset_or_confidence_slice = bool(
                "above60" in source_text
                or "conf>=" in text_for_priority
                or "ticker=" in text_for_priority
                or "regime=" in text_for_priority
                or "topk" in text_for_priority
            )
            full_scope_like = bool(
                (math.isfinite(old_row_count_num) and old_row_count_num >= 3000)
                or str(data.get("full_ticker_coverage", "")).lower() == "true"
                or str(data.get("final_coverage", "")) in {"1.0", "1"}
                or "selected_candidate" in source_text
                or "model_universe_benchmark/final_results" in source_text
                or "fair_tuning" in source_text
            )
            registry_priority = 2 if full_scope_like and not subset_or_confidence_slice else (1 if full_scope_like else 0)
            if "bull_bear_sideway_router" in text_for_priority:
                registry_priority = 3
            if "old_candidate_strict_final_result" in source_text:
                registry_priority = 3
            rows.append(
                {
                    "source_file": rel(path),
                    "candidate_id": candidate_id,
                    "old_model": old_model,
                    "old_feature_set": old_feature,
                    "old_horizon": old_horizon,
                    "old_threshold": old_threshold,
                    "old_accuracy": old_accuracy,
                    "old_row_count": old_rows,
                    "old_baseline_or_lift": data.get("final_lift_over_strongest_baseline", data.get("final_delta_vs_baseline", data.get("lift_vs_majority", ""))),
                    "old_split_rule": data.get("selection_source", data.get("selected_by_validation_yes_no", data.get("selected_by_preregistered_rule", ""))),
                    "strict_replay_status": "not_replayed",
                    "strict_replay_accuracy": math.nan,
                    "strict_replay_strongest_baseline": "",
                    "strict_replay_lift": math.nan,
                    "status": "insufficient_metadata",
                    "_registry_priority": registry_priority,
                    "_old_row_count_num": old_row_count_num,
                }
            )
    if not rows:
        return pd.DataFrame()
    registry = pd.DataFrame(rows).drop_duplicates(["source_file", "candidate_id"]).copy()
    registry = registry.sort_values(
        ["_registry_priority", "_old_row_count_num", "old_accuracy"],
        ascending=[False, False, False],
        na_position="last",
    ).head(500).reset_index(drop=True)
    final_lookup = final_leaderboard.set_index("candidate_id").to_dict("index") if not final_leaderboard.empty else {}
    for idx, row in registry.iterrows():
        cid = str(row["candidate_id"])
        text = " ".join(str(row.get(col, "")) for col in ["candidate_id", "old_model", "old_feature_set"]).lower()
        matched: dict[str, Any] | None = None
        if "old_selected_l2_logistic" in cid or (
            ("l2_logistic" in text or "logistic_l2" in text)
            and "feature_set_c_closest" in text
            and str(row.get("old_horizon", "")) == "40"
            and str(row.get("old_threshold", "")) in {"0.5", "0.50"}
        ):
            candidates = [key for key in final_lookup if key.startswith("forced_old_selected_l2_h40__t0p500")]
            candidates.extend([key for key in final_lookup if key.startswith("grid_")])
            for key in candidates:
                item = final_lookup[key]
                if item.get("model") == "logistic_regression" and item.get("feature_group") == "feature_set_C_closest" and int(item.get("horizon", -1)) == 40 and abs(float(item.get("threshold", 0)) - 0.5) < 1e-9:
                    matched = item
                    break
        if "bull_bear_sideway_router" in text and int(as_float(row.get("old_horizon", -1))) == 40:
            for key, item in final_lookup.items():
                if (
                    str(item.get("source")) == "regime_gate"
                    and str(item.get("feature_group")) in {"feature_set_C_closest", "old_baseline_C_closest"}
                    and int(item.get("horizon", -1)) == 40
                    and abs(float(item.get("threshold", 0)) - 0.5) < 1e-9
                ):
                    if "market_direction_regime" in str(key):
                        matched = item
                        break
        if matched:
            registry.loc[idx, "strict_replay_status"] = "completed"
            registry.loc[idx, "strict_replay_accuracy"] = matched.get("final_accuracy", math.nan)
            registry.loc[idx, "strict_replay_strongest_baseline"] = matched.get("strongest_baseline", "")
            registry.loc[idx, "strict_replay_lift"] = matched.get("final_lift", math.nan)
            registry.loc[idx, "status"] = "survives" if as_float(matched.get("final_accuracy")) >= 0.60 and as_float(matched.get("final_lift")) > 0 else "dies"
        elif any(token in text for token in ["router", "regime", "calibration", "isotonic", "platt", "xgboost", "lightgbm", "random_forest", "stacking", "ensemble"]):
            registry.loc[idx, "status"] = "needs_relock"
    return registry.drop(columns=[col for col in ["_registry_priority", "_old_row_count_num"] if col in registry.columns])


def write_reports(
    run_config: dict[str, Any],
    locked: dict[str, Any],
    final_once: dict[str, Any],
    validation_leaderboard: pd.DataFrame,
    exploratory_leaderboard: pd.DataFrame,
    historical_registry: pd.DataFrame,
    champion_decision: dict[str, Any],
) -> None:
    old63 = historical_registry[
        historical_registry["old_accuracy"].apply(lambda value: math.isfinite(as_float(value)) and 0.63 <= as_float(value) < 0.64)
    ].sort_values("old_accuracy", ascending=False)
    old63_router = old63[old63["candidate_id"].astype(str).str.contains("bull_bear_sideway_router", case=False, regex=False)]
    if not old63_router.empty:
        old63 = old63_router
    old63_row = old63.iloc[0].to_dict() if not old63.empty else {}
    old63_strict_accuracy = as_float(old63_row.get("strict_replay_accuracy", math.nan))
    old63_retained_63 = bool(math.isfinite(old63_strict_accuracy) and old63_strict_accuracy >= 0.63)
    old63_survival_text = (
        "survives as baseline60 but not at the old 63% level"
        if old63_row.get("status") == "survives" and not old63_retained_63
        else str(old63_row.get("status", "not_found"))
    )
    best_validation = validation_leaderboard.iloc[0].to_dict() if not validation_leaderboard.empty else {}
    best_exploratory = exploratory_leaderboard.iloc[0].to_dict() if not exploratory_leaderboard.empty else {}
    target62_validation = bool((validation_leaderboard.get("claim_label", pd.Series(dtype=str)).eq("target62_candidate")).any()) if not validation_leaderboard.empty else False
    baseline60_validation = bool((validation_leaderboard.get("claim_label", pd.Series(dtype=str)).isin(["baseline60_candidate", "target62_candidate"])).any()) if not validation_leaderboard.empty else False
    final65_validation = bool((validation_leaderboard.get("final_accuracy", pd.Series(dtype=float)).astype(float) >= 0.65).any()) if not validation_leaderboard.empty else False
    exploratory_over62 = bool((exploratory_leaderboard.get("final_accuracy", pd.Series(dtype=float)).astype(float) >= 0.62).any()) if not exploratory_leaderboard.empty else False
    exploratory_over63 = bool((exploratory_leaderboard.get("final_accuracy", pd.Series(dtype=float)).astype(float) >= 0.63).any()) if not exploratory_leaderboard.empty else False

    protocol = f"""# VN30 Full Model Resurrection And Index-Pretrain Protocol

## Scope

- Final target: VN30 stock hourly directional benchmark only.
- Index usage: main market indices are used only to build point-in-time lagged market-context features.
- Index benchmark results are not stock benchmark claims.
- Top-k ranking is not used as overall directional accuracy.
- Out of scope: trading, profitability, BUY/SELL, recommendations, live deployment, DOCX, paper generation, git tags.

## Split Rules

- Train rows require feature_timestamp <= `{TRAIN_END}` and target_timestamp <= `{TRAIN_END}`.
- Validation rows require feature_timestamp and target_timestamp from `{VAL_START}` through `{VAL_END}`.
- Final rows require feature_timestamp and target_timestamp >= `{FINAL_START}`.
- Candidate, model, threshold, and lock selection use validation only.
- Final-ranked exploration is written separately as `exploratory_final_leaderboard.csv` and is not claimable.

## Index Context Layer

- Required context features: market_direction_lag1, market_direction_lag5, market_return_lag1, market_return_lag5, market_volatility_5, market_volatility_20, index_agreement_score, risk_on_risk_off_state, market_momentum_5, market_momentum_20, cross_index_breadth_proxy.
- All index features are lagged or computed from rolling windows shifted one bar before merging to stock rows.
- Stock labels are not used in the index layer.

## Tuning

- Horizons: {HORIZONS}.
- Thresholds: {THRESHOLDS[0]:.3f} to {THRESHOLDS[-1]:.3f} step 0.005.
- Feature groups: {", ".join(FEATURE_GROUP_ORDER)}.
- Model families: logistic regression, elasticnet logistic, random forest, XGBoost, LightGBM, soft-vote ensemble, regime-gated ensemble.
- Full grid is enumerated in `candidate_grid.csv`; budgeted staged screening is recorded in `run_config.json`.

## Validation Composite Score

score = 0.35 * validation_lift + 0.25 * validation_accuracy + 0.15 * quarterly_stability_score + 0.10 * ticker_stability_score + 0.10 * prediction_balance_score + 0.05 * simplicity_score.
"""
    write_markdown(PROTOCOL_PATH, protocol)

    result = f"""# VN30 Full Model Resurrection And Index-Pretrain Result Summary

## Historical 63% Candidate

- Found old around-63% candidate: `{old63_row.get("candidate_id", "")}`.
- Source: `{old63_row.get("source_file", "")}`.
- Old model/feature/horizon/threshold: {old63_row.get("old_model", "")} / {old63_row.get("old_feature_set", "")} / h{old63_row.get("old_horizon", "")} / {old63_row.get("old_threshold", "")}.
- Old accuracy: {pct(old63_row.get("old_accuracy", math.nan))}.
- Strict replay status: {old63_survival_text}.
- Strict replay accuracy: {pct(old63_row.get("strict_replay_accuracy", math.nan))}.
- Strict replay retained old 63% level: {str(old63_retained_63).lower()}.

## Validation-Governed Result

- Locked candidate: `{locked.get("candidate_id", "")}`.
- Selected from strict shortlist: {str(bool(locked.get("selected_from_shortlist", False))).lower()}.
- Best validation-governed candidate: `{best_validation.get("candidate_id", "")}`.
- Best validation-governed final accuracy: {pct(best_validation.get("final_accuracy", math.nan))}.
- Best validation-governed final lift: {pp(best_validation.get("final_lift", math.nan))}.
- Locked final accuracy: {pct(final_once.get("final_accuracy", math.nan))}.
- Locked final lift: {pp(final_once.get("final_lift", math.nan))}.
- Locked final rows: {final_once.get("final_rows", "")}.

## Exploratory Final Result

- Best exploratory final candidate: `{best_exploratory.get("candidate_id", "")}`.
- Best exploratory final accuracy: {pct(best_exploratory.get("final_accuracy", math.nan))}.
- Best exploratory final lift: {pp(best_exploratory.get("final_lift", math.nan))}.
- Exploratory final candidate exceeded 62%: {str(exploratory_over62).lower()}.
- Exploratory final candidate exceeded 63%: {str(exploratory_over63).lower()}.

## Champion Decision

- Current champion: L2 Logistic, feature_set_C_closest, h40, threshold 0.50, 61.61% final accuracy, +10.90 pp lift, 4,074 rows.
- New champion replaces current champion: {str(bool(champion_decision.get("new_champion_replaces_current", False))).lower()}.
- Reason: {champion_decision.get("decision_reason", "")}.

## Claim Boundary Answers

1. Old 63% candidate found: `{old63_row.get("candidate_id", "")}` at {pct(old63_row.get("old_accuracy", math.nan))}.
2. Strict replay survival: {old63_survival_text}.
3. Any validation-governed candidate beat 61.61%: {str(bool(champion_decision.get("validation_governed_beats_accuracy", False))).lower()}.
4. Any candidate beat +10.90 pp lift: {str(bool(champion_decision.get("any_candidate_beats_lift", False))).lower()}.
5. Exploratory final exceeded 62% or 63%: 62={str(exploratory_over62).lower()}, 63={str(exploratory_over63).lower()}.
6. Claimable result: {final_once.get("claim_label", "not_claimable")}.
7. Exploratory-only result: best final-ranked rows in `exploratory_final_leaderboard.csv`.
8. Baseline60 defensible: {str(baseline60_validation).lower()}; target62 defensible: {str(target62_validation).lower()}; final65 defensible: {str(final65_validation and bool(champion_decision.get("new_champion_replaces_current", False))).lower()}.

Paper-safe wording:

> In a validation-governed VN30 stock hourly diagnostic benchmark with lagged market-index context features, the locked candidate reached {pct(final_once.get("final_accuracy", math.nan))} final pooled directional accuracy over {final_once.get("final_rows", "")} rows, {pp(final_once.get("final_lift", math.nan))} versus the strongest same-horizon simple baseline. The current strict-replay L2 Logistic h40 champion at 61.61% and +10.90 pp is {'' if champion_decision.get("new_champion_replaces_current", False) else 'not '}replaced. Final-ranked rows outside the validation-governed lock are exploratory only and require re-locking or future-blind confirmation.
"""
    write_markdown(RESULT_PATH, result)

    claim = f"""# VN30 Full Model Resurrection And Index-Pretrain Claim Boundary

- Claimable scope: VN30 stock hourly diagnostic benchmark only.
- Index layer: lagged market-context features only; no index benchmark result is claimed as stock accuracy.
- Locked candidate: `{locked.get("candidate_id", "")}`.
- Final role: one-time locked-candidate scoring after `locked_validation_candidate.json`.
- Locked final accuracy: {pct(final_once.get("final_accuracy", math.nan))}.
- Strongest final baseline: {final_once.get("strongest_baseline", "")}.
- Locked final lift: {pp(final_once.get("final_lift", math.nan))}.
- Claim label: {final_once.get("claim_label", "not_claimable")}.
- Current 61.61% champion replaced: {str(bool(champion_decision.get("new_champion_replaces_current", False))).lower()}.
- Baseline60, target62, and final65 claims are limited by `champion_comparison.csv` and validation-governed selection.
- No trading, profitability, BUY/SELL, recommendation, live deployment, DOCX, paper, index-as-stock, top-k-as-overall, push, merge, or tag claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def champion_comparison_rows(final_once: dict[str, Any], validation_leaderboard: pd.DataFrame, exploratory_leaderboard: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    best_val = validation_leaderboard.iloc[0].to_dict() if not validation_leaderboard.empty else {}
    best_exp = exploratory_leaderboard.iloc[0].to_dict() if not exploratory_leaderboard.empty else {}
    current = {
        "comparison_role": "current_champion",
        "candidate_id": CURRENT_CHAMPION["candidate_id"],
        "final_accuracy": CURRENT_CHAMPION["final_accuracy"],
        "final_lift": CURRENT_CHAMPION["final_lift"],
        "final_rows": CURRENT_CHAMPION["final_rows"],
        "claim_label": "baseline60_candidate",
        "validation_governed": True,
        "replaces_current": False,
        "reason": "incumbent",
    }
    def row_from(role: str, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "comparison_role": role,
            "candidate_id": row.get("candidate_id", ""),
            "final_accuracy": row.get("final_accuracy", math.nan),
            "final_lift": row.get("final_lift", math.nan),
            "final_rows": row.get("final_rows", math.nan),
            "claim_label": row.get("claim_label", ""),
            "validation_governed": bool(row.get("leaderboard_role") == "validation_governed" or row.get("locked_validation_candidate", False)),
            "replaces_current": False,
            "reason": "",
        }
    rows = [current, row_from("locked_validation_candidate", final_once), row_from("best_validation_governed", best_val), row_from("best_exploratory_final", best_exp)]
    best_claimable = best_val if best_val else final_once
    stability_not_worse = bool(
        as_float(best_claimable.get("quarter_min_accuracy")) >= CURRENT_CHAMPION["quarter_min_accuracy"]
        and as_float(best_claimable.get("ticker_median_accuracy")) >= CURRENT_CHAMPION["ticker_median_accuracy"] - 0.03
    )
    comparable_rows = bool(as_float(best_claimable.get("final_rows")) >= CURRENT_CHAMPION["final_rows"] * 0.95)
    beats_accuracy = bool(as_float(best_claimable.get("final_accuracy")) > CURRENT_CHAMPION["final_accuracy"])
    beats_lift = bool(as_float(best_claimable.get("final_lift")) > CURRENT_CHAMPION["final_lift"])
    validation_governed = bool(best_claimable.get("leaderboard_role") == "validation_governed" or best_claimable.get("locked_validation_candidate", False))
    replace = bool(beats_accuracy and beats_lift and validation_governed and comparable_rows and stability_not_worse)
    reason_parts = []
    if not beats_accuracy:
        reason_parts.append("no validation-governed candidate beat 61.61% final accuracy")
    if not beats_lift:
        reason_parts.append("no validation-governed candidate beat +10.90 pp lift")
    if beats_accuracy and beats_lift and not validation_governed:
        reason_parts.append("beating row was not validation-governed")
    if beats_accuracy and beats_lift and validation_governed and not stability_not_worse:
        reason_parts.append("stability was materially worse")
    if beats_accuracy and beats_lift and validation_governed and not comparable_rows:
        reason_parts.append("row count was not comparable")
    if replace:
        reason_parts.append("validation-governed candidate beat accuracy and lift with comparable rows and stability")
    decision = {
        "new_champion_replaces_current": replace,
        "validation_governed_beats_accuracy": beats_accuracy and validation_governed,
        "validation_governed_beats_lift": beats_lift and validation_governed,
        "any_candidate_beats_lift": bool(not exploratory_leaderboard.empty and (exploratory_leaderboard["final_lift"].astype(float) > CURRENT_CHAMPION["final_lift"]).any()),
        "decision_reason": "; ".join(reason_parts),
        "evaluated_candidate_id": best_claimable.get("candidate_id", ""),
    }
    for row in rows:
        if row["candidate_id"] == best_claimable.get("candidate_id", ""):
            row["replaces_current"] = replace
            row["reason"] = decision["decision_reason"]
    return pd.DataFrame(rows), decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 full model resurrection with index-context features.")
    parser.add_argument("--logistic-budget", type=int, default=70)
    parser.add_argument("--rf-budget", type=int, default=18)
    parser.add_argument("--xgb-budget", type=int, default=14)
    parser.add_argument("--lgbm-budget", type=int, default=14)
    parser.add_argument("--report-only", action="store_true", help="Regenerate registry, comparison, and markdown reports from existing outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        run_config = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))
        locked = json.loads((OUTPUT_DIR / "locked_validation_candidate.json").read_text(encoding="utf-8"))
        final_once = pd.read_csv(OUTPUT_DIR / "final_once_result.csv").iloc[0].to_dict()
        validation_governed = pd.read_csv(OUTPUT_DIR / "validation_governed_leaderboard.csv", low_memory=False)
        exploratory = pd.read_csv(OUTPUT_DIR / "exploratory_final_leaderboard.csv", low_memory=False)
        historical_registry = build_historical_candidate_registry(exploratory)
        champion_comparison, champion_decision = champion_comparison_rows(final_once, validation_governed, exploratory)
        write_frame(OUTPUT_DIR / "historical_candidate_registry.csv", historical_registry)
        write_frame(OUTPUT_DIR / "champion_comparison.csv", champion_comparison)
        manifest_path = OUTPUT_DIR / "tuning_manifest.json"
        tuning_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        tuning_manifest["historical_registry_regenerated_at_utc"] = now_utc()
        tuning_manifest["champion_decision"] = champion_decision
        write_json(manifest_path, tuning_manifest)
        write_reports(run_config, locked, final_once, validation_governed, exploratory, historical_registry, champion_decision)
        print(f"Report-only regeneration complete: {rel(OUTPUT_DIR)}")
        return 0
    budgets = {
        "logistic_regression": int(args.logistic_budget),
        "elasticnet_logistic": max(1, int(args.logistic_budget // 4)),
        "random_forest": int(args.rf_budget),
        "xgboost": int(args.xgb_budget),
        "lightgbm": int(args.lgbm_budget),
    }
    features, feature_groups, feature_manifest, index_audit, index_manifest = build_feature_frame()
    candidate_grid = enumerate_candidate_grid(feature_groups)
    fit_grid = select_budgeted_grid(candidate_grid, budgets)
    candidate_grid.loc[candidate_grid["grid_id"].isin(set(fit_grid["grid_id"].astype(str))), "selected_for_fit"] = True
    candidate_grid.loc[candidate_grid["selected_for_fit"].astype(bool), "stage"] = candidate_grid.loc[candidate_grid["selected_for_fit"].astype(bool), "stage"].where(
        candidate_grid.loc[candidate_grid["selected_for_fit"].astype(bool), "stage"].eq("forced_historical_replay"),
        "cheap_screening_selected",
    )
    run_config = {
        "created_at_utc": now_utc(),
        "scope": "VN30 stock hourly full model resurrection with index-context pretraining features",
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
        "horizons": HORIZONS,
        "threshold_start": THRESHOLDS[0],
        "threshold_end": THRESHOLDS[-1],
        "threshold_step": 0.005,
        "fit_budget_per_family": budgets,
        "candidate_grid_rows": int(len(candidate_grid)),
        "fit_grid_rows": int(len(fit_grid)),
        "final_accuracy_used_for_selection": False,
        "exploratory_final_ranking_claimable": False,
        "git_tags_created": False,
        "paper_docx_generated": False,
        "feature_manifest": feature_manifest,
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)
    write_frame(OUTPUT_DIR / "index_context_feature_audit.csv", index_audit)
    write_json(OUTPUT_DIR / "index_context_manifest.json", index_manifest)
    write_frame(OUTPUT_DIR / "candidate_grid.csv", candidate_grid)

    validation_results, payloads, baseline_comparison, quarter_stability, ticker_stability, prediction_balance, label_cache = fit_validation_candidates(features, feature_groups, fit_grid)
    regime_results, regime_q, regime_t, regime_b, regime_payloads = add_regime_gate_candidates(features, feature_groups, payloads, label_cache)
    if not regime_results.empty:
        validation_results = pd.concat([validation_results, regime_results], ignore_index=True)
        quarter_stability = pd.concat([quarter_stability, regime_q], ignore_index=True)
        ticker_stability = pd.concat([ticker_stability, regime_t], ignore_index=True)
        prediction_balance = pd.concat([prediction_balance, regime_b], ignore_index=True)
        payloads.update(regime_payloads)
    ensemble_results, ensemble_q, ensemble_t, ensemble_b, ensemble_payloads = add_soft_vote_candidates(features, validation_results, payloads)
    if not ensemble_results.empty:
        validation_results = pd.concat([validation_results, ensemble_results], ignore_index=True)
        quarter_stability = pd.concat([quarter_stability, ensemble_q], ignore_index=True)
        ticker_stability = pd.concat([ticker_stability, ensemble_t], ignore_index=True)
        prediction_balance = pd.concat([prediction_balance, ensemble_b], ignore_index=True)
        payloads.update(ensemble_payloads)

    locked = select_locked_candidate(validation_results)
    write_frame(OUTPUT_DIR / "validation_results.csv", validation_results)
    write_json(OUTPUT_DIR / "locked_validation_candidate.json", locked)

    final_all, final_q, final_t, final_b = evaluate_final_candidates(features, validation_results, payloads, str(locked["candidate_id"]))
    final_once = final_all[final_all["candidate_id"].astype(str).eq(str(locked["candidate_id"]))].copy()
    if final_once.empty:
        raise RuntimeError("locked candidate missing from final scoring")
    final_once_row = final_once.iloc[0].to_dict()
    write_frame(OUTPUT_DIR / "final_once_result.csv", final_once)

    validation_governed = final_all[final_all["leaderboard_role"].eq("validation_governed")].copy()
    validation_governed = validation_governed.sort_values(
        by=["locked_validation_candidate", "validation_composite_score", "final_accuracy", "final_lift"],
        ascending=[False, False, False, False],
    )
    exploratory = final_all.sort_values(by=["final_accuracy", "final_lift", "validation_accuracy"], ascending=[False, False, False]).copy()
    historical_registry = build_historical_candidate_registry(exploratory)
    champion_comparison, champion_decision = champion_comparison_rows(final_once_row, validation_governed, exploratory)

    write_frame(OUTPUT_DIR / "validation_governed_leaderboard.csv", validation_governed)
    write_frame(OUTPUT_DIR / "exploratory_final_leaderboard.csv", exploratory)
    write_frame(OUTPUT_DIR / "baseline_comparison.csv", baseline_comparison)
    write_frame(OUTPUT_DIR / "quarter_stability.csv", pd.concat([quarter_stability, final_q], ignore_index=True))
    write_frame(OUTPUT_DIR / "ticker_stability.csv", pd.concat([ticker_stability, final_t], ignore_index=True))
    write_frame(OUTPUT_DIR / "prediction_balance.csv", pd.concat([prediction_balance, final_b], ignore_index=True))
    write_frame(OUTPUT_DIR / "historical_candidate_registry.csv", historical_registry)
    write_frame(OUTPUT_DIR / "champion_comparison.csv", champion_comparison)

    tuning_manifest = {
        **run_config,
        "validation_result_rows": int(len(validation_results)),
        "regime_gate_payloads": int(len(regime_payloads)),
        "ensemble_payloads": int(len(ensemble_payloads)),
        "locked_validation_candidate": locked,
        "final_once_result": final_once_row,
        "champion_decision": champion_decision,
        "optional_dependencies": {
            "xgboost_available": XGBClassifier is not None,
            "lightgbm_available": LGBMClassifier is not None,
        },
    }
    write_json(OUTPUT_DIR / "tuning_manifest.json", tuning_manifest)
    write_reports(run_config, locked, final_once_row, validation_governed, exploratory, historical_registry, champion_decision)
    print(f"VN30 full model resurrection complete: {rel(OUTPUT_DIR)}")
    print(f"Locked candidate: {locked['candidate_id']}")
    print(f"Locked final accuracy: {pct(final_once_row['final_accuracy'])}")
    print(f"New champion replaces 61.61%: {str(bool(champion_decision['new_champion_replaces_current'])).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
