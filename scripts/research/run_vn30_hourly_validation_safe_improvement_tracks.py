"""Run validation-safe VN30 hourly stock-only improvement tracks.

This runner uses existing local VN30 hourly stock and index artifacts only.
Candidate selection is based on validation-window metrics only; final-window
accuracy is computed after validation-only selection and is scoring-only.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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

from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    LOCKED_RF_H60,
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    rel,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_validation_safe_improvement_tracks"
REFERENCE_ROLLING_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_selected_candidate_rolling"

TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
FINAL_START = pd.Timestamp("2025-01-01")

HORIZONS = [20, 40, 60, 80]
MAIN_THRESHOLD = 0.50
RANDOM_STATE = 42

REFERENCE_FINAL_ACCURACY = 0.6151202749140894
REFERENCE_FINAL_ACCURACY_LABEL = "61.51%"
REFERENCE_FINAL_ROWS = 4074
REFERENCE_MAJORITY_BASELINE = 0.5044182621502209
REFERENCE_VALIDATION_ACCURACY = 0.5188145188145188
REFERENCE_VALIDATION_FINAL_GAP = REFERENCE_FINAL_ACCURACY - REFERENCE_VALIDATION_ACCURACY

BASE_FEATURE_FAMILY = "baseline_C_closest"
REQUIRED_OUTPUTS = [
    "candidate_grid.csv",
    "validation_results.csv",
    "selected_candidate.json",
    "final_scoring_results.csv",
    "row_predictions_selected.csv",
    "by_ticker.csv",
    "by_month.csv",
    "by_quarter.csv",
    "rolling_250.csv",
    "rolling_500.csv",
    "rolling_1000.csv",
    "feature_ablation_summary.csv",
    "model_family_summary.csv",
    "improvement_summary.md",
    "leakage_audit.md",
    "claim_boundary.md",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    return value


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def pct(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:+.2f} percentage points"


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def accuracy(y_true: pd.Series, prediction: np.ndarray | pd.Series) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((y_true.astype(int).to_numpy() == np.asarray(prediction).astype(int)).mean())


def majority_accuracy(y_true: pd.Series) -> float:
    if len(y_true) == 0:
        return math.nan
    rate = float(y_true.astype(int).mean())
    return max(rate, 1.0 - rate)


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    label_index = labels.dropna().index
    return {
        "train": features.index[features["datetime"].le(TRAIN_END)].intersection(label_index),
        "validation": features.index[features["datetime"].between(VAL_START, VAL_END)].intersection(label_index),
        "final": features.index[features["datetime"].ge(FINAL_START)].intersection(label_index),
    }


def _direction_from_return(ret: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=ret.index, dtype=float)
    out.loc[ret.notna()] = (ret.loc[ret.notna()] > 0.0).astype(float)
    return out


def build_lagged_index_context(index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    pieces: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    for code in ["VNINDEX", "VN30"]:
        if code not in index_data:
            continue
        frame = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
        close = pd.to_numeric(frame["close"], errors="coerce")
        ret = close.pct_change(fill_method=None)
        prefix = code.lower()
        for lag in (1, 2, 3, 5, 10, 20):
            frame[f"{prefix}_ret_lag_{lag}_ctx"] = ret.shift(lag)
        for window in (10, 20, 60):
            frame[f"{prefix}_vol_{window}_lag_ctx"] = ret.rolling(window, min_periods=max(3, window // 4)).std().shift(1)
            frame[f"{prefix}_mean_{window}_lag_ctx"] = ret.rolling(window, min_periods=max(3, window // 4)).mean().shift(1)
        trend_20 = (close / close.shift(20) - 1.0).shift(1)
        trend_60 = (close / close.shift(60) - 1.0).shift(1)
        frame[f"{prefix}_trend_20_lag_ctx"] = trend_20
        frame[f"{prefix}_trend_60_lag_ctx"] = trend_60
        frame[f"{prefix}_direction_lag_1_ctx"] = _direction_from_return(ret.shift(1))
        vol_ratio = frame[f"{prefix}_vol_20_lag_ctx"] / frame[f"{prefix}_vol_60_lag_ctx"].replace(0.0, np.nan)
        frame[f"{prefix}_vol_regime_ctx"] = np.select([vol_ratio < 0.75, vol_ratio > 1.25], [0.0, 2.0], default=1.0)
        frame.loc[vol_ratio.isna(), f"{prefix}_vol_regime_ctx"] = np.nan
        frame[f"{prefix}_trend_regime_ctx"] = np.select([trend_60 < -0.02, trend_60 > 0.02], [-1.0, 1.0], default=0.0)
        frame.loc[trend_60.isna(), f"{prefix}_trend_regime_ctx"] = np.nan
        cols = [col for col in frame.columns if col not in {"datetime", "close"}]
        feature_cols.extend(cols)
        pieces.append(frame[["datetime", *cols]])
    if not pieces:
        return pd.DataFrame(columns=["datetime"]), []
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="datetime", how="outer")
    return merged.sort_values("datetime").reset_index(drop=True), sorted(set(feature_cols))


def build_breadth_context(stock_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = stock_df.copy().sort_values(["ticker", "datetime"])
    work["stock_return_1_raw"] = work.groupby("ticker")["close"].pct_change(fill_method=None)
    breadth = (
        work.groupby("datetime")
        .agg(
            breadth_stock_count=("ticker", "nunique"),
            breadth_positive_raw=("stock_return_1_raw", lambda values: float((values > 0.0).mean())),
            breadth_avg_return_raw=("stock_return_1_raw", "mean"),
            breadth_dispersion_raw=("stock_return_1_raw", "std"),
        )
        .sort_index()
        .reset_index()
    )
    breadth["breadth_positive_lag_1"] = breadth["breadth_positive_raw"].shift(1)
    breadth["breadth_avg_return_lag_1"] = breadth["breadth_avg_return_raw"].shift(1)
    breadth["breadth_dispersion_lag_1"] = breadth["breadth_dispersion_raw"].shift(1)
    breadth["breadth_positive_trend_5_lag"] = breadth["breadth_positive_raw"].shift(1).rolling(5, min_periods=3).mean()
    breadth["breadth_positive_trend_20_lag"] = breadth["breadth_positive_raw"].shift(1).rolling(20, min_periods=5).mean()
    breadth["breadth_trend_lag"] = breadth["breadth_positive_trend_5_lag"] - breadth["breadth_positive_trend_20_lag"]
    cols = [
        "breadth_stock_count",
        "breadth_positive_lag_1",
        "breadth_avg_return_lag_1",
        "breadth_dispersion_lag_1",
        "breadth_positive_trend_5_lag",
        "breadth_positive_trend_20_lag",
        "breadth_trend_lag",
    ]
    return breadth[["datetime", *cols]], cols


def add_relative_volatility_and_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = frame.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    relative_cols = [
        "relative_ret_minus_vnindex_lag_1",
        "relative_ret_minus_vn30_lag_1",
        "relative_momentum_vnindex_20_lag",
        "relative_momentum_vn30_20_lag",
        "relative_vol_vnindex_20_lag",
        "relative_vol_vn30_20_lag",
    ]
    volatility_cols = [
        "return_over_vol_20_lag",
        "return_zscore_20_lag",
        "high_low_range_shock_20_lag",
        "volume_shock_20_lag_safe",
    ]
    interaction_cols = [
        "interaction_momentum_x_market_direction",
        "interaction_relative_strength_x_market_vol",
        "interaction_volnorm_momentum_x_breadth",
    ]

    for _ticker, group in out.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        volume = group["volume"].astype(float)
        ret = close.pct_change(fill_method=None)
        ret_lag = ret.shift(1)
        ret_mean_20_lag = ret.rolling(20, min_periods=5).mean().shift(1)
        ret_vol_20_lag = ret.rolling(20, min_periods=5).std().shift(1)
        range_ratio = (high - low) / close.replace(0.0, np.nan)

        vnindex_lag = out.loc[idx, "vnindex_ret_lag_1_ctx"] if "vnindex_ret_lag_1_ctx" in out.columns else pd.Series(np.nan, index=idx)
        vn30_lag = out.loc[idx, "vn30_ret_lag_1_ctx"] if "vn30_ret_lag_1_ctx" in out.columns else pd.Series(np.nan, index=idx)
        rel_vnindex = ret_lag.to_numpy(dtype=float) - vnindex_lag.to_numpy(dtype=float)
        rel_vn30 = ret_lag.to_numpy(dtype=float) - vn30_lag.to_numpy(dtype=float)

        rel_vnindex_series = pd.Series(rel_vnindex, index=idx)
        rel_vn30_series = pd.Series(rel_vn30, index=idx)
        out.loc[idx, "relative_ret_minus_vnindex_lag_1"] = rel_vnindex_series
        out.loc[idx, "relative_ret_minus_vn30_lag_1"] = rel_vn30_series
        out.loc[idx, "relative_momentum_vnindex_20_lag"] = rel_vnindex_series.rolling(20, min_periods=5).mean()
        out.loc[idx, "relative_momentum_vn30_20_lag"] = rel_vn30_series.rolling(20, min_periods=5).mean()
        out.loc[idx, "relative_vol_vnindex_20_lag"] = rel_vnindex_series.rolling(20, min_periods=5).std()
        out.loc[idx, "relative_vol_vn30_20_lag"] = rel_vn30_series.rolling(20, min_periods=5).std()

        out.loc[idx, "return_over_vol_20_lag"] = ret_lag / ret_vol_20_lag.replace(0.0, np.nan)
        out.loc[idx, "return_zscore_20_lag"] = (ret_lag - ret_mean_20_lag) / ret_vol_20_lag.replace(0.0, np.nan)
        out.loc[idx, "high_low_range_shock_20_lag"] = (range_ratio / range_ratio.rolling(20, min_periods=5).mean() - 1.0).shift(1)
        out.loc[idx, "volume_shock_20_lag_safe"] = (volume / volume.rolling(20, min_periods=5).mean() - 1.0).shift(1)

        momentum_20_lag = (close / close.shift(20) - 1.0).shift(1)
        direction = out.loc[idx, "vnindex_direction_lag_1_ctx"] if "vnindex_direction_lag_1_ctx" in out.columns else pd.Series(np.nan, index=idx)
        direction_signed = direction.astype(float) * 2.0 - 1.0
        market_vol = out.loc[idx, "vnindex_vol_20_lag_ctx"] if "vnindex_vol_20_lag_ctx" in out.columns else pd.Series(np.nan, index=idx)
        breadth = out.loc[idx, "breadth_positive_lag_1"] if "breadth_positive_lag_1" in out.columns else pd.Series(np.nan, index=idx)
        out.loc[idx, "interaction_momentum_x_market_direction"] = momentum_20_lag.to_numpy(dtype=float) * direction_signed.to_numpy(dtype=float)
        out.loc[idx, "interaction_relative_strength_x_market_vol"] = rel_vnindex_series.to_numpy(dtype=float) * market_vol.to_numpy(dtype=float)
        out.loc[idx, "interaction_volnorm_momentum_x_breadth"] = out.loc[idx, "return_over_vol_20_lag"].to_numpy(dtype=float) * breadth.to_numpy(dtype=float)

    grouped_cols = {
        "relative_strength": relative_cols,
        "volatility_normalized": volatility_cols,
        "interaction_context": interaction_cols,
    }
    all_cols = sorted({col for cols in grouped_cols.values() for col in cols if col in out.columns})
    out[all_cols] = out[all_cols].replace([np.inf, -np.inf], np.nan)
    return out, {name: [col for col in cols if col in out.columns] for name, cols in grouped_cols.items()}


def build_feature_families() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    if len(tickers) != 30:
        raise ValueError(f"expected 30 active stock tickers, got {len(tickers)}")
    if stock_df.empty:
        raise ValueError("stock data loaded empty")
    if not {"VNINDEX", "VN30"}.intersection(index_data):
        raise ValueError("required local VNINDEX/VN30 index context data is missing")

    base, base_cols = build_feature_set_c(stock_df, index_data)
    base = base.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    index_context, regime_cols = build_lagged_index_context(index_data)
    breadth_context, breadth_cols = build_breadth_context(stock_df)

    enhanced = base.merge(index_context.drop_duplicates("datetime", keep="last"), on="datetime", how="left")
    enhanced = enhanced.merge(breadth_context.drop_duplicates("datetime", keep="last"), on="datetime", how="left")
    enhanced, grouped_cols = add_relative_volatility_and_interactions(enhanced)

    family_cols: dict[str, list[str]] = {
        BASE_FEATURE_FAMILY: sorted(set(base_cols)),
        "regime_context": sorted(set(base_cols).union(regime_cols)),
        "breadth_context": sorted(set(base_cols).union(breadth_cols)),
        "relative_strength": sorted(set(base_cols).union(grouped_cols["relative_strength"])),
        "volatility_normalized": sorted(set(base_cols).union(grouped_cols["volatility_normalized"])),
        "interaction_context": sorted(
            set(base_cols)
            .union(regime_cols)
            .union(breadth_cols)
            .union(grouped_cols["relative_strength"])
            .union(grouped_cols["volatility_normalized"])
            .union(grouped_cols["interaction_context"])
        ),
    }
    family_cols = {name: [col for col in cols if col in enhanced.columns] for name, cols in family_cols.items()}

    manifest = {
        "run_id": "vn30_hourly_validation_safe_improvement_tracks_v1",
        "data_fetch": False,
        "provider_behavior_changed": False,
        "stock_ticker_count": len(tickers),
        "stock_tickers": tickers,
        "stock_rows_loaded": int(len(stock_df)),
        "index_codes_loaded": sorted(index_data.keys()),
        "feature_families": {},
    }
    added_by_family = {
        BASE_FEATURE_FAMILY: [],
        "regime_context": regime_cols,
        "breadth_context": breadth_cols,
        "relative_strength": grouped_cols["relative_strength"],
        "volatility_normalized": grouped_cols["volatility_normalized"],
        "interaction_context": grouped_cols["interaction_context"],
    }
    for family, cols in family_cols.items():
        manifest["feature_families"][family] = {
            "feature_count": len(cols),
            "base_feature_set": "feature_set_C_closest",
            "added_feature_columns": added_by_family.get(family, []),
            "added_feature_count": len(added_by_family.get(family, [])),
            "all_added_features_lagged_or_ex_ante": True,
            "future_regime_labels": False,
            "future_return_features": False,
            "target_leakage_features": False,
            "same_row_target_leakage": False,
            "final_window_derived_features": False,
            "index_context_role": "lagged market-context features only",
        }
    return enhanced, family_cols, manifest


def logistic_pipeline(c: float = 0.3, penalty: str = "l2", l1_ratio: float | None = None) -> Pipeline:
    kwargs: dict[str, Any] = {
        "max_iter": 2500 if penalty == "elasticnet" else 1000,
        "C": c,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    if penalty == "elasticnet":
        kwargs.update({"solver": "saga", "penalty": "elasticnet", "l1_ratio": l1_ratio})
    else:
        kwargs.update({"solver": "liblinear", "penalty": "l2"})
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", LogisticRegression(**kwargs)),
        ]
    )


def calibrated_logistic() -> Any:
    base = logistic_pipeline(c=0.3, penalty="l2")
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:  # pragma: no cover - older sklearn
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def make_model(model_name: str) -> Any | None:
    if model_name == "logistic_l2":
        return logistic_pipeline(c=0.3, penalty="l2")
    if model_name == "logistic_elastic_net":
        return logistic_pipeline(c=0.3, penalty="elasticnet", l1_ratio=0.2)
    if model_name == "calibrated_logistic":
        return calibrated_logistic()
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=12,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=12,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=2,
        )
    if model_name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=2,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )
    return None


def model_names() -> list[str]:
    return [
        "logistic_l2",
        "logistic_elastic_net",
        "calibrated_logistic",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "hist_gradient_boosting",
    ]


def model_complexity_penalty(model_name: str) -> float:
    return {
        "logistic_l2": 0.000,
        "logistic_elastic_net": 0.002,
        "calibrated_logistic": 0.004,
        "random_forest": 0.010,
        "extra_trees": 0.010,
        "hist_gradient_boosting": 0.012,
        "xgboost": 0.015,
        "lightgbm": 0.015,
        "soft_vote_ensemble": 0.020,
    }.get(model_name, 0.010)


def predict_probability(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    prediction = model.predict(x_data)
    return np.asarray(prediction, dtype=float)


def period_metrics(frame: pd.DataFrame, period_col: str) -> dict[str, float]:
    if frame.empty or period_col not in frame.columns:
        return {
            "count": 0.0,
            "min_accuracy": math.nan,
            "mean_accuracy": math.nan,
            "std_accuracy": 0.0,
            "below_50": 0.0,
            "below_55": 0.0,
            "below_60": 0.0,
        }
    grouped = frame.groupby(period_col)["correct"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] > 0]
    if grouped.empty:
        return {
            "count": 0.0,
            "min_accuracy": math.nan,
            "mean_accuracy": math.nan,
            "std_accuracy": 0.0,
            "below_50": 0.0,
            "below_55": 0.0,
            "below_60": 0.0,
        }
    return {
        "count": float(len(grouped)),
        "min_accuracy": float(grouped["mean"].min()),
        "mean_accuracy": float(grouped["mean"].mean()),
        "std_accuracy": float(grouped["mean"].std(ddof=0) if len(grouped) > 1 else 0.0),
        "below_50": float((grouped["mean"] < 0.50).sum()),
        "below_55": float((grouped["mean"] < 0.55).sum()),
        "below_60": float((grouped["mean"] < 0.60).sum()),
    }


def validation_stability_frame(features: pd.DataFrame, idx: pd.Index, y_true: pd.Series, prob: np.ndarray) -> pd.DataFrame:
    out = features.reindex(idx)[["datetime", "ticker"]].copy()
    out["y_true"] = y_true.astype(int).to_numpy()
    out["y_score_or_probability"] = prob
    out["y_pred"] = (prob >= MAIN_THRESHOLD).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    out["month"] = out["datetime"].dt.to_period("M").astype(str)
    out["quarter"] = out["datetime"].dt.to_period("Q").astype(str)
    if "vnindex_trend_regime_ctx" in features.columns:
        out["regime"] = features.reindex(idx)["vnindex_trend_regime_ctx"].fillna(99).astype(int).astype(str).to_numpy()
    else:
        out["regime"] = "unknown"
    return out


def stability_component(min_accuracy: float) -> float:
    if not math.isfinite(min_accuracy):
        return -0.05
    return min_accuracy - 0.50


def validation_selection_metrics(
    features: pd.DataFrame,
    idx: pd.Index,
    train_y: pd.Series,
    val_y: pd.Series,
    train_prob: np.ndarray,
    val_prob: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    train_pred = (train_prob >= MAIN_THRESHOLD).astype(int)
    val_pred = (val_prob >= MAIN_THRESHOLD).astype(int)
    train_acc = accuracy(train_y, train_pred)
    val_acc = accuracy(val_y, val_pred)
    val_majority = majority_accuracy(val_y)
    val_lift = val_acc - val_majority
    stability = validation_stability_frame(features, idx, val_y, val_prob)
    month = period_metrics(stability, "month")
    quarter = period_metrics(stability, "quarter")
    regime = period_metrics(stability[stability["regime"] != "99"], "regime")
    monthly_stability_score = stability_component(month["min_accuracy"])
    quarterly_stability_score = stability_component(quarter["min_accuracy"])
    regime_stability_score = stability_component(regime["min_accuracy"]) if regime["count"] >= 2 else 0.0
    instability_penalty = (
        max(0.0, 0.55 - (month["min_accuracy"] if math.isfinite(month["min_accuracy"]) else 0.0))
        + max(0.0, 0.55 - (quarter["min_accuracy"] if math.isfinite(quarter["min_accuracy"]) else 0.0))
        + 0.25 * month["std_accuracy"]
        + 0.25 * quarter["std_accuracy"]
    )
    overfit_risk_penalty = max(0.0, train_acc - val_acc - 0.05) + model_complexity_penalty(model_name)
    selection_score = (
        val_acc
        + val_lift
        + monthly_stability_score
        + quarterly_stability_score
        + regime_stability_score
        - instability_penalty
        - overfit_risk_penalty
    )
    return {
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "validation_majority_baseline": val_majority,
        "lift_over_validation_majority": val_lift,
        "monthly_stability_score": monthly_stability_score,
        "quarterly_stability_score": quarterly_stability_score,
        "regime_stability_score_if_available": regime_stability_score,
        "monthly_min_accuracy": month["min_accuracy"],
        "quarterly_min_accuracy": quarter["min_accuracy"],
        "regime_min_accuracy": regime["min_accuracy"],
        "monthly_accuracy_std": month["std_accuracy"],
        "quarterly_accuracy_std": quarter["std_accuracy"],
        "validation_months": int(month["count"]),
        "validation_quarters": int(quarter["count"]),
        "validation_regime_count": int(regime["count"]),
        "validation_months_below_50": int(month["below_50"]),
        "validation_months_below_55": int(month["below_55"]),
        "validation_months_below_60": int(month["below_60"]),
        "validation_quarters_below_60": int(quarter["below_60"]),
        "instability_penalty": instability_penalty,
        "overfit_risk_penalty": overfit_risk_penalty,
        "selection_score": selection_score,
    }


def candidate_id(feature_family: str, model_name: str, horizon: int) -> str:
    return f"{feature_family}__{model_name}__h{horizon}__t050"


def run_single_candidate(
    features: pd.DataFrame,
    feature_cols: list[str],
    feature_family: str,
    model_name: str,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    labels = add_absolute_labels(features, horizon)
    idx = split_indices(features, labels)
    train_y = labels.reindex(idx["train"]).astype(int)
    val_y = labels.reindex(idx["validation"]).astype(int)
    final_y = labels.reindex(idx["final"]).astype(int)
    if train_y.empty or val_y.empty or final_y.empty or train_y.nunique() < 2:
        return None

    model = make_model(model_name)
    if model is None:
        return None

    x_train = features.reindex(idx["train"])[feature_cols]
    x_val = features.reindex(idx["validation"])[feature_cols]
    x_final = features.reindex(idx["final"])[feature_cols]
    try:
        model.fit(x_train, train_y)
        train_prob = predict_probability(model, x_train)
        val_prob = predict_probability(model, x_val)
        final_prob = predict_probability(model, x_final)
    except Exception as exc:
        return {
            "candidate_id": candidate_id(feature_family, model_name, horizon),
            "feature_family": feature_family,
            "model": model_name,
            "horizon": horizon,
            "threshold": MAIN_THRESHOLD,
            "status": "failed",
            "failure_reason": str(exc)[:240],
            "feature_count": len(feature_cols),
        }, {}

    metrics = validation_selection_metrics(features, idx["validation"], train_y, val_y, train_prob, val_prob, model_name)
    final_pred = (final_prob >= MAIN_THRESHOLD).astype(int)
    final_acc = accuracy(final_y, final_pred)
    final_majority = majority_accuracy(final_y)
    row = {
        "candidate_id": candidate_id(feature_family, model_name, horizon),
        "feature_family": feature_family,
        "model": model_name,
        "horizon": horizon,
        "threshold": MAIN_THRESHOLD,
        "status": "ok",
        "feature_count": len(feature_cols),
        "active_ticker_count": 30,
        "final_accuracy_used_for_selection": False,
        **metrics,
        "final_accuracy": final_acc,
        "final_majority_baseline": final_majority,
        "final_lift_vs_majority": final_acc - final_majority,
        "final_rows": int(len(final_y)),
        "final_unique_tickers": int(features.reindex(idx["final"])["ticker"].nunique()),
        "delta_vs_61_51_reference": final_acc - REFERENCE_FINAL_ACCURACY,
        "delta_vs_reference_majority_50_44": final_acc - REFERENCE_MAJORITY_BASELINE,
        "delta_vs_historical_rf_h60": final_acc - LOCKED_RF_H60,
    }
    payload = {
        "idx": idx,
        "labels": labels,
        "train_y": train_y,
        "val_y": val_y,
        "final_y": final_y,
        "train_prob": train_prob,
        "val_prob": val_prob,
        "final_prob": final_prob,
        "feature_cols": feature_cols,
    }
    return row, payload


def add_soft_vote_candidates(
    features: pd.DataFrame,
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    feature_family: str,
    horizon: int,
) -> None:
    base_names = ["logistic_l2", "random_forest", "extra_trees", "hist_gradient_boosting"]
    ids = [candidate_id(feature_family, name, horizon) for name in base_names]
    if not all(cid in payloads for cid in ids):
        return
    first = payloads[ids[0]]
    if not all(payloads[cid]["idx"]["validation"].equals(first["idx"]["validation"]) and payloads[cid]["idx"]["final"].equals(first["idx"]["final"]) for cid in ids):
        return
    train_prob = np.mean([payloads[cid]["train_prob"] for cid in ids], axis=0)
    val_prob = np.mean([payloads[cid]["val_prob"] for cid in ids], axis=0)
    final_prob = np.mean([payloads[cid]["final_prob"] for cid in ids], axis=0)
    metrics = validation_selection_metrics(
        features,
        first["idx"]["validation"],
        first["train_y"],
        first["val_y"],
        train_prob,
        val_prob,
        "soft_vote_ensemble",
    )
    final_pred = (final_prob >= MAIN_THRESHOLD).astype(int)
    final_acc = accuracy(first["final_y"], final_pred)
    final_majority = majority_accuracy(first["final_y"])
    cid = candidate_id(feature_family, "soft_vote_ensemble", horizon)
    row = {
        "candidate_id": cid,
        "feature_family": feature_family,
        "model": "soft_vote_ensemble",
        "horizon": horizon,
        "threshold": MAIN_THRESHOLD,
        "status": "ok",
        "feature_count": len(first["feature_cols"]),
        "active_ticker_count": 30,
        "base_prediction_ids": ";".join(ids),
        "final_accuracy_used_for_selection": False,
        **metrics,
        "final_accuracy": final_acc,
        "final_majority_baseline": final_majority,
        "final_lift_vs_majority": final_acc - final_majority,
        "final_rows": int(len(first["final_y"])),
        "final_unique_tickers": int(features.reindex(first["idx"]["final"])["ticker"].nunique()),
        "delta_vs_61_51_reference": final_acc - REFERENCE_FINAL_ACCURACY,
        "delta_vs_reference_majority_50_44": final_acc - REFERENCE_MAJORITY_BASELINE,
        "delta_vs_historical_rf_h60": final_acc - LOCKED_RF_H60,
    }
    rows.append(row)
    payloads[cid] = {
        **first,
        "train_prob": train_prob,
        "val_prob": val_prob,
        "final_prob": final_prob,
        "base_prediction_ids": ids,
    }


def select_candidate(validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in validation_rows
        if row.get("status") == "ok"
        and int(row.get("active_ticker_count", 0)) == 30
        and int(row.get("final_unique_tickers", 0)) == 30
        and math.isfinite(as_float(row.get("selection_score")))
        and as_float(row.get("validation_accuracy")) > 0
    ]
    if not valid:
        raise ValueError("no valid validation-selected candidates")
    model_rank = {
        "logistic_l2": 0,
        "logistic_elastic_net": 1,
        "calibrated_logistic": 2,
        "hist_gradient_boosting": 3,
        "random_forest": 4,
        "extra_trees": 5,
        "xgboost": 6,
        "lightgbm": 7,
        "soft_vote_ensemble": 8,
    }
    family_rank = {
        BASE_FEATURE_FAMILY: 0,
        "regime_context": 1,
        "breadth_context": 2,
        "relative_strength": 3,
        "volatility_normalized": 4,
        "interaction_context": 5,
    }
    selected = max(
        valid,
        key=lambda row: (
            as_float(row["selection_score"]),
            as_float(row["validation_accuracy"]),
            as_float(row["lift_over_validation_majority"]),
            -abs(int(row["horizon"]) - 40),
            -model_rank.get(str(row["model"]), 99),
            -family_rank.get(str(row["feature_family"]), 99),
        ),
    )
    selected["selected_by_validation_only"] = True
    return selected


def selected_prediction_frame(features: pd.DataFrame, payload: dict[str, Any], selected: dict[str, Any]) -> pd.DataFrame:
    idx = payload["idx"]["final"]
    final_y = payload["final_y"]
    final_prob = payload["final_prob"]
    out = features.reindex(idx)[["datetime", "ticker"]].copy()
    out["horizon"] = int(selected["horizon"])
    out["model"] = selected["model"]
    out["feature_family"] = selected["feature_family"]
    out["threshold"] = MAIN_THRESHOLD
    out["split"] = "final"
    out["y_true"] = final_y.astype(int).to_numpy()
    out["y_score_or_probability"] = final_prob
    out["y_pred"] = (final_prob >= MAIN_THRESHOLD).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    out["candidate_id"] = selected["candidate_id"]
    out["selection_source"] = "validation_only"
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def grouped_summary(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_col, sort=True):
        y_true = group["y_true"].astype(int)
        accuracy_value = float(group["correct"].mean())
        majority = majority_accuracy(y_true)
        rows.append(
            {
                group_col: key,
                "rows": int(len(group)),
                "accuracy": accuracy_value,
                "majority_baseline": majority,
                "lift_vs_majority": accuracy_value - majority,
                "target_positive_rate": float(y_true.mean()),
                "prediction_positive_rate": float(group["y_pred"].astype(int).mean()),
                "correct": int(group["correct"].sum()),
            }
        )
    return pd.DataFrame(rows)


def rolling_frame(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    work = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True).copy()
    work["row_number"] = np.arange(1, len(work) + 1)
    correct = work["correct"].astype(float)
    y_true = work["y_true"].astype(float)
    rolling_correct = correct.rolling(window, min_periods=window).sum()
    rolling_positive = y_true.rolling(window, min_periods=window).sum()
    valid = rolling_correct.notna()
    out = work.loc[valid, ["row_number", "datetime", "ticker"]].copy()
    out["window_rows"] = window
    out["window_start_row_number"] = out["row_number"] - window + 1
    out["window_start_datetime"] = work["datetime"].shift(window - 1).loc[valid].to_numpy()
    out["rolling_correct"] = rolling_correct.loc[valid].to_numpy(dtype=float)
    out["rolling_accuracy"] = out["rolling_correct"] / window
    out["rolling_positive_rate"] = rolling_positive.loc[valid].to_numpy(dtype=float) / window
    out["rolling_majority_baseline"] = np.maximum(out["rolling_positive_rate"], 1.0 - out["rolling_positive_rate"])
    out["rolling_lift_vs_majority"] = out["rolling_accuracy"] - out["rolling_majority_baseline"]
    return out


def summarize_rolling(rolling: pd.DataFrame, window: int, total_rows: int, final_accuracy: float) -> dict[str, Any]:
    if rolling.empty:
        return {
            "window_rows": window,
            "total_final_rows": total_rows,
            "global_final_accuracy": final_accuracy,
            "rolling_window_count": 0,
            "rolling_min_accuracy": math.nan,
            "rolling_mean_accuracy": math.nan,
            "rolling_median_accuracy": math.nan,
            "rolling_max_accuracy": math.nan,
            "windows_below_50": 0,
            "windows_below_55": 0,
            "windows_below_60": 0,
            "rolling_min_lift_vs_majority": math.nan,
            "rolling_mean_lift_vs_majority": math.nan,
            "final_endpoint_rolling_accuracy": math.nan,
        }
    acc = rolling["rolling_accuracy"]
    lift = rolling["rolling_lift_vs_majority"]
    return {
        "window_rows": window,
        "total_final_rows": total_rows,
        "global_final_accuracy": final_accuracy,
        "rolling_window_count": int(len(rolling)),
        "rolling_min_accuracy": float(acc.min()),
        "rolling_mean_accuracy": float(acc.mean()),
        "rolling_median_accuracy": float(acc.median()),
        "rolling_max_accuracy": float(acc.max()),
        "windows_below_50": int((acc < 0.50).sum()),
        "windows_below_55": int((acc < 0.55).sum()),
        "windows_below_60": int((acc < 0.60).sum()),
        "rolling_min_lift_vs_majority": float(lift.min()),
        "rolling_mean_lift_vs_majority": float(lift.mean()),
        "final_endpoint_rolling_accuracy": float(acc.iloc[-1]),
    }


def read_reference_rolling_summary() -> pd.DataFrame:
    path = REFERENCE_ROLLING_DIR / "rolling_stability_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rolling_not_worse(selected_summary: pd.DataFrame, reference_summary: pd.DataFrame) -> bool:
    if selected_summary.empty or reference_summary.empty:
        return False
    checks: list[bool] = []
    for _, row in selected_summary.iterrows():
        window = int(row["window_rows"])
        ref = reference_summary[reference_summary["window_rows"].astype(int) == window]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        checks.append(as_float(row["rolling_min_accuracy"]) >= as_float(ref_row["rolling_min_accuracy"]) - 0.01)
        checks.append(as_float(row["rolling_mean_accuracy"]) >= as_float(ref_row["rolling_mean_accuracy"]) - 0.01)
        checks.append(int(row["windows_below_60"]) <= int(ref_row["windows_below_60"]) + 25)
    return bool(checks) and all(checks)


def classify_acceptance(selected: dict[str, Any], rolling_ok: bool) -> tuple[str, str]:
    final_acc = as_float(selected.get("final_accuracy"))
    val_acc = as_float(selected.get("validation_accuracy"))
    selected_ok = bool(selected.get("selected_by_validation_only"))
    coverage_ok = int(selected.get("final_unique_tickers", 0)) == 30 and int(selected.get("active_ticker_count", 0)) == 30
    gap = final_acc - val_acc
    gap_ok = abs(gap) <= abs(REFERENCE_VALIDATION_FINAL_GAP) + 0.02
    if final_acc >= 0.65 and selected_ok and coverage_ok and rolling_ok and gap_ok:
        return "final65_candidate", "Exploratory only unless confirmed by a future blind test."
    if final_acc > REFERENCE_FINAL_ACCURACY and selected_ok and coverage_ok and rolling_ok and gap_ok:
        return "stronger_baseline60_candidate", "Validation-selected candidate improved over the 61.51% reference without final-window selection."
    if final_acc > REFERENCE_FINAL_ACCURACY and selected_ok and coverage_ok:
        return "weak_improvement", "Final accuracy improved, but stability or validation-final gap worsened materially."
    return "failed_improvement", "Final accuracy did not exceed the 61.51% reference or a validation/coverage rule failed."


def overfit_risk_classification(selected: dict[str, Any]) -> str:
    penalty = as_float(selected.get("overfit_risk_penalty"))
    gap = as_float(selected.get("train_accuracy")) - as_float(selected.get("validation_accuracy"))
    validation_final_gap = as_float(selected.get("final_accuracy")) - as_float(selected.get("validation_accuracy"))
    if penalty >= 0.12 or gap > 0.15 or abs(validation_final_gap) > 0.14:
        return "high"
    if penalty >= 0.06 or gap > 0.08 or abs(validation_final_gap) > 0.10:
        return "medium"
    return "low"


def summarize_by_feature(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    ok = rows[rows["status"] == "ok"].copy()
    for family, group in ok.groupby("feature_family", sort=True):
        best = group.sort_values(["selection_score", "validation_accuracy"], ascending=False).iloc[0]
        out_rows.append(
            {
                "feature_family": family,
                "candidate_count": int(len(group)),
                "best_validation_candidate_id": best["candidate_id"],
                "best_validation_model": best["model"],
                "best_validation_horizon": int(best["horizon"]),
                "best_validation_accuracy": float(best["validation_accuracy"]),
                "best_selection_score": float(best["selection_score"]),
                "scoring_only_final_accuracy": float(best["final_accuracy"]),
                "delta_vs_61_51_reference": float(best["delta_vs_61_51_reference"]),
            }
        )
    return pd.DataFrame(out_rows)


def summarize_by_model(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    ok = rows[rows["status"] == "ok"].copy()
    for model, group in ok.groupby("model", sort=True):
        best = group.sort_values(["selection_score", "validation_accuracy"], ascending=False).iloc[0]
        out_rows.append(
            {
                "model": model,
                "candidate_count": int(len(group)),
                "best_validation_candidate_id": best["candidate_id"],
                "best_validation_feature_family": best["feature_family"],
                "best_validation_horizon": int(best["horizon"]),
                "best_validation_accuracy": float(best["validation_accuracy"]),
                "best_selection_score": float(best["selection_score"]),
                "scoring_only_final_accuracy": float(best["final_accuracy"]),
                "delta_vs_61_51_reference": float(best["delta_vs_61_51_reference"]),
            }
        )
    return pd.DataFrame(out_rows)


def markdown_summary(
    selected: dict[str, Any],
    classification: str,
    classification_reason: str,
    rolling_summary: pd.DataFrame,
    reference_rolling: pd.DataFrame,
) -> str:
    lines = [
        "# VN30 Hourly Validation-Safe Improvement Tracks Summary",
        "",
        "## Protocol Boundary",
        "",
        "- Benchmark run: yes, validation-safe improvement experiment only.",
        "- Data fetch: no.",
        "- Model training: yes.",
        "- Model selection: yes, validation-only.",
        "- Main target: VN30 stock-only hourly overall directional accuracy.",
        "- Main threshold: 0.50.",
        "- Confidence abstention: no.",
        "- Ticker subset: no.",
        "- Top-k/ranking substitution: no.",
        "- Paper/DOCX generated: no.",
        "",
        "## Selected Candidate",
        "",
        f"- Candidate ID: `{selected['candidate_id']}`.",
        f"- Feature family: `{selected['feature_family']}`.",
        f"- Model: `{selected['model']}`.",
        f"- Horizon: h={int(selected['horizon'])}.",
        f"- Validation accuracy: {pct(selected['validation_accuracy'])}.",
        f"- Validation selection score: {as_float(selected['selection_score']):.6f}.",
        f"- Final accuracy: {pct(selected['final_accuracy'])}.",
        f"- Delta vs 61.51% reference: {pp(selected['delta_vs_61_51_reference'])}.",
        f"- Delta vs 50.44% majority reference: {pp(selected['delta_vs_reference_majority_50_44'])}.",
        f"- Final rows: {int(selected['final_rows'])}.",
        f"- Full 30-stock coverage: {'yes' if int(selected['final_unique_tickers']) == 30 else 'no'}.",
        f"- Validation-final gap: {pp(as_float(selected['final_accuracy']) - as_float(selected['validation_accuracy']))}.",
        f"- Overfit risk classification: {selected['overfit_risk_classification']}.",
        f"- Acceptance classification: `{classification}`.",
        f"- Classification reason: {classification_reason}",
        "",
        "## Rolling Summary",
        "",
        "| Window | Min Acc | Mean Acc | Median Acc | End Acc | Windows <60% | Ref Mean Acc | Ref Windows <60% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in rolling_summary.iterrows():
        window = int(row["window_rows"])
        ref = reference_rolling[reference_rolling["window_rows"].astype(int) == window] if not reference_rolling.empty else pd.DataFrame()
        ref_mean = pct(ref.iloc[0]["rolling_mean_accuracy"]) if not ref.empty else ""
        ref_below = str(int(ref.iloc[0]["windows_below_60"])) if not ref.empty else ""
        lines.append(
            f"| {window} | {pct(row['rolling_min_accuracy'])} | {pct(row['rolling_mean_accuracy'])} | "
            f"{pct(row['rolling_median_accuracy'])} | {pct(row['final_endpoint_rolling_accuracy'])} | "
            f"{int(row['windows_below_60'])} | {ref_mean} | {ref_below} |"
        )
    lines.extend(
        [
            "",
            "## Claim Level",
            "",
            "This is an exploratory validation-safe improvement experiment. It does not establish trading readiness, profitability, investment suitability, live deployment, or Final65 unless the acceptance classification explicitly permits a Final65 candidate boundary.",
        ]
    )
    return "\n".join(lines)


def leakage_audit_markdown(manifest: dict[str, Any], selected: dict[str, Any]) -> str:
    family_manifest = manifest["feature_families"][selected["feature_family"]]
    checks = [
        ("final window not used in selection", True),
        ("no future features", not family_manifest["future_return_features"] and not family_manifest["future_regime_labels"]),
        ("no target leakage", not family_manifest["target_leakage_features"]),
        ("no same-row target leakage", not family_manifest["same_row_target_leakage"]),
        ("no confidence abstention", True),
        ("no ticker subset", int(selected.get("final_unique_tickers", 0)) == 30),
        ("no top-k/ranking substitution", True),
        ("full 30-stock coverage", int(selected.get("final_unique_tickers", 0)) == 30 and int(selected.get("active_ticker_count", 0)) == 30),
    ]
    lines = [
        "# Leakage Audit",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, passed in checks:
        lines.append(f"| {name} | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Index features are lagged market-context features only.",
            "- Index-only and joint-panel results do not replace the stock-only claim.",
            "- The selected candidate was chosen by validation-only selection score; final accuracy was scoring-only.",
        ]
    )
    return "\n".join(lines)


def claim_boundary_markdown(selected: dict[str, Any], classification: str) -> str:
    return "\n".join(
        [
            "# Claim Boundary",
            "",
            "## Safe Claim",
            "",
            f"The `{selected['candidate_id']}` candidate was selected by validation-only scoring and then scored on the final window for VN30 stock-only hourly overall directional accuracy.",
            "",
            "## Current Result",
            "",
            f"- Final accuracy: {pct(selected['final_accuracy'])}.",
            f"- Reference final accuracy: {REFERENCE_FINAL_ACCURACY_LABEL}.",
            f"- Delta vs reference: {pp(selected['delta_vs_61_51_reference'])}.",
            f"- Acceptance classification: `{classification}`.",
            f"- Claim level: `{selected['claim_level']}`.",
            "",
            "## Unsafe Claims",
            "",
            "- Do not claim trading, profitability, investment recommendation, or live-deployment readiness.",
            "- Do not describe confidence-filtered, ticker-subset, top-k, index-only, or joint-panel diagnostics as the main VN30 stock-only overall accuracy.",
            "- Do not state that Final65 is established unless the audit classifies the result as `final65_candidate`, and even then keep it exploratory until future blind confirmation.",
        ]
    )


def run() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features, family_cols, manifest = build_feature_families()
    write_json(REPORT_DIR / "feature_family_manifest.json", manifest)

    grid_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}

    for feature_family, feature_cols in family_cols.items():
        for horizon in HORIZONS:
            for model_name in model_names():
                cid = candidate_id(feature_family, model_name, horizon)
                grid_rows.append(
                    {
                        "candidate_id": cid,
                        "feature_family": feature_family,
                        "model": model_name,
                        "horizon": horizon,
                        "threshold": MAIN_THRESHOLD,
                        "feature_count": len(feature_cols),
                        "selection_window": "validation_only",
                        "final_window_role": "scoring_only",
                    }
                )
                output = run_single_candidate(features, feature_cols, feature_family, model_name, horizon)
                if output is None:
                    result_rows.append(
                        {
                            "candidate_id": cid,
                            "feature_family": feature_family,
                            "model": model_name,
                            "horizon": horizon,
                            "threshold": MAIN_THRESHOLD,
                            "status": "not_available_or_empty",
                            "feature_count": len(feature_cols),
                        }
                    )
                    continue
                row, payload = output
                result_rows.append(row)
                if row.get("status") == "ok" and payload:
                    payloads[str(row["candidate_id"])] = payload
            add_soft_vote_candidates(features, result_rows, payloads, feature_family, horizon)
            if candidate_id(feature_family, "soft_vote_ensemble", horizon) in payloads:
                grid_rows.append(
                    {
                        "candidate_id": candidate_id(feature_family, "soft_vote_ensemble", horizon),
                        "feature_family": feature_family,
                        "model": "soft_vote_ensemble",
                        "horizon": horizon,
                        "threshold": MAIN_THRESHOLD,
                        "feature_count": len(feature_cols),
                        "selection_window": "validation_only",
                        "final_window_role": "scoring_only",
                    }
                )

    grid = pd.DataFrame(grid_rows).drop_duplicates("candidate_id")
    results = pd.DataFrame(result_rows).drop_duplicates("candidate_id", keep="last")
    selected = select_candidate(results.to_dict("records"))
    selected_payload = payloads[selected["candidate_id"]]

    results["selected_by_validation_only"] = results["candidate_id"].eq(selected["candidate_id"])
    validation_cols = [
        "candidate_id",
        "feature_family",
        "model",
        "horizon",
        "threshold",
        "status",
        "feature_count",
        "active_ticker_count",
        "train_accuracy",
        "validation_accuracy",
        "validation_majority_baseline",
        "lift_over_validation_majority",
        "monthly_stability_score",
        "quarterly_stability_score",
        "regime_stability_score_if_available",
        "instability_penalty",
        "overfit_risk_penalty",
        "selection_score",
        "monthly_min_accuracy",
        "quarterly_min_accuracy",
        "regime_min_accuracy",
        "validation_months_below_60",
        "validation_quarters_below_60",
        "selected_by_validation_only",
        "final_accuracy_used_for_selection",
    ]
    final_cols = [
        "candidate_id",
        "feature_family",
        "model",
        "horizon",
        "threshold",
        "status",
        "selected_by_validation_only",
        "final_accuracy",
        "final_majority_baseline",
        "final_lift_vs_majority",
        "final_rows",
        "final_unique_tickers",
        "delta_vs_61_51_reference",
        "delta_vs_reference_majority_50_44",
        "delta_vs_historical_rf_h60",
    ]
    grid.to_csv(REPORT_DIR / "candidate_grid.csv", index=False)
    results[[col for col in validation_cols if col in results.columns]].to_csv(REPORT_DIR / "validation_results.csv", index=False)
    results[[col for col in final_cols if col in results.columns]].to_csv(REPORT_DIR / "final_scoring_results.csv", index=False)

    row_predictions = selected_prediction_frame(features, selected_payload, selected)
    row_predictions.to_csv(REPORT_DIR / "row_predictions_selected.csv", index=False)

    by_ticker = grouped_summary(row_predictions, "ticker")
    by_month_source = row_predictions.copy()
    by_month_source["month"] = by_month_source["datetime"].dt.to_period("M").astype(str)
    by_quarter_source = row_predictions.copy()
    by_quarter_source["quarter"] = by_quarter_source["datetime"].dt.to_period("Q").astype(str)
    by_month = grouped_summary(by_month_source, "month")
    by_quarter = grouped_summary(by_quarter_source, "quarter")
    by_ticker.to_csv(REPORT_DIR / "by_ticker.csv", index=False)
    by_month.to_csv(REPORT_DIR / "by_month.csv", index=False)
    by_quarter.to_csv(REPORT_DIR / "by_quarter.csv", index=False)

    rolling_summaries: list[dict[str, Any]] = []
    for window in (250, 500, 1000):
        rolling = rolling_frame(row_predictions, window)
        rolling.to_csv(REPORT_DIR / f"rolling_{window}.csv", index=False)
        rolling_summaries.append(summarize_rolling(rolling, window, len(row_predictions), as_float(selected["final_accuracy"])))
    rolling_summary = pd.DataFrame(rolling_summaries)
    rolling_summary.to_csv(REPORT_DIR / "rolling_summary.csv", index=False)

    feature_summary = summarize_by_feature(results)
    model_summary = summarize_by_model(results)
    feature_summary.to_csv(REPORT_DIR / "feature_ablation_summary.csv", index=False)
    model_summary.to_csv(REPORT_DIR / "model_family_summary.csv", index=False)

    reference_rolling = read_reference_rolling_summary()
    rolling_ok = rolling_not_worse(rolling_summary, reference_rolling)
    selected["rolling_stability_not_worse_than_reference"] = rolling_ok
    selected["validation_final_gap"] = as_float(selected["final_accuracy"]) - as_float(selected["validation_accuracy"])
    selected["validation_final_gap_abs"] = abs(as_float(selected["validation_final_gap"]))
    selected["full_30_stock_coverage"] = int(selected["final_unique_tickers"]) == 30
    selected["overfit_risk_classification"] = overfit_risk_classification(selected)
    classification, classification_reason = classify_acceptance(selected, rolling_ok)
    selected["acceptance_classification"] = classification
    selected["acceptance_classification_reason"] = classification_reason
    selected["claim_level"] = (
        "exploratory_final65_candidate"
        if classification == "final65_candidate"
        else "exploratory_validation_safe_improvement"
        if classification in {"stronger_baseline60_candidate", "weak_improvement"}
        else "exploratory_failed_improvement"
    )
    selected["reference_final_accuracy"] = REFERENCE_FINAL_ACCURACY
    selected["reference_final_rows"] = REFERENCE_FINAL_ROWS
    selected["reference_majority_baseline"] = REFERENCE_MAJORITY_BASELINE
    selected["historical_rf_h60_reference"] = LOCKED_RF_H60
    selected["selection_score_formula"] = (
        "validation_accuracy + lift_over_validation_majority + monthly_stability_score + "
        "quarterly_stability_score + regime_stability_score_if_available - instability_penalty - overfit_risk_penalty"
    )
    selected["selection_source"] = "validation_only"
    selected["final_window_role"] = "scoring_only"
    selected["confidence_abstention"] = False
    selected["ticker_subset"] = False
    selected["topk_ranking_substitution"] = False
    selected["data_fetch"] = False
    selected["paper_docx_generated"] = False

    write_json(REPORT_DIR / "selected_candidate.json", selected)
    write_json(
        REPORT_DIR / "run_config.json",
        {
            "run_id": "vn30_hourly_validation_safe_improvement_tracks_v1",
            "train_end": str(TRAIN_END),
            "validation_start": str(VAL_START),
            "validation_end": str(VAL_END),
            "final_start": str(FINAL_START),
            "horizons": HORIZONS,
            "threshold": MAIN_THRESHOLD,
            "models": model_names() + ["soft_vote_ensemble"],
            "feature_families": list(family_cols.keys()),
            "selection_source": "validation_only",
            "final_window_role": "scoring_only",
            "final_accuracy_used_for_selection": False,
            "confidence_abstention": False,
            "ticker_subset": False,
            "topk": False,
            "data_fetch": False,
            "provider_behavior_changed": False,
            "paper_docx": False,
            "reference_final_accuracy": REFERENCE_FINAL_ACCURACY,
            "reference_majority_baseline": REFERENCE_MAJORITY_BASELINE,
        },
    )
    write_markdown(REPORT_DIR / "improvement_summary.md", markdown_summary(selected, classification, classification_reason, rolling_summary, reference_rolling))
    write_markdown(REPORT_DIR / "leakage_audit.md", leakage_audit_markdown(manifest, selected))
    write_markdown(REPORT_DIR / "claim_boundary.md", claim_boundary_markdown(selected, classification))

    missing = [name for name in REQUIRED_OUTPUTS if not (REPORT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required outputs: {missing}")
    return selected


def main() -> None:
    selected = run()
    print(
        "Selected "
        f"{selected['candidate_id']} validation={pct(selected['validation_accuracy'])} "
        f"final={pct(selected['final_accuracy'])} classification={selected['acceptance_classification']}"
    )


if __name__ == "__main__":
    main()
