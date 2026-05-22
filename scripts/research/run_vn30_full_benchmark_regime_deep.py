"""Run full VN30 hourly directional benchmark with regimes and deep models.

The benchmark uses existing local artifacts only. Selection is validation-only;
the final window is scoring-only for selected candidates.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    FINAL_START,
    REFERENCE_FINAL_ACCURACY,
    REFERENCE_FINAL_ROWS,
    REFERENCE_MAJORITY_BASELINE,
    REFERENCE_VALIDATION_FINAL_GAP,
    TRAIN_END,
    VAL_END,
    VAL_START,
    build_feature_families,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    active_stock_tickers,
    load_index_data,
    rel,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_full_benchmark_regime_deep"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
HORIZONS = [20, 40, 60, 80]
DEEP_HORIZONS = [20, 40, 60]
SEQUENCE_LENGTHS = [16, 32, 64]
THRESHOLD_GRID = [0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]
FEATURE_FAMILIES = [
    "baseline_C_closest",
    "regime_context",
    "breadth_context",
    "relative_strength",
    "volatility_normalized",
    "interaction_context",
    "combined_context",
]
CLASSICAL_MODELS = [
    "logistic_l2",
    "logistic_elastic_net",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "hist_gradient_boosting",
]
BASELINE_MODELS = [
    "majority_class",
    "random_walk_direction",
    "previous_direction",
    "moving_average_rule",
    "rolling_momentum_rule",
    "volatility_adjusted_momentum_rule",
]
DEEP_MODELS = ["lstm", "gru", "tcn"]


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
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def pct(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:+.2f} pp"


def accuracy(y_true: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(prediction, dtype=int)).mean())


def majority_accuracy(y_true: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    rate = float(np.asarray(y_true, dtype=int).mean())
    return max(rate, 1.0 - rate)


def majority_value(y_true: pd.Series | np.ndarray) -> int:
    if len(y_true) == 0:
        return 1
    return int(float(np.asarray(y_true, dtype=int).mean()) >= 0.5)


def label_frame(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _ticker, group in features.groupby("ticker", sort=True):
        ordered = group.sort_values("datetime")
        future_close = ordered["close"].shift(-horizon)
        future_dt = ordered["datetime"].shift(-horizon)
        y = (future_close > ordered["close"]).astype(float)
        y.loc[future_close.isna()] = np.nan
        pieces.append(pd.DataFrame({"y": y, "future_datetime": future_dt}, index=ordered.index))
    if not pieces:
        return pd.DataFrame(columns=["y", "future_datetime"])
    return pd.concat(pieces).sort_index()


def strict_split_indices(features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, pd.Index]:
    valid = labels["y"].notna() & labels["future_datetime"].notna()
    train_mask = (
        features["datetime"].le(TRAIN_END)
        & labels["future_datetime"].le(TRAIN_END)
        & valid
    )
    val_mask = (
        features["datetime"].between(VAL_START, VAL_END)
        & labels["future_datetime"].between(VAL_START, VAL_END)
        & valid
    )
    final_mask = features["datetime"].ge(FINAL_START) & valid
    return {
        "train": features.index[train_mask],
        "validation": features.index[val_mask],
        "final": features.index[final_mask],
    }


def compute_regime_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    index_data = load_index_data()
    code = "VNINDEX" if "VNINDEX" in index_data else ("VN30" if "VN30" in index_data else "")
    if not code:
        raise ValueError("VNINDEX/VN30 index context is required for regime features")
    frame = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    ret = frame["close"].pct_change(fill_method=None)
    trend_60 = (frame["close"] / frame["close"].shift(60) - 1.0).shift(1)
    vol_20 = ret.rolling(20, min_periods=5).std().shift(1)
    train_vol = vol_20.loc[frame["datetime"].le(TRAIN_END)].dropna()
    vol_threshold = float(train_vol.median()) if not train_vol.empty else float(vol_20.dropna().median())
    direction = np.select([trend_60 > 0.02, trend_60 < -0.02], ["bull", "bear"], default="sideway")
    volatility = np.where(vol_20 > vol_threshold, "high_volatility", "low_volatility")
    regime = pd.DataFrame(
        {
            "datetime": frame["datetime"],
            "market_direction_regime": direction,
            "volatility_regime": volatility,
            "market_direction_regime_code_lagged": np.select([direction == "bull", direction == "bear"], [1.0, -1.0], default=0.0),
            "volatility_regime_code_lagged": np.where(volatility == "high_volatility", 1.0, 0.0),
            "regime_source_code": code,
            "regime_trend_60_lag": trend_60,
            "regime_vol_20_lag": vol_20,
        }
    )
    manifest = {
        "regime_source_index": code,
        "market_direction_rule": "lagged 60-bar index return: bull > 2%, bear < -2%, otherwise sideway",
        "volatility_rule": "lagged 20-bar index return volatility above train-window median is high_volatility",
        "volatility_threshold_train_median": vol_threshold,
        "uses_future_returns": False,
        "uses_final_window_for_threshold": False,
        "feature_construction": "lagged rolling index returns and volatility only",
    }
    return regime, manifest


def load_benchmark_features() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any], pd.DataFrame]:
    features, family_cols, manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    regime, regime_manifest = compute_regime_features()
    features = features.merge(
        regime[
            [
                "datetime",
                "market_direction_regime",
                "volatility_regime",
                "market_direction_regime_code_lagged",
                "volatility_regime_code_lagged",
                "regime_trend_60_lag",
                "regime_vol_20_lag",
            ]
        ].drop_duplicates("datetime", keep="last"),
        on="datetime",
        how="left",
    )
    features["market_direction_regime"] = features["market_direction_regime"].fillna("sideway")
    features["volatility_regime"] = features["volatility_regime"].fillna("low_volatility")
    regime_code_cols = ["market_direction_regime_code_lagged", "volatility_regime_code_lagged", "regime_trend_60_lag", "regime_vol_20_lag"]
    for col in regime_code_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    family_cols = {name: [col for col in cols if col in features.columns] for name, cols in family_cols.items()}
    family_cols.setdefault("regime_context", [])
    family_cols["regime_context"] = sorted(set(family_cols["regime_context"]).union(regime_code_cols))
    combined = sorted(
        {
            col
            for name, cols in family_cols.items()
            if name != "combined_context"
            for col in cols
            if col in features.columns and pd.api.types.is_numeric_dtype(features[col])
        }
    )
    family_cols["combined_context"] = combined
    family_cols = {
        name: [col for col in cols if col in features.columns and pd.api.types.is_numeric_dtype(features[col])]
        for name, cols in family_cols.items()
        if name in FEATURE_FAMILIES or name == "combined_context"
    }
    manifest = dict(manifest)
    manifest["run_id"] = "vn30_full_benchmark_regime_deep_v1"
    manifest["data_fetch"] = False
    manifest["provider_behavior_changed"] = False
    manifest["strict_label_split"] = True
    manifest["feature_families"] = dict(manifest.get("feature_families", {}))
    for name in FEATURE_FAMILIES:
        manifest["feature_families"].setdefault(name, {})
        manifest["feature_families"][name].update(
            {
                "feature_count": len(family_cols.get(name, [])),
                "all_added_features_lagged_or_ex_ante": True,
                "future_regime_labels": False,
                "future_return_features": False,
                "target_leakage_features": False,
                "same_row_target_leakage": False,
                "final_window_derived_features": False,
                "index_context_role": "lagged market-context features only",
            }
        )
    manifest["feature_families"]["combined_context"]["base_feature_set"] = "union_of_requested_context_families"
    manifest["regime_features"] = regime_manifest
    return features, family_cols, manifest, regime


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.DataFrame,
    score: np.ndarray,
    threshold: float,
    *,
    method_group: str,
    model: str,
    feature_family: str,
    horizon: int,
    threshold_policy: str,
    candidate_id: str,
    split: str,
) -> pd.DataFrame:
    out = features.loc[idx, ["datetime", "ticker", "market_direction_regime", "volatility_regime"]].copy()
    out["future_datetime"] = labels.loc[idx, "future_datetime"].to_numpy()
    out["method_group"] = method_group
    out["model"] = model
    out["feature_family"] = feature_family
    out["horizon"] = int(horizon)
    out["threshold_policy"] = threshold_policy
    out["candidate_id"] = candidate_id
    out["split"] = split
    out["y_true"] = labels.loc[idx, "y"].astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(score, dtype=float)
    out["threshold"] = float(threshold)
    out["y_pred"] = (out["y_score_or_probability"].to_numpy(dtype=float) >= float(threshold)).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def stability_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "final_accuracy": math.nan,
            "final_rows": 0,
            "ticker_coverage": 0,
            "majority_baseline": math.nan,
            "lift_vs_majority": math.nan,
            "monthly_stability": math.nan,
            "quarterly_stability": math.nan,
            "regime_stability": math.nan,
            "rolling_250_mean": math.nan,
            "rolling_500_mean": math.nan,
            "rolling_1000_mean": math.nan,
            "rolling_250_min": math.nan,
            "rolling_500_min": math.nan,
            "rolling_1000_min": math.nan,
        }
    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    final_accuracy = float(work["correct"].mean())
    majority = majority_accuracy(work["y_true"].to_numpy(dtype=int))

    def grouped_min(key: str) -> float:
        grouped = work.groupby(key)["correct"].mean()
        return float(grouped.min()) if not grouped.empty else math.nan

    work["month"] = work["datetime"].dt.to_period("M").astype(str)
    work["quarter"] = work["datetime"].dt.to_period("Q").astype(str)
    work["regime_key"] = work["market_direction_regime"].astype(str) + "/" + work["volatility_regime"].astype(str)
    out = {
        "final_accuracy": final_accuracy,
        "final_rows": int(len(work)),
        "ticker_coverage": int(work["ticker"].nunique()),
        "majority_baseline": majority,
        "lift_vs_majority": final_accuracy - majority,
        "monthly_stability": grouped_min("month"),
        "quarterly_stability": grouped_min("quarter"),
        "regime_stability": grouped_min("regime_key"),
    }
    ordered = work.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    for window in (250, 500, 1000):
        rolling = ordered["correct"].astype(float).rolling(window=window, min_periods=window).mean().dropna()
        out[f"rolling_{window}_mean"] = float(rolling.mean()) if not rolling.empty else math.nan
        out[f"rolling_{window}_min"] = float(rolling.min()) if not rolling.empty else math.nan
        out[f"rolling_{window}_end"] = float(rolling.iloc[-1]) if not rolling.empty else math.nan
        out[f"rolling_{window}_windows"] = int(len(rolling))
    return out


def period_slice(frame: pd.DataFrame, candidate_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work["month"] = work["datetime"].dt.to_period("M").astype(str)
    work["quarter"] = work["datetime"].dt.to_period("Q").astype(str)
    work["regime_key"] = work["market_direction_regime"].astype(str) + "/" + work["volatility_regime"].astype(str)

    def agg(key: str, label: str) -> pd.DataFrame:
        if work.empty:
            return pd.DataFrame(columns=["candidate_id", label, "accuracy", "rows"])
        out = work.groupby(key)["correct"].agg(["mean", "count"]).reset_index()
        out.insert(0, "candidate_id", candidate_id)
        out = out.rename(columns={key: label, "mean": "accuracy", "count": "rows"})
        return out

    return agg("ticker", "ticker"), agg("month", "month"), agg("quarter", "quarter"), agg("regime_key", "regime")


def select_threshold(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_accuracy = -1.0
    y = np.asarray(y_true, dtype=int)
    for threshold in THRESHOLD_GRID:
        pred = (np.asarray(probability, dtype=float) >= threshold).astype(int)
        acc = accuracy(y, pred)
        if acc > best_accuracy + 1e-12 or (abs(acc - best_accuracy) <= 1e-12 and abs(threshold - 0.50) < abs(best_threshold - 0.50)):
            best_accuracy = acc
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def make_classical_model(model_name: str) -> Any | None:
    if model_name == "logistic_l2":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1200, solver="liblinear", C=0.3, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
    if model_name == "logistic_elastic_net":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2500,
                        solver="saga",
                        penalty="elasticnet",
                        C=0.3,
                        l1_ratio=0.2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=90,
                        max_depth=7,
                        min_samples_leaf=12,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=90,
                        max_depth=7,
                        min_samples_leaf=12,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "xgboost" and XGBClassifier is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=80,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_weight=10,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        verbosity=0,
                        n_jobs=2,
                    ),
                ),
            ]
        )
    if model_name == "lightgbm" and LGBMClassifier is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=80,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_samples=40,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                        n_jobs=2,
                    ),
                ),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=80,
                        max_leaf_nodes=15,
                        learning_rate=0.04,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    return None


def predict_probability(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    prediction = model.predict(x_data)
    return np.asarray(prediction, dtype=float)


def run_baselines(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        labels = label_frame(features, horizon)
        idx = strict_split_indices(features, labels)
        train_y = labels.loc[idx["train"], "y"].astype(int)
        majority_pred = majority_value(train_y)
        close = features["close"].astype(float)
        by_ticker = features.groupby("ticker", sort=True)
        random_walk = (close > by_ticker["close"].shift(1)).astype(float)
        previous_direction = labels["y"].groupby(features["ticker"]).shift(horizon)
        ma20 = by_ticker["close"].transform(lambda values: values.rolling(20, min_periods=5).mean())
        moving_average = (close > ma20).astype(float)
        momentum20 = close / by_ticker["close"].shift(20) - 1.0
        rolling_momentum = (momentum20 > 0.0).astype(float)
        ret1 = by_ticker["close"].pct_change(fill_method=None)
        vol20 = ret1.groupby(features["ticker"]).transform(lambda values: values.rolling(20, min_periods=5).std())
        vol_adjusted = ((momentum20 / vol20.replace(0.0, np.nan)) > 0.0).astype(float)
        raw_predictions = {
            "majority_class": pd.Series(float(majority_pred), index=features.index),
            "random_walk_direction": random_walk,
            "previous_direction": previous_direction,
            "moving_average_rule": moving_average,
            "rolling_momentum_rule": rolling_momentum,
            "volatility_adjusted_momentum_rule": vol_adjusted,
        }
        for model_name, raw_pred in raw_predictions.items():
            pred_series = pd.to_numeric(raw_pred, errors="coerce").reindex(features.index).fillna(float(majority_pred)).astype(float)
            val_frame = prediction_frame(
                features,
                idx["validation"],
                labels,
                pred_series.loc[idx["validation"]].to_numpy(dtype=float),
                0.50,
                method_group="baseline",
                model=model_name,
                feature_family="ex_ante_rule",
                horizon=horizon,
                threshold_policy="rule",
                candidate_id=f"baseline__{model_name}__h{horizon}",
                split="validation",
            )
            final_frame = prediction_frame(
                features,
                idx["final"],
                labels,
                pred_series.loc[idx["final"]].to_numpy(dtype=float),
                0.50,
                method_group="baseline",
                model=model_name,
                feature_family="ex_ante_rule",
                horizon=horizon,
                threshold_policy="rule",
                candidate_id=f"baseline__{model_name}__h{horizon}",
                split="final",
            )
            final_stats = stability_summary(final_frame)
            result_rows.append(
                {
                    "method_group": "baseline",
                    "model": model_name,
                    "feature_family": "ex_ante_rule",
                    "horizon": horizon,
                    "threshold_policy": "rule",
                    "candidate_id": f"baseline__{model_name}__h{horizon}",
                    "selection_source": "rule_no_model_selection",
                    "validation_accuracy": float(val_frame["correct"].mean()) if not val_frame.empty else math.nan,
                    "validation_rows": int(len(val_frame)),
                    "validation_ticker_coverage": int(val_frame["ticker"].nunique()) if not val_frame.empty else 0,
                    **final_stats,
                    "validation_final_gap": final_stats["final_accuracy"] - float(val_frame["correct"].mean()) if not val_frame.empty else math.nan,
                    "full_ticker_coverage": int(final_stats["ticker_coverage"]) == 30,
                    "leakage_status": "passed_ex_ante_rule",
                }
            )
            prediction_frames.append(final_frame)
    baseline_results = pd.DataFrame(result_rows)
    baseline_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    write_csv(REPORT_DIR / "baseline_results.csv", baseline_results)
    write_csv(REPORT_DIR / "baseline_row_predictions.csv", baseline_predictions)
    return baseline_results, baseline_predictions


def add_selection(selected: dict[str, set[str]], candidate_id: str, reason: str) -> None:
    selected.setdefault(candidate_id, set()).add(reason)


def run_classical_ml(features: pd.DataFrame, family_cols: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for feature_family in FEATURE_FAMILIES:
        cols = [col for col in family_cols.get(feature_family, []) if col in features.columns and pd.api.types.is_numeric_dtype(features[col])]
        if not cols:
            continue
        for horizon in HORIZONS:
            labels = label_frame(features, horizon)
            idx = strict_split_indices(features, labels)
            train_y = labels.loc[idx["train"], "y"].astype(int)
            val_y = labels.loc[idx["validation"], "y"].astype(int)
            if train_y.empty or val_y.empty or train_y.nunique() < 2:
                continue
            x_train = features.loc[idx["train"], cols]
            x_val = features.loc[idx["validation"], cols]
            for model_name in CLASSICAL_MODELS:
                candidate_base = f"classical__{feature_family}__{model_name}__h{horizon}"
                model = make_classical_model(model_name)
                if model is None:
                    validation_rows.append(
                        {
                            "candidate_id": f"{candidate_base}__skipped",
                            "method_group": "classical_ml",
                            "model": model_name,
                            "feature_family": feature_family,
                            "horizon": horizon,
                            "threshold_policy": "not_run",
                            "status": "skipped_optional_dependency_missing",
                        }
                    )
                    continue
                try:
                    model.fit(x_train, train_y)
                    val_prob = predict_probability(model, x_val)
                    train_prob = predict_probability(model, x_train)
                except Exception as exc:
                    validation_rows.append(
                        {
                            "candidate_id": f"{candidate_base}__failed",
                            "method_group": "classical_ml",
                            "model": model_name,
                            "feature_family": feature_family,
                            "horizon": horizon,
                            "threshold_policy": "not_run",
                            "status": "failed",
                            "error": str(exc)[:300],
                        }
                    )
                    continue
                train_acc = accuracy(train_y, (train_prob >= 0.50).astype(int))
                threshold_specs = [("fixed_0.50", 0.50), ("validation_selected_threshold", select_threshold(val_y, val_prob)[0])]
                for threshold_policy, threshold in threshold_specs:
                    val_pred = (val_prob >= threshold).astype(int)
                    val_acc = accuracy(val_y, val_pred)
                    candidate_id = f"{candidate_base}__{threshold_policy}__t{threshold:.3f}".replace(".", "p")
                    validation_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "method_group": "classical_ml",
                            "model": model_name,
                            "feature_family": feature_family,
                            "horizon": horizon,
                            "threshold_policy": threshold_policy,
                            "threshold": float(threshold),
                            "status": "ok",
                            "selection_source": "validation_only",
                            "final_window_role": "not_scored_until_selected",
                            "final_accuracy_used_for_selection": False,
                            "feature_count": len(cols),
                            "train_rows": int(len(train_y)),
                            "validation_rows": int(len(val_y)),
                            "validation_ticker_coverage": int(features.loc[idx["validation"], "ticker"].nunique()),
                            "validation_accuracy": val_acc,
                            "validation_majority_baseline": majority_accuracy(val_y),
                            "validation_lift_vs_majority": val_acc - majority_accuracy(val_y),
                            "train_accuracy": train_acc,
                        }
                    )
                    payloads[candidate_id] = {
                        "model": model,
                        "feature_cols": cols,
                        "labels": labels,
                        "idx": idx,
                        "val_prob": val_prob,
                        "threshold": float(threshold),
                        "threshold_policy": threshold_policy,
                        "feature_family": feature_family,
                        "model_name": model_name,
                        "horizon": horizon,
                    }
    results = pd.DataFrame(validation_rows)
    ok = results[results["status"].eq("ok")].copy() if not results.empty and "status" in results.columns else pd.DataFrame()
    selected: dict[str, set[str]] = {}
    if not ok.empty:
        for policy, group in ok.groupby("threshold_policy", sort=True):
            row = group.sort_values(["validation_accuracy", "validation_rows"], ascending=False).iloc[0]
            add_selection(selected, str(row["candidate_id"]), f"best_overall_{policy}")
        for family, group in ok.groupby("feature_family", sort=True):
            row = group.sort_values(["validation_accuracy", "validation_rows"], ascending=False).iloc[0]
            add_selection(selected, str(row["candidate_id"]), f"best_feature_family_{family}")
        for model_name, group in ok.groupby("model", sort=True):
            row = group.sort_values(["validation_accuracy", "validation_rows"], ascending=False).iloc[0]
            add_selection(selected, str(row["candidate_id"]), f"best_model_{model_name}")

    selected_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    final_updates: dict[str, dict[str, Any]] = {}
    for candidate_id, reasons in sorted(selected.items()):
        payload = payloads[candidate_id]
        labels = payload["labels"]
        idx = payload["idx"]
        threshold = float(payload["threshold"])
        cols = payload["feature_cols"]
        model = payload["model"]
        final_idx = idx["final"]
        val_idx = idx["validation"]
        final_y = labels.loc[final_idx, "y"].astype(int)
        x_final = features.loc[final_idx, cols]
        final_prob = predict_probability(model, x_final)
        val_frame = prediction_frame(
            features,
            val_idx,
            labels,
            payload["val_prob"],
            threshold,
            method_group="classical_ml",
            model=payload["model_name"],
            feature_family=payload["feature_family"],
            horizon=payload["horizon"],
            threshold_policy=payload["threshold_policy"],
            candidate_id=candidate_id,
            split="validation",
        )
        final_frame = prediction_frame(
            features,
            final_idx,
            labels,
            final_prob,
            threshold,
            method_group="classical_ml",
            model=payload["model_name"],
            feature_family=payload["feature_family"],
            horizon=payload["horizon"],
            threshold_policy=payload["threshold_policy"],
            candidate_id=candidate_id,
            split="final",
        )
        final_stats = stability_summary(final_frame)
        val_acc = float(val_frame["correct"].mean()) if not val_frame.empty else math.nan
        update = {
            **final_stats,
            "validation_accuracy": val_acc,
            "validation_final_gap": final_stats["final_accuracy"] - val_acc,
            "selected_by_validation": True,
            "selection_reasons": ";".join(sorted(reasons)),
            "full_ticker_coverage": int(final_stats["ticker_coverage"]) == 30,
            "leakage_status": "passed_validation_only",
        }
        final_updates[candidate_id] = update
        base_row = ok[ok["candidate_id"].eq(candidate_id)].iloc[0].to_dict()
        base_row.update(update)
        selected_rows.append(base_row)
        prediction_frames.append(final_frame)
        if len(final_y) != len(final_frame):
            raise ValueError("classical final prediction row mismatch")
    selected_df = pd.DataFrame(selected_rows)
    if not results.empty:
        for candidate_id, update in final_updates.items():
            mask = results["candidate_id"].eq(candidate_id)
            for key, value in update.items():
                results.loc[mask, key] = value
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    write_csv(REPORT_DIR / "classical_ml_results.csv", results)
    write_csv(REPORT_DIR / "classical_ml_selected_candidates.csv", selected_df)
    write_csv(REPORT_DIR / "classical_ml_row_predictions.csv", predictions)
    return results, selected_df, predictions


if torch is not None:

    class RecurrentDirectionModel(nn.Module):
        def __init__(self, input_dim: int, model_type: str) -> None:
            super().__init__()
            hidden = 16
            if model_type == "lstm":
                self.core = nn.LSTM(input_dim, hidden, batch_first=True)
            else:
                self.core = nn.GRU(input_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _out, state = self.core(x)
            hidden = state[0] if isinstance(state, tuple) else state
            return self.head(hidden[-1]).squeeze(-1)


    class TcnDirectionModel(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, 16, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv1d(16, 16, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(16, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.net(x.transpose(1, 2)).squeeze(-1)
            return self.head(encoded).squeeze(-1)


def make_deep_model(model_name: str, input_dim: int) -> Any:
    if torch is None or nn is None:
        raise RuntimeError("torch unavailable")
    if model_name in {"lstm", "gru"}:
        return RecurrentDirectionModel(input_dim, model_name)
    if model_name == "tcn":
        return TcnDirectionModel(input_dim)
    raise ValueError(f"unknown deep model: {model_name}")


def select_deep_feature_cols(features: pd.DataFrame, family_cols: dict[str, list[str]]) -> list[str]:
    candidates = [col for col in family_cols.get("baseline_C_closest", []) if col in features.columns and pd.api.types.is_numeric_dtype(features[col])]
    train_mask = features["datetime"].le(TRAIN_END)
    scored: list[tuple[str, float, int]] = []
    for col in candidates:
        series = pd.to_numeric(features.loc[train_mask, col], errors="coerce")
        scored.append((col, float(series.var(skipna=True) or 0.0), int(series.notna().sum())))
    scored = [item for item in scored if item[2] > 100 and math.isfinite(item[1])]
    selected = [col for col, _var, _count in sorted(scored, key=lambda item: (item[2], item[1]), reverse=True)[:32]]
    return selected or candidates[:32]


def standardize_feature_matrix(features: pd.DataFrame, cols: list[str], train_idx: pd.Index) -> pd.DataFrame:
    train_values = features.loc[train_idx, cols].apply(pd.to_numeric, errors="coerce")
    med = train_values.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scale = train_values.std(axis=0).replace([np.inf, -np.inf, 0.0], np.nan).fillna(1.0)
    out = features[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    out = (out - med) / scale
    return out.astype("float32")


def build_sequences(
    features: pd.DataFrame,
    matrix: pd.DataFrame,
    labels: pd.DataFrame,
    idx: pd.Index,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    idx_set = set(int(i) for i in idx)
    seqs: list[np.ndarray] = []
    ys: list[int] = []
    row_ids: list[int] = []
    times: list[pd.Timestamp] = []
    tickers: list[str] = []
    for ticker, group in features.groupby("ticker", sort=True):
        ordered = group.sort_values("datetime")
        values = matrix.loc[ordered.index].to_numpy(dtype=np.float32)
        ordered_indices = list(ordered.index)
        ordered_times = list(ordered["datetime"])
        for pos, row_id in enumerate(ordered_indices):
            if int(row_id) not in idx_set or pos + 1 < seq_len:
                continue
            y = labels.at[row_id, "y"]
            if pd.isna(y):
                continue
            seqs.append(values[pos - seq_len + 1 : pos + 1])
            ys.append(int(y))
            row_ids.append(int(row_id))
            times.append(pd.Timestamp(ordered_times[pos]))
            tickers.append(str(ticker))
    if not seqs:
        return np.empty((0, seq_len, matrix.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32), pd.Index([])
    order = np.lexsort((np.asarray(tickers, dtype=object), np.asarray(times, dtype="datetime64[ns]")))
    x = np.stack(seqs).astype(np.float32)[order]
    y = np.asarray(ys, dtype=np.float32)[order]
    row_index = pd.Index(np.asarray(row_ids, dtype=int)[order])
    return x, y, row_index


def predict_deep(model: Any, x: np.ndarray) -> np.ndarray:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("torch unavailable")
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=1024, shuffle=False)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs) if probs else np.asarray([], dtype=float)


def fit_deep_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[Any, dict[str, Any], np.ndarray]:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("torch unavailable")
    torch.manual_seed(RANDOM_STATE)
    model = make_deep_model(model_name, int(x_train.shape[2]))
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives > 0 else torch.tensor([1.0], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=512,
        shuffle=False,
    )
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = math.inf
    best_epoch = 0
    patience_used = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(yb)
            total_rows += len(yb)
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(x_val))
            val_loss = float(criterion(val_logits, torch.from_numpy(y_val.astype(np.float32))).detach())
            val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_acc = accuracy(y_val.astype(int), (val_prob >= 0.50).astype(int))
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_rows, 1), "validation_loss": val_loss, "validation_accuracy": val_acc})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            patience_used = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            patience_used += 1
            if patience_used >= 1:
                break
    model.load_state_dict(best_state)
    val_prob = predict_deep(model, x_val)
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history}, val_prob


def run_deep_learning(features: pd.DataFrame, family_cols: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    skip_lines: list[str] = ["# Deep Learning Skip Report", ""]
    if torch is None:
        skip_lines.append("- Status: skipped_with_reason.")
        skip_lines.append("- Reason: torch is not importable in the intended Python environment.")
        empty = pd.DataFrame()
        write_csv(REPORT_DIR / "deep_learning_results.csv", empty)
        write_csv(REPORT_DIR / "deep_learning_selected_candidates.csv", empty)
        write_csv(REPORT_DIR / "deep_learning_row_predictions.csv", empty)
        report = "\n".join(skip_lines)
        write_markdown(REPORT_DIR / "deep_learning_skip_report.md", report)
        return empty, empty, empty, report

    feature_cols = select_deep_feature_cols(features, family_cols)
    if not feature_cols:
        skip_lines.append("- Status: skipped_with_reason.")
        skip_lines.append("- Reason: no numeric feature columns available for sequence models.")
        empty = pd.DataFrame()
        write_csv(REPORT_DIR / "deep_learning_results.csv", empty)
        write_csv(REPORT_DIR / "deep_learning_selected_candidates.csv", empty)
        write_csv(REPORT_DIR / "deep_learning_row_predictions.csv", empty)
        report = "\n".join(skip_lines)
        write_markdown(REPORT_DIR / "deep_learning_skip_report.md", report)
        return empty, empty, empty, report

    result_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for horizon in DEEP_HORIZONS:
        labels = label_frame(features, horizon)
        idx = strict_split_indices(features, labels)
        if len(idx["train"]) == 0 or len(idx["validation"]) == 0 or len(idx["final"]) == 0:
            continue
        matrix = standardize_feature_matrix(features, feature_cols, idx["train"])
        for seq_len in SEQUENCE_LENGTHS:
            x_train, y_train, train_rows = build_sequences(features, matrix, labels, idx["train"], seq_len)
            x_val, y_val, val_rows = build_sequences(features, matrix, labels, idx["validation"], seq_len)
            if len(y_train) == 0 or len(y_val) == 0 or len(np.unique(y_train)) < 2:
                continue
            for model_name in DEEP_MODELS:
                candidate_id = f"deep_learning__{model_name}__baseline_C_32__h{horizon}__seq{seq_len}__fixed_0p50"
                try:
                    model, train_meta, val_prob = fit_deep_model(model_name, x_train, y_train, x_val, y_val)
                    val_pred = (val_prob >= 0.50).astype(int)
                    val_acc = accuracy(y_val.astype(int), val_pred)
                    result_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "method_group": "deep_learning",
                            "model": model_name,
                            "feature_family": "baseline_C_closest_sequence32",
                            "horizon": horizon,
                            "threshold_policy": "fixed_0.50",
                            "sequence_length": seq_len,
                            "threshold": 0.50,
                            "status": "ok",
                            "selection_source": "validation_only",
                            "final_window_role": "not_scored_until_selected",
                            "feature_count": len(feature_cols),
                            "train_rows": int(len(y_train)),
                            "validation_rows": int(len(y_val)),
                            "validation_ticker_coverage": int(features.loc[val_rows, "ticker"].nunique()),
                            "validation_accuracy": val_acc,
                            "validation_majority_baseline": majority_accuracy(y_val.astype(int)),
                            "validation_lift_vs_majority": val_acc - majority_accuracy(y_val.astype(int)),
                            "best_epoch": int(train_meta["best_epoch"]),
                            "early_stopping_source": "validation_loss_only",
                            "shuffle_across_time": False,
                        }
                    )
                    payloads[candidate_id] = {
                        "model": model,
                        "labels": labels,
                        "idx": idx,
                        "matrix": matrix,
                        "seq_len": seq_len,
                        "horizon": horizon,
                        "model_name": model_name,
                        "val_prob": val_prob,
                        "val_rows": val_rows,
                        "feature_cols": feature_cols,
                        "train_meta": train_meta,
                    }
                except Exception as exc:
                    result_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "method_group": "deep_learning",
                            "model": model_name,
                            "feature_family": "baseline_C_closest_sequence32",
                            "horizon": horizon,
                            "threshold_policy": "fixed_0.50",
                            "sequence_length": seq_len,
                            "status": "skipped_with_reason",
                            "skip_reason": str(exc)[:300],
                        }
                    )
                gc.collect()
    results = pd.DataFrame(result_rows)
    ok = results[results["status"].eq("ok")].copy() if not results.empty and "status" in results.columns else pd.DataFrame()
    selected: dict[str, set[str]] = {}
    if not ok.empty:
        row = ok.sort_values(["validation_accuracy", "validation_rows"], ascending=False).iloc[0]
        add_selection(selected, str(row["candidate_id"]), "best_overall_deep_validation")
        for model_name, group in ok.groupby("model", sort=True):
            row = group.sort_values(["validation_accuracy", "validation_rows"], ascending=False).iloc[0]
            add_selection(selected, str(row["candidate_id"]), f"best_deep_model_{model_name}")
    selected_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    final_updates: dict[str, dict[str, Any]] = {}
    for candidate_id, reasons in sorted(selected.items()):
        payload = payloads[candidate_id]
        labels = payload["labels"]
        idx = payload["idx"]
        x_final, y_final, final_rows = build_sequences(features, payload["matrix"], labels, idx["final"], int(payload["seq_len"]))
        final_prob = predict_deep(payload["model"], x_final)
        val_frame = prediction_frame(
            features,
            payload["val_rows"],
            labels,
            payload["val_prob"],
            0.50,
            method_group="deep_learning",
            model=payload["model_name"],
            feature_family="baseline_C_closest_sequence32",
            horizon=int(payload["horizon"]),
            threshold_policy="fixed_0.50",
            candidate_id=candidate_id,
            split="validation",
        )
        final_frame = prediction_frame(
            features,
            final_rows,
            labels,
            final_prob,
            0.50,
            method_group="deep_learning",
            model=payload["model_name"],
            feature_family="baseline_C_closest_sequence32",
            horizon=int(payload["horizon"]),
            threshold_policy="fixed_0.50",
            candidate_id=candidate_id,
            split="final",
        )
        final_stats = stability_summary(final_frame)
        val_acc = float(val_frame["correct"].mean()) if not val_frame.empty else math.nan
        update = {
            **final_stats,
            "validation_accuracy": val_acc,
            "validation_final_gap": final_stats["final_accuracy"] - val_acc,
            "selected_by_validation": True,
            "selection_reasons": ";".join(sorted(reasons)),
            "full_ticker_coverage": int(final_stats["ticker_coverage"]) == 30,
            "leakage_status": "passed_validation_only_sequence_safe",
        }
        final_updates[candidate_id] = update
        base_row = ok[ok["candidate_id"].eq(candidate_id)].iloc[0].to_dict()
        base_row.update(update)
        selected_rows.append(base_row)
        prediction_frames.append(final_frame)
        del x_final, y_final
        gc.collect()
    selected_df = pd.DataFrame(selected_rows)
    if not results.empty:
        for candidate_id, update in final_updates.items():
            mask = results["candidate_id"].eq(candidate_id)
            for key, value in update.items():
                results.loc[mask, key] = value
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    skip_lines.append("- Status: run.")
    skip_lines.append("- Reason: torch importable and sequence data shape was valid.")
    skip_lines.append(f"- Feature columns: {len(feature_cols)} train-only selected baseline columns.")
    skip_lines.append("- Sequence lengths evaluated: 16, 32, 64.")
    skip_lines.append("- Horizons evaluated: h20, h40, h60.")
    report = "\n".join(skip_lines)
    write_csv(REPORT_DIR / "deep_learning_results.csv", results)
    write_csv(REPORT_DIR / "deep_learning_selected_candidates.csv", selected_df)
    write_csv(REPORT_DIR / "deep_learning_row_predictions.csv", predictions)
    write_markdown(REPORT_DIR / "deep_learning_skip_report.md", report)
    return results, selected_df, predictions, report


def baseline_lookup(baseline_results: pd.DataFrame) -> dict[tuple[int, str], float]:
    lookup: dict[tuple[int, str], float] = {}
    if baseline_results.empty:
        return lookup
    for _, row in baseline_results.iterrows():
        lookup[(int(row["horizon"]), str(row["model"]))] = as_float(row.get("final_accuracy"))
    return lookup


def overfit_risk(validation_accuracy: float, final_accuracy: float, rolling_250_mean: float) -> str:
    if not math.isfinite(validation_accuracy) or not math.isfinite(final_accuracy):
        return "unknown"
    val_minus_final = validation_accuracy - final_accuracy
    if val_minus_final > 0.05 or (math.isfinite(rolling_250_mean) and rolling_250_mean < 0.52):
        return "high"
    if val_minus_final > 0.02 or (math.isfinite(rolling_250_mean) and rolling_250_mean < 0.56):
        return "moderate"
    return "low"


def classify_claim(row: dict[str, Any], audit_passed: bool = True) -> str:
    if not audit_passed or not bool(row.get("full_ticker_coverage", False)):
        return "rejected"
    final_acc = as_float(row.get("final_accuracy"))
    val_final_gap = as_float(row.get("validation_final_gap"))
    rolling_250 = as_float(row.get("rolling_250_mean"))
    if not math.isfinite(final_acc) or final_acc <= REFERENCE_FINAL_ACCURACY:
        return "failed_to_beat_reference"
    if final_acc >= 0.65:
        return "final65_candidate_exploratory"
    if (
        final_acc > REFERENCE_FINAL_ACCURACY
        and math.isfinite(val_final_gap)
        and val_final_gap >= REFERENCE_VALIDATION_FINAL_GAP - 0.02
        and math.isfinite(rolling_250)
        and rolling_250 >= 0.56
    ):
        return "stronger_full_coverage_candidate"
    return "exploratory_accuracy_gain"


def build_unified_outputs(
    baseline_results: pd.DataFrame,
    classical_selected: pd.DataFrame,
    deep_selected: pd.DataFrame,
    all_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pieces = [baseline_results.copy()]
    if not classical_selected.empty:
        pieces.append(classical_selected.copy())
    if not deep_selected.empty:
        pieces.append(deep_selected.copy())
    leaderboard = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    lookup = baseline_lookup(baseline_results)
    rows: list[dict[str, Any]] = []
    for _, raw in leaderboard.iterrows():
        row = raw.to_dict()
        horizon = int(as_float(row.get("horizon"))) if math.isfinite(as_float(row.get("horizon"))) else 0
        validation_accuracy = as_float(row.get("validation_accuracy"))
        final_accuracy = as_float(row.get("final_accuracy"))
        validation_final_gap = as_float(row.get("validation_final_gap"))
        if not math.isfinite(validation_final_gap) and math.isfinite(validation_accuracy) and math.isfinite(final_accuracy):
            validation_final_gap = final_accuracy - validation_accuracy
        full_coverage = int(as_float(row.get("ticker_coverage"))) == 30 if math.isfinite(as_float(row.get("ticker_coverage"))) else bool(row.get("full_ticker_coverage", False))
        out = {
            "method_group": row.get("method_group", ""),
            "model": row.get("model", ""),
            "feature_family": row.get("feature_family", ""),
            "horizon": horizon,
            "threshold_policy": row.get("threshold_policy", ""),
            "candidate_id": row.get("candidate_id", ""),
            "validation_accuracy": validation_accuracy,
            "final_accuracy": final_accuracy,
            "validation_final_gap": validation_final_gap,
            "final_rows": int(as_float(row.get("final_rows"))) if math.isfinite(as_float(row.get("final_rows"))) else 0,
            "ticker_coverage": int(as_float(row.get("ticker_coverage"))) if math.isfinite(as_float(row.get("ticker_coverage"))) else 0,
            "majority_baseline": as_float(row.get("majority_baseline")),
            "lift_vs_majority": as_float(row.get("lift_vs_majority")),
            "previous_direction_baseline": lookup.get((horizon, "previous_direction"), math.nan),
            "moving_average_baseline": lookup.get((horizon, "moving_average_rule"), math.nan),
            "rolling_250_mean": as_float(row.get("rolling_250_mean")),
            "rolling_500_mean": as_float(row.get("rolling_500_mean")),
            "rolling_1000_mean": as_float(row.get("rolling_1000_mean")),
            "monthly_stability": as_float(row.get("monthly_stability")),
            "quarterly_stability": as_float(row.get("quarterly_stability")),
            "regime_stability": as_float(row.get("regime_stability")),
            "leakage_status": row.get("leakage_status", "passed_validation_only"),
            "full_ticker_coverage": full_coverage,
            "overfit_risk_classification": overfit_risk(validation_accuracy, final_accuracy, as_float(row.get("rolling_250_mean"))),
            "selection_source": row.get("selection_source", ""),
            "selection_reasons": row.get("selection_reasons", ""),
        }
        out["claim_level"] = classify_claim(out, audit_passed=str(out["leakage_status"]).startswith("passed"))
        rows.append(out)
    unified = pd.DataFrame(rows)
    if unified.empty:
        best = {}
    else:
        selected_candidates = unified[
            unified["method_group"].isin(["classical_ml", "deep_learning"])
            & unified["validation_accuracy"].notna()
            & unified["final_accuracy"].notna()
            & unified["full_ticker_coverage"].eq(True)
        ].copy()
        if selected_candidates.empty:
            best = {}
        else:
            best = selected_candidates.sort_values(["validation_accuracy", "final_accuracy"], ascending=False).iloc[0].to_dict()
    best_by_group = (
        unified.sort_values(["method_group", "validation_accuracy", "final_accuracy"], ascending=[True, False, False])
        .groupby("method_group", as_index=False)
        .head(1)
        if not unified.empty
        else pd.DataFrame()
    )
    write_csv(REPORT_DIR / "unified_leaderboard.csv", unified)
    write_csv(REPORT_DIR / "best_by_group.csv", best_by_group)
    write_json(REPORT_DIR / "best_overall_validation_selected.json", best)
    build_comparison_summary(unified, best)
    return unified, best


def build_comparison_summary(unified: pd.DataFrame, best: dict[str, Any]) -> None:
    if unified.empty:
        write_markdown(REPORT_DIR / "comparison_summary.md", "# VN30 Full Benchmark Comparison Summary\n\nNo results were produced.")
        return
    best_baseline = unified[unified["method_group"].eq("baseline")].sort_values("final_accuracy", ascending=False).head(1)
    best_classical = unified[unified["method_group"].eq("classical_ml")].sort_values("validation_accuracy", ascending=False).head(1)
    best_deep = unified[unified["method_group"].eq("deep_learning")].sort_values("validation_accuracy", ascending=False).head(1)
    lines = [
        "# VN30 Full Benchmark Comparison Summary",
        "",
        "## Best Overall Validation-Selected Candidate",
        "",
        f"- Candidate: `{best.get('candidate_id', '')}`.",
        f"- Method group: {best.get('method_group', '')}.",
        f"- Model: {best.get('model', '')}.",
        f"- Feature family: {best.get('feature_family', '')}.",
        f"- Horizon: h{best.get('horizon', '')}.",
        f"- Threshold policy: {best.get('threshold_policy', '')}.",
        f"- Validation accuracy: {pct(best.get('validation_accuracy'))}.",
        f"- Final accuracy: {pct(best.get('final_accuracy'))}.",
        f"- Delta vs 61.51% reference: {pp(as_float(best.get('final_accuracy')) - REFERENCE_FINAL_ACCURACY)}.",
        f"- Final rows: {best.get('final_rows', '')}.",
        f"- Full ticker coverage: {'yes' if best.get('full_ticker_coverage') else 'no'}.",
        f"- Claim level: {best.get('claim_level', '')}.",
        "",
        "## Best By Group",
        "",
    ]
    for label, frame in [("Baseline", best_baseline), ("Classical ML", best_classical), ("Deep Learning", best_deep)]:
        if frame.empty:
            lines.append(f"- {label}: not available.")
            continue
        row = frame.iloc[0].to_dict()
        lines.append(
            f"- {label}: `{row.get('candidate_id', '')}` validation {pct(row.get('validation_accuracy'))}, "
            f"final {pct(row.get('final_accuracy'))}, rows {int(as_float(row.get('final_rows')))}."
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The benchmark uses existing local data only and does not fetch market data.",
            "- Candidate selection is validation-only; final rows are scoring-only.",
            "- No trading, profitability, investment recommendation, or live-deployment claim is made.",
            "- Any Final65-level result remains exploratory until future blind validation confirms it.",
        ]
    )
    write_markdown(REPORT_DIR / "comparison_summary.md", "\n".join(lines))


def run_regime_outputs(regime: pd.DataFrame, all_predictions: pd.DataFrame, regime_manifest: dict[str, Any]) -> pd.DataFrame:
    write_json(REPORT_DIR / "regime_feature_manifest.json", regime_manifest)
    distribution = (
        all_predictions[all_predictions["split"].eq("final")]
        .groupby(["candidate_id", "market_direction_regime", "volatility_regime"])
        .size()
        .reset_index(name="rows")
        if not all_predictions.empty
        else pd.DataFrame(columns=["candidate_id", "market_direction_regime", "volatility_regime", "rows"])
    )
    write_csv(REPORT_DIR / "regime_distribution.csv", distribution)
    if all_predictions.empty:
        slices = pd.DataFrame()
    else:
        final = all_predictions[all_predictions["split"].eq("final")].copy()
        final["regime_key"] = final["market_direction_regime"].astype(str) + "/" + final["volatility_regime"].astype(str)
        slices = final.groupby(["candidate_id", "method_group", "model", "feature_family", "horizon", "regime_key"])["correct"].agg(["mean", "count"]).reset_index()
        slices = slices.rename(columns={"mean": "accuracy", "count": "rows"})
    write_csv(REPORT_DIR / "regime_slice_results.csv", slices)
    return slices


def make_model_for_walk_forward(model_name: str) -> Any:
    model = make_classical_model(model_name)
    if model is None:
        raise ValueError(f"walk-forward model is unavailable: {model_name}")
    return model


def run_walk_forward(features: pd.DataFrame, family_cols: dict[str, list[str]], classical_selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if classical_selected.empty:
        empty = pd.DataFrame()
        write_json(
            REPORT_DIR / "walk_forward_config.json",
            {"status": "skipped", "reason": "no selected classical candidate available", "data_fetch": False},
        )
        write_csv(REPORT_DIR / "walk_forward_validation_results.csv", empty)
        write_csv(REPORT_DIR / "walk_forward_final_results.csv", empty)
        return empty, empty
    fixed = classical_selected[classical_selected["threshold_policy"].eq("fixed_0.50")]
    source = fixed if not fixed.empty else classical_selected
    selected = source.sort_values(["validation_accuracy", "final_accuracy"], ascending=False).iloc[0].to_dict()
    model_name = str(selected["model"])
    feature_family = str(selected["feature_family"])
    horizon = int(as_float(selected["horizon"]))
    threshold = float(as_float(selected.get("threshold", 0.50)))
    cols = [col for col in family_cols.get(feature_family, []) if col in features.columns and pd.api.types.is_numeric_dtype(features[col])]
    labels = label_frame(features, horizon)
    validation_windows = [
        ("2024Q1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31 23:59:59")),
        ("2024Q2", pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30 23:59:59")),
        ("2024Q3", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30 23:59:59")),
        ("2024Q4", pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31 23:59:59")),
    ]
    final_windows = [
        ("2025H1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30 23:59:59")),
        ("2025H2", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")),
        ("2026YTD", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31 23:59:59")),
    ]

    def eval_window(window_name: str, start: pd.Timestamp, end: pd.Timestamp, split_name: str) -> dict[str, Any] | None:
        train_mask = features["datetime"].lt(start) & labels["future_datetime"].lt(start) & labels["y"].notna()
        test_mask = features["datetime"].between(start, end) & labels["future_datetime"].le(end) & labels["y"].notna()
        train_idx = features.index[train_mask]
        test_idx = features.index[test_mask]
        if len(train_idx) == 0 or len(test_idx) == 0:
            return None
        y_train = labels.loc[train_idx, "y"].astype(int)
        y_test = labels.loc[test_idx, "y"].astype(int)
        if y_train.nunique() < 2:
            return None
        model = make_model_for_walk_forward(model_name)
        model.fit(features.loc[train_idx, cols], y_train)
        prob = predict_probability(model, features.loc[test_idx, cols])
        pred = (prob >= threshold).astype(int)
        acc = accuracy(y_test, pred)
        return {
            "candidate_id": selected.get("candidate_id", ""),
            "window": window_name,
            "split": split_name,
            "train_start": str(features.loc[train_idx, "datetime"].min()),
            "train_end": str(features.loc[train_idx, "datetime"].max()),
            "test_start": str(start),
            "test_end": str(end),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "ticker_coverage": int(features.loc[test_idx, "ticker"].nunique()),
            "accuracy": acc,
            "majority_baseline": majority_accuracy(y_test),
            "threshold": threshold,
            "model": model_name,
            "feature_family": feature_family,
            "horizon": horizon,
            "selection_source": "candidate_fixed_before_walk_forward_window",
        }

    val_rows = [row for args in validation_windows if (row := eval_window(*args, split_name="validation")) is not None]
    final_rows = [row for args in final_windows if (row := eval_window(*args, split_name="final")) is not None]
    val_df = pd.DataFrame(val_rows)
    final_df = pd.DataFrame(final_rows)
    config = {
        "status": "run",
        "candidate_id": selected.get("candidate_id", ""),
        "model": model_name,
        "feature_family": feature_family,
        "horizon": horizon,
        "threshold": threshold,
        "training_window": "expanding",
        "validation_windows": [{"name": name, "start": str(start), "end": str(end)} for name, start, end in validation_windows],
        "final_windows": [{"name": name, "start": str(start), "end": str(end)} for name, start, end in final_windows],
        "candidate_selection": "validation_only_before_final_claim",
        "data_fetch": False,
        "shuffle": False,
    }
    write_json(REPORT_DIR / "walk_forward_config.json", config)
    write_csv(REPORT_DIR / "walk_forward_validation_results.csv", val_df)
    write_csv(REPORT_DIR / "walk_forward_final_results.csv", final_df)
    return val_df, final_df


def collect_slice_outputs(all_predictions: pd.DataFrame) -> None:
    final = all_predictions[all_predictions["split"].eq("final")].copy() if not all_predictions.empty else pd.DataFrame()
    ticker_frames: list[pd.DataFrame] = []
    month_frames: list[pd.DataFrame] = []
    quarter_frames: list[pd.DataFrame] = []
    regime_frames: list[pd.DataFrame] = []
    for candidate_id, group in final.groupby("candidate_id", sort=True):
        t, m, q, r = period_slice(group, str(candidate_id))
        ticker_frames.append(t)
        month_frames.append(m)
        quarter_frames.append(q)
        regime_frames.append(r)
    write_csv(REPORT_DIR / "ticker_slice_results.csv", pd.concat(ticker_frames, ignore_index=True) if ticker_frames else pd.DataFrame())
    write_csv(REPORT_DIR / "monthly_slice_results.csv", pd.concat(month_frames, ignore_index=True) if month_frames else pd.DataFrame())
    write_csv(REPORT_DIR / "quarterly_slice_results.csv", pd.concat(quarter_frames, ignore_index=True) if quarter_frames else pd.DataFrame())


def make_figures(unified: pd.DataFrame, all_predictions: pd.DataFrame, regime_slices: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.figsize": (10, 5), "axes.grid": True, "font.size": 9})
    if unified.empty:
        return

    def save_current(name: str) -> None:
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / name, dpi=160)
        plt.close()

    best_group = unified.sort_values(["method_group", "final_accuracy"], ascending=[True, False]).groupby("method_group", as_index=False).head(1)
    plt.figure()
    plt.bar(best_group["method_group"], best_group["final_accuracy"] * 100.0, color=["#4c78a8", "#f58518", "#54a24b"][: len(best_group)])
    plt.axhline(REFERENCE_FINAL_ACCURACY * 100.0, color="#b279a2", linestyle="--", label="61.51 reference")
    plt.ylabel("Final accuracy (%)")
    plt.title("Accuracy by Method Group")
    plt.legend()
    save_current("01_accuracy_by_method_group.png")

    baselines = unified[unified["method_group"].eq("baseline")].sort_values("final_accuracy", ascending=False).head(6)
    best_model = unified[unified["method_group"].ne("baseline")].sort_values("validation_accuracy", ascending=False).head(1)
    compare = pd.concat([best_model, baselines], ignore_index=True)
    plt.figure(figsize=(12, 5))
    labels = [str(x)[:28] for x in compare["model"]]
    plt.bar(labels, compare["final_accuracy"] * 100.0, color="#4c78a8")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Final accuracy (%)")
    plt.title("Best Validation-Selected Model vs Simple Baselines")
    save_current("02_best_model_vs_simple_baselines.png")

    scored = unified[unified["validation_accuracy"].notna() & unified["final_accuracy"].notna()].copy()
    plt.figure()
    for group, frame in scored.groupby("method_group"):
        plt.scatter(frame["validation_accuracy"] * 100.0, frame["final_accuracy"] * 100.0, label=group, s=50)
    plt.axhline(REFERENCE_FINAL_ACCURACY * 100.0, color="#b279a2", linestyle="--")
    plt.xlabel("Validation accuracy (%)")
    plt.ylabel("Final accuracy (%)")
    plt.title("Validation vs Final Accuracy")
    plt.legend()
    save_current("03_validation_vs_final_accuracy_by_method.png")

    best = scored.sort_values("validation_accuracy", ascending=False).head(8)
    roll_cols = ["rolling_250_mean", "rolling_500_mean", "rolling_1000_mean"]
    plt.figure(figsize=(12, 5))
    x = np.arange(len(best))
    width = 0.25
    for i, col in enumerate(roll_cols):
        plt.bar(x + (i - 1) * width, best[col] * 100.0, width=width, label=col.replace("_mean", ""))
    plt.xticks(x, [str(c)[:18] for c in best["model"]], rotation=30, ha="right")
    plt.ylabel("Rolling mean accuracy (%)")
    plt.title("Rolling 250/500/1000 Comparison")
    plt.legend()
    save_current("04_rolling_250_500_1000_comparison.png")

    final = all_predictions[all_predictions["split"].eq("final")].copy() if not all_predictions.empty else pd.DataFrame()
    if not final.empty:
        chosen = list(scored.sort_values("validation_accuracy", ascending=False)["candidate_id"].head(6))
        heat = final[final["candidate_id"].isin(chosen)].pivot_table(index="candidate_id", columns="ticker", values="correct", aggfunc="mean")
        plt.figure(figsize=(13, 5))
        plt.imshow(heat.fillna(np.nan).to_numpy(dtype=float), aspect="auto", vmin=0.3, vmax=0.8, cmap="viridis")
        plt.colorbar(label="Accuracy")
        plt.yticks(range(len(heat.index)), [str(x)[:24] for x in heat.index])
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
        plt.title("Ticker Stability Heatmap")
        save_current("05_ticker_stability_heatmap.png")

        final["month"] = pd.to_datetime(final["datetime"]).dt.to_period("M").astype(str)
        month_heat = final[final["candidate_id"].isin(chosen)].pivot_table(index="candidate_id", columns="month", values="correct", aggfunc="mean")
        plt.figure(figsize=(13, 5))
        plt.imshow(month_heat.fillna(np.nan).to_numpy(dtype=float), aspect="auto", vmin=0.3, vmax=0.8, cmap="plasma")
        plt.colorbar(label="Accuracy")
        plt.yticks(range(len(month_heat.index)), [str(x)[:24] for x in month_heat.index])
        plt.xticks(range(len(month_heat.columns)), month_heat.columns, rotation=90)
        plt.title("Month Stability Heatmap")
        save_current("06_month_quarter_stability_heatmap.png")

        if not regime_slices.empty:
            top_candidate = str(scored.sort_values("validation_accuracy", ascending=False).iloc[0]["candidate_id"])
            rs = regime_slices[regime_slices["candidate_id"].eq(top_candidate)].sort_values("regime_key")
            plt.figure()
            plt.bar(rs["regime_key"], rs["accuracy"] * 100.0, color="#72b7b2")
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("Final accuracy (%)")
            plt.title("Regime Slice Accuracy")
            save_current("07_regime_slice_accuracy.png")

        comp = unified[unified["method_group"].isin(["classical_ml", "deep_learning"])].sort_values(["method_group", "validation_accuracy"], ascending=[True, False]).groupby("method_group").head(1)
        plt.figure()
        plt.bar(comp["method_group"], comp["final_accuracy"] * 100.0, color=["#f58518", "#54a24b"][: len(comp)])
        plt.ylabel("Final accuracy (%)")
        plt.title("Deep Learning vs Classical ML Comparison")
        save_current("08_deep_learning_vs_classical_ml.png")

        top_candidate = str(scored.sort_values("validation_accuracy", ascending=False).iloc[0]["candidate_id"])
        fp = final[final["candidate_id"].eq(top_candidate)].sort_values(["datetime", "ticker"]).head(300)
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(fp)), fp["y_true"].to_numpy(dtype=int), label="actual", linewidth=1.0)
        plt.plot(range(len(fp)), fp["y_pred"].to_numpy(dtype=int), label="forecast", linewidth=1.0, alpha=0.8)
        plt.ylim(-0.1, 1.1)
        plt.ylabel("Direction")
        plt.title("Forecast vs Actual for Best Validation-Selected Candidate")
        plt.legend()
        save_current("09_forecast_vs_actual_best_candidate.png")

    family = unified[unified["method_group"].eq("classical_ml")].copy()
    if not family.empty:
        fam_best = family.sort_values(["feature_family", "validation_accuracy"], ascending=[True, False]).groupby("feature_family").head(1)
        plt.figure(figsize=(12, 5))
        plt.bar(fam_best["feature_family"], fam_best["final_accuracy"] * 100.0, color="#eeca3b")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Final accuracy (%)")
        plt.title("Feature-Family Ablation Chart")
        save_current("10_feature_family_ablation_chart.png")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    features, family_cols, manifest, regime = load_benchmark_features()
    tickers = active_stock_tickers()
    if len(tickers) != 30:
        raise ValueError(f"expected 30 VN30 tickers, found {len(tickers)}")

    write_json(REPORT_DIR / "feature_family_manifest.json", manifest)
    run_config = {
        "run_id": "vn30_full_benchmark_regime_deep_v1",
        "data_fetch": False,
        "provider_behavior_changed": False,
        "ticker_subset": False,
        "confidence_abstention": False,
        "topk": False,
        "final_accuracy_used_for_selection": False,
        "main_target": "VN30 stock-only hourly overall directional accuracy",
        "ticker_count": len(tickers),
        "tickers": tickers,
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
        "strict_label_splits": True,
        "horizons": HORIZONS,
        "deep_horizons": DEEP_HORIZONS,
        "feature_families": FEATURE_FAMILIES,
        "classical_models": CLASSICAL_MODELS,
        "baseline_models": BASELINE_MODELS,
        "deep_models": DEEP_MODELS,
        "threshold_policies": ["fixed_0.50", "validation_selected_threshold"],
        "reference_final_accuracy": REFERENCE_FINAL_ACCURACY,
        "reference_final_rows": REFERENCE_FINAL_ROWS,
        "reference_majority_baseline": REFERENCE_MAJORITY_BASELINE,
    }
    write_json(REPORT_DIR / "run_config.json", run_config)

    baseline_results, baseline_predictions = run_baselines(features)
    classical_results, classical_selected, classical_predictions = run_classical_ml(features, family_cols)
    deep_results, deep_selected, deep_predictions, _deep_report = run_deep_learning(features, family_cols)

    pred_pieces = [frame for frame in [baseline_predictions, classical_predictions, deep_predictions] if frame is not None and not frame.empty]
    all_predictions = pd.concat(pred_pieces, ignore_index=True, sort=False) if pred_pieces else pd.DataFrame()
    regime_slices = run_regime_outputs(regime, all_predictions, manifest["regime_features"])
    collect_slice_outputs(all_predictions)
    run_walk_forward(features, family_cols, classical_selected)
    unified, best = build_unified_outputs(baseline_results, classical_selected, deep_selected, all_predictions)
    make_figures(unified, all_predictions, regime_slices)

    acceptance = str(best.get("claim_level", "failed_to_beat_reference")) if best else "failed_to_beat_reference"
    write_json(
        REPORT_DIR / "benchmark_completion_manifest.json",
        {
            "benchmark_run": True,
            "data_fetch": False,
            "model_training": True,
            "model_selection": "validation_only",
            "paper_docx_generated": False,
            "main_touched": False,
            "outputs_dir": rel(REPORT_DIR),
            "best_overall_validation_selected": best,
            "acceptance_classification": acceptance,
            "baselines_run": not baseline_results.empty,
            "classical_ml_run": not classical_results.empty,
            "deep_learning_run": not deep_results.empty,
            "regime_aware_layer_run": not regime_slices.empty,
            "walk_forward_validation_run": (REPORT_DIR / "walk_forward_validation_results.csv").exists(),
        },
    )
    print(f"VN30 full benchmark complete: {rel(REPORT_DIR)}")
    print(f"Best validation-selected candidate: {best.get('candidate_id', '') if best else ''}")
    print(f"Final accuracy: {pct(best.get('final_accuracy')) if best else ''}")


if __name__ == "__main__":
    main()
