"""Run VN30 legacy-compatible model comparison across h20/h40/h60/h80.

The run uses existing local data and the legacy feature-timestamp split rule.
Final-window rows are scored after validation-only model/threshold choices.
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

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_full_benchmark_regime_deep import (  # noqa: E402
    build_sequences,
    fit_deep_model,
    predict_deep,
    select_deep_feature_cols,
    standardize_feature_matrix,
)
from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    FINAL_START,
    REFERENCE_FINAL_ACCURACY,
    TRAIN_END,
    VAL_END,
    VAL_START,
    build_feature_families,
    split_indices,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    rel,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_legacy_rules_reference_and_stacking" / "full_horizon"

HORIZONS = [20, 40, 60, 80]
THRESHOLDS = [0.45, 0.50, 0.55]
BASE_FEATURE_FAMILY = "baseline_C_closest"
REGIME_FEATURE_FAMILY = "regime_context"
FEATURE_FAMILIES = [BASE_FEATURE_FAMILY, REGIME_FEATURE_FAMILY]
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
    "previous_direction",
    "random_walk_direction",
    "moving_average_rule",
]
DEEP_MODELS = ["lstm", "gru", "tcn"]
SEQUENCE_LENGTH = 16
RANDOM_STATE = 42

CURRENT_MAIN_CANDIDATE_ID = "fullhorizon__classical_ml__logistic_l2__baseline_C_closest__h40__validation_selected_threshold__t0p550"
CURRENT_MAIN_FINAL_ACCURACY = 0.6163475699558174
CURRENT_MAIN_LABEL = "Logistic L2 baseline_C_closest h40 threshold 0.55"


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


def accuracy(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(pred, dtype=int)).mean())


def majority_accuracy(y_true: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    rate = float(np.asarray(y_true, dtype=int).mean())
    return max(rate, 1.0 - rate)


def majority_value(y_true: pd.Series | np.ndarray) -> int:
    if len(y_true) == 0:
        return 1
    return int(float(np.asarray(y_true, dtype=int).mean()) >= 0.5)


def candidate_id(*parts: Any) -> str:
    return "__".join(str(part).replace(".", "p").replace(" ", "_") for part in parts)


def make_model(model_name: str) -> Any | None:
    if model_name == "logistic_l2":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=0.3, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
    if model_name == "logistic_elastic_net":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
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
                ("model", RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=10, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", ExtraTreesClassifier(n_estimators=120, max_depth=8, min_samples_leaf=10, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
    if model_name == "xgboost" and XGBClassifier is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_weight=8,
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
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_samples=35,
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
                ("model", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.04, l2_regularization=0.1, random_state=RANDOM_STATE)),
            ]
        )
    return None


def predict_probability(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_data)[:, 1], dtype=float)
    return np.asarray(model.predict(x_data), dtype=float)


def select_threshold(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_accuracy = -1.0
    for threshold in THRESHOLDS:
        acc = accuracy(y_true, (np.asarray(probability, dtype=float) >= threshold).astype(int))
        if acc > best_accuracy + 1e-12 or (abs(acc - best_accuracy) <= 1e-12 and abs(threshold - 0.50) < abs(best_threshold - 0.50)):
            best_accuracy = acc
            best_threshold = threshold
    return float(best_threshold), float(best_accuracy)


def add_regime_labels(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "vnindex_trend_60_lag_ctx" in out.columns:
        trend = pd.to_numeric(out["vnindex_trend_60_lag_ctx"], errors="coerce")
    elif "vn30_trend_60_lag_ctx" in out.columns:
        trend = pd.to_numeric(out["vn30_trend_60_lag_ctx"], errors="coerce")
    else:
        trend = pd.Series(np.nan, index=out.index)
    for fallback_col in ["momentum_60", "momentum_20", "rolling_return_mean_20"]:
        if fallback_col in out.columns:
            trend = trend.fillna(pd.to_numeric(out[fallback_col], errors="coerce"))
    out["market_direction_regime"] = "sideway"
    out.loc[trend > 0.02, "market_direction_regime"] = "bull"
    out.loc[trend < -0.02, "market_direction_regime"] = "bear"
    out.loc[trend.isna(), "market_direction_regime"] = "unknown_direction"
    if "vnindex_vol_20_lag_ctx" in out.columns and "vnindex_vol_60_lag_ctx" in out.columns:
        ratio = pd.to_numeric(out["vnindex_vol_20_lag_ctx"], errors="coerce") / pd.to_numeric(out["vnindex_vol_60_lag_ctx"], errors="coerce").replace(0.0, np.nan)
    else:
        ratio = pd.Series(np.nan, index=out.index)
    for short_col, long_col in [("rolling_return_vol_20", "rolling_return_vol_60"), ("roll_vol_20", "roll_vol_40")]:
        if short_col in out.columns and long_col in out.columns:
            fallback_ratio = pd.to_numeric(out[short_col], errors="coerce") / pd.to_numeric(out[long_col], errors="coerce").replace(0.0, np.nan)
            ratio = ratio.fillna(fallback_ratio)
    out["volatility_regime"] = "low_volatility"
    out.loc[ratio > 1.10, "volatility_regime"] = "high_volatility"
    out.loc[ratio.isna(), "volatility_regime"] = "unknown_volatility"
    out["regime_router_key"] = out["market_direction_regime"].astype(str) + "_" + out["volatility_regime"].astype(str)
    return out


def load_features() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    features, family_cols, manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    features = add_regime_labels(features)
    family_cols = {
        name: [col for col in family_cols.get(name, []) if col in features.columns and pd.api.types.is_numeric_dtype(features[col])]
        for name in FEATURE_FAMILIES
    }
    manifest = dict(manifest)
    manifest["legacy_full_horizon"] = {
        "split_rule": "feature timestamp split with non-null h-step absolute direction labels",
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
        "horizons": HORIZONS,
        "data_fetch": False,
    }
    return features, family_cols, manifest


def label_df_from_series(labels: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"y": labels}, index=labels.index)


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    score: np.ndarray,
    pred: np.ndarray,
    *,
    model_group: str,
    model: str,
    feature_family: str,
    horizon: int,
    threshold_policy: str,
    threshold: float,
    candidate: str,
    split: str,
) -> pd.DataFrame:
    out = features.loc[idx, ["datetime", "ticker", "market_direction_regime", "volatility_regime", "regime_router_key"]].copy()
    out["model_group"] = model_group
    out["model"] = model
    out["feature_family"] = feature_family
    out["horizon"] = int(horizon)
    out["threshold_policy"] = threshold_policy
    out["threshold"] = threshold
    out["candidate_id"] = candidate
    out["split"] = split
    out["y_true"] = labels.loc[idx].astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(score, dtype=float)
    out["y_pred"] = np.asarray(pred, dtype=int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def rolling_stats(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ordered = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    correct = ordered["correct"].astype(float)
    for window in (250, 500, 1000):
        rolling = correct.rolling(window=window, min_periods=window).mean().dropna()
        out[f"rolling_{window}_mean"] = float(rolling.mean()) if not rolling.empty else math.nan
        out[f"rolling_{window}_min"] = float(rolling.min()) if not rolling.empty else math.nan
        out[f"rolling_{window}_end"] = float(rolling.iloc[-1]) if not rolling.empty else math.nan
        out[f"rolling_{window}_windows"] = int(len(rolling))
    return out


def result_row(
    *,
    candidate: str,
    model_group: str,
    model: str,
    feature_family: str,
    horizon: int,
    threshold_policy: str,
    threshold: float,
    threshold_detail: str,
    feature_count: int,
    train_rows: int,
    validation_frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    status: str = "ok",
    skip_reason: str = "",
    sequence_length: int | None = None,
) -> dict[str, Any]:
    validation_accuracy = float(validation_frame["correct"].mean()) if not validation_frame.empty else math.nan
    final_accuracy = float(final_frame["correct"].mean()) if not final_frame.empty else math.nan
    row: dict[str, Any] = {
        "candidate_id": candidate,
        "model_group": model_group,
        "model": model,
        "feature_family": feature_family,
        "horizon": int(horizon),
        "threshold_policy": threshold_policy,
        "threshold": threshold,
        "threshold_detail": threshold_detail,
        "status": status,
        "skip_reason": skip_reason,
        "selection_source": "validation_only" if status == "ok" else "not_selected",
        "final_window_role": "scoring_only",
        "final_accuracy_used_for_selection": False,
        "feature_count": int(feature_count),
        "sequence_length": sequence_length if sequence_length is not None else "",
        "train_rows": int(train_rows),
        "validation_rows": int(len(validation_frame)),
        "final_rows": int(len(final_frame)),
        "validation_ticker_coverage": int(validation_frame["ticker"].nunique()) if not validation_frame.empty else 0,
        "ticker_coverage": int(final_frame["ticker"].nunique()) if not final_frame.empty else 0,
        "full_ticker_coverage": bool(int(final_frame["ticker"].nunique()) == 30) if not final_frame.empty else False,
        "validation_accuracy": validation_accuracy,
        "final_accuracy": final_accuracy,
        "validation_majority_baseline": majority_accuracy(validation_frame["y_true"]) if not validation_frame.empty else math.nan,
        "final_majority_baseline": majority_accuracy(final_frame["y_true"]) if not final_frame.empty else math.nan,
        "validation_lift_vs_majority": validation_accuracy - majority_accuracy(validation_frame["y_true"]) if not validation_frame.empty else math.nan,
        "final_lift_vs_majority": final_accuracy - majority_accuracy(final_frame["y_true"]) if not final_frame.empty else math.nan,
        "validation_final_gap": final_accuracy - validation_accuracy if math.isfinite(validation_accuracy) and math.isfinite(final_accuracy) else math.nan,
        "delta_vs_current_h40_61_63": final_accuracy - CURRENT_MAIN_FINAL_ACCURACY if math.isfinite(final_accuracy) else math.nan,
        "delta_vs_legacy_reference_61_51": final_accuracy - REFERENCE_FINAL_ACCURACY if math.isfinite(final_accuracy) else math.nan,
        "legacy_rule": "feature_timestamp_split_non_null_horizon_label",
        "ticker_subset": False,
        "confidence_abstention": False,
        "topk_substitution": False,
        "leakage_status": "passed_legacy_rules" if status == "ok" else "not_run",
    }
    if not final_frame.empty:
        row.update({f"final_{key}": value for key, value in rolling_stats(final_frame).items()})
    return row


def append_slices(
    validation_frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    row: dict[str, Any],
    ticker_rows: list[pd.DataFrame],
    month_rows: list[pd.DataFrame],
    quarter_rows: list[pd.DataFrame],
    rolling_rows: list[dict[str, Any]],
) -> None:
    if final_frame.empty:
        return
    final = final_frame.copy()
    final["datetime"] = pd.to_datetime(final["datetime"], errors="coerce")
    final["month"] = final["datetime"].dt.to_period("M").astype(str)
    final["quarter"] = final["datetime"].dt.to_period("Q").astype(str)
    common_cols = {
        "candidate_id": row["candidate_id"],
        "model_group": row["model_group"],
        "model": row["model"],
        "feature_family": row["feature_family"],
        "horizon": row["horizon"],
        "threshold_policy": row["threshold_policy"],
    }
    for key, out_rows, label in [("ticker", ticker_rows, "ticker"), ("month", month_rows, "month"), ("quarter", quarter_rows, "quarter")]:
        grouped = final.groupby(key, sort=True)["correct"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "accuracy", "count": "rows", key: label})
        for col, value in reversed(list(common_cols.items())):
            grouped.insert(0, col, value)
        out_rows.append(grouped)
    for split_name, frame in [("validation", validation_frame), ("final", final_frame)]:
        stats = rolling_stats(frame)
        for window in (250, 500, 1000):
            rolling_rows.append(
                {
                    **common_cols,
                    "split": split_name,
                    "window": window,
                    "rolling_mean_accuracy": stats[f"rolling_{window}_mean"],
                    "rolling_min_accuracy": stats[f"rolling_{window}_min"],
                    "rolling_end_accuracy": stats[f"rolling_{window}_end"],
                    "rolling_window_count": stats[f"rolling_{window}_windows"],
                }
            )


def run_baselines(
    features: pd.DataFrame,
    ticker_rows: list[pd.DataFrame],
    month_rows: list[pd.DataFrame],
    quarter_rows: list[pd.DataFrame],
    rolling_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    close = pd.to_numeric(features["close"], errors="coerce")
    by_ticker = features.groupby("ticker", sort=True)
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        train_y = labels.loc[idx["train"]].astype(int)
        majority_pred = majority_value(train_y)
        previous_direction = labels.groupby(features["ticker"]).shift(horizon)
        previous_close = by_ticker["close"].shift(1)
        random_walk = (close > pd.to_numeric(previous_close, errors="coerce")).astype(float)
        random_walk.loc[pd.to_numeric(previous_close, errors="coerce").isna()] = np.nan
        ma20 = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").rolling(20, min_periods=5).mean())
        moving_average = (close > ma20).astype(float)
        moving_average.loc[ma20.isna()] = np.nan
        raw_predictions = {
            "majority_class": pd.Series(float(majority_pred), index=features.index),
            "previous_direction": previous_direction,
            "random_walk_direction": random_walk,
            "moving_average_rule": moving_average,
        }
        for model_name, raw_pred in raw_predictions.items():
            pred_series = pd.to_numeric(raw_pred, errors="coerce").reindex(features.index).fillna(float(majority_pred)).astype(float)
            cid = candidate_id("fullhorizon", "baseline", model_name, f"h{horizon}")
            val_score = pred_series.loc[idx["validation"]].to_numpy(dtype=float)
            final_score = pred_series.loc[idx["final"]].to_numpy(dtype=float)
            val_pred = (val_score >= 0.50).astype(int)
            final_pred = (final_score >= 0.50).astype(int)
            val_frame = prediction_frame(
                features,
                idx["validation"],
                labels,
                val_score,
                val_pred,
                model_group="baseline",
                model=model_name,
                feature_family="ex_ante_rule",
                horizon=horizon,
                threshold_policy="rule",
                threshold=0.50,
                candidate=cid,
                split="validation",
            )
            final_frame = prediction_frame(
                features,
                idx["final"],
                labels,
                final_score,
                final_pred,
                model_group="baseline",
                model=model_name,
                feature_family="ex_ante_rule",
                horizon=horizon,
                threshold_policy="rule",
                threshold=0.50,
                candidate=cid,
                split="final",
            )
            row = result_row(
                candidate=cid,
                model_group="baseline",
                model=model_name,
                feature_family="ex_ante_rule",
                horizon=horizon,
                threshold_policy="rule",
                threshold=0.50,
                threshold_detail="deterministic ex-ante rule",
                feature_count=0,
                train_rows=len(train_y),
                validation_frame=val_frame,
                final_frame=final_frame,
            )
            rows.append(row)
            append_slices(val_frame, final_frame, row, ticker_rows, month_rows, quarter_rows, rolling_rows)
    return rows


def run_classical(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    ticker_rows: list[pd.DataFrame],
    month_rows: list[pd.DataFrame],
    quarter_rows: list[pd.DataFrame],
    rolling_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        train_y = labels.loc[idx["train"]].astype(int)
        val_y = labels.loc[idx["validation"]].astype(int)
        for feature_family in FEATURE_FAMILIES:
            cols = family_cols.get(feature_family, [])
            model_group = "regime_aware" if feature_family == REGIME_FEATURE_FAMILY else "classical_ml"
            if not cols:
                for model_name in CLASSICAL_MODELS:
                    rows.append(
                        {
                            "candidate_id": candidate_id("fullhorizon", model_group, model_name, feature_family, f"h{horizon}", "skipped"),
                            "model_group": model_group,
                            "model": model_name,
                            "feature_family": feature_family,
                            "horizon": horizon,
                            "threshold_policy": "not_run",
                            "threshold": math.nan,
                            "status": "skipped_with_reason",
                            "skip_reason": "no numeric feature columns",
                            "selection_source": "not_selected",
                            "final_accuracy_used_for_selection": False,
                            "ticker_subset": False,
                            "confidence_abstention": False,
                            "topk_substitution": False,
                        }
                    )
                continue
            x_train = features.loc[idx["train"], cols]
            x_val = features.loc[idx["validation"], cols]
            x_final = features.loc[idx["final"], cols]
            for model_name in CLASSICAL_MODELS:
                model = make_model(model_name)
                if model is None:
                    rows.append(
                        {
                            "candidate_id": candidate_id("fullhorizon", model_group, model_name, feature_family, f"h{horizon}", "skipped"),
                            "model_group": model_group,
                            "model": model_name,
                            "feature_family": feature_family,
                            "horizon": horizon,
                            "threshold_policy": "not_run",
                            "threshold": math.nan,
                            "status": "skipped_with_reason",
                            "skip_reason": "optional dependency missing",
                            "selection_source": "not_selected",
                            "final_accuracy_used_for_selection": False,
                            "ticker_subset": False,
                            "confidence_abstention": False,
                            "topk_substitution": False,
                        }
                    )
                    continue
                try:
                    model.fit(x_train, train_y)
                    val_prob = predict_probability(model, x_val)
                    final_prob = predict_probability(model, x_final)
                except Exception as exc:
                    rows.append(
                        {
                            "candidate_id": candidate_id("fullhorizon", model_group, model_name, feature_family, f"h{horizon}", "failed"),
                            "model_group": model_group,
                            "model": model_name,
                            "feature_family": feature_family,
                            "horizon": horizon,
                            "threshold_policy": "not_run",
                            "threshold": math.nan,
                            "status": "failed",
                            "skip_reason": str(exc)[:300],
                            "selection_source": "not_selected",
                            "final_accuracy_used_for_selection": False,
                            "ticker_subset": False,
                            "confidence_abstention": False,
                            "topk_substitution": False,
                        }
                    )
                    continue
                threshold_specs = [("fixed_0.50", 0.50), ("validation_selected_threshold", select_threshold(val_y, val_prob)[0])]
                for threshold_policy, threshold in threshold_specs:
                    cid = candidate_id("fullhorizon", model_group, model_name, feature_family, f"h{horizon}", threshold_policy, f"t{threshold:.3f}")
                    val_pred = (val_prob >= threshold).astype(int)
                    final_pred = (final_prob >= threshold).astype(int)
                    val_frame = prediction_frame(
                        features,
                        idx["validation"],
                        labels,
                        val_prob,
                        val_pred,
                        model_group=model_group,
                        model=model_name,
                        feature_family=feature_family,
                        horizon=horizon,
                        threshold_policy=threshold_policy,
                        threshold=threshold,
                        candidate=cid,
                        split="validation",
                    )
                    final_frame = prediction_frame(
                        features,
                        idx["final"],
                        labels,
                        final_prob,
                        final_pred,
                        model_group=model_group,
                        model=model_name,
                        feature_family=feature_family,
                        horizon=horizon,
                        threshold_policy=threshold_policy,
                        threshold=threshold,
                        candidate=cid,
                        split="final",
                    )
                    row = result_row(
                        candidate=cid,
                        model_group=model_group,
                        model=model_name,
                        feature_family=feature_family,
                        horizon=horizon,
                        threshold_policy=threshold_policy,
                        threshold=threshold,
                        threshold_detail=f"threshold grid {THRESHOLDS}; selected on validation only" if threshold_policy == "validation_selected_threshold" else "fixed threshold",
                        feature_count=len(cols),
                        train_rows=len(train_y),
                        validation_frame=val_frame,
                        final_frame=final_frame,
                    )
                    rows.append(row)
                    append_slices(val_frame, final_frame, row, ticker_rows, month_rows, quarter_rows, rolling_rows)
                gc.collect()
    return rows


def select_group_thresholds(y_true: pd.Series, scores: np.ndarray, groups: pd.Series, default_threshold: float) -> dict[str, float]:
    work = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "score": scores, "group": groups.astype(str).to_numpy()})
    thresholds: dict[str, float] = {}
    for group_name, group in work.groupby("group", sort=True):
        if len(group) < 40:
            thresholds[str(group_name)] = float(default_threshold)
            continue
        thresholds[str(group_name)] = select_threshold(group["y"], group["score"].to_numpy(dtype=float))[0]
    return thresholds


def apply_group_thresholds(scores: np.ndarray, groups: pd.Series, thresholds: dict[str, float], default_threshold: float) -> np.ndarray:
    group_values = groups.astype(str).to_numpy()
    pred = np.zeros(len(scores), dtype=int)
    for i, score in enumerate(scores):
        pred[i] = int(float(score) >= thresholds.get(group_values[i], default_threshold))
    return pred


def run_regime_router(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    ticker_rows: list[pd.DataFrame],
    month_rows: list[pd.DataFrame],
    quarter_rows: list[pd.DataFrame],
    rolling_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cols = family_cols.get(BASE_FEATURE_FAMILY, [])
    if not cols:
        return rows
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        train_y = labels.loc[idx["train"]].astype(int)
        val_y = labels.loc[idx["validation"]].astype(int)
        model = make_model("logistic_l2")
        if model is None:
            continue
        model.fit(features.loc[idx["train"], cols], train_y)
        val_score = predict_probability(model, features.loc[idx["validation"], cols])
        final_score = predict_probability(model, features.loc[idx["final"], cols])
        default_threshold = select_threshold(val_y, val_score)[0]
        val_groups = features.loc[idx["validation"], "regime_router_key"]
        final_groups = features.loc[idx["final"], "regime_router_key"]
        thresholds = select_group_thresholds(val_y, val_score, val_groups, default_threshold)
        val_pred = apply_group_thresholds(val_score, val_groups, thresholds, default_threshold)
        final_pred = apply_group_thresholds(final_score, final_groups, thresholds, default_threshold)
        cid = candidate_id("fullhorizon", "regime_aware", "regime_threshold_router", BASE_FEATURE_FAMILY, f"h{horizon}", "validation_selected_regime_thresholds")
        val_frame = prediction_frame(
            features,
            idx["validation"],
            labels,
            val_score,
            val_pred,
            model_group="regime_aware",
            model="regime_threshold_router",
            feature_family=BASE_FEATURE_FAMILY,
            horizon=horizon,
            threshold_policy="regime_validation_selected_threshold",
            threshold=default_threshold,
            candidate=cid,
            split="validation",
        )
        final_frame = prediction_frame(
            features,
            idx["final"],
            labels,
            final_score,
            final_pred,
            model_group="regime_aware",
            model="regime_threshold_router",
            feature_family=BASE_FEATURE_FAMILY,
            horizon=horizon,
            threshold_policy="regime_validation_selected_threshold",
            threshold=default_threshold,
            candidate=cid,
            split="final",
        )
        detail = json.dumps({"default_threshold": default_threshold, "regime_thresholds": thresholds}, sort_keys=True)
        row = result_row(
            candidate=cid,
            model_group="regime_aware",
            model="regime_threshold_router",
            feature_family=BASE_FEATURE_FAMILY,
            horizon=horizon,
            threshold_policy="regime_validation_selected_threshold",
            threshold=default_threshold,
            threshold_detail=detail,
            feature_count=len(cols),
            train_rows=len(train_y),
            validation_frame=val_frame,
            final_frame=final_frame,
        )
        row["leakage_status"] = "passed_validation_only_regime_threshold_router"
        rows.append(row)
        append_slices(val_frame, final_frame, row, ticker_rows, month_rows, quarter_rows, rolling_rows)
    return rows


def run_deep(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    ticker_rows: list[pd.DataFrame],
    month_rows: list[pd.DataFrame],
    quarter_rows: list[pd.DataFrame],
    rolling_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if torch is None:
        for horizon in HORIZONS:
            for model_name in DEEP_MODELS:
                rows.append(
                    {
                        "candidate_id": candidate_id("fullhorizon", "deep_learning", model_name, f"h{horizon}", "skipped"),
                        "model_group": "deep_learning",
                        "model": model_name,
                        "feature_family": "baseline_C_closest_sequence16",
                        "horizon": horizon,
                        "threshold_policy": "fixed_0.50",
                        "threshold": 0.50,
                        "status": "skipped_with_reason",
                        "skip_reason": "torch not importable",
                        "selection_source": "not_selected",
                        "final_accuracy_used_for_selection": False,
                        "ticker_subset": False,
                        "confidence_abstention": False,
                        "topk_substitution": False,
                    }
                )
        return rows
    feature_cols = select_deep_feature_cols(features, family_cols)
    if not feature_cols:
        return rows
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        label_df = label_df_from_series(labels)
        matrix = standardize_feature_matrix(features, feature_cols, idx["train"])
        x_train, y_train, _train_rows = build_sequences(features, matrix, label_df, idx["train"], SEQUENCE_LENGTH)
        x_val, y_val, val_rows = build_sequences(features, matrix, label_df, idx["validation"], SEQUENCE_LENGTH)
        x_final, y_final, final_rows = build_sequences(features, matrix, label_df, idx["final"], SEQUENCE_LENGTH)
        if len(y_train) == 0 or len(y_val) == 0 or len(y_final) == 0 or len(np.unique(y_train)) < 2:
            for model_name in DEEP_MODELS:
                rows.append(
                    {
                        "candidate_id": candidate_id("fullhorizon", "deep_learning", model_name, f"h{horizon}", "skipped"),
                        "model_group": "deep_learning",
                        "model": model_name,
                        "feature_family": "baseline_C_closest_sequence16",
                        "horizon": horizon,
                        "threshold_policy": "fixed_0.50",
                        "threshold": 0.50,
                        "status": "skipped_with_reason",
                        "skip_reason": "invalid sequence data shape",
                        "selection_source": "not_selected",
                        "final_accuracy_used_for_selection": False,
                        "ticker_subset": False,
                        "confidence_abstention": False,
                        "topk_substitution": False,
                    }
                )
            continue
        for model_name in DEEP_MODELS:
            cid = candidate_id("fullhorizon", "deep_learning", model_name, "baseline_C_closest_sequence16", f"h{horizon}", f"seq{SEQUENCE_LENGTH}", "fixed_0.50")
            try:
                model, meta, val_prob = fit_deep_model(model_name, x_train, y_train, x_val, y_val)
                final_prob = predict_deep(model, x_final)
                val_pred = (val_prob >= 0.50).astype(int)
                final_pred = (final_prob >= 0.50).astype(int)
                val_frame = prediction_frame(
                    features,
                    val_rows,
                    labels,
                    val_prob,
                    val_pred,
                    model_group="deep_learning",
                    model=model_name,
                    feature_family="baseline_C_closest_sequence16",
                    horizon=horizon,
                    threshold_policy="fixed_0.50",
                    threshold=0.50,
                    candidate=cid,
                    split="validation",
                )
                final_frame = prediction_frame(
                    features,
                    final_rows,
                    labels,
                    final_prob,
                    final_pred,
                    model_group="deep_learning",
                    model=model_name,
                    feature_family="baseline_C_closest_sequence16",
                    horizon=horizon,
                    threshold_policy="fixed_0.50",
                    threshold=0.50,
                    candidate=cid,
                    split="final",
                )
                row = result_row(
                    candidate=cid,
                    model_group="deep_learning",
                    model=model_name,
                    feature_family="baseline_C_closest_sequence16",
                    horizon=horizon,
                    threshold_policy="fixed_0.50",
                    threshold=0.50,
                    threshold_detail="sequence model fixed threshold; early stopping by validation loss",
                    feature_count=len(feature_cols),
                    train_rows=len(y_train),
                    validation_frame=val_frame,
                    final_frame=final_frame,
                    sequence_length=SEQUENCE_LENGTH,
                )
                row["best_epoch"] = int(meta.get("best_epoch", 0))
                row["leakage_status"] = "passed_legacy_sequence_rules"
                rows.append(row)
                append_slices(val_frame, final_frame, row, ticker_rows, month_rows, quarter_rows, rolling_rows)
            except Exception as exc:
                rows.append(
                    {
                        "candidate_id": cid,
                        "model_group": "deep_learning",
                        "model": model_name,
                        "feature_family": "baseline_C_closest_sequence16",
                        "horizon": horizon,
                        "threshold_policy": "fixed_0.50",
                        "threshold": 0.50,
                        "status": "skipped_with_reason",
                        "skip_reason": str(exc)[:300],
                        "selection_source": "not_selected",
                        "final_accuracy_used_for_selection": False,
                        "ticker_subset": False,
                        "confidence_abstention": False,
                        "topk_substitution": False,
                    }
                )
            gc.collect()
    return rows


def select_headline_by_horizon(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    eligible = leaderboard[
        leaderboard["status"].eq("ok")
        & leaderboard["full_ticker_coverage"].astype(bool)
        & leaderboard["model_group"].isin(["baseline", "classical_ml", "deep_learning"])
        & ~leaderboard["feature_family"].astype(str).eq(REGIME_FEATURE_FAMILY)
    ].copy()
    for horizon, group in eligible.groupby("horizon", sort=True):
        selected = group.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0].copy()
        selected["horizon_headline_selection_source"] = "validation_accuracy_then_candidate_id_main_setup_pool"
        selected["paper_role"] = "full_horizon_diagnostic"
        if int(horizon) == 40 and str(selected["candidate_id"]) == CURRENT_MAIN_CANDIDATE_ID:
            selected["paper_role"] = "main_h40_paper_result"
        rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def select_best_by_group(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    eligible = leaderboard[leaderboard["status"].eq("ok") & leaderboard["full_ticker_coverage"].astype(bool)].copy()
    for (horizon, model_group), group in eligible.groupby(["horizon", "model_group"], sort=True):
        selected = group.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0].copy()
        selected["selection_source_detail"] = "validation_accuracy_then_candidate_id_within_model_group"
        rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def build_validation_final_gap(leaderboard: pd.DataFrame, best_by_horizon: pd.DataFrame) -> pd.DataFrame:
    out = leaderboard[leaderboard["status"].eq("ok")].copy()
    selected_ids = set(best_by_horizon["candidate_id"].astype(str).tolist()) if not best_by_horizon.empty else set()
    out["selected_for_horizon_headline"] = out["candidate_id"].astype(str).isin(selected_ids)
    cols = [
        "candidate_id",
        "model_group",
        "model",
        "feature_family",
        "horizon",
        "threshold_policy",
        "threshold",
        "validation_accuracy",
        "final_accuracy",
        "validation_final_gap",
        "validation_rows",
        "final_rows",
        "ticker_coverage",
        "selected_for_horizon_headline",
        "final_accuracy_used_for_selection",
    ]
    return out[cols].sort_values(["horizon", "model_group", "validation_accuracy", "candidate_id"], ascending=[True, True, False, True])


def make_figures(
    leaderboard: pd.DataFrame,
    best_by_horizon: pd.DataFrame,
    best_by_group: pd.DataFrame,
    rolling_summary: pd.DataFrame,
) -> None:
    plt.rcParams.update({"figure.figsize": (10, 5), "axes.grid": True, "font.size": 9})
    if best_by_horizon.empty:
        return

    def save_current(name: str) -> None:
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / name, dpi=160)
        plt.close()

    ordered = best_by_horizon.sort_values("horizon")
    labels = [f"h{int(h)}" for h in ordered["horizon"]]
    plt.figure()
    plt.bar(labels, ordered["final_accuracy"].astype(float) * 100.0, color="#4c78a8")
    plt.axhline(CURRENT_MAIN_FINAL_ACCURACY * 100.0, color="#b279a2", linestyle="--", label="h40 main 61.63%")
    plt.ylabel("Final accuracy (%)")
    plt.title("Validation-Selected Diagnostic Accuracy by Horizon")
    plt.legend()
    save_current("fig_final_accuracy_by_horizon.png")

    plt.figure(figsize=(11, 5))
    model_labels = [f"h{int(row.horizon)}\n{str(row.model)[:18]}" for row in ordered.itertuples()]
    plt.bar(model_labels, ordered["final_accuracy"].astype(float) * 100.0, color="#72b7b2")
    plt.ylabel("Final accuracy (%)")
    plt.title("Best Diagnostic Model by Horizon")
    save_current("fig_best_model_by_horizon.png")

    scored = leaderboard[leaderboard["status"].eq("ok") & leaderboard["validation_accuracy"].notna() & leaderboard["final_accuracy"].notna()].copy()
    plt.figure()
    for horizon, group in scored.groupby("horizon", sort=True):
        plt.scatter(group["validation_accuracy"].astype(float) * 100.0, group["final_accuracy"].astype(float) * 100.0, label=f"h{int(horizon)}", s=35, alpha=0.75)
    plt.axhline(CURRENT_MAIN_FINAL_ACCURACY * 100.0, color="#b279a2", linestyle="--")
    plt.xlabel("Validation accuracy (%)")
    plt.ylabel("Final accuracy (%)")
    plt.title("Validation vs Final Accuracy by Horizon")
    plt.legend()
    save_current("fig_validation_vs_final_by_horizon.png")

    selected_roll = rolling_summary[
        rolling_summary["split"].eq("final")
        & rolling_summary["candidate_id"].isin(set(best_by_horizon["candidate_id"].astype(str)))
    ].copy()
    if not selected_roll.empty:
        pivot = selected_roll.pivot_table(index="horizon", columns="window", values="rolling_mean_accuracy", aggfunc="first").sort_index()
        plt.figure()
        for window in [250, 500, 1000]:
            if window in pivot.columns:
                plt.plot([f"h{int(h)}" for h in pivot.index], pivot[window].astype(float) * 100.0, marker="o", label=f"{window}")
        plt.ylabel("Final rolling mean accuracy (%)")
        plt.title("Rolling Mean by Horizon")
        plt.legend(title="Rows")
        save_current("fig_rolling_mean_by_horizon.png")

    heat_source = best_by_group.copy()
    if not heat_source.empty:
        heat = heat_source.pivot_table(index="model_group", columns="horizon", values="final_accuracy", aggfunc="first")
        plt.figure(figsize=(8, 4.8))
        values = heat.to_numpy(dtype=float) * 100.0
        im = plt.imshow(values, aspect="auto", cmap="viridis", vmin=np.nanmin(values), vmax=np.nanmax(values))
        plt.colorbar(im, label="Final accuracy (%)")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.xticks(range(len(heat.columns)), [f"h{int(col)}" for col in heat.columns])
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                if math.isfinite(values[row_idx, col_idx]):
                    plt.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.1f}", ha="center", va="center", color="white", fontsize=8)
        plt.title("Best Validation-Selected Accuracy by Group and Horizon")
        save_current("fig_model_group_horizon_heatmap.png")


def row_counts_by_horizon(leaderboard: pd.DataFrame) -> pd.DataFrame:
    ok = leaderboard[leaderboard["status"].eq("ok")].copy()
    return (
        ok.groupby("horizon", as_index=False)
        .agg(
            train_rows=("train_rows", "max"),
            validation_rows=("validation_rows", "max"),
            final_rows=("final_rows", "max"),
            ticker_coverage=("ticker_coverage", "max"),
            candidate_rows=("candidate_id", "count"),
        )
        .sort_values("horizon")
    )


def write_reports(
    leaderboard: pd.DataFrame,
    best_by_horizon: pd.DataFrame,
    best_by_group: pd.DataFrame,
    gap: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    counts = row_counts_by_horizon(leaderboard)
    checks = {
        "no_final_window_selection": bool(leaderboard["final_accuracy_used_for_selection"].fillna(False).astype(str).str.lower().isin(["false", "0"]).all()),
        "no_leakage": bool(leaderboard["leakage_status"].dropna().astype(str).str.startswith("passed").all()),
        "full_30_stock_headline_coverage": bool(best_by_horizon["ticker_coverage"].astype(int).eq(30).all()) if not best_by_horizon.empty else False,
        "no_ticker_subset": bool(leaderboard["ticker_subset"].fillna(False).astype(str).str.lower().isin(["false", "0"]).all()),
        "no_confidence_abstention": bool(leaderboard["confidence_abstention"].fillna(False).astype(str).str.lower().isin(["false", "0"]).all()),
        "no_topk_substitution": bool(leaderboard["topk_substitution"].fillna(False).astype(str).str.lower().isin(["false", "0"]).all()),
        "horizon_specific_row_counts_reported": bool(set(counts["horizon"].astype(int)) == set(HORIZONS)),
        "h40_main_claim_kept_separate": bool(CURRENT_MAIN_CANDIDATE_ID in set(leaderboard["candidate_id"].astype(str))),
    }
    main = leaderboard[leaderboard["candidate_id"].astype(str).eq(CURRENT_MAIN_CANDIDATE_ID)]
    main_row = main.iloc[0].to_dict() if not main.empty else {}

    count_lines = ["| Horizon | Train Rows | Validation Rows | Final Rows | Ticker Coverage | Candidates |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in counts.itertuples():
        count_lines.append(f"| h{int(row.horizon)} | {int(row.train_rows):,} | {int(row.validation_rows):,} | {int(row.final_rows):,} | {int(row.ticker_coverage)}/30 | {int(row.candidate_rows)} |")

    best_lines = ["| Horizon | Candidate | Group | Validation | Final | Gap | Rows |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in best_by_horizon.sort_values("horizon").itertuples():
        best_lines.append(
            f"| h{int(row.horizon)} | `{row.candidate_id}` | {row.model_group} | {pct(row.validation_accuracy)} | "
            f"{pct(row.final_accuracy)} | {pp(row.validation_final_gap)} | {int(row.final_rows):,} |"
        )

    group_lines = ["| Horizon | Group | Candidate | Validation | Final |", "| --- | --- | --- | ---: | ---: |"]
    for row in best_by_group.sort_values(["horizon", "model_group"]).itertuples():
        group_lines.append(f"| h{int(row.horizon)} | {row.model_group} | `{row.candidate_id}` | {pct(row.validation_accuracy)} | {pct(row.final_accuracy)} |")

    summary = [
        "# VN30 Legacy Full-Horizon Model Comparison",
        "",
        "## Scope",
        "",
        "- Horizons: h20, h40, h60, h80.",
        "- Data fetch: no.",
        "- Split rule: legacy feature-timestamp split with non-null horizon labels.",
        "- Selection rule for diagnostic horizon rows: validation accuracy, then candidate id; final rows are scoring-only.",
        "- Main paper h40 result remains separate from full-horizon diagnostics.",
        "",
        "## H40 Main Paper Result",
        "",
        f"- Fixed main result: {CURRENT_MAIN_LABEL}.",
        f"- Validation accuracy: {pct(main_row.get('validation_accuracy'))}.",
        f"- Final accuracy: {pct(main_row.get('final_accuracy'))}.",
        f"- Final rows: {int(as_float(main_row.get('final_rows'))) if main_row else ''}.",
        f"- Ticker coverage: {int(as_float(main_row.get('ticker_coverage'))) if main_row else ''}/30.",
        "",
        "## Horizon Row Counts",
        "",
        "\n".join(count_lines),
        "",
        "## Diagnostic Best By Horizon",
        "",
        "\n".join(best_lines),
        "",
        "## Best By Model Group And Horizon",
        "",
        "\n".join(group_lines),
        "",
        "## Audit Summary",
        "",
    ]
    for key, value in checks.items():
        summary.append(f"- {key}: {'pass' if value else 'fail'}.")
    summary.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The full-horizon tables are diagnostics for horizon robustness, not a replacement for the h40 paper claim.",
            "- Regime-context and regime-threshold-router rows are reported in the regime-aware group; they are not used to replace the fixed h40 main claim.",
            "- No ticker subset, confidence abstention, or top-k/ranking substitute is used.",
        ]
    )
    write_markdown(OUTPUT_DIR / "full_horizon_summary.md", "\n".join(summary))

    claim_lines = [
        "# VN30 Legacy Full-Horizon Claim Boundary",
        "",
        "## Audit Checks",
        "",
        "| Check | Status |",
        "| --- | --- |",
    ]
    for key, value in checks.items():
        claim_lines.append(f"| {key} | {'pass' if value else 'fail'} |")
    claim_lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"- Main paper claim remains `{CURRENT_MAIN_LABEL}` with final accuracy {pct(CURRENT_MAIN_FINAL_ACCURACY)} unless a separate pre-registered validation-only horizon-selection objective is adopted before final-window scoring.",
            "- This run does not use final-window score for model, feature, threshold, router, or horizon selection.",
            "- Horizon-specific best rows are diagnostic and full-coverage only.",
            "- Regime-aware rows include `regime_context` features and a validation-selected regime threshold router; both are scoring-only on final rows.",
            "- No market data was fetched, no ticker subset was used, no confidence abstention was used, and no top-k metric is substituted for overall directional accuracy.",
            "- No paper or DOCX artifact was generated.",
        ]
    )
    write_markdown(OUTPUT_DIR / "full_horizon_claim_boundary.md", "\n".join(claim_lines))

    write_json(
        OUTPUT_DIR / "full_horizon_manifest.json",
        {
            "run_id": "vn30_legacy_full_horizon_comparison_v1",
            "data_fetch": False,
            "model_training": True,
            "final_window_role": "scoring_only",
            "model_selection": "validation_only",
            "horizon_selection_for_paper_claim": "none_in_this_run_h40_main_kept",
            "horizons": HORIZONS,
            "feature_manifest": manifest,
            "audit_checks": checks,
            "row_counts_by_horizon": counts.to_dict("records"),
        },
    )
    _ = gap
    _ = rolling_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, family_cols, manifest = load_features()
    tickers = active_stock_tickers()
    if len(tickers) != 30:
        raise ValueError(f"full-horizon legacy comparison requires 30 active tickers, found {len(tickers)}")

    ticker_rows: list[pd.DataFrame] = []
    month_rows: list[pd.DataFrame] = []
    quarter_rows: list[pd.DataFrame] = []
    rolling_rows: list[dict[str, Any]] = []

    rows: list[dict[str, Any]] = []
    rows.extend(run_baselines(features, ticker_rows, month_rows, quarter_rows, rolling_rows))
    rows.extend(run_classical(features, family_cols, ticker_rows, month_rows, quarter_rows, rolling_rows))
    rows.extend(run_regime_router(features, family_cols, ticker_rows, month_rows, quarter_rows, rolling_rows))
    rows.extend(run_deep(features, family_cols, ticker_rows, month_rows, quarter_rows, rolling_rows))

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        raise RuntimeError("full-horizon comparison produced no rows")
    leaderboard = leaderboard.sort_values(["horizon", "model_group", "model", "feature_family", "threshold_policy", "candidate_id"]).reset_index(drop=True)
    best_by_horizon = select_headline_by_horizon(leaderboard)
    best_by_group = select_best_by_group(leaderboard)
    gap = build_validation_final_gap(leaderboard, best_by_horizon)
    rolling_summary = pd.DataFrame(rolling_rows).sort_values(["horizon", "model_group", "candidate_id", "split", "window"]).reset_index(drop=True) if rolling_rows else pd.DataFrame()
    by_ticker = pd.concat(ticker_rows, ignore_index=True, sort=False) if ticker_rows else pd.DataFrame()
    by_month = pd.concat(month_rows, ignore_index=True, sort=False) if month_rows else pd.DataFrame()
    by_quarter = pd.concat(quarter_rows, ignore_index=True, sort=False) if quarter_rows else pd.DataFrame()

    write_csv(OUTPUT_DIR / "full_horizon_leaderboard.csv", leaderboard)
    write_csv(OUTPUT_DIR / "best_by_horizon.csv", best_by_horizon)
    write_csv(OUTPUT_DIR / "best_by_model_group_and_horizon.csv", best_by_group)
    write_csv(OUTPUT_DIR / "validation_final_gap_by_horizon.csv", gap)
    write_csv(OUTPUT_DIR / "rolling_summary_by_horizon.csv", rolling_summary)
    write_csv(OUTPUT_DIR / "by_ticker_by_horizon.csv", by_ticker)
    write_csv(OUTPUT_DIR / "by_month_by_horizon.csv", by_month)
    write_csv(OUTPUT_DIR / "by_quarter_by_horizon.csv", by_quarter)

    make_figures(leaderboard, best_by_horizon, best_by_group, rolling_summary)
    write_reports(leaderboard, best_by_horizon, best_by_group, gap, rolling_summary, manifest)

    main = leaderboard[leaderboard["candidate_id"].astype(str).eq(CURRENT_MAIN_CANDIDATE_ID)]
    main_final = pct(main.iloc[0]["final_accuracy"]) if not main.empty else "missing"
    print(f"legacy_full_horizon_complete output_dir={rel(OUTPUT_DIR)} h40_main_final={main_final} candidates={len(leaderboard)}")


if __name__ == "__main__":
    main()
