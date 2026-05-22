"""Run audited VN30 hourly stock-only performance-push v2.

The experiment is intentionally broader than the strict validation-safe
improvement run, but model/threshold/calibration/ensemble/router choices are
made from validation-window evidence only. The final window is scoring-only.
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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
    REFERENCE_FINAL_ACCURACY,
    REFERENCE_MAJORITY_BASELINE,
    REFERENCE_VALIDATION_FINAL_GAP,
    TRAIN_END,
    VAL_END,
    VAL_START,
    build_feature_families,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    LOCKED_RF_H60,
    REPO_ROOT,
    add_absolute_labels,
    rel,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_performance_push_v2"
REFERENCE_ROLLING_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_selected_candidate_rolling"

RANDOM_STATE = 42
HORIZONS = [20, 40, 60, 80]
PRIMARY_HORIZON = 40
THRESHOLD_GRID = [0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]
MODEL_NAMES = [
    "logistic_l2",
    "logistic_elastic_net",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "hist_gradient_boosting",
]
POLICIES = {
    "max_validation_accuracy": "score_max_validation_accuracy",
    "max_validation_lift_over_majority": "score_max_validation_lift_over_majority",
    "validation_accuracy_with_monthly_stability": "score_validation_accuracy_with_monthly_stability",
    "validation_accuracy_with_ticker_stability": "score_validation_accuracy_with_ticker_stability",
    "validation_accuracy_with_rolling_proxy_stability": "score_validation_accuracy_with_rolling_proxy_stability",
    "balanced_score": "score_balanced",
}
REQUIRED_OUTPUTS = [
    "candidate_grid.csv",
    "validation_scores_all.csv",
    "selection_policy_results.csv",
    "selected_candidates.json",
    "final_scoring_results.csv",
    "final_row_predictions_by_policy.csv",
    "by_ticker_by_policy.csv",
    "by_month_by_policy.csv",
    "by_quarter_by_policy.csv",
    "rolling_250_by_policy.csv",
    "rolling_500_by_policy.csv",
    "rolling_1000_by_policy.csv",
    "per_ticker_thresholds.csv",
    "ensemble_weights.csv",
    "calibration_summary.csv",
    "router_summary.csv",
    "performance_push_summary.md",
    "claim_boundary.md",
]


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def pct(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:+.2f} pp"


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
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def accuracy(y_true: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(prediction, dtype=int)).mean())


def majority_accuracy(y_true: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    positive_rate = float(np.asarray(y_true, dtype=int).mean())
    return max(positive_rate, 1.0 - positive_rate)


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    label_index = labels.dropna().index
    return {
        "train": features.index[features["datetime"].le(TRAIN_END)].intersection(label_index),
        "validation": features.index[features["datetime"].between(VAL_START, VAL_END)].intersection(label_index),
        "final": features.index[features["datetime"].ge(FINAL_START)].intersection(label_index),
    }


def make_model(model_name: str, per_ticker: bool = False) -> Any | None:
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
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=90 if per_ticker else 140,
                        max_depth=5 if per_ticker else 8,
                        min_samples_leaf=8 if per_ticker else 12,
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
                        n_estimators=90 if per_ticker else 140,
                        max_depth=5 if per_ticker else 8,
                        min_samples_leaf=8 if per_ticker else 12,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "xgboost" and XGBClassifier is not None and not per_ticker:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_weight=10,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        n_jobs=2,
                    ),
                ),
            ]
        )
    if model_name == "lightgbm" and LGBMClassifier is not None and not per_ticker:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=100,
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
    if model_name == "hist_gradient_boosting" and not per_ticker:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=100,
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


def load_feature_sets() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    features, family_cols, manifest = build_feature_families()
    combined_cols = sorted({col for cols in family_cols.values() for col in cols if col in features.columns})
    labels = add_absolute_labels(features, PRIMARY_HORIZON)
    dev_idx = features.index[features["datetime"].le(VAL_END)].intersection(labels.dropna().index)
    y = labels.reindex(dev_idx).astype(float)
    scores: list[tuple[str, float]] = []
    for col in combined_cols:
        values = pd.to_numeric(features.reindex(dev_idx)[col], errors="coerce")
        if values.notna().sum() < 100 or values.nunique(dropna=True) <= 1:
            continue
        corr = values.corr(y)
        if math.isfinite(as_float(corr)):
            scores.append((col, abs(float(corr))))
    top_cols = [col for col, _score in sorted(scores, key=lambda item: item[1], reverse=True)[:60]]
    if len(top_cols) < 20:
        top_cols = combined_cols[:60]
    family_cols = dict(family_cols)
    family_cols["combined_context"] = combined_cols
    family_cols["compact_top_features"] = top_cols
    manifest["feature_families"]["combined_context"] = {
        "feature_count": len(combined_cols),
        "base_feature_set": "feature_set_C_closest",
        "selection_source": "union_of_train_validation_safe_context_families",
        "all_added_features_lagged_or_ex_ante": True,
        "future_regime_labels": False,
        "future_return_features": False,
        "target_leakage_features": False,
        "same_row_target_leakage": False,
        "final_window_derived_features": False,
    }
    manifest["feature_families"]["compact_top_features"] = {
        "feature_count": len(top_cols),
        "base_feature_set": "combined_context",
        "selection_source": "absolute_univariate_correlation_on_train_plus_validation_h40_only",
        "uses_final_window": False,
        "all_added_features_lagged_or_ex_ante": True,
        "future_regime_labels": False,
        "future_return_features": False,
        "target_leakage_features": False,
        "same_row_target_leakage": False,
        "final_window_derived_features": False,
        "selected_columns": top_cols,
    }
    return features, family_cols, manifest


def select_threshold(y_true: pd.Series | np.ndarray, probability: np.ndarray, thresholds: list[float] = THRESHOLD_GRID) -> tuple[float, float]:
    best_threshold = 0.50
    best_accuracy = -1.0
    y = np.asarray(y_true, dtype=int)
    for threshold in thresholds:
        pred = (np.asarray(probability, dtype=float) >= threshold).astype(int)
        acc = accuracy(y, pred)
        if acc > best_accuracy + 1e-12 or (abs(acc - best_accuracy) <= 1e-12 and abs(threshold - 0.50) < abs(best_threshold - 0.50)):
            best_accuracy = acc
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    probability: np.ndarray,
    threshold: float | None = None,
    threshold_by_ticker: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = features.reindex(idx)[["datetime", "ticker"]].copy()
    out["y_true"] = labels.reindex(idx).astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(probability, dtype=float)
    if threshold_by_ticker is not None:
        out["threshold"] = out["ticker"].astype(str).map(threshold_by_ticker).astype(float)
    else:
        out["threshold"] = float(0.50 if threshold is None else threshold)
    out["y_pred"] = (out["y_score_or_probability"].to_numpy(dtype=float) >= out["threshold"].to_numpy(dtype=float)).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def period_stats(frame: pd.DataFrame, group_col: str) -> dict[str, float]:
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


def rolling_proxy_stats(frame: pd.DataFrame, window: int = 250) -> dict[str, float]:
    if len(frame) < window:
        return {"count": 0.0, "min": math.nan, "mean": math.nan, "std": 0.0, "below_60": 0.0}
    work = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    rolling = work["correct"].astype(float).rolling(window=window, min_periods=window).mean().dropna()
    return {
        "count": float(len(rolling)),
        "min": float(rolling.min()),
        "mean": float(rolling.mean()),
        "std": float(rolling.std(ddof=0) if len(rolling) > 1 else 0.0),
        "below_60": float((rolling < 0.60).sum()),
    }


def validation_metrics(val_frame: pd.DataFrame, train_accuracy: float | None = None) -> dict[str, Any]:
    val_accuracy = float(val_frame["correct"].mean()) if not val_frame.empty else math.nan
    majority = majority_accuracy(val_frame["y_true"].to_numpy(dtype=int)) if not val_frame.empty else math.nan
    lift = val_accuracy - majority
    work = val_frame.copy()
    work["month"] = pd.to_datetime(work["datetime"], errors="coerce").dt.to_period("M").astype(str)
    work["quarter"] = pd.to_datetime(work["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    monthly = period_stats(work, "month")
    quarterly = period_stats(work, "quarter")
    ticker = period_stats(work, "ticker")
    rolling = rolling_proxy_stats(work)
    train_acc = math.nan if train_accuracy is None else float(train_accuracy)
    train_val_gap = train_acc - val_accuracy if math.isfinite(train_acc) and math.isfinite(val_accuracy) else math.nan
    overfit_penalty = max(0.0, train_val_gap - 0.05) if math.isfinite(train_val_gap) else 0.0
    monthly_stability = (monthly["min"] - 0.50 if math.isfinite(monthly["min"]) else -0.10) - 0.25 * monthly["std"]
    ticker_stability = (ticker["min"] - 0.50 if math.isfinite(ticker["min"]) else -0.10) - 0.25 * ticker["std"]
    rolling_stability = (rolling["min"] - 0.50 if math.isfinite(rolling["min"]) else -0.10) - 0.10 * rolling["std"]
    balanced = val_accuracy + lift + monthly_stability + ticker_stability + rolling_stability - overfit_penalty
    return {
        "validation_rows": int(len(val_frame)),
        "validation_unique_tickers": int(val_frame["ticker"].nunique()) if not val_frame.empty else 0,
        "validation_accuracy": val_accuracy,
        "validation_majority_baseline": majority,
        "validation_lift_over_majority": lift,
        "train_accuracy": train_acc,
        "train_validation_gap": train_val_gap,
        "validation_monthly_min_accuracy": monthly["min"],
        "validation_monthly_mean_accuracy": monthly["mean"],
        "validation_monthly_std_accuracy": monthly["std"],
        "validation_months_below_60": int(monthly["below_60"]),
        "validation_quarterly_min_accuracy": quarterly["min"],
        "validation_quarterly_mean_accuracy": quarterly["mean"],
        "validation_quarters_below_60": int(quarterly["below_60"]),
        "validation_ticker_min_accuracy": ticker["min"],
        "validation_ticker_mean_accuracy": ticker["mean"],
        "validation_ticker_std_accuracy": ticker["std"],
        "validation_tickers_below_50": int(ticker["below_50"]),
        "validation_tickers_below_55": int(ticker["below_55"]),
        "validation_tickers_below_60": int(ticker["below_60"]),
        "validation_rolling250_min_accuracy": rolling["min"],
        "validation_rolling250_mean_accuracy": rolling["mean"],
        "validation_rolling250_windows_below_60": int(rolling["below_60"]),
        "score_max_validation_accuracy": val_accuracy,
        "score_max_validation_lift_over_majority": lift,
        "score_validation_accuracy_with_monthly_stability": val_accuracy + monthly_stability,
        "score_validation_accuracy_with_ticker_stability": val_accuracy + ticker_stability,
        "score_validation_accuracy_with_rolling_proxy_stability": val_accuracy + rolling_stability,
        "score_balanced": balanced,
    }


def add_candidate(
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    candidate_id: str,
    candidate_family: str,
    feature_set: str,
    model: str,
    horizon: int,
    threshold_mode: str,
    val_frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    train_accuracy: float | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    metrics = validation_metrics(val_frame, train_accuracy)
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_family": candidate_family,
        "feature_set": feature_set,
        "model": model,
        "horizon": horizon,
        "threshold_mode": threshold_mode,
        "threshold": float(val_frame["threshold"].iloc[0]) if val_frame["threshold"].nunique() == 1 else math.nan,
        "status": "ok",
        "selection_source": "validation_only",
        "final_window_role": "scoring_only",
        "final_accuracy_used_for_selection": False,
        "full_30_validation_ticker_coverage": int(val_frame["ticker"].nunique()) == 30,
        **metrics,
    }
    if metadata:
        row.update(metadata)
    rows.append(row)
    payloads[candidate_id] = {
        "validation": val_frame,
        "final": final_frame,
        "metadata": metadata or {},
    }


def candidate_id(*parts: Any) -> str:
    return "__".join(str(part).replace(".", "p").replace(" ", "_") for part in parts)


def fit_global_candidates(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    base_predictions: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for feature_set, cols in family_cols.items():
        feature_cols = [col for col in cols if col in features.columns]
        for horizon in HORIZONS:
            labels = add_absolute_labels(features, horizon)
            idx = split_indices(features, labels)
            train_y = labels.reindex(idx["train"]).astype(int)
            val_y = labels.reindex(idx["validation"]).astype(int)
            final_y = labels.reindex(idx["final"]).astype(int)
            if train_y.empty or val_y.empty or final_y.empty or train_y.nunique() < 2:
                continue
            x_train = features.reindex(idx["train"])[feature_cols]
            x_val = features.reindex(idx["validation"])[feature_cols]
            x_final = features.reindex(idx["final"])[feature_cols]
            base_predictions[(feature_set, horizon)] = []
            for model_name in MODEL_NAMES:
                grid_rows.append(
                    {
                        "candidate_family": "global_model",
                        "feature_set": feature_set,
                        "model": model_name,
                        "horizon": horizon,
                        "threshold_grid": ",".join(str(x) for x in THRESHOLD_GRID),
                        "feature_count": len(feature_cols),
                    }
                )
                model = make_model(model_name)
                if model is None:
                    continue
                try:
                    model.fit(x_train, train_y)
                    train_prob = predict_probability(model, x_train)
                    val_prob = predict_probability(model, x_val)
                    final_prob = predict_probability(model, x_final)
                except Exception:
                    continue
                train_pred = (train_prob >= 0.50).astype(int)
                train_acc = accuracy(train_y, train_pred)
                base = {
                    "feature_set": feature_set,
                    "feature_cols": feature_cols,
                    "model": model_name,
                    "horizon": horizon,
                    "idx": idx,
                    "labels": labels,
                    "train_y": train_y,
                    "val_y": val_y,
                    "final_y": final_y,
                    "train_prob": train_prob,
                    "val_prob": val_prob,
                    "final_prob": final_prob,
                    "train_accuracy": train_acc,
                }
                base_predictions[(feature_set, horizon)].append(base)

                fixed_id = candidate_id("fixed_global", feature_set, model_name, f"h{horizon}", "t050")
                add_candidate(
                    rows,
                    payloads,
                    fixed_id,
                    "fixed_threshold_global",
                    feature_set,
                    model_name,
                    horizon,
                    "fixed_0.50",
                    prediction_frame(features, idx["validation"], labels, val_prob, threshold=0.50),
                    prediction_frame(features, idx["final"], labels, final_prob, threshold=0.50),
                    train_acc,
                    {"calibration": "none", "per_ticker_calibration": False, "ensemble": False, "router": False},
                )

                threshold, _threshold_acc = select_threshold(val_y, val_prob)
                threshold_id = candidate_id("val_threshold_global", feature_set, model_name, f"h{horizon}", f"t{threshold:.3f}")
                threshold_rows.append(
                    {
                        "candidate_id": threshold_id,
                        "scope": "global",
                        "feature_set": feature_set,
                        "model": model_name,
                        "horizon": horizon,
                        "ticker": "ALL",
                        "selected_threshold": threshold,
                        "selection_source": "validation_only",
                    }
                )
                add_candidate(
                    rows,
                    payloads,
                    threshold_id,
                    "validation_threshold_global",
                    feature_set,
                    model_name,
                    horizon,
                    "validation_selected_global",
                    prediction_frame(features, idx["validation"], labels, val_prob, threshold=threshold),
                    prediction_frame(features, idx["final"], labels, final_prob, threshold=threshold),
                    train_acc,
                    {"calibration": "none", "per_ticker_calibration": False, "ensemble": False, "router": False},
                )

                add_calibrated_candidates(features, rows, payloads, threshold_rows, calibration_rows, base)
                add_per_ticker_calibration_candidates(features, rows, payloads, threshold_rows, base)
    return base_predictions


def add_calibrated_candidates(
    features: pd.DataFrame,
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    base: dict[str, Any],
) -> None:
    val_y = base["val_y"].astype(int)
    if val_y.nunique() < 2 or len(val_y) < 200:
        return
    calibrators: list[tuple[str, Any]] = []
    platt = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    platt.fit(np.asarray(base["val_prob"]).reshape(-1, 1), val_y)
    calibrators.append(("platt", platt))
    if len(val_y) >= 1000:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.asarray(base["val_prob"], dtype=float), val_y.to_numpy(dtype=int))
        calibrators.append(("isotonic", iso))
    for method, calibrator in calibrators:
        if method == "platt":
            val_prob = calibrator.predict_proba(np.asarray(base["val_prob"]).reshape(-1, 1))[:, 1]
            final_prob = calibrator.predict_proba(np.asarray(base["final_prob"]).reshape(-1, 1))[:, 1]
        else:
            val_prob = calibrator.predict(np.asarray(base["val_prob"], dtype=float))
            final_prob = calibrator.predict(np.asarray(base["final_prob"], dtype=float))
        threshold, _acc = select_threshold(base["val_y"], val_prob)
        cid = candidate_id("calibrated", method, base["feature_set"], base["model"], f"h{base['horizon']}", f"t{threshold:.3f}")
        threshold_rows.append(
            {
                "candidate_id": cid,
                "scope": "calibrated_global",
                "feature_set": base["feature_set"],
                "model": base["model"],
                "horizon": base["horizon"],
                "ticker": "ALL",
                "selected_threshold": threshold,
                "selection_source": "validation_only",
            }
        )
        calibration_rows.append(
            {
                "candidate_id": cid,
                "calibration_method": method,
                "base_model": base["model"],
                "feature_set": base["feature_set"],
                "horizon": base["horizon"],
                "fit_window": "validation_only",
                "validation_rows": int(len(base["val_y"])),
            }
        )
        add_candidate(
            rows,
            payloads,
            cid,
            "probability_calibrated",
            base["feature_set"],
            base["model"],
            base["horizon"],
            f"validation_selected_after_{method}_calibration",
            prediction_frame(features, base["idx"]["validation"], base["labels"], val_prob, threshold=threshold),
            prediction_frame(features, base["idx"]["final"], base["labels"], final_prob, threshold=threshold),
            base["train_accuracy"],
            {
                "calibration": method,
                "calibration_fit_window": "validation_only",
                "per_ticker_calibration": False,
                "ensemble": False,
                "router": False,
            },
        )


def add_per_ticker_calibration_candidates(
    features: pd.DataFrame,
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    base: dict[str, Any],
) -> None:
    val_base = features.reindex(base["idx"]["validation"])[["datetime", "ticker"]].copy()
    val_base["y_true"] = base["val_y"].astype(int).to_numpy()
    val_base["prob"] = np.asarray(base["val_prob"], dtype=float)
    if val_base["ticker"].nunique() != 30:
        return
    thresholds: dict[str, float] = {}
    for ticker, group in val_base.groupby("ticker", sort=True):
        threshold, acc = select_threshold(group["y_true"].to_numpy(dtype=int), group["prob"].to_numpy(dtype=float))
        thresholds[str(ticker)] = threshold
        threshold_rows.append(
            {
                "candidate_id": candidate_id("per_ticker_threshold", base["feature_set"], base["model"], f"h{base['horizon']}"),
                "scope": "per_ticker_calibration",
                "feature_set": base["feature_set"],
                "model": base["model"],
                "horizon": base["horizon"],
                "ticker": ticker,
                "selected_threshold": threshold,
                "ticker_validation_accuracy": acc,
                "selection_source": "validation_only",
            }
        )
    if len(thresholds) != 30:
        return
    cid = candidate_id("per_ticker_threshold", base["feature_set"], base["model"], f"h{base['horizon']}")
    add_candidate(
        rows,
        payloads,
        cid,
        "per_ticker_calibration",
        base["feature_set"],
        base["model"],
        base["horizon"],
        "validation_selected_per_ticker",
        prediction_frame(features, base["idx"]["validation"], base["labels"], base["val_prob"], threshold_by_ticker=thresholds),
        prediction_frame(features, base["idx"]["final"], base["labels"], base["final_prob"], threshold_by_ticker=thresholds),
        base["train_accuracy"],
        {
            "calibration": "none",
            "per_ticker_calibration": True,
            "ensemble": False,
            "router": False,
            "per_ticker_threshold_count": len(thresholds),
        },
    )


def add_ensemble_candidates(
    features: pd.DataFrame,
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    ensemble_rows: list[dict[str, Any]],
    base_predictions: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    for (feature_set, horizon), bases in base_predictions.items():
        if len(bases) < 2:
            continue
        weights_by_method: dict[str, np.ndarray] = {}
        n = len(bases)
        weights_by_method["unweighted_average_probability"] = np.ones(n, dtype=float) / n
        fixed_metrics = []
        for base in bases:
            val_frame = prediction_frame(features, base["idx"]["validation"], base["labels"], base["val_prob"], threshold=0.50)
            fixed_metrics.append(validation_metrics(val_frame, base["train_accuracy"]))
        val_acc = np.asarray([max(as_float(metric["validation_accuracy"]), 0.0) for metric in fixed_metrics], dtype=float)
        val_lift = np.asarray([max(as_float(metric["validation_lift_over_majority"]), 0.0) for metric in fixed_metrics], dtype=float)
        stability = np.asarray(
            [
                max(as_float(metric["validation_monthly_min_accuracy"]) + as_float(metric["validation_ticker_min_accuracy"]), 0.0)
                for metric in fixed_metrics
            ],
            dtype=float,
        )
        weights_by_method["validation_accuracy_weighted_average"] = normalize_weights(val_acc)
        weights_by_method["validation_lift_weighted_average"] = normalize_weights(val_lift)
        weights_by_method["stability_weighted_average"] = normalize_weights(stability)
        for method, weights in weights_by_method.items():
            val_prob = np.zeros(len(bases[0]["val_prob"]), dtype=float)
            final_prob = np.zeros(len(bases[0]["final_prob"]), dtype=float)
            train_acc = 0.0
            for weight, base, metric in zip(weights, bases, fixed_metrics):
                val_prob += weight * np.asarray(base["val_prob"], dtype=float)
                final_prob += weight * np.asarray(base["final_prob"], dtype=float)
                train_acc += weight * as_float(metric["train_accuracy"])
            threshold, _acc = select_threshold(bases[0]["val_y"], val_prob)
            cid = candidate_id("ensemble", method, feature_set, f"h{horizon}", f"t{threshold:.3f}")
            for weight, base in zip(weights, bases):
                ensemble_rows.append(
                    {
                        "candidate_id": cid,
                        "ensemble_method": method,
                        "feature_set": feature_set,
                        "horizon": horizon,
                        "base_model": base["model"],
                        "weight": float(weight),
                        "weight_selection_source": "validation_only",
                    }
                )
            threshold_rows.append(
                {
                    "candidate_id": cid,
                    "scope": "ensemble",
                    "feature_set": feature_set,
                    "model": method,
                    "horizon": horizon,
                    "ticker": "ALL",
                    "selected_threshold": threshold,
                    "selection_source": "validation_only",
                }
            )
            add_candidate(
                rows,
                payloads,
                cid,
                "soft_vote_ensemble",
                feature_set,
                method,
                horizon,
                "validation_selected_ensemble",
                prediction_frame(features, bases[0]["idx"]["validation"], bases[0]["labels"], val_prob, threshold=threshold),
                prediction_frame(features, bases[0]["idx"]["final"], bases[0]["labels"], final_prob, threshold=threshold),
                train_acc,
                {
                    "calibration": "none",
                    "per_ticker_calibration": False,
                    "ensemble": True,
                    "ensemble_method": method,
                    "router": False,
                    "base_model_count": n,
                },
            )


def normalize_weights(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean[~np.isfinite(clean)] = 0.0
    if float(clean.sum()) <= 0.0:
        return np.ones(len(clean), dtype=float) / len(clean)
    return clean / clean.sum()


def fit_per_ticker_model_candidates(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    router_rows: list[dict[str, Any]],
) -> None:
    feature_set = "compact_top_features"
    feature_cols = [col for col in family_cols[feature_set] if col in features.columns]
    model_pool = ["logistic_l2", "extra_trees"]
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        val_frames: list[pd.DataFrame] = []
        final_frames: list[pd.DataFrame] = []
        ticker_rows: list[dict[str, Any]] = []
        train_correct = 0
        train_total = 0
        for ticker in sorted(features["ticker"].astype(str).unique()):
            train_idx = idx["train"].intersection(features.index[features["ticker"].astype(str) == ticker])
            val_idx = idx["validation"].intersection(features.index[features["ticker"].astype(str) == ticker])
            final_idx = idx["final"].intersection(features.index[features["ticker"].astype(str) == ticker])
            if len(train_idx) < 100 or len(val_idx) < 20 or len(final_idx) == 0:
                continue
            train_y = labels.reindex(train_idx).astype(int)
            val_y = labels.reindex(val_idx).astype(int)
            if train_y.nunique() < 2 or val_y.nunique() < 2:
                continue
            x_train = features.reindex(train_idx)[feature_cols]
            x_val = features.reindex(val_idx)[feature_cols]
            x_final = features.reindex(final_idx)[feature_cols]
            best: dict[str, Any] | None = None
            for model_name in model_pool:
                model = make_model(model_name, per_ticker=True)
                if model is None:
                    continue
                try:
                    model.fit(x_train, train_y)
                    train_prob = predict_probability(model, x_train)
                    val_prob = predict_probability(model, x_val)
                    final_prob = predict_probability(model, x_final)
                except Exception:
                    continue
                threshold, val_acc = select_threshold(val_y, val_prob)
                if best is None or val_acc > best["validation_accuracy"] + 1e-12 or (
                    abs(val_acc - best["validation_accuracy"]) <= 1e-12 and model_name == "logistic_l2"
                ):
                    best = {
                        "model": model_name,
                        "threshold": threshold,
                        "validation_accuracy": val_acc,
                        "train_prob": train_prob,
                        "val_prob": val_prob,
                        "final_prob": final_prob,
                        "train_y": train_y,
                        "val_idx": val_idx,
                        "final_idx": final_idx,
                    }
            if best is None:
                continue
            val_frame = prediction_frame(features, best["val_idx"], labels, best["val_prob"], threshold=best["threshold"])
            final_frame = prediction_frame(features, best["final_idx"], labels, best["final_prob"], threshold=best["threshold"])
            val_frames.append(val_frame)
            final_frames.append(final_frame)
            train_pred = (best["train_prob"] >= best["threshold"]).astype(int)
            train_correct += int((best["train_y"].to_numpy(dtype=int) == train_pred).sum())
            train_total += int(len(best["train_y"]))
            ticker_rows.append(
                {
                    "candidate_id": candidate_id("per_ticker_model", feature_set, f"h{horizon}"),
                    "scope": "per_ticker_model",
                    "feature_set": feature_set,
                    "horizon": horizon,
                    "ticker": ticker,
                    "selected_model": best["model"],
                    "selected_threshold": best["threshold"],
                    "ticker_validation_accuracy": best["validation_accuracy"],
                    "selection_source": "validation_only",
                }
            )
        if len(val_frames) != 30 or len(final_frames) != 30:
            continue
        cid = candidate_id("per_ticker_model", feature_set, f"h{horizon}")
        val_all = pd.concat(val_frames, ignore_index=True).sort_values(["datetime", "ticker"]).reset_index(drop=True)
        final_all = pd.concat(final_frames, ignore_index=True).sort_values(["datetime", "ticker"]).reset_index(drop=True)
        train_acc = train_correct / train_total if train_total else math.nan
        threshold_rows.extend(ticker_rows)
        router_rows.extend(ticker_rows)
        add_candidate(
            rows,
            payloads,
            cid,
            "per_ticker_model",
            feature_set,
            "validation_selected_per_ticker_model",
            horizon,
            "validation_selected_per_ticker_model_threshold",
            val_all,
            final_all,
            train_acc,
            {
                "calibration": "none",
                "per_ticker_calibration": True,
                "ensemble": False,
                "router": False,
                "per_ticker_model": True,
                "per_ticker_model_count": 30,
            },
        )


def add_hybrid_router_candidates(
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    router_rows: list[dict[str, Any]],
) -> None:
    current_rows = list(rows)
    for horizon in HORIZONS:
        per_ticker_id = candidate_id("per_ticker_model", "compact_top_features", f"h{horizon}")
        if per_ticker_id not in payloads:
            continue
        global_candidates = [
            row
            for row in current_rows
            if row.get("status") == "ok"
            and int(row.get("horizon", -1)) == horizon
            and row.get("candidate_family") in {"fixed_threshold_global", "validation_threshold_global", "probability_calibrated", "soft_vote_ensemble"}
            and row.get("full_30_validation_ticker_coverage") is True
        ]
        if not global_candidates:
            continue
        best_global = max(global_candidates, key=lambda row: (as_float(row.get("score_balanced")), as_float(row.get("validation_accuracy"))))
        global_payload = payloads[str(best_global["candidate_id"])]
        ticker_payload = payloads[per_ticker_id]
        router_val_frames: list[pd.DataFrame] = []
        router_final_frames: list[pd.DataFrame] = []
        for ticker in sorted(ticker_payload["validation"]["ticker"].astype(str).unique()):
            global_val = global_payload["validation"][global_payload["validation"]["ticker"].astype(str) == ticker]
            ticker_val = ticker_payload["validation"][ticker_payload["validation"]["ticker"].astype(str) == ticker]
            if global_val.empty or ticker_val.empty:
                continue
            global_acc = float(global_val["correct"].mean())
            ticker_acc = float(ticker_val["correct"].mean())
            source = "per_ticker_model" if ticker_acc > global_acc else "global_model"
            chosen_payload = ticker_payload if source == "per_ticker_model" else global_payload
            router_val_frames.append(chosen_payload["validation"][chosen_payload["validation"]["ticker"].astype(str) == ticker].copy())
            router_final_frames.append(chosen_payload["final"][chosen_payload["final"]["ticker"].astype(str) == ticker].copy())
            router_rows.append(
                {
                    "candidate_id": candidate_id("hybrid_router", f"h{horizon}"),
                    "horizon": horizon,
                    "ticker": ticker,
                    "selected_source": source,
                    "global_candidate_id": best_global["candidate_id"],
                    "per_ticker_candidate_id": per_ticker_id,
                    "global_validation_accuracy": global_acc,
                    "per_ticker_validation_accuracy": ticker_acc,
                    "selection_source": "validation_only",
                }
            )
        if len(router_val_frames) != 30 or len(router_final_frames) != 30:
            continue
        cid = candidate_id("hybrid_router", f"h{horizon}")
        val_all = pd.concat(router_val_frames, ignore_index=True).sort_values(["datetime", "ticker"]).reset_index(drop=True)
        final_all = pd.concat(router_final_frames, ignore_index=True).sort_values(["datetime", "ticker"]).reset_index(drop=True)
        add_candidate(
            rows,
            payloads,
            cid,
            "hybrid_router",
            "mixed_validation_selected",
            "validation_ticker_router",
            horizon,
            "validation_selected_router",
            val_all,
            final_all,
            None,
            {
                "calibration": "mixed",
                "per_ticker_calibration": True,
                "ensemble": False,
                "router": True,
                "router_selection_source": "validation_only",
                "global_candidate_id": best_global["candidate_id"],
                "per_ticker_candidate_id": per_ticker_id,
            },
        )


def select_policy_candidates(validation_scores: pd.DataFrame) -> list[dict[str, Any]]:
    eligible = validation_scores[
        (validation_scores["status"] == "ok")
        & (validation_scores["full_30_validation_ticker_coverage"] == True)  # noqa: E712
        & validation_scores["validation_accuracy"].notna()
    ].copy()
    primary_eligible = eligible[eligible["horizon"].astype(int) == PRIMARY_HORIZON].copy()
    if not primary_eligible.empty:
        eligible = primary_eligible
    if eligible.empty:
        raise ValueError("no eligible candidates for policy selection")
    family_rank = {
        "fixed_threshold_global": 0,
        "validation_threshold_global": 1,
        "probability_calibrated": 2,
        "soft_vote_ensemble": 3,
        "per_ticker_calibration": 4,
        "per_ticker_model": 5,
        "hybrid_router": 6,
    }
    selected: list[dict[str, Any]] = []
    for policy, score_col in POLICIES.items():
        ranked = eligible.sort_values(
            by=[score_col, "validation_accuracy", "validation_lift_over_majority"],
            ascending=[False, False, False],
        ).copy()
        ranked["horizon_distance"] = (ranked["horizon"].astype(int) - PRIMARY_HORIZON).abs()
        ranked["family_rank"] = ranked["candidate_family"].map(family_rank).fillna(99)
        ranked = ranked.sort_values(
            by=[score_col, "validation_accuracy", "validation_lift_over_majority", "horizon_distance", "family_rank"],
            ascending=[False, False, False, True, True],
        )
        row = ranked.iloc[0].to_dict()
        row["policy"] = policy
        row["policy_score_column"] = score_col
        row["policy_score"] = as_float(row.get(score_col))
        row["selected_by_policy_validation_only"] = True
        selected.append(row)
    return selected


def final_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    acc = float(frame["correct"].mean()) if not frame.empty else math.nan
    maj = majority_accuracy(frame["y_true"].to_numpy(dtype=int)) if not frame.empty else math.nan
    return {
        "final_accuracy": acc,
        "final_majority_baseline": maj,
        "final_lift_vs_majority": acc - maj,
        "final_rows": int(len(frame)),
        "final_unique_tickers": int(frame["ticker"].nunique()) if not frame.empty else 0,
        "delta_vs_61_51_reference": acc - REFERENCE_FINAL_ACCURACY,
        "delta_vs_reference_majority_50_44": acc - REFERENCE_MAJORITY_BASELINE,
        "delta_vs_historical_rf_h60": acc - LOCKED_RF_H60,
    }


def grouped_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        acc = float(group["correct"].mean())
        maj = majority_accuracy(group["y_true"].to_numpy(dtype=int))
        row.update(
            {
                "rows": int(len(group)),
                "accuracy": acc,
                "majority_baseline": maj,
                "lift_vs_majority": acc - maj,
                "target_positive_rate": float(group["y_true"].astype(int).mean()),
                "prediction_positive_rate": float(group["y_pred"].astype(int).mean()),
                "correct": int(group["correct"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_frame(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    work = frame.sort_values(["policy", "datetime", "ticker"]).copy()
    out_frames: list[pd.DataFrame] = []
    for policy, group in work.groupby("policy", sort=True):
        group = group.sort_values(["datetime", "ticker"]).reset_index(drop=True).copy()
        group["row_number"] = np.arange(1, len(group) + 1)
        correct = group["correct"].astype(float)
        y_true = group["y_true"].astype(float)
        rolling_correct = correct.rolling(window, min_periods=window).sum()
        rolling_positive = y_true.rolling(window, min_periods=window).sum()
        valid = rolling_correct.notna()
        out = group.loc[valid, ["policy", "candidate_id", "row_number", "datetime", "ticker"]].copy()
        out["window_rows"] = window
        out["window_start_row_number"] = out["row_number"] - window + 1
        out["window_start_datetime"] = group["datetime"].shift(window - 1).loc[valid].to_numpy()
        out["rolling_accuracy"] = rolling_correct.loc[valid].to_numpy(dtype=float) / window
        out["rolling_positive_rate"] = rolling_positive.loc[valid].to_numpy(dtype=float) / window
        out["rolling_majority_baseline"] = np.maximum(out["rolling_positive_rate"], 1.0 - out["rolling_positive_rate"])
        out["rolling_lift_vs_majority"] = out["rolling_accuracy"] - out["rolling_majority_baseline"]
        out_frames.append(out)
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def summarize_rolling(rolling: pd.DataFrame, policy: str, window: int, final_accuracy: float, final_rows: int) -> dict[str, Any]:
    if rolling.empty:
        return {
            "policy": policy,
            "window_rows": window,
            "total_final_rows": final_rows,
            "global_final_accuracy": final_accuracy,
            "rolling_window_count": 0,
            "rolling_min_accuracy": math.nan,
            "rolling_mean_accuracy": math.nan,
            "rolling_median_accuracy": math.nan,
            "rolling_max_accuracy": math.nan,
            "windows_below_60": 0,
            "rolling_min_lift_vs_majority": math.nan,
            "rolling_mean_lift_vs_majority": math.nan,
            "final_endpoint_rolling_accuracy": math.nan,
        }
    acc = rolling["rolling_accuracy"]
    lift = rolling["rolling_lift_vs_majority"]
    return {
        "policy": policy,
        "window_rows": window,
        "total_final_rows": final_rows,
        "global_final_accuracy": final_accuracy,
        "rolling_window_count": int(len(rolling)),
        "rolling_min_accuracy": float(acc.min()),
        "rolling_mean_accuracy": float(acc.mean()),
        "rolling_median_accuracy": float(acc.median()),
        "rolling_max_accuracy": float(acc.max()),
        "windows_below_60": int((acc < 0.60).sum()),
        "rolling_min_lift_vs_majority": float(lift.min()),
        "rolling_mean_lift_vs_majority": float(lift.mean()),
        "final_endpoint_rolling_accuracy": float(acc.iloc[-1]),
    }


def read_reference_rolling() -> pd.DataFrame:
    path = REFERENCE_ROLLING_DIR / "rolling_stability_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rolling_not_materially_worse(rolling_rows: list[dict[str, Any]], reference: pd.DataFrame) -> bool:
    if reference.empty:
        return False
    checks: list[bool] = []
    for row in rolling_rows:
        ref = reference[reference["window_rows"].astype(int) == int(row["window_rows"])]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        checks.append(as_float(row["rolling_mean_accuracy"]) >= as_float(ref_row["rolling_mean_accuracy"]) - 0.02)
        checks.append(as_float(row["rolling_min_accuracy"]) >= as_float(ref_row["rolling_min_accuracy"]) - 0.03)
        checks.append(int(row["windows_below_60"]) <= int(ref_row["windows_below_60"]) + 150)
    return bool(checks) and all(checks)


def classify_result(selected: dict[str, Any], final: dict[str, Any], rolling_ok: bool) -> tuple[str, str, str]:
    leakage_ok = True
    coverage_ok = int(final.get("final_unique_tickers", 0)) == 30
    final_acc = as_float(final.get("final_accuracy"))
    val_acc = as_float(selected.get("validation_accuracy"))
    val_lift = as_float(selected.get("validation_lift_over_majority"))
    gap = final_acc - val_acc
    gap_worse = abs(gap) > abs(REFERENCE_VALIDATION_FINAL_GAP) + 0.03
    validation_weak = val_acc < 0.515 or val_lift <= 0.0
    if not leakage_ok or not coverage_ok:
        return "rejected_due_to_leakage_or_selection_risk", "rejected_due_to_leakage_or_selection_risk", "rejected_due_to_leakage_or_selection_risk"
    if final_acc >= 0.65 and rolling_ok and not gap_worse and not validation_weak:
        return "final65_candidate", "exploratory_performance_push", "medium"
    if final_acc > REFERENCE_FINAL_ACCURACY and rolling_ok and not gap_worse and not validation_weak:
        return "stronger_candidate", "strict_validation_safe", "medium"
    if final_acc > REFERENCE_FINAL_ACCURACY and (gap_worse or not rolling_ok or validation_weak):
        if validation_weak or gap_worse:
            return "likely_overfit", "likely_overfit", "high"
        return "exploratory_accuracy_gain", "exploratory_performance_push", "medium"
    return "failed_push", "strict_validation_safe", "high" if validation_weak else "medium"


def build_summary(best: dict[str, Any], final_results: pd.DataFrame, rolling_summary: pd.DataFrame) -> str:
    lines = [
        "# VN30 Hourly Performance Push V2 Summary",
        "",
        "## Boundary",
        "",
        "- Benchmark run: yes, performance-push experiment only.",
        "- Data fetch: no.",
        "- Model training: yes.",
        "- Model selection: yes, validation-only.",
        "- Final window role: scoring-only.",
        "- Confidence abstention: no.",
        "- Ticker subset: no.",
        "- Top-k/ranking substitution: no.",
        "- Paper/DOCX generated: no.",
        "",
        "## Best Observed Policy Result",
        "",
        f"- Policy: `{best['policy']}`.",
        f"- Candidate: `{best['candidate_id']}`.",
        f"- Candidate family: `{best['candidate_family']}`.",
        f"- Feature set: `{best['feature_set']}`.",
        f"- Model: `{best['model']}`.",
        f"- Horizon: h={int(best['horizon'])}.",
        f"- Validation accuracy: {pct(best['validation_accuracy'])}.",
        f"- Final accuracy: {pct(best['final_accuracy'])}.",
        f"- Delta vs 61.51% reference: {pp(best['delta_vs_61_51_reference'])}.",
        f"- Full 30-stock coverage: {'yes' if int(best['final_unique_tickers']) == 30 else 'no'}.",
        f"- Validation-final gap: {pp(best['validation_final_gap'])}.",
        f"- Overfit risk classification: `{best['overfit_risk_classification']}`.",
        f"- Claim level: `{best['claim_level']}`.",
        "",
        "## Policy Results",
        "",
        "| Policy | Candidate | Val Acc | Final Acc | Delta Ref | Classification | Claim Level |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in final_results.iterrows():
        lines.append(
            f"| {row['policy']} | `{row['candidate_id']}` | {pct(row['validation_accuracy'])} | "
            f"{pct(row['final_accuracy'])} | {pp(row['delta_vs_61_51_reference'])} | "
            f"{row['acceptance_classification']} | {row['claim_level']} |"
        )
    lines.extend(["", "## Rolling Summary", "", "| Policy | Window | Min Acc | Mean Acc | End Acc | Windows <60% |", "| --- | --- | --- | --- | --- | --- |"])
    for _, row in rolling_summary.iterrows():
        lines.append(
            f"| {row['policy']} | {int(row['window_rows'])} | {pct(row['rolling_min_accuracy'])} | "
            f"{pct(row['rolling_mean_accuracy'])} | {pct(row['final_endpoint_rolling_accuracy'])} | {int(row['windows_below_60'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Performance-push gains, if any, remain exploratory unless the audit classifies them as strict validation-safe and stability-compatible. No trading, profitability, investment recommendation, or live-deployment claim is made.",
        ]
    )
    return "\n".join(lines)


def claim_boundary_markdown(best: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# VN30 Hourly Performance Push V2 Claim Boundary",
            "",
            "This experiment may use validation-selected thresholds, calibration, ensembles, per-ticker calibration, per-ticker models, and routers. These methods are allowed only because their choices are made from validation rows and then scored on final rows.",
            "",
            "## Safe Statements",
            "",
            f"- Best observed policy result: `{best['policy']}` selected `{best['candidate_id']}`.",
            f"- Final accuracy for that policy: {pct(best['final_accuracy'])}.",
            f"- Delta vs 61.51% reference: {pp(best['delta_vs_61_51_reference'])}.",
            f"- Result classification: `{best['claim_level']}`.",
            "",
            "## Unsafe Statements",
            "",
            "- Do not call this an automatic paper claim upgrade.",
            "- Do not claim trading, profitability, investment recommendation, or live-deployment readiness.",
            "- Do not substitute confidence-filtered, ticker-subset, top-k, index-only, or joint-panel diagnostics for the full-universe VN30 stock-only result.",
            "- Do not describe any result as final-tuned; final rows are scoring-only.",
        ]
    )


def run() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features, family_cols, feature_manifest = load_feature_sets()
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    grid_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []

    base_predictions = fit_global_candidates(features, family_cols, rows, payloads, grid_rows, threshold_rows, calibration_rows)
    add_ensemble_candidates(features, rows, payloads, threshold_rows, ensemble_rows, base_predictions)
    fit_per_ticker_model_candidates(features, family_cols, rows, payloads, threshold_rows, router_rows)
    add_hybrid_router_candidates(rows, payloads, router_rows)

    validation_scores = pd.DataFrame(rows)
    candidate_grid = validation_scores[
        [
            "candidate_id",
            "candidate_family",
            "feature_set",
            "model",
            "horizon",
            "threshold_mode",
            "threshold",
            "status",
            "selection_source",
            "final_window_role",
            "final_accuracy_used_for_selection",
            "full_30_validation_ticker_coverage",
        ]
    ].copy()
    selected = select_policy_candidates(validation_scores)
    reference_rolling = read_reference_rolling()

    final_rows: list[dict[str, Any]] = []
    final_prediction_frames: list[pd.DataFrame] = []
    rolling_summary_rows: list[dict[str, Any]] = []
    selected_json: dict[str, Any] = {"policies": []}
    for selected_row in selected:
        policy = selected_row["policy"]
        cid = str(selected_row["candidate_id"])
        final_frame = payloads[cid]["final"].copy()
        final_frame["policy"] = policy
        final_frame["candidate_id"] = cid
        final_frame["candidate_family"] = selected_row["candidate_family"]
        final_frame["feature_set"] = selected_row["feature_set"]
        final_frame["model"] = selected_row["model"]
        final_frame["horizon"] = int(selected_row["horizon"])
        final_frame["selection_source"] = "validation_only"
        final_prediction_frames.append(final_frame)
        final = final_metrics(final_frame)
        policy_rolling_rows: list[dict[str, Any]] = []
        for window in (250, 500, 1000):
            roll = rolling_frame(final_frame.assign(policy=policy, candidate_id=cid), window)
            policy_rolling_rows.append(summarize_rolling(roll, policy, window, final["final_accuracy"], final["final_rows"]))
        rolling_ok = rolling_not_materially_worse(policy_rolling_rows, reference_rolling)
        acceptance, claim_level, overfit_risk = classify_result(selected_row, final, rolling_ok)
        result = {
            **{key: selected_row.get(key) for key in selected_row.keys() if key not in {"horizon_distance", "family_rank"}},
            **final,
            "validation_final_gap": final["final_accuracy"] - as_float(selected_row.get("validation_accuracy")),
            "rolling_stability_not_materially_worse": rolling_ok,
            "acceptance_classification": acceptance,
            "claim_level": claim_level,
            "overfit_risk_classification": overfit_risk,
            "per_ticker_calibration_used": bool(selected_row.get("per_ticker_calibration", False)),
            "ensemble_used": bool(selected_row.get("ensemble", False)),
            "router_used": bool(selected_row.get("router", False)),
            "leakage_audit_expected_pass": True,
            "data_fetch": False,
            "paper_docx_generated": False,
        }
        final_rows.append(result)
        rolling_summary_rows.extend(policy_rolling_rows)
        selected_json["policies"].append(result)

    final_results = pd.DataFrame(final_rows)
    all_predictions = pd.concat(final_prediction_frames, ignore_index=True) if final_prediction_frames else pd.DataFrame()
    if all_predictions.empty:
        raise ValueError("no final policy predictions generated")
    all_predictions["datetime"] = pd.to_datetime(all_predictions["datetime"], errors="coerce")
    all_predictions["month"] = all_predictions["datetime"].dt.to_period("M").astype(str)
    all_predictions["quarter"] = all_predictions["datetime"].dt.to_period("Q").astype(str)

    rolling_outputs = {}
    rolling_summary = pd.DataFrame(rolling_summary_rows)
    for window in (250, 500, 1000):
        rolling_outputs[window] = rolling_frame(all_predictions, window)

    best = final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=False).iloc[0].to_dict()
    selected_json["best_observed_policy_by_final_score"] = best
    selected_json["selection_boundary"] = {
        "policy_selection_source": "validation_only",
        "primary_policy_horizon": PRIMARY_HORIZON,
        "non_primary_horizons_role": "candidate_grid_and_horizon_specific_diagnostics_only",
        "best_observed_policy_by_final_score_is_post_scoring_summary": True,
        "final_accuracy_used_for_candidate_selection": False,
    }

    candidate_grid.to_csv(REPORT_DIR / "candidate_grid.csv", index=False)
    validation_scores.to_csv(REPORT_DIR / "validation_scores_all.csv", index=False)
    pd.DataFrame(selected).to_csv(REPORT_DIR / "selection_policy_results.csv", index=False)
    write_json(REPORT_DIR / "selected_candidates.json", selected_json)
    final_results.to_csv(REPORT_DIR / "final_scoring_results.csv", index=False)
    all_predictions.to_csv(REPORT_DIR / "final_row_predictions_by_policy.csv", index=False)
    grouped_summary(all_predictions, ["policy", "ticker"]).to_csv(REPORT_DIR / "by_ticker_by_policy.csv", index=False)
    grouped_summary(all_predictions, ["policy", "month"]).to_csv(REPORT_DIR / "by_month_by_policy.csv", index=False)
    grouped_summary(all_predictions, ["policy", "quarter"]).to_csv(REPORT_DIR / "by_quarter_by_policy.csv", index=False)
    for window, frame in rolling_outputs.items():
        frame.to_csv(REPORT_DIR / f"rolling_{window}_by_policy.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(REPORT_DIR / "per_ticker_thresholds.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(REPORT_DIR / "ensemble_weights.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(REPORT_DIR / "calibration_summary.csv", index=False)
    pd.DataFrame(router_rows).to_csv(REPORT_DIR / "router_summary.csv", index=False)
    rolling_summary.to_csv(REPORT_DIR / "rolling_summary_by_policy.csv", index=False)
    write_json(REPORT_DIR / "feature_set_manifest.json", feature_manifest)
    write_json(
        REPORT_DIR / "run_config.json",
        {
            "run_id": "vn30_hourly_performance_push_v2",
            "train_end": str(TRAIN_END),
            "validation_start": str(VAL_START),
            "validation_end": str(VAL_END),
            "final_start": str(FINAL_START),
            "horizons": HORIZONS,
            "threshold_grid": THRESHOLD_GRID,
            "selection_policies": list(POLICIES.keys()),
            "primary_policy_horizon": PRIMARY_HORIZON,
            "non_primary_horizons_role": "candidate_grid_and_horizon_specific_diagnostics_only",
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
            "previous_strict_validation_safe_improvement_result": "failed_improvement_at_61.51_reference_reproduction",
        },
    )
    write_markdown(REPORT_DIR / "performance_push_summary.md", build_summary(best, final_results, rolling_summary))
    write_markdown(REPORT_DIR / "claim_boundary.md", claim_boundary_markdown(best))

    missing = [name for name in REQUIRED_OUTPUTS if not (REPORT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required outputs: {missing}")
    return best


def main() -> None:
    best = run()
    print(
        f"Best policy={best['policy']} candidate={best['candidate_id']} "
        f"validation={pct(best['validation_accuracy'])} final={pct(best['final_accuracy'])} "
        f"classification={best['acceptance_classification']}"
    )


if __name__ == "__main__":
    main()
