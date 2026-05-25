"""Run fair validation-only VN30 model-zoo tuning.

This runner uses existing local VN30 benchmark data only. It tunes every
feasible model family under a documented budget, selects candidates by
validation-only objectives, and treats the final window as scoring-only.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research import run_vn30_comprehensive_model_universe_benchmark as universe  # noqa: E402
from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    FINAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, active_stock_tickers, add_absolute_labels, rel  # noqa: E402

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "fair_tuning"
FIGURE_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42
PRIMARY_HORIZON = 40
SECONDARY_HORIZONS = [20, 60, 80]
FEATURE_FAMILIES = [
    "baseline_C_closest",
    "volatility_normalized",
    "relative_strength",
    "regime_context",
    "combined_context",
]
THRESHOLD_GRID = [0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]
CURRENT_MAIN_FINAL_ACCURACY = 0.6163475699558174
CURRENT_MAIN_LABEL = "Logistic L2, baseline_C_closest, h40, threshold 0.55, final accuracy 61.63%"
PRIOR_DESCRIPTIVE_LABEL = "bull_bear_sideway_router h40 fixed 0.50, final accuracy 63.33%"

NAIVE_MODELS = [
    "majority_class",
    "random_walk_direction",
    "previous_direction",
    "persistence_rule",
    "moving_average_rule",
    "rolling_momentum_rule",
    "volatility_adjusted_momentum_rule",
]
TECHNICAL_MODELS = [
    "sma_crossover",
    "ema_crossover",
    "macd_rule",
    "rsi_rule",
    "bollinger_band_rule",
    "price_momentum_rule",
    "volume_momentum_rule",
    "mean_reversion_rule",
    "breakout_rule",
]
LINEAR_MODELS = [
    "logistic_l2",
    "logistic_l1",
    "logistic_elastic_net",
    "ridge_classifier",
    "lda",
    "qda",
    "passive_aggressive",
    "perceptron",
    "sgd_hinge",
    "sgd_log_loss",
]
SVM_MODELS = ["linear_svm", "svm_rbf", "svm_poly", "calibrated_svm"]
DISTANCE_MODELS = ["knn", "radius_neighbors", "nearest_centroid"]
PROBABILISTIC_MODELS = ["gaussian_naive_bayes", "bernoulli_naive_bayes", "complement_naive_bayes"]
TREE_MODELS = ["decision_tree", "random_forest", "extra_trees"]
BOOSTING_MODELS = [
    "adaboost",
    "sklearn_gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
]
NEURAL_MODELS = ["mlp_classifier", "lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"]
ENSEMBLE_MODELS = [
    "hard_voting",
    "soft_voting",
    "validation_weighted_soft_vote",
    "stacking_logistic_meta",
    "stacking_lightgbm_meta",
    "stacking_xgboost_meta",
]
CALIBRATION_MODELS = [
    "platt_logistic",
    "isotonic_logistic",
    "calibrated_random_forest",
    "calibrated_xgboost",
    "calibrated_lightgbm",
]
REGIME_MODELS = [
    "regime_context_logistic",
    "regime_context_xgboost",
    "regime_context_lightgbm",
    "bull_bear_sideway_router",
    "high_low_volatility_router",
    "regime_threshold_router",
    "regime_model_router",
    "conservative_router_fallback",
]
STATISTICAL_MODELS = [
    "arima_direction",
    "sarima_direction",
    "ets_direction",
    "var_direction",
    "garch_volatility_diagnostic",
]

MODEL_GROUPS: dict[str, list[str]] = {
    "naive_baseline": NAIVE_MODELS,
    "technical_rules": TECHNICAL_MODELS,
    "linear_models": LINEAR_MODELS,
    "svm_and_kernel_models": SVM_MODELS,
    "distance_based_models": DISTANCE_MODELS,
    "probabilistic_models": PROBABILISTIC_MODELS,
    "tree_models": TREE_MODELS,
    "boosting_models": BOOSTING_MODELS,
    "neural_deep_models": NEURAL_MODELS,
    "ensemble_stacking_models": ENSEMBLE_MODELS,
    "calibration_variants": CALIBRATION_MODELS,
    "regime_aware_models": REGIME_MODELS,
    "statistical_models": STATISTICAL_MODELS,
}
MODEL_TO_GROUP = {model_id: group for group, models in MODEL_GROUPS.items() for model_id in models}
INTERPRETABILITY_SCORE = {
    "naive_baseline": 5,
    "technical_rules": 5,
    "linear_models": 4,
    "svm_and_kernel_models": 2,
    "distance_based_models": 3,
    "probabilistic_models": 4,
    "tree_models": 3,
    "boosting_models": 2,
    "neural_deep_models": 1,
    "ensemble_stacking_models": 1,
    "calibration_variants": 3,
    "regime_aware_models": 3,
    "statistical_models": 4,
}
ENSEMBLE_BASE_MODELS = [
    "logistic_l2",
    "logistic_l1",
    "linear_svm",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "gaussian_naive_bayes",
]


@dataclass(frozen=True)
class FitConfig:
    config_key: str
    model_group: str
    model_id: str
    config_name: str
    params: dict[str, Any]
    feature_families: tuple[str, ...]
    factory: Callable[[pd.Series, dict[str, Any]], Any]
    scaling_required: str
    runtime_risk: str
    hyperparameter_space: str


@dataclass
class ScoreCache:
    candidate_id: str
    model_group: str
    model_id: str
    feature_family: str
    horizon: int
    threshold_policy: str
    threshold: float
    validation_index: pd.Index
    final_index: pd.Index
    validation_score: np.ndarray
    final_score: np.ndarray


@dataclass
class BasePrediction:
    model_id: str
    config_key: str
    config_name: str
    candidate_id: str
    validation_score: np.ndarray
    final_score: np.ndarray
    validation_accuracy: float


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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    work = frame.head(max_rows).copy()
    headers = list(work.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def dependency_status(model_id: str) -> str:
    if model_id in {"xgboost", "calibrated_xgboost", "regime_context_xgboost", "stacking_xgboost_meta"}:
        return "available" if XGBClassifier is not None else "missing:xgboost"
    if model_id in {"lightgbm", "calibrated_lightgbm", "regime_context_lightgbm", "stacking_lightgbm_meta"}:
        return "available" if LGBMClassifier is not None else "missing:lightgbm"
    if model_id == "catboost":
        return "available" if CatBoostClassifier is not None else "missing:catboost"
    if model_id in {"lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"}:
        return "available" if universe.torch is not None else "missing:torch"
    if model_id == "garch_volatility_diagnostic":
        return "available" if importlib.util.find_spec("arch") is not None else "missing:arch"
    if model_id in {"arima_direction", "sarima_direction", "ets_direction", "var_direction"}:
        return "available" if universe.ARIMA is not None else "missing:statsmodels"
    return "available"


def candidate_id(*parts: Any) -> str:
    return universe.candidate_id("fair", *parts)


def threshold_specs(y_true: pd.Series | np.ndarray, score: np.ndarray) -> list[tuple[str, float]]:
    specs = [("fixed_0.50", 0.50)]
    best_threshold = 0.50
    best_accuracy = -1.0
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(score, dtype=float)
    for threshold in THRESHOLD_GRID:
        pred = (scores >= threshold).astype(int)
        acc = float((pred == y).mean()) if len(y) else math.nan
        if math.isfinite(acc) and (
            acc > best_accuracy + 1e-12
            or (abs(acc - best_accuracy) <= 1e-12 and abs(threshold - 0.50) < abs(best_threshold - 0.50))
        ):
            best_accuracy = acc
            best_threshold = float(threshold)
    specs.append(("validation_selected_threshold", best_threshold))
    return specs


def majority_accuracy(y_true: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(y) == 0:
        return math.nan
    rate = float(y.mean())
    return max(rate, 1.0 - rate)


def balanced_accuracy(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred, dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    return float(balanced_accuracy_score(y, p))


def rolling_values(frame: pd.DataFrame, window: int) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    ordered = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    return ordered["correct"].astype(float).rolling(window=window, min_periods=window).mean().dropna()


def period_stability(frame: pd.DataFrame, freq: str) -> tuple[float, float, float]:
    if frame.empty:
        return math.nan, math.nan, math.nan
    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    grouped = work.dropna(subset=["datetime"]).groupby(work["datetime"].dt.to_period(freq))["correct"].mean()
    if grouped.empty:
        return math.nan, math.nan, math.nan
    return float(grouped.mean()), float(grouped.min()), float(grouped.std(ddof=0))


def group_stability(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
    if frame.empty or column not in frame.columns:
        return math.nan, math.nan, math.nan
    grouped = frame.groupby(column)["correct"].mean()
    if grouped.empty:
        return math.nan, math.nan, math.nan
    return float(grouped.mean()), float(grouped.min()), float(grouped.std(ddof=0))


def validation_metric_overlay(validation_frame: pd.DataFrame) -> dict[str, Any]:
    if validation_frame.empty:
        return {
            "validation_balanced_accuracy": math.nan,
            "validation_majority_accuracy": math.nan,
            "validation_lift_over_majority": math.nan,
            "validation_rolling_250_mean": math.nan,
            "validation_rolling_500_mean": math.nan,
            "validation_rolling_1000_mean": math.nan,
            "validation_rolling_stability": math.nan,
            "validation_instability": math.nan,
            "validation_monthly_stability": math.nan,
            "validation_ticker_stability": math.nan,
            "balanced_robust_score": math.nan,
        }
    y = validation_frame["y_true"].astype(int).to_numpy()
    pred = validation_frame["y_pred"].astype(int).to_numpy()
    val_acc = float(validation_frame["correct"].mean())
    val_bal = balanced_accuracy(y, pred)
    maj = majority_accuracy(y)
    lift = val_acc - maj if math.isfinite(maj) else math.nan
    rolling_means = []
    rolling_stds = []
    out: dict[str, Any] = {}
    for window in (250, 500, 1000):
        roll = rolling_values(validation_frame, window)
        mean_value = float(roll.mean()) if not roll.empty else math.nan
        std_value = float(roll.std(ddof=0)) if not roll.empty else math.nan
        out[f"validation_rolling_{window}_mean"] = mean_value
        out[f"validation_rolling_{window}_min"] = float(roll.min()) if not roll.empty else math.nan
        out[f"validation_rolling_{window}_windows_below_60"] = int((roll < 0.60).sum()) if not roll.empty else 0
        if math.isfinite(mean_value):
            rolling_means.append(mean_value)
        if math.isfinite(std_value):
            rolling_stds.append(std_value)
    monthly_mean, monthly_min, monthly_std = period_stability(validation_frame, "M")
    quarterly_mean, quarterly_min, quarterly_std = period_stability(validation_frame, "Q")
    ticker_mean, ticker_min, ticker_std = group_stability(validation_frame, "ticker")
    rolling_stability = float(np.nanmean(rolling_means)) if rolling_means else val_acc
    instability_parts = [value for value in [*rolling_stds, monthly_std, quarterly_std, ticker_std] if math.isfinite(value)]
    instability = float(np.nanmean(instability_parts)) if instability_parts else math.nan
    monthly_stability = monthly_min if math.isfinite(monthly_min) else monthly_mean
    ticker_stability = ticker_min if math.isfinite(ticker_min) else ticker_mean
    robust_score = (
        0.20 * val_acc
        + 0.20 * (val_bal if math.isfinite(val_bal) else val_acc)
        + 0.15 * ((0.50 + lift) if math.isfinite(lift) else 0.50)
        + 0.15 * rolling_stability
        + 0.10 * (monthly_stability if math.isfinite(monthly_stability) else val_acc)
        + 0.10 * (ticker_stability if math.isfinite(ticker_stability) else val_acc)
        - 0.10 * (instability if math.isfinite(instability) else 0.0)
    )
    out.update(
        {
            "validation_balanced_accuracy": val_bal,
            "validation_majority_accuracy": maj,
            "validation_lift_over_majority": lift,
            "validation_monthly_mean_accuracy": monthly_mean,
            "validation_monthly_min_accuracy": monthly_min,
            "validation_monthly_std_accuracy": monthly_std,
            "validation_quarterly_mean_accuracy": quarterly_mean,
            "validation_quarterly_min_accuracy": quarterly_min,
            "validation_quarterly_std_accuracy": quarterly_std,
            "validation_ticker_mean_accuracy": ticker_mean,
            "validation_ticker_min_accuracy": ticker_min,
            "validation_ticker_std_accuracy": ticker_std,
            "validation_rolling_stability": rolling_stability,
            "validation_instability": instability,
            "validation_monthly_stability": monthly_stability,
            "validation_ticker_stability": ticker_stability,
            "balanced_robust_score": robust_score,
        }
    )
    return out


def classify_overfit(row: pd.Series) -> tuple[str, str]:
    validation = as_float(row.get("validation_accuracy"))
    final = as_float(row.get("final_accuracy"))
    gap = validation - final if math.isfinite(validation) and math.isfinite(final) else math.nan
    rolling = as_float(row.get("rolling_250_mean"))
    monthly_min = as_float(row.get("monthly_min_accuracy"))
    quarterly_min = as_float(row.get("quarterly_min_accuracy"))
    ticker_min = as_float(row.get("ticker_min_accuracy"))
    below = as_float(row.get("rolling_250_windows_below_60"))
    reasons: list[str] = []
    if math.isfinite(gap) and gap > 0.05:
        reasons.append(f"validation-final gap {gap * 100.0:.2f} pp")
    if math.isfinite(rolling) and rolling < 0.56:
        reasons.append(f"rolling 250 mean {rolling * 100.0:.2f}%")
    if math.isfinite(below) and int(below) > 0:
        reasons.append(f"{int(below)} rolling 250 windows below 60%")
    if math.isfinite(monthly_min) and monthly_min < 0.55:
        reasons.append(f"monthly minimum {monthly_min * 100.0:.2f}%")
    if math.isfinite(quarterly_min) and quarterly_min < 0.55:
        reasons.append(f"quarterly minimum {quarterly_min * 100.0:.2f}%")
    if math.isfinite(ticker_min) and ticker_min < 0.55:
        reasons.append(f"ticker minimum {ticker_min * 100.0:.2f}%")
    if (math.isfinite(gap) and gap > 0.05) or (math.isfinite(rolling) and rolling < 0.52):
        return "high", "; ".join(reasons) or "large validation-final deterioration"
    if reasons:
        return "medium", "; ".join(reasons)
    return "low", "validation-only metrics and final stability do not show a major warning"


def make_result_row(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    validation_index: pd.Index,
    final_index: pd.Index,
    validation_score: np.ndarray,
    final_score: np.ndarray,
    model_group: str,
    model_id: str,
    feature_family: str,
    horizon: int,
    threshold_policy: str,
    threshold: float,
    candidate: str,
    train_rows: int,
    feature_count: int,
    config_name: str,
    hyperparameters: dict[str, Any],
    selection_stage: str,
    fit_runtime_seconds: float,
    implementation_note: str,
    sequence_length: int | str = "",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    validation_score = np.asarray(validation_score, dtype=float)
    final_score = np.asarray(final_score, dtype=float)
    validation_pred = (validation_score >= threshold).astype(int)
    final_pred = (final_score >= threshold).astype(int)
    validation_frame = universe.prediction_frame(
        features,
        validation_index,
        labels,
        validation_score,
        validation_pred,
        model_group=model_group,
        model_id=model_id,
        feature_family=feature_family,
        horizon=horizon,
        threshold_policy=threshold_policy,
        threshold=threshold,
        candidate=candidate,
        split="validation",
    )
    final_frame = universe.prediction_frame(
        features,
        final_index,
        labels,
        final_score,
        final_pred,
        model_group=model_group,
        model_id=model_id,
        feature_family=feature_family,
        horizon=horizon,
        threshold_policy=threshold_policy,
        threshold=threshold,
        candidate=candidate,
        split="final",
    )
    row = universe.result_row(
        candidate=candidate,
        model_group=model_group,
        model_id=model_id,
        feature_family=feature_family,
        horizon=horizon,
        threshold_policy=threshold_policy,
        threshold=threshold,
        validation_frame=validation_frame,
        final_frame=final_frame,
        train_rows=train_rows,
        feature_count=feature_count,
        reason_not_claim_eligible="not selected by validation-only objectives",
        sequence_length=sequence_length,
        implementation_note=implementation_note,
    )
    row.update(validation_metric_overlay(validation_frame))
    row.update(
        {
            "config_name": config_name,
            "hyperparameters": json.dumps(json_safe(hyperparameters), sort_keys=True),
            "selection_stage": selection_stage,
            "fit_runtime_seconds": fit_runtime_seconds,
            "primary_horizon_policy": "primary_h40" if horizon == PRIMARY_HORIZON else "secondary_validation_diagnostic",
            "interpretability_score": INTERPRETABILITY_SCORE.get(model_group, 1),
            "prior_final_score_privileged": False,
            "prior_63_33_context_only": True,
            "current_61_63_context_only": True,
        }
    )
    risk, reason = classify_overfit(pd.Series(row))
    row["overfit_risk"] = risk
    row["overfit_risk_reason"] = reason
    return row, validation_frame, final_frame


def add_grid_row(
    candidate_rows: list[dict[str, Any]],
    *,
    candidate: str,
    model_group: str,
    model_id: str,
    feature_family: str,
    horizon: int,
    threshold_policy: str,
    planned_status: str,
    config_name: str,
    hyperparameters: dict[str, Any] | None = None,
    selection_stage: str = "primary_h40_full_zoo",
    fit_runtime_seconds: float = math.nan,
    reason: str = "",
) -> None:
    candidate_rows.append(
        {
            "candidate_id": candidate,
            "model_group": model_group,
            "model_id": model_id,
            "feature_family": feature_family,
            "horizon": int(horizon),
            "threshold_policy": threshold_policy,
            "planned_status": planned_status,
            "config_name": config_name,
            "hyperparameters": json.dumps(json_safe(hyperparameters or {}), sort_keys=True),
            "selection_stage": selection_stage,
            "fit_runtime_seconds": fit_runtime_seconds,
            "reason": reason,
            "final_accuracy_used_for_selection": False,
            "ticker_subset": False,
            "confidence_abstention": False,
            "topk_substitution": False,
        }
    )


def add_evaluation(
    *,
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_cache: dict[str, ScoreCache],
    features: pd.DataFrame,
    labels: pd.Series,
    validation_index: pd.Index,
    final_index: pd.Index,
    validation_score: np.ndarray,
    final_score: np.ndarray,
    model_group: str,
    model_id: str,
    feature_family: str,
    horizon: int,
    threshold_policy: str,
    threshold: float,
    train_rows: int,
    feature_count: int,
    config_name: str,
    hyperparameters: dict[str, Any],
    selection_stage: str,
    fit_runtime_seconds: float,
    implementation_note: str,
    sequence_length: int | str = "",
) -> dict[str, Any]:
    cid = candidate_id(model_group, model_id, feature_family, f"h{horizon}", config_name, threshold_policy, f"t{threshold:.3f}")
    add_grid_row(
        candidate_rows,
        candidate=cid,
        model_group=model_group,
        model_id=model_id,
        feature_family=feature_family,
        horizon=horizon,
        threshold_policy=threshold_policy,
        planned_status="run",
        config_name=config_name,
        hyperparameters=hyperparameters,
        selection_stage=selection_stage,
        fit_runtime_seconds=fit_runtime_seconds,
    )
    row, _val_frame, _final_frame = make_result_row(
        features=features,
        labels=labels,
        validation_index=validation_index,
        final_index=final_index,
        validation_score=validation_score,
        final_score=final_score,
        model_group=model_group,
        model_id=model_id,
        feature_family=feature_family,
        horizon=horizon,
        threshold_policy=threshold_policy,
        threshold=threshold,
        candidate=cid,
        train_rows=train_rows,
        feature_count=feature_count,
        config_name=config_name,
        hyperparameters=hyperparameters,
        selection_stage=selection_stage,
        fit_runtime_seconds=fit_runtime_seconds,
        implementation_note=implementation_note,
        sequence_length=sequence_length,
    )
    result_rows.append(row)
    score_cache[cid] = ScoreCache(
        candidate_id=cid,
        model_group=model_group,
        model_id=model_id,
        feature_family=feature_family,
        horizon=horizon,
        threshold_policy=threshold_policy,
        threshold=threshold,
        validation_index=validation_index,
        final_index=final_index,
        validation_score=np.asarray(validation_score, dtype=np.float32),
        final_score=np.asarray(final_score, dtype=np.float32),
    )
    return row


def estimator_logistic_l2(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    solver="liblinear",
                    penalty="l2",
                    C=float(params.get("C", 0.3)),
                    class_weight=params.get("class_weight", "balanced"),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_logistic_l1(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    solver="liblinear",
                    penalty="l1",
                    C=float(params.get("C", 0.2)),
                    class_weight=params.get("class_weight", "balanced"),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_logistic_elastic(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    solver="saga",
                    penalty="elasticnet",
                    C=float(params.get("C", 0.3)),
                    l1_ratio=float(params.get("l1_ratio", 0.2)),
                    class_weight=params.get("class_weight", "balanced"),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_ridge(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeClassifier(alpha=float(params.get("alpha", 1.0)), class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def estimator_lda(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    shrinkage = params.get("shrinkage", "auto")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)),
        ]
    )


def estimator_qda(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", QuadraticDiscriminantAnalysis(reg_param=float(params.get("reg_param", 0.2)))),
        ]
    )


def estimator_passive_aggressive(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                PassiveAggressiveClassifier(
                    C=float(params.get("C", 0.2)),
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_perceptron(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                Perceptron(
                    penalty=params.get("penalty", "l2"),
                    alpha=float(params.get("alpha", 0.0001)),
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_sgd(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SGDClassifier(
                    loss=params.get("loss", "hinge"),
                    penalty=params.get("penalty", "l2"),
                    alpha=float(params.get("alpha", 0.0005)),
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_linear_svm(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=float(params.get("C", 0.3)),
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=5000,
                    dual="auto",
                ),
            ),
        ]
    )


def estimator_svm(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    kernel = params.get("kernel", "rbf")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel=kernel,
                    C=float(params.get("C", 1.0)),
                    gamma=params.get("gamma", "scale"),
                    degree=int(params.get("degree", 3)),
                    coef0=float(params.get("coef0", 1.0)),
                    class_weight="balanced",
                    probability=False,
                    cache_size=1600,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_knn(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=int(params.get("n_neighbors", 31)),
                    weights=params.get("weights", "distance"),
                    metric=params.get("metric", "minkowski"),
                    n_jobs=-1,
                ),
            ),
        ]
    )


def estimator_radius_neighbors(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    majority = int(float(np.asarray(y_train, dtype=int).mean()) >= 0.5) if len(y_train) else 1
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RadiusNeighborsClassifier(
                    radius=float(params.get("radius", 12.0)),
                    weights=params.get("weights", "distance"),
                    outlier_label=majority,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def estimator_nearest_centroid(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", NearestCentroid())])


def estimator_gaussian_nb(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GaussianNB(var_smoothing=float(params.get("var_smoothing", 1e-9)))),
        ]
    )


def estimator_bernoulli_nb(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", BernoulliNB(alpha=float(params.get("alpha", 1.0)), binarize=float(params.get("binarize", 0.0)))),
        ]
    )


def estimator_complement_nb(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("minmax", MinMaxScaler()),
            ("model", ComplementNB(alpha=float(params.get("alpha", 1.0)))),
        ]
    )


def estimator_decision_tree(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=params.get("max_depth"),
                    min_samples_leaf=int(params.get("min_samples_leaf", 25)),
                    class_weight=params.get("class_weight", "balanced"),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_random_forest(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=int(params.get("n_estimators", 120)),
                    max_depth=params.get("max_depth", 8),
                    min_samples_leaf=int(params.get("min_samples_leaf", 10)),
                    max_features=params.get("max_features", "sqrt"),
                    class_weight=params.get("class_weight", "balanced"),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def estimator_extra_trees(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=int(params.get("n_estimators", 120)),
                    max_depth=params.get("max_depth", 8),
                    min_samples_leaf=int(params.get("min_samples_leaf", 10)),
                    max_features=params.get("max_features", "sqrt"),
                    class_weight=params.get("class_weight", "balanced"),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def estimator_adaboost(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", AdaBoostClassifier(n_estimators=int(params.get("n_estimators", 80)), learning_rate=float(params.get("learning_rate", 0.05)), random_state=RANDOM_STATE)),
        ]
    )


def estimator_gradient_boosting(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=int(params.get("n_estimators", 80)),
                    max_depth=int(params.get("max_depth", 2)),
                    learning_rate=float(params.get("learning_rate", 0.04)),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_hist_gradient(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=int(params.get("max_iter", 90)),
                    max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                    learning_rate=float(params.get("learning_rate", 0.04)),
                    l2_regularization=float(params.get("l2_regularization", 0.1)),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_xgboost(y_train: pd.Series, params: dict[str, Any]) -> Pipeline | None:
    if XGBClassifier is None:
        return None
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                XGBClassifier(
                    n_estimators=int(params.get("n_estimators", 90)),
                    max_depth=int(params.get("max_depth", 3)),
                    learning_rate=float(params.get("learning_rate", 0.05)),
                    min_child_weight=float(params.get("min_child_weight", 8)),
                    subsample=float(params.get("subsample", 0.85)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.85)),
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=2,
                ),
            ),
        ]
    )


def estimator_lightgbm(y_train: pd.Series, params: dict[str, Any]) -> Pipeline | None:
    if LGBMClassifier is None:
        return None
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=int(params.get("n_estimators", 90)),
                    num_leaves=int(params.get("num_leaves", 15)),
                    max_depth=int(params.get("max_depth", 3)),
                    learning_rate=float(params.get("learning_rate", 0.05)),
                    min_child_samples=int(params.get("min_child_samples", 35)),
                    subsample=float(params.get("subsample", 0.85)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.85)),
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    n_jobs=2,
                ),
            ),
        ]
    )


def estimator_catboost(y_train: pd.Series, params: dict[str, Any]) -> Pipeline | None:
    if CatBoostClassifier is None:
        return None
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                CatBoostClassifier(
                    iterations=int(params.get("iterations", 90)),
                    depth=int(params.get("depth", 4)),
                    learning_rate=float(params.get("learning_rate", 0.05)),
                    loss_function="Logloss",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


def estimator_mlp(y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    hidden = tuple(params.get("hidden_layer_sizes", (48, 16)))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    alpha=float(params.get("alpha", 0.001)),
                    learning_rate_init=float(params.get("learning_rate_init", 0.001)),
                    max_iter=int(params.get("max_iter", 160)),
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_fit_configs() -> list[FitConfig]:
    configs: list[FitConfig] = []

    def add(
        model_group: str,
        model_id: str,
        config_name: str,
        params: dict[str, Any],
        feature_families: tuple[str, ...],
        factory: Callable[[pd.Series, dict[str, Any]], Any],
        scaling_required: str,
        runtime_risk: str,
        hyperparameter_space: str,
    ) -> None:
        configs.append(
            FitConfig(
                config_key=f"{model_id}:{config_name}",
                model_group=model_group,
                model_id=model_id,
                config_name=config_name,
                params=params,
                feature_families=feature_families,
                factory=factory,
                scaling_required=scaling_required,
                runtime_risk=runtime_risk,
                hyperparameter_space=hyperparameter_space,
            )
        )

    all_families = tuple(FEATURE_FAMILIES)
    three_families = ("baseline_C_closest", "volatility_normalized", "combined_context")
    two_families = ("baseline_C_closest", "combined_context")
    svm_families = ("baseline_C_closest", "volatility_normalized")

    for c_value in (0.1, 0.3, 1.0):
        add("linear_models", "logistic_l2", f"C{c_value:g}_balanced", {"C": c_value, "class_weight": "balanced"}, all_families, estimator_logistic_l2, "pipeline_train_only", "low", "C={0.1,0.3,1.0}; class_weight=balanced")
    for c_value in (0.1, 0.3):
        add("linear_models", "logistic_l1", f"C{c_value:g}_balanced", {"C": c_value, "class_weight": "balanced"}, all_families, estimator_logistic_l1, "pipeline_train_only", "low", "C={0.1,0.3}; class_weight=balanced")
    for ratio in (0.2, 0.5):
        add("linear_models", "logistic_elastic_net", f"C0p3_l1r{ratio:g}", {"C": 0.3, "l1_ratio": ratio}, all_families, estimator_logistic_elastic, "pipeline_train_only", "medium", "C=0.3; l1_ratio={0.2,0.5}; class_weight=balanced")
    for alpha in (0.5, 1.0, 2.0):
        add("linear_models", "ridge_classifier", f"alpha{alpha:g}", {"alpha": alpha}, all_families, estimator_ridge, "pipeline_train_only", "low", "alpha={0.5,1.0,2.0}")
    for shrinkage in ("auto", 0.2):
        add("linear_models", "lda", f"shrinkage{str(shrinkage).replace('.', 'p')}", {"shrinkage": shrinkage}, three_families, estimator_lda, "pipeline_train_only", "low", "solver=lsqr; shrinkage={auto,0.2}")
    for reg_param in (0.1, 0.3):
        add("linear_models", "qda", f"reg{reg_param:g}", {"reg_param": reg_param}, two_families, estimator_qda, "pipeline_train_only", "medium", "reg_param={0.1,0.3}")
    for c_value in (0.1, 0.3):
        add("linear_models", "passive_aggressive", f"C{c_value:g}", {"C": c_value}, all_families, estimator_passive_aggressive, "pipeline_train_only", "low", "C={0.1,0.3}")
    for penalty in ("l2", "l1"):
        add("linear_models", "perceptron", f"penalty_{penalty}", {"penalty": penalty, "alpha": 0.0001}, all_families, estimator_perceptron, "pipeline_train_only", "low", "penalty={l2,l1}; alpha=0.0001")
    for loss in ("hinge", "log_loss"):
        for alpha in (0.0001, 0.0005):
            model_id = "sgd_hinge" if loss == "hinge" else "sgd_log_loss"
            add("linear_models", model_id, f"alpha{alpha:g}", {"loss": loss, "alpha": alpha, "penalty": "l2"}, all_families, estimator_sgd, "pipeline_train_only", "low", "loss-specific; alpha={0.0001,0.0005}; penalty=l2")

    for c_value in (0.1, 0.3, 1.0):
        add("svm_and_kernel_models", "linear_svm", f"C{c_value:g}", {"C": c_value}, svm_families, estimator_linear_svm, "pipeline_train_only", "medium", "C={0.1,0.3,1.0}")
    for c_value, gamma in ((0.5, "scale"), (1.0, "scale"), (1.0, "auto")):
        add("svm_and_kernel_models", "svm_rbf", f"C{c_value:g}_g{gamma}", {"kernel": "rbf", "C": c_value, "gamma": gamma}, ("baseline_C_closest",), estimator_svm, "pipeline_train_only", "high", "C={0.5,1.0}; gamma={scale,auto}")
    for c_value, degree in ((0.3, 2), (0.5, 3)):
        add("svm_and_kernel_models", "svm_poly", f"C{c_value:g}_d{degree}", {"kernel": "poly", "C": c_value, "degree": degree, "gamma": "scale"}, ("baseline_C_closest",), estimator_svm, "pipeline_train_only", "high", "C={0.3,0.5}; degree={2,3}; gamma=scale")

    for n_neighbors, weights in ((15, "uniform"), (31, "distance"), (51, "distance")):
        add("distance_based_models", "knn", f"k{n_neighbors}_{weights}", {"n_neighbors": n_neighbors, "weights": weights}, three_families, estimator_knn, "pipeline_train_only", "medium", "k={15,31,51}; weights={uniform,distance}")
    for radius in (8.0, 12.0):
        add("distance_based_models", "radius_neighbors", f"r{radius:g}", {"radius": radius}, ("baseline_C_closest", "volatility_normalized"), estimator_radius_neighbors, "pipeline_train_only", "medium", "radius={8,12}; weights=distance")
    add("distance_based_models", "nearest_centroid", "default", {}, three_families, estimator_nearest_centroid, "pipeline_train_only", "low", "default nearest centroid")

    for smoothing in (1e-9, 1e-8, 1e-7):
        add("probabilistic_models", "gaussian_naive_bayes", f"var{smoothing:g}", {"var_smoothing": smoothing}, three_families, estimator_gaussian_nb, "not_required", "low", "var_smoothing={1e-9,1e-8,1e-7}")
    for alpha in (0.5, 1.0):
        add("probabilistic_models", "bernoulli_naive_bayes", f"alpha{alpha:g}", {"alpha": alpha, "binarize": 0.0}, three_families, estimator_bernoulli_nb, "not_required", "low", "alpha={0.5,1.0}; binarize=0")
        add("probabilistic_models", "complement_naive_bayes", f"alpha{alpha:g}", {"alpha": alpha}, three_families, estimator_complement_nb, "minmax_train_only", "low", "alpha={0.5,1.0}; MinMax train-only transform")

    for depth, leaf in ((4, 20), (6, 25), (8, 35), (None, 50)):
        add("tree_models", "decision_tree", f"d{depth}_leaf{leaf}", {"max_depth": depth, "min_samples_leaf": leaf}, three_families, estimator_decision_tree, "not_required", "low", "max_depth={4,6,8,None}; min_samples_leaf={20,25,35,50}")
    for n_estimators, depth, leaf in ((80, 6, 10), (120, 8, 10), (160, None, 25)):
        add("tree_models", "random_forest", f"n{n_estimators}_d{depth}_leaf{leaf}", {"n_estimators": n_estimators, "max_depth": depth, "min_samples_leaf": leaf, "max_features": "sqrt"}, three_families, estimator_random_forest, "not_required", "medium", "n_estimators={80,120,160}; max_depth={6,8,None}; min_samples_leaf={10,25}")
        add("tree_models", "extra_trees", f"n{n_estimators}_d{depth}_leaf{leaf}", {"n_estimators": n_estimators, "max_depth": depth, "min_samples_leaf": leaf, "max_features": "sqrt"}, three_families, estimator_extra_trees, "not_required", "medium", "n_estimators={80,120,160}; max_depth={6,8,None}; min_samples_leaf={10,25}")

    for n_estimators, learning_rate in ((80, 0.05), (120, 0.03)):
        add("boosting_models", "adaboost", f"n{n_estimators}_lr{learning_rate:g}", {"n_estimators": n_estimators, "learning_rate": learning_rate}, three_families, estimator_adaboost, "not_required", "medium", "n_estimators={80,120}; learning_rate={0.05,0.03}")
        add("boosting_models", "sklearn_gradient_boosting", f"n{n_estimators}_lr{learning_rate:g}", {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": 2}, three_families, estimator_gradient_boosting, "not_required", "medium", "n_estimators={80,120}; learning_rate={0.05,0.03}; max_depth=2")
    for max_iter, leaves, lr in ((80, 15, 0.05), (120, 31, 0.03)):
        add("boosting_models", "hist_gradient_boosting", f"i{max_iter}_leaf{leaves}_lr{lr:g}", {"max_iter": max_iter, "max_leaf_nodes": leaves, "learning_rate": lr}, three_families, estimator_hist_gradient, "not_required", "medium", "max_iter={80,120}; max_leaf_nodes={15,31}; learning_rate={0.05,0.03}")
    for n_estimators, depth, lr in ((80, 3, 0.05), (120, 2, 0.03)):
        add("boosting_models", "xgboost", f"n{n_estimators}_d{depth}_lr{lr:g}", {"n_estimators": n_estimators, "max_depth": depth, "learning_rate": lr, "subsample": 0.85, "colsample_bytree": 0.85}, three_families, estimator_xgboost, "not_required", "medium", "n_estimators={80,120}; max_depth={3,2}; learning_rate={0.05,0.03}; subsample=0.85; colsample=0.85")
        add("boosting_models", "lightgbm", f"n{n_estimators}_d{depth}_lr{lr:g}", {"n_estimators": n_estimators, "max_depth": depth, "num_leaves": 15, "learning_rate": lr, "subsample": 0.85, "colsample_bytree": 0.85}, three_families, estimator_lightgbm, "not_required", "medium", "n_estimators={80,120}; num_leaves=15; learning_rate={0.05,0.03}; subsample=0.85; colsample=0.85")
        add("boosting_models", "catboost", f"n{n_estimators}_d{depth}_lr{lr:g}", {"iterations": n_estimators, "depth": depth + 1, "learning_rate": lr}, two_families, estimator_catboost, "not_required", "medium", "iterations={80,120}; depth={4,3}; learning_rate={0.05,0.03}")

    for hidden, alpha, lr in (((48, 16), 0.001, 0.001), ((64, 32), 0.001, 0.0007), ((32,), 0.0005, 0.001)):
        name = "h" + "x".join(str(item) for item in hidden) + f"_a{alpha:g}_lr{lr:g}"
        add("neural_deep_models", "mlp_classifier", name, {"hidden_layer_sizes": hidden, "alpha": alpha, "learning_rate_init": lr, "max_iter": 160}, two_families, estimator_mlp, "pipeline_train_only", "medium", "hidden_layer_sizes={(48,16),(64,32),(32,)}; alpha={0.001,0.0005}; learning_rate_init={0.001,0.0007}")

    return configs


def score_technical_rule(features: pd.DataFrame, model_id: str, params: dict[str, Any], majority: int) -> pd.Series:
    close = pd.to_numeric(features["close"], errors="coerce")
    high = pd.to_numeric(features.get("high", close), errors="coerce")
    low = pd.to_numeric(features.get("low", close), errors="coerce")
    volume = pd.to_numeric(features.get("volume", pd.Series(1.0, index=features.index)), errors="coerce")
    by_ticker = features.groupby("ticker", sort=False)
    close_by_ticker = close.groupby(features["ticker"], sort=False)
    if model_id == "sma_crossover":
        short = int(params.get("short_window", 10))
        long = int(params.get("long_window", 40))
        sma_short = close_by_ticker.transform(lambda values: values.rolling(short, min_periods=max(3, short // 2)).mean())
        sma_long = close_by_ticker.transform(lambda values: values.rolling(long, min_periods=max(5, long // 2)).mean())
        raw = (sma_short / sma_long.replace(0.0, np.nan) - 1.0) * 40.0
    elif model_id == "ema_crossover":
        short = int(params.get("short_window", 12))
        long = int(params.get("long_window", 36))
        ema_short = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").ewm(span=short, adjust=False, min_periods=max(3, short // 2)).mean())
        ema_long = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").ewm(span=long, adjust=False, min_periods=max(5, long // 2)).mean())
        raw = (ema_short / ema_long.replace(0.0, np.nan) - 1.0) * 40.0
    elif model_id == "macd_rule":
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal = int(params.get("signal", 9))
        ema_fast = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").ewm(span=fast, adjust=False, min_periods=max(3, fast // 2)).mean())
        ema_slow = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").ewm(span=slow, adjust=False, min_periods=max(5, slow // 2)).mean())
        macd = ema_fast - ema_slow
        sig = macd.groupby(features["ticker"], sort=False).transform(lambda values: values.ewm(span=signal, adjust=False, min_periods=max(3, signal // 2)).mean())
        raw = (macd - sig) / close.replace(0.0, np.nan) * 100.0
    elif model_id == "rsi_rule":
        window = int(params.get("window", 14))
        delta = close_by_ticker.diff()
        gain = delta.clip(lower=0.0).groupby(features["ticker"], sort=False).transform(lambda values: values.rolling(window, min_periods=max(3, window // 2)).mean())
        loss = (-delta.clip(upper=0.0)).groupby(features["ticker"], sort=False).transform(lambda values: values.rolling(window, min_periods=max(3, window // 2)).mean())
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        raw = (50.0 - rsi) / 10.0
    elif model_id == "bollinger_band_rule":
        window = int(params.get("window", 20))
        k = float(params.get("std_k", 2.0))
        ma = close_by_ticker.transform(lambda values: values.rolling(window, min_periods=max(5, window // 2)).mean())
        sd = close_by_ticker.transform(lambda values: values.rolling(window, min_periods=max(5, window // 2)).std()).replace(0.0, np.nan)
        z = (close - ma) / (k * sd)
        raw = -z
    elif model_id == "price_momentum_rule":
        lookback = int(params.get("lookback", 20))
        lag = close_by_ticker.shift(lookback)
        raw = (close / lag.replace(0.0, np.nan) - 1.0) * 20.0
    elif model_id == "volume_momentum_rule":
        lookback = int(params.get("lookback", 5))
        vol_window = int(params.get("volume_window", 20))
        ret = close / close_by_ticker.shift(lookback).replace(0.0, np.nan) - 1.0
        vol_ma = volume.groupby(features["ticker"], sort=False).transform(lambda values: values.rolling(vol_window, min_periods=max(5, vol_window // 2)).mean())
        vol_ratio = volume / vol_ma.replace(0.0, np.nan) - 1.0
        raw = ret * (1.0 + vol_ratio.clip(lower=-0.5, upper=2.0).fillna(0.0)) * 15.0
    elif model_id == "mean_reversion_rule":
        window = int(params.get("window", 20))
        ma = close_by_ticker.transform(lambda values: values.rolling(window, min_periods=max(5, window // 2)).mean())
        sd = close_by_ticker.transform(lambda values: values.rolling(window, min_periods=max(5, window // 2)).std()).replace(0.0, np.nan)
        raw = (ma - close) / sd
    elif model_id == "breakout_rule":
        window = int(params.get("window", 20))
        rolling_high = high.groupby(features["ticker"], sort=False).transform(lambda values: values.rolling(window, min_periods=max(5, window // 2)).max().shift(1))
        rolling_low = low.groupby(features["ticker"], sort=False).transform(lambda values: values.rolling(window, min_periods=max(5, window // 2)).min().shift(1))
        raw = pd.Series(0.0, index=features.index)
        raw.loc[close > rolling_high] = 1.0
        raw.loc[close < rolling_low] = -1.0
        raw = raw * 4.0
    else:
        score = universe.score_series_rule(features, add_absolute_labels(features, PRIMARY_HORIZON), PRIMARY_HORIZON, model_id, majority)
        return pd.Series(score, index=features.index).fillna(float(majority)).clip(0.0, 1.0)
    return pd.Series(universe.sigmoid(pd.to_numeric(raw, errors="coerce").fillna(0.0).to_numpy()), index=features.index).fillna(float(majority)).clip(0.0, 1.0)


def technical_param_grid(model_id: str) -> list[dict[str, Any]]:
    grids = {
        "sma_crossover": [{"short_window": 5, "long_window": 20}, {"short_window": 10, "long_window": 40}, {"short_window": 20, "long_window": 60}],
        "ema_crossover": [{"short_window": 8, "long_window": 24}, {"short_window": 12, "long_window": 36}, {"short_window": 20, "long_window": 60}],
        "macd_rule": [{"fast": 8, "slow": 21, "signal": 5}, {"fast": 12, "slow": 26, "signal": 9}, {"fast": 16, "slow": 39, "signal": 9}],
        "rsi_rule": [{"window": 10}, {"window": 14}, {"window": 21}],
        "bollinger_band_rule": [{"window": 15, "std_k": 2.0}, {"window": 20, "std_k": 2.0}, {"window": 30, "std_k": 2.5}],
        "price_momentum_rule": [{"lookback": 5}, {"lookback": 20}, {"lookback": 60}],
        "volume_momentum_rule": [{"lookback": 5, "volume_window": 20}, {"lookback": 10, "volume_window": 30}, {"lookback": 20, "volume_window": 60}],
        "mean_reversion_rule": [{"window": 10}, {"window": 20}, {"window": 40}],
        "breakout_rule": [{"window": 10}, {"window": 20}, {"window": 40}],
    }
    return grids[model_id]


def fit_predict_estimator(
    *,
    features: pd.DataFrame,
    labels_by_horizon: dict[int, pd.Series],
    split_by_horizon: dict[int, dict[str, pd.Index]],
    family_cols: dict[str, list[str]],
    config: FitConfig,
    feature_family: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index, int, int, float]:
    labels = labels_by_horizon[horizon]
    idx = split_by_horizon[horizon]
    train_y = labels.loc[idx["train"]].astype(int)
    val_index = idx["validation"]
    final_index = idx["final"]
    cols = family_cols.get(feature_family, [])
    if not cols:
        raise RuntimeError(f"no numeric feature columns for {feature_family}")
    if train_y.empty or train_y.nunique() < 2:
        raise RuntimeError("invalid train labels")
    model = config.factory(train_y, config.params)
    if model is None:
        raise RuntimeError(f"dependency unavailable for {config.model_id}")
    start = time.perf_counter()
    model.fit(features.loc[idx["train"], cols], train_y)
    val_score = np.clip(universe.predict_score(model, features.loc[val_index, cols]), 0.0, 1.0)
    final_score = np.clip(universe.predict_score(model, features.loc[final_index, cols]), 0.0, 1.0)
    runtime = time.perf_counter() - start
    return val_score, final_score, val_index, final_index, len(train_y), len(cols), runtime


def run_rule_families(
    features: pd.DataFrame,
    labels_by_horizon: dict[int, pd.Series],
    split_by_horizon: dict[int, dict[str, pd.Index]],
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_cache: dict[str, ScoreCache],
    failures: list[dict[str, Any]],
) -> None:
    horizon = PRIMARY_HORIZON
    labels = labels_by_horizon[horizon]
    idx = split_by_horizon[horizon]
    train_y = labels.loc[idx["train"]].astype(int)
    majority = int(float(train_y.mean()) >= 0.5)
    val_y = labels.loc[idx["validation"]].astype(int)
    for model_id in NAIVE_MODELS:
        model_group = "naive_baseline"
        params: dict[str, Any] = {"fixed_rule": True}
        config_name = "fixed_rule"
        start = time.perf_counter()
        try:
            score_series = pd.Series(universe.score_series_rule(features, labels, horizon, model_id, majority), index=features.index)
            score_series = pd.to_numeric(score_series, errors="coerce").reindex(features.index).fillna(float(majority)).clip(0.0, 1.0)
            runtime = time.perf_counter() - start
            add_evaluation(
                candidate_rows=candidate_rows,
                result_rows=result_rows,
                score_cache=score_cache,
                features=features,
                labels=labels,
                validation_index=idx["validation"],
                final_index=idx["final"],
                validation_score=score_series.loc[idx["validation"]].to_numpy(dtype=float),
                final_score=score_series.loc[idx["final"]].to_numpy(dtype=float),
                model_group=model_group,
                model_id=model_id,
                feature_family="ex_ante_rule",
                horizon=horizon,
                threshold_policy="fixed_0.50",
                threshold=0.50,
                train_rows=len(train_y),
                feature_count=0,
                config_name=config_name,
                hyperparameters=params,
                selection_stage="primary_h40_full_zoo",
                fit_runtime_seconds=runtime,
                implementation_note="fixed ex-ante naive baseline; no tuning beyond documented fixed rule",
            )
        except Exception as exc:
            failures.append({"model_group": model_group, "model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
            add_grid_row(candidate_rows, candidate=candidate_id(model_group, model_id, "failed"), model_group=model_group, model_id=model_id, feature_family="ex_ante_rule", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=config_name, hyperparameters=params, reason=str(exc)[:500])

    for model_id in TECHNICAL_MODELS:
        model_group = "technical_rules"
        for params in technical_param_grid(model_id):
            config_name = "_".join(f"{key}{value}" for key, value in params.items()).replace(".", "p")
            start = time.perf_counter()
            try:
                score_series = score_technical_rule(features, model_id, params, majority)
                runtime = time.perf_counter() - start
                for threshold_policy, threshold in threshold_specs(val_y, score_series.loc[idx["validation"]].to_numpy(dtype=float)):
                    add_evaluation(
                        candidate_rows=candidate_rows,
                        result_rows=result_rows,
                        score_cache=score_cache,
                        features=features,
                        labels=labels,
                        validation_index=idx["validation"],
                        final_index=idx["final"],
                        validation_score=score_series.loc[idx["validation"]].to_numpy(dtype=float),
                        final_score=score_series.loc[idx["final"]].to_numpy(dtype=float),
                        model_group=model_group,
                        model_id=model_id,
                        feature_family="ex_ante_rule",
                        horizon=horizon,
                        threshold_policy=threshold_policy,
                        threshold=threshold,
                        train_rows=len(train_y),
                        feature_count=0,
                        config_name=config_name,
                        hyperparameters=params,
                        selection_stage="primary_h40_full_zoo",
                        fit_runtime_seconds=runtime,
                        implementation_note="technical rule window tuned by validation-only threshold/objective protocol",
                    )
            except Exception as exc:
                failures.append({"model_group": model_group, "model_id": model_id, "horizon": horizon, "config_name": config_name, "reason": str(exc)[:500]})
                add_grid_row(candidate_rows, candidate=candidate_id(model_group, model_id, config_name, "failed"), model_group=model_group, model_id=model_id, feature_family="ex_ante_rule", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=config_name, hyperparameters=params, reason=str(exc)[:500])


def run_classifier_configs(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    labels_by_horizon: dict[int, pd.Series],
    split_by_horizon: dict[int, dict[str, pd.Index]],
    fit_configs: list[FitConfig],
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_cache: dict[str, ScoreCache],
    failures: list[dict[str, Any]],
) -> dict[str, list[BasePrediction]]:
    base_pool: dict[str, list[BasePrediction]] = {model_id: [] for model_id in ENSEMBLE_BASE_MODELS}
    horizon = PRIMARY_HORIZON
    labels = labels_by_horizon[horizon]
    val_y = labels.loc[split_by_horizon[horizon]["validation"]].astype(int)
    for config in fit_configs:
        if dependency_status(config.model_id).startswith("missing"):
            for feature_family in config.feature_families:
                reason = dependency_status(config.model_id)
                failures.append({"model_group": config.model_group, "model_id": config.model_id, "feature_family": feature_family, "horizon": horizon, "config_name": config.config_name, "reason": reason})
                add_grid_row(candidate_rows, candidate=candidate_id(config.model_group, config.model_id, feature_family, config.config_name, "skipped"), model_group=config.model_group, model_id=config.model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", config_name=config.config_name, hyperparameters=config.params, reason=reason)
            continue
        for feature_family in config.feature_families:
            try:
                val_score, final_score, val_index, final_index, train_rows, feature_count, runtime = fit_predict_estimator(
                    features=features,
                    labels_by_horizon=labels_by_horizon,
                    split_by_horizon=split_by_horizon,
                    family_cols=family_cols,
                    config=config,
                    feature_family=feature_family,
                    horizon=horizon,
                )
                for threshold_policy, threshold in threshold_specs(val_y, val_score):
                    row = add_evaluation(
                        candidate_rows=candidate_rows,
                        result_rows=result_rows,
                        score_cache=score_cache,
                        features=features,
                        labels=labels,
                        validation_index=val_index,
                        final_index=final_index,
                        validation_score=val_score,
                        final_score=final_score,
                        model_group=config.model_group,
                        model_id=config.model_id,
                        feature_family=feature_family,
                        horizon=horizon,
                        threshold_policy=threshold_policy,
                        threshold=threshold,
                        train_rows=train_rows,
                        feature_count=feature_count,
                        config_name=config.config_name,
                        hyperparameters=config.params,
                        selection_stage="primary_h40_full_zoo",
                        fit_runtime_seconds=runtime,
                        implementation_note="scaler/imputer inside train-only sklearn pipeline; validation-only threshold/objective selection",
                    )
                    if (
                        config.model_id in base_pool
                        and feature_family == "baseline_C_closest"
                        and threshold_policy == "fixed_0.50"
                    ):
                        base_pool[config.model_id].append(
                            BasePrediction(
                                model_id=config.model_id,
                                config_key=config.config_key,
                                config_name=config.config_name,
                                candidate_id=str(row["candidate_id"]),
                                validation_score=np.asarray(val_score, dtype=np.float32),
                                final_score=np.asarray(final_score, dtype=np.float32),
                                validation_accuracy=as_float(row["validation_accuracy"]),
                            )
                        )
                gc.collect()
            except Exception as exc:
                failures.append({"model_group": config.model_group, "model_id": config.model_id, "feature_family": feature_family, "horizon": horizon, "config_name": config.config_name, "reason": str(exc)[:500]})
                add_grid_row(candidate_rows, candidate=candidate_id(config.model_group, config.model_id, feature_family, f"h{horizon}", config.config_name, "failed"), model_group=config.model_group, model_id=config.model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=config.config_name, hyperparameters=config.params, reason=str(exc)[:500])
    return base_pool


def calibrate_scores(raw_cal: np.ndarray, y_cal: pd.Series, raw_val: np.ndarray, raw_final: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "sigmoid":
        calibrator = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
        calibrator.fit(np.asarray(raw_cal).reshape(-1, 1), y_cal.astype(int))
        return (
            calibrator.predict_proba(np.asarray(raw_val).reshape(-1, 1))[:, 1],
            calibrator.predict_proba(np.asarray(raw_final).reshape(-1, 1))[:, 1],
        )
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(raw_cal, dtype=float), y_cal.astype(int).to_numpy())
    return calibrator.predict(np.asarray(raw_val, dtype=float)), calibrator.predict(np.asarray(raw_final, dtype=float))


def run_time_safe_calibration(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    labels_by_horizon: dict[int, pd.Series],
    split_by_horizon: dict[int, dict[str, pd.Index]],
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_cache: dict[str, ScoreCache],
    failures: list[dict[str, Any]],
) -> None:
    horizon = PRIMARY_HORIZON
    labels = labels_by_horizon[horizon]
    idx = split_by_horizon[horizon]
    train_index = features.loc[idx["train"]].sort_values(["datetime", "ticker"]).index
    cut = int(len(train_index) * 0.80)
    core_index = pd.Index(train_index[:cut])
    calibration_index = pd.Index(train_index[cut:])
    if len(core_index) < 100 or len(calibration_index) < 100:
        return
    val_y = labels.loc[idx["validation"]].astype(int)
    specs: list[tuple[str, str, Callable[[pd.Series, dict[str, Any]], Any], dict[str, Any], str]] = [
        ("platt_logistic", "sigmoid", estimator_logistic_l2, {"C": 0.3, "class_weight": "balanced"}, "time-safe train-core logistic with Platt calibration on trailing train split"),
        ("isotonic_logistic", "isotonic", estimator_logistic_l2, {"C": 0.3, "class_weight": "balanced"}, "time-safe train-core logistic with isotonic calibration on trailing train split"),
        ("calibrated_svm", "sigmoid", estimator_linear_svm, {"C": 0.3}, "time-safe train-core linear SVM with Platt calibration on trailing train split"),
        ("calibrated_random_forest", "sigmoid", estimator_random_forest, {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 10, "max_features": "sqrt"}, "time-safe train-core RF with Platt calibration on trailing train split"),
        ("calibrated_xgboost", "sigmoid", estimator_xgboost, {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05}, "time-safe train-core XGBoost with Platt calibration on trailing train split"),
        ("calibrated_lightgbm", "sigmoid", estimator_lightgbm, {"n_estimators": 80, "max_depth": 3, "num_leaves": 15, "learning_rate": 0.05}, "time-safe train-core LightGBM with Platt calibration on trailing train split"),
    ]
    for model_id, method, factory, params, note in specs:
        if dependency_status(model_id).startswith("missing"):
            reason = dependency_status(model_id)
            failures.append({"model_group": "calibration_variants", "model_id": model_id, "horizon": horizon, "reason": reason})
            add_grid_row(candidate_rows, candidate=candidate_id("calibration_variants", model_id, "skipped"), model_group="calibration_variants", model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", config_name=method, hyperparameters=params, reason=reason)
            continue
        for feature_family in ("baseline_C_closest", "combined_context"):
            cols = family_cols.get(feature_family, [])
            if not cols:
                continue
            start = time.perf_counter()
            try:
                base = factory(labels.loc[core_index].astype(int), params)
                if base is None:
                    raise RuntimeError("base dependency unavailable")
                base.fit(features.loc[core_index, cols], labels.loc[core_index].astype(int))
                raw_cal = np.clip(universe.predict_score(base, features.loc[calibration_index, cols]), 0.0, 1.0)
                raw_val = np.clip(universe.predict_score(base, features.loc[idx["validation"], cols]), 0.0, 1.0)
                raw_final = np.clip(universe.predict_score(base, features.loc[idx["final"], cols]), 0.0, 1.0)
                val_score, final_score = calibrate_scores(raw_cal, labels.loc[calibration_index].astype(int), raw_val, raw_final, method)
                runtime = time.perf_counter() - start
                for threshold_policy, threshold in threshold_specs(val_y, val_score):
                    add_evaluation(
                        candidate_rows=candidate_rows,
                        result_rows=result_rows,
                        score_cache=score_cache,
                        features=features,
                        labels=labels,
                        validation_index=idx["validation"],
                        final_index=idx["final"],
                        validation_score=np.clip(val_score, 0.0, 1.0),
                        final_score=np.clip(final_score, 0.0, 1.0),
                        model_group="calibration_variants",
                        model_id=model_id,
                        feature_family=feature_family,
                        horizon=horizon,
                        threshold_policy=threshold_policy,
                        threshold=threshold,
                        train_rows=len(core_index),
                        feature_count=len(cols),
                        config_name=method,
                        hyperparameters=params | {"calibration_method": method, "calibration_split": "last_20pct_train"},
                        selection_stage="primary_h40_full_zoo",
                        fit_runtime_seconds=runtime,
                        implementation_note=note,
                    )
            except Exception as exc:
                failures.append({"model_group": "calibration_variants", "model_id": model_id, "feature_family": feature_family, "horizon": horizon, "reason": str(exc)[:500]})
                add_grid_row(candidate_rows, candidate=candidate_id("calibration_variants", model_id, feature_family, "failed"), model_group="calibration_variants", model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=method, hyperparameters=params, reason=str(exc)[:500])


def run_ensembles(
    features: pd.DataFrame,
    labels_by_horizon: dict[int, pd.Series],
    split_by_horizon: dict[int, dict[str, pd.Index]],
    base_pool: dict[str, list[BasePrediction]],
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_cache: dict[str, ScoreCache],
    failures: list[dict[str, Any]],
) -> None:
    horizon = PRIMARY_HORIZON
    labels = labels_by_horizon[horizon]
    idx = split_by_horizon[horizon]
    val_y = labels.loc[idx["validation"]].astype(int)
    selected_base: list[BasePrediction] = []
    for model_id, items in base_pool.items():
        valid = [item for item in items if math.isfinite(item.validation_accuracy)]
        if valid:
            selected_base.append(sorted(valid, key=lambda item: (-item.validation_accuracy, item.candidate_id))[0])
    if len(selected_base) < 2:
        reason = "fewer than two successful validation-selected base models"
        for model_id in ENSEMBLE_MODELS:
            failures.append({"model_group": "ensemble_stacking_models", "model_id": model_id, "horizon": horizon, "reason": reason})
            add_grid_row(candidate_rows, candidate=candidate_id("ensemble_stacking_models", model_id, "skipped"), model_group="ensemble_stacking_models", model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", config_name="base_pool", reason=reason)
        return
    val_matrix = np.column_stack([item.validation_score for item in selected_base])
    final_matrix = np.column_stack([item.final_score for item in selected_base])
    base_names = [item.model_id for item in selected_base]
    base_acc = np.asarray([item.validation_accuracy for item in selected_base], dtype=float)
    weights = base_acc / base_acc.sum() if float(base_acc.sum()) > 0 else np.repeat(1.0 / len(selected_base), len(selected_base))
    score_specs: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any], str, float]] = {}
    score_specs["hard_voting"] = ((val_matrix >= 0.50).mean(axis=1), (final_matrix >= 0.50).mean(axis=1), {"base_models": base_names}, "hard vote fraction from validation-selected base configs", 0.0)
    score_specs["soft_voting"] = (val_matrix.mean(axis=1), final_matrix.mean(axis=1), {"base_models": base_names}, "unweighted soft vote from validation-selected base configs", 0.0)
    score_specs["validation_weighted_soft_vote"] = (val_matrix @ weights, final_matrix @ weights, {"base_models": base_names, "weights": weights.tolist()}, "weights proportional to validation accuracy only", 0.0)
    meta_specs: dict[str, Any] = {
        "stacking_logistic_meta": LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
    }
    if LGBMClassifier is not None:
        meta_specs["stacking_lightgbm_meta"] = LGBMClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, min_child_samples=40, random_state=RANDOM_STATE, verbose=-1, n_jobs=2)
    if XGBClassifier is not None:
        meta_specs["stacking_xgboost_meta"] = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, min_child_weight=20, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=2)
    for model_id, estimator in meta_specs.items():
        start = time.perf_counter()
        try:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
            pipe.fit(pd.DataFrame(val_matrix, columns=base_names), val_y)
            val_score = np.clip(universe.predict_score(pipe, pd.DataFrame(val_matrix, columns=base_names)), 0.0, 1.0)
            final_score = np.clip(universe.predict_score(pipe, pd.DataFrame(final_matrix, columns=base_names)), 0.0, 1.0)
            score_specs[model_id] = (val_score, final_score, {"base_models": base_names}, "meta model trained only on validation base predictions; final labels not used", time.perf_counter() - start)
        except Exception as exc:
            failures.append({"model_group": "ensemble_stacking_models", "model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
            add_grid_row(candidate_rows, candidate=candidate_id("ensemble_stacking_models", model_id, "failed"), model_group="ensemble_stacking_models", model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name="meta", reason=str(exc)[:500])
    for model_id in ENSEMBLE_MODELS:
        if model_id not in score_specs:
            add_grid_row(candidate_rows, candidate=candidate_id("ensemble_stacking_models", model_id, "skipped"), model_group="ensemble_stacking_models", model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", config_name="dependency_or_fit", reason="dependency missing or meta fit failed")
            continue
        val_score, final_score, params, note, runtime = score_specs[model_id]
        for threshold_policy, threshold in threshold_specs(val_y, val_score):
            add_evaluation(
                candidate_rows=candidate_rows,
                result_rows=result_rows,
                score_cache=score_cache,
                features=features,
                labels=labels,
                validation_index=idx["validation"],
                final_index=idx["final"],
                validation_score=val_score,
                final_score=final_score,
                model_group="ensemble_stacking_models",
                model_id=model_id,
                feature_family="validation_selected_base_models",
                horizon=horizon,
                threshold_policy=threshold_policy,
                threshold=threshold,
                train_rows=len(labels.loc[idx["train"]].dropna()),
                feature_count=len(base_names),
                config_name="base_pool_validation_selected",
                hyperparameters=params,
                selection_stage="primary_h40_full_zoo",
                fit_runtime_seconds=runtime,
                implementation_note=note,
            )


def run_regime_models(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    labels_by_horizon: dict[int, pd.Series],
    split_by_horizon: dict[int, dict[str, pd.Index]],
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_cache: dict[str, ScoreCache],
    failures: list[dict[str, Any]],
) -> None:
    horizon = PRIMARY_HORIZON
    labels = labels_by_horizon[horizon]
    idx = split_by_horizon[horizon]
    val_y = labels.loc[idx["validation"]].astype(int)
    context_specs = [
        ("regime_context_logistic", estimator_logistic_l2, {"C": 0.3}),
        ("regime_context_xgboost", estimator_xgboost, {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05}),
        ("regime_context_lightgbm", estimator_lightgbm, {"n_estimators": 80, "max_depth": 3, "num_leaves": 15, "learning_rate": 0.05}),
    ]
    cols = family_cols.get("regime_context", [])
    for model_id, factory, params in context_specs:
        if dependency_status(model_id).startswith("missing"):
            reason = dependency_status(model_id)
            failures.append({"model_group": "regime_aware_models", "model_id": model_id, "horizon": horizon, "reason": reason})
            add_grid_row(candidate_rows, candidate=candidate_id("regime_aware_models", model_id, "skipped"), model_group="regime_aware_models", model_id=model_id, feature_family="regime_context", horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", config_name="default", hyperparameters=params, reason=reason)
            continue
        start = time.perf_counter()
        try:
            train_y = labels.loc[idx["train"]].astype(int)
            model = factory(train_y, params)
            if model is None:
                raise RuntimeError("dependency unavailable")
            model.fit(features.loc[idx["train"], cols], train_y)
            val_score = np.clip(universe.predict_score(model, features.loc[idx["validation"], cols]), 0.0, 1.0)
            final_score = np.clip(universe.predict_score(model, features.loc[idx["final"], cols]), 0.0, 1.0)
            runtime = time.perf_counter() - start
            for threshold_policy, threshold in threshold_specs(val_y, val_score):
                add_evaluation(
                    candidate_rows=candidate_rows,
                    result_rows=result_rows,
                    score_cache=score_cache,
                    features=features,
                    labels=labels,
                    validation_index=idx["validation"],
                    final_index=idx["final"],
                    validation_score=val_score,
                    final_score=final_score,
                    model_group="regime_aware_models",
                    model_id=model_id,
                    feature_family="regime_context",
                    horizon=horizon,
                    threshold_policy=threshold_policy,
                    threshold=threshold,
                    train_rows=len(train_y),
                    feature_count=len(cols),
                    config_name="lagged_regime_context",
                    hyperparameters=params,
                    selection_stage="primary_h40_full_zoo",
                    fit_runtime_seconds=runtime,
                    implementation_note="regime features are lagged/ex-ante; scaler/imputer train-only where applicable",
                )
        except Exception as exc:
            failures.append({"model_group": "regime_aware_models", "model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
            add_grid_row(candidate_rows, candidate=candidate_id("regime_aware_models", model_id, "failed"), model_group="regime_aware_models", model_id=model_id, feature_family="regime_context", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name="lagged_regime_context", hyperparameters=params, reason=str(exc)[:500])

    base_cols = family_cols.get("baseline_C_closest", [])
    router_specs = {
        "bull_bear_sideway_router": "market_direction_regime",
        "high_low_volatility_router": "volatility_regime",
        "regime_model_router": "regime_router_key",
    }
    for model_id, group_col in router_specs.items():
        start = time.perf_counter()
        try:
            val_score, final_score = universe.fit_router_models(features, labels, idx, base_cols, group_col)
            runtime = time.perf_counter() - start
            for threshold_policy, threshold in threshold_specs(val_y, val_score):
                add_evaluation(
                    candidate_rows=candidate_rows,
                    result_rows=result_rows,
                    score_cache=score_cache,
                    features=features,
                    labels=labels,
                    validation_index=idx["validation"],
                    final_index=idx["final"],
                    validation_score=val_score,
                    final_score=final_score,
                    model_group="regime_aware_models",
                    model_id=model_id,
                    feature_family="baseline_C_closest",
                    horizon=horizon,
                    threshold_policy=threshold_policy,
                    threshold=threshold,
                    train_rows=len(labels.loc[idx["train"]].dropna()),
                    feature_count=len(base_cols),
                    config_name=f"router_by_{group_col}",
                    hyperparameters={"group_col": group_col, "base_model": "logistic_l2"},
                    selection_stage="primary_h40_full_zoo",
                    fit_runtime_seconds=runtime,
                    implementation_note=f"ex-ante router by {group_col}; per-regime models fit on train only",
                )
        except Exception as exc:
            failures.append({"model_group": "regime_aware_models", "model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
            add_grid_row(candidate_rows, candidate=candidate_id("regime_aware_models", model_id, "failed"), model_group="regime_aware_models", model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=f"router_by_{group_col}", reason=str(exc)[:500])

    for model_id in ("regime_threshold_router", "conservative_router_fallback"):
        start = time.perf_counter()
        try:
            train_y = labels.loc[idx["train"]].astype(int)
            model = estimator_logistic_l2(train_y, {"C": 0.3, "class_weight": "balanced"})
            model.fit(features.loc[idx["train"], base_cols], train_y)
            val_score = np.clip(universe.predict_score(model, features.loc[idx["validation"], base_cols]), 0.0, 1.0)
            final_score = np.clip(universe.predict_score(model, features.loc[idx["final"], base_cols]), 0.0, 1.0)
            if model_id == "regime_threshold_router":
                thresholds = universe.group_thresholds(val_y, val_score, features.loc[idx["validation"], "regime_router_key"])
                val_pred = universe.apply_group_thresholds(val_score, features.loc[idx["validation"], "regime_router_key"], thresholds)
                final_pred = universe.apply_group_thresholds(final_score, features.loc[idx["final"], "regime_router_key"], thresholds)
                threshold_like_score_val = val_pred.astype(float)
                threshold_like_score_final = final_pred.astype(float)
                threshold_policy = "validation_selected_threshold"
                threshold = 0.50
                params = {"group_col": "regime_router_key", "thresholds": thresholds}
            else:
                threshold_policy = "fixed_0.50"
                threshold = 0.50
                threshold_like_score_val = np.where(np.abs(val_score - 0.50) < 0.025, 0.50, val_score)
                threshold_like_score_final = np.where(np.abs(final_score - 0.50) < 0.025, 0.50, final_score)
                params = {"fallback_band": 0.025, "base_model": "logistic_l2"}
            runtime = time.perf_counter() - start
            add_evaluation(
                candidate_rows=candidate_rows,
                result_rows=result_rows,
                score_cache=score_cache,
                features=features,
                labels=labels,
                validation_index=idx["validation"],
                final_index=idx["final"],
                validation_score=threshold_like_score_val,
                final_score=threshold_like_score_final,
                model_group="regime_aware_models",
                model_id=model_id,
                feature_family="baseline_C_closest",
                horizon=horizon,
                threshold_policy=threshold_policy,
                threshold=threshold,
                train_rows=len(train_y),
                feature_count=len(base_cols),
                config_name=model_id,
                hyperparameters=params,
                selection_stage="primary_h40_full_zoo",
                fit_runtime_seconds=runtime,
                implementation_note="regime thresholds/fallback selected from validation or fixed ex-ante; final labels not used",
            )
        except Exception as exc:
            failures.append({"model_group": "regime_aware_models", "model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
            add_grid_row(candidate_rows, candidate=candidate_id("regime_aware_models", model_id, "failed"), model_group="regime_aware_models", model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=model_id, reason=str(exc)[:500])


def run_deep_sequence_models(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    labels_by_horizon: dict[int, pd.Series],
    split_by_horizon: dict[int, dict[str, pd.Index]],
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_cache: dict[str, ScoreCache],
    failures: list[dict[str, Any]],
) -> None:
    horizon = PRIMARY_HORIZON
    if universe.torch is None:
        for model_id in ["lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"]:
            failures.append({"model_group": "neural_deep_models", "model_id": model_id, "horizon": horizon, "reason": "torch dependency is not installed"})
            add_grid_row(candidate_rows, candidate=candidate_id("neural_deep_models", model_id, "skipped"), model_group="neural_deep_models", model_id=model_id, feature_family="sequence", horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", config_name="sequence", reason="torch dependency is not installed")
        return
    labels = labels_by_horizon[horizon]
    idx = split_by_horizon[horizon]
    val_y_series = labels.loc[idx["validation"]].astype(int)
    for feature_family in ("baseline_C_closest",):
        base_cols = family_cols.get(feature_family, [])
        cols = universe.select_deep_cols(features, base_cols, idx["train"], limit=24)
        if not cols:
            continue
        matrix = universe.standardize_matrix(features, cols, idx["train"])
        for seq_len in (16, 32):
            x_train, y_train, _train_rows = universe.build_sequences(features, matrix, labels, idx["train"], seq_len)
            x_val, y_val, val_rows = universe.build_sequences(features, matrix, labels, idx["validation"], seq_len)
            x_final, _y_final, final_rows = universe.build_sequences(features, matrix, labels, idx["final"], seq_len)
            if len(y_train) < 100 or len(y_val) < 100 or len(x_final) == 0 or len(np.unique(y_train.astype(int))) < 2:
                reason = f"invalid sequence shape train={x_train.shape} validation={x_val.shape} final={x_final.shape}"
                for model_id in ["lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"]:
                    failures.append({"model_group": "neural_deep_models", "model_id": model_id, "horizon": horizon, "sequence_length": seq_len, "reason": reason})
                    add_grid_row(candidate_rows, candidate=candidate_id("neural_deep_models", model_id, feature_family, f"seq{seq_len}", "failed"), model_group="neural_deep_models", model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=f"seq{seq_len}", reason=reason)
                continue
            for model_id in ["lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"]:
                start = time.perf_counter()
                try:
                    model, fit_info, val_score = universe.fit_deep(model_id, x_train, y_train, x_val, y_val)
                    final_score = np.clip(universe.predict_deep(model, x_final), 0.0, 1.0)
                    val_score = np.clip(val_score, 0.0, 1.0)
                    runtime = time.perf_counter() - start
                    for threshold_policy, threshold in threshold_specs(pd.Series(y_val.astype(int), index=val_rows), val_score):
                        add_evaluation(
                            candidate_rows=candidate_rows,
                            result_rows=result_rows,
                            score_cache=score_cache,
                            features=features,
                            labels=labels,
                            validation_index=val_rows,
                            final_index=final_rows,
                            validation_score=val_score,
                            final_score=final_score,
                            model_group="neural_deep_models",
                            model_id=model_id,
                            feature_family=feature_family,
                            horizon=horizon,
                            threshold_policy=threshold_policy,
                            threshold=threshold,
                            train_rows=len(y_train),
                            feature_count=len(cols),
                            config_name=f"seq{seq_len}",
                            hyperparameters={"sequence_length": seq_len, "selected_feature_count": len(cols), "epochs_max": 3},
                            selection_stage="primary_h40_full_zoo",
                            fit_runtime_seconds=runtime,
                            implementation_note=f"early stopping uses validation loss only; best_epoch={fit_info.get('best_epoch')}",
                            sequence_length=seq_len,
                        )
                    del model
                    gc.collect()
                except Exception as exc:
                    failures.append({"model_group": "neural_deep_models", "model_id": model_id, "horizon": horizon, "sequence_length": seq_len, "reason": str(exc)[:500]})
                    add_grid_row(candidate_rows, candidate=candidate_id("neural_deep_models", model_id, feature_family, f"seq{seq_len}", "failed"), model_group="neural_deep_models", model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=f"seq{seq_len}", reason=str(exc)[:500])


def remap_statistical_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remapped = []
    for row in rows:
        item = dict(row)
        model_id = str(item.get("model_id", ""))
        item["model_group"] = MODEL_TO_GROUP.get(model_id, "statistical_models")
        item["config_name"] = item.get("config_name", "statsmodels_default")
        item["hyperparameters"] = item.get("hyperparameters", "{}")
        item["selection_stage"] = "primary_h40_full_zoo"
        item["fit_runtime_seconds"] = item.get("fit_runtime_seconds", math.nan)
        item["primary_horizon_policy"] = "primary_h40"
        item["interpretability_score"] = INTERPRETABILITY_SCORE.get(str(item["model_group"]), 4)
        item["prior_final_score_privileged"] = False
        item["prior_63_33_context_only"] = True
        item["current_61_63_context_only"] = True
        remapped.append(item)
    return remapped


def remap_statistical_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        model_id = str(item.get("model_id", ""))
        item["model_group"] = MODEL_TO_GROUP.get(model_id, "statistical_models")
        item["config_name"] = item.get("config_name", "statsmodels_default")
        item["hyperparameters"] = item.get("hyperparameters", "{}")
        item["selection_stage"] = "primary_h40_full_zoo"
        item["fit_runtime_seconds"] = item.get("fit_runtime_seconds", math.nan)
        item["final_accuracy_used_for_selection"] = False
        item["ticker_subset"] = False
        item["confidence_abstention"] = False
        item["topk_substitution"] = False
        out.append(item)
    return out


def run_statistical_family(
    features: pd.DataFrame,
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    old_horizons = list(universe.HORIZONS)
    universe.HORIZONS = [PRIMARY_HORIZON]
    external_grid: list[dict[str, Any]] = []
    external_predictions: list[pd.DataFrame] = []
    try:
        rows, fail, summary, garch_text = universe.run_statistical_models(features, external_grid, external_predictions)
    finally:
        universe.HORIZONS = old_horizons
    for item in fail:
        item = dict(item)
        item["model_group"] = "statistical_models"
        failures.append(item)
    candidate_rows.extend(remap_statistical_grid(external_grid))
    remapped_rows = remap_statistical_group_rows(rows)
    for row in remapped_rows:
        row.update(validation_metric_overlay(pd.DataFrame()))
        row["config_name"] = row.get("config_name", "statsmodels_default")
        row["hyperparameters"] = row.get("hyperparameters", "{}")
        row["selection_stage"] = "primary_h40_full_zoo"
        row["interpretability_score"] = INTERPRETABILITY_SCORE["statistical_models"]
        row["prior_final_score_privileged"] = False
        row["prior_63_33_context_only"] = True
        row["current_61_63_context_only"] = True
        risk, reason = classify_overfit(pd.Series(row))
        row["overfit_risk"] = risk
        row["overfit_risk_reason"] = reason
        result_rows.append(row)
    write_markdown(OUTPUT_DIR / "garch_diagnostic_summary.md", garch_text)
    return summary


def select_by_objectives(final_results: pd.DataFrame) -> pd.DataFrame:
    pool = final_results[
        final_results["status"].astype(str).eq("ok")
        & final_results["full_ticker_coverage"].astype(bool)
        & final_results["horizon"].astype(int).eq(PRIMARY_HORIZON)
        & ~final_results["model_id"].astype(str).eq("garch_volatility_diagnostic")
    ].copy()
    pool = pool[pool["validation_accuracy"].apply(lambda value: math.isfinite(as_float(value)))]
    objectives = [
        ("max_validation_accuracy", "validation_accuracy", False),
        ("max_validation_balanced_accuracy", "validation_balanced_accuracy", False),
        ("max_validation_lift_over_majority", "validation_lift_over_majority", False),
        ("max_validation_rolling_stability", "validation_rolling_stability", False),
        ("min_validation_instability", "validation_instability", True),
        ("max_validation_monthly_stability", "validation_monthly_stability", False),
        ("max_validation_ticker_stability", "validation_ticker_stability", False),
        ("balanced_robust_score", "balanced_robust_score", False),
    ]
    rows: list[dict[str, Any]] = []
    if pool.empty:
        return pd.DataFrame(rows)
    for objective, metric, ascending in objectives:
        work = pool[pool[metric].apply(lambda value: math.isfinite(as_float(value)))].copy() if metric in pool.columns else pd.DataFrame()
        if work.empty:
            continue
        selected = work.sort_values([metric, "validation_accuracy", "candidate_id"], ascending=[ascending, False, True]).iloc[0].copy()
        selected["selection_objective"] = objective
        selected["selection_metric"] = metric
        selected["selection_metric_value"] = selected.get(metric)
        selected["selection_scope"] = "primary_h40_validation_only"
        rows.append(dict(selected))
    return pd.DataFrame(rows)


def apply_selection_flags(final_results: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = final_results.copy()
    out["selected_by_validation_objective_yes_no"] = "no"
    out["selection_objectives_won"] = ""
    if not selected.empty:
        objective_map = selected.groupby("candidate_id")["selection_objective"].apply(lambda values: ";".join(sorted(set(map(str, values))))).to_dict()
        for candidate, objectives in objective_map.items():
            mask = out["candidate_id"].astype(str).eq(str(candidate))
            out.loc[mask, "selected_by_validation_objective_yes_no"] = "yes"
            out.loc[mask, "selection_objectives_won"] = objectives
    out["claim_eligible_yes_no"] = "no"
    out["reason_not_claim_eligible"] = "not selected by validation-only objectives"
    selected_mask = out["selected_by_validation_objective_yes_no"].eq("yes")
    eligible_mask = (
        selected_mask
        & out["full_ticker_coverage"].astype(bool)
        & ~out["model_id"].astype(str).eq("garch_volatility_diagnostic")
        & ~out["overfit_risk"].astype(str).eq("high")
    )
    out.loc[eligible_mask, "claim_eligible_yes_no"] = "yes"
    out.loc[eligible_mask, "reason_not_claim_eligible"] = ""
    out.loc[selected_mask & out["overfit_risk"].astype(str).eq("high"), "reason_not_claim_eligible"] = "selected by validation objective but high overfit risk"
    out.loc[selected_mask & ~out["full_ticker_coverage"].astype(bool), "reason_not_claim_eligible"] = "selected but lacks full 30-stock coverage"
    return out


def run_secondary_diagnostics(
    selected: pd.DataFrame,
    fit_config_by_key: dict[str, FitConfig],
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    labels_by_horizon: dict[int, pd.Series],
    split_by_horizon: dict[int, dict[str, pd.Index]],
    candidate_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_cache: dict[str, ScoreCache],
    failures: list[dict[str, Any]],
) -> None:
    if selected.empty:
        return
    unique = selected.drop_duplicates("candidate_id").copy()
    for _, selected_row in unique.iterrows():
        model_group = str(selected_row["model_group"])
        model_id = str(selected_row["model_id"])
        feature_family = str(selected_row["feature_family"])
        config_name = str(selected_row.get("config_name", ""))
        config_key = f"{model_id}:{config_name}"
        if config_key not in fit_config_by_key:
            continue
        config = fit_config_by_key[config_key]
        for horizon in SECONDARY_HORIZONS:
            labels = labels_by_horizon[horizon]
            val_y = labels.loc[split_by_horizon[horizon]["validation"]].astype(int)
            try:
                val_score, final_score, val_index, final_index, train_rows, feature_count, runtime = fit_predict_estimator(
                    features=features,
                    labels_by_horizon=labels_by_horizon,
                    split_by_horizon=split_by_horizon,
                    family_cols=family_cols,
                    config=config,
                    feature_family=feature_family,
                    horizon=horizon,
                )
                for threshold_policy, threshold in threshold_specs(val_y, val_score):
                    add_evaluation(
                        candidate_rows=candidate_rows,
                        result_rows=result_rows,
                        score_cache=score_cache,
                        features=features,
                        labels=labels,
                        validation_index=val_index,
                        final_index=final_index,
                        validation_score=val_score,
                        final_score=final_score,
                        model_group=model_group,
                        model_id=model_id,
                        feature_family=feature_family,
                        horizon=horizon,
                        threshold_policy=threshold_policy,
                        threshold=threshold,
                        train_rows=train_rows,
                        feature_count=feature_count,
                        config_name=config_name,
                        hyperparameters=config.params,
                        selection_stage="secondary_horizon_diagnostic_for_h40_validation_selected",
                        fit_runtime_seconds=runtime,
                        implementation_note="secondary h20/h60/h80 diagnostic for h40 validation-selected candidate; final not used for horizon selection",
                    )
            except Exception as exc:
                failures.append({"model_group": model_group, "model_id": model_id, "feature_family": feature_family, "horizon": horizon, "config_name": config_name, "reason": str(exc)[:500]})
                add_grid_row(candidate_rows, candidate=candidate_id(model_group, model_id, feature_family, f"h{horizon}", config_name, "secondary_failed"), model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", config_name=config_name, hyperparameters=config.params, selection_stage="secondary_horizon_diagnostic_for_h40_validation_selected", reason=str(exc)[:500])


def prediction_rows_for_candidates(
    candidate_ids: list[str],
    score_cache: dict[str, ScoreCache],
    features: pd.DataFrame,
    labels_by_horizon: dict[int, pd.Series],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for candidate in candidate_ids:
        cache = score_cache.get(candidate)
        if cache is None:
            continue
        labels = labels_by_horizon[cache.horizon]
        val_pred = (cache.validation_score >= cache.threshold).astype(int)
        final_pred = (cache.final_score >= cache.threshold).astype(int)
        frames.append(
            universe.prediction_frame(
                features,
                cache.validation_index,
                labels,
                cache.validation_score,
                val_pred,
                model_group=cache.model_group,
                model_id=cache.model_id,
                feature_family=cache.feature_family,
                horizon=cache.horizon,
                threshold_policy=cache.threshold_policy,
                threshold=cache.threshold,
                candidate=cache.candidate_id,
                split="validation",
            )
        )
        frames.append(
            universe.prediction_frame(
                features,
                cache.final_index,
                labels,
                cache.final_score,
                final_pred,
                model_group=cache.model_group,
                model_id=cache.model_id,
                feature_family=cache.feature_family,
                horizon=cache.horizon,
                threshold_policy=cache.threshold_policy,
                threshold=cache.threshold,
                candidate=cache.candidate_id,
                split="final",
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_slice_outputs(row_predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    if row_predictions.empty:
        empty = pd.DataFrame()
        for name in ["by_ticker", "by_month", "by_quarter", "rolling_250", "rolling_500", "rolling_1000"]:
            outputs[name] = empty
        return outputs
    work = row_predictions.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    group_cols = ["candidate_id", "split", "model_group", "model_id", "feature_family", "horizon", "threshold_policy"]
    outputs["by_ticker"] = work.groupby([*group_cols, "ticker"], dropna=False)["correct"].agg(rows="size", accuracy="mean").reset_index()
    month = work.dropna(subset=["datetime"]).copy()
    month["month"] = month["datetime"].dt.to_period("M").astype(str)
    month["quarter"] = month["datetime"].dt.to_period("Q").astype(str)
    outputs["by_month"] = month.groupby([*group_cols, "month"], dropna=False)["correct"].agg(rows="size", accuracy="mean").reset_index()
    outputs["by_quarter"] = month.groupby([*group_cols, "quarter"], dropna=False)["correct"].agg(rows="size", accuracy="mean").reset_index()
    for window in (250, 500, 1000):
        rows = []
        for candidate, group in work.groupby("candidate_id", sort=True):
            ordered = group.sort_values(["split", "datetime", "ticker"]).copy()
            for split, split_group in ordered.groupby("split", sort=True):
                split_group = split_group.sort_values(["datetime", "ticker"]).reset_index(drop=True)
                roll = split_group["correct"].astype(float).rolling(window=window, min_periods=window).mean()
                temp = split_group.loc[roll.notna(), ["candidate_id", "split", "datetime", "ticker", "model_group", "model_id", "feature_family", "horizon", "threshold_policy"]].copy()
                temp["window"] = window
                temp["rolling_accuracy"] = roll.dropna().to_numpy()
                rows.append(temp)
        outputs[f"rolling_{window}"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return outputs


def aggregate_outputs(final_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ok = final_results[final_results["status"].astype(str).eq("ok")].copy()
    aggregations = {
        "candidates": ("candidate_id", "nunique"),
        "mean_validation_accuracy": ("validation_accuracy", "mean"),
        "mean_final_accuracy": ("final_accuracy", "mean"),
        "best_final_accuracy": ("final_accuracy", "max"),
        "mean_validation_final_gap": ("validation_final_gap", "mean"),
        "mean_balanced_robust_score": ("balanced_robust_score", "mean"),
        "mean_rolling_250": ("rolling_250_mean", "mean"),
        "mean_runtime_seconds": ("fit_runtime_seconds", "mean"),
    }
    by_group = ok.groupby("model_group").agg(**aggregations).reset_index() if not ok.empty else pd.DataFrame()
    by_horizon = ok.groupby("horizon").agg(**aggregations).reset_index() if not ok.empty else pd.DataFrame()
    transfer = by_group.copy()
    if not transfer.empty:
        gap_abs = ok.assign(abs_gap=ok["validation_final_gap"].abs()).groupby("model_group")["abs_gap"].median().reset_index(name="median_abs_validation_final_gap")
        transfer = transfer.merge(gap_abs, on="model_group", how="left")
        transfer["transfer_quality_score"] = transfer["mean_final_accuracy"] - transfer["median_abs_validation_final_gap"].fillna(0.0)
        transfer = transfer.sort_values("transfer_quality_score", ascending=False)
    runtime = ok.groupby(["model_group", "model_id", "config_name"], dropna=False).agg(configs=("candidate_id", "nunique"), mean_runtime_seconds=("fit_runtime_seconds", "mean"), best_final_accuracy=("final_accuracy", "max"), best_validation_accuracy=("validation_accuracy", "max")).reset_index() if not ok.empty else pd.DataFrame()
    return {"by_model_group": by_group, "by_horizon": by_horizon, "transfer_quality": transfer, "runtime": runtime}


def planned_budget_registry(fit_configs: list[FitConfig]) -> pd.DataFrame:
    config_counts = pd.DataFrame([{"model_group": cfg.model_group, "model_id": cfg.model_id, "feature_count": len(cfg.feature_families)} for cfg in fit_configs])
    planned_by_group = config_counts.groupby("model_group")["feature_count"].sum().to_dict() if not config_counts.empty else {}
    planned_by_group["naive_baseline"] = len(NAIVE_MODELS)
    planned_by_group["technical_rules"] = sum(len(technical_param_grid(model_id)) * 2 for model_id in TECHNICAL_MODELS)
    planned_by_group["ensemble_stacking_models"] = len(ENSEMBLE_MODELS) * 2
    planned_by_group["calibration_variants"] = len(CALIBRATION_MODELS) * 2 * 2 + 2 * 2
    planned_by_group["regime_aware_models"] = 3 * 2 + 3 * 2 + 2
    planned_by_group["statistical_models"] = 4 + 1
    rows = []
    for model_group, models in MODEL_GROUPS.items():
        if model_group == "naive_baseline":
            hyper = "fixed ex-ante baselines; no hyperparameter tuning"
            reason = "small fixed budget because baselines are controls"
        elif model_group == "technical_rules":
            hyper = "rule windows for SMA/EMA/MACD/RSI/Bollinger/momentum/volume/mean-reversion/breakout; fixed and validation-selected thresholds"
            reason = "non-trivial rule-window sweep without privileging any rule"
        elif model_group == "statistical_models":
            hyper = "ARIMA/SARIMA/ETS/VAR train-window direction; GARCH volatility diagnostic only"
            reason = "controlled diagnostic budget due per-ticker time-series runtime"
        else:
            spaces = sorted({cfg.hyperparameter_space for cfg in fit_configs if cfg.model_group == model_group})
            hyper = "; ".join(spaces) if spaces else "validation-only controlled family budget"
            reason = "controlled non-trivial budget scaled by runtime risk and dependency availability"
        runtime_values = sorted({cfg.runtime_risk for cfg in fit_configs if cfg.model_group == model_group})
        scaling_values = sorted({cfg.scaling_required for cfg in fit_configs if cfg.model_group == model_group})
        dep_values = sorted({dependency_status(model_id) for model_id in models})
        rows.append(
            {
                "model_group": model_group,
                "models_in_group": ",".join(models),
                "planned_config_count": int(planned_by_group.get(model_group, 0)),
                "actual_config_count": 0,
                "hyperparameter_space": hyper,
                "reason_for_budget_size": reason,
                "runtime_risk": ",".join(runtime_values) if runtime_values else "low",
                "scaling_required": ",".join(scaling_values) if scaling_values else "not_required",
                "dependency_status": "; ".join(f"{model}:{dependency_status(model)}" for model in models),
                "selection_objectives": "max_validation_accuracy;max_validation_balanced_accuracy;max_validation_lift_over_majority;max_validation_rolling_stability;min_validation_instability;max_validation_monthly_stability;max_validation_ticker_stability;balanced_robust_score",
                "claim_eligibility_rule": "validation-only selected, full 30-stock coverage, non-diagnostic, audit-passed, not high overfit risk",
            }
        )
    return pd.DataFrame(rows)


def update_budget_actuals(registry: pd.DataFrame, candidate_grid: pd.DataFrame) -> pd.DataFrame:
    out = registry.copy()
    actuals = candidate_grid[candidate_grid["planned_status"].astype(str).eq("run")].groupby("model_group")["candidate_id"].nunique().to_dict()
    for idx, row in out.iterrows():
        out.at[idx, "actual_config_count"] = int(actuals.get(str(row["model_group"]), 0))
    return out


def write_budget_registry(registry: pd.DataFrame) -> None:
    write_csv(OUTPUT_DIR / "tuning_budget_registry.csv", registry)
    lines = [
        "# Fair Tuning Budget Registry",
        "",
        "- This is a fair full-model-zoo tuning run, not a targeted tuning run.",
        "- No model family is privileged because of prior final-window performance.",
        "- The prior 63.33% regime-router row is context only and does not alter budget priority.",
        "- The 61.63% Logistic L2 row is current-main context only and does not block another validation-selected, audit-passed model.",
        "",
        markdown_table(registry, max_rows=len(registry)),
    ]
    write_markdown(OUTPUT_DIR / "tuning_budget_registry.md", "\n".join(lines))


def make_figures(final_results: pd.DataFrame, selected: pd.DataFrame, registry: pd.DataFrame, aggregates: dict[str, pd.DataFrame]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    def safe_plot(name: str, func: Callable[[], None]) -> None:
        plt.figure(figsize=(10, 6))
        try:
            func()
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / name, dpi=160)
        finally:
            plt.close()

    def budget() -> None:
        work = registry.sort_values("model_group")
        x = np.arange(len(work))
        plt.bar(x - 0.2, work["planned_config_count"], width=0.4, label="planned")
        plt.bar(x + 0.2, work["actual_config_count"], width=0.4, label="actual")
        plt.xticks(x, work["model_group"], rotation=75, ha="right")
        plt.ylabel("candidate rows")
        plt.title("Fair Tuning Budget by Family")
        plt.legend()

    safe_plot("fig_fair_tuning_budget_by_family.png", budget)

    def claim_vs_desc() -> None:
        claim = final_results[final_results["claim_eligible_yes_no"].astype(str).eq("yes")]
        values = [
            claim["final_accuracy"].max() if not claim.empty else math.nan,
            final_results["final_accuracy"].max() if not final_results.empty else math.nan,
        ]
        plt.bar(["claim eligible", "descriptive final"], values, color=["#2a9d8f", "#8d99ae"])
        plt.axhline(CURRENT_MAIN_FINAL_ACCURACY, color="#d62828", linestyle="--", label="current 61.63%")
        plt.ylim(0.45, max([v for v in values if math.isfinite(v)] + [CURRENT_MAIN_FINAL_ACCURACY]) + 0.03)
        plt.ylabel("final accuracy")
        plt.title("Claim Eligible vs Descriptive Accuracy")
        plt.legend()

    safe_plot("fig_claim_eligible_vs_descriptive_accuracy.png", claim_vs_desc)

    def val_final_selected() -> None:
        work = selected.drop_duplicates("candidate_id") if not selected.empty else pd.DataFrame()
        if work.empty:
            plt.text(0.5, 0.5, "No selected candidates", ha="center")
            return
        plt.scatter(work["validation_accuracy"], work["final_accuracy"], s=50)
        for _, row in work.iterrows():
            plt.annotate(str(row["model_id"])[:18], (row["validation_accuracy"], row["final_accuracy"]), fontsize=8)
        plt.xlabel("validation accuracy")
        plt.ylabel("final accuracy")
        plt.title("Validation vs Final for Selected Candidates")

    safe_plot("fig_validation_vs_final_selected_candidates.png", val_final_selected)

    def family_tradeoff() -> None:
        work = aggregates["by_model_group"]
        if work.empty:
            return
        plt.scatter(work["mean_final_accuracy"], work["mean_rolling_250"], s=80)
        for _, row in work.iterrows():
            plt.annotate(str(row["model_group"]), (row["mean_final_accuracy"], row["mean_rolling_250"]), fontsize=8)
        plt.xlabel("mean final accuracy")
        plt.ylabel("mean rolling 250 final accuracy")
        plt.title("Model Family Accuracy-Stability Tradeoff")

    safe_plot("fig_model_family_accuracy_stability_tradeoff.png", family_tradeoff)

    def gap_by_family() -> None:
        work = aggregates["by_model_group"].sort_values("mean_validation_final_gap")
        plt.barh(work["model_group"], work["mean_validation_final_gap"], color="#457b9d")
        plt.axvline(0.0, color="black", linewidth=0.8)
        plt.xlabel("validation minus final accuracy")
        plt.title("Validation-Final Gap by Family")

    safe_plot("fig_validation_final_gap_by_family.png", gap_by_family)

    def overfit_family() -> None:
        work = final_results.groupby(["model_group", "overfit_risk"]).size().unstack(fill_value=0)
        for col in ["low", "medium", "high", "unknown"]:
            if col not in work.columns:
                work[col] = 0
        bottom = np.zeros(len(work))
        x = np.arange(len(work))
        colors = {"low": "#2a9d8f", "medium": "#e9c46a", "high": "#e76f51", "unknown": "#8d99ae"}
        for col in ["low", "medium", "high", "unknown"]:
            plt.bar(x, work[col], bottom=bottom, label=col, color=colors[col])
            bottom += work[col].to_numpy()
        plt.xticks(x, work.index, rotation=75, ha="right")
        plt.ylabel("candidate rows")
        plt.title("Overfit Risk by Family")
        plt.legend()

    safe_plot("fig_overfit_risk_by_family.png", overfit_family)

    def runtime_accuracy() -> None:
        work = final_results[final_results["fit_runtime_seconds"].apply(lambda value: math.isfinite(as_float(value)))].copy()
        plt.scatter(work["fit_runtime_seconds"], work["final_accuracy"], s=16, alpha=0.5)
        plt.xlabel("fit runtime seconds")
        plt.ylabel("final accuracy")
        plt.title("Runtime vs Accuracy")

    safe_plot("fig_runtime_vs_accuracy.png", runtime_accuracy)

    def interpretability_accuracy() -> None:
        work = final_results.groupby("model_group").agg(best_final=("final_accuracy", "max"), interpretability=("interpretability_score", "max")).reset_index()
        plt.scatter(work["interpretability"], work["best_final"], s=80)
        for _, row in work.iterrows():
            plt.annotate(str(row["model_group"]), (row["interpretability"], row["best_final"]), fontsize=8)
        plt.xlabel("interpretability score")
        plt.ylabel("best final accuracy")
        plt.title("Interpretability vs Accuracy")

    safe_plot("fig_interpretability_vs_accuracy.png", interpretability_accuracy)

    def transfer_quality() -> None:
        work = aggregates["transfer_quality"].sort_values("transfer_quality_score")
        plt.barh(work["model_group"], work["transfer_quality_score"], color="#2a9d8f")
        plt.xlabel("transfer quality score")
        plt.title("Transfer Quality by Family")

    safe_plot("fig_transfer_quality_by_family.png", transfer_quality)

    def current_vs_best() -> None:
        best_selected = selected[selected["selection_objective"].astype(str).eq("max_validation_accuracy")]
        best_value = as_float(best_selected.iloc[0]["final_accuracy"]) if not best_selected.empty else math.nan
        labels = ["current main 61.63", "best validation selected"]
        values = [CURRENT_MAIN_FINAL_ACCURACY, best_value]
        plt.bar(labels, values, color=["#8d99ae", "#2a9d8f"])
        plt.ylim(0.45, max([v for v in values if math.isfinite(v)] + [CURRENT_MAIN_FINAL_ACCURACY]) + 0.03)
        plt.ylabel("final accuracy")
        plt.title("Current Main vs Best Fair-Tuned")

    safe_plot("fig_current_main_vs_best_fair_tuned.png", current_vs_best)


def write_summary_outputs(
    final_results: pd.DataFrame,
    selected: pd.DataFrame,
    registry: pd.DataFrame,
    aggregates: dict[str, pd.DataFrame],
    candidate_grid: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    best_validation = selected[selected["selection_objective"].astype(str).eq("max_validation_accuracy")] if not selected.empty else pd.DataFrame()
    best_robust = selected[selected["selection_objective"].astype(str).eq("balanced_robust_score")] if not selected.empty else pd.DataFrame()
    descriptive = final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False]).head(10)
    beaters = final_results[final_results["final_accuracy"] > CURRENT_MAIN_FINAL_ACCURACY].sort_values("final_accuracy", ascending=False)
    transfer = aggregates["transfer_quality"]
    lines = [
        "# Fair Exhaustive Model-Zoo Tuning Summary",
        "",
        f"- Primary horizon: h{PRIMARY_HORIZON}.",
        "- Secondary horizons: h20/h60/h80 only for h40 validation-selected candidates where rerun is feasible.",
        f"- Feature families: {', '.join(FEATURE_FAMILIES)}.",
        f"- Candidate rows planned/attempted: {len(candidate_grid)}.",
        f"- Successful result rows: {int(final_results['status'].astype(str).eq('ok').sum())}.",
        f"- Model groups tuned: {registry['model_group'].nunique()}.",
        f"- Current main context: {CURRENT_MAIN_LABEL}.",
        f"- Prior descriptive context only: {PRIOR_DESCRIPTIVE_LABEL}.",
        "- Final window role: scoring-only.",
        "- Data fetch: no.",
        "- Provider behavior changed: no.",
        "- Paper/DOCX generation: no.",
        "",
        "## Best Validation-Selected Candidate",
        "",
        markdown_table(best_validation[["selection_objective", "candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "overfit_risk"]], max_rows=3),
        "",
        "## Best Balanced-Robust Candidate",
        "",
        markdown_table(best_robust[["selection_objective", "candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "balanced_robust_score", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "overfit_risk"]], max_rows=3),
        "",
        "## Descriptive Final Leaderboard",
        "",
        markdown_table(descriptive[["candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "validation_accuracy", "final_accuracy", "selected_by_validation_objective_yes_no", "claim_eligible_yes_no", "overfit_risk"]], max_rows=10),
        "",
        "## Rows Beating 61.63%",
        "",
        markdown_table(beaters[["candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "validation_accuracy", "final_accuracy", "selected_by_validation_objective_yes_no", "claim_eligible_yes_no", "overfit_risk", "overfit_risk_reason"]], max_rows=20),
        "",
        "## Transfer Quality",
        "",
        markdown_table(transfer, max_rows=len(transfer)),
        "",
        "## Failures And Skips",
        "",
        markdown_table(failures.head(50), max_rows=50),
    ]
    write_markdown(OUTPUT_DIR / "fair_tuning_summary.md", "\n".join(lines))
    claim_lines = [
        "# Fair Tuning Claim Boundary",
        "",
        "- Claim eligibility requires validation-only selection, full 30-stock coverage, non-diagnostic model role, audit pass, and no high overfit risk.",
        "- Descriptive final-window leaderboard rows are descriptive only.",
        "- The final window is not used for model, family, feature, threshold, horizon, calibration, ensemble, router, or claim selection.",
        "- No trading, profitability, investment recommendation, or live-deployment claim is made.",
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected[["selection_objective", "candidate_id", "model_group", "model_id", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "reason_not_claim_eligible", "overfit_risk"]], max_rows=20),
    ]
    write_markdown(OUTPUT_DIR / "fair_tuning_claim_boundary.md", "\n".join(claim_lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tickers = active_stock_tickers()
    if len(tickers) != 30:
        raise RuntimeError(f"full 30-stock VN30 coverage required, got {len(tickers)}")

    features, family_cols, manifest = universe.prepare_features()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")

    all_horizons = [PRIMARY_HORIZON, *SECONDARY_HORIZONS]
    labels_by_horizon = {horizon: add_absolute_labels(features, horizon) for horizon in all_horizons}
    split_by_horizon = {horizon: universe.split_indices(features, labels_by_horizon[horizon]) for horizon in all_horizons}

    fit_configs = make_fit_configs()
    fit_config_by_key = {config.config_key: config for config in fit_configs}
    budget_registry = planned_budget_registry(fit_configs)
    write_budget_registry(budget_registry)

    candidate_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    score_cache: dict[str, ScoreCache] = {}

    run_rule_families(features, labels_by_horizon, split_by_horizon, candidate_rows, result_rows, score_cache, failures)
    base_pool = run_classifier_configs(features, family_cols, labels_by_horizon, split_by_horizon, fit_configs, candidate_rows, result_rows, score_cache, failures)
    run_time_safe_calibration(features, family_cols, labels_by_horizon, split_by_horizon, candidate_rows, result_rows, score_cache, failures)
    run_deep_sequence_models(features, family_cols, labels_by_horizon, split_by_horizon, candidate_rows, result_rows, score_cache, failures)
    run_regime_models(features, family_cols, labels_by_horizon, split_by_horizon, candidate_rows, result_rows, score_cache, failures)
    run_ensembles(features, labels_by_horizon, split_by_horizon, base_pool, candidate_rows, result_rows, score_cache, failures)
    statistical_summary = run_statistical_family(features, candidate_rows, result_rows, failures)

    candidate_grid = pd.DataFrame(candidate_rows)
    final_results = pd.DataFrame(result_rows)
    if final_results.empty:
        raise RuntimeError("fair tuning produced no successful result rows")
    selected = select_by_objectives(final_results)
    final_results = apply_selection_flags(final_results, selected)
    selected = select_by_objectives(final_results)
    if not selected.empty:
        selected = selected.merge(
            final_results[["candidate_id", "claim_eligible_yes_no", "reason_not_claim_eligible", "selected_by_validation_objective_yes_no", "selection_objectives_won"]],
            on="candidate_id",
            how="left",
            suffixes=("", "_updated"),
        )
        for col in ["claim_eligible_yes_no", "reason_not_claim_eligible", "selected_by_validation_objective_yes_no", "selection_objectives_won"]:
            updated = f"{col}_updated"
            if updated in selected.columns:
                selected[col] = selected[updated].combine_first(selected.get(col))
                selected = selected.drop(columns=[updated])

    run_secondary_diagnostics(selected, fit_config_by_key, features, family_cols, labels_by_horizon, split_by_horizon, candidate_rows, result_rows, score_cache, failures)
    candidate_grid = pd.DataFrame(candidate_rows)
    final_results = pd.DataFrame(result_rows)
    selected = select_by_objectives(final_results)
    final_results = apply_selection_flags(final_results, selected)
    selected = select_by_objectives(final_results)
    if not selected.empty:
        selected = selected.merge(
            final_results[["candidate_id", "claim_eligible_yes_no", "reason_not_claim_eligible", "selected_by_validation_objective_yes_no", "selection_objectives_won"]],
            on="candidate_id",
            how="left",
            suffixes=("", "_updated"),
        )
        for col in ["claim_eligible_yes_no", "reason_not_claim_eligible", "selected_by_validation_objective_yes_no", "selection_objectives_won"]:
            updated = f"{col}_updated"
            if updated in selected.columns:
                selected[col] = selected[updated].combine_first(selected.get(col))
                selected = selected.drop(columns=[updated])

    budget_registry = update_budget_actuals(budget_registry, candidate_grid)
    failures_frame = pd.DataFrame(failures)
    aggregates = aggregate_outputs(final_results)

    selected_ids = selected["candidate_id"].astype(str).drop_duplicates().tolist() if not selected.empty else []
    if selected_ids:
        row_prediction_ids = sorted(set(selected_ids))
    else:
        row_prediction_ids = (
            final_results.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True])["candidate_id"]
            .astype(str)
            .head(1)
            .tolist()
        )
    row_predictions = prediction_rows_for_candidates(row_prediction_ids, score_cache, features, labels_by_horizon)
    slice_outputs = build_slice_outputs(row_predictions)

    validation_results = final_results.copy()
    write_csv(OUTPUT_DIR / "fair_tuning_candidate_grid.csv", candidate_grid)
    write_csv(OUTPUT_DIR / "fair_tuning_validation_results.csv", validation_results)
    write_csv(OUTPUT_DIR / "fair_tuning_selected_by_objective.csv", selected)
    write_csv(OUTPUT_DIR / "fair_tuning_final_results.csv", final_results)
    write_csv(OUTPUT_DIR / "fair_tuning_row_predictions.csv", row_predictions)
    write_csv(OUTPUT_DIR / "fair_tuning_by_model_group.csv", aggregates["by_model_group"])
    write_csv(OUTPUT_DIR / "fair_tuning_by_horizon.csv", aggregates["by_horizon"])
    write_csv(OUTPUT_DIR / "fair_tuning_by_ticker.csv", slice_outputs["by_ticker"])
    write_csv(OUTPUT_DIR / "fair_tuning_by_month.csv", slice_outputs["by_month"])
    write_csv(OUTPUT_DIR / "fair_tuning_by_quarter.csv", slice_outputs["by_quarter"])
    write_csv(OUTPUT_DIR / "fair_tuning_rolling_250.csv", slice_outputs["rolling_250"])
    write_csv(OUTPUT_DIR / "fair_tuning_rolling_500.csv", slice_outputs["rolling_500"])
    write_csv(OUTPUT_DIR / "fair_tuning_rolling_1000.csv", slice_outputs["rolling_1000"])
    write_csv(OUTPUT_DIR / "fair_tuning_transfer_quality.csv", aggregates["transfer_quality"])
    write_csv(OUTPUT_DIR / "fair_tuning_runtime_summary.csv", aggregates["runtime"])
    write_csv(OUTPUT_DIR / "statistical_models_summary.csv", statistical_summary)
    write_csv(OUTPUT_DIR / "fair_tuning_failures.csv", failures_frame)
    write_budget_registry(budget_registry)
    write_summary_outputs(final_results, selected, budget_registry, aggregates, candidate_grid, failures_frame)
    make_figures(final_results, selected, budget_registry, aggregates)
    write_json(
        OUTPUT_DIR / "fair_tuning_run_manifest.json",
        {
            "data_fetch": False,
            "provider_behavior_changed": False,
            "paper_docx_generated": False,
            "primary_horizon": PRIMARY_HORIZON,
            "secondary_horizons": SECONDARY_HORIZONS,
            "feature_families": FEATURE_FAMILIES,
            "threshold_grid": THRESHOLD_GRID,
            "train_end": str(TRAIN_END),
            "validation_start": str(VAL_START),
            "validation_end": str(VAL_END),
            "final_start": str(FINAL_START),
            "model_groups_tuned": int(budget_registry["model_group"].nunique()),
            "candidate_rows": int(len(candidate_grid)),
            "successful_result_rows": int(final_results["status"].astype(str).eq("ok").sum()),
            "full_coverage": bool(final_results[final_results["selected_by_validation_objective_yes_no"].eq("yes")]["full_ticker_coverage"].astype(bool).all()) if not final_results.empty else False,
            "prior_63_33_context_only": PRIOR_DESCRIPTIVE_LABEL,
            "current_61_63_context_only": CURRENT_MAIN_LABEL,
            "manifest_source": manifest,
        },
    )

    print(f"Wrote fair tuning outputs to {rel(OUTPUT_DIR)}")
    print(f"Model groups tuned: {budget_registry['model_group'].nunique()}")
    print(f"Candidate rows: {len(candidate_grid)}")
    if not selected.empty:
        best = selected[selected["selection_objective"].eq("max_validation_accuracy")].iloc[0]
        print(f"Best validation-selected: {best['model_id']} {pct(best['final_accuracy'])}")


if __name__ == "__main__":
    main()
