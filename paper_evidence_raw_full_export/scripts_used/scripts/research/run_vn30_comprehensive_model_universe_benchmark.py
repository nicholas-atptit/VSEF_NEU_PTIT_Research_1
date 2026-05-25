"""Run the exhaustive VN30 comprehensive model-universe benchmark.

The run uses existing local VN30 hourly artifacts only. Model, feature,
threshold, ensemble, calibration, and router choices are validation-only; the
final window is scoring-only.
"""

from __future__ import annotations

import gc
import importlib.util
import importlib.metadata as importlib_metadata
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import accuracy_score
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

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from statsmodels.tsa.api import VAR, ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # pragma: no cover - optional dependency
    ARIMA = None
    SARIMAX = None
    ExponentialSmoothing = None
    VAR = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    FINAL_START,
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

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark"
FIGURE_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42
EXHAUSTIVE_FULL_RUN = True
HORIZONS = [20, 40, 60, 80]
FEATURE_FAMILIES = [
    "baseline_C_closest",
    "volatility_normalized",
    "relative_strength",
    "regime_context",
    "combined_context",
]
DEEP_FEATURE_FAMILIES = ["baseline_C_closest", "combined_context"]
THRESHOLD_GRID = [0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]
CURRENT_MAIN_FINAL_ACCURACY = 0.6163475699558174
CURRENT_MAIN_LABEL = "Logistic L2 baseline_C_closest h40 validation-selected threshold 0.55"
CURRENT_MAIN_CANDIDATE_PATTERN = ("logistic_l2", "baseline_C_closest", 40, "validation_selected_threshold")

NAIVE_BASELINES = [
    "majority_class",
    "random_walk_direction",
    "previous_direction",
    "persistence_rule",
    "moving_average_rule",
    "rolling_momentum_rule",
    "volatility_adjusted_momentum_rule",
]
TECHNICAL_RULES = [
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
    "linear_svm",
]
KERNEL_DISTANCE_MODELS = ["svm_rbf", "svm_poly", "knn", "radius_neighbors", "nearest_centroid"]
PROBABILISTIC_MODELS = ["gaussian_naive_bayes", "bernoulli_naive_bayes", "complement_naive_bayes"]
TREE_MODELS = ["decision_tree", "random_forest", "extra_trees"]
BOOSTING_MODELS = ["adaboost", "sklearn_gradient_boosting", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"]
DEEP_MODELS = ["mlp_classifier", "lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"]
ENSEMBLE_MODELS = [
    "hard_voting",
    "soft_voting",
    "validation_weighted_soft_vote",
    "stacking_logistic_meta",
    "stacking_lightgbm_meta",
    "stacking_xgboost_meta",
    "blending",
]
CALIBRATION_MODELS = [
    "platt_logistic",
    "isotonic_logistic",
    "calibrated_svm",
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
]
STATISTICAL_MODELS = ["arima_direction", "sarima_direction", "ets_direction", "var_direction", "garch_volatility_diagnostic"]

CLASSIFIER_MODEL_IDS = (
    LINEAR_MODELS
    + KERNEL_DISTANCE_MODELS
    + PROBABILISTIC_MODELS
    + TREE_MODELS
    + BOOSTING_MODELS
    + ["mlp_classifier"]
    + CALIBRATION_MODELS
)
ENSEMBLE_BASE_MODEL_IDS = [
    "logistic_l2",
    "logistic_l1",
    "linear_svm",
    "svm_rbf",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "hist_gradient_boosting",
    "gaussian_naive_bayes",
]


@dataclass
class BasePrediction:
    model_id: str
    candidate_id: str
    validation_score: np.ndarray
    final_score: np.ndarray
    validation_pred: np.ndarray
    final_pred: np.ndarray
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


def accuracy(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(accuracy_score(np.asarray(y_true, dtype=int), np.asarray(pred, dtype=int)))


def majority_value(y_true: pd.Series | np.ndarray) -> int:
    if len(y_true) == 0:
        return 1
    return int(float(np.asarray(y_true, dtype=int).mean()) >= 0.5)


def majority_accuracy(y_true: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    rate = float(np.asarray(y_true, dtype=int).mean())
    return max(rate, 1.0 - rate)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


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


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    work = frame.copy()
    if columns is not None:
        work = work[[col for col in columns if col in work.columns]]
    work = work.head(max_rows)
    headers = list(work.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def candidate_id(*parts: Any) -> str:
    return "__".join(str(part).replace(".", "p").replace(" ", "_").replace("/", "_") for part in parts)


def dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return ""


def dependency_summary_rows() -> list[dict[str, str]]:
    return [
        {
            "package": "catboost",
            "import_name": "catboost",
            "installed": "yes" if dependency_available("catboost") else "no",
            "version": dependency_version("catboost"),
            "benchmark_role": "boosting model family classifier",
        },
        {
            "package": "arch",
            "import_name": "arch",
            "installed": "yes" if dependency_available("arch") else "no",
            "version": dependency_version("arch"),
            "benchmark_role": "GARCH volatility diagnostic only",
        },
    ]


def write_dependency_install_report() -> None:
    rows = pd.DataFrame(dependency_summary_rows())
    lines = [
        "# Dependency Install Report",
        "",
        "- Install command requested for this rerun: `<repo-approved-venv>\\Scripts\\python.exe -m pip install catboost arch`.",
        "- Import verification command requested for CatBoost: `<repo-approved-venv>\\Scripts\\python.exe -c \"import catboost; print('catboost ok', catboost.__version__)\"`.",
        "- Import verification command requested for arch: `<repo-approved-venv>\\Scripts\\python.exe -c \"import arch; print('arch ok', arch.__version__)\"`.",
        "- Installation error: none recorded for this successful rerun.",
        "",
        markdown_table(rows, max_rows=len(rows)),
    ]
    write_markdown(OUTPUT_DIR / "dependency_install_report.md", "\n".join(lines))


def model_group_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for model_id in NAIVE_BASELINES:
        mapping[model_id] = "naive_baselines"
    for model_id in TECHNICAL_RULES:
        mapping[model_id] = "technical_rule_baselines"
    for model_id in LINEAR_MODELS:
        mapping[model_id] = "linear_generalized_linear_models"
    for model_id in KERNEL_DISTANCE_MODELS:
        mapping[model_id] = "kernel_distance_based_models"
    for model_id in PROBABILISTIC_MODELS:
        mapping[model_id] = "probabilistic_models"
    for model_id in TREE_MODELS:
        mapping[model_id] = "tree_based_models"
    for model_id in BOOSTING_MODELS:
        mapping[model_id] = "boosting_models"
    for model_id in DEEP_MODELS:
        mapping[model_id] = "neural_deep_models"
    for model_id in ENSEMBLE_MODELS:
        mapping[model_id] = "ensemble_stacking"
    for model_id in CALIBRATION_MODELS:
        mapping[model_id] = "calibration_variants"
    for model_id in REGIME_MODELS:
        mapping[model_id] = "regime_aware_models"
    for model_id in STATISTICAL_MODELS:
        mapping[model_id] = "traditional_statistical_financial_models"
    return mapping


def scaling_required(model_id: str) -> str:
    scaled = {
        "svm_rbf",
        "svm_poly",
        "linear_svm",
        "knn",
        "radius_neighbors",
        "nearest_centroid",
        "passive_aggressive",
        "perceptron",
        "sgd_hinge",
        "sgd_log_loss",
        "mlp_classifier",
        "calibrated_svm",
        "lda",
        "qda",
    }
    return "yes" if model_id in scaled else "no"


def dependency_required(model_id: str) -> str:
    if model_id in {"xgboost", "stacking_xgboost_meta", "calibrated_xgboost", "regime_context_xgboost"}:
        return "xgboost"
    if model_id in {"lightgbm", "stacking_lightgbm_meta", "calibrated_lightgbm", "regime_context_lightgbm"}:
        return "lightgbm"
    if model_id == "catboost":
        return "catboost"
    if model_id in {"lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"}:
        return "torch"
    if model_id in {"arima_direction", "sarima_direction", "ets_direction", "var_direction"}:
        return "statsmodels"
    if model_id == "garch_volatility_diagnostic":
        return "arch"
    return "sklearn" if model_id in CLASSIFIER_MODEL_IDS else "none"


def runtime_risk(model_id: str) -> str:
    if model_id in {"svm_rbf", "svm_poly", "calibrated_svm", "lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"}:
        return "high"
    if model_id in {"xgboost", "lightgbm", "catboost", "calibrated_xgboost", "calibrated_lightgbm", "random_forest", "extra_trees"}:
        return "medium"
    return "low"


def data_requirement(model_id: str) -> str:
    if model_id in TECHNICAL_RULES or model_id in NAIVE_BASELINES:
        return "OHLCV rows with non-null horizon labels"
    if model_id in {"lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"}:
        return "time-safe per-ticker sequence tensors with non-null labels"
    if model_id in STATISTICAL_MODELS:
        return "per-ticker close or return series from local historical rows"
    return "numeric feature matrix with non-null horizon labels"


def paper_role(model_id: str) -> str:
    if model_id == "garch_volatility_diagnostic":
        return "volatility_diagnostic_only"
    if model_id in NAIVE_BASELINES or model_id in TECHNICAL_RULES:
        return "baseline_comparison"
    if model_id in ENSEMBLE_MODELS:
        return "expanded_model_universe_ensemble"
    if model_id in CALIBRATION_MODELS:
        return "probability_calibration_diagnostic"
    return "expanded_model_universe_candidate"


def initial_registry() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = model_group_map()
    for model_id in (
        NAIVE_BASELINES
        + TECHNICAL_RULES
        + LINEAR_MODELS
        + KERNEL_DISTANCE_MODELS
        + PROBABILISTIC_MODELS
        + TREE_MODELS
        + BOOSTING_MODELS
        + DEEP_MODELS
        + ENSEMBLE_MODELS
        + CALIBRATION_MODELS
        + REGIME_MODELS
        + STATISTICAL_MODELS
    ):
        rows.append(
            {
                "model_id": model_id,
                "model_group": groups[model_id],
                "model_name": model_id.replace("_", " ").title(),
                "implementation_status": "implemented",
                "run_status": "planned",
                "dependency_required": dependency_required(model_id),
                "scaling_required": scaling_required(model_id),
                "runtime_risk": runtime_risk(model_id),
                "data_requirement": data_requirement(model_id),
                "reason_if_skipped": "",
                "reason_if_failed": "",
                "reason_if_not_recommended": "GARCH is a volatility diagnostic, not a direct direction classifier." if model_id == "garch_volatility_diagnostic" else "",
                "paper_role": paper_role(model_id),
                "claim_eligible": "no" if model_id == "garch_volatility_diagnostic" else "yes",
            }
        )
    return pd.DataFrame(rows)


def prepare_features() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    tickers = active_stock_tickers()
    if len(tickers) != 30:
        raise ValueError(f"expected full 30-stock VN30 universe, got {len(tickers)}")
    features, family_cols, manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    features = add_regime_labels(features)
    combined_cols = sorted(
        {
            col
            for family in ["baseline_C_closest", "volatility_normalized", "relative_strength", "regime_context"]
            for col in family_cols.get(family, [])
            if col in features.columns and pd.api.types.is_numeric_dtype(features[col])
        }
    )
    family_cols["combined_context"] = combined_cols
    family_cols = {
        family: [
            col
            for col in family_cols.get(family, [])
            if col in features.columns and pd.api.types.is_numeric_dtype(features[col])
        ]
        for family in FEATURE_FAMILIES
    }
    manifest = dict(manifest)
    manifest["comprehensive_model_universe_benchmark"] = {
        "exhaustive_full_run": EXHAUSTIVE_FULL_RUN,
        "horizons": HORIZONS,
        "feature_families": FEATURE_FAMILIES,
        "threshold_grid": THRESHOLD_GRID,
        "data_fetch": False,
        "provider_behavior_changed": False,
        "legacy_compatible_row_rules": True,
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
    }
    return features, family_cols, manifest


def add_regime_labels(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    trend = pd.Series(np.nan, index=out.index, dtype=float)
    if "vnindex_trend_60_lag_ctx" in out.columns:
        trend = pd.to_numeric(out["vnindex_trend_60_lag_ctx"], errors="coerce")
    elif "vn30_trend_60_lag_ctx" in out.columns:
        trend = pd.to_numeric(out["vn30_trend_60_lag_ctx"], errors="coerce")
    for fallback_col in ["momentum_60", "momentum_20", "rolling_return_mean_20"]:
        if fallback_col in out.columns:
            trend = trend.fillna(pd.to_numeric(out[fallback_col], errors="coerce"))
    out["market_direction_regime"] = "sideway"
    out.loc[trend > 0.02, "market_direction_regime"] = "bull"
    out.loc[trend < -0.02, "market_direction_regime"] = "bear"
    out.loc[trend.isna(), "market_direction_regime"] = "unknown_direction"
    ratio = pd.Series(np.nan, index=out.index, dtype=float)
    if "vnindex_vol_20_lag_ctx" in out.columns and "vnindex_vol_60_lag_ctx" in out.columns:
        ratio = pd.to_numeric(out["vnindex_vol_20_lag_ctx"], errors="coerce") / pd.to_numeric(out["vnindex_vol_60_lag_ctx"], errors="coerce").replace(0.0, np.nan)
    for short_col, long_col in [("rolling_return_vol_20", "rolling_return_vol_60"), ("roll_vol_20", "roll_vol_40")]:
        if short_col in out.columns and long_col in out.columns:
            fallback = pd.to_numeric(out[short_col], errors="coerce") / pd.to_numeric(out[long_col], errors="coerce").replace(0.0, np.nan)
            ratio = ratio.fillna(fallback)
    out["volatility_regime"] = "low_volatility"
    out.loc[ratio > 1.10, "volatility_regime"] = "high_volatility"
    out.loc[ratio.isna(), "volatility_regime"] = "unknown_volatility"
    out["regime_router_key"] = out["market_direction_regime"].astype(str) + "_" + out["volatility_regime"].astype(str)
    return out


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    score: np.ndarray,
    pred: np.ndarray,
    *,
    model_group: str,
    model_id: str,
    feature_family: str,
    horizon: int,
    threshold_policy: str,
    threshold: float,
    candidate: str,
    split: str,
) -> pd.DataFrame:
    out = features.loc[idx, ["datetime", "ticker", "market_direction_regime", "volatility_regime", "regime_router_key"]].copy()
    out["model_group"] = model_group
    out["model_id"] = model_id
    out["feature_family"] = feature_family
    out["horizon"] = int(horizon)
    out["threshold_policy"] = threshold_policy
    out["threshold"] = float(threshold)
    out["candidate_id"] = candidate
    out["split"] = split
    out["y_true"] = labels.loc[idx].astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(score, dtype=float)
    out["y_pred"] = np.asarray(pred, dtype=int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def rolling_stats(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "rolling_250_mean": math.nan,
            "rolling_500_mean": math.nan,
            "rolling_1000_mean": math.nan,
            "rolling_250_windows_below_60": 0,
            "rolling_500_windows_below_60": 0,
            "rolling_1000_windows_below_60": 0,
        }
    ordered = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    out: dict[str, float] = {}
    correct = ordered["correct"].astype(float)
    for window in (250, 500, 1000):
        roll = correct.rolling(window=window, min_periods=window).mean().dropna()
        out[f"rolling_{window}_mean"] = float(roll.mean()) if not roll.empty else math.nan
        out[f"rolling_{window}_windows_below_60"] = int((roll < 0.60).sum()) if not roll.empty else 0
    return out


def period_accuracy_stats(frame: pd.DataFrame, freq: str, prefix: str) -> dict[str, float]:
    if frame.empty:
        return {f"{prefix}_mean_accuracy": math.nan, f"{prefix}_median_accuracy": math.nan, f"{prefix}_min_accuracy": math.nan}
    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    grouped = work.dropna(subset=["datetime"]).groupby(work["datetime"].dt.to_period(freq))["correct"].mean()
    if grouped.empty:
        return {f"{prefix}_mean_accuracy": math.nan, f"{prefix}_median_accuracy": math.nan, f"{prefix}_min_accuracy": math.nan}
    return {
        f"{prefix}_mean_accuracy": float(grouped.mean()),
        f"{prefix}_median_accuracy": float(grouped.median()),
        f"{prefix}_min_accuracy": float(grouped.min()),
    }


def ticker_accuracy_stats(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"ticker_mean_accuracy": math.nan, "ticker_median_accuracy": math.nan, "ticker_min_accuracy": math.nan}
    grouped = frame.groupby("ticker")["correct"].mean()
    if grouped.empty:
        return {"ticker_mean_accuracy": math.nan, "ticker_median_accuracy": math.nan, "ticker_min_accuracy": math.nan}
    return {
        "ticker_mean_accuracy": float(grouped.mean()),
        "ticker_median_accuracy": float(grouped.median()),
        "ticker_min_accuracy": float(grouped.min()),
    }


def slice_accuracy_summary(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    grouped = frame.groupby(column, dropna=True)["correct"].mean().sort_index()
    if grouped.empty:
        return ""
    return "; ".join(f"{key}:{value:.4f}" for key, value in grouped.items())


def overfit_risk_from_values(validation_accuracy: float, final_accuracy: float, rolling_250_mean: float) -> str:
    if not math.isfinite(validation_accuracy) or not math.isfinite(final_accuracy):
        return "unknown"
    if validation_accuracy - final_accuracy > 0.05 or (math.isfinite(rolling_250_mean) and rolling_250_mean < 0.52):
        return "high"
    if validation_accuracy - final_accuracy > 0.02 or (math.isfinite(rolling_250_mean) and rolling_250_mean < 0.56):
        return "medium"
    return "low"


def classify_overfit_risk(row: pd.Series) -> tuple[str, str]:
    validation_accuracy = as_float(row.get("validation_accuracy"))
    final_accuracy = as_float(row.get("final_accuracy"))
    gap = validation_accuracy - final_accuracy if math.isfinite(validation_accuracy) and math.isfinite(final_accuracy) else math.nan
    rolling_250_mean = as_float(row.get("rolling_250_mean"))
    monthly_min = as_float(row.get("monthly_min_accuracy"))
    quarterly_min = as_float(row.get("quarterly_min_accuracy"))
    ticker_min = as_float(row.get("ticker_min_accuracy"))
    rolling_250_below_value = as_float(row.get("rolling_250_windows_below_60"))
    rolling_250_below = int(rolling_250_below_value) if math.isfinite(rolling_250_below_value) else 0
    selected = str(row.get("selected_by_validation_yes_no", "no")).lower() == "yes"
    beats_current = math.isfinite(final_accuracy) and final_accuracy > CURRENT_MAIN_FINAL_ACCURACY

    reasons: list[str] = []
    if beats_current and not selected:
        reasons.append("beats current main result only in post-hoc final leaderboard and was not validation-selected")
    if math.isfinite(gap) and gap > 0.05:
        reasons.append(f"validation accuracy exceeds final accuracy by {gap * 100.0:.2f} pp")
    if math.isfinite(rolling_250_mean) and rolling_250_mean < 0.56:
        reasons.append(f"rolling 250 mean is {rolling_250_mean * 100.0:.2f}%")
    if rolling_250_below > 0:
        reasons.append(f"{rolling_250_below} rolling 250 windows fall below 60%")
    if math.isfinite(monthly_min) and monthly_min < 0.55:
        reasons.append(f"weak monthly slice minimum {monthly_min * 100.0:.2f}%")
    if math.isfinite(quarterly_min) and quarterly_min < 0.55:
        reasons.append(f"weak quarterly slice minimum {quarterly_min * 100.0:.2f}%")
    if math.isfinite(ticker_min) and ticker_min < 0.55:
        reasons.append(f"weak ticker slice minimum {ticker_min * 100.0:.2f}%")

    if beats_current and not selected:
        return "high", "; ".join(reasons)
    if (math.isfinite(gap) and gap > 0.05) or (math.isfinite(rolling_250_mean) and rolling_250_mean < 0.52):
        return "high", "; ".join(reasons) or "large validation-final deterioration"
    if reasons:
        return "medium", "; ".join(reasons)
    return "low", "validation-final gap and stability slices do not show a major post-hoc warning"


def result_row(
    *,
    candidate: str,
    model_group: str,
    model_id: str,
    feature_family: str,
    horizon: int,
    threshold_policy: str,
    threshold: float,
    validation_frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    train_rows: int,
    feature_count: int,
    status: str = "ok",
    reason_not_claim_eligible: str = "not selected by validation",
    sequence_length: int | str = "",
    implementation_note: str = "",
) -> dict[str, Any]:
    validation_accuracy = float(validation_frame["correct"].mean()) if not validation_frame.empty else math.nan
    final_accuracy = float(final_frame["correct"].mean()) if not final_frame.empty else math.nan
    final_rows = int(len(final_frame))
    ticker_coverage = int(final_frame["ticker"].nunique()) if not final_frame.empty else 0
    rolling = rolling_stats(final_frame)
    monthly = period_accuracy_stats(final_frame, "M", "monthly")
    quarterly = period_accuracy_stats(final_frame, "Q", "quarterly")
    ticker_stats = ticker_accuracy_stats(final_frame)
    return {
        "candidate_id": candidate,
        "model_group": model_group,
        "model_id": model_id,
        "feature_family": feature_family,
        "horizon": int(horizon),
        "threshold_policy": threshold_policy,
        "threshold": float(threshold) if math.isfinite(float(threshold)) else math.nan,
        "status": status,
        "validation_accuracy": validation_accuracy,
        "final_accuracy": final_accuracy,
        "validation_final_gap": validation_accuracy - final_accuracy if math.isfinite(validation_accuracy) and math.isfinite(final_accuracy) else math.nan,
        "validation_rows": int(len(validation_frame)),
        "final_rows": final_rows,
        "ticker_coverage": ticker_coverage,
        "full_ticker_coverage": ticker_coverage == 30,
        "rolling_250_mean": rolling["rolling_250_mean"],
        "rolling_500_mean": rolling["rolling_500_mean"],
        "rolling_1000_mean": rolling["rolling_1000_mean"],
        "rolling_250_windows_below_60": rolling["rolling_250_windows_below_60"],
        "rolling_500_windows_below_60": rolling["rolling_500_windows_below_60"],
        "rolling_1000_windows_below_60": rolling["rolling_1000_windows_below_60"],
        **monthly,
        **quarterly,
        **ticker_stats,
        "market_regime_accuracy_summary": slice_accuracy_summary(final_frame, "market_direction_regime"),
        "volatility_regime_accuracy_summary": slice_accuracy_summary(final_frame, "volatility_regime"),
        "router_regime_accuracy_summary": slice_accuracy_summary(final_frame, "regime_router_key"),
        "beats_61_63_yes_no": "yes" if math.isfinite(final_accuracy) and final_accuracy > CURRENT_MAIN_FINAL_ACCURACY else "no",
        "selected_by_validation_yes_no": "no",
        "claim_eligible_yes_no": "no",
        "reason_not_claim_eligible": reason_not_claim_eligible,
        "train_rows": int(train_rows),
        "feature_count": int(feature_count),
        "sequence_length": sequence_length,
        "selection_source": "validation_only",
        "final_window_role": "scoring_only",
        "final_accuracy_used_for_selection": False,
        "ticker_subset": False,
        "confidence_abstention": False,
        "topk_substitution": False,
        "leakage_status": "passed_train_only_preprocessing_and_validation_only_selection",
        "delta_vs_61_63": final_accuracy - CURRENT_MAIN_FINAL_ACCURACY if math.isfinite(final_accuracy) else math.nan,
        "overfit_risk": overfit_risk_from_values(validation_accuracy, final_accuracy, rolling["rolling_250_mean"]),
        "overfit_risk_reason": "provisional before validation-selection update",
        "implementation_note": implementation_note,
    }


def select_threshold(y_true: pd.Series | np.ndarray, score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_accuracy = -1.0
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(score, dtype=float)
    for threshold in THRESHOLD_GRID:
        pred = (scores >= threshold).astype(int)
        acc = accuracy(y, pred)
        if acc > best_accuracy + 1e-12 or (abs(acc - best_accuracy) <= 1e-12 and abs(threshold - 0.50) < abs(best_threshold - 0.50)):
            best_accuracy = acc
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def make_classifier(model_id: str, y_train: pd.Series | np.ndarray) -> Any | None:
    majority = majority_value(y_train)
    if model_id == "logistic_l2":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LogisticRegression(max_iter=1000, solver="liblinear", penalty="l2", C=0.3, class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "logistic_l1":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LogisticRegression(max_iter=1200, solver="liblinear", penalty="l1", C=0.2, class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "logistic_elastic_net":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LogisticRegression(max_iter=2500, solver="saga", penalty="elasticnet", C=0.3, l1_ratio=0.2, class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "ridge_classifier":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", RidgeClassifier(class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "lda":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
    if model_id == "qda":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", QuadraticDiscriminantAnalysis(reg_param=0.2))])
    if model_id == "passive_aggressive":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", PassiveAggressiveClassifier(C=0.2, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "perceptron":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Perceptron(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "sgd_hinge":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SGDClassifier(loss="hinge", alpha=0.0005, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "sgd_log_loss":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SGDClassifier(loss="log_loss", alpha=0.0005, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "linear_svm":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LinearSVC(C=0.3, class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000, dual="auto"))])
    if model_id == "svm_rbf":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, cache_size=1600, random_state=RANDOM_STATE))])
    if model_id == "svm_poly":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(kernel="poly", degree=3, coef0=1.0, C=0.5, gamma="scale", class_weight="balanced", probability=True, cache_size=1600, random_state=RANDOM_STATE))])
    if model_id == "knn":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=-1))])
    if model_id == "radius_neighbors":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", RadiusNeighborsClassifier(radius=12.0, weights="distance", outlier_label=majority, n_jobs=-1))])
    if model_id == "nearest_centroid":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", NearestCentroid())])
    if model_id == "gaussian_naive_bayes":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", GaussianNB())])
    if model_id == "bernoulli_naive_bayes":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", BernoulliNB(binarize=0.0))])
    if model_id == "complement_naive_bayes":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("minmax", MinMaxScaler()), ("model", ComplementNB())])
    if model_id == "decision_tree":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_STATE))])
    if model_id == "random_forest":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", RandomForestClassifier(n_estimators=140, max_depth=8, min_samples_leaf=10, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])
    if model_id == "extra_trees":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", ExtraTreesClassifier(n_estimators=140, max_depth=8, min_samples_leaf=10, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])
    if model_id == "adaboost":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE))])
    if model_id == "sklearn_gradient_boosting":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.04, random_state=RANDOM_STATE))])
    if model_id == "hist_gradient_boosting":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.04, l2_regularization=0.1, random_state=RANDOM_STATE))])
    if model_id == "xgboost" and XGBClassifier is not None:
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, min_child_weight=8, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=2))])
    if model_id == "lightgbm" and LGBMClassifier is not None:
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, min_child_samples=35, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, verbose=-1, n_jobs=2))])
    if model_id == "catboost" and CatBoostClassifier is not None:
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", CatBoostClassifier(iterations=120, depth=4, learning_rate=0.05, loss_function="Logloss", random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False))])
    if model_id == "mlp_classifier":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(48, 16), alpha=0.001, max_iter=160, early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE))])
    if model_id == "platt_logistic":
        base = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=0.3, class_weight="balanced", random_state=RANDOM_STATE))])
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    if model_id == "isotonic_logistic":
        base = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=0.3, class_weight="balanced", random_state=RANDOM_STATE))])
        return CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    if model_id == "calibrated_svm":
        base = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LinearSVC(C=0.3, class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000, dual="auto"))])
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    if model_id == "calibrated_random_forest":
        base = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    if model_id == "calibrated_xgboost" and XGBClassifier is not None:
        base = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, min_child_weight=8, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=2))])
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    if model_id == "calibrated_lightgbm" and LGBMClassifier is not None:
        base = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, min_child_samples=35, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, verbose=-1, n_jobs=2))])
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    return None


def predict_score(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_data)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).ravel()
    if hasattr(model, "decision_function"):
        return sigmoid(np.asarray(model.decision_function(x_data), dtype=float))
    pred = model.predict(x_data)
    return np.asarray(pred, dtype=float)


def threshold_specs(y_true: pd.Series, score: np.ndarray, allow_validation_threshold: bool = True) -> list[tuple[str, float]]:
    specs = [("fixed_0.50", 0.50)]
    if allow_validation_threshold:
        specs.append(("validation_selected_threshold", select_threshold(y_true, score)[0]))
    return specs


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
    reason: str = "",
) -> None:
    candidate_rows.append(
        {
            "candidate_id": candidate,
            "model_group": model_group,
            "model_id": model_id,
            "feature_family": feature_family,
            "horizon": horizon,
            "threshold_policy": threshold_policy,
            "planned_status": planned_status,
            "reason": reason,
        }
    )


def score_series_rule(features: pd.DataFrame, labels: pd.Series, horizon: int, model_id: str, majority: int) -> pd.Series:
    close = pd.to_numeric(features["close"], errors="coerce")
    high = pd.to_numeric(features["high"], errors="coerce")
    low = pd.to_numeric(features["low"], errors="coerce")
    volume = pd.to_numeric(features["volume"], errors="coerce")
    by_ticker = features.groupby("ticker", sort=True)
    if model_id == "majority_class":
        return pd.Series(float(majority), index=features.index)
    if model_id == "random_walk_direction":
        previous_close = by_ticker["close"].shift(1)
        score = (close > pd.to_numeric(previous_close, errors="coerce")).astype(float)
        score.loc[pd.to_numeric(previous_close, errors="coerce").isna()] = np.nan
        return score
    if model_id == "previous_direction":
        return labels.groupby(features["ticker"]).shift(horizon)
    if model_id == "persistence_rule":
        lag_close = by_ticker["close"].shift(horizon)
        score = (close > pd.to_numeric(lag_close, errors="coerce")).astype(float)
        score.loc[pd.to_numeric(lag_close, errors="coerce").isna()] = np.nan
        return score
    if model_id == "moving_average_rule":
        ma = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").rolling(20, min_periods=5).mean())
        return (close > ma).astype(float).where(ma.notna(), np.nan)
    if model_id == "rolling_momentum_rule":
        lag = by_ticker["close"].shift(20)
        mom = close / pd.to_numeric(lag, errors="coerce") - 1.0
        return (mom > 0.0).astype(float).where(mom.notna(), np.nan)
    if model_id == "volatility_adjusted_momentum_rule":
        lag = by_ticker["close"].shift(20)
        mom = close / pd.to_numeric(lag, errors="coerce") - 1.0
        ret = close.groupby(features["ticker"]).pct_change(fill_method=None)
        vol = ret.groupby(features["ticker"]).transform(lambda values: values.rolling(20, min_periods=5).std())
        score = mom / vol.replace(0.0, np.nan)
        return sigmoid(score.fillna(0.0).to_numpy())
    if model_id == "sma_crossover":
        sma_short = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").rolling(10, min_periods=5).mean())
        sma_long = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").rolling(30, min_periods=10).mean())
        raw = (sma_short / sma_long.replace(0.0, np.nan) - 1.0) * 25.0
        return pd.Series(sigmoid(raw.fillna(0.0).to_numpy()), index=features.index)
    if model_id == "ema_crossover":
        ema_short = close.groupby(features["ticker"]).transform(lambda values: values.ewm(span=12, adjust=False, min_periods=12).mean())
        ema_long = close.groupby(features["ticker"]).transform(lambda values: values.ewm(span=26, adjust=False, min_periods=26).mean())
        raw = (ema_short / ema_long.replace(0.0, np.nan) - 1.0) * 25.0
        return pd.Series(sigmoid(raw.fillna(0.0).to_numpy()), index=features.index)
    if model_id == "macd_rule":
        if "macd_hist" in features.columns:
            raw = pd.to_numeric(features["macd_hist"], errors="coerce")
        else:
            ema_12 = close.groupby(features["ticker"]).transform(lambda values: values.ewm(span=12, adjust=False, min_periods=12).mean())
            ema_26 = close.groupby(features["ticker"]).transform(lambda values: values.ewm(span=26, adjust=False, min_periods=26).mean())
            macd = ema_12 - ema_26
            signal = macd.groupby(features["ticker"]).transform(lambda values: values.ewm(span=9, adjust=False, min_periods=9).mean())
            raw = macd - signal
        scale = raw.groupby(features["ticker"]).transform(lambda values: values.rolling(60, min_periods=10).std()).replace(0.0, np.nan)
        return pd.Series(sigmoid((raw / scale).fillna(0.0).to_numpy()), index=features.index)
    if model_id == "rsi_rule":
        rsi = pd.to_numeric(features["rsi_14"], errors="coerce") if "rsi_14" in features.columns else pd.Series(np.nan, index=features.index)
        score = (50.0 - rsi) / 20.0
        return pd.Series(sigmoid(score.fillna(0.0).to_numpy()), index=features.index)
    if model_id == "bollinger_band_rule":
        ma = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").rolling(20, min_periods=10).mean())
        sd = close.groupby(features["ticker"]).transform(lambda values: values.rolling(20, min_periods=10).std())
        lower = ma - 2.0 * sd
        upper = ma + 2.0 * sd
        raw = (ma - close) / (upper - lower).replace(0.0, np.nan)
        return pd.Series(np.clip(0.5 + raw.fillna(0.0).to_numpy(), 0.0, 1.0), index=features.index)
    if model_id == "price_momentum_rule":
        lag = by_ticker["close"].shift(20)
        raw = (close / pd.to_numeric(lag, errors="coerce") - 1.0) * 20.0
        return pd.Series(sigmoid(raw.fillna(0.0).to_numpy()), index=features.index)
    if model_id == "volume_momentum_rule":
        lag = by_ticker["close"].shift(5)
        ret5 = close / pd.to_numeric(lag, errors="coerce") - 1.0
        vol_ma = volume.groupby(features["ticker"]).transform(lambda values: values.rolling(20, min_periods=5).mean())
        vol_ratio = volume / vol_ma.replace(0.0, np.nan) - 1.0
        raw = ret5.fillna(0.0) * (1.0 + vol_ratio.clip(lower=-0.5, upper=2.0).fillna(0.0)) * 15.0
        return pd.Series(sigmoid(raw.to_numpy()), index=features.index)
    if model_id == "mean_reversion_rule":
        ma = by_ticker["close"].transform(lambda values: pd.to_numeric(values, errors="coerce").rolling(20, min_periods=10).mean())
        sd = close.groupby(features["ticker"]).transform(lambda values: values.rolling(20, min_periods=10).std()).replace(0.0, np.nan)
        raw = (ma - close) / sd
        return pd.Series(sigmoid(raw.fillna(0.0).to_numpy()), index=features.index)
    if model_id == "breakout_rule":
        rolling_high = high.groupby(features["ticker"]).transform(lambda values: values.rolling(20, min_periods=10).max().shift(1))
        rolling_low = low.groupby(features["ticker"]).transform(lambda values: values.rolling(20, min_periods=10).min().shift(1))
        raw = pd.Series(0.0, index=features.index)
        raw.loc[close > rolling_high] = 1.0
        raw.loc[close < rolling_low] = -1.0
        return pd.Series(sigmoid(raw.to_numpy() * 4.0), index=features.index)
    raise ValueError(f"unknown rule model: {model_id}")


def run_rule_models(features: pd.DataFrame, candidate_rows: list[dict[str, Any]], row_predictions: list[pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    groups = model_group_map()
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        train_y = labels.loc[idx["train"]].astype(int)
        majority = majority_value(train_y)
        for model_id in NAIVE_BASELINES + TECHNICAL_RULES:
            model_group = groups[model_id]
            try:
                score_series = pd.Series(score_series_rule(features, labels, horizon, model_id, majority), index=features.index)
                score_series = pd.to_numeric(score_series, errors="coerce").reindex(features.index).fillna(float(majority)).clip(0.0, 1.0)
                val_y = labels.loc[idx["validation"]].astype(int)
                specs = threshold_specs(val_y, score_series.loc[idx["validation"]].to_numpy(dtype=float), allow_validation_threshold=model_id in TECHNICAL_RULES)
                for threshold_policy, threshold in specs:
                    cid = candidate_id("universe", model_group, model_id, f"h{horizon}", threshold_policy, f"t{threshold:.3f}")
                    add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family="ex_ante_rule", horizon=horizon, threshold_policy=threshold_policy, planned_status="run")
                    val_score = score_series.loc[idx["validation"]].to_numpy(dtype=float)
                    final_score = score_series.loc[idx["final"]].to_numpy(dtype=float)
                    val_pred = (val_score >= threshold).astype(int)
                    final_pred = (final_score >= threshold).astype(int)
                    val_frame = prediction_frame(features, idx["validation"], labels, val_score, val_pred, model_group=model_group, model_id=model_id, feature_family="ex_ante_rule", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="validation")
                    final_frame = prediction_frame(features, idx["final"], labels, final_score, final_pred, model_group=model_group, model_id=model_id, feature_family="ex_ante_rule", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="final")
                    rows.append(result_row(candidate=cid, model_group=model_group, model_id=model_id, feature_family="ex_ante_rule", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, validation_frame=val_frame, final_frame=final_frame, train_rows=len(train_y), feature_count=0, implementation_note="deterministic ex-ante rule"))
                    if horizon == 40 or model_id in {"majority_class", "sma_crossover", "macd_rule"}:
                        row_predictions.extend([val_frame, final_frame])
            except Exception as exc:
                failures.append({"model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
                cid = candidate_id("universe", model_group, model_id, f"h{horizon}", "failed")
                add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family="ex_ante_rule", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", reason=str(exc)[:500])
    return rows, failures


def run_classifier_models(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    candidate_rows: list[dict[str, Any]],
    row_predictions: list[pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[BasePrediction]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ensemble_pool: dict[int, list[BasePrediction]] = {horizon: [] for horizon in HORIZONS}
    groups = model_group_map()
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        train_y = labels.loc[idx["train"]].astype(int)
        val_y = labels.loc[idx["validation"]].astype(int)
        if train_y.empty or val_y.empty or train_y.nunique() < 2:
            failures.append({"model_id": "all_classifiers", "horizon": horizon, "reason": "invalid train/validation label shape"})
            continue
        for feature_family in FEATURE_FAMILIES:
            cols = family_cols.get(feature_family, [])
            if not cols:
                for model_id in CLASSIFIER_MODEL_IDS:
                    model_group = groups[model_id]
                    cid = candidate_id("universe", model_group, model_id, feature_family, f"h{horizon}", "skipped")
                    add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", reason="no numeric feature columns")
                continue
            x_train = features.loc[idx["train"], cols]
            x_val = features.loc[idx["validation"], cols]
            x_final = features.loc[idx["final"], cols]
            for model_id in CLASSIFIER_MODEL_IDS:
                model_group = groups[model_id]
                if model_id == "catboost" and CatBoostClassifier is None:
                    cid = candidate_id("universe", model_group, model_id, feature_family, f"h{horizon}", "skipped")
                    add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", reason="catboost dependency is not installed")
                    continue
                if model_id in {"xgboost", "calibrated_xgboost"} and XGBClassifier is None:
                    cid = candidate_id("universe", model_group, model_id, feature_family, f"h{horizon}", "skipped")
                    add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", reason="xgboost dependency is not installed")
                    continue
                if model_id in {"lightgbm", "calibrated_lightgbm"} and LGBMClassifier is None:
                    cid = candidate_id("universe", model_group, model_id, feature_family, f"h{horizon}", "skipped")
                    add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", reason="lightgbm dependency is not installed")
                    continue
                try:
                    model = make_classifier(model_id, train_y)
                    if model is None:
                        raise RuntimeError("classifier implementation unavailable")
                    model.fit(x_train, train_y)
                    val_score = np.clip(predict_score(model, x_val), 0.0, 1.0)
                    final_score = np.clip(predict_score(model, x_final), 0.0, 1.0)
                    if len(val_score) != len(val_y) or len(final_score) != len(idx["final"]):
                        raise RuntimeError("prediction length mismatch")
                except Exception as exc:
                    failures.append({"model_id": model_id, "feature_family": feature_family, "horizon": horizon, "reason": str(exc)[:500]})
                    cid = candidate_id("universe", model_group, model_id, feature_family, f"h{horizon}", "failed")
                    add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", reason=str(exc)[:500])
                    continue
                specs = threshold_specs(val_y, val_score, allow_validation_threshold=True)
                stored_for_ensemble = False
                for threshold_policy, threshold in specs:
                    cid = candidate_id("universe", model_group, model_id, feature_family, f"h{horizon}", threshold_policy, f"t{threshold:.3f}")
                    add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy=threshold_policy, planned_status="run")
                    val_pred = (val_score >= threshold).astype(int)
                    final_pred = (final_score >= threshold).astype(int)
                    val_frame = prediction_frame(features, idx["validation"], labels, val_score, val_pred, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="validation")
                    final_frame = prediction_frame(features, idx["final"], labels, final_score, final_pred, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="final")
                    rows.append(result_row(candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, validation_frame=val_frame, final_frame=final_frame, train_rows=len(train_y), feature_count=len(cols)))
                    if model_id == "logistic_l2" and feature_family == "baseline_C_closest" and horizon == 40:
                        row_predictions.extend([val_frame, final_frame])
                    if feature_family == "baseline_C_closest" and model_id in ENSEMBLE_BASE_MODEL_IDS and threshold_policy == "fixed_0.50" and not stored_for_ensemble:
                        ensemble_pool[horizon].append(BasePrediction(model_id=model_id, candidate_id=cid, validation_score=val_score, final_score=final_score, validation_pred=val_pred, final_pred=final_pred, validation_accuracy=float(val_frame["correct"].mean())))
                        stored_for_ensemble = True
                gc.collect()
    return rows, failures, ensemble_pool


class LstmDirectionModel(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, 24, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, 1))

    def forward(self, x: Any) -> Any:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GruDirectionModel(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, 24, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, 1))

    def forward(self, x: Any) -> Any:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TcnDirectionModel(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 24, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(24, 24, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(24, 1)

    def forward(self, x: Any) -> Any:
        y = self.net(x.transpose(1, 2))
        return self.head(y[:, :, -1]).squeeze(-1)


class Cnn1dDirectionModel(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(input_dim, 32, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(32, 1)

    def forward(self, x: Any) -> Any:
        y = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(y).squeeze(-1)


class CnnLstmDirectionModel(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_dim, 24, kernel_size=3, padding=1), nn.ReLU())
        self.rnn = nn.LSTM(24, 20, batch_first=True)
        self.head = nn.Linear(20, 1)

    def forward(self, x: Any) -> Any:
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(conv)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_deep_model(model_id: str, input_dim: int) -> Any:
    if model_id == "lstm":
        return LstmDirectionModel(input_dim)
    if model_id == "gru":
        return GruDirectionModel(input_dim)
    if model_id == "tcn":
        return TcnDirectionModel(input_dim)
    if model_id == "cnn_1d":
        return Cnn1dDirectionModel(input_dim)
    if model_id == "cnn_lstm":
        return CnnLstmDirectionModel(input_dim)
    raise ValueError(f"unknown deep model: {model_id}")


def select_deep_cols(features: pd.DataFrame, cols: list[str], train_idx: pd.Index, limit: int = 32) -> list[str]:
    scored: list[tuple[str, float, int]] = []
    for col in cols:
        series = pd.to_numeric(features.loc[train_idx, col], errors="coerce")
        scored.append((col, float(series.var(skipna=True) or 0.0), int(series.notna().sum())))
    scored = [item for item in scored if item[2] > 100 and math.isfinite(item[1])]
    return [col for col, _var, _count in sorted(scored, key=lambda item: (item[2], item[1]), reverse=True)[:limit]]


def standardize_matrix(features: pd.DataFrame, cols: list[str], train_idx: pd.Index) -> pd.DataFrame:
    train_values = features.loc[train_idx, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = train_values.median(axis=0).fillna(0.0)
    scale = train_values.std(axis=0).replace(0.0, np.nan).fillna(1.0)
    out = features[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    return ((out - med) / scale).astype("float32")


def build_sequences(features: pd.DataFrame, matrix: pd.DataFrame, labels: pd.Series, idx: pd.Index, seq_len: int) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    idx_set = set(int(i) for i in idx)
    seqs: list[np.ndarray] = []
    ys: list[int] = []
    row_ids: list[int] = []
    times: list[pd.Timestamp] = []
    tickers: list[str] = []
    for ticker, group in features.groupby("ticker", sort=True):
        ordered = group.sort_values("datetime")
        values = matrix.loc[ordered.index].to_numpy(dtype=np.float32)
        indices = list(ordered.index)
        ordered_times = list(ordered["datetime"])
        for pos, row_id in enumerate(indices):
            if int(row_id) not in idx_set or pos + 1 < seq_len:
                continue
            y = labels.at[row_id]
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
    return np.stack(seqs).astype(np.float32)[order], np.asarray(ys, dtype=np.float32)[order], pd.Index(np.asarray(row_ids, dtype=int)[order])


def predict_deep(model: Any, x: np.ndarray) -> np.ndarray:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("torch unavailable")
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=2048, shuffle=False)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            probs.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(probs) if probs else np.asarray([], dtype=float)


def fit_deep(model_id: str, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[Any, dict[str, Any], np.ndarray]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("torch unavailable")
    torch.manual_seed(RANDOM_STATE)
    model = make_deep_model(model_id, int(x_train.shape[2]))
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives > 0 else torch.tensor([1.0], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))), batch_size=512, shuffle=False)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = math.inf
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(yb)
            total_rows += len(yb)
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val.astype(np.float32))), batch_size=4096, shuffle=False)
            val_losses: list[float] = []
            val_probs: list[np.ndarray] = []
            for xb, yb in val_loader:
                logits = model(xb)
                val_losses.append(float(criterion(logits, yb).detach()) * len(yb))
                val_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            val_prob = np.concatenate(val_probs) if val_probs else np.asarray([], dtype=float)
            val_loss = float(sum(val_losses) / max(len(y_val), 1))
            val_acc = accuracy(y_val.astype(int), (val_prob >= 0.50).astype(int))
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_rows, 1), "validation_loss": val_loss, "validation_accuracy": val_acc})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            patience = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 1:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history}, predict_deep(model, x_val)


def run_deep_models(features: pd.DataFrame, family_cols: dict[str, list[str]], candidate_rows: list[dict[str, Any]], row_predictions: list[pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    groups = model_group_map()
    deep_sequence_models = ["lstm", "gru", "tcn", "cnn_1d", "cnn_lstm"]
    if torch is None:
        for model_id in deep_sequence_models:
            add_grid_row(candidate_rows, candidate=candidate_id("universe", "neural_deep_models", model_id, "skipped"), model_group=groups[model_id], model_id=model_id, feature_family="sequence", horizon=0, threshold_policy="not_run", planned_status="skipped_with_reason", reason="torch dependency is not installed")
        return rows, [{"model_id": model_id, "reason": "torch dependency is not installed"} for model_id in deep_sequence_models]
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        for feature_family in DEEP_FEATURE_FAMILIES:
            base_cols = family_cols.get(feature_family, [])
            cols = select_deep_cols(features, base_cols, idx["train"], limit=32)
            if not cols:
                for model_id in deep_sequence_models:
                    failures.append({"model_id": model_id, "feature_family": feature_family, "horizon": horizon, "reason": "no numeric sequence feature columns"})
                continue
            matrix = standardize_matrix(features, cols, idx["train"])
            for seq_len in (16, 32, 64):
                x_train, y_train, train_rows = build_sequences(features, matrix, labels, idx["train"], seq_len)
                x_val, y_val, val_rows = build_sequences(features, matrix, labels, idx["validation"], seq_len)
                x_final, y_final, final_rows = build_sequences(features, matrix, labels, idx["final"], seq_len)
                if len(y_train) < 100 or len(y_val) < 100 or len(y_final) == 0 or len(np.unique(y_train.astype(int))) < 2:
                    reason = f"invalid sequence shape train={x_train.shape} validation={x_val.shape} final={x_final.shape}"
                    for model_id in deep_sequence_models:
                        failures.append({"model_id": model_id, "feature_family": feature_family, "horizon": horizon, "sequence_length": seq_len, "reason": reason})
                        cid = candidate_id("universe", groups[model_id], model_id, feature_family, f"h{horizon}", f"seq{seq_len}", "failed")
                        add_grid_row(candidate_rows, candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", reason=reason)
                    continue
                for model_id in deep_sequence_models:
                    model_group = groups[model_id]
                    try:
                        model, fit_info, val_score = fit_deep(model_id, x_train, y_train, x_val, y_val)
                        final_score = np.clip(predict_deep(model, x_final), 0.0, 1.0)
                        val_score = np.clip(val_score, 0.0, 1.0)
                    except Exception as exc:
                        failures.append({"model_id": model_id, "feature_family": feature_family, "horizon": horizon, "sequence_length": seq_len, "reason": str(exc)[:500]})
                        cid = candidate_id("universe", model_group, model_id, feature_family, f"h{horizon}", f"seq{seq_len}", "failed")
                        add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", reason=str(exc)[:500])
                        continue
                    val_y_series = pd.Series(y_val.astype(int), index=val_rows)
                    for threshold_policy, threshold in threshold_specs(val_y_series, val_score, allow_validation_threshold=True):
                        cid = candidate_id("universe", model_group, model_id, feature_family, f"h{horizon}", f"seq{seq_len}", threshold_policy, f"t{threshold:.3f}")
                        add_grid_row(candidate_rows, candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy=threshold_policy, planned_status="run")
                        val_pred = (val_score >= threshold).astype(int)
                        final_pred = (final_score >= threshold).astype(int)
                        val_frame = prediction_frame(features, val_rows, labels, val_score, val_pred, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="validation")
                        final_frame = prediction_frame(features, final_rows, labels, final_score, final_pred, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="final")
                        rows.append(result_row(candidate=cid, model_group=model_group, model_id=model_id, feature_family=feature_family, horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, validation_frame=val_frame, final_frame=final_frame, train_rows=len(y_train), feature_count=len(cols), sequence_length=seq_len, implementation_note=f"early stopping validation only; best_epoch={fit_info.get('best_epoch')}"))
                    gc.collect()
    return rows, failures


def run_regime_models(features: pd.DataFrame, family_cols: dict[str, list[str]], candidate_rows: list[dict[str, Any]], row_predictions: list[pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    groups = model_group_map()
    context_map = {
        "regime_context_logistic": "logistic_l2",
        "regime_context_xgboost": "xgboost",
        "regime_context_lightgbm": "lightgbm",
    }
    for model_id, base_model_id in context_map.items():
        if base_model_id == "xgboost" and XGBClassifier is None:
            add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, "skipped"), model_group=groups[model_id], model_id=model_id, feature_family="regime_context", horizon=0, threshold_policy="not_run", planned_status="skipped_with_reason", reason="xgboost dependency is not installed")
            continue
        if base_model_id == "lightgbm" and LGBMClassifier is None:
            add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, "skipped"), model_group=groups[model_id], model_id=model_id, feature_family="regime_context", horizon=0, threshold_policy="not_run", planned_status="skipped_with_reason", reason="lightgbm dependency is not installed")
            continue
        for horizon in HORIZONS:
            labels = add_absolute_labels(features, horizon)
            idx = split_indices(features, labels)
            train_y = labels.loc[idx["train"]].astype(int)
            val_y = labels.loc[idx["validation"]].astype(int)
            cols = family_cols.get("regime_context", [])
            try:
                model = make_classifier(base_model_id, train_y)
                if model is None:
                    raise RuntimeError("base classifier unavailable")
                model.fit(features.loc[idx["train"], cols], train_y)
                val_score = np.clip(predict_score(model, features.loc[idx["validation"], cols]), 0.0, 1.0)
                final_score = np.clip(predict_score(model, features.loc[idx["final"], cols]), 0.0, 1.0)
            except Exception as exc:
                failures.append({"model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
                add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, f"h{horizon}", "failed"), model_group=groups[model_id], model_id=model_id, feature_family="regime_context", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", reason=str(exc)[:500])
                continue
            for threshold_policy, threshold in threshold_specs(val_y, val_score, True):
                cid = candidate_id("universe", groups[model_id], model_id, "regime_context", f"h{horizon}", threshold_policy, f"t{threshold:.3f}")
                add_grid_row(candidate_rows, candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="regime_context", horizon=horizon, threshold_policy=threshold_policy, planned_status="run")
                val_frame = prediction_frame(features, idx["validation"], labels, val_score, (val_score >= threshold).astype(int), model_group=groups[model_id], model_id=model_id, feature_family="regime_context", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="validation")
                final_frame = prediction_frame(features, idx["final"], labels, final_score, (final_score >= threshold).astype(int), model_group=groups[model_id], model_id=model_id, feature_family="regime_context", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="final")
                rows.append(result_row(candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="regime_context", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, validation_frame=val_frame, final_frame=final_frame, train_rows=len(train_y), feature_count=len(cols), implementation_note="regime-context features are lagged/ex-ante"))
    rows2, failures2 = run_regime_routers(features, family_cols, candidate_rows)
    rows.extend(rows2)
    failures.extend(failures2)
    return rows, failures


def fit_router_models(features: pd.DataFrame, labels: pd.Series, idx: dict[str, pd.Index], cols: list[str], group_col: str) -> tuple[np.ndarray, np.ndarray]:
    train_y = labels.loc[idx["train"]].astype(int)
    global_model = make_classifier("logistic_l2", train_y)
    if global_model is None:
        raise RuntimeError("global router classifier unavailable")
    global_model.fit(features.loc[idx["train"], cols], train_y)
    val_score = np.clip(predict_score(global_model, features.loc[idx["validation"], cols]), 0.0, 1.0)
    final_score = np.clip(predict_score(global_model, features.loc[idx["final"], cols]), 0.0, 1.0)
    for group_name, train_group in features.loc[idx["train"]].groupby(group_col, sort=True):
        train_ids = train_group.index
        y_group = labels.loc[train_ids].astype(int)
        if len(y_group) < 100 or y_group.nunique() < 2:
            continue
        model = clone(global_model)
        model.fit(features.loc[train_ids, cols], y_group)
        val_mask = features.loc[idx["validation"], group_col].astype(str).eq(str(group_name)).to_numpy()
        final_mask = features.loc[idx["final"], group_col].astype(str).eq(str(group_name)).to_numpy()
        if val_mask.any():
            val_score[val_mask] = np.clip(predict_score(model, features.loc[idx["validation"], cols].loc[val_mask]), 0.0, 1.0)
        if final_mask.any():
            final_score[final_mask] = np.clip(predict_score(model, features.loc[idx["final"], cols].loc[final_mask]), 0.0, 1.0)
    return val_score, final_score


def group_thresholds(y_true: pd.Series, score: np.ndarray, groups: pd.Series) -> dict[str, float]:
    work = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "score": score, "group": groups.astype(str).to_numpy()})
    thresholds: dict[str, float] = {}
    for group_name, group in work.groupby("group", sort=True):
        thresholds[group_name] = select_threshold(group["y"], group["score"].to_numpy(dtype=float))[0] if len(group) >= 40 else 0.50
    return thresholds


def apply_group_thresholds(score: np.ndarray, groups: pd.Series, thresholds: dict[str, float], default: float = 0.50) -> np.ndarray:
    group_values = groups.astype(str).to_numpy()
    return np.asarray([int(float(s) >= thresholds.get(str(g), default)) for s, g in zip(score, group_values)], dtype=int)


def run_regime_routers(features: pd.DataFrame, family_cols: dict[str, list[str]], candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    groups = model_group_map()
    cols = family_cols.get("baseline_C_closest", [])
    router_specs = {
        "bull_bear_sideway_router": "market_direction_regime",
        "high_low_volatility_router": "volatility_regime",
        "regime_model_router": "regime_router_key",
    }
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        val_y = labels.loc[idx["validation"]].astype(int)
        train_y = labels.loc[idx["train"]].astype(int)
        for model_id, group_col in router_specs.items():
            try:
                val_score, final_score = fit_router_models(features, labels, idx, cols, group_col)
            except Exception as exc:
                failures.append({"model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
                add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, f"h{horizon}", "failed"), model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", reason=str(exc)[:500])
                continue
            for threshold_policy, threshold in threshold_specs(val_y, val_score, True):
                cid = candidate_id("universe", groups[model_id], model_id, "baseline_C_closest", f"h{horizon}", threshold_policy, f"t{threshold:.3f}")
                add_grid_row(candidate_rows, candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy=threshold_policy, planned_status="run")
                val_frame = prediction_frame(features, idx["validation"], labels, val_score, (val_score >= threshold).astype(int), model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="validation")
                final_frame = prediction_frame(features, idx["final"], labels, final_score, (final_score >= threshold).astype(int), model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="final")
                rows.append(result_row(candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, validation_frame=val_frame, final_frame=final_frame, train_rows=len(train_y), feature_count=len(cols), implementation_note=f"router trained by {group_col}; validation only selection"))
        model_id = "regime_threshold_router"
        try:
            model = make_classifier("logistic_l2", train_y)
            if model is None:
                raise RuntimeError("logistic base unavailable")
            model.fit(features.loc[idx["train"], cols], train_y)
            val_score = np.clip(predict_score(model, features.loc[idx["validation"], cols]), 0.0, 1.0)
            final_score = np.clip(predict_score(model, features.loc[idx["final"], cols]), 0.0, 1.0)
            thresholds = group_thresholds(val_y, val_score, features.loc[idx["validation"], "regime_router_key"])
            val_pred = apply_group_thresholds(val_score, features.loc[idx["validation"], "regime_router_key"], thresholds)
            final_pred = apply_group_thresholds(final_score, features.loc[idx["final"], "regime_router_key"], thresholds)
            cid = candidate_id("universe", groups[model_id], model_id, "baseline_C_closest", f"h{horizon}", "validation_selected_regime_thresholds")
            add_grid_row(candidate_rows, candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="validation_selected_threshold", planned_status="run")
            val_frame = prediction_frame(features, idx["validation"], labels, val_score, val_pred, model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="validation_selected_threshold", threshold=0.50, candidate=cid, split="validation")
            final_frame = prediction_frame(features, idx["final"], labels, final_score, final_pred, model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="validation_selected_threshold", threshold=0.50, candidate=cid, split="final")
            rows.append(result_row(candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="validation_selected_threshold", threshold=0.50, validation_frame=val_frame, final_frame=final_frame, train_rows=len(train_y), feature_count=len(cols), implementation_note="per-regime thresholds selected on validation only"))
        except Exception as exc:
            failures.append({"model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
            add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, f"h{horizon}", "failed"), model_group=groups[model_id], model_id=model_id, feature_family="baseline_C_closest", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", reason=str(exc)[:500])
    return rows, failures


def run_ensembles(features: pd.DataFrame, ensemble_pool: dict[int, list[BasePrediction]], candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    groups = model_group_map()
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        val_y = labels.loc[idx["validation"]].astype(int)
        train_rows = len(labels.loc[idx["train"]].dropna())
        pool = ensemble_pool.get(horizon, [])
        if len(pool) < 2:
            reason = "fewer than two successful baseline base models"
            for model_id in ENSEMBLE_MODELS:
                failures.append({"model_id": model_id, "horizon": horizon, "reason": reason})
                add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, f"h{horizon}", "skipped"), model_group=groups[model_id], model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", reason=reason)
            continue
        val_matrix = np.column_stack([item.validation_score for item in pool])
        final_matrix = np.column_stack([item.final_score for item in pool])
        base_names = [item.model_id for item in pool]
        base_acc = np.asarray([max(item.validation_accuracy, 0.0) for item in pool], dtype=float)
        weights = base_acc / base_acc.sum() if base_acc.sum() > 0 else np.repeat(1.0 / len(pool), len(pool))
        scores: dict[str, tuple[np.ndarray, np.ndarray, str]] = {
            "hard_voting": ((val_matrix >= 0.50).mean(axis=1), (final_matrix >= 0.50).mean(axis=1), "mean hard vote fraction"),
            "soft_voting": (val_matrix.mean(axis=1), final_matrix.mean(axis=1), "unweighted mean probability"),
            "validation_weighted_soft_vote": (val_matrix @ weights, final_matrix @ weights, "weights proportional to validation accuracy"),
            "blending": (0.5 * val_matrix.mean(axis=1) + 0.5 * val_matrix[:, int(base_acc.argmax())], 0.5 * final_matrix.mean(axis=1) + 0.5 * final_matrix[:, int(base_acc.argmax())], "blend of soft vote and best validation base model"),
        }
        meta_specs: dict[str, Any] = {
            "stacking_logistic_meta": LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        }
        if LGBMClassifier is not None:
            meta_specs["stacking_lightgbm_meta"] = LGBMClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, min_child_samples=40, random_state=RANDOM_STATE, verbose=-1, n_jobs=2)
        else:
            failures.append({"model_id": "stacking_lightgbm_meta", "horizon": horizon, "reason": "lightgbm dependency is not installed"})
        if XGBClassifier is not None:
            meta_specs["stacking_xgboost_meta"] = XGBClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, min_child_weight=20, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=2)
        else:
            failures.append({"model_id": "stacking_xgboost_meta", "horizon": horizon, "reason": "xgboost dependency is not installed"})
        for model_id, estimator in meta_specs.items():
            try:
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
                pipe.fit(val_matrix, val_y)
                scores[model_id] = (np.clip(predict_score(pipe, pd.DataFrame(val_matrix, columns=base_names)), 0.0, 1.0), np.clip(predict_score(pipe, pd.DataFrame(final_matrix, columns=base_names)), 0.0, 1.0), "meta model trained on validation base predictions only")
            except Exception as exc:
                failures.append({"model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
                add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, f"h{horizon}", "failed"), model_group=groups[model_id], model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy="not_run", planned_status="failed_with_reason", reason=str(exc)[:500])
        for model_id in ENSEMBLE_MODELS:
            if model_id not in scores:
                add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, f"h{horizon}", "skipped"), model_group=groups[model_id], model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy="not_run", planned_status="skipped_with_reason", reason="dependency missing or meta fit failed")
                continue
            val_score, final_score, note = scores[model_id]
            for threshold_policy, threshold in threshold_specs(val_y, val_score, True):
                cid = candidate_id("universe", groups[model_id], model_id, "validation_selected_base_models", f"h{horizon}", threshold_policy, f"t{threshold:.3f}")
                add_grid_row(candidate_rows, candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy=threshold_policy, planned_status="run")
                val_frame = prediction_frame(features, idx["validation"], labels, val_score, (val_score >= threshold).astype(int), model_group=groups[model_id], model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="validation")
                final_frame = prediction_frame(features, idx["final"], labels, final_score, (final_score >= threshold).astype(int), model_group=groups[model_id], model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="final")
                rows.append(result_row(candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="validation_selected_base_models", horizon=horizon, threshold_policy=threshold_policy, threshold=threshold, validation_frame=val_frame, final_frame=final_frame, train_rows=train_rows, feature_count=len(base_names), implementation_note=f"{note}; base_models={','.join(base_names)}"))
    return rows, failures


def run_statistical_models(features: pd.DataFrame, candidate_rows: list[dict[str, Any]], row_predictions: list[pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    groups = model_group_map()
    if ARIMA is None or SARIMAX is None or ExponentialSmoothing is None:
        reason = "statsmodels implementation is not available"
        for model_id in ["arima_direction", "sarima_direction", "ets_direction", "var_direction"]:
            failures.append({"model_id": model_id, "reason": reason})
            add_grid_row(candidate_rows, candidate=candidate_id("universe", groups[model_id], model_id, "skipped"), model_group=groups[model_id], model_id=model_id, feature_family="statistical_series", horizon=0, threshold_policy="not_run", planned_status="skipped_with_reason", reason=reason)
        garch_rows, garch_text = run_garch_diagnostic(features, candidate_rows)
        summary_rows.extend(garch_rows)
        return rows, failures, pd.DataFrame(summary_rows), garch_text
    ticker_train_signs: dict[tuple[str, int, str], int] = {}
    fit_notes: dict[tuple[str, int, str], str] = {}
    for horizon in HORIZONS:
        for ticker, group in features.groupby("ticker", sort=True):
            train_close = pd.to_numeric(group.loc[group["datetime"].le(TRAIN_END), "close"], errors="coerce").dropna()
            returns = train_close.pct_change(fill_method=None).dropna()
            for model_id in ["arima_direction", "sarima_direction", "ets_direction"]:
                key = (model_id, horizon, str(ticker))
                try:
                    if len(train_close) < 120 or len(returns) < 100:
                        raise RuntimeError("insufficient train series length")
                    if model_id == "arima_direction":
                        fit = ARIMA(returns, order=(1, 0, 1)).fit()
                        forecast = float(np.asarray(fit.forecast(steps=horizon))[-1])
                        sign = int(forecast > 0.0)
                    elif model_id == "sarima_direction":
                        fit = SARIMAX(returns, order=(1, 0, 1), seasonal_order=(1, 0, 0, 5), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                        forecast = float(np.asarray(fit.forecast(steps=horizon))[-1])
                        sign = int(forecast > 0.0)
                    else:
                        fit = ExponentialSmoothing(train_close, trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True)
                        forecast_level = float(np.asarray(fit.forecast(steps=horizon))[-1])
                        sign = int(forecast_level > float(train_close.iloc[-1]))
                    ticker_train_signs[key] = sign
                    fit_notes[key] = "ok"
                except Exception as exc:
                    ticker_train_signs[key] = 1
                    fit_notes[key] = f"fallback_to_up_due_fit_failure: {str(exc)[:120]}"
        if VAR is not None:
            model_id = "var_direction"
            try:
                pivot = features.loc[features["datetime"].le(TRAIN_END)].pivot_table(index="datetime", columns="ticker", values="close", aggfunc="last").sort_index()
                ret = pivot.pct_change(fill_method=None).dropna(how="all").fillna(0.0)
                ret = ret.loc[:, ret.notna().sum() > 100]
                if ret.shape[1] < 2:
                    raise RuntimeError("insufficient multivariate train series")
                fit = VAR(ret).fit(maxlags=2, ic=None, trend="c")
                forecast = fit.forecast(ret.values[-fit.k_ar :], steps=horizon)
                signs = pd.Series((forecast[-1] > 0.0).astype(int), index=ret.columns)
                for ticker in features["ticker"].astype(str).unique():
                    ticker_train_signs[(model_id, horizon, str(ticker))] = int(signs.get(str(ticker), 1))
                    fit_notes[(model_id, horizon, str(ticker))] = "ok" if str(ticker) in signs.index else "fallback_missing_var_column"
            except Exception as exc:
                failures.append({"model_id": model_id, "horizon": horizon, "reason": str(exc)[:500]})
                for ticker in features["ticker"].astype(str).unique():
                    ticker_train_signs[(model_id, horizon, str(ticker))] = 1
                    fit_notes[(model_id, horizon, str(ticker))] = f"fallback_to_up_due_fit_failure: {str(exc)[:120]}"
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        train_y = labels.loc[idx["train"]].astype(int)
        for model_id in ["arima_direction", "sarima_direction", "ets_direction", "var_direction"]:
            score = pd.Series(index=features.index, dtype=float)
            for ticker in features["ticker"].astype(str).unique():
                sign = ticker_train_signs.get((model_id, horizon, ticker), 1)
                score.loc[features["ticker"].astype(str).eq(ticker)] = float(sign)
            cid = candidate_id("universe", groups[model_id], model_id, "statistical_series", f"h{horizon}", "fixed_0p50")
            add_grid_row(candidate_rows, candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="statistical_series", horizon=horizon, threshold_policy="fixed_0.50", planned_status="run")
            val_score = score.loc[idx["validation"]].fillna(1.0).to_numpy(dtype=float)
            final_score = score.loc[idx["final"]].fillna(1.0).to_numpy(dtype=float)
            val_frame = prediction_frame(features, idx["validation"], labels, val_score, (val_score >= 0.50).astype(int), model_group=groups[model_id], model_id=model_id, feature_family="statistical_series", horizon=horizon, threshold_policy="fixed_0.50", threshold=0.50, candidate=cid, split="validation")
            final_frame = prediction_frame(features, idx["final"], labels, final_score, (final_score >= 0.50).astype(int), model_group=groups[model_id], model_id=model_id, feature_family="statistical_series", horizon=horizon, threshold_policy="fixed_0.50", threshold=0.50, candidate=cid, split="final")
            rows.append(result_row(candidate=cid, model_group=groups[model_id], model_id=model_id, feature_family="statistical_series", horizon=horizon, threshold_policy="fixed_0.50", threshold=0.50, validation_frame=val_frame, final_frame=final_frame, train_rows=len(train_y), feature_count=1, implementation_note="train-window statistical forecast sign converted to direction"))
            notes = [fit_notes.get((model_id, horizon, str(ticker)), "") for ticker in features["ticker"].astype(str).unique()]
            summary_rows.append({"model_id": model_id, "horizon": horizon, "ticker_fits_ok": sum(note == "ok" for note in notes), "ticker_fit_fallbacks": sum(note != "ok" for note in notes), "validation_accuracy": rows[-1]["validation_accuracy"], "final_accuracy": rows[-1]["final_accuracy"], "claim_role": "direction_by_forecast_sign"})
    garch_rows, garch_text = run_garch_diagnostic(features, candidate_rows)
    summary_rows.extend(garch_rows)
    return rows, failures, pd.DataFrame(summary_rows), garch_text


def run_garch_diagnostic(features: pd.DataFrame, candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    model_id = "garch_volatility_diagnostic"
    groups = model_group_map()
    if not dependency_available("arch"):
        add_grid_row(
            candidate_rows,
            candidate=candidate_id("universe", groups[model_id], model_id, "garch_diagnostic", "skipped"),
            model_group=groups[model_id],
            model_id=model_id,
            feature_family="garch_diagnostic",
            horizon=0,
            threshold_policy="not_applicable",
            planned_status="skipped_with_reason",
            reason="arch dependency is not installed",
        )
        return [], "\n".join(
            [
                "# GARCH Diagnostic Summary",
                "",
                "- Attempted: yes.",
                "- Status: skipped_with_reason.",
                "- Reason: `arch` dependency is not installed in the intended Python environment.",
                "- Direction role: not used as a main direction classifier.",
                "- Claim eligible: no.",
            ]
        )

    try:
        from arch import arch_model
    except Exception as exc:
        add_grid_row(
            candidate_rows,
            candidate=candidate_id("universe", groups[model_id], model_id, "garch_diagnostic", "failed"),
            model_group=groups[model_id],
            model_id=model_id,
            feature_family="garch_diagnostic",
            horizon=0,
            threshold_policy="not_applicable",
            planned_status="failed_with_reason",
            reason=str(exc)[:500],
        )
        return [], "\n".join(
            [
                "# GARCH Diagnostic Summary",
                "",
                "- Attempted: yes.",
                "- Status: failed_with_reason.",
                f"- Reason: `arch` import failed: {str(exc)[:500]}.",
                "- Direction role: not used as a main direction classifier.",
                "- Claim eligible: no.",
            ]
        )

    horizon_values: dict[int, list[float]] = {horizon: [] for horizon in HORIZONS}
    fit_notes: dict[str, str] = {}
    max_horizon = max(HORIZONS)
    for ticker, group in features.groupby("ticker", sort=True):
        close = pd.to_numeric(group.loc[group["datetime"].le(TRAIN_END), "close"], errors="coerce").dropna()
        returns = close.pct_change(fill_method=None).dropna() * 100.0
        try:
            if len(returns) < 250:
                raise RuntimeError("insufficient train return series length")
            fit = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(
                disp="off",
                show_warning=False,
                options={"maxiter": 200},
            )
            forecast = fit.forecast(horizon=max_horizon, reindex=False)
            variance = np.asarray(forecast.variance.iloc[-1], dtype=float)
            if len(variance) < max_horizon:
                raise RuntimeError("GARCH forecast returned too few horizons")
            for horizon in HORIZONS:
                horizon_values[horizon].append(float(math.sqrt(max(variance[horizon - 1], 0.0))))
            fit_notes[str(ticker)] = "ok"
        except Exception as exc:
            fit_notes[str(ticker)] = f"fit_failed: {str(exc)[:120]}"

    summary_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        values = np.asarray(horizon_values[horizon], dtype=float)
        add_grid_row(
            candidate_rows,
            candidate=candidate_id("universe", groups[model_id], model_id, "garch_diagnostic", f"h{horizon}", "not_recommended"),
            model_group=groups[model_id],
            model_id=model_id,
            feature_family="garch_diagnostic",
            horizon=horizon,
            threshold_policy="not_applicable",
            planned_status="not_recommended_with_reason",
            reason="GARCH is a volatility diagnostic and not a direct directional classifier",
        )
        summary_rows.append(
            {
                "model_id": model_id,
                "horizon": horizon,
                "ticker_fits_ok": sum(note == "ok" for note in fit_notes.values()),
                "ticker_fit_fallbacks": sum(note != "ok" for note in fit_notes.values()),
                "validation_accuracy": math.nan,
                "final_accuracy": math.nan,
                "claim_role": "volatility_diagnostic_only",
                "mean_forecast_volatility_pct": float(np.nanmean(values)) if values.size else math.nan,
                "median_forecast_volatility_pct": float(np.nanmedian(values)) if values.size else math.nan,
                "fit_status": "ok" if values.size else "failed_with_reason",
            }
        )

    detail_frame = pd.DataFrame(
        {
            "ticker": list(fit_notes.keys()),
            "fit_note": list(fit_notes.values()),
        }
    )
    status = "not_recommended_with_reason" if any(note == "ok" for note in fit_notes.values()) else "failed_with_reason"
    text_lines = [
            "# GARCH Diagnostic Summary",
            "",
            "- Attempted: yes.",
            f"- Status: {status}.",
            f"- arch version: {dependency_version('arch')}.",
            f"- Ticker fits ok: {sum(note == 'ok' for note in fit_notes.values())}.",
            f"- Ticker fit failures: {sum(note != 'ok' for note in fit_notes.values())}.",
            "- Reason: GARCH is a volatility diagnostic and not a direct directional classifier in this benchmark.",
            "- Direction role: not used as a main direction classifier.",
            "- Claim eligible: no.",
            "",
            "## Horizon Forecast Volatility",
            "",
            markdown_table(pd.DataFrame(summary_rows), max_rows=len(summary_rows)),
            "",
            "## Fit Notes",
            "",
            markdown_table(detail_frame, max_rows=len(detail_frame)),
        ]
    return summary_rows, "\n".join(text_lines)


def update_selection(final_results: pd.DataFrame) -> pd.DataFrame:
    out = final_results.copy()
    if out.empty:
        return out
    selection_pool = out[
        out["status"].eq("ok")
        & out["full_ticker_coverage"].astype(bool)
        & out["validation_accuracy"].apply(lambda value: math.isfinite(as_float(value)))
        & ~out["model_id"].eq("garch_volatility_diagnostic")
    ].copy()
    if selection_pool.empty:
        return out
    selected = selection_pool.sort_values(["validation_accuracy", "validation_rows", "candidate_id"], ascending=[False, False, True]).iloc[0]
    selected_id = str(selected["candidate_id"])
    out.loc[out["candidate_id"].astype(str).eq(selected_id), "selected_by_validation_yes_no"] = "yes"
    mask = out["candidate_id"].astype(str).eq(selected_id)
    out.loc[mask, "claim_eligible_yes_no"] = np.where(
        out.loc[mask, "full_ticker_coverage"].astype(bool),
        "yes",
        "no",
    )
    out.loc[mask & out["claim_eligible_yes_no"].eq("yes"), "reason_not_claim_eligible"] = ""
    out.loc[mask & out["claim_eligible_yes_no"].eq("no"), "reason_not_claim_eligible"] = "selected but missing full ticker coverage"
    risk_values = out.apply(classify_overfit_risk, axis=1)
    out["overfit_risk"] = [risk for risk, _ in risk_values]
    out["overfit_risk_reason"] = [reason for _, reason in risk_values]
    return out


def update_registry_from_results(registry: pd.DataFrame, candidate_grid: pd.DataFrame, final_results: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    out = registry.copy()
    run_ids = set(final_results.loc[final_results["status"].eq("ok"), "model_id"].astype(str)) if not final_results.empty else set()
    failed = failures.groupby("model_id")["reason"].apply(lambda values: "; ".join(sorted({str(v) for v in values if str(v)}))[:1000]).to_dict() if not failures.empty else {}
    skipped_grid = candidate_grid[candidate_grid["planned_status"].eq("skipped_with_reason")] if not candidate_grid.empty else pd.DataFrame()
    skipped = skipped_grid.groupby("model_id")["reason"].apply(lambda values: "; ".join(sorted({str(v) for v in values if str(v)}))[:1000]).to_dict() if not skipped_grid.empty else {}
    for idx, row in out.iterrows():
        model_id = str(row["model_id"])
        if model_id in run_ids:
            out.at[idx, "run_status"] = "run"
        elif model_id == "garch_volatility_diagnostic":
            if dependency_available("arch"):
                out.at[idx, "run_status"] = "not_recommended_with_reason"
            else:
                out.at[idx, "run_status"] = "skipped_with_reason"
                out.at[idx, "reason_if_skipped"] = "arch dependency is not installed"
        elif model_id in skipped:
            out.at[idx, "run_status"] = "skipped_with_reason"
            out.at[idx, "reason_if_skipped"] = skipped[model_id]
        elif model_id in failed:
            out.at[idx, "run_status"] = "failed_with_reason"
            out.at[idx, "reason_if_failed"] = failed[model_id]
        else:
            out.at[idx, "run_status"] = "failed_with_reason"
            out.at[idx, "reason_if_failed"] = "planned model has no successful result and no recorded skip"
        if model_id in failed:
            out.at[idx, "reason_if_failed"] = failed[model_id]
        dep = str(row["dependency_required"])
        if dep in {"catboost", "arch"} and not dependency_available(dep):
            out.at[idx, "implementation_status"] = "dependency_missing"
        if model_id == "garch_volatility_diagnostic":
            out.at[idx, "claim_eligible"] = "no"
    return out


def write_registry_outputs(registry: pd.DataFrame) -> None:
    write_csv(OUTPUT_DIR / "model_universe_registry.csv", registry)
    counts = registry_status_counts(registry)
    lines = [
        "# Model Universe Registry",
        "",
        f"- Total model groups listed: {registry['model_group'].nunique()}.",
        f"- Total model variants planned: {len(registry)}.",
        f"- Total model variants attempted: {counts['attempted']}.",
        f"- Total model variants run: {counts['run']}.",
        f"- Total model variants failed: {counts['failed']}.",
        f"- Total model variants skipped: {counts['skipped']}.",
        f"- Total model variants not recommended: {counts['not_recommended']}.",
        f"- CatBoost status: {model_status(registry, 'catboost')}.",
        f"- GARCH diagnostic status: {model_status(registry, 'garch_volatility_diagnostic')}.",
        "- GARCH used as main directional classifier: no.",
        "",
    ]
    lines.append(markdown_table(registry, max_rows=len(registry)))
    write_markdown(OUTPUT_DIR / "model_universe_registry.md", "\n".join(lines))


def registry_status_counts(registry: pd.DataFrame) -> dict[str, int]:
    status = registry["run_status"].astype(str) if "run_status" in registry.columns else pd.Series(dtype=str)
    return {
        "run": int(status.eq("run").sum()),
        "failed": int(status.eq("failed_with_reason").sum()),
        "skipped": int(status.eq("skipped_with_reason").sum()),
        "not_recommended": int(status.eq("not_recommended_with_reason").sum()),
        "attempted": int(status.isin(["run", "failed_with_reason", "skipped_with_reason", "not_recommended_with_reason"]).sum()),
    }


def model_status(registry: pd.DataFrame, model_id: str) -> str:
    rows = registry[registry["model_id"].astype(str).eq(model_id)]
    if rows.empty:
        return "missing"
    return str(rows.iloc[0].get("run_status", "unknown"))


def write_reports(
    registry: pd.DataFrame,
    candidate_grid: pd.DataFrame,
    final_results: pd.DataFrame,
    failures: pd.DataFrame,
    statistical_summary: pd.DataFrame,
    garch_text: str,
) -> None:
    validation_results = final_results.copy()
    counts = registry_status_counts(registry)
    catboost_status = model_status(registry, "catboost")
    garch_status = model_status(registry, "garch_volatility_diagnostic")
    write_csv(OUTPUT_DIR / "candidate_grid.csv", candidate_grid)
    write_csv(OUTPUT_DIR / "validation_results.csv", validation_results)
    write_csv(OUTPUT_DIR / "final_results.csv", final_results)
    augmented = final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False]).reset_index(drop=True)
    write_csv(OUTPUT_DIR / "augmented_leaderboard.csv", augmented)
    best_by_group = final_results[final_results["status"].eq("ok")].sort_values(["model_group", "final_accuracy"], ascending=[True, False]).groupby("model_group", as_index=False).head(1)
    best_by_horizon = final_results[final_results["status"].eq("ok")].sort_values(["horizon", "final_accuracy"], ascending=[True, False]).groupby("horizon", as_index=False).head(1)
    best_by_family = final_results[final_results["status"].eq("ok")].sort_values(["model_id", "final_accuracy"], ascending=[True, False]).groupby("model_id", as_index=False).head(1)
    write_csv(OUTPUT_DIR / "best_by_model_group.csv", best_by_group)
    write_csv(OUTPUT_DIR / "best_by_horizon.csv", best_by_horizon)
    write_csv(OUTPUT_DIR / "best_by_model_family.csv", best_by_family)
    comparison = final_results.copy()
    comparison["current_main_result"] = CURRENT_MAIN_LABEL
    comparison["current_main_final_accuracy"] = CURRENT_MAIN_FINAL_ACCURACY
    comparison["delta_vs_current_main"] = comparison["final_accuracy"] - CURRENT_MAIN_FINAL_ACCURACY
    write_csv(OUTPUT_DIR / "comparison_vs_current_best.csv", comparison)
    write_csv(OUTPUT_DIR / "technical_rules_summary.csv", final_results[final_results["model_group"].isin(["naive_baselines", "technical_rule_baselines"])])
    write_csv(OUTPUT_DIR / "calibration_summary.csv", final_results[final_results["model_group"].eq("calibration_variants")])
    write_csv(OUTPUT_DIR / "ensemble_summary.csv", final_results[final_results["model_group"].eq("ensemble_stacking")])
    write_csv(OUTPUT_DIR / "regime_summary.csv", final_results[final_results["model_group"].eq("regime_aware_models")])
    write_csv(OUTPUT_DIR / "statistical_models_summary.csv", statistical_summary)
    write_markdown(OUTPUT_DIR / "garch_diagnostic_summary.md", garch_text)

    skipped = registry[registry["run_status"].eq("skipped_with_reason")].copy()
    failed = registry[registry["run_status"].eq("failed_with_reason")].copy()
    not_rec = registry[(registry["run_status"].eq("not_recommended_with_reason")) | (registry["reason_if_not_recommended"].astype(str).ne(""))].copy()
    status_preamble = "\n".join(
        [
            f"- Total model groups listed: {registry['model_group'].nunique()}.",
            f"- Total planned: {len(registry)}.",
            f"- Total attempted: {counts['attempted']}.",
            f"- Total run: {counts['run']}.",
            f"- Total failed: {counts['failed']}.",
            f"- Total skipped: {counts['skipped']}.",
            f"- Total not recommended: {counts['not_recommended']}.",
            f"- CatBoost status: {catboost_status}.",
            f"- GARCH diagnostic status: {garch_status}.",
            "- GARCH used as main directional classifier: no.",
        ]
    )
    write_markdown(OUTPUT_DIR / "skipped_models_report.md", "# Skipped Models Report\n\n" + status_preamble + "\n\n" + markdown_table(skipped[["model_id", "model_group", "reason_if_skipped"]], max_rows=len(skipped)))
    write_markdown(OUTPUT_DIR / "failed_models_report.md", "# Failed Models Report\n\n" + status_preamble + "\n\n" + markdown_table(failed[["model_id", "model_group", "reason_if_failed"]], max_rows=len(failed)))
    write_markdown(OUTPUT_DIR / "not_recommended_models_report.md", "# Not Recommended Models Report\n\n" + status_preamble + "\n\n" + markdown_table(not_rec[["model_id", "model_group", "reason_if_not_recommended"]], max_rows=len(not_rec)))
    coverage = registry.groupby(["model_group", "run_status"]).size().reset_index(name="models")
    write_markdown(OUTPUT_DIR / "model_coverage_audit.md", "# Model Coverage Audit\n\n" + status_preamble + "\n\n" + markdown_table(coverage, max_rows=len(coverage)))
    selected = final_results[final_results["selected_by_validation_yes_no"].eq("yes")]
    best = augmented.head(10)
    lines = [
        "# Model Universe Summary",
        "",
        f"- Exhaustive full run: {EXHAUSTIVE_FULL_RUN}.",
        f"- Total model groups listed: {registry['model_group'].nunique()}.",
        f"- Total model variants planned: {len(registry)}.",
        f"- Total model variants attempted: {counts['attempted']}.",
        f"- Total model variants run: {counts['run']}.",
        f"- Total model variants failed: {counts['failed']}.",
        f"- Total model variants skipped: {counts['skipped']}.",
        f"- Total model variants not recommended: {counts['not_recommended']}.",
        f"- Candidate rows planned/attempted: {len(candidate_grid)}.",
        f"- Successful result rows: {int(final_results['status'].eq('ok').sum())}.",
        f"- CatBoost status: {catboost_status}.",
        f"- GARCH diagnostic status: {garch_status}.",
        "- GARCH used as main directional classifier: no.",
        f"- Current main result: {CURRENT_MAIN_LABEL}, {pct(CURRENT_MAIN_FINAL_ACCURACY)}.",
        "",
        "## Validation-Selected Row",
        "",
        markdown_table(selected[["candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "validation_accuracy", "final_accuracy", "ticker_coverage", "claim_eligible_yes_no"]], max_rows=5),
        "",
        "## Top Final Accuracy Rows",
        "",
        markdown_table(best[["candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "validation_accuracy", "final_accuracy", "ticker_coverage", "selected_by_validation_yes_no", "claim_eligible_yes_no"]], max_rows=10),
    ]
    write_markdown(OUTPUT_DIR / "model_universe_summary.md", "\n".join(lines))
    claim_lines = [
        "# Model Universe Claim Boundary",
        "",
        "- Final-window scores are scoring-only and are not used for model, feature, threshold, horizon, ensemble, calibration, or router selection.",
        "- The current h40 paper result remains Logistic L2 / baseline_C_closest / h40 / validation-selected threshold 0.55 / 61.63% unless a new model is validation-selected, full-coverage, and audit-passed.",
        "- GARCH is diagnostic only and not a direct headline direction classifier.",
        f"- Total planned/run/failed/skipped/not recommended: {len(registry)}/{counts['run']}/{counts['failed']}/{counts['skipped']}/{counts['not_recommended']}.",
        f"- CatBoost status: {catboost_status}.",
        f"- GARCH diagnostic status: {garch_status}.",
        "- No trading, profitability, investment recommendation, or live-deployment claim is made.",
        "",
        markdown_table(selected[["candidate_id", "model_id", "validation_accuracy", "final_accuracy", "beats_61_63_yes_no", "claim_eligible_yes_no", "overfit_risk"]], max_rows=5),
    ]
    write_markdown(OUTPUT_DIR / "model_universe_claim_boundary.md", "\n".join(claim_lines))


def write_row_predictions(row_predictions: list[pd.DataFrame]) -> None:
    if row_predictions:
        frame = pd.concat(row_predictions, ignore_index=True)
    else:
        frame = pd.DataFrame()
    write_csv(OUTPUT_DIR / "row_predictions.csv", frame)


def safe_plot(path: Path, title: str, draw_fn: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    try:
        draw_fn()
        plt.title(title)
        plt.tight_layout()
    except Exception as exc:
        plt.clf()
        plt.text(0.5, 0.5, f"{title}\n{exc}", ha="center", va="center")
        plt.axis("off")
    plt.savefig(path, dpi=150)
    plt.close()


def write_figures(registry: pd.DataFrame, final_results: pd.DataFrame, statistical_summary: pd.DataFrame) -> None:
    ok = final_results[final_results["status"].eq("ok")].copy()

    def coverage() -> None:
        counts = registry["run_status"].value_counts().sort_index()
        plt.bar(counts.index, counts.values, color="#3b6ea8")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Models")

    safe_plot(FIGURE_DIR / "fig_model_universe_coverage.png", "Model Universe Coverage", coverage)

    def final_by_group() -> None:
        data = ok.groupby("model_group")["final_accuracy"].max().sort_values()
        plt.barh(data.index, data.values, color="#5b8a72")
        plt.axvline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Best final accuracy")

    safe_plot(FIGURE_DIR / "fig_final_accuracy_by_model_group.png", "Final Accuracy by Model Group", final_by_group)

    def validation_vs_final() -> None:
        for group, data in ok.groupby("model_group"):
            plt.scatter(data["validation_accuracy"], data["final_accuracy"], s=18, label=group, alpha=0.7)
        plt.axhline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Validation accuracy")
        plt.ylabel("Final accuracy")
        plt.legend(fontsize=6, loc="best")

    safe_plot(FIGURE_DIR / "fig_validation_vs_final_by_model_group.png", "Validation vs Final by Model Group", validation_vs_final)

    def best_by_family() -> None:
        data = ok.groupby("model_id")["final_accuracy"].max().sort_values().tail(20)
        plt.barh(data.index, data.values, color="#7a6f9b")
        plt.axvline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Best final accuracy")

    safe_plot(FIGURE_DIR / "fig_best_by_model_family.png", "Best by Model Family", best_by_family)

    def horizon_heatmap() -> None:
        pivot = ok.pivot_table(index="model_group", columns="horizon", values="final_accuracy", aggfunc="max")
        plt.imshow(pivot.fillna(np.nan).to_numpy(dtype=float), aspect="auto", cmap="viridis")
        plt.colorbar(label="Best final accuracy")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns)

    safe_plot(FIGURE_DIR / "fig_horizon_accuracy_heatmap.png", "Horizon Accuracy Heatmap", horizon_heatmap)

    def technical_vs_ml() -> None:
        groups = ["naive_baselines", "technical_rule_baselines", "linear_generalized_linear_models", "tree_based_models", "boosting_models"]
        data = ok[ok["model_group"].isin(groups)].groupby("model_group")["final_accuracy"].max().reindex(groups)
        plt.bar(data.index, data.values, color="#4c7899")
        plt.xticks(rotation=25, ha="right")
        plt.axhline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)
        plt.ylabel("Best final accuracy")

    safe_plot(FIGURE_DIR / "fig_technical_rules_vs_ml.png", "Technical Rules vs ML", technical_vs_ml)

    def svm_tree_boosting() -> None:
        groups = ["kernel_distance_based_models", "tree_based_models", "boosting_models"]
        data = ok[ok["model_group"].isin(groups)].groupby("model_id")["final_accuracy"].max().sort_values()
        plt.barh(data.index, data.values, color="#8f6b4a")
        plt.axvline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)

    safe_plot(FIGURE_DIR / "fig_svm_tree_boosting_comparison.png", "SVM Tree Boosting Comparison", svm_tree_boosting)

    def deep_vs_classical() -> None:
        groups = ["linear_generalized_linear_models", "kernel_distance_based_models", "tree_based_models", "boosting_models", "neural_deep_models"]
        data = ok[ok["model_group"].isin(groups)].groupby("model_group")["final_accuracy"].max().reindex(groups)
        plt.bar(data.index, data.values, color="#5c7f67")
        plt.xticks(rotation=25, ha="right")
        plt.axhline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)

    safe_plot(FIGURE_DIR / "fig_deep_vs_classical_comparison.png", "Deep vs Classical Comparison", deep_vs_classical)

    def calibration_effects() -> None:
        data = ok[ok["model_group"].isin(["calibration_variants", "linear_generalized_linear_models", "tree_based_models", "boosting_models"])].groupby("model_id")["final_accuracy"].max().sort_values().tail(20)
        plt.barh(data.index, data.values, color="#6f8fb7")
        plt.axvline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)

    safe_plot(FIGURE_DIR / "fig_calibration_effects.png", "Calibration Effects", calibration_effects)

    def current_best_vs_expansion() -> None:
        data = ok.groupby("model_id")["final_accuracy"].max().sort_values().tail(12)
        labels = list(data.index) + ["current_main_61_63"]
        values = list(data.values) + [CURRENT_MAIN_FINAL_ACCURACY]
        plt.barh(labels, values, color=["#577590"] * len(data) + ["#111111"])
        plt.xlabel("Final accuracy")

    safe_plot(FIGURE_DIR / "fig_current_best_vs_expansion.png", "Current Best vs Expansion", current_best_vs_expansion)

    def skipped_failed_reasons() -> None:
        scoped = registry[registry["run_status"].isin(["skipped_with_reason", "failed_with_reason", "not_recommended_with_reason"])]
        reasons = []
        for _, row in scoped.iterrows():
            reason = row.get("reason_if_skipped") or row.get("reason_if_failed") or row.get("reason_if_not_recommended") or row.get("run_status")
            reasons.append(str(reason)[:60])
        counts = pd.Series(reasons).value_counts().head(10)
        plt.barh(counts.index, counts.values, color="#9b5c5c")
        plt.xlabel("Models")

    safe_plot(FIGURE_DIR / "fig_skipped_failed_model_reasons.png", "Skipped Failed Model Reasons", skipped_failed_reasons)

    def statistical_diag() -> None:
        data = ok[ok["model_group"].eq("traditional_statistical_financial_models")].groupby("model_id")["final_accuracy"].max().sort_values()
        if data.empty:
            data = registry[registry["model_group"].eq("traditional_statistical_financial_models")]["run_status"].value_counts()
            plt.bar(data.index, data.values, color="#9a7f4f")
            plt.ylabel("Models")
        else:
            plt.barh(data.index, data.values, color="#9a7f4f")
            plt.axvline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)

    safe_plot(FIGURE_DIR / "fig_statistical_models_diagnostic.png", "Statistical Models Diagnostic", statistical_diag)

    def overfit_risk_beating_rows() -> None:
        data = ok[pd.to_numeric(ok["final_accuracy"], errors="coerce") > CURRENT_MAIN_FINAL_ACCURACY].copy()
        if data.empty:
            plt.text(0.5, 0.5, "No beating rows", ha="center", va="center")
            plt.axis("off")
            return
        counts = data["overfit_risk"].fillna("unknown").value_counts().reindex(["low", "medium", "high", "unknown"]).dropna()
        plt.bar(counts.index, counts.values, color="#7a4f9a")
        plt.ylabel("Rows beating 61.63%")

    safe_plot(FIGURE_DIR / "fig_overfit_risk_beating_rows.png", "Overfit Risk for Beating Rows", overfit_risk_beating_rows)

    def catboost_vs_boosting_family() -> None:
        data = ok[ok["model_group"].eq("boosting_models")].groupby("model_id")["final_accuracy"].max().sort_values()
        if data.empty:
            plt.text(0.5, 0.5, "No boosting rows", ha="center", va="center")
            plt.axis("off")
            return
        colors = ["#b85c38" if model_id == "catboost" else "#5d7f9a" for model_id in data.index]
        plt.barh(data.index, data.values, color=colors)
        plt.axvline(CURRENT_MAIN_FINAL_ACCURACY, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Best final accuracy")

    safe_plot(FIGURE_DIR / "fig_catboost_vs_boosting_family.png", "CatBoost vs Boosting Family", catboost_vs_boosting_family)

    def garch_volatility_diagnostic() -> None:
        data = statistical_summary[statistical_summary["model_id"].astype(str).eq("garch_volatility_diagnostic")].copy()
        if data.empty or "mean_forecast_volatility_pct" not in data.columns:
            plt.text(0.5, 0.5, "GARCH diagnostic unavailable", ha="center", va="center")
            plt.axis("off")
            return
        data = data.sort_values("horizon")
        plt.plot(data["horizon"], data["mean_forecast_volatility_pct"], marker="o", label="mean")
        if "median_forecast_volatility_pct" in data.columns:
            plt.plot(data["horizon"], data["median_forecast_volatility_pct"], marker="s", label="median")
        plt.xlabel("Horizon")
        plt.ylabel("Forecast volatility (%)")
        plt.legend()

    if not statistical_summary.empty and statistical_summary["model_id"].astype(str).eq("garch_volatility_diagnostic").any():
        safe_plot(FIGURE_DIR / "fig_garch_volatility_diagnostic.png", "GARCH Volatility Diagnostic", garch_volatility_diagnostic)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_dependency_install_report()
    registry = initial_registry()
    write_registry_outputs(registry)
    features, family_cols, manifest = prepare_features()
    write_json(OUTPUT_DIR / "run_manifest.json", manifest)

    candidate_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    row_predictions: list[pd.DataFrame] = []

    rule_rows, rule_failures = run_rule_models(features, candidate_rows, row_predictions)
    all_rows.extend(rule_rows)
    all_failures.extend(rule_failures)

    classifier_rows, classifier_failures, ensemble_pool = run_classifier_models(features, family_cols, candidate_rows, row_predictions)
    all_rows.extend(classifier_rows)
    all_failures.extend(classifier_failures)

    deep_rows, deep_failures = run_deep_models(features, family_cols, candidate_rows, row_predictions)
    all_rows.extend(deep_rows)
    all_failures.extend(deep_failures)

    regime_rows, regime_failures = run_regime_models(features, family_cols, candidate_rows, row_predictions)
    all_rows.extend(regime_rows)
    all_failures.extend(regime_failures)

    ensemble_rows, ensemble_failures = run_ensembles(features, ensemble_pool, candidate_rows)
    all_rows.extend(ensemble_rows)
    all_failures.extend(ensemble_failures)

    statistical_rows, statistical_failures, statistical_summary, garch_text = run_statistical_models(features, candidate_rows, row_predictions)
    all_rows.extend(statistical_rows)
    all_failures.extend(statistical_failures)

    candidate_grid = pd.DataFrame(candidate_rows)
    final_results = pd.DataFrame(all_rows)
    failures = pd.DataFrame(all_failures)
    if not final_results.empty:
        final_results = update_selection(final_results)
    registry = update_registry_from_results(registry, candidate_grid, final_results, failures)
    write_registry_outputs(registry)
    write_reports(registry, candidate_grid, final_results, failures, statistical_summary, garch_text)
    write_row_predictions(row_predictions)
    write_figures(registry, final_results, statistical_summary)
    print(f"Wrote VN30 comprehensive model-universe benchmark outputs to {rel(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
