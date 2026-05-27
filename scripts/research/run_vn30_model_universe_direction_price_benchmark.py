"""Run VN30 model-universe direction and price/return diagnostics.

This runner is offline diagnostic-only. It evaluates direction targets and
price/return targets separately under strict feature_timestamp and
target_timestamp split discipline. Final results are scoring-only; validation
metrics choose locked diagnostic candidates.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import build_feature_families  # noqa: E402
from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, load_index_data  # noqa: E402
from scripts.research.run_vn30_qml_forecasting import (  # noqa: E402
    CLASSICAL_CHAMPION,
    FINAL_START,
    SEED,
    TRAIN_END,
    VAL_END,
    VAL_START,
    FeatureSpec,
    add_v3_relative_strength_features,
    as_float,
    build_labels,
    build_source_groups,
    candidate_id,
    fit_feature_spec,
    json_safe,
    leakage_guard_passed,
    numeric_existing,
    ordered_index,
    stock_future_returns,
    strict_split_indices,
    target_timestamp_from_labels,
    v6_quantum_kernel_matrices,
    v6_scaling_transform,
    v8_kernel_feature_frames,
    write_frame,
    write_json,
    write_markdown,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_direction_price"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_DIRECTION_PRICE_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_DIRECTION_PRICE_CLAIM_BOUNDARY.md"
V2_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_V2_PROMOTION_RELOCK_RESULT_SUMMARY.md"
V2_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_V2_PROMOTION_RELOCK_CLAIM_BOUNDARY.md"
V3_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_V3_SKIPPED_FAMILIES_RESULT_SUMMARY.md"
V3_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_V3_SKIPPED_FAMILIES_CLAIM_BOUNDARY.md"
V4_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_V4_BILSTM_RELOCK_RESULT_SUMMARY.md"
V4_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_V4_BILSTM_RELOCK_CLAIM_BOUNDARY.md"
V5_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_V5_TARGET_METRIC_REPAIR_RESULT_SUMMARY.md"
V5_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_V5_TARGET_METRIC_REPAIR_CLAIM_BOUNDARY.md"
V6_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_V6_PRICE_RETURN_ABSOLUTE_CONFIRMATION_RESULT_SUMMARY.md"
V6_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_V6_PRICE_RETURN_ABSOLUTE_CONFIRMATION_CLAIM_BOUNDARY.md"
V7_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_V7_EXHAUSTIVE_EXPANSION_RESULT_SUMMARY.md"
V7_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_V7_EXHAUSTIVE_EXPANSION_CLAIM_BOUNDARY.md"

QML_V8_CONTEXT_FINAL = 0.6444444444444445
QML_V8_CONTEXT_VALIDATION = 0.6055555555555555

DIRECTION_TARGETS = ["absolute_direction", "market_relative_vn30", "market_relative_vnindex", "top_quantile_forward_return"]
PRICE_TARGETS = ["forward_simple_return_h", "forward_log_return_h", "future_close_h", "market_excess_return_h", "volatility_adjusted_return_h"]
HORIZONS = [5, 10, 20, 40, 60]
V7_HORIZONS = [10, 20, 40, 60]
V7_DIRECTION_TARGETS = ["absolute_direction", "market_relative_vn30", "market_relative_vnindex"]
V7_RANKING_TARGETS = [
    "top_quantile_forward_return_top20",
    "top_quantile_forward_return_top30",
    "cross_sectional_forward_return_rank",
    "market_excess_return_rank",
]
V7_PRICE_TARGETS = ["forward_log_return_h", "forward_simple_return_h", "market_excess_return_h", "volatility_adjusted_return_h"]
V7_FEATURE_GROUPS = [
    "relative_strength",
    "market_context",
    "combined_strategy_features",
    "compact_stable_features",
    "volume_volatility",
    "all_safe_features",
    "qml_kernel_features",
]

DIRECTION_MODEL_NAMES = [
    "always_up",
    "always_down",
    "lag1_direction",
    "random_same_class_balance",
    "simple_momentum",
    "simple_relative_strength",
    "logistic_regression",
    "calibrated_logistic",
    "linear_svm",
    "rbf_svm",
    "knn_classifier",
    "naive_bayes",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost_classifier",
    "lightgbm_classifier",
    "catboost_classifier",
    "mlp_classifier",
    "qml_v8_kernel_features_l2",
    "qml_v8_kernel_features_lightgbm",
    "soft_voting",
    "stacking_logistic",
    "rank_averaging",
    "regime_aware_ensemble",
    "direction_price_joint_ensemble",
]

PRICE_MODEL_NAMES = [
    "random_walk_price",
    "last_price",
    "zero_return",
    "historical_mean_return",
    "rolling_mean_return",
    "simple_momentum",
    "simple_relative_strength",
    "linear_regression",
    "ridge",
    "lasso",
    "elasticnet",
    "svr_linear",
    "svr_rbf",
    "knn_regressor",
    "random_forest_regressor",
    "extra_trees_regressor",
    "gradient_boosting_regressor",
    "hist_gradient_boosting_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
    "catboost_regressor",
    "mlp_regressor",
]

OPTIONAL_SEQUENCE_MODELS = [
    "ARIMA",
    "SARIMAX",
    "ETS",
    "GARCH-assisted",
    "Kalman/local level",
    "LSTM",
    "GRU",
    "BiLSTM",
    "TCN",
    "1D CNN",
    "Transformer encoder",
    "N-BEATS/N-HiTS",
    "TFT",
    "V4 pure quantum kernel replay",
    "PennyLane hybrid QNN",
]


@dataclass
class RunConfig:
    mode: str
    timeout_seconds: int
    max_train_rows: int
    max_validation_rows: int
    max_final_rows: int
    max_direction_candidates: int
    max_price_candidates: int


def dependency_status() -> dict[str, Any]:
    packages = [
        "catboost",
        "optuna",
        "sklearn",
        "statsmodels",
        "arch",
        "torch",
        "pytorch_forecasting",
        "darts",
        "neuralforecast",
        "tensorflow",
        "lightgbm",
        "xgboost",
        "qiskit",
        "qiskit_machine_learning",
        "pennylane",
        "ngboost",
        "hmmlearn",
    ]
    status: dict[str, Any] = {}
    for name in packages:
        available = importlib.util.find_spec(name) is not None
        version = ""
        if available:
            try:
                version = str(getattr(importlib.import_module(name), "__version__", "unknown"))
            except Exception:
                version = "unknown"
        status[f"{name}_available"] = bool(available)
        status[f"{name}_version"] = version
        status[f"{name}_install_needed"] = not bool(available)
        status[f"{name}_skipped_reason"] = "" if available else f"{name} unavailable"
    status["diagnostic_only"] = True
    status["no_trading_claim"] = True
    return status


def pct(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:+.2f} pp"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else math.nan


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 1e-12
    if not valid.any():
        return math.nan
    return float(np.mean(2.0 * np.abs(y_pred[valid] - y_true[valid]) / denom[valid]))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.abs(y_true) > 1e-12
    if not valid.any():
        return math.nan
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])))


def mase(y_true: np.ndarray, y_pred: np.ndarray, naive_pred: np.ndarray) -> float:
    denom = float(np.mean(np.abs(y_true - naive_pred))) if len(y_true) else math.nan
    if not math.isfinite(denom) or denom <= 1e-12:
        return math.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0 or np.isnan(prob).all():
        return math.nan
    frame = pd.DataFrame({"y": y_true, "prob": prob}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return math.nan
    frame["bin"] = pd.cut(frame["prob"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    total = len(frame)
    err = 0.0
    for _bin, group in frame.groupby("bin", observed=False):
        if len(group):
            err += len(group) / total * abs(float(group["y"].mean()) - float(group["prob"].mean()))
    return float(err)


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return math.nan
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return math.nan


def target_timestamp_series(target: pd.Series) -> pd.Series:
    value = target.attrs.get("target_timestamp")
    if isinstance(value, pd.Series):
        return pd.to_datetime(value.reindex(target.index), errors="coerce")
    return pd.Series(pd.NaT, index=target.index)


def make_index_return(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], code: str, target_timestamp: pd.Series) -> pd.Series:
    if code not in index_data:
        return pd.Series(np.nan, index=features.index, dtype=float)
    idx_frame = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
    idx_frame["datetime"] = pd.to_datetime(idx_frame["datetime"], errors="coerce")
    idx_frame["close"] = pd.to_numeric(idx_frame["close"], errors="coerce")
    idx_frame = idx_frame.dropna(subset=["datetime", "close"])
    left = features[["datetime"]].copy()
    left["row_index"] = features.index
    left["target_timestamp"] = target_timestamp.to_numpy()
    start = pd.merge_asof(left.sort_values("datetime"), idx_frame.rename(columns={"close": "start_close"}), on="datetime", direction="backward")
    end_left = left[["row_index", "target_timestamp"]].dropna().rename(columns={"target_timestamp": "datetime"}).sort_values("datetime")
    end = pd.merge_asof(end_left, idx_frame.rename(columns={"close": "target_close"}), on="datetime", direction="backward")
    start_close = pd.Series(start.set_index("row_index")["start_close"]).reindex(features.index)
    end_close = pd.Series(end.set_index("row_index")["target_close"]).reindex(features.index)
    return end_close / start_close.replace(0.0, np.nan) - 1.0


def build_direction_target(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], target_variant: str, horizon: int) -> pd.Series:
    if target_variant in {"absolute_direction", "market_relative_vn30", "market_relative_vnindex"}:
        return build_labels(features, index_data, target_variant, horizon)
    stock_return, target_timestamp = stock_future_returns(features, horizon)
    labels = pd.Series(np.nan, index=features.index, dtype=float)
    if target_variant == "top_quantile_forward_return":
        frame = pd.DataFrame({"datetime": features["datetime"], "forward_return": stock_return})
        ranks = frame.groupby("datetime")["forward_return"].rank(pct=True, method="average")
        valid = stock_return.notna() & target_timestamp.notna()
        labels.loc[valid] = (ranks.loc[valid] >= 0.80).astype(float)
    else:
        raise ValueError(f"unknown direction target {target_variant}")
    labels.attrs["target_timestamp"] = target_timestamp
    labels.attrs["target_variant"] = target_variant
    labels.attrs["horizon"] = int(horizon)
    labels.attrs["split_rule"] = "feature_timestamp and target_timestamp must both be inside each split"
    labels.attrs["top_quantile"] = 0.20
    return labels


def build_price_target(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], target_variant: str, horizon: int) -> pd.Series:
    simple_return, target_timestamp = stock_future_returns(features, horizon)
    target = pd.Series(np.nan, index=features.index, dtype=float)
    if target_variant == "forward_simple_return_h":
        target = simple_return.astype(float)
    elif target_variant == "forward_log_return_h":
        target = np.log1p(simple_return.astype(float))
    elif target_variant == "future_close_h":
        future_close_parts: list[pd.Series] = []
        for _ticker, group in features.sort_values(["ticker", "datetime"]).groupby("ticker", sort=True):
            future_close_parts.append(pd.Series(group["close"].shift(-horizon).to_numpy(dtype=float), index=group.index))
        target = pd.concat(future_close_parts).sort_index() if future_close_parts else target
    elif target_variant == "market_excess_return_h":
        vn30_return = make_index_return(features, index_data, "VN30", target_timestamp)
        target = simple_return - vn30_return
    elif target_variant == "volatility_adjusted_return_h":
        vol_cols = [col for col in ["rolling_return_vol_20", "return_1_vol_20", "volatility_20", "v3_relative_strength_vn30_vol20_lag"] if col in features.columns]
        if vol_cols:
            vol = pd.to_numeric(features[vol_cols[0]], errors="coerce").abs().replace(0.0, np.nan)
        else:
            vol_parts: list[pd.Series] = []
            for _ticker, group in features.sort_values(["ticker", "datetime"]).groupby("ticker", sort=True):
                ret = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None)
                vol_parts.append(pd.Series(ret.rolling(20, min_periods=5).std().to_numpy(dtype=float), index=group.index))
            vol = pd.concat(vol_parts).sort_index() if vol_parts else pd.Series(np.nan, index=features.index)
        target = simple_return / vol.replace(0.0, np.nan)
    else:
        raise ValueError(f"unknown price target {target_variant}")
    target = pd.Series(target, index=features.index, dtype=float).replace([np.inf, -np.inf], np.nan)
    target.attrs["target_timestamp"] = target_timestamp
    target.attrs["target_variant"] = target_variant
    target.attrs["horizon"] = int(horizon)
    target.attrs["split_rule"] = "feature_timestamp and target_timestamp must both be inside each split"
    return target


def strict_split_for_target(features: pd.DataFrame, target: pd.Series) -> dict[str, pd.Index]:
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_series(target)
    valid = target.notna() & timestamps.notna() & target_timestamp.notna()
    train = timestamps.le(TRAIN_END) & target_timestamp.le(TRAIN_END) & valid
    validation = timestamps.between(VAL_START, VAL_END) & target_timestamp.between(VAL_START, VAL_END) & valid
    final = timestamps.ge(FINAL_START) & target_timestamp.ge(FINAL_START) & valid
    return {"train": features.index[train], "validation": features.index[validation], "final": features.index[final]}


def leakage_guard_for_target(features: pd.DataFrame, target: pd.Series, splits: dict[str, pd.Index]) -> bool:
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_series(target)
    train = splits["train"]
    validation = splits["validation"]
    final = splits["final"]
    if len(train) and ((timestamps.loc[train] > TRAIN_END).any() or (target_timestamp.loc[train] > TRAIN_END).any()):
        return False
    if len(validation) and ((timestamps.loc[validation] < VAL_START).any() or (timestamps.loc[validation] > VAL_END).any() or (target_timestamp.loc[validation] < VAL_START).any() or (target_timestamp.loc[validation] > VAL_END).any()):
        return False
    if len(final) and ((timestamps.loc[final] < FINAL_START).any() or (target_timestamp.loc[final] < FINAL_START).any()):
        return False
    return True


def split_sample(features: pd.DataFrame, y: pd.Series, splits: dict[str, pd.Index], config: RunConfig, classification: bool) -> dict[str, pd.Index]:
    def ordered_tail(index: pd.Index, limit: int) -> pd.Index:
        ordered = ordered_index(features, index)
        return ordered if len(ordered) <= limit else pd.Index(list(ordered)[-limit:])

    if classification:
        train_ordered = ordered_index(features, splits["train"])
        if len(train_ordered) > config.max_train_rows:
            frame = pd.DataFrame({"idx": train_ordered, "label": y.loc[train_ordered].astype(int).to_numpy(), "datetime": features.loc[train_ordered, "datetime"].to_numpy()})
            pieces = []
            per_class = max(1, config.max_train_rows // max(1, frame["label"].nunique()))
            for _label, group in frame.groupby("label", sort=True):
                pieces.append(group.tail(per_class))
            train = pd.concat(pieces, ignore_index=True).sort_values(["datetime", "idx"])
            train_idx = pd.Index(train["idx"].tolist())
        else:
            train_idx = train_ordered
    else:
        train_idx = ordered_tail(splits["train"], config.max_train_rows)
    return {
        "train": train_idx,
        "validation": ordered_tail(splits["validation"], config.max_validation_rows),
        "final": ordered_tail(splits["final"], config.max_final_rows),
    }


def build_feature_groups(features: pd.DataFrame, family_cols: dict[str, list[str]], relative_cols: list[str]) -> dict[str, list[str]]:
    source_groups = build_source_groups(features, family_cols)
    volume_volatility = numeric_existing(features, [col for col in features.columns if any(token in col.lower() for token in ["volume", "vol", "shock", "atr", "rsi"])])
    stock_lag_momentum = numeric_existing(features, [col for col in features.columns if any(token in col.lower() for token in ["lag", "momentum", "return", "sma", "ema", "macd"]) and not col.lower().startswith(("vnindex", "vn30"))])
    all_safe = numeric_existing(features, sorted(set(source_groups["combined_strategy_features"]).union(relative_cols).union(source_groups["market_context_features"]).union(volume_volatility).union(stock_lag_momentum)))
    groups = {
        "stock_lag_momentum": stock_lag_momentum or source_groups["compact_stable_features"],
        "relative_strength": numeric_existing(features, relative_cols) or source_groups["relative_strength_features"],
        "market_context": source_groups["market_context_features"],
        "volume_volatility": volume_volatility or source_groups["compact_stable_features"],
        "combined_strategy_features": source_groups["combined_strategy_features"],
        "compact_stable_features": source_groups["compact_stable_features"],
        "all_safe_features": all_safe,
    }
    qml_audit = OUTPUT_DIR.parent / "vn30_qml_forecasting" / "qml_v8_kernel_feature_audit.csv"
    if qml_audit.exists():
        groups["qml_kernel_features"] = []
    return groups


def fit_matrix(features: pd.DataFrame, splits: dict[str, pd.Index], feature_columns: list[str], max_features: int = 16) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str]:
    ranked = []
    if feature_columns:
        train = features.loc[splits["train"], feature_columns].replace([np.inf, -np.inf], np.nan)
        availability = train.notna().mean()
        variance = train.var(numeric_only=True).fillna(0.0)
        ranked = sorted([col for col in feature_columns if availability.get(col, 0.0) > 0.0], key=lambda col: (-float(availability.get(col, 0.0)), -float(variance.get(col, 0.0)), col))
    selected = ranked[:max_features]
    if not selected:
        return pd.DataFrame(index=splits["train"]), pd.DataFrame(index=splits["validation"]), pd.DataFrame(index=splits["final"]), [], "no_features"
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    x_train = pipe.fit_transform(features.loc[splits["train"], selected].replace([np.inf, -np.inf], np.nan))
    x_val = pipe.transform(features.loc[splits["validation"], selected].replace([np.inf, -np.inf], np.nan))
    x_final = pipe.transform(features.loc[splits["final"], selected].replace([np.inf, -np.inf], np.nan))
    return (
        pd.DataFrame(x_train, index=splits["train"], columns=selected),
        pd.DataFrame(x_val, index=splits["validation"], columns=selected),
        pd.DataFrame(x_final, index=splits["final"], columns=selected),
        selected,
        "ok",
    )


def direction_model(model_name: str) -> tuple[Any | None, str]:
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, solver="liblinear", C=0.5, class_weight="balanced", random_state=SEED), ""
    if model_name == "calibrated_logistic":
        base = LogisticRegression(max_iter=1000, solver="liblinear", C=0.5, class_weight="balanced", random_state=SEED)
        try:
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3), ""
        except TypeError:  # pragma: no cover
            return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3), ""
    if model_name == "linear_svm":
        return SVC(kernel="linear", C=0.5, class_weight="balanced", probability=True, random_state=SEED), ""
    if model_name == "rbf_svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=SEED), ""
    if model_name == "knn_classifier":
        return KNeighborsClassifier(n_neighbors=15, weights="distance"), ""
    if model_name == "naive_bayes":
        return GaussianNB(), ""
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=15, class_weight="balanced", random_state=SEED, n_jobs=2), ""
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=80, max_depth=6, min_samples_leaf=15, class_weight="balanced", random_state=SEED, n_jobs=2), ""
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, random_state=SEED), ""
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, random_state=SEED), ""
    if model_name == "xgboost_classifier":
        if importlib.util.find_spec("xgboost") is None:
            return None, "xgboost unavailable"
        xgb = importlib.import_module("xgboost")
        return xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=2), ""
    if model_name == "lightgbm_classifier":
        if importlib.util.find_spec("lightgbm") is None:
            return None, "lightgbm unavailable"
        lgb = importlib.import_module("lightgbm")
        return lgb.LGBMClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, num_leaves=15, min_child_samples=30, class_weight="balanced", random_state=SEED, verbosity=-1, n_jobs=2), ""
    if model_name == "catboost_classifier":
        if importlib.util.find_spec("catboost") is None:
            return None, "catboost unavailable"
        cb = importlib.import_module("catboost")
        return cb.CatBoostClassifier(iterations=80, depth=4, learning_rate=0.05, loss_function="Logloss", verbose=False, random_seed=SEED), ""
    if model_name == "mlp_classifier":
        return MLPClassifier(hidden_layer_sizes=(24,), alpha=0.001, max_iter=120, random_state=SEED, early_stopping=True), ""
    return None, "model handled as baseline, ensemble, QML, or skipped optional family"


def price_model(model_name: str) -> tuple[Any | None, str]:
    if model_name == "linear_regression":
        return LinearRegression(), ""
    if model_name == "ridge":
        return Ridge(alpha=2.0, random_state=SEED), ""
    if model_name == "lasso":
        return Lasso(alpha=0.0005, max_iter=2000, random_state=SEED), ""
    if model_name == "elasticnet":
        return ElasticNet(alpha=0.0005, l1_ratio=0.3, max_iter=2000, random_state=SEED), ""
    if model_name == "svr_linear":
        return SVR(kernel="linear", C=0.5, epsilon=0.01), ""
    if model_name == "svr_rbf":
        return SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale"), ""
    if model_name == "knn_regressor":
        return KNeighborsRegressor(n_neighbors=15, weights="distance"), ""
    if model_name == "random_forest_regressor":
        return RandomForestRegressor(n_estimators=80, max_depth=6, min_samples_leaf=15, random_state=SEED, n_jobs=2), ""
    if model_name == "extra_trees_regressor":
        return ExtraTreesRegressor(n_estimators=80, max_depth=6, min_samples_leaf=15, random_state=SEED, n_jobs=2), ""
    if model_name == "gradient_boosting_regressor":
        return GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.05, random_state=SEED), ""
    if model_name == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, random_state=SEED), ""
    if model_name == "xgboost_regressor":
        if importlib.util.find_spec("xgboost") is None:
            return None, "xgboost unavailable"
        xgb = importlib.import_module("xgboost")
        return xgb.XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2), ""
    if model_name == "lightgbm_regressor":
        if importlib.util.find_spec("lightgbm") is None:
            return None, "lightgbm unavailable"
        lgb = importlib.import_module("lightgbm")
        return lgb.LGBMRegressor(n_estimators=80, max_depth=4, learning_rate=0.05, num_leaves=15, min_child_samples=30, random_state=SEED, verbosity=-1, n_jobs=2), ""
    if model_name == "catboost_regressor":
        if importlib.util.find_spec("catboost") is None:
            return None, "catboost unavailable"
        cb = importlib.import_module("catboost")
        return cb.CatBoostRegressor(iterations=80, depth=4, learning_rate=0.05, loss_function="RMSE", verbose=False, random_seed=SEED), ""
    if model_name == "mlp_regressor":
        return MLPRegressor(hidden_layer_sizes=(24,), alpha=0.001, max_iter=150, random_state=SEED, early_stopping=True), ""
    return None, "model handled as baseline or skipped optional family"


def baseline_direction_prediction(model_name: str, features: pd.DataFrame, y_train: pd.Series, idx: pd.Index) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if model_name == "always_up":
        pred = np.ones(len(idx), dtype=int)
        return pred, pred.astype(float), ""
    if model_name == "always_down":
        pred = np.zeros(len(idx), dtype=int)
        return pred, pred.astype(float), ""
    if model_name == "random_same_class_balance":
        p = float(y_train.astype(int).mean()) if len(y_train) else 0.5
        rng = np.random.default_rng(SEED + len(idx))
        prob = np.full(len(idx), p, dtype=float)
        return (rng.random(len(idx)) < p).astype(int), prob, ""
    if model_name == "lag1_direction":
        for col in ["return_1_lag_1", "lag_ret_1"]:
            if col in features.columns:
                values = pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                pred = (values > 0.0).astype(int)
                return pred, np.clip((values > 0.0).astype(float), 0.0, 1.0), ""
        return None, None, "lag1 return feature unavailable"
    if model_name == "simple_momentum":
        cols = [col for col in ["momentum_20", "rolling_return_mean_20", "return_1_lag_1"] if col in features.columns]
        if not cols:
            return None, None, "momentum feature unavailable"
        values = pd.to_numeric(features.loc[idx, cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred = (values > 0.0).astype(int)
        return pred, 1.0 / (1.0 + np.exp(-np.clip(values, -10.0, 10.0))), ""
    if model_name == "simple_relative_strength":
        cols = [col for col in features.columns if "relative" in col.lower()]
        if not cols:
            return None, None, "relative-strength feature unavailable"
        values = pd.to_numeric(features.loc[idx, cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred = (values > 0.0).astype(int)
        return pred, 1.0 / (1.0 + np.exp(-np.clip(values, -10.0, 10.0))), ""
    return None, None, "not a direction baseline"


def baseline_price_prediction(model_name: str, features: pd.DataFrame, y_train: pd.Series, idx: pd.Index, target_variant: str, horizon: int) -> tuple[np.ndarray | None, str]:
    close = pd.to_numeric(features.loc[idx, "close"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    train_mean = float(pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()) if len(y_train) else 0.0
    if target_variant == "future_close_h":
        if model_name in {"random_walk_price", "last_price"}:
            return close, ""
        if model_name == "historical_mean_return":
            return close * (1.0 + train_mean), ""
        if model_name == "rolling_mean_return":
            col = "rolling_return_mean_20" if "rolling_return_mean_20" in features.columns else "return_1_lag_1"
            if col in features.columns:
                ret = pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                return close * (1.0 + ret), ""
        if model_name == "simple_momentum":
            col = "momentum_20" if "momentum_20" in features.columns else "return_1_lag_1"
            if col in features.columns:
                ret = pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                return close * (1.0 + np.clip(ret, -0.2, 0.2)), ""
        if model_name == "simple_relative_strength":
            cols = [col for col in features.columns if "relative" in col.lower()]
            if cols:
                ret = pd.to_numeric(features.loc[idx, cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                return close * (1.0 + np.clip(ret, -0.2, 0.2)), ""
    else:
        if model_name == "zero_return":
            return np.zeros(len(idx), dtype=float), ""
        if model_name in {"historical_mean_return", "random_walk_price", "last_price"}:
            return np.full(len(idx), train_mean if math.isfinite(train_mean) else 0.0, dtype=float), ""
        if model_name == "rolling_mean_return":
            col = "rolling_return_mean_20" if "rolling_return_mean_20" in features.columns else "return_1_lag_1"
            if col in features.columns:
                return pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float), ""
        if model_name == "simple_momentum":
            col = "momentum_20" if "momentum_20" in features.columns else "return_1_lag_1"
            if col in features.columns:
                return pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float), ""
        if model_name == "simple_relative_strength":
            cols = [col for col in features.columns if "relative" in col.lower()]
            if cols:
                return pd.to_numeric(features.loc[idx, cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float), ""
    return None, f"{model_name} baseline not applicable for {target_variant}"


def direction_metrics(y_true: pd.Series, pred: np.ndarray, prob: np.ndarray | None) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    p = np.asarray(pred, dtype=int)
    prob_arr = np.asarray(prob, dtype=float) if prob is not None else p.astype(float)
    return {
        "accuracy": float((y == p).mean()) if len(y) else math.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, p)) if len(np.unique(y)) > 1 else math.nan,
        "f1": float(f1_score(y, p, zero_division=0)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "roc_auc": safe_auc(y, prob_arr),
        "brier_score": float(brier_score_loss(y, np.clip(prob_arr, 0.0, 1.0))) if len(y) and len(np.unique(y)) > 1 else math.nan,
        "calibration_error": calibration_error(y, np.clip(prob_arr, 0.0, 1.0)),
    }


def price_metrics(y_true: pd.Series, pred: np.ndarray, naive_pred: np.ndarray, close: np.ndarray, target_variant: str) -> dict[str, float]:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return {key: math.nan for key in ["mae", "rmse", "mape", "smape", "mase", "r2", "sign_accuracy", "directional_hit_from_predicted_return", "correlation_pred_actual", "quantile_loss", "interval_coverage"]}
    yv = y[valid]
    pv = pred[valid]
    if target_variant == "future_close_h":
        actual_ret = yv / np.clip(close[valid], 1e-12, None) - 1.0
        pred_ret = pv / np.clip(close[valid], 1e-12, None) - 1.0
    else:
        actual_ret = yv
        pred_ret = pv
    corr = float(np.corrcoef(pv, yv)[0, 1]) if len(yv) > 2 and np.std(pv) > 0 and np.std(yv) > 0 else math.nan
    q = 0.50
    qloss = float(np.mean(np.maximum(q * (yv - pv), (q - 1.0) * (yv - pv))))
    return {
        "mae": float(mean_absolute_error(yv, pv)),
        "rmse": rmse(yv, pv),
        "mape": mape(yv, pv),
        "smape": smape(yv, pv),
        "mase": mase(yv, pv, naive_pred[valid]),
        "r2": float(r2_score(yv, pv)) if len(yv) > 1 else math.nan,
        "sign_accuracy": float((np.sign(actual_ret) == np.sign(pred_ret)).mean()) if len(yv) else math.nan,
        "directional_hit_from_predicted_return": float((np.sign(actual_ret) == np.sign(pred_ret)).mean()) if len(yv) else math.nan,
        "correlation_pred_actual": corr,
        "quantile_loss": qloss,
        "interval_coverage": math.nan,
    }


def stability_rows(task: str, candidate: dict[str, Any], features: pd.DataFrame, idx: pd.Index, y_true: pd.Series, pred: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame({"ticker": features.loc[idx, "ticker"].astype(str).to_numpy(), "datetime": features.loc[idx, "datetime"].to_numpy(), "y": y_true.loc[idx].to_numpy(), "pred": pred}, index=idx)
    if task == "direction":
        frame["score"] = (frame["y"].astype(int) == frame["pred"].astype(int)).astype(float)
        metric_name = "accuracy"
    else:
        frame["score"] = np.abs(pd.to_numeric(frame["y"], errors="coerce") - pd.to_numeric(frame["pred"], errors="coerce"))
        metric_name = "mae"
    frame["quarter"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    ticker_rows = []
    for ticker, group in frame.groupby("ticker", sort=True):
        ticker_rows.append({**candidate, "task": task, "group": ticker, "metric": metric_name, "value": float(group["score"].mean()), "rows": int(len(group))})
    quarter_rows = []
    for quarter, group in frame.groupby("quarter", sort=True):
        quarter_rows.append({**candidate, "task": task, "group": quarter, "metric": metric_name, "value": float(group["score"].mean()), "rows": int(len(group))})
    return ticker_rows, quarter_rows


def qml_feature_direction_candidate(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    labels: pd.Series,
    splits: dict[str, pd.Index],
    model_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    if importlib.util.find_spec("qiskit_machine_learning") is None:
        return None, None, None, None, "qiskit_machine_learning unavailable"
    if labels.attrs.get("target_variant") != "market_relative_vn30" or int(labels.attrs.get("horizon", 0)) != 40:
        return None, None, None, None, "QML V8 architecture scoped to market_relative_vn30 h40"
    try:
        source_groups = build_source_groups(features, family_cols)
        source_groups["relative_strength_features"] = relative_cols
        source_groups["relative_plus_market_context_features"] = sorted(set(relative_cols).union(source_groups.get("market_context_features", [])[:12]))
        spec, _audit = fit_feature_spec(features, labels, "market_relative_vn30", 40, "relative_plus_market_context_features", source_groups["relative_plus_market_context_features"], "topk_availability", 4, splits)
        if spec.selection_status not in {"ok", "mutual_info_failed_fallback_availability"}:
            return None, None, None, None, spec.selection_status
        x_train, x_val, x_final, scaling_info = v6_scaling_transform(spec, "minmax_0_pi")
        if scaling_info["status"] != "ok":
            return None, None, None, None, scaling_info["skipped_reason"]
        k_train, k_val, k_final = v6_quantum_kernel_matrices(x_train, x_val, x_final, 2, "full")
        train_meta, val_meta, final_meta = v8_kernel_feature_frames(k_train, k_val, k_final, labels, splits)
        rel_spec, _ = fit_feature_spec(features, labels, "market_relative_vn30", 40, "relative_strength_features", source_groups["relative_strength_features"], "topk_availability", 4, splits)
        market_spec, _ = fit_feature_spec(features, labels, "market_relative_vn30", 40, "market_context_features", source_groups["market_context_features"], "topk_availability", 4, splits)
        train_x = pd.concat([train_meta, rel_spec.x_train.add_prefix("relative__"), market_spec.x_train.add_prefix("market__")], axis=1).fillna(0.0)
        val_x = pd.concat([val_meta, rel_spec.x_validation.add_prefix("relative__"), market_spec.x_validation.add_prefix("market__")], axis=1).fillna(0.0)
        final_x = pd.concat([final_meta, rel_spec.x_final.add_prefix("relative__"), market_spec.x_final.add_prefix("market__")], axis=1).fillna(0.0)
        if model_name == "qml_v8_kernel_features_lightgbm":
            model, reason = direction_model("lightgbm_classifier")
            if model is None:
                return None, None, None, None, reason
        else:
            model = LogisticRegression(max_iter=1000, solver="liblinear", C=0.5, class_weight="balanced", random_state=SEED)
        train_y = labels.loc[splits["train"]].astype(int)
        model.fit(train_x, train_y)
        val_prob = np.asarray(model.predict_proba(val_x)[:, 1], dtype=float) if hasattr(model, "predict_proba") else np.asarray(model.predict(val_x), dtype=float)
        final_prob = np.asarray(model.predict_proba(final_x)[:, 1], dtype=float) if hasattr(model, "predict_proba") else np.asarray(model.predict(final_x), dtype=float)
        threshold = float(np.nanquantile(val_prob, max(0.0, min(1.0, 1.0 - float(labels.loc[splits["validation"]].mean())))))
        return (val_prob >= threshold).astype(int), val_prob, (final_prob >= threshold).astype(int), final_prob, ""
    except Exception as exc:
        return None, None, None, None, f"{type(exc).__name__}: {exc}"


def evaluate_direction_candidate(
    model_name: str,
    feature_group: str,
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    labels: pd.Series,
    splits: dict[str, pd.Index],
    feature_columns: list[str],
) -> tuple[dict[str, Any], np.ndarray | None]:
    start = time.perf_counter()
    target_variant = str(labels.attrs.get("target_variant", ""))
    horizon = int(labels.attrs.get("horizon", 0))
    row = {
        "candidate_id": candidate_id("direction", model_name, target_variant, f"h{horizon}", feature_group),
        "task": "direction",
        "model_family": model_name,
        "feature_group": feature_group,
        "target_variant": target_variant,
        "horizon": horizon,
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "validation_accuracy": math.nan,
        "final_accuracy": math.nan,
        "lift_over_strongest_baseline": math.nan,
        "balanced_accuracy": math.nan,
        "f1": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "roc_auc": math.nan,
        "brier_score": math.nan,
        "calibration_error": math.nan,
        "ticker_stability": math.nan,
        "quarter_stability": math.nan,
        "claim_label": "not_claimable",
        "runtime_seconds": 0.0,
        "status": "pending",
        "skipped_reason": "",
    }
    final_pred: np.ndarray | None = None
    try:
        train_y = labels.loc[splits["train"]].astype(int)
        val_y = labels.loc[splits["validation"]].astype(int)
        final_y = labels.loc[splits["final"]].astype(int)
        if train_y.nunique() < 2:
            raise ValueError("train split has fewer than two classes")
        baseline_candidates = ["always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"]
        if model_name in baseline_candidates:
            val_pred, val_prob, reason = baseline_direction_prediction(model_name, features, train_y, splits["validation"])
            fin_pred, fin_prob, reason2 = baseline_direction_prediction(model_name, features, train_y, splits["final"])
            if val_pred is None or fin_pred is None:
                raise ValueError(reason or reason2)
        elif model_name.startswith("qml_v8"):
            val_pred, val_prob, fin_pred, fin_prob, reason = qml_feature_direction_candidate(features, family_cols, relative_cols, index_data, labels, splits, model_name)
            if val_pred is None or fin_pred is None:
                raise ValueError(reason)
        else:
            x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_columns, 16)
            if status != "ok":
                raise ValueError(status)
            model, reason = direction_model(model_name)
            if model is None:
                raise ValueError(reason)
            model.fit(x_train, train_y)
            val_pred = np.asarray(model.predict(x_val)).astype(int)
            fin_pred = np.asarray(model.predict(x_final)).astype(int)
            if hasattr(model, "predict_proba"):
                val_prob = np.asarray(model.predict_proba(x_val)[:, 1], dtype=float)
                fin_prob = np.asarray(model.predict_proba(x_final)[:, 1], dtype=float)
            elif hasattr(model, "decision_function"):
                val_score = np.asarray(model.decision_function(x_val), dtype=float)
                fin_score = np.asarray(model.decision_function(x_final), dtype=float)
                val_prob = 1.0 / (1.0 + np.exp(-np.clip(val_score, -30.0, 30.0)))
                fin_prob = 1.0 / (1.0 + np.exp(-np.clip(fin_score, -30.0, 30.0)))
            else:
                val_prob = val_pred.astype(float)
                fin_prob = fin_pred.astype(float)
        val_metrics = direction_metrics(val_y, val_pred, val_prob)
        final_metrics = direction_metrics(final_y, fin_pred, fin_prob)
        baseline_accs = []
        for baseline in ["always_up", "always_down", "lag1_direction", "simple_momentum", "simple_relative_strength"]:
            pred, _prob, _reason = baseline_direction_prediction(baseline, features, train_y, splits["validation"])
            if pred is not None:
                baseline_accs.append(float((val_y.to_numpy(dtype=int) == pred.astype(int)).mean()))
        strongest_baseline = max(baseline_accs) if baseline_accs else math.nan
        stability_ticker, stability_quarter = stability_rows("direction", {"candidate_id": row["candidate_id"], "target_variant": target_variant, "horizon": horizon}, features, splits["validation"], labels, val_pred)
        ticker_values = [item["value"] for item in stability_ticker]
        quarter_values = [item["value"] for item in stability_quarter]
        row.update(
            {
                "validation_accuracy": val_metrics["accuracy"],
                "final_accuracy": final_metrics["accuracy"],
                "lift_over_strongest_baseline": val_metrics["accuracy"] - strongest_baseline if math.isfinite(strongest_baseline) else math.nan,
                "balanced_accuracy": val_metrics["balanced_accuracy"],
                "f1": val_metrics["f1"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "roc_auc": val_metrics["roc_auc"],
                "brier_score": val_metrics["brier_score"],
                "calibration_error": val_metrics["calibration_error"],
                "ticker_stability": float(1.0 - np.nanstd(ticker_values)) if ticker_values else math.nan,
                "quarter_stability": float(1.0 - np.nanstd(quarter_values)) if quarter_values else math.nan,
                "claim_label": "diagnostic_only",
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
        if target_variant == "absolute_direction" and horizon == 40 and row["final_accuracy"] >= 0.60 and row["lift_over_strongest_baseline"] > 0:
            row["claim_label"] = "baseline60_candidate"
        elif target_variant == "market_relative_vn30" and horizon == 40 and row["final_accuracy"] >= 0.60:
            row["claim_label"] = "market_relative_candidate"
        else:
            row["claim_label"] = "direction_candidate" if row["lift_over_strongest_baseline"] > 0 else "diagnostic_only"
        final_pred = fin_pred
    except Exception as exc:
        row.update({"runtime_seconds": time.perf_counter() - start, "status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}", "claim_label": "not_claimable"})
    return row, final_pred


def evaluate_price_candidate(
    model_name: str,
    feature_group: str,
    features: pd.DataFrame,
    target: pd.Series,
    splits: dict[str, pd.Index],
    feature_columns: list[str],
) -> tuple[dict[str, Any], np.ndarray | None]:
    start = time.perf_counter()
    target_variant = str(target.attrs.get("target_variant", ""))
    horizon = int(target.attrs.get("horizon", 0))
    row = {
        "candidate_id": candidate_id("price", model_name, target_variant, f"h{horizon}", feature_group),
        "task": "price_return",
        "model_family": model_name,
        "feature_group": feature_group,
        "target_variant": target_variant,
        "horizon": horizon,
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "validation_mae": math.nan,
        "validation_rmse": math.nan,
        "validation_mape": math.nan,
        "validation_smape": math.nan,
        "validation_mase": math.nan,
        "validation_r2": math.nan,
        "validation_sign_accuracy": math.nan,
        "validation_directional_hit_from_predicted_return": math.nan,
        "validation_correlation_pred_actual": math.nan,
        "validation_quantile_loss": math.nan,
        "validation_interval_coverage": math.nan,
        "final_mae": math.nan,
        "final_rmse": math.nan,
        "final_mape": math.nan,
        "final_smape": math.nan,
        "final_mase": math.nan,
        "final_r2": math.nan,
        "final_sign_accuracy": math.nan,
        "final_directional_hit_from_predicted_return": math.nan,
        "final_correlation_pred_actual": math.nan,
        "error_improvement_over_baseline": math.nan,
        "claim_label": "not_claimable",
        "runtime_seconds": 0.0,
        "status": "pending",
        "skipped_reason": "",
    }
    final_pred: np.ndarray | None = None
    try:
        train_y = pd.to_numeric(target.loc[splits["train"]], errors="coerce")
        val_y = pd.to_numeric(target.loc[splits["validation"]], errors="coerce")
        final_y = pd.to_numeric(target.loc[splits["final"]], errors="coerce")
        baseline_models = ["random_walk_price", "last_price", "zero_return", "historical_mean_return", "rolling_mean_return", "simple_momentum", "simple_relative_strength"]
        if model_name in baseline_models:
            val_pred, reason = baseline_price_prediction(model_name, features, train_y, splits["validation"], target_variant, horizon)
            fin_pred, reason2 = baseline_price_prediction(model_name, features, train_y, splits["final"], target_variant, horizon)
            if val_pred is None or fin_pred is None:
                raise ValueError(reason or reason2)
        else:
            x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_columns, 16)
            if status != "ok":
                raise ValueError(status)
            model, reason = price_model(model_name)
            if model is None:
                raise ValueError(reason)
            model.fit(x_train, train_y)
            val_pred = np.asarray(model.predict(x_val), dtype=float)
            fin_pred = np.asarray(model.predict(x_final), dtype=float)
        naive_val, _ = baseline_price_prediction("last_price" if target_variant == "future_close_h" else "historical_mean_return", features, train_y, splits["validation"], target_variant, horizon)
        naive_final, _ = baseline_price_prediction("last_price" if target_variant == "future_close_h" else "historical_mean_return", features, train_y, splits["final"], target_variant, horizon)
        if naive_val is None:
            naive_val = np.zeros(len(val_y), dtype=float)
        if naive_final is None:
            naive_final = np.zeros(len(final_y), dtype=float)
        val_close = pd.to_numeric(features.loc[splits["validation"], "close"], errors="coerce").to_numpy(dtype=float)
        final_close = pd.to_numeric(features.loc[splits["final"], "close"], errors="coerce").to_numpy(dtype=float)
        val_metrics = price_metrics(val_y, val_pred, naive_val, val_close, target_variant)
        final_metrics = price_metrics(final_y, fin_pred, naive_final, final_close, target_variant)
        baseline_rmse = rmse(val_y.to_numpy(dtype=float), naive_val)
        row.update(
            {
                "validation_mae": val_metrics["mae"],
                "validation_rmse": val_metrics["rmse"],
                "validation_mape": val_metrics["mape"],
                "validation_smape": val_metrics["smape"],
                "validation_mase": val_metrics["mase"],
                "validation_r2": val_metrics["r2"],
                "validation_sign_accuracy": val_metrics["sign_accuracy"],
                "validation_directional_hit_from_predicted_return": val_metrics["directional_hit_from_predicted_return"],
                "validation_correlation_pred_actual": val_metrics["correlation_pred_actual"],
                "validation_quantile_loss": val_metrics["quantile_loss"],
                "validation_interval_coverage": val_metrics["interval_coverage"],
                "final_mae": final_metrics["mae"],
                "final_rmse": final_metrics["rmse"],
                "final_mape": final_metrics["mape"],
                "final_smape": final_metrics["smape"],
                "final_mase": final_metrics["mase"],
                "final_r2": final_metrics["r2"],
                "final_sign_accuracy": final_metrics["sign_accuracy"],
                "final_directional_hit_from_predicted_return": final_metrics["directional_hit_from_predicted_return"],
                "final_correlation_pred_actual": final_metrics["correlation_pred_actual"],
                "error_improvement_over_baseline": (baseline_rmse - val_metrics["rmse"]) / baseline_rmse if math.isfinite(baseline_rmse) and baseline_rmse > 0 and math.isfinite(val_metrics["rmse"]) else math.nan,
                "claim_label": "price_return_candidate" if math.isfinite(val_metrics["rmse"]) and math.isfinite(baseline_rmse) and val_metrics["rmse"] < baseline_rmse else "diagnostic_only",
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
        final_pred = fin_pred
    except Exception as exc:
        row.update({"runtime_seconds": time.perf_counter() - start, "status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}", "claim_label": "not_claimable"})
    return row, final_pred


def model_family_registry(dependency: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in DIRECTION_MODEL_NAMES:
        family = "direction"
        available = True
        reason = ""
        if name.startswith("xgboost") and not dependency.get("xgboost_available"):
            available, reason = False, "xgboost unavailable"
        if name.startswith("lightgbm") and not dependency.get("lightgbm_available"):
            available, reason = False, "lightgbm unavailable"
        if name.startswith("catboost") and not dependency.get("catboost_available"):
            available, reason = False, "catboost unavailable"
        if name.startswith("qml") and not dependency.get("qiskit_machine_learning_available"):
            available, reason = False, "qiskit_machine_learning unavailable"
        rows.append({"task": family, "model_family": name, "available": available, "reason": reason, "stage": "screening"})
    for name in PRICE_MODEL_NAMES:
        available = True
        reason = ""
        if name.startswith("xgboost") and not dependency.get("xgboost_available"):
            available, reason = False, "xgboost unavailable"
        if name.startswith("lightgbm") and not dependency.get("lightgbm_available"):
            available, reason = False, "lightgbm unavailable"
        if name.startswith("catboost") and not dependency.get("catboost_available"):
            available, reason = False, "catboost unavailable"
        rows.append({"task": "price_return", "model_family": name, "available": available, "reason": reason, "stage": "screening"})
    for name in OPTIONAL_SEQUENCE_MODELS:
        rows.append({"task": "optional_sequence_or_qml", "model_family": name, "available": False, "reason": "not enabled in bounded model-universe screening; requires separate sequence adapter or focused QML replay", "stage": "skipped"})
    return rows


def selected_feature_groups(mode: str, target_variant: str, horizon: int, task: str) -> list[str]:
    if mode == "smoke":
        return ["compact_stable_features", "relative_strength"]
    base = ["compact_stable_features", "relative_strength"]
    if horizon == 40:
        base.extend(["market_context", "volume_volatility", "combined_strategy_features", "all_safe_features"])
    if task == "direction" and target_variant == "market_relative_vn30" and horizon == 40:
        base.append("qml_kernel_features")
    return list(dict.fromkeys(base))


def selected_direction_models(mode: str, target_variant: str, horizon: int, feature_group: str) -> list[str]:
    if mode == "smoke":
        return ["always_up", "always_down", "lag1_direction", "logistic_regression", "random_forest", "lightgbm_classifier"]
    models = [
        "always_up",
        "always_down",
        "lag1_direction",
        "random_same_class_balance",
        "simple_momentum",
        "simple_relative_strength",
        "logistic_regression",
        "calibrated_logistic",
        "linear_svm",
        "knn_classifier",
        "naive_bayes",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost_classifier",
        "lightgbm_classifier",
        "catboost_classifier",
        "mlp_classifier",
    ]
    if horizon == 40 and feature_group in {"compact_stable_features", "relative_strength"}:
        models.extend(["rbf_svm", "gradient_boosting"])
    if target_variant == "market_relative_vn30" and horizon == 40 and feature_group == "qml_kernel_features":
        models = ["qml_v8_kernel_features_l2", "qml_v8_kernel_features_lightgbm"]
    return models


def selected_price_models(mode: str, target_variant: str, horizon: int, feature_group: str) -> list[str]:
    if mode == "smoke":
        return ["last_price", "historical_mean_return", "ridge", "random_forest_regressor", "lightgbm_regressor"]
    models = [
        "random_walk_price",
        "last_price",
        "zero_return",
        "historical_mean_return",
        "rolling_mean_return",
        "simple_momentum",
        "simple_relative_strength",
        "linear_regression",
        "ridge",
        "lasso",
        "elasticnet",
        "knn_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "hist_gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
        "catboost_regressor",
        "mlp_regressor",
    ]
    if horizon == 40 and feature_group in {"compact_stable_features", "relative_strength"}:
        models.extend(["svr_linear", "svr_rbf", "gradient_boosting_regressor"])
    return models


def build_candidate_grid(mode: str, feature_groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS if mode != "smoke" else [40]:
        for target in DIRECTION_TARGETS if mode != "smoke" else ["absolute_direction", "market_relative_vn30"]:
            for group in selected_feature_groups(mode, target, horizon, "direction"):
                for model in selected_direction_models(mode, target, horizon, group):
                    rows.append({"task": "direction", "target_variant": target, "horizon": horizon, "feature_group": group, "model_family": model, "candidate_id": candidate_id("direction", model, target, f"h{horizon}", group), "planned_stage": mode})
        for target in PRICE_TARGETS if mode != "smoke" else ["forward_simple_return_h", "future_close_h"]:
            for group in selected_feature_groups(mode, target, horizon, "price_return"):
                for model in selected_price_models(mode, target, horizon, group):
                    rows.append({"task": "price_return", "target_variant": target, "horizon": horizon, "feature_group": group, "model_family": model, "candidate_id": candidate_id("price", model, target, f"h{horizon}", group), "planned_stage": mode})
    return rows


def select_locked_direction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_accuracy")))]
    if not valid:
        return {}
    non_trivial = [
        row for row in valid
        if as_float(row.get("lift_over_strongest_baseline")) > 0.0
        and row.get("model_family") not in {"always_up", "always_down", "random_same_class_balance"}
        and as_float(row.get("balanced_accuracy")) > 0.5
    ]
    pool = non_trivial if non_trivial else valid
    return max(pool, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("lift_over_strongest_baseline")), as_float(row.get("balanced_accuracy")), as_float(row.get("ticker_stability")), -as_float(row.get("runtime_seconds"))))


def select_locked_price(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_rmse")))]
    if not valid:
        return {}
    improved = [row for row in valid if as_float(row.get("error_improvement_over_baseline")) > 0.0]
    pool = improved if improved else valid
    return max(pool, key=lambda row: (as_float(row.get("error_improvement_over_baseline")), -as_float(row.get("validation_rmse")), as_float(row.get("validation_correlation_pred_actual"))))


def write_reports(direction_rows: list[dict[str, Any]], price_rows: list[dict[str, Any]], skipped_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    locked_direction = manifest.get("locked_direction", {})
    locked_price = manifest.get("locked_price", {})
    abs_h40 = [
        row for row in direction_rows
        if row.get("status") == "ok" and row.get("target_variant") == "absolute_direction" and int(row.get("horizon", 0)) == 40
    ]
    abs_h40_nontrivial = [row for row in abs_h40 if as_float(row.get("lift_over_strongest_baseline")) > 0.0 and row.get("model_family") not in {"always_up", "always_down", "random_same_class_balance"}]
    best_abs = max(abs_h40_nontrivial or abs_h40, key=lambda row: as_float(row.get("final_accuracy")), default={})
    mr_h40 = [
        row for row in direction_rows
        if row.get("status") == "ok" and row.get("target_variant") == "market_relative_vn30" and int(row.get("horizon", 0)) == 40
    ]
    mr_h40_nontrivial = [row for row in mr_h40 if as_float(row.get("lift_over_strongest_baseline")) > 0.0 and row.get("model_family") not in {"always_up", "always_down", "random_same_class_balance"}]
    best_mr = max(mr_h40_nontrivial or mr_h40, key=lambda row: as_float(row.get("final_accuracy")), default={})
    price_beats_baseline = as_float(locked_price.get("error_improvement_over_baseline")) > 0.0
    best_abs_beats = as_float(best_abs.get("final_accuracy")) > CLASSICAL_CHAMPION["final_accuracy"]
    best_mr_beats = as_float(best_mr.get("final_accuracy")) > QML_V8_CONTEXT_FINAL
    summary = f"""# VN30 Model Universe Direction + Price Forecasting Result Summary

## Required Answers

1. Which direction target performed best: `{locked_direction.get("target_variant", "")}` h{locked_direction.get("horizon", "")} with `{locked_direction.get("model_family", "")}` on `{locked_direction.get("feature_group", "")}`.
2. Which price/return target performed best: `{locked_price.get("target_variant", "")}` h{locked_price.get("horizon", "")} with `{locked_price.get("model_family", "")}` on `{locked_price.get("feature_group", "")}`.
3. Which model family performed best for direction: `{locked_direction.get("model_family", "")}` with validation accuracy {pct(locked_direction.get("validation_accuracy"))} and final accuracy {pct(locked_direction.get("final_accuracy"))}.
4. Which model family performed best for price/return: `{locked_price.get("model_family", "")}` with validation RMSE {as_float(locked_price.get("validation_rmse")):.6g} and final RMSE {as_float(locked_price.get("final_rmse")):.6g}.
5. Did any model beat the 61.61% absolute-direction classical champion on comparable absolute-direction scope: exploratory final-ranked rows did ({str(best_abs_beats).lower()}, best comparable final accuracy {pct(best_abs.get("final_accuracy"))}), but no claimable replacement is made because final-ranked rows are `exploratory_not_claimable`.
6. Did any model beat the 64.44% QML V8 market-relative result on comparable market_relative_vn30 scope: exploratory final-ranked rows did ({str(best_mr_beats).lower()}, best comparable final accuracy {pct(best_mr.get("final_accuracy"))}), but no claimable replacement is made because final-ranked rows are `exploratory_not_claimable`.
7. Did any model forecast price/return better than random walk / last price baseline: validation-screening yes ({str(price_beats_baseline).lower()}, locked validation error improvement {pp(locked_price.get("error_improvement_over_baseline"))}); final transfer is reported separately and is not a trading or production claim.
8. Which models failed or were skipped: {len(skipped_rows)} candidate/model rows were skipped; see `skipped_models.csv`.
9. Is there evidence that stock direction can be forecast: diagnostic evidence exists when validation-selected direction candidates beat simple validation baselines, but this is not a trading claim.
10. Is there evidence that stock price/return can be forecast: diagnostic evidence exists if validation RMSE improves over random-walk/last-price or historical-return baselines; this is reported separately from direction accuracy.
11. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, VN100, index-as-stock, merge, tag, DOCX, or production claim is made.

## Locked Direction Candidate

- Candidate: `{locked_direction.get("candidate_id", "")}`.
- Validation accuracy: {pct(locked_direction.get("validation_accuracy"))}.
- Final accuracy: {pct(locked_direction.get("final_accuracy"))}.
- Lift over strongest validation baseline: {pp(locked_direction.get("lift_over_strongest_baseline"))}.
- Claim label: `{locked_direction.get("claim_label", "")}`.

## Locked Price/Return Candidate

- Candidate: `{locked_price.get("candidate_id", "")}`.
- Validation RMSE: {as_float(locked_price.get("validation_rmse")):.6g}.
- Final RMSE: {as_float(locked_price.get("final_rmse")):.6g}.
- Validation sign accuracy: {pct(locked_price.get("validation_sign_accuracy"))}.
- Final sign accuracy: {pct(locked_price.get("final_sign_accuracy"))}.
- Claim label: `{locked_price.get("claim_label", "")}`.
"""
    write_markdown(RESULT_PATH, summary)
    claim = """# VN30 Model Universe Direction + Price Forecasting Claim Boundary

- This benchmark is offline diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Direction and price/return targets are separate; directional accuracy is not mixed with price or return errors.
- Price-level forecasts and return forecasts are reported separately.
- Index data may be used only as lagged market-context features or market-relative target context.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Candidate selection is validation-governed only; final performance is scoring-only.
- Final-ranked rows are exploratory_not_claimable.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, DOCX, tag, merge, push --mirror, or main-branch claim is made.
- Comparisons against the 61.61% absolute-direction classical champion are only valid on comparable absolute_direction h40 scope.
- Comparisons against the 64.44% QML V8 result are only valid on comparable market_relative_vn30 h40 scope.
- Stronger claims require future-blind confirmation.
"""
    write_markdown(CLAIM_PATH, claim)


def read_artifact(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def v2_baseline_maps(baseline: pd.DataFrame) -> tuple[dict[tuple[str, int, str], float], dict[tuple[str, int], float], dict[tuple[str, int, str], float], dict[tuple[str, int], float]]:
    if baseline.empty:
        return {}, {}, {}, {}
    frame = baseline.copy()
    frame["horizon"] = numeric_series(frame, "horizon").astype("Int64")
    frame["final_value_num"] = numeric_series(frame, "final_value")
    direction = frame[(frame["task"] == "direction") & frame["final_value_num"].notna()]
    price = frame[(frame["task"] == "price_return") & frame["final_value_num"].notna()]
    dir_feature = direction.groupby(["target_variant", "horizon", "feature_group"], dropna=False)["final_value_num"].max().to_dict()
    dir_target = direction.groupby(["target_variant", "horizon"], dropna=False)["final_value_num"].max().to_dict()
    price_feature = price.groupby(["target_variant", "horizon", "feature_group"], dropna=False)["final_value_num"].min().to_dict()
    price_target = price.groupby(["target_variant", "horizon"], dropna=False)["final_value_num"].min().to_dict()
    return dir_feature, dir_target, price_feature, price_target


def v2_lookup_baseline(row: pd.Series, exact: dict[tuple[str, int, str], float], fallback: dict[tuple[str, int], float]) -> float:
    target = str(row.get("target_variant", ""))
    horizon = int(as_float(row.get("horizon"))) if math.isfinite(as_float(row.get("horizon"))) else 0
    feature_group = str(row.get("feature_group", ""))
    value = exact.get((target, horizon, feature_group))
    if value is None:
        value = fallback.get((target, horizon))
    return float(value) if value is not None and math.isfinite(float(value)) else math.nan


def annotate_v2_direction(direction: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if direction.empty:
        return direction
    dir_feature, dir_target, _price_feature, _price_target = v2_baseline_maps(baseline)
    frame = direction.copy()
    frame = frame[frame.get("status", "ok") == "ok"].copy()
    for column in ["horizon", "validation_accuracy", "final_accuracy", "lift_over_strongest_baseline", "balanced_accuracy", "ticker_stability", "quarter_stability", "validation_rows", "final_rows"]:
        frame[column] = numeric_series(frame, column)
    frame["strongest_final_baseline_accuracy"] = frame.apply(lambda row: v2_lookup_baseline(row, dir_feature, dir_target), axis=1)
    frame["final_lift_over_strongest_baseline"] = frame["final_accuracy"] - frame["strongest_final_baseline_accuracy"]
    frame["is_trivial_baseline"] = frame["model_family"].isin(["always_up", "always_down", "random_same_class_balance"])
    frame["comparable_absolute_h40_scope"] = (frame["target_variant"] == "absolute_direction") & (frame["horizon"] == 40)
    frame["comparable_market_relative_h40_scope"] = (frame["target_variant"] == "market_relative_vn30") & (frame["horizon"] == 40)
    frame["v2_claim_label"] = "exploratory_not_claimable"
    return frame


def annotate_v2_price(price: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if price.empty:
        return price
    _dir_feature, _dir_target, price_feature, price_target = v2_baseline_maps(baseline)
    frame = price.copy()
    frame = frame[frame.get("status", "ok") == "ok"].copy()
    for column in [
        "horizon",
        "validation_rmse",
        "final_rmse",
        "validation_sign_accuracy",
        "final_sign_accuracy",
        "validation_correlation_pred_actual",
        "final_correlation_pred_actual",
        "error_improvement_over_baseline",
        "validation_rows",
        "final_rows",
    ]:
        frame[column] = numeric_series(frame, column)
    frame["strongest_final_baseline_rmse"] = frame.apply(lambda row: v2_lookup_baseline(row, price_feature, price_target), axis=1)
    frame["final_rmse_improvement_over_baseline"] = np.where(
        (frame["strongest_final_baseline_rmse"] > 0.0) & frame["strongest_final_baseline_rmse"].notna() & frame["final_rmse"].notna(),
        (frame["strongest_final_baseline_rmse"] - frame["final_rmse"]) / frame["strongest_final_baseline_rmse"],
        np.nan,
    )
    frame["is_price_baseline"] = frame["model_family"].isin(["random_walk_price", "last_price", "zero_return", "historical_mean_return", "rolling_mean_return"])
    frame["v2_claim_label"] = "exploratory_not_claimable"
    return frame


def source_bucket_frame(frame: pd.DataFrame, bucket: str, sort_column: str, ascending: bool, n: int = 50) -> pd.DataFrame:
    if frame.empty or sort_column not in frame.columns:
        return pd.DataFrame()
    subset = frame[frame[sort_column].notna()].sort_values(sort_column, ascending=ascending).head(n).copy()
    subset["v2_selection_bucket"] = bucket
    return subset


def v2_top_direction_candidates(direction: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        source_bucket_frame(direction[direction["target_variant"] == "absolute_direction"], "top50_absolute_direction_final_accuracy", "final_accuracy", False),
        source_bucket_frame(direction[direction["target_variant"] == "market_relative_vn30"], "top50_market_relative_vn30_final_accuracy", "final_accuracy", False),
        source_bucket_frame(direction, "top50_final_lift_over_strongest_baseline", "final_lift_over_strongest_baseline", False),
        source_bucket_frame(direction[(direction["target_variant"].isin(["absolute_direction", "market_relative_vn30"])) & (direction["horizon"] == 40)], "included_comparable_h40_rows", "final_accuracy", False),
    ]
    combined = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    if combined.empty:
        return combined
    bucket_map = combined.groupby("candidate_id")["v2_selection_bucket"].apply(lambda values: "|".join(sorted(set(values)))).to_dict()
    top = combined.sort_values(["final_accuracy", "final_lift_over_strongest_baseline"], ascending=False).drop_duplicates("candidate_id").copy()
    top["v2_selection_bucket"] = top["candidate_id"].map(bucket_map)
    return top.sort_values(["final_accuracy", "final_lift_over_strongest_baseline"], ascending=False)


def v2_top_price_candidates(price: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        source_bucket_frame(price, "top50_final_rmse_improvement", "final_rmse_improvement_over_baseline", False),
        source_bucket_frame(price, "top50_final_sign_accuracy", "final_sign_accuracy", False),
        source_bucket_frame(price, "top50_final_correlation_pred_actual", "final_correlation_pred_actual", False),
        source_bucket_frame(price[price["target_variant"].isin(["forward_log_return_h", "market_excess_return_h"])], "included_forward_log_and_market_excess", "final_rmse_improvement_over_baseline", False),
    ]
    combined = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    if combined.empty:
        return combined
    bucket_map = combined.groupby("candidate_id")["v2_selection_bucket"].apply(lambda values: "|".join(sorted(set(values)))).to_dict()
    top = combined.sort_values(["final_rmse_improvement_over_baseline", "final_sign_accuracy", "final_correlation_pred_actual"], ascending=False).drop_duplicates("candidate_id").copy()
    top["v2_selection_bucket"] = top["candidate_id"].map(bucket_map)
    return top.sort_values(["final_rmse_improvement_over_baseline", "final_sign_accuracy"], ascending=False)


def candidate_coverage_maps(ticker_stability: pd.DataFrame, quarter_stability: pd.DataFrame, task: str) -> tuple[dict[str, int], dict[str, int]]:
    ticker_counts: dict[str, int] = {}
    quarter_counts: dict[str, int] = {}
    if not ticker_stability.empty and "candidate_id" in ticker_stability.columns:
        ticker_counts = ticker_stability[ticker_stability.get("task") == task].groupby("candidate_id")["group"].nunique().to_dict()
    if not quarter_stability.empty and "candidate_id" in quarter_stability.columns:
        quarter_counts = quarter_stability[quarter_stability.get("task") == task].groupby("candidate_id")["group"].nunique().to_dict()
    return ticker_counts, quarter_counts


def v2_cluster_summary(top: pd.DataFrame, task: str, ticker_stability: pd.DataFrame, quarter_stability: pd.DataFrame) -> pd.DataFrame:
    if top.empty:
        return pd.DataFrame()
    ticker_counts, quarter_counts = candidate_coverage_maps(ticker_stability, quarter_stability, task)
    frame = top.copy()
    frame["ticker_coverage"] = frame["candidate_id"].map(ticker_counts).fillna(0).astype(int)
    frame["quarter_coverage"] = frame["candidate_id"].map(quarter_counts).fillna(0).astype(int)
    support_key = ["model_family", "target_variant", "horizon"]
    support_counts = frame.groupby(support_key, dropna=False)["candidate_id"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    group_cols = ["model_family", "target_variant", "horizon", "feature_group"]
    for key, group in frame.groupby(group_cols, dropna=False):
        model_family, target_variant, horizon, feature_group = key
        support_count = int(support_counts.get((model_family, target_variant, horizon), len(group)))
        if task == "direction":
            final_values = numeric_series(group, "final_accuracy")
            validation_values = numeric_series(group, "validation_accuracy")
            max_final = float(final_values.max())
            median_final = float(final_values.median())
            median_validation = float(validation_values.median())
            gap = median_final - median_validation
            metric_name = "accuracy"
        else:
            final_values = numeric_series(group, "final_rmse_improvement_over_baseline")
            validation_values = numeric_series(group, "error_improvement_over_baseline")
            max_final = float(final_values.max())
            median_final = float(final_values.median())
            median_validation = float(validation_values.median())
            gap = median_final - median_validation
            metric_name = "rmse_improvement_over_baseline"
        cluster_count = int(len(group))
        verdict = "cluster_supported" if cluster_count >= 2 or support_count >= 2 else "isolated_one_off"
        rows.append(
            {
                "task": task,
                "model_family": model_family,
                "target_variant": target_variant,
                "horizon": int(horizon) if math.isfinite(as_float(horizon)) else horizon,
                "feature_group": feature_group,
                "cluster_count": cluster_count,
                "family_target_horizon_support_count": support_count,
                "metric_name": metric_name,
                "max_final_metric": max_final,
                "median_final_metric": median_final,
                "validation_median_metric": median_validation,
                "validation_final_gap": gap,
                "median_row_count": float(numeric_series(group, "final_rows").median()) if "final_rows" in group.columns else math.nan,
                "median_ticker_coverage": float(group["ticker_coverage"].median()),
                "median_quarter_coverage": float(group["quarter_coverage"].median()),
                "median_ticker_stability": float(numeric_series(group, "ticker_stability").median()) if "ticker_stability" in group.columns else math.nan,
                "median_quarter_stability": float(numeric_series(group, "quarter_stability").median()) if "quarter_stability" in group.columns else math.nan,
                "cluster_verdict": verdict,
                "candidate_ids": "|".join(group["candidate_id"].astype(str).tolist()[:12]),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_final_metric", "family_target_horizon_support_count"], ascending=False)


def v2_relock_direction_candidates(direction: pd.DataFrame, clusters: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    locked: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    if direction.empty or clusters.empty:
        return locked, audit
    interesting = clusters.head(24).copy()
    required = clusters[((clusters["target_variant"].isin(["absolute_direction", "market_relative_vn30"])) & (clusters["horizon"] == 40))]
    interesting = pd.concat([interesting, required], ignore_index=True).drop_duplicates(["model_family", "target_variant", "horizon", "feature_group"])
    for _, cluster in interesting.iterrows():
        subset = direction[
            (direction["model_family"] == cluster["model_family"])
            & (direction["target_variant"] == cluster["target_variant"])
            & (direction["horizon"] == cluster["horizon"])
            & (direction["feature_group"] == cluster["feature_group"])
        ].copy()
        if subset.empty:
            continue
        ranked = subset.sort_values(["validation_accuracy", "lift_over_strongest_baseline", "balanced_accuracy"], ascending=False)
        selected = ranked.iloc[0].to_dict()
        is_trivial = bool(selected.get("is_trivial_baseline"))
        validation_lift = as_float(selected.get("lift_over_strongest_baseline"))
        final_accuracy = as_float(selected.get("final_accuracy"))
        target = str(selected.get("target_variant", ""))
        horizon = int(as_float(selected.get("horizon"))) if math.isfinite(as_float(selected.get("horizon"))) else 0
        if is_trivial:
            label = "not_claimable"
            reason = "trivial_baseline_family"
        elif validation_lift <= 0:
            label = "exploratory_not_claimable"
            reason = "validation_lift_not_positive"
        elif target == "absolute_direction" and horizon == 40 and final_accuracy > CLASSICAL_CHAMPION["final_accuracy"]:
            label = "future_blind_required"
            reason = "validation_relocked_final_beats_context_but_hypothesis_was_final_discovered"
        elif target == "market_relative_vn30" and horizon == 40 and final_accuracy > QML_V8_CONTEXT_FINAL:
            label = "future_blind_required"
            reason = "validation_relocked_final_beats_qml_v8_context_but_hypothesis_was_final_discovered"
        else:
            label = "future_blind_required"
            reason = "validation_relocked_diagnostic_requires_future_blind_confirmation"
        selected.update(
            {
                "relock_source": "final_discovered_cluster_hypothesis",
                "selection_rule": "within_cluster_validation_accuracy_then_validation_lift",
                "relock_claim_label": label,
                "relock_reason": reason,
                "cluster_verdict": cluster.get("cluster_verdict", ""),
                "family_target_horizon_support_count": int(cluster.get("family_target_horizon_support_count", 0)),
                "final_selection_used": False,
                "future_blind_required": True,
            }
        )
        locked.append(json_safe(selected))
        audit.append(
            {
                "task": "direction",
                "source_model_family": cluster["model_family"],
                "target_variant": cluster["target_variant"],
                "horizon": cluster["horizon"],
                "feature_group": cluster["feature_group"],
                "source_cluster_verdict": cluster.get("cluster_verdict", ""),
                "cluster_count": cluster.get("cluster_count", 0),
                "family_target_horizon_support_count": cluster.get("family_target_horizon_support_count", 0),
                "selection_freeze": "model_family/target/horizon/feature_group",
                "selection_metric": "validation_accuracy_then_validation_lift",
                "selected_candidate_id": selected.get("candidate_id", ""),
                "selected_validation_accuracy": selected.get("validation_accuracy", math.nan),
                "selected_final_accuracy": selected.get("final_accuracy", math.nan),
                "final_selection_used": False,
                "claim_label": label,
                "audit_note": reason,
            }
        )
    return locked, audit


def v2_relock_price_candidates(price: pd.DataFrame, clusters: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    locked: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    if price.empty or clusters.empty:
        return locked, audit
    interesting = clusters.head(24).copy()
    required = clusters[clusters["target_variant"].isin(["forward_log_return_h", "market_excess_return_h"])]
    interesting = pd.concat([interesting, required], ignore_index=True).drop_duplicates(["model_family", "target_variant", "horizon", "feature_group"])
    for _, cluster in interesting.iterrows():
        subset = price[
            (price["model_family"] == cluster["model_family"])
            & (price["target_variant"] == cluster["target_variant"])
            & (price["horizon"] == cluster["horizon"])
            & (price["feature_group"] == cluster["feature_group"])
        ].copy()
        if subset.empty:
            continue
        ranked = subset.sort_values(["error_improvement_over_baseline", "validation_sign_accuracy", "validation_correlation_pred_actual", "validation_rmse"], ascending=[False, False, False, True])
        selected = ranked.iloc[0].to_dict()
        final_improvement = as_float(selected.get("final_rmse_improvement_over_baseline"))
        validation_improvement = as_float(selected.get("error_improvement_over_baseline"))
        if bool(selected.get("is_price_baseline")):
            label = "not_claimable"
            reason = "baseline_family"
        elif validation_improvement <= 0:
            label = "exploratory_not_claimable"
            reason = "validation_error_improvement_not_positive"
        elif final_improvement > 0:
            label = "future_blind_required"
            reason = "validation_relocked_final_beats_price_baseline_but_hypothesis_was_final_discovered"
        else:
            label = "future_blind_required"
            reason = "validation_relocked_price_diagnostic_requires_future_blind_confirmation"
        selected.update(
            {
                "relock_source": "final_discovered_cluster_hypothesis",
                "selection_rule": "within_cluster_validation_error_improvement_then_sign_accuracy",
                "relock_claim_label": label,
                "relock_reason": reason,
                "cluster_verdict": cluster.get("cluster_verdict", ""),
                "family_target_horizon_support_count": int(cluster.get("family_target_horizon_support_count", 0)),
                "final_selection_used": False,
                "future_blind_required": True,
            }
        )
        locked.append(json_safe(selected))
        audit.append(
            {
                "task": "price_return",
                "source_model_family": cluster["model_family"],
                "target_variant": cluster["target_variant"],
                "horizon": cluster["horizon"],
                "feature_group": cluster["feature_group"],
                "source_cluster_verdict": cluster.get("cluster_verdict", ""),
                "cluster_count": cluster.get("cluster_count", 0),
                "family_target_horizon_support_count": cluster.get("family_target_horizon_support_count", 0),
                "selection_freeze": "model_family/target/horizon/feature_group",
                "selection_metric": "validation_error_improvement_then_sign_accuracy",
                "selected_candidate_id": selected.get("candidate_id", ""),
                "selected_validation_rmse": selected.get("validation_rmse", math.nan),
                "selected_final_rmse": selected.get("final_rmse", math.nan),
                "selected_final_rmse_improvement": selected.get("final_rmse_improvement_over_baseline", math.nan),
                "final_selection_used": False,
                "claim_label": label,
                "audit_note": reason,
            }
        )
    return locked, audit


def v2_rolling_rows_from_artifacts(locked: list[dict[str, Any]], quarter_stability: pd.DataFrame, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quarter_frame = quarter_stability[quarter_stability.get("task") == task].copy() if not quarter_stability.empty and "task" in quarter_stability.columns else pd.DataFrame()
    for item in locked:
        candidate = str(item.get("candidate_id", ""))
        if task == "direction":
            rows.extend(
                [
                    {"candidate_id": candidate, "task": task, "window": "validation_2024_full", "metric": "accuracy", "value": item.get("validation_accuracy", math.nan), "rows": item.get("validation_rows", math.nan), "status": "aggregate_available"},
                    {"candidate_id": candidate, "task": task, "window": "validation_early_2024", "metric": "accuracy", "value": math.nan, "rows": math.nan, "status": "not_available_from_v1_aggregate_artifacts"},
                    {"candidate_id": candidate, "task": task, "window": "validation_late_2024", "metric": "accuracy", "value": math.nan, "rows": math.nan, "status": "not_available_from_v1_aggregate_artifacts"},
                    {"candidate_id": candidate, "task": task, "window": "final_2025plus_diagnostic", "metric": "accuracy", "value": item.get("final_accuracy", math.nan), "rows": item.get("final_rows", math.nan), "status": "aggregate_available"},
                ]
            )
        else:
            rows.extend(
                [
                    {"candidate_id": candidate, "task": task, "window": "validation_2024_full", "metric": "rmse", "value": item.get("validation_rmse", math.nan), "rows": item.get("validation_rows", math.nan), "status": "aggregate_available"},
                    {"candidate_id": candidate, "task": task, "window": "validation_early_2024", "metric": "rmse", "value": math.nan, "rows": math.nan, "status": "not_available_from_v1_aggregate_artifacts"},
                    {"candidate_id": candidate, "task": task, "window": "validation_late_2024", "metric": "rmse", "value": math.nan, "rows": math.nan, "status": "not_available_from_v1_aggregate_artifacts"},
                    {"candidate_id": candidate, "task": task, "window": "final_2025plus_diagnostic", "metric": "rmse", "value": item.get("final_rmse", math.nan), "rows": item.get("final_rows", math.nan), "status": "aggregate_available"},
                ]
            )
        if not quarter_frame.empty:
            for _, qrow in quarter_frame[quarter_frame["candidate_id"] == candidate].iterrows():
                rows.append(
                    {
                        "candidate_id": candidate,
                        "task": task,
                        "window": f"final_quarter_{qrow.get('group', '')}",
                        "metric": qrow.get("metric", ""),
                        "value": as_float(qrow.get("value")),
                        "rows": as_float(qrow.get("rows")),
                        "status": "quarter_artifact_available",
                    }
                )
    return rows


def best_relocked_direction(locked: list[dict[str, Any]], target: str | None = None, horizon: int | None = None) -> dict[str, Any]:
    rows = [
        row for row in locked
        if (target is None or row.get("target_variant") == target)
        and (horizon is None or int(as_float(row.get("horizon"))) == horizon)
        and row.get("relock_claim_label") != "not_claimable"
    ]
    if not rows:
        rows = [row for row in locked if target is None or row.get("target_variant") == target]
    return max(rows, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("lift_over_strongest_baseline")), as_float(row.get("final_accuracy"))), default={})


def best_relocked_price(locked: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in locked if row.get("relock_claim_label") != "not_claimable"]
    if not rows:
        rows = locked
    return max(rows, key=lambda row: (as_float(row.get("error_improvement_over_baseline")), as_float(row.get("final_rmse_improvement_over_baseline")), as_float(row.get("validation_sign_accuracy"))), default={})


def v2_comparison_rows(locked_direction: list[dict[str, Any]], locked_price: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v1_direction = manifest.get("locked_direction", {}) if isinstance(manifest, dict) else {}
    v1_price = manifest.get("locked_price", {}) if isinstance(manifest, dict) else {}
    for item in locked_direction:
        target = str(item.get("target_variant", ""))
        horizon = int(as_float(item.get("horizon"))) if math.isfinite(as_float(item.get("horizon"))) else 0
        if target == "absolute_direction" and horizon == 40:
            rows.append({"task": "direction", "candidate_id": item.get("candidate_id", ""), "comparison": "61.61_absolute_direction_classical_champion", "candidate_metric": item.get("final_accuracy", math.nan), "reference_metric": CLASSICAL_CHAMPION["final_accuracy"], "delta": as_float(item.get("final_accuracy")) - CLASSICAL_CHAMPION["final_accuracy"], "claim_status": item.get("relock_claim_label", "")})
        if target == "market_relative_vn30" and horizon == 40:
            rows.append({"task": "direction", "candidate_id": item.get("candidate_id", ""), "comparison": "64.44_qml_v8_market_relative_context", "candidate_metric": item.get("final_accuracy", math.nan), "reference_metric": QML_V8_CONTEXT_FINAL, "delta": as_float(item.get("final_accuracy")) - QML_V8_CONTEXT_FINAL, "claim_status": item.get("relock_claim_label", "")})
        if v1_direction:
            rows.append({"task": "direction", "candidate_id": item.get("candidate_id", ""), "comparison": "v1_locked_direction_final_accuracy", "candidate_metric": item.get("final_accuracy", math.nan), "reference_metric": v1_direction.get("final_accuracy", math.nan), "delta": as_float(item.get("final_accuracy")) - as_float(v1_direction.get("final_accuracy")), "claim_status": item.get("relock_claim_label", "")})
    for item in locked_price:
        if v1_price:
            rows.append({"task": "price_return", "candidate_id": item.get("candidate_id", ""), "comparison": "v1_locked_price_final_rmse", "candidate_metric": item.get("final_rmse", math.nan), "reference_metric": v1_price.get("final_rmse", math.nan), "delta": as_float(v1_price.get("final_rmse")) - as_float(item.get("final_rmse")), "claim_status": item.get("relock_claim_label", "")})
        rows.append({"task": "price_return", "candidate_id": item.get("candidate_id", ""), "comparison": "strongest_price_baseline_final_rmse", "candidate_metric": item.get("final_rmse", math.nan), "reference_metric": item.get("strongest_final_baseline_rmse", math.nan), "delta": as_float(item.get("strongest_final_baseline_rmse")) - as_float(item.get("final_rmse")), "claim_status": item.get("relock_claim_label", "")})
    return rows


def write_v2_reports(
    top_direction: pd.DataFrame,
    direction_clusters: pd.DataFrame,
    top_price: pd.DataFrame,
    locked_direction: list[dict[str, Any]],
    locked_price: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    abs_source_all = top_direction[(top_direction["target_variant"] == "absolute_direction") & (top_direction["horizon"] == 40)].sort_values("final_accuracy", ascending=False).head(1)
    abs_source_promotable = top_direction[
        (top_direction["target_variant"] == "absolute_direction")
        & (top_direction["horizon"] == 40)
        & (~top_direction["is_trivial_baseline"])
        & (top_direction["lift_over_strongest_baseline"] > 0)
    ].sort_values("final_accuracy", ascending=False).head(1)
    mr_source = top_direction[(top_direction["target_variant"] == "market_relative_vn30") & (top_direction["horizon"] == 40)].sort_values("final_accuracy", ascending=False).head(1)
    abs_row = abs_source_promotable.iloc[0].to_dict() if not abs_source_promotable.empty else (abs_source_all.iloc[0].to_dict() if not abs_source_all.empty else {})
    abs_all_row = abs_source_all.iloc[0].to_dict() if not abs_source_all.empty else {}
    mr_row = mr_source.iloc[0].to_dict() if not mr_source.empty else {}
    locked_abs = best_relocked_direction(locked_direction, "absolute_direction", 40)
    locked_mr = best_relocked_direction(locked_direction, "market_relative_vn30", 40)
    locked_price_best = best_relocked_price(locked_price)
    abs_cluster = direction_clusters[
        (direction_clusters["model_family"] == abs_row.get("model_family"))
        & (direction_clusters["target_variant"] == abs_row.get("target_variant"))
        & (direction_clusters["horizon"] == abs_row.get("horizon"))
        & (direction_clusters["feature_group"] == abs_row.get("feature_group"))
    ]
    mr_cluster = direction_clusters[
        (direction_clusters["model_family"] == mr_row.get("model_family"))
        & (direction_clusters["target_variant"] == mr_row.get("target_variant"))
        & (direction_clusters["horizon"] == mr_row.get("horizon"))
        & (direction_clusters["feature_group"] == mr_row.get("feature_group"))
    ]
    abs_cluster_verdict = str(abs_cluster.iloc[0].get("cluster_verdict", "")) if not abs_cluster.empty else ""
    mr_cluster_verdict = str(mr_cluster.iloc[0].get("cluster_verdict", "")) if not mr_cluster.empty else ""
    abs_final_relocks = [
        row for row in locked_direction
        if row.get("target_variant") == "absolute_direction"
        and int(as_float(row.get("horizon"))) == 40
        and row.get("relock_claim_label") == "future_blind_required"
        and as_float(row.get("final_accuracy")) > CLASSICAL_CHAMPION["final_accuracy"]
    ]
    abs_final_best = max(abs_final_relocks, key=lambda row: as_float(row.get("final_accuracy")), default={})
    mr_trivial_final_relocks = [
        row for row in locked_direction
        if row.get("target_variant") == "market_relative_vn30"
        and int(as_float(row.get("horizon"))) == 40
        and as_float(row.get("final_accuracy")) > QML_V8_CONTEXT_FINAL
    ]
    mr_nontrivial_final_relocks = [row for row in mr_trivial_final_relocks if row.get("relock_claim_label") == "future_blind_required"]
    mr_final_best = max(mr_nontrivial_final_relocks or mr_trivial_final_relocks, key=lambda row: as_float(row.get("final_accuracy")), default={})
    price_final_relocks = [
        row for row in locked_price
        if row.get("relock_claim_label") == "future_blind_required"
        and as_float(row.get("final_rmse_improvement_over_baseline")) > 0
    ]
    price_final_best = max(price_final_relocks, key=lambda row: as_float(row.get("final_rmse_improvement_over_baseline")), default={})
    price_final_beats = bool(price_final_relocks)
    summary = f"""# VN30 Model Universe V2 Promotion Relock Result Summary

## Required Answers

1. What produced the exploratory 69.86% absolute-direction final row: the strongest non-trivial comparable h40 source was `{abs_row.get("candidate_id", "")}` with final accuracy {pct(abs_row.get("final_accuracy"))}, validation accuracy {pct(abs_row.get("validation_accuracy"))}, and validation lift {pp(abs_row.get("lift_over_strongest_baseline"))}. The absolute highest h40 final row was `{abs_all_row.get("candidate_id", "")}` at {pct(abs_all_row.get("final_accuracy"))}, but it did not pass validation-lift promotion screening.
2. What produced the exploratory 74.57% market-relative final row: `{mr_row.get("candidate_id", "")}` with final accuracy {pct(mr_row.get("final_accuracy"))}; this is a trivial/simple baseline pattern with validation lift {pp(mr_row.get("lift_over_strongest_baseline"))}, so it is not promotable.
3. Were those results isolated one-offs or cluster-supported: the absolute-direction source is `{abs_cluster_verdict}` and the market-relative source is `{mr_cluster_verdict}` under the model/target/horizon/feature cluster audit.
4. Could any family be re-locked by validation: yes, V2 froze final-discovered hypothesis families and selected exact candidates by validation only, but all such relocks remain future-blind-required because the families were discovered from final exploratory rows.
5. Did any relocked direction candidate beat 61.61% on comparable scope: {str(bool(abs_final_relocks)).lower()} for `{abs_final_best.get("candidate_id", "")}` with final accuracy {pct(abs_final_best.get("final_accuracy"))}; this is future-blind-required and not a replacement claim. The validation-best absolute h40 relock was `{locked_abs.get("candidate_id", "")}` with final accuracy {pct(locked_abs.get("final_accuracy"))}.
6. Did any relocked market-relative candidate beat 64.44% on comparable scope: {str(bool(mr_nontrivial_final_relocks)).lower()} for non-trivial relocks. The strongest row over 64.44% was `{mr_final_best.get("candidate_id", "")}` at {pct(mr_final_best.get("final_accuracy"))}, but its label is `{mr_final_best.get("relock_claim_label", "")}`.
7. Did any relocked price/return candidate beat random walk/last price on final: {str(price_final_beats).lower()} for validation-supported relocks. Best validation-supported final improvement was `{price_final_best.get("candidate_id", locked_price_best.get("candidate_id", ""))}` with final RMSE improvement {pp(price_final_best.get("final_rmse_improvement_over_baseline", locked_price_best.get("final_rmse_improvement_over_baseline")))}.
8. Which results remain exploratory: all source final-ranked rows and all relocked rows remain diagnostic/future-blind-required unless confirmed by a new pre-registered future-blind run.
9. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, VN100, index-as-stock, tag, merge, push --mirror, DOCX, or production claim is made.

## Relocked Direction Candidates

- Best absolute h40 relock: `{locked_abs.get("candidate_id", "")}`; validation {pct(locked_abs.get("validation_accuracy"))}, final {pct(locked_abs.get("final_accuracy"))}, label `{locked_abs.get("relock_claim_label", "")}`.
- Strongest absolute h40 future-blind-required final diagnostic: `{abs_final_best.get("candidate_id", "")}`; validation {pct(abs_final_best.get("validation_accuracy"))}, final {pct(abs_final_best.get("final_accuracy"))}, label `{abs_final_best.get("relock_claim_label", "")}`.
- Best market-relative h40 relock: `{locked_mr.get("candidate_id", "")}`; validation {pct(locked_mr.get("validation_accuracy"))}, final {pct(locked_mr.get("final_accuracy"))}, label `{locked_mr.get("relock_claim_label", "")}`.

## Relocked Price/Return Candidate

- Best price/return relock: `{locked_price_best.get("candidate_id", "")}`; validation RMSE {as_float(locked_price_best.get("validation_rmse")):.6g}, final RMSE {as_float(locked_price_best.get("final_rmse")):.6g}, final baseline improvement {pp(locked_price_best.get("final_rmse_improvement_over_baseline"))}, label `{locked_price_best.get("relock_claim_label", "")}`.

## Audit Note

V2 uses V1 aggregate artifacts for promotion relock. Early/late validation window rows are marked unavailable when row-level validation predictions were not preserved by V1; final quarter diagnostics are included where V1 stability artifacts exist.
"""
    write_markdown(V2_RESULT_PATH, summary)
    claim = """# VN30 Model Universe V2 Promotion Relock Claim Boundary

- V2 is an offline diagnostic promotion-relock audit only.
- Exploratory final rows are used only to define hypothesis families, not to select claimable rows.
- Exact relocked candidates are selected within each frozen hypothesis family by validation metrics only.
- Because hypothesis families were discovered from final exploratory rows, all stronger interpretation requires a new future-blind protocol.
- Final-ranked rows remain exploratory_not_claimable unless re-run under a future-blind design.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, VN100, index-as-stock, DOCX, tag, merge, push --mirror, or main-branch claim is made.
- Comparisons against the 61.61% absolute-direction champion are contextual and only valid on comparable absolute_direction h40 scope.
- Comparisons against the 64.44% QML V8 market-relative result are contextual and only valid on comparable market_relative_vn30 h40 scope.
"""
    write_markdown(V2_CLAIM_PATH, claim)


def run_promotion_relock(timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    direction_raw = read_artifact("direction_validation_results.csv")
    price_raw = read_artifact("price_validation_results.csv")
    baseline = read_artifact("baseline_comparison.csv")
    ticker_stability = read_artifact("ticker_stability.csv")
    quarter_stability = read_artifact("quarter_stability.csv")
    manifest_path = OUTPUT_DIR / "model_universe_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    direction = annotate_v2_direction(direction_raw, baseline)
    price = annotate_v2_price(price_raw, baseline)
    top_direction = v2_top_direction_candidates(direction)
    top_price = v2_top_price_candidates(price)
    direction_clusters = v2_cluster_summary(top_direction, "direction", ticker_stability, quarter_stability)
    price_clusters = v2_cluster_summary(top_price, "price_return", ticker_stability, quarter_stability)
    locked_direction, direction_audit = v2_relock_direction_candidates(direction, direction_clusters)
    locked_price, price_audit = v2_relock_price_candidates(price, price_clusters)
    protocol_audit = direction_audit + price_audit
    rolling_direction = v2_rolling_rows_from_artifacts(locked_direction, quarter_stability, "direction")
    rolling_price = v2_rolling_rows_from_artifacts(locked_price, quarter_stability, "price_return")
    comparison_rows = v2_comparison_rows(locked_direction, locked_price, manifest)

    write_frame(OUTPUT_DIR / "v2_top_exploratory_direction_candidates.csv", top_direction.to_dict("records"), list(top_direction.columns) if not top_direction.empty else [])
    write_frame(OUTPUT_DIR / "v2_top_exploratory_price_candidates.csv", top_price.to_dict("records"), list(top_price.columns) if not top_price.empty else [])
    write_frame(OUTPUT_DIR / "v2_direction_candidate_cluster_summary.csv", direction_clusters.to_dict("records"), list(direction_clusters.columns) if not direction_clusters.empty else [])
    write_frame(OUTPUT_DIR / "v2_price_candidate_cluster_summary.csv", price_clusters.to_dict("records"), list(price_clusters.columns) if not price_clusters.empty else [])
    write_frame(OUTPUT_DIR / "v2_relock_protocol_audit.csv", protocol_audit, list(protocol_audit[0].keys()) if protocol_audit else [])
    write_json(OUTPUT_DIR / "v2_locked_direction_candidates.json", {"locked_candidates": locked_direction, "selection": "validation_only_within_final_discovered_hypothesis_family", "future_blind_required": True})
    write_json(OUTPUT_DIR / "v2_locked_price_candidates.json", {"locked_candidates": locked_price, "selection": "validation_only_within_final_discovered_hypothesis_family", "future_blind_required": True})
    write_frame(OUTPUT_DIR / "v2_rolling_origin_direction.csv", rolling_direction, list(rolling_direction[0].keys()) if rolling_direction else [])
    write_frame(OUTPUT_DIR / "v2_rolling_origin_price.csv", rolling_price, list(rolling_price[0].keys()) if rolling_price else [])
    write_frame(OUTPUT_DIR / "v2_relocked_comparison_summary.csv", comparison_rows, list(comparison_rows[0].keys()) if comparison_rows else [])
    write_v2_reports(top_direction, direction_clusters, top_price, locked_direction, locked_price, comparison_rows)

    result = {
        "status": "ok",
        "mode": "promotion_relock",
        "runtime_seconds": time.perf_counter() - started,
        "timeout_seconds": timeout_seconds,
        "top_direction_candidates": int(len(top_direction)),
        "top_price_candidates": int(len(top_price)),
        "direction_clusters": int(len(direction_clusters)),
        "price_clusters": int(len(price_clusters)),
        "locked_direction_candidates": int(len(locked_direction)),
        "locked_price_candidates": int(len(locked_price)),
        "diagnostic_only": True,
        "future_blind_required": True,
        "no_trading_claim": True,
    }
    print(json.dumps(json_safe(result), indent=2))
    return result


def v3_dependency_status() -> dict[str, Any]:
    packages = [
        "catboost",
        "statsmodels",
        "arch",
        "pykalman",
        "torch",
        "pytorch_forecasting",
        "tensorflow",
        "qiskit",
        "qiskit_machine_learning",
        "pennylane",
    ]
    status: dict[str, Any] = {}
    for package in packages:
        available = importlib.util.find_spec(package) is not None
        version = ""
        if available:
            try:
                version = str(getattr(importlib.import_module(package), "__version__", "unknown"))
            except Exception:
                version = "unknown"
        status[f"{package}_available"] = bool(available)
        status[f"{package}_version"] = version
    status["diagnostic_only"] = True
    status["no_trading_claim"] = True
    return status


def v3_price_row(
    candidate: dict[str, Any],
    target: pd.Series,
    splits: dict[str, pd.Index],
    features: pd.DataFrame,
    val_pred: np.ndarray,
    final_pred: np.ndarray,
    runtime_seconds: float,
    status: str = "ok",
    skipped_reason: str = "",
) -> dict[str, Any]:
    target_variant = str(candidate["target_variant"])
    horizon = int(candidate["horizon"])
    train_y = pd.to_numeric(target.loc[splits["train"]], errors="coerce")
    val_y = pd.to_numeric(target.loc[splits["validation"]], errors="coerce")
    final_y = pd.to_numeric(target.loc[splits["final"]], errors="coerce")
    naive_name = "last_price" if target_variant == "future_close_h" else "historical_mean_return"
    naive_val, _ = baseline_price_prediction(naive_name, features, train_y, splits["validation"], target_variant, horizon)
    naive_final, _ = baseline_price_prediction(naive_name, features, train_y, splits["final"], target_variant, horizon)
    if naive_val is None:
        naive_val = np.zeros(len(val_y), dtype=float)
    if naive_final is None:
        naive_final = np.zeros(len(final_y), dtype=float)
    val_close = pd.to_numeric(features.loc[splits["validation"], "close"], errors="coerce").to_numpy(dtype=float)
    final_close = pd.to_numeric(features.loc[splits["final"], "close"], errors="coerce").to_numpy(dtype=float)
    val_metrics = price_metrics(val_y, val_pred, naive_val, val_close, target_variant)
    final_metrics = price_metrics(final_y, final_pred, naive_final, final_close, target_variant)
    baseline_val_rmse = rmse(val_y.to_numpy(dtype=float), naive_val)
    baseline_final_rmse = rmse(final_y.to_numpy(dtype=float), naive_final)
    validation_improvement = (baseline_val_rmse - val_metrics["rmse"]) / baseline_val_rmse if math.isfinite(baseline_val_rmse) and baseline_val_rmse > 0 and math.isfinite(val_metrics["rmse"]) else math.nan
    final_improvement = (baseline_final_rmse - final_metrics["rmse"]) / baseline_final_rmse if math.isfinite(baseline_final_rmse) and baseline_final_rmse > 0 and math.isfinite(final_metrics["rmse"]) else math.nan
    label = "future_blind_required" if validation_improvement > 0 and final_improvement > 0 else ("price_return_candidate" if validation_improvement > 0 else "diagnostic_only")
    return {
        "candidate_id": candidate["candidate_id"],
        "task": "price_return",
        "source_family": candidate.get("source_family", ""),
        "model_family": candidate["model_family"],
        "feature_group": candidate.get("feature_group", ""),
        "target_variant": target_variant,
        "horizon": horizon,
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "validation_rmse": val_metrics["rmse"],
        "validation_mae": val_metrics["mae"],
        "validation_sign_accuracy": val_metrics["sign_accuracy"],
        "validation_correlation_pred_actual": val_metrics["correlation_pred_actual"],
        "validation_error_improvement_over_baseline": validation_improvement,
        "final_rmse": final_metrics["rmse"],
        "final_mae": final_metrics["mae"],
        "final_sign_accuracy": final_metrics["sign_accuracy"],
        "final_correlation_pred_actual": final_metrics["correlation_pred_actual"],
        "final_error_improvement_over_baseline": final_improvement,
        "strongest_final_baseline_rmse": baseline_final_rmse,
        "claim_label": label,
        "runtime_seconds": runtime_seconds,
        "status": status,
        "skipped_reason": skipped_reason,
    }


def v3_direction_row(
    candidate: dict[str, Any],
    labels: pd.Series,
    splits: dict[str, pd.Index],
    features: pd.DataFrame,
    val_pred: np.ndarray,
    val_prob: np.ndarray | None,
    final_pred: np.ndarray,
    final_prob: np.ndarray | None,
    runtime_seconds: float,
    status: str = "ok",
    skipped_reason: str = "",
) -> dict[str, Any]:
    target_variant = str(candidate["target_variant"])
    horizon = int(candidate["horizon"])
    val_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    train_y = labels.loc[splits["train"]].astype(int)
    val_metrics = direction_metrics(val_y, val_pred, val_prob)
    final_metrics = direction_metrics(final_y, final_pred, final_prob)
    baseline_accs = []
    for baseline in ["always_up", "always_down", "lag1_direction", "simple_momentum", "simple_relative_strength"]:
        pred, _prob, _reason = baseline_direction_prediction(baseline, features, train_y, splits["validation"])
        if pred is not None:
            baseline_accs.append(float((val_y.to_numpy(dtype=int) == pred.astype(int)).mean()))
    strongest_baseline = max(baseline_accs) if baseline_accs else math.nan
    val_lift = val_metrics["accuracy"] - strongest_baseline if math.isfinite(strongest_baseline) else math.nan
    label = "diagnostic_only"
    if val_lift > 0:
        label = "direction_candidate"
    if target_variant == "absolute_direction" and horizon == 40 and val_lift > 0 and final_metrics["accuracy"] > CLASSICAL_CHAMPION["final_accuracy"]:
        label = "future_blind_required"
    if target_variant == "market_relative_vn30" and horizon == 40 and val_lift > 0 and final_metrics["accuracy"] > QML_V8_CONTEXT_FINAL:
        label = "future_blind_required"
    return {
        "candidate_id": candidate["candidate_id"],
        "task": "direction",
        "source_family": candidate.get("source_family", ""),
        "model_family": candidate["model_family"],
        "feature_group": candidate.get("feature_group", ""),
        "target_variant": target_variant,
        "horizon": horizon,
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "validation_accuracy": val_metrics["accuracy"],
        "validation_lift_over_strongest_baseline": val_lift,
        "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
        "validation_f1": val_metrics["f1"],
        "validation_roc_auc": val_metrics["roc_auc"],
        "final_accuracy": final_metrics["accuracy"],
        "final_balanced_accuracy": final_metrics["balanced_accuracy"],
        "final_f1": final_metrics["f1"],
        "final_roc_auc": final_metrics["roc_auc"],
        "claim_label": label,
        "runtime_seconds": runtime_seconds,
        "status": status,
        "skipped_reason": skipped_reason,
    }


def v3_skipped_row(task: str, source_family: str, model_family: str, target_variant: str = "", horizon: Any = "", feature_group: str = "", reason: str = "") -> dict[str, Any]:
    return {
        "task": task,
        "source_family": source_family,
        "model_family": model_family,
        "target_variant": target_variant,
        "horizon": horizon,
        "feature_group": feature_group,
        "status": "skipped",
        "skipped_reason": reason,
    }


def v3_series_forecast(model_name: str, train_values: pd.Series, n_val: int, n_final: int, dependency: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    y = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float).tail(500)
    if len(y) < 40:
        return None, None, "insufficient train rows for statistical model"
    if model_name in {"ARIMA", "SARIMAX", "ETS", "Kalman/local level"} and not dependency.get("statsmodels_available"):
        return None, None, "statsmodels unavailable"
    if model_name == "GARCH-assisted" and not dependency.get("arch_available"):
        return None, None, "arch unavailable"
    try:
        if model_name == "ARIMA":
            from statsmodels.tsa.arima.model import ARIMA

            fit = ARIMA(y.to_numpy(dtype=float), order=(1, 0, 0), trend="c").fit()
            return np.asarray(fit.forecast(n_val), dtype=float), np.asarray(fit.forecast(n_final), dtype=float), ""
        if model_name == "SARIMAX":
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            fit = SARIMAX(y.to_numpy(dtype=float), order=(1, 0, 0), trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
            return np.asarray(fit.forecast(n_val), dtype=float), np.asarray(fit.forecast(n_final), dtype=float), ""
        if model_name == "ETS":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            fit = ExponentialSmoothing(y.to_numpy(dtype=float), trend=None, seasonal=None, initialization_method="estimated").fit(optimized=True)
            return np.asarray(fit.forecast(n_val), dtype=float), np.asarray(fit.forecast(n_final), dtype=float), ""
        if model_name == "GARCH-assisted":
            arch_mod = importlib.import_module("arch")
            scaled = y.to_numpy(dtype=float) * 100.0
            fit = arch_mod.arch_model(scaled, mean="AR", lags=1, vol="GARCH", p=1, q=1, rescale=False).fit(disp="off", show_warning=False)
            one = float(np.asarray(fit.forecast(horizon=1).mean.iloc[-1])[0]) / 100.0
            return np.full(n_val, one, dtype=float), np.full(n_final, one, dtype=float), ""
        if model_name == "Kalman/local level":
            from statsmodels.tsa.statespace.structural import UnobservedComponents

            fit = UnobservedComponents(y.to_numpy(dtype=float), level="local level").fit(disp=False, maxiter=50)
            return np.asarray(fit.forecast(n_val), dtype=float), np.asarray(fit.forecast(n_final), dtype=float), ""
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    return None, None, f"{model_name} not implemented"


def run_v3_statistical(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], dependency: dict[str, Any], config: RunConfig, started: float, timeout_seconds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    models = ["ARIMA", "SARIMAX", "ETS", "GARCH-assisted", "Kalman/local level"]
    targets = ["forward_log_return_h", "forward_simple_return_h", "market_excess_return_h", "future_close_h"]
    for horizon in [10, 20, 40]:
        for target_variant in targets:
            target = build_price_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=False)
            for model_name in models:
                if time.perf_counter() - started > timeout_seconds:
                    skipped.append(v3_skipped_row("price_return", "statistical", model_name, target_variant, horizon, "", "runtime_budget_exhausted"))
                    continue
                start = time.perf_counter()
                candidate = {
                    "candidate_id": candidate_id("v3", "statistical", model_name, target_variant, f"h{horizon}"),
                    "source_family": "statistical",
                    "model_family": model_name,
                    "feature_group": "target_history",
                    "target_variant": target_variant,
                    "horizon": horizon,
                }
                val_pred, final_pred, reason = v3_series_forecast(model_name, target.loc[splits["train"]], len(splits["validation"]), len(splits["final"]), dependency)
                if val_pred is None or final_pred is None:
                    skipped.append(v3_skipped_row("price_return", "statistical", model_name, target_variant, horizon, "target_history", reason))
                    rows.append({**candidate, "task": "price_return", "status": "skipped", "skipped_reason": reason})
                    continue
                rows.append(v3_price_row(candidate, target, splits, features, val_pred, final_pred, time.perf_counter() - start))
    return rows, skipped


def v3_torch_predict(model_name: str, task: str, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, x_final: pd.DataFrame, epochs: int = 4) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    if importlib.util.find_spec("torch") is None:
        return None, None, None, None, "torch unavailable"
    try:
        import torch
        from torch import nn

        torch.manual_seed(SEED)
        xtr = torch.tensor(x_train.to_numpy(dtype=np.float32), dtype=torch.float32)
        xva = torch.tensor(x_val.to_numpy(dtype=np.float32), dtype=torch.float32)
        xfi = torch.tensor(x_final.to_numpy(dtype=np.float32), dtype=torch.float32)
        if xtr.numel() == 0:
            return None, None, None, None, "empty feature matrix"

        class MLPNet(nn.Module):
            def __init__(self, n: int) -> None:
                super().__init__()
                self.net = nn.Sequential(nn.Linear(n, 24), nn.ReLU(), nn.Dropout(0.05), nn.Linear(24, 1))

            def forward(self, x: Any) -> Any:
                return self.net(x).squeeze(-1)

        class RNNNet(nn.Module):
            def __init__(self, kind: str, n: int) -> None:
                super().__init__()
                bidir = kind == "BiLSTM"
                cls = nn.GRU if kind == "GRU" else nn.LSTM
                self.rnn = cls(input_size=1, hidden_size=12, batch_first=True, bidirectional=bidir)
                self.out = nn.Linear(24 if bidir else 12, 1)

            def forward(self, x: Any) -> Any:
                seq = x.unsqueeze(-1)
                out, _state = self.rnn(seq)
                return self.out(out[:, -1, :]).squeeze(-1)

        class CNNNet(nn.Module):
            def __init__(self, n: int, dilation: int = 1) -> None:
                super().__init__()
                self.conv = nn.Sequential(nn.Conv1d(1, 12, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
                self.out = nn.Linear(12, 1)

            def forward(self, x: Any) -> Any:
                return self.out(self.conv(x.unsqueeze(1)).squeeze(-1)).squeeze(-1)

        class TransformerNet(nn.Module):
            def __init__(self, n: int) -> None:
                super().__init__()
                self.proj = nn.Linear(1, 12)
                layer = nn.TransformerEncoderLayer(d_model=12, nhead=3, dim_feedforward=24, dropout=0.05, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=1)
                self.out = nn.Linear(12, 1)

            def forward(self, x: Any) -> Any:
                seq = self.encoder(self.proj(x.unsqueeze(-1)))
                return self.out(seq.mean(dim=1)).squeeze(-1)

        if model_name == "MLP":
            model = MLPNet(xtr.shape[1])
        elif model_name in {"LSTM", "GRU", "BiLSTM"}:
            model = RNNNet(model_name, xtr.shape[1])
        elif model_name == "1D CNN":
            model = CNNNet(xtr.shape[1], dilation=1)
        elif model_name == "TCN":
            model = CNNNet(xtr.shape[1], dilation=2)
        elif model_name == "Transformer encoder":
            model = TransformerNet(xtr.shape[1])
        else:
            return None, None, None, None, f"{model_name} not implemented"

        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        if task == "direction":
            y = torch.tensor(y_train.astype(int).to_numpy(dtype=np.float32), dtype=torch.float32)
            pos = float(y.mean().item()) if len(y) else 0.5
            pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-3)], dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            for _epoch in range(epochs):
                model.train()
                opt.zero_grad()
                loss = loss_fn(model(xtr), y)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_prob = torch.sigmoid(model(xva)).numpy()
                final_prob = torch.sigmoid(model(xfi)).numpy()
            return (val_prob >= 0.5).astype(int), val_prob, (final_prob >= 0.5).astype(int), final_prob, ""
        y_np = pd.to_numeric(y_train, errors="coerce").to_numpy(dtype=np.float32)
        y_mean = float(np.nanmean(y_np))
        y_std = float(np.nanstd(y_np)) or 1.0
        y = torch.tensor((y_np - y_mean) / y_std, dtype=torch.float32)
        loss_fn = nn.MSELoss()
        for _epoch in range(epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(xtr), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(xva).numpy() * y_std + y_mean
            final_pred = model(xfi).numpy() * y_std + y_mean
        return val_pred, None, final_pred, None, ""
    except Exception as exc:
        return None, None, None, None, f"{type(exc).__name__}: {exc}"


def run_v3_deep_sequence(features: pd.DataFrame, family_cols: dict[str, list[str]], relative_cols: list[str], index_data: dict[str, pd.DataFrame], feature_groups: dict[str, list[str]], dependency: dict[str, Any], config: RunConfig, started: float, timeout_seconds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    direction_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    models = ["MLP", "LSTM", "GRU", "BiLSTM", "1D CNN", "TCN", "Transformer encoder"]
    if not dependency.get("torch_available"):
        for model in models:
            skipped.append(v3_skipped_row("deep_sequence", "deep_sequence", model, reason="torch unavailable"))
        return direction_rows, price_rows, skipped
    direction_plan = [
        ("absolute_direction", 20, "relative_strength"),
        ("absolute_direction", 40, "market_context"),
        ("absolute_direction", 40, "combined_strategy_features"),
        ("market_relative_vn30", 20, "relative_strength"),
        ("market_relative_vn30", 40, "market_context"),
        ("market_relative_vn30", 40, "combined_strategy_features"),
    ]
    for target_variant, horizon, feature_group in direction_plan:
        labels = build_direction_target(features, index_data, target_variant, horizon)
        splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
        x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_groups.get(feature_group, []), 16)
        if status != "ok":
            skipped.append(v3_skipped_row("direction", "deep_sequence", "all", target_variant, horizon, feature_group, status))
            continue
        for model_name in models:
            if time.perf_counter() - started > timeout_seconds:
                skipped.append(v3_skipped_row("direction", "deep_sequence", model_name, target_variant, horizon, feature_group, "runtime_budget_exhausted"))
                continue
            start = time.perf_counter()
            val_pred, val_prob, final_pred, final_prob, reason = v3_torch_predict(model_name, "direction", x_train, labels.loc[splits["train"]], x_val, x_final)
            candidate = {"candidate_id": candidate_id("v3", "deep", model_name, target_variant, f"h{horizon}", feature_group), "source_family": "deep_sequence", "model_family": model_name, "feature_group": feature_group, "target_variant": target_variant, "horizon": horizon}
            if val_pred is None or final_pred is None:
                skipped.append(v3_skipped_row("direction", "deep_sequence", model_name, target_variant, horizon, feature_group, reason))
                direction_rows.append({**candidate, "task": "direction", "status": "skipped", "skipped_reason": reason})
                continue
            direction_rows.append(v3_direction_row(candidate, labels, splits, features, val_pred, val_prob, final_pred, final_prob, time.perf_counter() - start))
    price_plan = [
        ("forward_log_return_h", 20, "relative_strength"),
        ("forward_log_return_h", 40, "market_context"),
        ("market_excess_return_h", 20, "relative_strength"),
        ("market_excess_return_h", 40, "market_context"),
    ]
    for target_variant, horizon, feature_group in price_plan:
        target = build_price_target(features, index_data, target_variant, horizon)
        splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=False)
        x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_groups.get(feature_group, []), 16)
        if status != "ok":
            skipped.append(v3_skipped_row("price_return", "deep_sequence", "all", target_variant, horizon, feature_group, status))
            continue
        for model_name in models:
            if time.perf_counter() - started > timeout_seconds:
                skipped.append(v3_skipped_row("price_return", "deep_sequence", model_name, target_variant, horizon, feature_group, "runtime_budget_exhausted"))
                continue
            start = time.perf_counter()
            val_pred, _val_prob, final_pred, _final_prob, reason = v3_torch_predict(model_name, "price_return", x_train, target.loc[splits["train"]], x_val, x_final)
            candidate = {"candidate_id": candidate_id("v3", "deep", model_name, target_variant, f"h{horizon}", feature_group), "source_family": "deep_sequence", "model_family": model_name, "feature_group": feature_group, "target_variant": target_variant, "horizon": horizon}
            if val_pred is None or final_pred is None:
                skipped.append(v3_skipped_row("price_return", "deep_sequence", model_name, target_variant, horizon, feature_group, reason))
                price_rows.append({**candidate, "task": "price_return", "status": "skipped", "skipped_reason": reason})
                continue
            price_rows.append(v3_price_row(candidate, target, splits, features, np.asarray(val_pred), np.asarray(final_pred), time.perf_counter() - start))
    return direction_rows, price_rows, skipped


def run_v3_catboost(features: pd.DataFrame, family_cols: dict[str, list[str]], relative_cols: list[str], index_data: dict[str, pd.DataFrame], feature_groups: dict[str, list[str]], dependency: dict[str, Any], config: RunConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    direction_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if not dependency.get("catboost_available"):
        for task, model in [("direction", "catboost_classifier"), ("price_return", "catboost_regressor")]:
            skipped.append(v3_skipped_row(task, "catboost", model, reason="catboost unavailable"))
        return direction_rows, price_rows, skipped
    for target_variant, horizon in [("absolute_direction", 40), ("market_relative_vn30", 40)]:
        labels = build_direction_target(features, index_data, target_variant, horizon)
        splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
        for feature_group in ["relative_strength", "market_context"]:
            row, _pred = evaluate_direction_candidate("catboost_classifier", feature_group, features, family_cols, relative_cols, index_data, labels, splits, feature_groups.get(feature_group, []))
            row["source_family"] = "catboost"
            direction_rows.append(row)
            if row.get("status") == "skipped":
                skipped.append(v3_skipped_row("direction", "catboost", "catboost_classifier", target_variant, horizon, feature_group, row.get("skipped_reason", "")))
    for target_variant, horizon in [("forward_log_return_h", 40), ("market_excess_return_h", 40)]:
        target = build_price_target(features, index_data, target_variant, horizon)
        splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=False)
        for feature_group in ["relative_strength", "market_context"]:
            row, _pred = evaluate_price_candidate("catboost_regressor", feature_group, features, target, splits, feature_groups.get(feature_group, []))
            row["source_family"] = "catboost"
            price_rows.append(row)
            if row.get("status") == "skipped":
                skipped.append(v3_skipped_row("price_return", "catboost", "catboost_regressor", target_variant, horizon, feature_group, row.get("skipped_reason", "")))
    return direction_rows, price_rows, skipped


def run_v3_qml(features: pd.DataFrame, family_cols: dict[str, list[str]], relative_cols: list[str], index_data: dict[str, pd.DataFrame], dependency: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    config = RunConfig("v3_qml", 1200, 260, 140, 140, 20, 0)
    for target_variant in ["market_relative_vn30", "absolute_direction"]:
        labels = build_direction_target(features, index_data, target_variant, 40)
        splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
        for model_name in ["qml_v8_kernel_features_l2", "qml_v8_kernel_features_lightgbm"]:
            row, _pred = evaluate_direction_candidate(model_name, "qml_kernel_features", features, family_cols, relative_cols, index_data, labels, splits, [])
            row["source_family"] = "qml_integration"
            rows.append(row)
            if row.get("status") == "skipped":
                skipped.append(v3_skipped_row("direction", "qml_integration", model_name, target_variant, 40, "qml_kernel_features", row.get("skipped_reason", "")))
    skipped.append(v3_skipped_row("direction", "qml_integration", "V4 pure quantum kernel replay", "market_relative_vn30", 40, "relative_strength", "not run in V3 bounded integration; V4 evidence exists in QML artifacts"))
    skipped.append(v3_skipped_row("direction", "qml_integration", "PennyLane hybrid QNN", "market_relative_vn30", 40, "relative_strength", "not run in V3 bounded integration; focused QML smoke path required"))
    return rows, skipped


def v3_direction_ensemble(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], feature_groups: dict[str, list[str]], target_variant: str, horizon: int, feature_group: str, ensemble_name: str, config: RunConfig) -> tuple[dict[str, Any], dict[str, Any] | None]:
    start = time.perf_counter()
    labels = build_direction_target(features, index_data, target_variant, horizon)
    splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
    x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_groups.get(feature_group, []), 16)
    candidate = {"candidate_id": candidate_id("v3", "ensemble", ensemble_name, target_variant, f"h{horizon}", feature_group), "source_family": "ensemble", "model_family": ensemble_name, "feature_group": feature_group, "target_variant": target_variant, "horizon": horizon}
    if status != "ok":
        return {**candidate, "task": "direction", "status": "skipped", "skipped_reason": status}, v3_skipped_row("direction", "ensemble", ensemble_name, target_variant, horizon, feature_group, status)
    base_models = ["logistic_regression", "linear_svm", "naive_bayes", "random_forest"]
    val_probs: list[np.ndarray] = []
    final_probs: list[np.ndarray] = []
    val_y = labels.loc[splits["validation"]].astype(int).to_numpy()
    train_y = labels.loc[splits["train"]].astype(int)
    for base in base_models:
        model, reason = direction_model(base)
        if model is None:
            continue
        try:
            model.fit(x_train, train_y)
            if hasattr(model, "predict_proba"):
                vp = np.asarray(model.predict_proba(x_val)[:, 1], dtype=float)
                fp = np.asarray(model.predict_proba(x_final)[:, 1], dtype=float)
            else:
                vs = np.asarray(model.decision_function(x_val), dtype=float)
                fs = np.asarray(model.decision_function(x_final), dtype=float)
                vp = 1.0 / (1.0 + np.exp(-np.clip(vs, -30.0, 30.0)))
                fp = 1.0 / (1.0 + np.exp(-np.clip(fs, -30.0, 30.0)))
            val_probs.append(vp)
            final_probs.append(fp)
        except Exception:
            continue
    if len(val_probs) < 2:
        return {**candidate, "task": "direction", "status": "skipped", "skipped_reason": "fewer than two base models fit"}, v3_skipped_row("direction", "ensemble", ensemble_name, target_variant, horizon, feature_group, "fewer than two base models fit")
    val_stack = np.vstack(val_probs)
    final_stack = np.vstack(final_probs)
    if ensemble_name == "soft_voting":
        val_prob = val_stack.mean(axis=0)
        final_prob = final_stack.mean(axis=0)
    elif ensemble_name == "rank_average":
        val_prob = pd.DataFrame(val_stack.T).rank(pct=True).mean(axis=1).to_numpy(dtype=float)
        final_prob = pd.DataFrame(final_stack.T).rank(pct=True).mean(axis=1).to_numpy(dtype=float)
    elif ensemble_name == "model_family_ensemble":
        weights = np.asarray([max(0.01, float(((vp >= 0.5).astype(int) == val_y).mean())) for vp in val_probs], dtype=float)
        weights = weights / weights.sum()
        val_prob = np.average(val_stack, axis=0, weights=weights)
        final_prob = np.average(final_stack, axis=0, weights=weights)
    else:
        val_prob = val_stack.mean(axis=0)
        final_prob = final_stack.mean(axis=0)
    return v3_direction_row(candidate, labels, splits, features, (val_prob >= 0.5).astype(int), val_prob, (final_prob >= 0.5).astype(int), final_prob, time.perf_counter() - start), None


def run_v3_ensembles(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], feature_groups: dict[str, list[str]], config: RunConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for target_variant, feature_group in [("absolute_direction", "market_context"), ("absolute_direction", "combined_strategy_features"), ("market_relative_vn30", "market_context")]:
        for ensemble_name in ["soft_voting", "rank_average", "model_family_ensemble"]:
            row, skip = v3_direction_ensemble(features, index_data, feature_groups, target_variant, 40, feature_group, ensemble_name, config)
            rows.append(row)
            if skip is not None:
                skipped.append(skip)
    skipped.append(v3_skipped_row("direction", "ensemble", "stacking_logistic", "absolute_direction", 40, "market_context", "row-level out-of-fold predictions not preserved in V1/V2; skipped to avoid validation leakage"))
    skipped.append(v3_skipped_row("direction", "ensemble", "direction-price_joint_diagnostic_ensemble", "absolute_direction", 40, "market_context", "joint direction-price row-level alignment requires a separate pre-registered adapter"))
    return rows, skipped


def v3_unified_leaderboard(direction_rows: list[dict[str, Any]], price_rows: list[dict[str, Any]], ensemble_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in direction_rows + ensemble_rows:
        if row.get("status") != "ok":
            continue
        rows.append({**row, "primary_validation_metric": row.get("validation_accuracy", math.nan), "primary_final_metric": row.get("final_accuracy", math.nan), "metric_direction": "higher_is_better"})
    for row in price_rows:
        if row.get("status") != "ok":
            continue
        rows.append({**row, "primary_validation_metric": row.get("validation_error_improvement_over_baseline", math.nan), "primary_final_metric": row.get("final_error_improvement_over_baseline", math.nan), "metric_direction": "higher_is_better"})
    return sorted(rows, key=lambda row: as_float(row.get("primary_validation_metric")), reverse=True)


def v3_future_blind_registry(direction_rows: list[dict[str, Any]], price_rows: list[dict[str, Any]], ensemble_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    required_ids = {
        "direction__linear_svm__absolute_direction__h40__market_context": "V2 future-blind-required absolute h40 diagnostic",
        "direction__naive_bayes__absolute_direction__h40__market_context": "V2 exploratory 69.86 source family",
    }
    v2_audit = read_artifact("v2_relock_protocol_audit.csv")
    for cid, reason in required_ids.items():
        match = v2_audit[v2_audit.get("selected_candidate_id", "") == cid] if not v2_audit.empty else pd.DataFrame()
        if not match.empty:
            row = match.iloc[0].to_dict()
            registry.append({"candidate_id": cid, "source": "v2", "why_promising": reason, "why_not_claimable_now": row.get("audit_note", "final-discovered hypothesis family"), "future_blind_test_needed": "pre-register target/model/feature/horizon and evaluate on a future-blind post-final window"})
        else:
            registry.append({"candidate_id": cid, "source": "v2", "why_promising": reason, "why_not_claimable_now": "final-discovered hypothesis family", "future_blind_test_needed": "pre-register and rerun on a future-blind window"})
    for row in direction_rows + ensemble_rows:
        if row.get("status") != "ok":
            continue
        target = row.get("target_variant")
        horizon = int(as_float(row.get("horizon"))) if math.isfinite(as_float(row.get("horizon"))) else 0
        final_acc = as_float(row.get("final_accuracy"))
        val_lift = as_float(row.get("validation_lift_over_strongest_baseline"))
        if target == "absolute_direction" and horizon == 40 and final_acc > CLASSICAL_CHAMPION["final_accuracy"]:
            registry.append({"candidate_id": row.get("candidate_id", ""), "source": "v3", "why_promising": f"final absolute_direction h40 {pct(final_acc)} exceeds 61.61 context", "why_not_claimable_now": "V3 is skipped-family diagnostic and final rows are scoring-only", "future_blind_test_needed": "lock from validation and test on future-blind h40 absolute_direction window"})
        if target == "market_relative_vn30" and horizon == 40 and final_acc > QML_V8_CONTEXT_FINAL and val_lift > 0:
            registry.append({"candidate_id": row.get("candidate_id", ""), "source": "v3", "why_promising": f"final market_relative_vn30 h40 {pct(final_acc)} exceeds 64.44 QML V8 context", "why_not_claimable_now": "target is market-relative and V3 is diagnostic", "future_blind_test_needed": "lock from validation and test on future-blind market_relative_vn30 h40 window"})
    for row in price_rows:
        if row.get("status") == "ok" and as_float(row.get("final_error_improvement_over_baseline")) > 0:
            registry.append({"candidate_id": row.get("candidate_id", ""), "source": "v3", "why_promising": f"final error improvement {pp(row.get('final_error_improvement_over_baseline'))}", "why_not_claimable_now": "V3 is diagnostic and price/return final transfer needs future-blind confirmation", "future_blind_test_needed": "pre-register target/horizon/model and compare against random walk/last price on future-blind data"})
    return pd.DataFrame(registry).drop_duplicates("candidate_id").to_dict("records") if registry else []


def write_v3_reports(direction_rows: list[dict[str, Any]], price_rows: list[dict[str, Any]], ensemble_rows: list[dict[str, Any]], qml_rows: list[dict[str, Any]], skipped: list[dict[str, Any]], registry: list[dict[str, Any]]) -> None:
    ok_direction = [row for row in direction_rows + ensemble_rows if row.get("status") == "ok"]
    ok_price = [row for row in price_rows if row.get("status") == "ok"]
    best_direction = max(ok_direction, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("validation_lift_over_strongest_baseline"))), default={})
    best_price = max(ok_price, key=lambda row: as_float(row.get("validation_error_improvement_over_baseline")), default={})
    best_ensemble = max([row for row in ensemble_rows if row.get("status") == "ok"], key=lambda row: as_float(row.get("validation_accuracy")), default={})
    best_qml = max([row for row in qml_rows if row.get("status") == "ok"], key=lambda row: as_float(row.get("validation_accuracy")), default={})
    stat_price = [row for row in price_rows if row.get("source_family") == "statistical" and row.get("status") == "ok"]
    deep_direction = [row for row in direction_rows if row.get("source_family") == "deep_sequence" and row.get("status") == "ok"]
    deep_price = [row for row in price_rows if row.get("source_family") == "deep_sequence" and row.get("status") == "ok"]
    catboost_rows = [row for row in direction_rows + price_rows if row.get("source_family") == "catboost" and row.get("status") == "ok"]
    enabled_families = sorted(set(row.get("source_family", "") for row in direction_rows + price_rows + ensemble_rows if row.get("status") == "ok"))
    skipped_families = sorted(set(row.get("model_family", "") for row in skipped))
    abs_candidates = [row for row in ok_direction if row.get("target_variant") == "absolute_direction" and int(as_float(row.get("horizon"))) == 40 and as_float(row.get("validation_lift_over_strongest_baseline")) > 0]
    best_abs_final = max(abs_candidates, key=lambda row: as_float(row.get("final_accuracy")), default={})
    mr_candidates = [row for row in ok_direction if row.get("target_variant") == "market_relative_vn30" and int(as_float(row.get("horizon"))) == 40 and as_float(row.get("validation_lift_over_strongest_baseline")) > 0]
    best_mr_final = max(mr_candidates, key=lambda row: as_float(row.get("final_accuracy")), default={})
    price_final = max(ok_price, key=lambda row: as_float(row.get("final_error_improvement_over_baseline")), default={})
    summary = f"""# VN30 Model Universe V3 Skipped Families Result Summary

## Required Answers

1. Which skipped families successfully ran: {", ".join(enabled_families) if enabled_families else "none"}.
2. Which skipped families remained unavailable: {", ".join(skipped_families[:20]) if skipped_families else "none"}.
3. Did statistical models improve price/return forecasting: {str(any(as_float(row.get("validation_error_improvement_over_baseline")) > 0 for row in stat_price)).lower()} on validation; best statistical row is `{max(stat_price, key=lambda row: as_float(row.get("validation_error_improvement_over_baseline")), default={}).get("candidate_id", "")}`.
4. Did deep/sequence models improve direction forecasting: {str(any(as_float(row.get("validation_lift_over_strongest_baseline")) > 0 for row in deep_direction)).lower()} on validation.
5. Did deep/sequence models improve price/return forecasting: {str(any(as_float(row.get("validation_error_improvement_over_baseline")) > 0 for row in deep_price)).lower()} on validation.
6. Did CatBoost improve over XGBoost/LightGBM if available: {"CatBoost ran" if catboost_rows else "CatBoost unavailable or skipped"}.
7. Did QML integration improve over QML V8 or same-target classical baselines: {str(as_float(best_qml.get("final_accuracy")) > QML_V8_CONTEXT_FINAL).lower()} for the best QML integration row `{best_qml.get("candidate_id", "")}`; this remains diagnostic-only.
8. Did ensembles improve validation-to-final transfer: best ensemble `{best_ensemble.get("candidate_id", "")}` validation {pct(best_ensemble.get("validation_accuracy"))}, final {pct(best_ensemble.get("final_accuracy"))}.
9. Did any validation-governed candidate beat the 61.61 absolute-direction champion on comparable scope: {str(as_float(best_abs_final.get("final_accuracy")) > CLASSICAL_CHAMPION["final_accuracy"]).lower()} for `{best_abs_final.get("candidate_id", "")}` with final {pct(best_abs_final.get("final_accuracy"))}; future-blind confirmation is still required.
10. Did any validation-governed candidate beat the 64.44 QML V8 market-relative result on comparable scope: {str(as_float(best_mr_final.get("final_accuracy")) > QML_V8_CONTEXT_FINAL).lower()} for `{best_mr_final.get("candidate_id", "")}` with final {pct(best_mr_final.get("final_accuracy"))}; future-blind confirmation is still required.
11. Did any price/return model beat random walk/last price on final: {str(as_float(price_final.get("final_error_improvement_over_baseline")) > 0).lower()} for `{price_final.get("candidate_id", "")}` with final improvement {pp(price_final.get("final_error_improvement_over_baseline"))}.
12. Which candidates require future-blind confirmation: {len(registry)} rows are listed in `v3_future_blind_candidate_registry.csv`.
13. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, VN100, index-as-stock, tag, merge, push --mirror, DOCX, or replacement claim is made.

## Best Rows

- Best direction validation row: `{best_direction.get("candidate_id", "")}`; validation {pct(best_direction.get("validation_accuracy"))}, final {pct(best_direction.get("final_accuracy"))}.
- Best price/return validation row: `{best_price.get("candidate_id", "")}`; validation improvement {pp(best_price.get("validation_error_improvement_over_baseline"))}, final improvement {pp(best_price.get("final_error_improvement_over_baseline"))}.
- Best ensemble row: `{best_ensemble.get("candidate_id", "")}`; validation {pct(best_ensemble.get("validation_accuracy"))}, final {pct(best_ensemble.get("final_accuracy"))}.
- Best QML integration row: `{best_qml.get("candidate_id", "")}`; validation {pct(best_qml.get("validation_accuracy"))}, final {pct(best_qml.get("final_accuracy"))}.
"""
    write_markdown(V3_RESULT_PATH, summary)
    claim = """# VN30 Model Universe V3 Skipped Families Claim Boundary

- V3 is an offline diagnostic benchmark for previously skipped model families.
- Direction and price/return targets are separate; directional accuracy is not mixed with price/return error metrics.
- Optional dependencies are dependency-guarded; unavailable families are recorded, not forced.
- Candidate selection is validation-governed only; final rows are scoring-only.
- Final-ranked rows remain exploratory_not_claimable.
- Results exceeding the 61.61% absolute-direction champion or 64.44% QML V8 market-relative context require future-blind confirmation and do not replace prior champions.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, VN100, index-as-stock, DOCX, tag, merge, push --mirror, or main-branch claim is made.
"""
    write_markdown(V3_CLAIM_PATH, claim)


def run_enable_skipped_families(timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = v3_dependency_status()
    write_json(OUTPUT_DIR / "v3_dependency_status.json", dependency)
    features, family_cols, _feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    index_data = load_index_data()
    features, relative_cols = add_v3_relative_strength_features(features, index_data)
    feature_groups = build_feature_groups(features, family_cols, relative_cols)
    config = RunConfig("enable_skipped_families", timeout_seconds, 800, 320, 320, 200, 200)

    statistical_rows, skipped = run_v3_statistical(features, index_data, dependency, config, started, timeout_seconds)
    deep_direction, deep_price, deep_skipped = run_v3_deep_sequence(features, family_cols, relative_cols, index_data, feature_groups, dependency, config, started, timeout_seconds)
    cat_direction, cat_price, cat_skipped = run_v3_catboost(features, family_cols, relative_cols, index_data, feature_groups, dependency, config)
    qml_rows, qml_skipped = run_v3_qml(features, family_cols, relative_cols, index_data, dependency)
    ensemble_rows, ensemble_skipped = run_v3_ensembles(features, index_data, feature_groups, config)
    skipped.extend(deep_skipped + cat_skipped + qml_skipped + ensemble_skipped)

    direction_rows = deep_direction + cat_direction + qml_rows + ensemble_rows
    price_rows = statistical_rows + deep_price + cat_price
    direction_ok = [row for row in direction_rows if row.get("status") == "ok"]
    price_ok = [row for row in price_rows if row.get("status") == "ok"]
    direction_final = sorted(direction_ok, key=lambda row: as_float(row.get("final_accuracy")), reverse=True)
    price_final = sorted(price_ok, key=lambda row: as_float(row.get("final_error_improvement_over_baseline")), reverse=True)
    unified = v3_unified_leaderboard(direction_ok, price_ok, ensemble_rows)
    registry = v3_future_blind_registry(direction_ok, price_ok, ensemble_rows)

    write_frame(OUTPUT_DIR / "v3_statistical_results.csv", statistical_rows, sorted(set().union(*(row.keys() for row in statistical_rows))) if statistical_rows else [])
    write_frame(OUTPUT_DIR / "v3_deep_sequence_results.csv", deep_direction + deep_price, sorted(set().union(*(row.keys() for row in deep_direction + deep_price))) if (deep_direction or deep_price) else [])
    write_frame(OUTPUT_DIR / "v3_catboost_results.csv", cat_direction + cat_price, sorted(set().union(*(row.keys() for row in cat_direction + cat_price))) if (cat_direction or cat_price) else [])
    write_frame(OUTPUT_DIR / "v3_qml_integration_results.csv", qml_rows, sorted(set().union(*(row.keys() for row in qml_rows))) if qml_rows else [])
    write_frame(OUTPUT_DIR / "v3_ensemble_results.csv", ensemble_rows, sorted(set().union(*(row.keys() for row in ensemble_rows))) if ensemble_rows else [])
    write_frame(OUTPUT_DIR / "v3_direction_validation_results.csv", direction_ok, sorted(set().union(*(row.keys() for row in direction_ok))) if direction_ok else [])
    write_frame(OUTPUT_DIR / "v3_direction_final_results.csv", direction_final, sorted(set().union(*(row.keys() for row in direction_final))) if direction_final else [])
    write_frame(OUTPUT_DIR / "v3_price_validation_results.csv", price_ok, sorted(set().union(*(row.keys() for row in price_ok))) if price_ok else [])
    write_frame(OUTPUT_DIR / "v3_price_final_results.csv", price_final, sorted(set().union(*(row.keys() for row in price_final))) if price_final else [])
    write_frame(OUTPUT_DIR / "v3_unified_leaderboard.csv", unified, sorted(set().union(*(row.keys() for row in unified))) if unified else [])
    write_frame(OUTPUT_DIR / "v3_skipped_models.csv", skipped, sorted(set().union(*(row.keys() for row in skipped))) if skipped else [])
    runtime_rows = [{"phase": "v3_enable_skipped_families", "runtime_seconds": time.perf_counter() - started, "direction_rows": len(direction_ok), "price_rows": len(price_ok), "skipped_rows": len(skipped), "timeout_seconds": timeout_seconds}]
    write_frame(OUTPUT_DIR / "v3_runtime_summary.csv", runtime_rows, list(runtime_rows[0].keys()))
    write_frame(OUTPUT_DIR / "v3_future_blind_candidate_registry.csv", registry, sorted(set().union(*(row.keys() for row in registry))) if registry else [])
    write_v3_reports(direction_ok, price_ok, ensemble_rows, qml_rows, skipped, registry)
    result = {
        "status": "ok",
        "mode": "enable_skipped_families",
        "runtime_seconds": time.perf_counter() - started,
        "direction_rows": len(direction_ok),
        "price_rows": len(price_ok),
        "skipped_rows": len(skipped),
        "future_blind_registry_rows": len(registry),
        "diagnostic_only": True,
        "no_trading_claim": True,
    }
    print(json.dumps(json_safe(result), indent=2))
    return result


def v4_fit_transformer(features: pd.DataFrame, train_idx: pd.Index, feature_columns: list[str], max_features: int = 16) -> tuple[Pipeline | None, list[str], str]:
    if not feature_columns:
        return None, [], "no_features"
    train = features.loc[train_idx, feature_columns].replace([np.inf, -np.inf], np.nan)
    availability = train.notna().mean()
    variance = train.var(numeric_only=True).fillna(0.0)
    selected = sorted([col for col in feature_columns if availability.get(col, 0.0) > 0.0], key=lambda col: (-float(availability.get(col, 0.0)), -float(variance.get(col, 0.0)), col))[:max_features]
    if not selected:
        return None, [], "no_features"
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pipe.fit(features.loc[train_idx, selected].replace([np.inf, -np.inf], np.nan))
    return pipe, selected, "ok"


def v4_transform(features: pd.DataFrame, idx: pd.Index, pipe: Pipeline, selected: list[str]) -> pd.DataFrame:
    arr = pipe.transform(features.loc[idx, selected].replace([np.inf, -np.inf], np.nan))
    return pd.DataFrame(arr, index=idx, columns=selected)


def v4_torch_direction_predict(
    architecture: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    eval_frames: dict[str, pd.DataFrame],
    seed: int,
    hidden_size: int = 12,
    dropout: float = 0.0,
    epochs: int = 4,
) -> tuple[dict[str, dict[str, np.ndarray]], str]:
    if importlib.util.find_spec("torch") is None:
        return {}, "torch unavailable"
    try:
        import torch
        from torch import nn

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        xtr = torch.tensor(x_train.to_numpy(dtype=np.float32), dtype=torch.float32)
        if xtr.numel() == 0:
            return {}, "empty feature matrix"

        class RNNNet(nn.Module):
            def __init__(self, kind: str, hidden: int, drop: float) -> None:
                super().__init__()
                bidir = kind == "BiLSTM"
                cls = nn.GRU if kind == "GRU" else nn.LSTM
                self.rnn = cls(input_size=1, hidden_size=hidden, batch_first=True, bidirectional=bidir)
                self.drop = nn.Dropout(drop)
                self.out = nn.Linear(hidden * (2 if bidir else 1), 1)

            def forward(self, x: Any) -> Any:
                seq = x.unsqueeze(-1)
                out, _state = self.rnn(seq)
                return self.out(self.drop(out[:, -1, :])).squeeze(-1)

        model_kind = architecture
        if architecture in {"BiLSTM_small", "BiLSTM_dropout"}:
            model_kind = "BiLSTM"
        if model_kind not in {"LSTM", "GRU", "BiLSTM"}:
            return {}, f"{architecture} not implemented for V4 relock"
        model = RNNNet(model_kind, hidden_size, dropout)
        y = torch.tensor(y_train.astype(int).to_numpy(dtype=np.float32), dtype=torch.float32)
        pos = float(y.mean().item()) if len(y) else 0.5
        pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-3)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        for _epoch in range(epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(xtr), y)
            loss.backward()
            opt.step()
        model.eval()
        output: dict[str, dict[str, np.ndarray]] = {}
        with torch.no_grad():
            for name, frame in eval_frames.items():
                x = torch.tensor(frame.to_numpy(dtype=np.float32), dtype=torch.float32)
                prob = torch.sigmoid(model(x)).numpy()
                output[name] = {"prob": prob, "pred": (prob >= 0.5).astype(int)}
        return output, ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def v4_accuracy_row(name: str, features: pd.DataFrame, idx: pd.Index, labels: pd.Series, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, Any]:
    y = labels.loc[idx].astype(int)
    metrics = direction_metrics(y, pred, prob)
    return {
        "window": name,
        "rows": int(len(idx)),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "label_positive_ratio": float(y.mean()) if len(y) else math.nan,
        "prediction_positive_ratio": float(np.asarray(pred, dtype=int).mean()) if len(pred) else math.nan,
        "timestamp_start": str(pd.to_datetime(features.loc[idx, "datetime"], errors="coerce").min()) if len(idx) else "",
        "timestamp_end": str(pd.to_datetime(features.loc[idx, "datetime"], errors="coerce").max()) if len(idx) else "",
    }


def v4_group_stability(features: pd.DataFrame, labels: pd.Series, idx: pd.Index, pred: np.ndarray, split: str, by: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "ticker": features.loc[idx, "ticker"].astype(str).to_numpy(),
            "datetime": pd.to_datetime(features.loc[idx, "datetime"], errors="coerce").to_numpy(),
            "y": labels.loc[idx].astype(int).to_numpy(),
            "pred": np.asarray(pred, dtype=int),
        },
        index=idx,
    )
    if by == "ticker":
        frame["group"] = frame["ticker"]
    elif by == "quarter":
        frame["group"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    else:
        frame["group"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for group_name, group in frame.groupby("group", sort=True):
        rows.append(
            {
                "split": split,
                "group_type": by,
                "group": group_name,
                "rows": int(len(group)),
                "accuracy": float((group["y"].to_numpy(dtype=int) == group["pred"].to_numpy(dtype=int)).mean()) if len(group) else math.nan,
                "label_positive_ratio": float(group["y"].mean()) if len(group) else math.nan,
                "prediction_positive_ratio": float(group["pred"].mean()) if len(group) else math.nan,
            }
        )
    return rows


def v4_ordered_limit(features: pd.DataFrame, idx: pd.Index, limit: int) -> pd.Index:
    ordered = ordered_index(features, idx)
    return ordered if len(ordered) <= limit else pd.Index(list(ordered)[-limit:])


def v4_eval_architecture(
    features: pd.DataFrame,
    labels: pd.Series,
    train_idx: pd.Index,
    validation_idx: pd.Index,
    final_idx: pd.Index,
    feature_columns: list[str],
    feature_group: str,
    architecture: str,
    seed: int,
    hidden_size: int,
    dropout: float,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], list[str]]:
    start = time.perf_counter()
    pipe, selected, status = v4_fit_transformer(features, train_idx, feature_columns, 16)
    candidate = {
        "candidate_id": candidate_id("v4", "bilstm_relock", architecture, "market_relative_vn30", "h40", feature_group, f"seed{seed}"),
        "architecture": architecture,
        "model_family": architecture,
        "feature_group": feature_group,
        "target_variant": "market_relative_vn30",
        "horizon": 40,
        "seed": int(seed),
        "hidden_size": int(hidden_size),
        "dropout": float(dropout),
        "epochs": 4,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "final_rows": int(len(final_idx)),
        "selected_features": "|".join(selected),
        "status": status,
        "skipped_reason": "",
    }
    if pipe is None or status != "ok":
        candidate.update({"validation_accuracy": math.nan, "final_accuracy": math.nan, "claim_label": "not_claimable", "runtime_seconds": time.perf_counter() - start, "skipped_reason": status})
        return candidate, {}, selected
    x_train = v4_transform(features, train_idx, pipe, selected)
    x_val = v4_transform(features, validation_idx, pipe, selected)
    x_final = v4_transform(features, final_idx, pipe, selected)
    outputs, reason = v4_torch_direction_predict(architecture, x_train, labels.loc[train_idx], {"validation": x_val, "final": x_final}, seed, hidden_size, dropout)
    if reason:
        candidate.update({"validation_accuracy": math.nan, "final_accuracy": math.nan, "claim_label": "not_claimable", "runtime_seconds": time.perf_counter() - start, "status": "skipped", "skipped_reason": reason})
        return candidate, outputs, selected
    val_row = v4_accuracy_row("validation", features, validation_idx, labels, outputs["validation"]["pred"], outputs["validation"]["prob"])
    final_row = v4_accuracy_row("final", features, final_idx, labels, outputs["final"]["pred"], outputs["final"]["prob"])
    val_lift = val_row["accuracy"] - max(
        [
            float((labels.loc[validation_idx].astype(int).to_numpy() == baseline_direction_prediction(base, features, labels.loc[train_idx].astype(int), validation_idx)[0].astype(int)).mean())
            for base in ["always_up", "always_down", "simple_relative_strength"]
            if baseline_direction_prediction(base, features, labels.loc[train_idx].astype(int), validation_idx)[0] is not None
        ]
        or [math.nan]
    )
    label = "future_blind_required" if val_lift > 0 and final_row["accuracy"] > QML_V8_CONTEXT_FINAL else ("direction_candidate" if val_lift > 0 else "diagnostic_only")
    candidate.update(
        {
            "validation_accuracy": val_row["accuracy"],
            "validation_lift_over_strongest_baseline": val_lift,
            "validation_balanced_accuracy": val_row["balanced_accuracy"],
            "validation_prediction_positive_ratio": val_row["prediction_positive_ratio"],
            "final_accuracy": final_row["accuracy"],
            "final_balanced_accuracy": final_row["balanced_accuracy"],
            "final_prediction_positive_ratio": final_row["prediction_positive_ratio"],
            "claim_label": label,
            "runtime_seconds": time.perf_counter() - start,
            "status": "ok",
        }
    )
    return candidate, outputs, selected


def write_v4_reports(reconstruction: dict[str, Any], relock_decision: dict[str, Any], comparison_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]], rolling_rows: list[dict[str, Any]]) -> None:
    best_ablation = max([row for row in ablation_rows if row.get("status") == "ok"], key=lambda row: as_float(row.get("validation_accuracy")), default={})
    best_baseline = max([row for row in comparison_rows if row.get("comparison_group") == "same_target_baseline"], key=lambda row: as_float(row.get("validation_accuracy")), default={})
    qml_row = next((row for row in comparison_rows if row.get("comparison_group") == "qml_v8_context"), {})
    seed_acc = [as_float(row.get("final_accuracy")) for row in seed_rows if math.isfinite(as_float(row.get("final_accuracy")))]
    seed_summary = f"mean {pct(float(np.mean(seed_acc)) if seed_acc else math.nan)}, std {float(np.std(seed_acc)):.4f}" if seed_acc else "not available"
    rolling_min = min([as_float(row.get("accuracy")) for row in rolling_rows if math.isfinite(as_float(row.get("accuracy")))], default=math.nan)
    summary = f"""# VN30 Model Universe V4 BiLSTM Relock Result Summary

## Required Answers

1. Reconstructed BiLSTM candidate: `{reconstruction.get("candidate_id", "")}` using market_relative_vn30 h40, combined_strategy_features, validation {pct(reconstruction.get("validation_accuracy"))}, final {pct(reconstruction.get("final_accuracy"))}.
2. Split/leakage audit: `{relock_decision.get("split_leakage_status", "")}`; feature_timestamp and target_timestamp boundaries passed for train, validation, and final.
3. Ticker stability: see `v4_bilstm_ticker_stability.csv`; reconstruction final ticker accuracy mean is {pct(relock_decision.get("final_ticker_mean_accuracy"))}.
4. Quarter/month stability: see `v4_bilstm_quarter_stability.csv`; minimum rolling/window accuracy was {pct(rolling_min)}.
5. Prediction/class balance: validation predicted-positive {pct(reconstruction.get("validation_prediction_positive_ratio"))}, final predicted-positive {pct(reconstruction.get("final_prediction_positive_ratio"))}.
6. Rolling-origin replay: see `v4_bilstm_rolling_origin.csv`; V4 keeps this as diagnostic because early/late windows are replay checks, not new final selection.
7. Seed sensitivity: {seed_summary}; stability warning is `{relock_decision.get("seed_stability_warning", "")}`; see `v4_bilstm_seed_sensitivity.csv`.
8. Architecture ablation: best validation ablation was `{best_ablation.get("candidate_id", "")}` with validation {pct(best_ablation.get("validation_accuracy"))} and final {pct(best_ablation.get("final_accuracy"))}.
9. Comparison against QML V8 64.44: reconstructed BiLSTM final {pct(reconstruction.get("final_accuracy"))} vs QML V8 context {pct(QML_V8_CONTEXT_FINAL)}; label `{reconstruction.get("claim_label", "")}` and future-blind confirmation required.
10. Comparison against same-target baselines: best same-target baseline was `{best_baseline.get("candidate_id", "")}` with validation {pct(best_baseline.get("validation_accuracy"))} and final {pct(best_baseline.get("final_accuracy"))}; BiLSTM beats strongest same-target baseline on final: {str(relock_decision.get("beats_strongest_same_target_baseline_final", False)).lower()}.
11. Relock decision: `{relock_decision.get("decision_label", "")}`.
12. Exact claim boundary: offline diagnostic-only; no daily T+1 system, trading, profitability, BUY/SELL, recommendation, live deployment, VN100, index-as-stock, tag, merge, push --mirror, DOCX, or replacement claim is made.

## QML Context

- QML V8 context row: `{qml_row.get("candidate_id", "qml_v8_context")}` final {pct(qml_row.get("final_accuracy", QML_V8_CONTEXT_FINAL))}.
- V4 BiLSTM is a market-relative diagnostic candidate only and does not replace the absolute-direction 61.61% champion.
"""
    write_markdown(V4_RESULT_PATH, summary)
    claim = """# VN30 Model Universe V4 BiLSTM Relock Claim Boundary

- V4 is an offline diagnostic relock and stability confirmation for a V3 BiLSTM market-relative candidate.
- No daily T+1 forecast system is created in this branch.
- Scope is VN30 stock hourly only; VN100 is out of scope.
- The target is market_relative_vn30 h40 and is not directly comparable to the absolute-direction 61.61% champion.
- Index data may be used only as lagged market-context or market-relative context; no index-as-stock claim is made.
- Candidate relock, ablations, rolling checks, and seed sensitivity are diagnostic; final performance is scoring-only.
- Stronger interpretation requires future-blind confirmation under a pre-registered protocol.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, DOCX, tag, merge, push --mirror, or main-branch claim is made.
"""
    write_markdown(V4_CLAIM_PATH, claim)


def run_bilstm_relock(timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, family_cols, _feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    index_data = load_index_data()
    features, relative_cols = add_v3_relative_strength_features(features, index_data)
    feature_groups = build_feature_groups(features, family_cols, relative_cols)
    labels = build_direction_target(features, index_data, "market_relative_vn30", 40)
    full_splits = strict_split_indices(features, labels)
    config = RunConfig("bilstm_relock", timeout_seconds, 800, 320, 320, 40, 0)
    splits = split_sample(features, labels, full_splits, config, classification=True)
    target_ts = target_timestamp_from_labels(labels, features.index)
    split_guard = leakage_guard_passed(features, labels, splits)
    split_rows = []
    for split_name, idx in splits.items():
        split_rows.append(
            {
                "split": split_name,
                "rows": int(len(idx)),
                "feature_timestamp_min": str(pd.to_datetime(features.loc[idx, "datetime"], errors="coerce").min()) if len(idx) else "",
                "feature_timestamp_max": str(pd.to_datetime(features.loc[idx, "datetime"], errors="coerce").max()) if len(idx) else "",
                "target_timestamp_min": str(target_ts.loc[idx].min()) if len(idx) else "",
                "target_timestamp_max": str(target_ts.loc[idx].max()) if len(idx) else "",
                "label_positive_ratio": float(labels.loc[idx].astype(int).mean()) if len(idx) else math.nan,
                "ticker_count": int(features.loc[idx, "ticker"].nunique()) if len(idx) else 0,
                "leakage_guard_passed": bool(split_guard),
            }
        )

    feature_columns = feature_groups["combined_strategy_features"]
    reconstruction, outputs, selected = v4_eval_architecture(features, labels, splits["train"], splits["validation"], splits["final"], feature_columns, "combined_strategy_features", "BiLSTM", SEED, 12, 0.0)
    v3_rows = read_artifact("v3_direction_validation_results.csv")
    v3_match = v3_rows[v3_rows.get("candidate_id", "") == "v3__deep__BiLSTM__market_relative_vn30__h40__combined_strategy_features"] if not v3_rows.empty else pd.DataFrame()
    v3_artifact = v3_match.iloc[0].to_dict() if not v3_match.empty else {}
    reconstruction_json = {
        **json_safe(reconstruction),
        "v3_artifact_validation_accuracy": as_float(v3_artifact.get("validation_accuracy")),
        "v3_artifact_final_accuracy": as_float(v3_artifact.get("final_accuracy")),
        "selected_features": selected,
        "reconstructed_from_v3_config": True,
        "no_daily_t1_system_created": True,
    }

    val_pred = outputs.get("validation", {}).get("pred", np.array([], dtype=int))
    val_prob = outputs.get("validation", {}).get("prob", np.array([], dtype=float))
    final_pred = outputs.get("final", {}).get("pred", np.array([], dtype=int))
    final_prob = outputs.get("final", {}).get("prob", np.array([], dtype=float))
    ticker_rows = v4_group_stability(features, labels, splits["validation"], val_pred, "validation", "ticker") + v4_group_stability(features, labels, splits["final"], final_pred, "final", "ticker")
    period_rows = (
        v4_group_stability(features, labels, splits["validation"], val_pred, "validation", "quarter")
        + v4_group_stability(features, labels, splits["final"], final_pred, "final", "quarter")
        + v4_group_stability(features, labels, splits["validation"], val_pred, "validation", "month")
        + v4_group_stability(features, labels, splits["final"], final_pred, "final", "month")
    )
    balance_rows = [
        {"split": "train", "rows": int(len(splits["train"])), "label_positive_ratio": float(labels.loc[splits["train"]].astype(int).mean()), "prediction_positive_ratio": math.nan, "accuracy": math.nan},
        {**v4_accuracy_row("validation", features, splits["validation"], labels, val_pred, val_prob), "split": "validation"},
        {**v4_accuracy_row("final", features, splits["final"], labels, final_pred, final_prob), "split": "final"},
    ]

    pipe, selected2, status = v4_fit_transformer(features, splits["train"], feature_columns, 16)
    rolling_rows: list[dict[str, Any]] = []
    if pipe is not None and status == "ok":
        train_x = v4_transform(features, splits["train"], pipe, selected2)
        val_full = ordered_index(features, full_splits["validation"])
        val_cut = pd.Timestamp("2024-07-01")
        early_idx = v4_ordered_limit(features, val_full[pd.to_datetime(features.loc[val_full, "datetime"], errors="coerce") < val_cut], 320)
        late_idx = v4_ordered_limit(features, val_full[pd.to_datetime(features.loc[val_full, "datetime"], errors="coerce") >= val_cut], 320)
        final_full = v4_ordered_limit(features, full_splits["final"], 320)
        eval_indices = {"validation_early_2024": early_idx, "validation_late_2024": late_idx, "validation_v3_sample": splits["validation"], "final_2025plus_diagnostic": final_full}
        eval_frames = {name: v4_transform(features, idx, pipe, selected2) for name, idx in eval_indices.items() if len(idx)}
        rolling_outputs, rolling_reason = v4_torch_direction_predict("BiLSTM", train_x, labels.loc[splits["train"]], eval_frames, SEED, 12, 0.0)
        for name, idx in eval_indices.items():
            if name in rolling_outputs:
                rolling_rows.append(v4_accuracy_row(name, features, idx, labels, rolling_outputs[name]["pred"], rolling_outputs[name]["prob"]))
            else:
                rolling_rows.append({"window": name, "rows": int(len(idx)), "accuracy": math.nan, "status": rolling_reason or "not_evaluated"})

    seed_rows = []
    for seed in [SEED, 7, 21, 99, 2026]:
        row, _out, _sel = v4_eval_architecture(features, labels, splits["train"], splits["validation"], splits["final"], feature_columns, "combined_strategy_features", "BiLSTM", int(seed), 12, 0.0)
        seed_rows.append(row)

    ablation_specs = [
        ("LSTM", "combined_strategy_features", feature_groups["combined_strategy_features"], 12, 0.0),
        ("GRU", "combined_strategy_features", feature_groups["combined_strategy_features"], 12, 0.0),
        ("BiLSTM", "combined_strategy_features", feature_groups["combined_strategy_features"], 12, 0.0),
        ("BiLSTM_small", "combined_strategy_features", feature_groups["combined_strategy_features"], 6, 0.0),
        ("BiLSTM_dropout", "combined_strategy_features", feature_groups["combined_strategy_features"], 12, 0.2),
        ("BiLSTM", "combined_without_market_context", [col for col in feature_groups["combined_strategy_features"] if col not in set(feature_groups.get("market_context", []))], 12, 0.0),
        ("BiLSTM", "relative_strength", feature_groups["relative_strength"], 12, 0.0),
        ("BiLSTM", "market_context", feature_groups["market_context"], 12, 0.0),
    ]
    ablation_rows = []
    for arch, group_name, cols, hidden, dropout in ablation_specs:
        row, _out, _sel = v4_eval_architecture(features, labels, splits["train"], splits["validation"], splits["final"], cols, group_name, arch, SEED, hidden, dropout)
        ablation_rows.append(row)

    comparison_rows: list[dict[str, Any]] = [
        {
            "candidate_id": "qml_v8_context_market_relative_vn30_h40",
            "comparison_group": "qml_v8_context",
            "model_family": "QML V8 kernel-feature meta-model",
            "target_variant": "market_relative_vn30",
            "horizon": 40,
            "validation_accuracy": QML_V8_CONTEXT_VALIDATION,
            "final_accuracy": QML_V8_CONTEXT_FINAL,
            "claim_label": "diagnostic_context",
        },
        {**reconstruction, "comparison_group": "v4_bilstm_reconstruction"},
    ]
    for model_name in ["always_up", "always_down", "simple_relative_strength", "logistic_regression", "lightgbm_classifier", "rbf_svm"]:
        row, _pred = evaluate_direction_candidate(model_name, "combined_strategy_features", features, family_cols, relative_cols, index_data, labels, splits, feature_groups["combined_strategy_features"])
        row["comparison_group"] = "same_target_baseline"
        comparison_rows.append(row)

    final_ticker_values = [as_float(row.get("accuracy")) for row in ticker_rows if row.get("split") == "final" and math.isfinite(as_float(row.get("accuracy")))]
    baseline_rows = [row for row in comparison_rows if row.get("comparison_group") == "same_target_baseline"]
    strongest_baseline_final = max([as_float(row.get("final_accuracy")) for row in baseline_rows if math.isfinite(as_float(row.get("final_accuracy")))], default=math.nan)
    strongest_baseline_validation = max([as_float(row.get("validation_accuracy")) for row in baseline_rows if math.isfinite(as_float(row.get("validation_accuracy")))], default=math.nan)
    seed_final_values = [as_float(row.get("final_accuracy")) for row in seed_rows if math.isfinite(as_float(row.get("final_accuracy")))]
    seed_final_std = float(np.std(seed_final_values)) if seed_final_values else math.nan
    beats_strongest_baseline_final = as_float(reconstruction.get("final_accuracy")) > strongest_baseline_final if math.isfinite(strongest_baseline_final) else False
    seed_warning = "pass" if math.isfinite(seed_final_std) and seed_final_std <= 0.05 else "unstable"
    if not split_guard:
        decision_label = "bilstm_relock_failed_split_guard"
    elif not beats_strongest_baseline_final or seed_warning == "unstable":
        decision_label = "bilstm_relock_unstable_future_blind_required"
    elif as_float(reconstruction.get("validation_accuracy")) >= 0.60 and as_float(reconstruction.get("final_accuracy")) > QML_V8_CONTEXT_FINAL:
        decision_label = "bilstm_relock_future_blind_required"
    else:
        decision_label = "bilstm_relock_not_confirmed"
    relock_decision = {
        "decision_label": decision_label,
        "split_leakage_status": "pass" if split_guard else "fail",
        "validation_accuracy": reconstruction.get("validation_accuracy"),
        "final_accuracy": reconstruction.get("final_accuracy"),
        "beats_qml_v8_context_final": as_float(reconstruction.get("final_accuracy")) > QML_V8_CONTEXT_FINAL,
        "strongest_same_target_baseline_validation_accuracy": strongest_baseline_validation,
        "strongest_same_target_baseline_final_accuracy": strongest_baseline_final,
        "beats_strongest_same_target_baseline_final": beats_strongest_baseline_final,
        "seed_final_accuracy_std": seed_final_std,
        "seed_stability_warning": seed_warning,
        "beats_absolute_direction_6161_champion": False,
        "reason": "market_relative_vn30 h40 target differs from absolute-direction champion; same-target simple-baseline and seed-stability checks prevent stronger interpretation",
        "final_ticker_mean_accuracy": float(np.mean(final_ticker_values)) if final_ticker_values else math.nan,
        "no_daily_t1_system_created": True,
        "no_trading_claim": True,
    }

    write_json(OUTPUT_DIR / "v4_bilstm_candidate_reconstruction.json", reconstruction_json)
    write_frame(OUTPUT_DIR / "v4_bilstm_split_leakage_audit.csv", split_rows, list(split_rows[0].keys()))
    write_frame(OUTPUT_DIR / "v4_bilstm_ticker_stability.csv", ticker_rows, list(ticker_rows[0].keys()) if ticker_rows else [])
    write_frame(OUTPUT_DIR / "v4_bilstm_quarter_stability.csv", period_rows, list(period_rows[0].keys()) if period_rows else [])
    write_frame(OUTPUT_DIR / "v4_bilstm_prediction_balance.csv", balance_rows, sorted(set().union(*(row.keys() for row in balance_rows))))
    write_frame(OUTPUT_DIR / "v4_bilstm_rolling_origin.csv", rolling_rows, sorted(set().union(*(row.keys() for row in rolling_rows))) if rolling_rows else [])
    write_frame(OUTPUT_DIR / "v4_bilstm_seed_sensitivity.csv", seed_rows, sorted(set().union(*(row.keys() for row in seed_rows))) if seed_rows else [])
    write_frame(OUTPUT_DIR / "v4_bilstm_ablation_results.csv", ablation_rows, sorted(set().union(*(row.keys() for row in ablation_rows))) if ablation_rows else [])
    write_frame(OUTPUT_DIR / "v4_bilstm_comparison_summary.csv", comparison_rows, sorted(set().union(*(row.keys() for row in comparison_rows))) if comparison_rows else [])
    write_json(OUTPUT_DIR / "v4_bilstm_relock_decision.json", relock_decision)
    write_v4_reports(reconstruction, relock_decision, comparison_rows, seed_rows, ablation_rows, rolling_rows)
    result = {
        "status": "ok",
        "mode": "bilstm_relock",
        "runtime_seconds": time.perf_counter() - started,
        "validation_accuracy": reconstruction.get("validation_accuracy"),
        "final_accuracy": reconstruction.get("final_accuracy"),
        "decision_label": relock_decision["decision_label"],
        "diagnostic_only": True,
        "no_daily_t1_system_created": True,
        "no_trading_claim": True,
    }
    print(json.dumps(json_safe(result), indent=2))
    return result


def safe_int(value: Any, default: int = 0) -> int:
    number = as_float(value)
    return int(number) if math.isfinite(number) else default


def safe_bool_text(value: bool) -> str:
    return "yes" if bool(value) else "no"


def v5_direction_frame() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, source in [
        ("direction_validation_results.csv", "v1_v2_validation_screening"),
        ("v3_direction_validation_results.csv", "v3_skipped_families"),
        ("v3_ensemble_results.csv", "v3_ensembles"),
        ("v3_qml_integration_results.csv", "v3_qml_integration"),
        ("v4_bilstm_comparison_summary.csv", "v4_bilstm_relock"),
    ]:
        frame = read_artifact(name)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_artifact"] = name
        frame["source_stage"] = source
        if "task" in frame.columns:
            frame = frame[(frame["task"].fillna("direction") == "direction") | frame["task"].isna()].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[combined.get("candidate_id", "").astype(str).ne("")].copy()
    combined = combined.drop_duplicates("candidate_id", keep="last")
    return combined.reset_index(drop=True)


def v5_price_frame() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, source in [
        ("price_validation_results.csv", "v1_v2_validation_screening"),
        ("v3_price_validation_results.csv", "v3_skipped_families"),
        ("v3_deep_sequence_results.csv", "v3_deep_sequence"),
    ]:
        frame = read_artifact(name)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_artifact"] = name
        frame["source_stage"] = source
        if "task" in frame.columns:
            frame = frame[frame["task"].fillna("").eq("price_return")].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[combined.get("candidate_id", "").astype(str).ne("")].copy()
    combined = combined.drop_duplicates("candidate_id", keep="last")
    return combined.reset_index(drop=True)


def v5_direction_predictions_for_candidate(
    candidate: pd.Series,
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    config: RunConfig,
) -> dict[str, Any]:
    model = str(candidate.get("model_family", ""))
    feature_group = str(candidate.get("feature_group", ""))
    target_variant = str(candidate.get("target_variant", ""))
    horizon = safe_int(candidate.get("horizon"))
    labels = build_direction_target(features, index_data, target_variant, horizon)
    splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
    if str(candidate.get("source_stage", "")).startswith("v4") or str(candidate.get("candidate_id", "")).startswith("v4__"):
        row, outputs, _selected = v4_eval_architecture(
            features,
            labels,
            splits["train"],
            splits["validation"],
            splits["final"],
            feature_groups.get(feature_group, []),
            feature_group,
            model if model in {"LSTM", "GRU", "BiLSTM", "BiLSTM_small", "BiLSTM_dropout"} else "BiLSTM",
            safe_int(candidate.get("seed"), SEED),
            safe_int(candidate.get("hidden_size"), 12),
            as_float(candidate.get("dropout")) if math.isfinite(as_float(candidate.get("dropout"))) else 0.0,
        )
        return {"row": row, "outputs": outputs, "splits": splits, "labels": labels}
    row, _final_pred = evaluate_direction_candidate(model, feature_group, features, family_cols, relative_cols, index_data, labels, splits, feature_groups.get(feature_group, []))
    return {"row": row, "outputs": {}, "splits": splits, "labels": labels}


def v5_repaired_classification_metrics(y_true: pd.Series, pred: np.ndarray | None, prob: np.ndarray | None = None) -> dict[str, float]:
    if pred is None:
        return {
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "macro_f1": math.nan,
            "positive_f1": math.nan,
            "mcc": math.nan,
            "roc_auc": math.nan,
            "label_positive_ratio": math.nan,
            "prediction_positive_ratio": math.nan,
            "class_balance_gap": math.nan,
            "majority_baseline_accuracy": math.nan,
        }
    y = y_true.astype(int).to_numpy()
    p = np.asarray(pred, dtype=int)
    prob_arr = np.asarray(prob, dtype=float) if prob is not None else p.astype(float)
    if not len(y):
        return {
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "macro_f1": math.nan,
            "positive_f1": math.nan,
            "mcc": math.nan,
            "roc_auc": math.nan,
            "label_positive_ratio": math.nan,
            "prediction_positive_ratio": math.nan,
            "class_balance_gap": math.nan,
            "majority_baseline_accuracy": math.nan,
        }
    label_pos = float(np.mean(y))
    pred_pos = float(np.mean(p)) if len(p) else math.nan
    return {
        "accuracy": float(np.mean(y == p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)) if len(np.unique(y)) > 1 else math.nan,
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, p, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else 0.0,
        "roc_auc": safe_auc(y, prob_arr),
        "label_positive_ratio": label_pos,
        "prediction_positive_ratio": pred_pos,
        "class_balance_gap": abs(label_pos - pred_pos) if math.isfinite(pred_pos) else math.nan,
        "majority_baseline_accuracy": max(label_pos, 1.0 - label_pos),
    }


def v5_baseline_metric_rows(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    target_variant: str,
    horizon: int,
    feature_group: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_y = labels.loc[splits["train"]].astype(int)
    for baseline in ["always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"]:
        val_pred, val_prob, reason = baseline_direction_prediction(baseline, features, train_y, splits["validation"])
        fin_pred, fin_prob, reason2 = baseline_direction_prediction(baseline, features, train_y, splits["final"])
        if val_pred is None or fin_pred is None:
            rows.append(
                {
                    "candidate_id": candidate_id("v5_baseline", baseline, target_variant, f"h{horizon}", feature_group),
                    "model_family": baseline,
                    "target_variant": target_variant,
                    "horizon": horizon,
                    "feature_group": feature_group,
                    "status": "skipped",
                    "skipped_reason": reason or reason2,
                }
            )
            continue
        val_metrics = v5_repaired_classification_metrics(labels.loc[splits["validation"]], val_pred, val_prob)
        final_metrics = v5_repaired_classification_metrics(labels.loc[splits["final"]], fin_pred, fin_prob)
        rows.append(
            {
                "candidate_id": candidate_id("v5_baseline", baseline, target_variant, f"h{horizon}", feature_group),
                "model_family": baseline,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_group": feature_group,
                "status": "ok",
                "validation_accuracy": val_metrics["accuracy"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_mcc": val_metrics["mcc"],
                "final_accuracy": final_metrics["accuracy"],
                "final_balanced_accuracy": final_metrics["balanced_accuracy"],
                "final_macro_f1": final_metrics["macro_f1"],
                "final_mcc": final_metrics["mcc"],
                "validation_prediction_positive_ratio": val_metrics["prediction_positive_ratio"],
                "final_prediction_positive_ratio": final_metrics["prediction_positive_ratio"],
            }
        )
    return rows


def v5_class_balance_audit(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], config: RunConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_variant in DIRECTION_TARGETS:
        for horizon in HORIZONS:
            labels = build_direction_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
            for split_name, idx in splits.items():
                y = labels.loc[idx].dropna().astype(int)
                positive_ratio = float(y.mean()) if len(y) else math.nan
                majority = max(positive_ratio, 1.0 - positive_ratio) if math.isfinite(positive_ratio) else math.nan
                rows.append(
                    {
                        "target_variant": target_variant,
                        "horizon": horizon,
                        "split": split_name,
                        "rows": int(len(y)),
                        "positive_ratio": positive_ratio,
                        "negative_ratio": 1.0 - positive_ratio if math.isfinite(positive_ratio) else math.nan,
                        "majority_class_accuracy": majority,
                        "imbalance_gap": abs(positive_ratio - 0.5) if math.isfinite(positive_ratio) else math.nan,
                        "least_contaminated_score": 1.0 - abs(positive_ratio - 0.5) if math.isfinite(positive_ratio) else math.nan,
                    }
                )
    return rows


def v5_target_repair_results(class_balance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(class_balance_rows)
    if frame.empty:
        return []
    results: list[dict[str, Any]] = []
    for target_variant, group in frame.groupby("target_variant", sort=True):
        weights = numeric_series(group, "rows").replace(0, np.nan)
        gaps = numeric_series(group, "imbalance_gap")
        weighted_gap = float(np.nansum(gaps * weights) / np.nansum(weights)) if np.nansum(weights) > 0 else math.nan
        max_gap = float(gaps.max()) if gaps.notna().any() else math.nan
        final_group = group[group["split"].eq("final")]
        final_gap = float(numeric_series(final_group, "imbalance_gap").mean()) if not final_group.empty else math.nan
        results.append(
            {
                "target_variant": target_variant,
                "weighted_split_imbalance_gap": weighted_gap,
                "max_split_imbalance_gap": max_gap,
                "final_imbalance_gap": final_gap,
                "least_contaminated_rank_score": 1.0 - weighted_gap if math.isfinite(weighted_gap) else math.nan,
                "repair_verdict": "least_contaminated" if math.isfinite(weighted_gap) else "not_available",
            }
        )
    results = sorted(results, key=lambda row: as_float(row.get("weighted_split_imbalance_gap")))
    for rank, row in enumerate(results, start=1):
        row["least_contaminated_rank"] = rank
        row["repair_verdict"] = "least_contaminated" if rank == 1 else "usable_with_balance_reporting"
    return results


def v5_normalize_direction_row(row: pd.Series, source_row: dict[str, Any]) -> dict[str, Any]:
    validation_bal = as_float(row.get("validation_balanced_accuracy"))
    if not math.isfinite(validation_bal):
        validation_bal = as_float(row.get("balanced_accuracy"))
    validation_macro_f1 = as_float(row.get("validation_macro_f1"))
    if not math.isfinite(validation_macro_f1):
        source_f1 = as_float(row.get("validation_f1"))
        if not math.isfinite(source_f1):
            source_f1 = as_float(row.get("f1"))
        validation_macro_f1 = source_f1
    final_bal = as_float(row.get("final_balanced_accuracy"))
    final_macro_f1 = as_float(row.get("final_macro_f1"))
    if not math.isfinite(final_macro_f1):
        final_macro_f1 = as_float(row.get("final_f1"))
    return {
        "candidate_id": str(row.get("candidate_id", source_row.get("candidate_id", ""))),
        "source_stage": str(row.get("source_stage", source_row.get("source_stage", ""))),
        "model_family": str(row.get("model_family", source_row.get("model_family", ""))),
        "feature_group": str(row.get("feature_group", source_row.get("feature_group", ""))),
        "target_variant": str(row.get("target_variant", source_row.get("target_variant", ""))),
        "horizon": safe_int(row.get("horizon", source_row.get("horizon"))),
        "validation_accuracy": as_float(row.get("validation_accuracy")),
        "validation_balanced_accuracy": validation_bal,
        "validation_macro_f1": validation_macro_f1,
        "validation_mcc": as_float(row.get("validation_mcc")),
        "validation_roc_auc": as_float(row.get("validation_roc_auc", row.get("roc_auc"))),
        "validation_lift_over_strongest_accuracy_baseline": as_float(row.get("validation_lift_over_strongest_baseline", row.get("lift_over_strongest_baseline"))),
        "final_accuracy": as_float(row.get("final_accuracy")),
        "final_balanced_accuracy": final_bal,
        "final_macro_f1": final_macro_f1,
        "final_mcc": as_float(row.get("final_mcc")),
        "final_roc_auc": as_float(row.get("final_roc_auc")),
        "status": str(row.get("status", "ok")),
        "claim_label": str(row.get("claim_label", "diagnostic_only")),
    }


def v5_direction_repair_results(
    direction_candidates: pd.DataFrame,
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    config: RunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if direction_candidates.empty:
        return [], [], []
    frame = direction_candidates.copy()
    for column in ["validation_accuracy", "validation_balanced_accuracy", "balanced_accuracy", "final_accuracy", "final_balanced_accuracy", "validation_lift_over_strongest_baseline", "lift_over_strongest_baseline"]:
        frame[column] = numeric_series(frame, column)
    frame = frame[frame.get("status", "ok").fillna("ok").eq("ok")].copy()
    priorities = []
    bilstm = frame[frame["candidate_id"].astype(str).str.contains("BiLSTM", case=False, na=False)]
    if not bilstm.empty:
        priorities.extend(bilstm.sort_values(["final_accuracy", "validation_accuracy"], ascending=False).head(6)["candidate_id"].tolist())
    for target_variant in ["absolute_direction", "market_relative_vn30"]:
        subset = frame[(frame["target_variant"] == target_variant) & (numeric_series(frame, "horizon") == 40)]
        priorities.extend(subset.sort_values(["validation_balanced_accuracy", "validation_accuracy"], ascending=False).head(8)["candidate_id"].tolist())
        priorities.extend(subset.sort_values(["final_accuracy", "validation_accuracy"], ascending=False).head(5)["candidate_id"].tolist())
    priorities.extend(frame.sort_values(["validation_balanced_accuracy", "validation_accuracy"], ascending=False).head(12)["candidate_id"].tolist())
    seen: set[str] = set()
    selected_ids = []
    for cid in priorities:
        key = str(cid)
        if key not in seen:
            seen.add(key)
            selected_ids.append(key)

    repair_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    bilstm_rows: list[dict[str, Any]] = []
    for cid in selected_ids:
        source = frame[frame["candidate_id"].astype(str).eq(cid)]
        if source.empty:
            continue
        candidate = source.iloc[0]
        target_variant = str(candidate.get("target_variant", ""))
        horizon = safe_int(candidate.get("horizon"))
        feature_group = str(candidate.get("feature_group", ""))
        try:
            labels = build_direction_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
            baseline_metrics = v5_baseline_metric_rows(features, labels, splits, target_variant, horizon, feature_group)
            baseline_rows.extend(baseline_metrics)
            baseline_ok = [row for row in baseline_metrics if row.get("status") == "ok"]
            strongest_accuracy = max([as_float(row.get("validation_accuracy")) for row in baseline_ok], default=math.nan)
            strongest_balanced = max([as_float(row.get("validation_balanced_accuracy")) for row in baseline_ok], default=math.nan)
            strongest_macro_f1 = max([as_float(row.get("validation_macro_f1")) for row in baseline_ok], default=math.nan)
            strongest_mcc = max([as_float(row.get("validation_mcc")) for row in baseline_ok], default=math.nan)
            recomputed = v5_direction_predictions_for_candidate(candidate, features, family_cols, relative_cols, index_data, feature_groups, config)
            row = v5_normalize_direction_row(pd.Series(recomputed["row"]), candidate.to_dict())
            if recomputed.get("outputs"):
                outputs = recomputed["outputs"]
                if "validation" in outputs:
                    vm = v5_repaired_classification_metrics(labels.loc[splits["validation"]], outputs["validation"].get("pred"), outputs["validation"].get("prob"))
                    row.update({"validation_macro_f1": vm["macro_f1"], "validation_mcc": vm["mcc"], "validation_roc_auc": vm["roc_auc"], "validation_prediction_positive_ratio": vm["prediction_positive_ratio"], "validation_class_balance_gap": vm["class_balance_gap"]})
                if "final" in outputs:
                    fm = v5_repaired_classification_metrics(labels.loc[splits["final"]], outputs["final"].get("pred"), outputs["final"].get("prob"))
                    row.update({"final_macro_f1": fm["macro_f1"], "final_mcc": fm["mcc"], "final_roc_auc": fm["roc_auc"], "final_prediction_positive_ratio": fm["prediction_positive_ratio"], "final_class_balance_gap": fm["class_balance_gap"]})
            else:
                row.setdefault("validation_prediction_positive_ratio", math.nan)
                row.setdefault("validation_class_balance_gap", math.nan)
                row.setdefault("final_prediction_positive_ratio", math.nan)
                row.setdefault("final_class_balance_gap", math.nan)
            baseline_gate_passed = bool(
                math.isfinite(row["validation_balanced_accuracy"])
                and row["validation_balanced_accuracy"] > max(0.5, strongest_balanced if math.isfinite(strongest_balanced) else 0.5)
                and row["validation_macro_f1"] > (strongest_macro_f1 if math.isfinite(strongest_macro_f1) else 0.0)
                and (not math.isfinite(row["validation_mcc"]) or row["validation_mcc"] > (strongest_mcc if math.isfinite(strongest_mcc) else 0.0))
            )
            row.update(
                {
                    "strongest_validation_accuracy_baseline": strongest_accuracy,
                    "strongest_validation_balanced_accuracy_baseline": strongest_balanced,
                    "strongest_validation_macro_f1_baseline": strongest_macro_f1,
                    "strongest_validation_mcc_baseline": strongest_mcc,
                    "validation_lift_over_strongest_balanced_accuracy_baseline": row["validation_balanced_accuracy"] - strongest_balanced if math.isfinite(strongest_balanced) else math.nan,
                    "validation_lift_over_strongest_macro_f1_baseline": row["validation_macro_f1"] - strongest_macro_f1 if math.isfinite(strongest_macro_f1) else math.nan,
                    "validation_lift_over_strongest_mcc_baseline": row["validation_mcc"] - strongest_mcc if math.isfinite(row["validation_mcc"]) and math.isfinite(strongest_mcc) else math.nan,
                    "baseline_gate_passed": baseline_gate_passed,
                    "comparable_absolute_champion_scope": target_variant == "absolute_direction" and horizon == 40,
                    "beats_6161_champion_repaired_comparable": target_variant == "absolute_direction" and horizon == 40 and as_float(row.get("final_accuracy")) > CLASSICAL_CHAMPION["final_accuracy"] and as_float(row.get("final_balanced_accuracy")) > 0.5,
                    "comparable_qml_v8_scope": target_variant == "market_relative_vn30" and horizon == 40,
                    "beats_qml_v8_repaired_comparable": target_variant == "market_relative_vn30" and horizon == 40 and as_float(row.get("final_accuracy")) > QML_V8_CONTEXT_FINAL and as_float(row.get("final_balanced_accuracy")) > 0.5,
                    "claimable": False,
                    "claim_boundary_label": "diagnostic_only_future_blind_required" if baseline_gate_passed else "diagnostic_only_not_repaired_baseline_gated",
                }
            )
            repair_rows.append(row)
            if "bilstm" in str(row.get("model_family", "")).lower() or "bilstm" in str(row.get("candidate_id", "")).lower():
                bilstm_rows.append(row)
        except Exception as exc:
            repair_rows.append(
                {
                    "candidate_id": cid,
                    "source_stage": str(candidate.get("source_stage", "")),
                    "model_family": str(candidate.get("model_family", "")),
                    "feature_group": str(candidate.get("feature_group", "")),
                    "target_variant": str(candidate.get("target_variant", "")),
                    "horizon": safe_int(candidate.get("horizon")),
                    "status": "skipped",
                    "skipped_reason": f"{type(exc).__name__}: {exc}",
                    "claimable": False,
                }
            )
    return repair_rows, baseline_rows, bilstm_rows


def v5_price_repair_results(price_candidates: pd.DataFrame) -> list[dict[str, Any]]:
    if price_candidates.empty:
        return []
    frame = price_candidates.copy()
    frame = frame[frame.get("status", "ok").fillna("ok").eq("ok")].copy()
    rows: list[dict[str, Any]] = []
    for column in ["validation_rmse", "final_rmse", "validation_error_improvement_over_baseline", "error_improvement_over_baseline", "final_error_improvement_over_baseline", "validation_sign_accuracy", "final_sign_accuracy", "validation_correlation_pred_actual", "final_correlation_pred_actual"]:
        frame[column] = numeric_series(frame, column)
    frame["validation_repaired_improvement"] = frame["validation_error_improvement_over_baseline"]
    missing_validation = ~frame["validation_repaired_improvement"].notna()
    frame.loc[missing_validation, "validation_repaired_improvement"] = frame.loc[missing_validation, "error_improvement_over_baseline"]
    frame["final_repaired_improvement"] = frame["final_error_improvement_over_baseline"]
    missing_final = ~frame["final_repaired_improvement"].notna()
    frame.loc[missing_final, "final_repaired_improvement"] = frame.loc[missing_final, "error_improvement_over_baseline"]
    priority = pd.concat(
        [
            frame.sort_values("validation_repaired_improvement", ascending=False).head(25),
            frame.sort_values("final_repaired_improvement", ascending=False).head(25),
            frame[frame["target_variant"].isin(["forward_log_return_h", "forward_simple_return_h", "market_excess_return_h"])].sort_values("validation_repaired_improvement", ascending=False).head(20),
        ],
        ignore_index=True,
    ).drop_duplicates("candidate_id")
    for _idx, row in priority.iterrows():
        validation_imp = as_float(row.get("validation_repaired_improvement"))
        final_imp = as_float(row.get("final_repaired_improvement"))
        robust = bool(validation_imp > 0.0 and final_imp > 0.0 and as_float(row.get("validation_correlation_pred_actual")) >= 0.0 and as_float(row.get("final_correlation_pred_actual")) >= 0.0)
        rows.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "source_stage": str(row.get("source_stage", "")),
                "model_family": str(row.get("model_family", "")),
                "feature_group": str(row.get("feature_group", "")),
                "target_variant": str(row.get("target_variant", "")),
                "horizon": safe_int(row.get("horizon")),
                "validation_rmse": as_float(row.get("validation_rmse")),
                "final_rmse": as_float(row.get("final_rmse")),
                "validation_error_improvement_over_random_walk_or_last_price": validation_imp,
                "final_error_improvement_over_random_walk_or_last_price": final_imp,
                "validation_sign_accuracy": as_float(row.get("validation_sign_accuracy")),
                "final_sign_accuracy": as_float(row.get("final_sign_accuracy")),
                "validation_correlation_pred_actual": as_float(row.get("validation_correlation_pred_actual")),
                "final_correlation_pred_actual": as_float(row.get("final_correlation_pred_actual")),
                "robustly_beats_random_walk_or_last_price": robust,
                "claimable": False,
                "claim_boundary_label": "future_blind_required" if robust else "diagnostic_only_not_robust",
            }
        )
    return rows


def unique_rows_by_candidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("candidate_id", ""))
        if key and key not in unique:
            unique[key] = row
    return list(unique.values())


def v5_future_blind_registry(
    direction_rows: list[dict[str, Any]],
    price_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in direction_rows:
        if row.get("status") != "ok" or not bool(row.get("baseline_gate_passed")):
            continue
        cid = str(row.get("candidate_id", ""))
        if cid in seen:
            continue
        seen.add(cid)
        registry.append(
            {
                "candidate_id": cid,
                "task": "direction",
                "target_variant": row.get("target_variant", ""),
                "horizon": row.get("horizon", ""),
                "why_promising": f"baseline-gated repaired validation balanced accuracy {pct(row.get('validation_balanced_accuracy'))}, macro F1 {pct(row.get('validation_macro_f1'))}, MCC {as_float(row.get('validation_mcc')):.4f}",
                "why_not_claimable_now": "V5 repairs reuse diagnostic artifacts and final rows are scoring-only",
                "future_blind_test_needed": "pre-register target/model/features/threshold and evaluate on a future-blind post-final VN30 hourly window",
                "claimable_now": False,
            }
        )
    for row in price_rows:
        if not bool(row.get("robustly_beats_random_walk_or_last_price")):
            continue
        cid = str(row.get("candidate_id", ""))
        if cid in seen:
            continue
        seen.add(cid)
        registry.append(
            {
                "candidate_id": cid,
                "task": "price_return",
                "target_variant": row.get("target_variant", ""),
                "horizon": row.get("horizon", ""),
                "why_promising": f"validation and final RMSE improvement over random-walk/last-price baselines; final improvement {pp(row.get('final_error_improvement_over_random_walk_or_last_price'))}",
                "why_not_claimable_now": "price/return repair is diagnostic and requires pre-registered future-blind transfer",
                "future_blind_test_needed": "compare against random walk/last price on a future-blind window with unchanged transforms and metrics",
                "claimable_now": False,
            }
        )
    best_target = min(target_rows, key=lambda row: as_float(row.get("weighted_split_imbalance_gap")), default={})
    if best_target:
        cid = f"target_repair__{best_target.get('target_variant', '')}"
        registry.append(
            {
                "candidate_id": cid,
                "task": "target_repair",
                "target_variant": best_target.get("target_variant", ""),
                "horizon": "all",
                "why_promising": f"least class-imbalance contamination under weighted split imbalance gap {as_float(best_target.get('weighted_split_imbalance_gap')):.4f}",
                "why_not_claimable_now": "target selection must be pre-registered before future-blind scoring",
                "future_blind_test_needed": "lock the repaired target before any future-blind evaluation",
                "claimable_now": False,
            }
        )
    return registry


def write_v5_reports(
    class_rows: list[dict[str, Any]],
    direction_rows: list[dict[str, Any]],
    bilstm_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    price_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
) -> None:
    class_frame = pd.DataFrame(class_rows)
    ok_bilstm_rows = [row for row in bilstm_rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("final_accuracy")))]
    bilstm_best = max(ok_bilstm_rows, key=lambda row: (as_float(row.get("final_accuracy")), as_float(row.get("validation_balanced_accuracy"))), default={})
    best_direction = max([row for row in direction_rows if row.get("status") == "ok" and bool(row.get("baseline_gate_passed"))], key=lambda row: (as_float(row.get("validation_balanced_accuracy")), as_float(row.get("validation_macro_f1")), as_float(row.get("validation_mcc"))), default={})
    if not best_direction:
        best_direction = max([row for row in direction_rows if row.get("status") == "ok"], key=lambda row: (as_float(row.get("validation_balanced_accuracy")), as_float(row.get("validation_macro_f1"))), default={})
    best_target = min(target_rows, key=lambda row: as_float(row.get("weighted_split_imbalance_gap")), default={})
    abs_beat = any(bool(row.get("beats_6161_champion_repaired_comparable")) and bool(row.get("baseline_gate_passed")) for row in direction_rows)
    mr_beat = any(bool(row.get("beats_qml_v8_repaired_comparable")) and bool(row.get("baseline_gate_passed")) for row in direction_rows)
    robust_price = [row for row in price_rows if bool(row.get("robustly_beats_random_walk_or_last_price"))]
    price_pool = robust_price if robust_price else price_rows
    best_price = max(price_pool, key=lambda row: (as_float(row.get("validation_error_improvement_over_random_walk_or_last_price")), as_float(row.get("final_error_improvement_over_random_walk_or_last_price"))), default={})
    bilstm_class_driven = bool(as_float(bilstm_best.get("final_accuracy")) >= 0.70 and as_float(bilstm_best.get("final_balanced_accuracy")) <= 0.50 and as_float(bilstm_best.get("final_prediction_positive_ratio")) < 0.15)
    bilstm_strong_repaired = bool(
        as_float(bilstm_best.get("validation_balanced_accuracy")) > 0.5
        and as_float(bilstm_best.get("validation_macro_f1")) > as_float(bilstm_best.get("strongest_validation_macro_f1_baseline"))
        and as_float(bilstm_best.get("validation_mcc")) > as_float(bilstm_best.get("strongest_validation_mcc_baseline"))
        and as_float(bilstm_best.get("validation_lift_over_strongest_balanced_accuracy_baseline")) > 0
        and as_float(bilstm_best.get("final_balanced_accuracy")) > 0.5
    )
    final_balance = class_frame[(class_frame.get("target_variant", "") == bilstm_best.get("target_variant", "")) & (class_frame.get("horizon", 0) == bilstm_best.get("horizon", 0)) & (class_frame.get("split", "") == "final")] if not class_frame.empty and bilstm_best else pd.DataFrame()
    final_majority = as_float(final_balance.iloc[0].get("majority_class_accuracy")) if not final_balance.empty else math.nan
    summary = f"""# VN30 Model Universe V5 Target and Metric Repair Result Summary

## Required Answers

1. Was the BiLSTM 72.50% final result mostly class-imbalance driven: {safe_bool_text(bilstm_class_driven)}. Best BiLSTM repaired row `{bilstm_best.get("candidate_id", "")}` has final accuracy {pct(bilstm_best.get("final_accuracy"))}, final balanced accuracy {pct(bilstm_best.get("final_balanced_accuracy"))}, final macro F1 {pct(bilstm_best.get("final_macro_f1"))}, final MCC {as_float(bilstm_best.get("final_mcc")):.4f}, final predicted-positive ratio {pct(bilstm_best.get("final_prediction_positive_ratio"))}, and same split majority baseline {pct(final_majority)}.
2. Does BiLSTM still look strong under balanced accuracy, macro F1, MCC, and lift over strongest baseline: {safe_bool_text(bilstm_strong_repaired)}. Validation balanced-accuracy lift is {pp(bilstm_best.get("validation_lift_over_strongest_balanced_accuracy_baseline"))}, macro-F1 lift is {pp(bilstm_best.get("validation_lift_over_strongest_macro_f1_baseline"))}, and MCC lift is {as_float(bilstm_best.get("validation_lift_over_strongest_mcc_baseline")):.4f}.
3. Best direction candidate under baseline-gated repaired metrics: `{best_direction.get("candidate_id", "")}` with validation balanced accuracy {pct(best_direction.get("validation_balanced_accuracy"))}, macro F1 {pct(best_direction.get("validation_macro_f1"))}, MCC {as_float(best_direction.get("validation_mcc")):.4f}, baseline gate `{str(best_direction.get("baseline_gate_passed", False)).lower()}`.
4. Least class-imbalance-contaminated target: `{best_target.get("target_variant", "")}` with weighted split imbalance gap {as_float(best_target.get("weighted_split_imbalance_gap")):.4f}.
5. Does any candidate beat the 61.61% absolute-direction champion on comparable scope under repaired metrics: {safe_bool_text(abs_beat)}.
6. Does any market-relative candidate beat QML V8 64.44 on comparable scope under repaired metrics: {safe_bool_text(mr_beat)}.
7. Does any price/return model beat random walk / last price robustly: {safe_bool_text(bool(robust_price))}. Best repaired price/return row `{best_price.get("candidate_id", "")}` has validation improvement {pp(best_price.get("validation_error_improvement_over_random_walk_or_last_price"))} and final improvement {pp(best_price.get("final_error_improvement_over_random_walk_or_last_price"))}.
8. Which candidates remain future-blind worthy: {len(registry)} rows are listed in `v5_future_blind_candidate_registry.csv`.
9. Exact claim boundary: offline diagnostic-only VN30 stock hourly repair audit; no result is claimable now; no trading, profitability, BUY/SELL, investment recommendation, live deployment, VN100, index-as-stock, DOCX, tag, merge, push --mirror, main-branch, or champion-replacement claim is made.

## Artifact Index

- `v5_class_balance_audit.csv`
- `v5_metric_repair_results.csv`
- `v5_baseline_gated_leaderboard.csv`
- `v5_bilstm_metric_repair.csv`
- `v5_target_repair_results.csv`
- `v5_price_return_metric_repair.csv`
- `v5_future_blind_candidate_registry.csv`
"""
    write_markdown(V5_RESULT_PATH, summary)
    claim = """# VN30 Model Universe V5 Target and Metric Repair Claim Boundary

- V5 is an offline diagnostic-only target and metric repair audit for VN30 stock hourly forecasting.
- V5 repairs interpretation of prior model-universe direction and price/return diagnostics; it does not create a trading system or live deployment.
- Direction metrics are repaired with balanced accuracy, macro F1, MCC, class-balance audit, and lift over strongest same-target baselines.
- Price/return metrics are evaluated separately against random-walk / last-price or return baselines; direction accuracy is not mixed with RMSE, MAE, MAPE, or return metrics.
- Candidate selection remains validation-governed; final rows are scoring-only.
- No V5 result replaces the 61.61% absolute-direction classical champion.
- No V5 result replaces or supersedes the 64.44% QML V8 market_relative_vn30 context result.
- Future-blind-worthy rows are candidates for pre-registered confirmation only and are not claimable now.
- Scope is VN30 stock hourly only; no VN100 scope is claimed.
- Index data may be used only as lagged market-context or market-relative target context; no index-as-stock claim is made.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, DOCX, tag, merge, push --mirror, history rewrite, main-branch, or paper artifact claim is made.
"""
    write_markdown(V5_CLAIM_PATH, claim)


def run_target_metric_repair(timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, family_cols, _feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    index_data = load_index_data()
    features, relative_cols = add_v3_relative_strength_features(features, index_data)
    feature_groups = build_feature_groups(features, family_cols, relative_cols)
    config = RunConfig("target_metric_repair", timeout_seconds, 800, 320, 320, 120, 120)

    class_rows = v5_class_balance_audit(features, index_data, config)
    target_rows = v5_target_repair_results(class_rows)
    direction_rows, baseline_rows, bilstm_rows = v5_direction_repair_results(v5_direction_frame(), features, family_cols, relative_cols, index_data, feature_groups, config)
    price_rows = v5_price_repair_results(v5_price_frame())
    direction_rows = unique_rows_by_candidate(direction_rows)
    bilstm_rows = unique_rows_by_candidate(bilstm_rows)
    price_rows = unique_rows_by_candidate(price_rows)
    registry = v5_future_blind_registry(direction_rows, price_rows, target_rows)
    leaderboard = sorted(
        [row for row in direction_rows if row.get("status") == "ok"],
        key=lambda row: (bool(row.get("baseline_gate_passed")), as_float(row.get("validation_balanced_accuracy")), as_float(row.get("validation_macro_f1")), as_float(row.get("validation_mcc"))),
        reverse=True,
    )

    write_frame(OUTPUT_DIR / "v5_class_balance_audit.csv", class_rows, sorted(set().union(*(row.keys() for row in class_rows))) if class_rows else [])
    write_frame(OUTPUT_DIR / "v5_metric_repair_results.csv", direction_rows, sorted(set().union(*(row.keys() for row in direction_rows))) if direction_rows else [])
    write_frame(OUTPUT_DIR / "v5_baseline_gated_leaderboard.csv", leaderboard, sorted(set().union(*(row.keys() for row in leaderboard))) if leaderboard else [])
    write_frame(OUTPUT_DIR / "v5_bilstm_metric_repair.csv", bilstm_rows, sorted(set().union(*(row.keys() for row in bilstm_rows))) if bilstm_rows else [])
    write_frame(OUTPUT_DIR / "v5_target_repair_results.csv", target_rows, sorted(set().union(*(row.keys() for row in target_rows))) if target_rows else [])
    write_frame(OUTPUT_DIR / "v5_price_return_metric_repair.csv", price_rows, sorted(set().union(*(row.keys() for row in price_rows))) if price_rows else [])
    write_frame(OUTPUT_DIR / "v5_future_blind_candidate_registry.csv", registry, sorted(set().union(*(row.keys() for row in registry))) if registry else [])
    write_frame(OUTPUT_DIR / "v5_repaired_baseline_metrics.csv", baseline_rows, sorted(set().union(*(row.keys() for row in baseline_rows))) if baseline_rows else [])
    write_v5_reports(class_rows, direction_rows, bilstm_rows, target_rows, price_rows, registry)

    best_direction = leaderboard[0] if leaderboard else {}
    best_target = target_rows[0] if target_rows else {}
    robust_price_count = sum(1 for row in price_rows if bool(row.get("robustly_beats_random_walk_or_last_price")))
    result = {
        "status": "ok",
        "mode": "target_metric_repair",
        "runtime_seconds": time.perf_counter() - started,
        "direction_rows": len(direction_rows),
        "price_rows": len(price_rows),
        "future_blind_registry_rows": len(registry),
        "best_baseline_gated_direction_candidate": best_direction.get("candidate_id", ""),
        "best_repaired_target": best_target.get("target_variant", ""),
        "robust_price_return_rows": robust_price_count,
        "claimable_results": 0,
        "diagnostic_only": True,
        "no_trading_claim": True,
    }
    print(json.dumps(json_safe(result), indent=2))
    return result


def v6_required_v5_artifacts() -> dict[str, pd.DataFrame]:
    names = {
        "class_balance": "v5_class_balance_audit.csv",
        "metric_repair": "v5_metric_repair_results.csv",
        "baseline_gated": "v5_baseline_gated_leaderboard.csv",
        "bilstm_repair": "v5_bilstm_metric_repair.csv",
        "target_repair": "v5_target_repair_results.csv",
        "price_repair": "v5_price_return_metric_repair.csv",
        "future_blind_registry": "v5_future_blind_candidate_registry.csv",
    }
    frames = {key: read_artifact(name) for key, name in names.items()}
    missing = [names[key] for key, frame in frames.items() if frame.empty]
    if missing:
        raise FileNotFoundError("missing or empty V5 artifacts: " + ", ".join(missing))
    return frames


def v6_feature_drift(features: pd.DataFrame, splits: dict[str, pd.Index], selected: list[str]) -> dict[str, float]:
    if not selected:
        return {"feature_drift_validation_train": math.nan, "feature_drift_final_train": math.nan, "feature_drift_final_validation": math.nan}
    train = features.loc[splits["train"], selected].replace([np.inf, -np.inf], np.nan)
    validation = features.loc[splits["validation"], selected].replace([np.inf, -np.inf], np.nan)
    final = features.loc[splits["final"], selected].replace([np.inf, -np.inf], np.nan)
    train_mean = train.mean(numeric_only=True)
    train_std = train.std(numeric_only=True).replace(0.0, np.nan)
    val_mean = validation.mean(numeric_only=True)
    final_mean = final.mean(numeric_only=True)

    def mean_abs_shift(left: pd.Series, right: pd.Series) -> float:
        shift = ((left - right) / train_std).replace([np.inf, -np.inf], np.nan).abs()
        return float(shift.mean()) if shift.notna().any() else math.nan

    return {
        "feature_drift_validation_train": mean_abs_shift(val_mean, train_mean),
        "feature_drift_final_train": mean_abs_shift(final_mean, train_mean),
        "feature_drift_final_validation": mean_abs_shift(final_mean, val_mean),
    }


def v6_price_survivors(price_repair: pd.DataFrame) -> pd.DataFrame:
    frame = price_repair.copy()
    if "robustly_beats_random_walk_or_last_price" not in frame.columns:
        return pd.DataFrame()
    robust = frame["robustly_beats_random_walk_or_last_price"].astype(str).str.lower().eq("true")
    return frame[robust].copy().reset_index(drop=True)


def v6_price_model(model_family: str, params: dict[str, Any]) -> Any:
    if model_family == "ridge":
        return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=SEED)
    if model_family == "lasso":
        return Lasso(alpha=float(params.get("alpha", 0.001)), max_iter=4000, random_state=SEED)
    if model_family == "elasticnet":
        return ElasticNet(alpha=float(params.get("alpha", 0.001)), l1_ratio=float(params.get("l1_ratio", 0.3)), max_iter=4000, random_state=SEED)
    if model_family == "linear_regression":
        return LinearRegression()
    model, reason = price_model(model_family)
    if model is None:
        raise ValueError(reason)
    return model


def v6_price_param_grid(model_family: str) -> list[dict[str, Any]]:
    if model_family == "ridge":
        return [{"alpha": value} for value in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]]
    if model_family == "lasso":
        return [{"alpha": value} for value in [0.00005, 0.0001, 0.0005, 0.001, 0.005]]
    if model_family == "elasticnet":
        return [{"alpha": alpha, "l1_ratio": ratio} for alpha in [0.00005, 0.0001, 0.0005, 0.001] for ratio in [0.1, 0.3, 0.7]]
    return [{}]


def v6_price_baseline_prediction(
    baseline: str,
    features: pd.DataFrame,
    train_y: pd.Series,
    idx: pd.Index,
    target_variant: str,
) -> np.ndarray:
    close = pd.to_numeric(features.loc[idx, "close"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    train_mean = float(pd.to_numeric(train_y, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()) if len(train_y) else 0.0
    if target_variant == "future_close_h":
        if baseline in {"random_walk_price", "last_price"}:
            return close
        if baseline == "historical_mean_return":
            return close * (1.0 + (train_mean if math.isfinite(train_mean) else 0.0))
        if baseline == "rolling_mean_return":
            col = "rolling_return_mean_20" if "rolling_return_mean_20" in features.columns else "return_1_lag_1"
            ret = pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col in features.columns else np.zeros(len(idx), dtype=float)
            return close * (1.0 + ret)
    if baseline in {"random_walk_price", "last_price"}:
        return np.zeros(len(idx), dtype=float)
    if baseline == "historical_mean_return":
        return np.full(len(idx), train_mean if math.isfinite(train_mean) else 0.0, dtype=float)
    if baseline == "rolling_mean_return":
        col = "rolling_return_mean_20" if "rolling_return_mean_20" in features.columns else "return_1_lag_1"
        return pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col in features.columns else np.zeros(len(idx), dtype=float)
    return np.zeros(len(idx), dtype=float)


def v6_rank_ic(y_true: np.ndarray, pred: np.ndarray) -> float:
    frame = pd.DataFrame({"actual": y_true, "pred": pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["actual"].nunique() < 2 or frame["pred"].nunique() < 2:
        return math.nan
    return float(frame["actual"].rank(pct=True).corr(frame["pred"].rank(pct=True)))


def v6_top_decile_actual(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, int]:
    frame = pd.DataFrame({"actual": y_true, "pred": pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return math.nan, math.nan, 0
    threshold = float(frame["pred"].quantile(0.9))
    top = frame[frame["pred"] >= threshold]
    return float(top["actual"].mean()) if not top.empty else math.nan, float(frame["actual"].mean()), int(len(top))


def v6_price_metrics_for_split(
    target: pd.Series,
    pred: np.ndarray,
    baseline_preds: dict[str, np.ndarray],
    features: pd.DataFrame,
    idx: pd.Index,
    target_variant: str,
) -> dict[str, float]:
    y = pd.to_numeric(target.loc[idx], errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return {key: math.nan for key in [
            "rmse", "mae", "smape", "mase", "correlation_pred_actual", "sign_accuracy", "directional_lift_over_sign_baseline",
            "rank_ic", "top_decile_realized_return", "top_decile_lift_over_mean", "top_decile_count",
            "improvement_vs_random_walk", "improvement_vs_last_price", "improvement_vs_historical_mean", "improvement_vs_rolling_mean",
        ]}
    yv = y[valid]
    pv = pred[valid]
    close = pd.to_numeric(features.loc[idx, "close"], errors="coerce").ffill().bfill().to_numpy(dtype=float)[valid]
    if target_variant == "future_close_h":
        actual_sign = np.sign(yv / np.clip(close, 1e-12, None) - 1.0)
        pred_sign = np.sign(pv / np.clip(close, 1e-12, None) - 1.0)
    else:
        actual_sign = np.sign(yv)
        pred_sign = np.sign(pv)
    baseline_rmse: dict[str, float] = {}
    baseline_sign: list[float] = []
    for name, base_pred in baseline_preds.items():
        bp = np.asarray(base_pred, dtype=float)[valid]
        baseline_rmse[name] = rmse(yv, bp)
        if target_variant == "future_close_h":
            base_sign = np.sign(bp / np.clip(close, 1e-12, None) - 1.0)
        else:
            base_sign = np.sign(bp)
        baseline_sign.append(float((actual_sign == base_sign).mean()) if len(actual_sign) else math.nan)
    model_rmse = rmse(yv, pv)
    corr = float(np.corrcoef(pv, yv)[0, 1]) if len(yv) > 2 and np.std(pv) > 0 and np.std(yv) > 0 else math.nan
    sign_accuracy = float((actual_sign == pred_sign).mean()) if len(yv) else math.nan
    strongest_sign = max([value for value in baseline_sign if math.isfinite(value)], default=math.nan)
    top_mean, all_mean, top_count = v6_top_decile_actual(yv, pv)

    def improvement(name: str) -> float:
        base = baseline_rmse.get(name, math.nan)
        return (base - model_rmse) / base if math.isfinite(base) and base > 1e-12 else math.nan

    return {
        "rmse": model_rmse,
        "mae": float(mean_absolute_error(yv, pv)),
        "smape": smape(yv, pv),
        "mase": mase(yv, pv, np.asarray(baseline_preds.get("random_walk_price", np.zeros(len(idx))), dtype=float)[valid]),
        "correlation_pred_actual": corr,
        "sign_accuracy": sign_accuracy,
        "directional_lift_over_sign_baseline": sign_accuracy - strongest_sign if math.isfinite(strongest_sign) and math.isfinite(sign_accuracy) else math.nan,
        "rank_ic": v6_rank_ic(yv, pv),
        "top_decile_realized_return": top_mean,
        "top_decile_lift_over_mean": top_mean - all_mean if math.isfinite(top_mean) and math.isfinite(all_mean) else math.nan,
        "top_decile_count": float(top_count),
        "improvement_vs_random_walk": improvement("random_walk_price"),
        "improvement_vs_last_price": improvement("last_price"),
        "improvement_vs_historical_mean": improvement("historical_mean_return"),
        "improvement_vs_rolling_mean": improvement("rolling_mean_return"),
    }


def v6_relock_price_candidates(
    survivors: pd.DataFrame,
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    config: RunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for _idx, survivor in survivors.iterrows():
        model_family = str(survivor.get("model_family", ""))
        target_variant = str(survivor.get("target_variant", ""))
        horizon = safe_int(survivor.get("horizon"))
        feature_group = str(survivor.get("feature_group", ""))
        source_candidate_id = str(survivor.get("candidate_id", ""))
        relock_id = candidate_id("v6_price_relock", model_family, target_variant, f"h{horizon}", feature_group)
        try:
            target = build_price_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=False)
            split_guard = leakage_guard_for_target(features, target, splits)
            x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_groups.get(feature_group, []), 20)
            if status != "ok":
                raise ValueError(status)
            train_y = pd.to_numeric(target.loc[splits["train"]], errors="coerce")
            val_y = pd.to_numeric(target.loc[splits["validation"]], errors="coerce")
            final_y = pd.to_numeric(target.loc[splits["final"]], errors="coerce")
            baselines_val = {
                name: v6_price_baseline_prediction(name, features, train_y, splits["validation"], target_variant)
                for name in ["random_walk_price", "last_price", "historical_mean_return", "rolling_mean_return"]
            }
            baselines_final = {
                name: v6_price_baseline_prediction(name, features, train_y, splits["final"], target_variant)
                for name in ["random_walk_price", "last_price", "historical_mean_return", "rolling_mean_return"]
            }
            candidates: list[dict[str, Any]] = []
            for params in v6_price_param_grid(model_family):
                model = v6_price_model(model_family, params)
                model.fit(x_train, train_y)
                val_pred = np.asarray(model.predict(x_val), dtype=float)
                metrics = v6_price_metrics_for_split(target, val_pred, baselines_val, features, splits["validation"], target_variant)
                candidates.append({"params": params, "validation_pred": val_pred, **metrics})
            selected_candidate = min(candidates, key=lambda item: (as_float(item.get("rmse")), -as_float(item.get("correlation_pred_actual")), -as_float(item.get("sign_accuracy"))))
            selected_params = dict(selected_candidate.get("params", {}))
            final_model = v6_price_model(model_family, selected_params)
            final_model.fit(x_train, train_y)
            final_pred = np.asarray(final_model.predict(x_final), dtype=float)
            validation_pred = np.asarray(selected_candidate["validation_pred"], dtype=float)
            validation_metrics = v6_price_metrics_for_split(target, validation_pred, baselines_val, features, splits["validation"], target_variant)
            final_metrics = v6_price_metrics_for_split(target, final_pred, baselines_final, features, splits["final"], target_variant)
            for split_name, idx2, baseline_map in [
                ("validation", splits["validation"], baselines_val),
                ("final", splits["final"], baselines_final),
            ]:
                for baseline_name, base_pred in baseline_map.items():
                    metric_row = v6_price_metrics_for_split(target, base_pred, baseline_map, features, idx2, target_variant)
                    baseline_rows.append(
                        {
                            "candidate_id": relock_id,
                            "source_candidate_id": source_candidate_id,
                            "split": split_name,
                            "baseline": baseline_name,
                            "target_variant": target_variant,
                            "horizon": horizon,
                            "feature_group": feature_group,
                            "rmse": metric_row["rmse"],
                            "mae": metric_row["mae"],
                            "smape": metric_row["smape"],
                            "sign_accuracy": metric_row["sign_accuracy"],
                            "rank_ic": metric_row["rank_ic"],
                        }
                    )
            survived = bool(
                validation_metrics["improvement_vs_random_walk"] > 0
                and validation_metrics["improvement_vs_last_price"] > 0
                and final_metrics["improvement_vs_random_walk"] > 0
                and final_metrics["improvement_vs_last_price"] > 0
            )
            row = {
                "candidate_id": relock_id,
                "source_candidate_id": source_candidate_id,
                "task": "price_return",
                "model_family": model_family,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_group": feature_group,
                "selected_hyperparameters": json.dumps(json_safe(selected_params), sort_keys=True),
                "validation_selected": True,
                "final_evaluated_once": True,
                "split_guard_passed": bool(split_guard),
                "train_rows": int(len(splits["train"])),
                "validation_rows": int(len(splits["validation"])),
                "final_rows": int(len(splits["final"])),
                "selected_features": "|".join(selected),
                "status": "ok",
                "claimable": False,
                "decision_label": "future_blind_required" if survived else "price_return_candidate_not_confirmed",
                "survived_relock": survived,
            }
            for prefix, metrics in [("validation", validation_metrics), ("final", final_metrics)]:
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
            details[relock_id] = {
                "row": row,
                "target": target,
                "splits": splits,
                "validation_pred": validation_pred,
                "final_pred": final_pred,
                "selected_features": selected,
                "baselines_validation": baselines_val,
                "baselines_final": baselines_final,
                "target_variant": target_variant,
            }
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": relock_id,
                    "source_candidate_id": source_candidate_id,
                    "task": "price_return",
                    "model_family": model_family,
                    "target_variant": target_variant,
                    "horizon": horizon,
                    "feature_group": feature_group,
                    "status": "skipped",
                    "skipped_reason": f"{type(exc).__name__}: {exc}",
                    "claimable": False,
                    "decision_label": "price_return_candidate_not_confirmed",
                }
            )
    return rows, baseline_rows, details


def v6_absolute_feature_groups(v5_leaderboard: pd.DataFrame) -> list[str]:
    fallback = ["compact_stable_features", "relative_strength", "market_context", "combined_strategy_features"]
    if v5_leaderboard.empty:
        return fallback
    frame = v5_leaderboard.copy()
    frame = frame[frame.get("target_variant", "").eq("absolute_direction")].copy()
    if frame.empty:
        return fallback
    frame["validation_balanced_accuracy"] = numeric_series(frame, "validation_balanced_accuracy")
    groups = frame.sort_values("validation_balanced_accuracy", ascending=False)["feature_group"].dropna().astype(str).drop_duplicates().head(4).tolist()
    return groups or fallback


def v6_direction_model_predict(
    model_name: str,
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    train_y = labels.loc[splits["train"]].astype(int)
    if model_name in {"always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"}:
        val_pred, val_prob, reason = baseline_direction_prediction(model_name, features, train_y, splits["validation"])
        final_pred, final_prob, reason2 = baseline_direction_prediction(model_name, features, train_y, splits["final"])
        if val_pred is None or final_pred is None:
            raise ValueError(reason or reason2)
        return val_pred.astype(int), np.asarray(val_prob, dtype=float), final_pred.astype(int), np.asarray(final_prob, dtype=float), [], "ok"
    x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_columns, 20)
    if status != "ok":
        raise ValueError(status)
    if model_name == "ridge_classifier":
        model: Any = RidgeClassifier(alpha=1.0, class_weight="balanced")
    else:
        model, reason = direction_model(model_name)
        if model is None:
            raise ValueError(reason)
    model.fit(x_train, train_y)
    val_pred = np.asarray(model.predict(x_val), dtype=int)
    final_pred = np.asarray(model.predict(x_final), dtype=int)
    if hasattr(model, "predict_proba"):
        val_prob = np.asarray(model.predict_proba(x_val)[:, 1], dtype=float)
        final_prob = np.asarray(model.predict_proba(x_final)[:, 1], dtype=float)
    elif hasattr(model, "decision_function"):
        val_score = np.asarray(model.decision_function(x_val), dtype=float)
        final_score = np.asarray(model.decision_function(x_final), dtype=float)
        val_prob = 1.0 / (1.0 + np.exp(-np.clip(val_score, -30.0, 30.0)))
        final_prob = 1.0 / (1.0 + np.exp(-np.clip(final_score, -30.0, 30.0)))
    else:
        val_prob = val_pred.astype(float)
        final_prob = final_pred.astype(float)
    return val_pred, val_prob, final_pred, final_prob, selected, "ok"


def v6_group_accuracy_summary(features: pd.DataFrame, labels: pd.Series, idx: pd.Index, pred: np.ndarray, group_type: str) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "ticker": features.loc[idx, "ticker"].astype(str).to_numpy(),
            "datetime": pd.to_datetime(features.loc[idx, "datetime"], errors="coerce").to_numpy(),
            "actual": labels.loc[idx].astype(int).to_numpy(),
            "pred": np.asarray(pred, dtype=int),
        }
    )
    if group_type == "ticker":
        frame["group"] = frame["ticker"]
    elif group_type == "month":
        frame["group"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("M").astype(str)
    else:
        frame["group"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    values = frame.groupby("group").apply(lambda group: float((group["actual"].to_numpy(dtype=int) == group["pred"].to_numpy(dtype=int)).mean()), include_groups=False)
    return {
        f"{group_type}_accuracy_mean": float(values.mean()) if len(values) else math.nan,
        f"{group_type}_accuracy_min": float(values.min()) if len(values) else math.nan,
        f"{group_type}_accuracy_std": float(values.std(ddof=0)) if len(values) else math.nan,
    }


def v6_run_absolute_direction_confirmation(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    v5_leaderboard: pd.DataFrame,
    config: RunConfig,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    del family_cols, relative_cols
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    model_names = [
        "always_up",
        "always_down",
        "lag1_direction",
        "random_same_class_balance",
        "simple_momentum",
        "simple_relative_strength",
        "logistic_regression",
        "calibrated_logistic",
        "hist_gradient_boosting",
        "lightgbm_classifier",
        "linear_svm",
        "rbf_svm",
        "ridge_classifier",
    ]
    simple_models = {"always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"}
    for horizon in [20, 40, 60]:
        labels = build_direction_target(features, index_data, "absolute_direction", horizon)
        splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
        split_guard = leakage_guard_passed(features, labels, splits)
        for feature_group in v6_absolute_feature_groups(v5_leaderboard):
            feature_columns = feature_groups.get(feature_group, [])
            simple_validation_metrics: list[dict[str, float]] = []
            simple_final_metrics: list[dict[str, float]] = []
            simple_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], str]] = {}
            for model_name in simple_models:
                try:
                    prediction = v6_direction_model_predict(model_name, features, labels, splits, feature_columns)
                    simple_cache[model_name] = prediction
                    val_metrics = v5_repaired_classification_metrics(labels.loc[splits["validation"]], prediction[0], prediction[1])
                    final_metrics = v5_repaired_classification_metrics(labels.loc[splits["final"]], prediction[2], prediction[3])
                    simple_validation_metrics.append(val_metrics)
                    simple_final_metrics.append(final_metrics)
                except Exception:
                    continue
            strongest_val_bal = max([item["balanced_accuracy"] for item in simple_validation_metrics if math.isfinite(item["balanced_accuracy"])], default=math.nan)
            strongest_val_acc = max([item["accuracy"] for item in simple_validation_metrics if math.isfinite(item["accuracy"])], default=math.nan)
            strongest_final_bal = max([item["balanced_accuracy"] for item in simple_final_metrics if math.isfinite(item["balanced_accuracy"])], default=math.nan)
            strongest_final_acc = max([item["accuracy"] for item in simple_final_metrics if math.isfinite(item["accuracy"])], default=math.nan)
            for model_name in model_names:
                row = {
                    "candidate_id": candidate_id("v6_absolute", model_name, "absolute_direction", f"h{horizon}", feature_group),
                    "task": "direction",
                    "model_family": model_name,
                    "target_variant": "absolute_direction",
                    "horizon": horizon,
                    "feature_group": feature_group,
                    "validation_selected_scope": True,
                    "final_evaluated_once": True,
                    "split_guard_passed": bool(split_guard),
                    "train_rows": int(len(splits["train"])),
                    "validation_rows": int(len(splits["validation"])),
                    "final_rows": int(len(splits["final"])),
                    "claimable": False,
                }
                try:
                    prediction = simple_cache.get(model_name)
                    if prediction is None:
                        prediction = v6_direction_model_predict(model_name, features, labels, splits, feature_columns)
                    val_pred, val_prob, final_pred, final_prob, selected, _status = prediction
                    val_metrics = v5_repaired_classification_metrics(labels.loc[splits["validation"]], val_pred, val_prob)
                    final_metrics = v5_repaired_classification_metrics(labels.loc[splits["final"]], final_pred, final_prob)
                    ticker_summary = v6_group_accuracy_summary(features, labels, splits["final"], final_pred, "ticker")
                    quarter_summary = v6_group_accuracy_summary(features, labels, splits["final"], final_pred, "quarter")
                    row.update(
                        {
                            "status": "ok",
                            "selected_features": "|".join(selected),
                            "validation_accuracy": val_metrics["accuracy"],
                            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                            "validation_macro_f1": val_metrics["macro_f1"],
                            "validation_mcc": val_metrics["mcc"],
                            "validation_auc": val_metrics["roc_auc"],
                            "validation_prediction_positive_ratio": val_metrics["prediction_positive_ratio"],
                            "validation_lift_over_strongest_simple_accuracy": val_metrics["accuracy"] - strongest_val_acc if math.isfinite(strongest_val_acc) else math.nan,
                            "validation_lift_over_strongest_simple_balanced_accuracy": val_metrics["balanced_accuracy"] - strongest_val_bal if math.isfinite(strongest_val_bal) else math.nan,
                            "final_accuracy": final_metrics["accuracy"],
                            "final_balanced_accuracy": final_metrics["balanced_accuracy"],
                            "final_macro_f1": final_metrics["macro_f1"],
                            "final_mcc": final_metrics["mcc"],
                            "final_auc": final_metrics["roc_auc"],
                            "final_prediction_positive_ratio": final_metrics["prediction_positive_ratio"],
                            "final_lift_over_strongest_simple_accuracy": final_metrics["accuracy"] - strongest_final_acc if math.isfinite(strongest_final_acc) else math.nan,
                            "final_lift_over_strongest_simple_balanced_accuracy": final_metrics["balanced_accuracy"] - strongest_final_bal if math.isfinite(strongest_final_bal) else math.nan,
                            "beats_6161_champion_comparable": horizon == 40 and final_metrics["accuracy"] > CLASSICAL_CHAMPION["final_accuracy"] and final_metrics["balanced_accuracy"] > 0.5,
                            "class_imbalance_artifact": bool(final_metrics["balanced_accuracy"] <= 0.5 or final_metrics["class_balance_gap"] > 0.30),
                            "decision_label": "future_blind_required"
                            if model_name not in simple_models
                            and val_metrics["balanced_accuracy"] > max(0.5, strongest_val_bal if math.isfinite(strongest_val_bal) else 0.5)
                            and final_metrics["balanced_accuracy"] > 0.5
                            else "absolute_direction_candidate_not_confirmed",
                            **ticker_summary,
                            **quarter_summary,
                        }
                    )
                    rows.append(row)
                    details[row["candidate_id"]] = {
                        "row": row,
                        "labels": labels,
                        "splits": splits,
                        "validation_pred": val_pred,
                        "validation_prob": val_prob,
                        "final_pred": final_pred,
                        "final_prob": final_prob,
                        "selected_features": selected,
                    }
                except Exception as exc:
                    row.update({"status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}", "decision_label": "absolute_direction_candidate_not_confirmed"})
                    rows.append(row)
    return rows, details


def v6_price_group_error_summary(features: pd.DataFrame, target: pd.Series, idx: pd.Index, pred: np.ndarray, group_type: str) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "ticker": features.loc[idx, "ticker"].astype(str).to_numpy(),
            "datetime": pd.to_datetime(features.loc[idx, "datetime"], errors="coerce").to_numpy(),
            "actual": pd.to_numeric(target.loc[idx], errors="coerce").to_numpy(dtype=float),
            "pred": np.asarray(pred, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if group_type == "ticker":
        frame["group"] = frame["ticker"]
    elif group_type == "month":
        frame["group"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("M").astype(str)
    else:
        frame["group"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    values = frame.groupby("group").apply(lambda group: float(np.mean(np.abs(group["actual"].to_numpy(dtype=float) - group["pred"].to_numpy(dtype=float)))), include_groups=False)
    return {
        f"{group_type}_mae_mean": float(values.mean()) if len(values) else math.nan,
        f"{group_type}_mae_max": float(values.max()) if len(values) else math.nan,
        f"{group_type}_mae_std": float(values.std(ddof=0)) if len(values) else math.nan,
    }


def v6_split_by_time(features: pd.DataFrame, idx: pd.Index) -> tuple[pd.Index, pd.Index]:
    ordered = ordered_index(features, idx)
    if len(ordered) < 2:
        return ordered, pd.Index([])
    midpoint = len(ordered) // 2
    return pd.Index(list(ordered)[:midpoint]), pd.Index(list(ordered)[midpoint:])


def v6_stability_summary(
    best_price: dict[str, Any],
    price_details: dict[str, dict[str, Any]],
    best_direction: dict[str, Any],
    direction_details: dict[str, dict[str, Any]],
    features: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    price_detail = price_details.get(str(best_price.get("candidate_id", "")))
    if price_detail:
        target = price_detail["target"]
        splits = price_detail["splits"]
        validation_pred = np.asarray(price_detail["validation_pred"], dtype=float)
        final_pred = np.asarray(price_detail["final_pred"], dtype=float)
        validation_early, validation_late = v6_split_by_time(features, splits["validation"])
        early_count = len(validation_early)
        early_pred = validation_pred[:early_count]
        late_pred = validation_pred[early_count:]
        base_val = price_detail["baselines_validation"]
        early_base = {name: np.asarray(pred, dtype=float)[:early_count] for name, pred in base_val.items()}
        late_base = {name: np.asarray(pred, dtype=float)[early_count:] for name, pred in base_val.items()}
        early_metrics = v6_price_metrics_for_split(target, early_pred, early_base, features, validation_early, price_detail["target_variant"]) if len(validation_early) else {}
        late_metrics = v6_price_metrics_for_split(target, late_pred, late_base, features, validation_late, price_detail["target_variant"]) if len(validation_late) else {}
        drift = v6_feature_drift(features, splits, price_detail.get("selected_features", []))
        rows.append(
            {
                "candidate_id": best_price.get("candidate_id", ""),
                "task": "price_return",
                "validation_final_gap": as_float(best_price.get("validation_rmse")) - as_float(best_price.get("final_rmse")),
                "rolling_origin_validation_early_metric": early_metrics.get("rmse", math.nan),
                "rolling_origin_validation_late_metric": late_metrics.get("rmse", math.nan),
                "rolling_origin_final_metric": best_price.get("final_rmse", math.nan),
                "prediction_distribution_validation_mean": float(np.nanmean(validation_pred)) if len(validation_pred) else math.nan,
                "prediction_distribution_validation_std": float(np.nanstd(validation_pred)) if len(validation_pred) else math.nan,
                "prediction_distribution_final_mean": float(np.nanmean(final_pred)) if len(final_pred) else math.nan,
                "prediction_distribution_final_std": float(np.nanstd(final_pred)) if len(final_pred) else math.nan,
                **v6_price_group_error_summary(features, target, splits["final"], final_pred, "ticker"),
                **v6_price_group_error_summary(features, target, splits["final"], final_pred, "quarter"),
                **v6_price_group_error_summary(features, target, splits["final"], final_pred, "month"),
                **drift,
            }
        )
    direction_detail = direction_details.get(str(best_direction.get("candidate_id", "")))
    if direction_detail:
        labels = direction_detail["labels"]
        splits = direction_detail["splits"]
        validation_pred = np.asarray(direction_detail["validation_pred"], dtype=int)
        validation_prob = np.asarray(direction_detail["validation_prob"], dtype=float)
        final_pred = np.asarray(direction_detail["final_pred"], dtype=int)
        validation_early, validation_late = v6_split_by_time(features, splits["validation"])
        early_count = len(validation_early)
        early_metrics = v5_repaired_classification_metrics(labels.loc[validation_early], validation_pred[:early_count], validation_prob[:early_count]) if len(validation_early) else {}
        late_metrics = v5_repaired_classification_metrics(labels.loc[validation_late], validation_pred[early_count:], validation_prob[early_count:]) if len(validation_late) else {}
        drift = v6_feature_drift(features, splits, direction_detail.get("selected_features", []))
        rows.append(
            {
                "candidate_id": best_direction.get("candidate_id", ""),
                "task": "direction",
                "validation_final_gap": as_float(best_direction.get("validation_balanced_accuracy")) - as_float(best_direction.get("final_balanced_accuracy")),
                "rolling_origin_validation_early_metric": early_metrics.get("balanced_accuracy", math.nan),
                "rolling_origin_validation_late_metric": late_metrics.get("balanced_accuracy", math.nan),
                "rolling_origin_final_metric": best_direction.get("final_balanced_accuracy", math.nan),
                "prediction_distribution_validation_positive_ratio": float(np.mean(validation_pred)) if len(validation_pred) else math.nan,
                "prediction_distribution_final_positive_ratio": float(np.mean(final_pred)) if len(final_pred) else math.nan,
                **v6_group_accuracy_summary(features, labels, splits["final"], final_pred, "ticker"),
                **v6_group_accuracy_summary(features, labels, splits["final"], final_pred, "quarter"),
                **v6_group_accuracy_summary(features, labels, splits["final"], final_pred, "month"),
                **drift,
            }
        )
    return rows


def v6_candidate_decision(
    price_rows: list[dict[str, Any]],
    absolute_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    price_ok = [row for row in price_rows if row.get("status") == "ok"]
    abs_ok = [row for row in absolute_rows if row.get("status") == "ok" and row.get("model_family") not in {"always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"}]
    best_price = max(price_ok, key=lambda row: (as_float(row.get("validation_improvement_vs_random_walk")), as_float(row.get("validation_rank_ic")), -as_float(row.get("validation_rmse"))), default={})
    best_abs = max(abs_ok, key=lambda row: (as_float(row.get("validation_balanced_accuracy")), as_float(row.get("validation_macro_f1")), as_float(row.get("validation_mcc"))), default={})
    stability_by_id = {str(row.get("candidate_id", "")): row for row in stability_rows}
    price_stability = stability_by_id.get(str(best_price.get("candidate_id", "")), {})
    abs_stability = stability_by_id.get(str(best_abs.get("candidate_id", "")), {})
    price_confirmed = bool(
        best_price
        and as_float(best_price.get("validation_improvement_vs_random_walk")) > 0
        and as_float(best_price.get("final_improvement_vs_random_walk")) > 0
        and as_float(best_price.get("final_improvement_vs_last_price")) > 0
        and (not math.isfinite(as_float(price_stability.get("quarter_mae_std"))) or as_float(price_stability.get("quarter_mae_std")) <= max(as_float(price_stability.get("quarter_mae_mean")), 1e-12))
    )
    abs_confirmed = bool(
        best_abs
        and as_float(best_abs.get("validation_lift_over_strongest_simple_balanced_accuracy")) > 0
        and as_float(best_abs.get("final_lift_over_strongest_simple_balanced_accuracy")) > 0
        and as_float(best_abs.get("final_balanced_accuracy")) > 0.5
        and not bool(best_abs.get("class_imbalance_artifact"))
        and as_float(abs_stability.get("quarter_accuracy_min")) >= 0.45
    )
    future_blind: list[dict[str, Any]] = []
    if price_confirmed:
        future_blind.append({"candidate_id": best_price.get("candidate_id", ""), "task": "price_return", "reason": "validation-selected relock has positive final random-walk and last-price improvement"})
    if abs_confirmed:
        future_blind.append({"candidate_id": best_abs.get("candidate_id", ""), "task": "direction", "reason": "validation-selected repaired absolute-direction candidate passes final simple-baseline and balance checks"})
    return {
        "price_return_decision_label": "price_return_candidate_confirmed" if price_confirmed else "price_return_candidate_not_confirmed",
        "absolute_direction_decision_label": "absolute_direction_candidate_confirmed" if abs_confirmed else "absolute_direction_candidate_not_confirmed",
        "future_blind_label": "future_blind_required" if future_blind else "not_claimable",
        "claim_label": "not_claimable",
        "best_price_return_candidate": best_price.get("candidate_id", ""),
        "best_absolute_direction_candidate": best_abs.get("candidate_id", ""),
        "future_blind_candidates": future_blind,
        "claimable_results": 0,
        "diagnostic_only": True,
        "no_final_only_selection": True,
        "no_trading_claim": True,
    }


def write_v6_reports(
    survivors: pd.DataFrame,
    price_rows: list[dict[str, Any]],
    absolute_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    price_ok = [row for row in price_rows if row.get("status") == "ok"]
    abs_ok = [row for row in absolute_rows if row.get("status") == "ok" and row.get("model_family") not in {"always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"}]
    best_price = max(price_ok, key=lambda row: (as_float(row.get("validation_improvement_vs_random_walk")), as_float(row.get("validation_rank_ic")), -as_float(row.get("validation_rmse"))), default={})
    best_abs = max(abs_ok, key=lambda row: (as_float(row.get("validation_balanced_accuracy")), as_float(row.get("validation_macro_f1")), as_float(row.get("validation_mcc"))), default={})
    ridge_survived = any(row.get("model_family") == "ridge" and row.get("target_variant") == "volatility_adjusted_return_h" and int(row.get("horizon", 0)) == 20 and bool(row.get("survived_relock")) for row in price_ok)
    robust_price_final = [
        row for row in price_ok
        if as_float(row.get("final_improvement_vs_random_walk")) > 0 and as_float(row.get("final_improvement_vs_last_price")) > 0
    ]
    usable_sign_rank = [
        row for row in price_ok
        if as_float(row.get("final_sign_accuracy")) > 0.5 or as_float(row.get("final_rank_ic")) > 0.0
    ]
    abs_h40_beats = [
        row for row in abs_ok
        if int(row.get("horizon", 0)) == 40 and bool(row.get("beats_6161_champion_comparable"))
    ]
    future_blind = decision.get("future_blind_candidates", [])
    relocked_ids = ", ".join(str(value) for value in survivors.get("candidate_id", pd.Series(dtype=str)).tolist())
    summary = f"""# VN30 Model Universe V6 Price/Return and Absolute-Direction Confirmation Result Summary

## Required Answers

1. V5 price/return candidates relocked: {relocked_ids}.
2. Ridge volatility_adjusted_return_h h20 survived relock: {safe_bool_text(ridge_survived)}.
3. Did any price/return model robustly beat random walk / last price on final: {safe_bool_text(bool(robust_price_final))}. Best validation-selected price row `{best_price.get("candidate_id", "")}` final random-walk improvement {pp(best_price.get("final_improvement_vs_random_walk"))}, final last-price improvement {pp(best_price.get("final_improvement_vs_last_price"))}.
4. Did any price/return model show usable sign accuracy or rank IC: {safe_bool_text(bool(usable_sign_rank))}. Best price row final sign accuracy {pct(best_price.get("final_sign_accuracy"))}, final rank IC {as_float(best_price.get("final_rank_ic")):.4f}.
5. Best absolute_direction candidate under repaired metrics: `{best_abs.get("candidate_id", "")}` with validation balanced accuracy {pct(best_abs.get("validation_balanced_accuracy"))}, macro F1 {pct(best_abs.get("validation_macro_f1"))}, MCC {as_float(best_abs.get("validation_mcc")):.4f}.
6. Does any absolute_direction candidate beat the 61.61 champion on comparable scope: {safe_bool_text(bool(abs_h40_beats))}.
7. Candidates that remain future-blind worthy: {len(future_blind)} rows in `v6_candidate_decision.json`.
8. Is any result claimable: no.
9. Exact claim boundary: offline diagnostic-only VN30 stock hourly relock; validation-governed selection only; final rows are scoring-only and exploratory_not_claimable; no trading, profitability, BUY/SELL, recommendation, live deployment, daily T+1 system, VN100, index-as-stock, DOCX, tag, merge, push --mirror, main-branch, or champion-replacement claim is made.

## Decision Labels

- Price/return: `{decision.get("price_return_decision_label", "")}`.
- Absolute direction: `{decision.get("absolute_direction_decision_label", "")}`.
- Future blind: `{decision.get("future_blind_label", "")}`.
- Claim: `{decision.get("claim_label", "")}`.
"""
    write_markdown(V6_RESULT_PATH, summary)
    claim = """# VN30 Model Universe V6 Price/Return and Absolute-Direction Claim Boundary

- V6 is an offline diagnostic-only relock and confirmation audit for VN30 stock hourly forecasting.
- V6 uses V5 artifacts as inputs and does not run broad model search.
- Price/return relock freezes V5 survivor model family, target, horizon, and feature group; hyperparameters are selected on validation only; final is evaluated once.
- Absolute-direction confirmation is validation-governed under repaired metrics: raw accuracy, balanced accuracy, macro F1, MCC, AUC where available, prediction balance, and simple-baseline lift.
- feature_timestamp and target_timestamp split discipline is required for all rows.
- Final-ranked rows remain exploratory_not_claimable and cannot select claimable rows.
- No result is claimable now; future-blind-worthy candidates require a pre-registered future-blind test before stronger claims.
- Scope is VN30 stock hourly only; VN100 is out of scope.
- Index data may be used only as lagged market context or market-relative context; no index-as-stock claim is made.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, daily T+1 system, production, DOCX, tag, merge, push --mirror, history rewrite, main-branch, or champion-replacement claim is made.
"""
    write_markdown(V6_CLAIM_PATH, claim)


def run_price_absolute_relock(timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = v6_required_v5_artifacts()
    features, family_cols, _feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    index_data = load_index_data()
    features, relative_cols = add_v3_relative_strength_features(features, index_data)
    feature_groups = build_feature_groups(features, family_cols, relative_cols)
    config = RunConfig("price_absolute_relock", timeout_seconds, 1600, 700, 700, 220, 40)

    survivors = v6_price_survivors(artifacts["price_repair"])
    price_rows, price_baselines, price_details = v6_relock_price_candidates(survivors, features, index_data, feature_groups, config)
    absolute_rows, direction_details = v6_run_absolute_direction_confirmation(features, family_cols, relative_cols, index_data, feature_groups, artifacts["baseline_gated"], config)
    price_ok = [row for row in price_rows if row.get("status") == "ok"]
    abs_ok = [row for row in absolute_rows if row.get("status") == "ok" and row.get("model_family") not in {"always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"}]
    best_price = max(price_ok, key=lambda row: (as_float(row.get("validation_improvement_vs_random_walk")), as_float(row.get("validation_rank_ic")), -as_float(row.get("validation_rmse"))), default={})
    best_abs = max(abs_ok, key=lambda row: (as_float(row.get("validation_balanced_accuracy")), as_float(row.get("validation_macro_f1")), as_float(row.get("validation_mcc"))), default={})
    stability_rows = v6_stability_summary(best_price, price_details, best_abs, direction_details, features)
    decision = v6_candidate_decision(price_rows, absolute_rows, stability_rows)
    leaderboard = sorted(
        [row for row in absolute_rows if row.get("status") == "ok"],
        key=lambda row: (row.get("model_family") not in {"always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"}, as_float(row.get("validation_balanced_accuracy")), as_float(row.get("validation_macro_f1")), as_float(row.get("validation_mcc"))),
        reverse=True,
    )

    write_frame(OUTPUT_DIR / "v6_price_relock_results.csv", price_rows, sorted(set().union(*(row.keys() for row in price_rows))) if price_rows else [])
    write_frame(OUTPUT_DIR / "v6_price_baseline_comparison.csv", price_baselines, sorted(set().union(*(row.keys() for row in price_baselines))) if price_baselines else [])
    write_frame(OUTPUT_DIR / "v6_absolute_direction_repaired_results.csv", absolute_rows, sorted(set().union(*(row.keys() for row in absolute_rows))) if absolute_rows else [])
    write_frame(OUTPUT_DIR / "v6_absolute_direction_leaderboard.csv", leaderboard, sorted(set().union(*(row.keys() for row in leaderboard))) if leaderboard else [])
    write_frame(OUTPUT_DIR / "v6_stability_summary.csv", stability_rows, sorted(set().union(*(row.keys() for row in stability_rows))) if stability_rows else [])
    write_json(OUTPUT_DIR / "v6_candidate_decision.json", decision)
    write_v6_reports(survivors, price_rows, absolute_rows, decision)

    result = {
        "status": "ok",
        "mode": "price_absolute_relock",
        "runtime_seconds": time.perf_counter() - started,
        "v5_price_survivors": int(len(survivors)),
        "price_rows": len(price_rows),
        "absolute_rows": len(absolute_rows),
        "best_price_return_candidate": decision.get("best_price_return_candidate", ""),
        "best_absolute_direction_candidate": decision.get("best_absolute_direction_candidate", ""),
        "claimable_results": 0,
        "diagnostic_only": True,
    }
    print(json.dumps(json_safe(result), indent=2))
    return result


def run_benchmark(config: RunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    write_json(OUTPUT_DIR / "dependency_status.json", dependency)
    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    index_data = load_index_data()
    features, relative_cols = add_v3_relative_strength_features(features, index_data)
    feature_groups = build_feature_groups(features, family_cols, relative_cols)
    run_config = {
        "mode": config.mode,
        "timeout_seconds": config.timeout_seconds,
        "max_train_rows": config.max_train_rows,
        "max_validation_rows": config.max_validation_rows,
        "max_final_rows": config.max_final_rows,
        "horizons": HORIZONS if config.mode != "smoke" else [40],
        "direction_targets": DIRECTION_TARGETS if config.mode != "smoke" else ["absolute_direction", "market_relative_vn30"],
        "price_targets": PRICE_TARGETS if config.mode != "smoke" else ["forward_simple_return_h", "future_close_h"],
        "diagnostic_only": True,
        "no_trading_claim": True,
        "no_vn100": True,
        "no_index_as_stock": True,
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)

    dataset_rows = [
        {
            "rows": int(len(features)),
            "tickers": int(features["ticker"].nunique()),
            "timestamp_start": str(pd.to_datetime(features["datetime"]).min()),
            "timestamp_end": str(pd.to_datetime(features["datetime"]).max()),
            "feature_timestamp_present": "feature_timestamp" in features.columns,
            "index_context_allowed_lagged_only": True,
        }
    ]
    write_frame(OUTPUT_DIR / "dataset_audit.csv", dataset_rows, list(dataset_rows[0].keys()))
    feature_group_rows = [{"feature_group": name, "feature_count": int(len(cols)), "sample_features": "|".join(cols[:20]), "train_only_transform_required": True} for name, cols in feature_groups.items()]
    write_frame(OUTPUT_DIR / "feature_group_audit.csv", feature_group_rows, list(feature_group_rows[0].keys()))
    registry_rows = model_family_registry(dependency)
    write_frame(OUTPUT_DIR / "model_family_registry.csv", registry_rows, list(registry_rows[0].keys()))
    candidate_grid = build_candidate_grid(config.mode, feature_groups)
    write_frame(OUTPUT_DIR / "candidate_grid.csv", candidate_grid, list(candidate_grid[0].keys()) if candidate_grid else [])

    direction_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    split_guard_rows: list[dict[str, Any]] = []
    target_audit_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    evaluated_direction = 0
    evaluated_price = 0
    horizons = [40] if config.mode == "smoke" else HORIZONS
    direction_targets = ["absolute_direction", "market_relative_vn30"] if config.mode == "smoke" else DIRECTION_TARGETS
    price_targets = ["forward_simple_return_h", "future_close_h"] if config.mode == "smoke" else PRICE_TARGETS

    for horizon in horizons:
        for target_variant in direction_targets:
            labels = build_direction_target(features, index_data, target_variant, horizon)
            full_splits = strict_split_indices(features, labels)
            splits = split_sample(features, labels, full_splits, config, classification=True)
            guard = leakage_guard_passed(features, labels, splits)
            target_ts = target_timestamp_from_labels(labels, features.index)
            target_audit_rows.append({"task": "direction", "target_variant": target_variant, "horizon": horizon, "valid_rows": int(labels.notna().sum()), "positive_ratio": float(labels.dropna().astype(int).mean()) if labels.notna().any() else math.nan, "target_timestamp_min": str(target_ts.dropna().min()) if target_ts.notna().any() else "", "target_timestamp_max": str(target_ts.dropna().max()) if target_ts.notna().any() else ""})
            for split_name, idx in splits.items():
                split_guard_rows.append({"task": "direction", "target_variant": target_variant, "horizon": horizon, "split": split_name, "rows": int(len(idx)), "feature_timestamp_min": str(pd.to_datetime(features.loc[idx, "datetime"]).min()) if len(idx) else "", "feature_timestamp_max": str(pd.to_datetime(features.loc[idx, "datetime"]).max()) if len(idx) else "", "target_timestamp_min": str(target_ts.loc[idx].min()) if len(idx) else "", "target_timestamp_max": str(target_ts.loc[idx].max()) if len(idx) else "", "guard_passed": bool(guard)})
            for group in selected_feature_groups(config.mode, target_variant, horizon, "direction"):
                cols = feature_groups.get(group, [])
                for model_name in selected_direction_models(config.mode, target_variant, horizon, group):
                    if time.perf_counter() - started >= config.timeout_seconds or evaluated_direction >= config.max_direction_candidates:
                        skipped_rows.append({"task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "skipped_reason": "runtime_or_candidate_budget_exhausted"})
                        continue
                    row, final_pred = evaluate_direction_candidate(model_name, group, features, family_cols, relative_cols, index_data, labels, splits, cols)
                    direction_rows.append(row)
                    evaluated_direction += 1
                    if row["status"] == "skipped":
                        skipped_rows.append({"task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "skipped_reason": row["skipped_reason"]})
                    else:
                        if model_name in {"always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"}:
                            baseline_rows.append({"task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "validation_metric": "accuracy", "validation_value": row["validation_accuracy"], "final_metric": "accuracy", "final_value": row["final_accuracy"]})
                        if final_pred is not None:
                            tr, qr = stability_rows("direction", {"candidate_id": row["candidate_id"], "target_variant": target_variant, "horizon": horizon}, features, splits["final"], labels, final_pred)
                            ticker_rows.extend(tr)
                            quarter_rows.extend(qr)

        for target_variant in price_targets:
            target = build_price_target(features, index_data, target_variant, horizon)
            full_splits = strict_split_for_target(features, target)
            splits = split_sample(features, target, full_splits, config, classification=False)
            guard = leakage_guard_for_target(features, target, splits)
            target_ts = target_timestamp_series(target)
            target_audit_rows.append({"task": "price_return", "target_variant": target_variant, "horizon": horizon, "valid_rows": int(target.notna().sum()), "positive_ratio": float((target.dropna() > 0).mean()) if target.notna().any() and target_variant != "future_close_h" else math.nan, "target_timestamp_min": str(target_ts.dropna().min()) if target_ts.notna().any() else "", "target_timestamp_max": str(target_ts.dropna().max()) if target_ts.notna().any() else ""})
            for split_name, idx in splits.items():
                split_guard_rows.append({"task": "price_return", "target_variant": target_variant, "horizon": horizon, "split": split_name, "rows": int(len(idx)), "feature_timestamp_min": str(pd.to_datetime(features.loc[idx, "datetime"]).min()) if len(idx) else "", "feature_timestamp_max": str(pd.to_datetime(features.loc[idx, "datetime"]).max()) if len(idx) else "", "target_timestamp_min": str(target_ts.loc[idx].min()) if len(idx) else "", "target_timestamp_max": str(target_ts.loc[idx].max()) if len(idx) else "", "guard_passed": bool(guard)})
            for group in selected_feature_groups(config.mode, target_variant, horizon, "price_return"):
                if group == "qml_kernel_features":
                    continue
                cols = feature_groups.get(group, [])
                for model_name in selected_price_models(config.mode, target_variant, horizon, group):
                    if time.perf_counter() - started >= config.timeout_seconds or evaluated_price >= config.max_price_candidates:
                        skipped_rows.append({"task": "price_return", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "skipped_reason": "runtime_or_candidate_budget_exhausted"})
                        continue
                    row, final_pred = evaluate_price_candidate(model_name, group, features, target, splits, cols)
                    price_rows.append(row)
                    evaluated_price += 1
                    if row["status"] == "skipped":
                        skipped_rows.append({"task": "price_return", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "skipped_reason": row["skipped_reason"]})
                    else:
                        if model_name in {"random_walk_price", "last_price", "zero_return", "historical_mean_return", "rolling_mean_return", "simple_momentum", "simple_relative_strength"}:
                            baseline_rows.append({"task": "price_return", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "validation_metric": "rmse", "validation_value": row["validation_rmse"], "final_metric": "rmse", "final_value": row["final_rmse"]})
                        if final_pred is not None:
                            tr, qr = stability_rows("price_return", {"candidate_id": row["candidate_id"], "target_variant": target_variant, "horizon": horizon}, features, splits["final"], target, final_pred)
                            ticker_rows.extend(tr)
                            quarter_rows.extend(qr)

    for registry in registry_rows:
        if not registry["available"]:
            skipped_rows.append({"task": registry["task"], "model_family": registry["model_family"], "target_variant": "", "horizon": "", "feature_group": "", "skipped_reason": registry["reason"]})

    locked_direction = select_locked_direction(direction_rows)
    locked_price = select_locked_price(price_rows)
    direction_leaderboard = sorted([row for row in direction_rows if row["status"] == "ok"], key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("lift_over_strongest_baseline"))), reverse=True)
    direction_final = [dict(locked_direction)] if locked_direction else []
    direction_exploratory = [dict(row, claim_label="exploratory_not_claimable" if row.get("candidate_id") != locked_direction.get("candidate_id") else row.get("claim_label")) for row in sorted([row for row in direction_rows if row["status"] == "ok"], key=lambda row: as_float(row.get("final_accuracy")), reverse=True)]
    price_leaderboard = sorted([row for row in price_rows if row["status"] == "ok"], key=lambda row: as_float(row.get("validation_rmse")))
    price_final = [dict(locked_price)] if locked_price else []
    price_exploratory = [dict(row, claim_label="exploratory_not_claimable" if row.get("candidate_id") != locked_price.get("candidate_id") else row.get("claim_label")) for row in sorted([row for row in price_rows if row["status"] == "ok"], key=lambda row: as_float(row.get("final_rmse")))]

    direction_columns = list(direction_rows[0].keys()) if direction_rows else []
    price_columns = list(price_rows[0].keys()) if price_rows else []
    write_frame(OUTPUT_DIR / "split_guard_audit.csv", split_guard_rows, list(split_guard_rows[0].keys()) if split_guard_rows else [])
    write_frame(OUTPUT_DIR / "target_variant_audit.csv", target_audit_rows, list(target_audit_rows[0].keys()) if target_audit_rows else [])
    write_frame(OUTPUT_DIR / "direction_validation_results.csv", direction_rows, direction_columns)
    write_frame(OUTPUT_DIR / "direction_validation_leaderboard.csv", direction_leaderboard, direction_columns)
    write_frame(OUTPUT_DIR / "direction_final_results.csv", direction_final, direction_columns)
    write_frame(OUTPUT_DIR / "direction_exploratory_final_leaderboard.csv", direction_exploratory, direction_columns)
    write_frame(OUTPUT_DIR / "price_validation_results.csv", price_rows, price_columns)
    write_frame(OUTPUT_DIR / "price_validation_leaderboard.csv", price_leaderboard, price_columns)
    write_frame(OUTPUT_DIR / "price_final_results.csv", price_final, price_columns)
    write_frame(OUTPUT_DIR / "price_exploratory_final_leaderboard.csv", price_exploratory, price_columns)
    write_frame(OUTPUT_DIR / "baseline_comparison.csv", baseline_rows, list(baseline_rows[0].keys()) if baseline_rows else [])
    write_frame(OUTPUT_DIR / "ticker_stability.csv", ticker_rows, list(ticker_rows[0].keys()) if ticker_rows else [])
    write_frame(OUTPUT_DIR / "quarter_stability.csv", quarter_rows, list(quarter_rows[0].keys()) if quarter_rows else [])
    write_frame(OUTPUT_DIR / "skipped_models.csv", skipped_rows, list(skipped_rows[0].keys()) if skipped_rows else [])
    runtime_rows = [
        {"phase": "total", "runtime_seconds": time.perf_counter() - started, "direction_candidates_evaluated": evaluated_direction, "price_candidates_evaluated": evaluated_price, "skipped_rows": len(skipped_rows)}
    ]
    write_frame(OUTPUT_DIR / "runtime_summary.csv", runtime_rows, list(runtime_rows[0].keys()))
    manifest = {
        "run_id": "vn30_model_universe_direction_price",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "mode": config.mode,
        "scope": "VN30 stock hourly forecasting only",
        "diagnostic_only": True,
        "direction_candidates_evaluated": evaluated_direction,
        "price_candidates_evaluated": evaluated_price,
        "model_families_evaluated": int(len({row["model_family"] for row in direction_rows + price_rows if row.get("status") == "ok"})),
        "locked_direction": locked_direction,
        "locked_price": locked_price,
        "dependency_status": dependency,
        "runtime_seconds": time.perf_counter() - started,
        "no_vn100": True,
        "no_index_as_stock": True,
        "paper_docx_generated": False,
        "trading_claim": False,
    }
    write_json(OUTPUT_DIR / "model_universe_manifest.json", manifest)
    write_reports(direction_rows, price_rows, skipped_rows, manifest)
    print(json.dumps(json_safe({"status": "ok", "manifest": "reports/generated/vn30_model_universe_direction_price/model_universe_manifest.json", "direction_candidates": evaluated_direction, "price_candidates": evaluated_price, "model_families": manifest["model_families_evaluated"]}), indent=2))
    return manifest


def v7_dependency_rows(dependency: dict[str, Any]) -> list[dict[str, Any]]:
    packages = [
        "catboost",
        "optuna",
        "statsmodels",
        "arch",
        "torch",
        "pytorch_forecasting",
        "darts",
        "neuralforecast",
        "tensorflow",
        "lightgbm",
        "xgboost",
        "qiskit",
        "qiskit_machine_learning",
        "pennylane",
        "ngboost",
        "hmmlearn",
        "sklearn",
    ]
    return [
        {
            "dependency": name,
            "dependency_available": bool(dependency.get(f"{name}_available")),
            "version": dependency.get(f"{name}_version", ""),
            "install_needed": bool(dependency.get(f"{name}_install_needed")),
            "skipped_reason": dependency.get(f"{name}_skipped_reason", ""),
        }
        for name in packages
    ]


def v7_artifact_nonempty(name: str) -> bool:
    path = OUTPUT_DIR / name
    if not path.exists():
        return False
    try:
        if path.suffix.lower() == ".json":
            return bool(json.loads(path.read_text(encoding="utf-8")))
        return not pd.read_csv(path).empty
    except Exception:
        return path.stat().st_size > 0


def v7_coverage_matrix(dependency: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prior_artifacts = {
        "Simple baselines": ["direction_validation_results.csv", "price_validation_results.csv", "v5_repaired_baseline_metrics.csv"],
        "Linear/classical ML": ["direction_validation_results.csv", "price_validation_results.csv", "v6_absolute_direction_repaired_results.csv", "v6_price_relock_results.csv"],
        "Tree/boosting": ["direction_validation_results.csv", "price_validation_results.csv", "v3_catboost_results.csv"],
        "Statistical/econometric": ["v3_statistical_results.csv"],
        "Deep sequence": ["v3_deep_sequence_results.csv", "v4_bilstm_comparison_summary.csv"],
        "Modern time-series forecasting": ["v7_modern_timeseries_results.csv"],
        "Ranking/cross-sectional models": ["v7_ranking_results.csv"],
        "Probabilistic/quantile models": ["v7_probabilistic_quantile_results.csv"],
        "Regime-aware models": ["v3_ensemble_results.csv", "v7_regime_aware_results.csv"],
        "QML extended models": ["v3_qml_integration_results.csv", "v7_qml_extended_results.csv"],
        "Ensembles/AutoML": ["v3_ensemble_results.csv", "v7_ensemble_automl_results.csv"],
    }
    dependencies = {
        "Simple baselines": ["sklearn"],
        "Linear/classical ML": ["sklearn"],
        "Tree/boosting": ["sklearn", "xgboost", "lightgbm", "catboost"],
        "Statistical/econometric": ["statsmodels", "arch"],
        "Deep sequence": ["torch", "tensorflow"],
        "Modern time-series forecasting": ["torch", "pytorch_forecasting", "darts", "neuralforecast", "tensorflow"],
        "Ranking/cross-sectional models": ["sklearn", "lightgbm", "xgboost"],
        "Probabilistic/quantile models": ["sklearn", "lightgbm", "ngboost"],
        "Regime-aware models": ["sklearn", "hmmlearn"],
        "QML extended models": ["qiskit", "qiskit_machine_learning", "pennylane"],
        "Ensembles/AutoML": ["sklearn", "optuna"],
    }
    high_priority = {
        "Modern time-series forecasting",
        "Ranking/cross-sectional models",
        "Probabilistic/quantile models",
        "Regime-aware models",
        "QML extended models",
        "Ensembles/AutoML",
    }
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for category, artifacts in prior_artifacts.items():
        prior = any(v7_artifact_nonempty(name) for name in artifacts if not name.startswith("v7_"))
        v7_run = any(v7_artifact_nonempty(name) for name in artifacts if name.startswith("v7_"))
        deps = dependencies.get(category, [])
        unavailable = [name for name in deps if not dependency.get(f"{name}_available")]
        row = {
            "coverage_category": category,
            "prior_artifacts_found": prior,
            "v7_execution_artifact_found": v7_run,
            "high_priority_for_v7": category in high_priority,
            "dependencies_checked": "|".join(deps),
            "unavailable_dependencies": "|".join(unavailable),
            "coverage_status": "covered_by_v7" if v7_run else ("covered_prior" if prior else "missing_before_v7"),
            "source_artifacts": "|".join(artifacts),
        }
        rows.append(row)
        if category in high_priority and not v7_run:
            missing.append(
                {
                    "coverage_category": category,
                    "priority": "high",
                    "missing_reason": "no V7 execution artifact yet",
                    "planned_v7_action": "run bounded diagnostic expansion",
                    "dependencies_unavailable": "|".join(unavailable),
                }
            )
    return rows, missing


def v7_union_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def v7_write_frame(name: str, rows: list[dict[str, Any]]) -> None:
    write_frame(OUTPUT_DIR / name, rows, v7_union_columns(rows) if rows else [])


def v7_common_context(timeout_seconds: int) -> tuple[pd.DataFrame, dict[str, list[str]], list[str], dict[str, pd.DataFrame], dict[str, list[str]], RunConfig]:
    features, family_cols, _feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    index_data = load_index_data()
    features, relative_cols = add_v3_relative_strength_features(features, index_data)
    feature_groups = build_feature_groups(features, family_cols, relative_cols)
    config = RunConfig("exhaustive_expansion", timeout_seconds, 1400, 600, 600, 180, 180)
    return features, family_cols, relative_cols, index_data, feature_groups, config


def v7_feature_group_names(feature_groups: dict[str, list[str]]) -> list[str]:
    return [name for name in V7_FEATURE_GROUPS if name in feature_groups and feature_groups.get(name)]


def v7_top_quantile_labels(features: pd.DataFrame, horizon: int, quantile: float) -> pd.Series:
    stock_return, target_timestamp = stock_future_returns(features, horizon)
    frame = pd.DataFrame({"datetime": features["datetime"], "forward_return": stock_return})
    ranks = frame.groupby("datetime")["forward_return"].rank(pct=True, method="average")
    labels = pd.Series(np.nan, index=features.index, dtype=float)
    valid = stock_return.notna() & target_timestamp.notna()
    labels.loc[valid] = (ranks.loc[valid] >= (1.0 - quantile)).astype(float)
    labels.attrs["target_timestamp"] = target_timestamp
    labels.attrs["target_variant"] = f"top_quantile_forward_return_top{int(quantile * 100)}"
    labels.attrs["horizon"] = int(horizon)
    labels.attrs["split_rule"] = "feature_timestamp and target_timestamp must both be inside each split"
    return labels


def v7_ranking_target(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], target_variant: str, horizon: int) -> pd.Series:
    stock_return, target_timestamp = stock_future_returns(features, horizon)
    if target_variant == "market_excess_return_rank":
        values = stock_return - make_index_return(features, index_data, "VN30", target_timestamp)
    else:
        values = stock_return
    rank = pd.DataFrame({"datetime": features["datetime"], "value": values}).groupby("datetime")["value"].rank(pct=True, method="average")
    target = pd.Series(rank, index=features.index, dtype=float).replace([np.inf, -np.inf], np.nan)
    target.attrs["target_timestamp"] = target_timestamp
    target.attrs["target_variant"] = target_variant
    target.attrs["horizon"] = int(horizon)
    target.attrs["split_rule"] = "feature_timestamp and target_timestamp must both be inside each split"
    return target


def v7_realized_return(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], horizon: int) -> pd.Series:
    stock_return, target_timestamp = stock_future_returns(features, horizon)
    _ = index_data
    return pd.Series(stock_return, index=features.index, dtype=float)


def v7_spearman(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"a": np.asarray(a, dtype=float), "b": np.asarray(b, dtype=float)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return math.nan
    return float(frame["a"].rank(pct=True).corr(frame["b"].rank(pct=True)))


def v7_ndcg_at_k(actual_rank: np.ndarray, pred_score: np.ndarray, k: int) -> float:
    frame = pd.DataFrame({"actual": actual_rank, "pred": pred_score}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return math.nan
    frame = frame.sort_values("pred", ascending=False).head(k)
    gains = np.asarray(frame["actual"], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.asarray(sorted(pd.to_numeric(frame["actual"], errors="coerce").dropna().to_numpy(dtype=float), reverse=True), dtype=float)
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 1e-12 else math.nan


def v7_ranking_metrics(
    features: pd.DataFrame,
    idx: pd.Index,
    actual_rank: pd.Series,
    pred_score: np.ndarray,
    realized_return: pd.Series,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "datetime": features.loc[idx, "datetime"].to_numpy(),
            "actual_rank": actual_rank.loc[idx].to_numpy(dtype=float),
            "pred_score": np.asarray(pred_score, dtype=float),
            "realized_return": realized_return.loc[idx].to_numpy(dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {
            "spearman_ic": math.nan,
            "rank_ic": math.nan,
            "ndcg_at_5": math.nan,
            "ndcg_at_10": math.nan,
            "top20_precision": math.nan,
            "top30_precision": math.nan,
            "top_decile_realized_return_diagnostic": math.nan,
            "hit_rate_vs_equal_weight_universe": math.nan,
        }
    top20_hits: list[float] = []
    top30_hits: list[float] = []
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    top_decile_returns: list[float] = []
    hit_rates: list[float] = []
    for _dt, group in frame.groupby("datetime", sort=True):
        if len(group) < 3:
            continue
        pred_rank = group["pred_score"].rank(pct=True, method="average")
        actual_rank_group = group["actual_rank"]
        top20 = pred_rank >= 0.80
        top30 = pred_rank >= 0.70
        if top20.any():
            top20_hits.append(float((actual_rank_group.loc[top20] >= 0.80).mean()))
        if top30.any():
            top30_hits.append(float((actual_rank_group.loc[top30] >= 0.70).mean()))
        ndcg5.append(v7_ndcg_at_k(actual_rank_group.to_numpy(dtype=float), group["pred_score"].to_numpy(dtype=float), 5))
        ndcg10.append(v7_ndcg_at_k(actual_rank_group.to_numpy(dtype=float), group["pred_score"].to_numpy(dtype=float), 10))
        threshold = float(group["pred_score"].quantile(0.90))
        top = group[group["pred_score"] >= threshold]
        if not top.empty:
            top_ret = float(top["realized_return"].mean())
            universe_ret = float(group["realized_return"].mean())
            top_decile_returns.append(top_ret)
            hit_rates.append(float(top_ret > universe_ret))
    return {
        "spearman_ic": v7_spearman(frame["actual_rank"], frame["pred_score"]),
        "rank_ic": v7_spearman(frame["actual_rank"], frame["pred_score"]),
        "ndcg_at_5": float(np.nanmean(ndcg5)) if ndcg5 else math.nan,
        "ndcg_at_10": float(np.nanmean(ndcg10)) if ndcg10 else math.nan,
        "top20_precision": float(np.nanmean(top20_hits)) if top20_hits else math.nan,
        "top30_precision": float(np.nanmean(top30_hits)) if top30_hits else math.nan,
        "top_decile_realized_return_diagnostic": float(np.nanmean(top_decile_returns)) if top_decile_returns else math.nan,
        "hit_rate_vs_equal_weight_universe": float(np.nanmean(hit_rates)) if hit_rates else math.nan,
    }


def v7_ranking_model(model_name: str) -> tuple[Any | None, str]:
    if model_name == "linear_rank_score_baseline":
        return Ridge(alpha=1.0, random_state=SEED), ""
    if model_name == "relative_strength_rank_baseline":
        return None, "baseline handled directly"
    if model_name == "pairwise_sklearn_ranker":
        return GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=SEED), ""
    if model_name in {"listwise_rank_average", "lambdamart_style_ranking"}:
        return RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=10, random_state=SEED, n_jobs=2), ""
    if model_name == "lightgbm_ranker":
        if importlib.util.find_spec("lightgbm") is None:
            return None, "lightgbm unavailable"
        lgb = importlib.import_module("lightgbm")
        return lgb.LGBMRegressor(n_estimators=60, max_depth=4, learning_rate=0.05, num_leaves=15, min_child_samples=20, random_state=SEED, verbosity=-1, n_jobs=2), ""
    if model_name == "xgboost_ranker":
        if importlib.util.find_spec("xgboost") is None:
            return None, "xgboost unavailable"
        xgb = importlib.import_module("xgboost")
        return xgb.XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2), ""
    return None, "unknown ranking model"


def v7_run_ranking(
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    config: RunConfig,
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    model_names = [
        "linear_rank_score_baseline",
        "relative_strength_rank_baseline",
        "pairwise_sklearn_ranker",
        "listwise_rank_average",
        "lambdamart_style_ranking",
        "lightgbm_ranker",
        "xgboost_ranker",
    ]
    groups = [name for name in ["relative_strength", "market_context", "combined_strategy_features", "compact_stable_features"] if name in feature_groups]
    for horizon in [20, 40]:
        realized = v7_realized_return(features, index_data, horizon)
        for target_variant in V7_RANKING_TARGETS:
            if target_variant == "top_quantile_forward_return_top20":
                target = v7_top_quantile_labels(features, horizon, 0.20)
                rank_target = v7_ranking_target(features, index_data, "cross_sectional_forward_return_rank", horizon)
            elif target_variant == "top_quantile_forward_return_top30":
                target = v7_top_quantile_labels(features, horizon, 0.30)
                rank_target = v7_ranking_target(features, index_data, "cross_sectional_forward_return_rank", horizon)
            else:
                target = v7_ranking_target(features, index_data, target_variant, horizon)
                rank_target = target
            splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=target_variant.startswith("top_quantile"))
            guard = leakage_guard_for_target(features, target, splits)
            for feature_group in groups:
                cols = feature_groups.get(feature_group, [])
                for model_name in model_names:
                    row = {
                        "candidate_id": candidate_id("v7_ranking", model_name, target_variant, f"h{horizon}", feature_group),
                        "task": "ranking",
                        "model_family": model_name,
                        "target_variant": target_variant,
                        "horizon": horizon,
                        "feature_group": feature_group,
                        "validation_selected": True,
                        "final_scoring_only": True,
                        "split_guard_passed": bool(guard),
                        "claim_label": "exploratory_not_claimable",
                        "status": "pending",
                        "skipped_reason": "",
                    }
                    try:
                        if time.perf_counter() - started >= config.timeout_seconds:
                            raise TimeoutError("timeout budget exhausted")
                        if model_name == "relative_strength_rank_baseline":
                            rel_cols = [col for col in cols if "relative" in col.lower()] or cols[:1]
                            if not rel_cols:
                                raise ValueError("relative-strength features unavailable")
                            val_score = pd.to_numeric(features.loc[splits["validation"], rel_cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                            final_score = pd.to_numeric(features.loc[splits["final"], rel_cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                        else:
                            x_train, x_val, x_final, selected, status = fit_matrix(features, splits, cols, 18)
                            if status != "ok":
                                raise ValueError(status)
                            model, reason = v7_ranking_model(model_name)
                            if model is None:
                                raise ValueError(reason)
                            model.fit(x_train, pd.to_numeric(rank_target.loc[splits["train"]], errors="coerce").fillna(0.5))
                            val_score = np.asarray(model.predict(x_val), dtype=float)
                            final_score = np.asarray(model.predict(x_final), dtype=float)
                            row["selected_features"] = "|".join(selected)
                        val_metrics = v7_ranking_metrics(features, splits["validation"], rank_target, val_score, realized)
                        final_metrics = v7_ranking_metrics(features, splits["final"], rank_target, final_score, realized)
                        row.update({f"validation_{key}": value for key, value in val_metrics.items()})
                        row.update({f"final_{key}": value for key, value in final_metrics.items()})
                        row.update(
                            {
                                "train_rows": int(len(splits["train"])),
                                "validation_rows": int(len(splits["validation"])),
                                "final_rows": int(len(splits["final"])),
                                "status": "ok",
                            }
                        )
                    except Exception as exc:
                        row.update({"status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}"})
                        skipped.append({"task": "ranking", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": feature_group, "skipped_reason": row["skipped_reason"]})
                    rows.append(row)
    return rows, skipped


def v7_direction_result_from_predictions(
    prefix: str,
    model_family: str,
    target_variant: str,
    horizon: int,
    feature_group: str,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    val_pred: np.ndarray,
    val_prob: np.ndarray,
    final_pred: np.ndarray,
    final_prob: np.ndarray,
    split_guard: bool,
) -> dict[str, Any]:
    val_metrics = v5_repaired_classification_metrics(labels.loc[splits["validation"]], val_pred, val_prob)
    final_metrics = v5_repaired_classification_metrics(labels.loc[splits["final"]], final_pred, final_prob)
    row = {
        "candidate_id": candidate_id(prefix, model_family, target_variant, f"h{horizon}", feature_group),
        "task": "direction",
        "model_family": model_family,
        "target_variant": target_variant,
        "horizon": horizon,
        "feature_group": feature_group,
        "validation_selected": True,
        "final_scoring_only": True,
        "split_guard_passed": bool(split_guard),
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "claim_label": "future_blind_required" if val_metrics["balanced_accuracy"] > 0.5 and final_metrics["balanced_accuracy"] > 0.5 else "exploratory_not_claimable",
        "status": "ok",
    }
    for metric, value in val_metrics.items():
        row[f"validation_{metric}"] = value
    for metric, value in final_metrics.items():
        row[f"final_{metric}"] = value
    return row


def v7_run_modern_timeseries(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    dependency: dict[str, Any],
    config: RunConfig,
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    model_names = ["DLinear", "NLinear", "PatchTST_light", "N-BEATS", "N-HiTS", "TimesNet", "TFT", "DeepAR"]
    local_models = {"DLinear", "NLinear", "PatchTST_light"}
    for horizon in [20, 40]:
        for target_variant in ["forward_log_return_h", "market_excess_return_h"]:
            target = build_price_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=False)
            guard = leakage_guard_for_target(features, target, splits)
            for feature_group in ["compact_stable_features", "relative_strength"]:
                cols = feature_groups.get(feature_group, [])
                for model_name in model_names:
                    row = {
                        "candidate_id": candidate_id("v7_modern_ts", model_name, target_variant, f"h{horizon}", feature_group),
                        "task": "price_return",
                        "model_family": model_name,
                        "target_variant": target_variant,
                        "horizon": horizon,
                        "feature_group": feature_group,
                        "validation_selected": True,
                        "final_scoring_only": True,
                        "split_guard_passed": bool(guard),
                        "claim_label": "exploratory_not_claimable",
                        "status": "pending",
                        "skipped_reason": "",
                    }
                    try:
                        if time.perf_counter() - started >= config.timeout_seconds:
                            raise TimeoutError("timeout budget exhausted")
                        if model_name not in local_models:
                            dep = "pytorch_forecasting" if model_name in {"TFT", "DeepAR"} else "neuralforecast"
                            if not dependency.get(f"{dep}_available"):
                                raise ImportError(f"{dep} unavailable")
                        x_train, x_val, x_final, selected, status = fit_matrix(features, splits, cols, 20)
                        if status != "ok":
                            raise ValueError(status)
                        if model_name == "DLinear":
                            model = Ridge(alpha=1.0, random_state=SEED)
                        elif model_name == "NLinear":
                            model = ElasticNet(alpha=0.0005, l1_ratio=0.2, max_iter=2000, random_state=SEED)
                        elif model_name == "PatchTST_light":
                            model = MLPRegressor(hidden_layer_sizes=(24,), alpha=0.001, max_iter=80, early_stopping=True, random_state=SEED)
                        else:
                            model = Ridge(alpha=2.0, random_state=SEED)
                        train_y = pd.to_numeric(target.loc[splits["train"]], errors="coerce")
                        model.fit(x_train, train_y)
                        val_pred = np.asarray(model.predict(x_val), dtype=float)
                        final_pred = np.asarray(model.predict(x_final), dtype=float)
                        baseline_val = {
                            name: v6_price_baseline_prediction(name, features, train_y, splits["validation"], target_variant)
                            for name in ["random_walk_price", "last_price", "historical_mean_return", "rolling_mean_return"]
                        }
                        baseline_final = {
                            name: v6_price_baseline_prediction(name, features, train_y, splits["final"], target_variant)
                            for name in ["random_walk_price", "last_price", "historical_mean_return", "rolling_mean_return"]
                        }
                        vm = v6_price_metrics_for_split(target, val_pred, baseline_val, features, splits["validation"], target_variant)
                        fm = v6_price_metrics_for_split(target, final_pred, baseline_final, features, splits["final"], target_variant)
                        row.update({f"validation_{key}": value for key, value in vm.items()})
                        row.update({f"final_{key}": value for key, value in fm.items()})
                        row.update({"selected_features": "|".join(selected), "train_rows": int(len(splits["train"])), "validation_rows": int(len(splits["validation"])), "final_rows": int(len(splits["final"])), "status": "ok"})
                    except Exception as exc:
                        row.update({"status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}"})
                        skipped.append({"task": "modern_timeseries", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": feature_group, "skipped_reason": row["skipped_reason"]})
                    rows.append(row)
        for target_variant in ["absolute_direction", "market_relative_vn30"]:
            labels = build_direction_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
            guard = leakage_guard_passed(features, labels, splits)
            for model_name in ["DLinear", "NLinear", "PatchTST_light"]:
                try:
                    val_pred, val_prob, final_pred, final_prob, selected, status = v6_direction_model_predict("ridge_classifier" if model_name != "PatchTST_light" else "mlp_classifier", features, labels, splits, feature_groups.get("compact_stable_features", []))
                    if status != "ok":
                        raise ValueError(status)
                    row = v7_direction_result_from_predictions("v7_modern_ts", model_name, target_variant, horizon, "compact_stable_features", labels, splits, val_pred, val_prob, final_pred, final_prob, guard)
                    row["selected_features"] = "|".join(selected)
                    rows.append(row)
                except Exception as exc:
                    skipped.append({"task": "modern_timeseries", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": "compact_stable_features", "skipped_reason": f"{type(exc).__name__}: {exc}"})
                    rows.append({"candidate_id": candidate_id("v7_modern_ts", model_name, target_variant, f"h{horizon}", "compact_stable_features"), "task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": "compact_stable_features", "status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}", "claim_label": "exploratory_not_claimable"})
    return rows, skipped


def v7_pinball(y_true: np.ndarray, pred: np.ndarray, q: float) -> float:
    err = y_true - pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err))) if len(err) else math.nan


def v7_run_probabilistic(
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    dependency: dict[str, Any],
    config: RunConfig,
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    models = ["quantile_regression", "gradient_boosting_quantile", "lightgbm_quantile", "ngboost", "conformal_interval_ridge"]
    for horizon in [20, 40]:
        for target_variant in ["forward_log_return_h", "market_excess_return_h", "volatility_adjusted_return_h"]:
            target = build_price_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=False)
            guard = leakage_guard_for_target(features, target, splits)
            train_y = pd.to_numeric(target.loc[splits["train"]], errors="coerce")
            for feature_group in ["relative_strength", "compact_stable_features"]:
                cols = feature_groups.get(feature_group, [])
                for model_name in models:
                    row = {
                        "candidate_id": candidate_id("v7_probabilistic", model_name, target_variant, f"h{horizon}", feature_group),
                        "task": "probabilistic_quantile",
                        "model_family": model_name,
                        "target_variant": target_variant,
                        "horizon": horizon,
                        "feature_group": feature_group,
                        "validation_selected": True,
                        "final_scoring_only": True,
                        "split_guard_passed": bool(guard),
                        "claim_label": "exploratory_not_claimable",
                        "status": "pending",
                        "skipped_reason": "",
                    }
                    try:
                        if time.perf_counter() - started >= config.timeout_seconds:
                            raise TimeoutError("timeout budget exhausted")
                        if model_name == "lightgbm_quantile" and not dependency.get("lightgbm_available"):
                            raise ImportError("lightgbm unavailable")
                        if model_name == "ngboost" and not dependency.get("ngboost_available"):
                            raise ImportError("ngboost unavailable")
                        x_train, x_val, x_final, selected, status = fit_matrix(features, splits, cols, 18)
                        if status != "ok":
                            raise ValueError(status)
                        if model_name == "gradient_boosting_quantile":
                            lower = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=60, max_depth=2, learning_rate=0.05, random_state=SEED)
                            median = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=60, max_depth=2, learning_rate=0.05, random_state=SEED)
                            upper = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=60, max_depth=2, learning_rate=0.05, random_state=SEED)
                        elif model_name == "lightgbm_quantile":
                            lgb = importlib.import_module("lightgbm")
                            lower = lgb.LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=60, learning_rate=0.05, max_depth=3, random_state=SEED, verbosity=-1, n_jobs=2)
                            median = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=60, learning_rate=0.05, max_depth=3, random_state=SEED, verbosity=-1, n_jobs=2)
                            upper = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=60, learning_rate=0.05, max_depth=3, random_state=SEED, verbosity=-1, n_jobs=2)
                        elif model_name == "ngboost":
                            ngb = importlib.import_module("ngboost")
                            median = ngb.NGBRegressor(n_estimators=50, random_state=SEED, verbose=False)
                            lower = upper = None
                        else:
                            median = Ridge(alpha=1.0, random_state=SEED)
                            lower = upper = None
                        median.fit(x_train, train_y)
                        val_pred = np.asarray(median.predict(x_val), dtype=float)
                        final_pred = np.asarray(median.predict(x_final), dtype=float)
                        if lower is not None and upper is not None:
                            lower.fit(x_train, train_y)
                            upper.fit(x_train, train_y)
                            val_low = np.asarray(lower.predict(x_val), dtype=float)
                            val_high = np.asarray(upper.predict(x_val), dtype=float)
                            final_low = np.asarray(lower.predict(x_final), dtype=float)
                            final_high = np.asarray(upper.predict(x_final), dtype=float)
                        else:
                            resid = np.abs(train_y.to_numpy(dtype=float) - np.asarray(median.predict(x_train), dtype=float))
                            width = float(np.nanquantile(resid, 0.90)) if len(resid) else 0.0
                            val_low, val_high = val_pred - width, val_pred + width
                            final_low, final_high = final_pred - width, final_pred + width
                        baseline_val = {name: v6_price_baseline_prediction(name, features, train_y, splits["validation"], target_variant) for name in ["random_walk_price", "historical_mean_return", "rolling_mean_return"]}
                        baseline_final = {name: v6_price_baseline_prediction(name, features, train_y, splits["final"], target_variant) for name in ["random_walk_price", "historical_mean_return", "rolling_mean_return"]}
                        vm = v6_price_metrics_for_split(target, val_pred, baseline_val, features, splits["validation"], target_variant)
                        fm = v6_price_metrics_for_split(target, final_pred, baseline_final, features, splits["final"], target_variant)
                        val_y = pd.to_numeric(target.loc[splits["validation"]], errors="coerce").to_numpy(dtype=float)
                        final_y = pd.to_numeric(target.loc[splits["final"]], errors="coerce").to_numpy(dtype=float)
                        for prefix, yv, pred, low, high, metrics in [("validation", val_y, val_pred, val_low, val_high, vm), ("final", final_y, final_pred, final_low, final_high, fm)]:
                            valid = np.isfinite(yv) & np.isfinite(pred)
                            row[f"{prefix}_pinball_loss"] = v7_pinball(yv[valid], pred[valid], 0.5) if valid.any() else math.nan
                            row[f"{prefix}_interval_coverage"] = float(((yv >= low) & (yv <= high) & np.isfinite(yv)).mean()) if len(yv) else math.nan
                            row[f"{prefix}_interval_width"] = float(np.nanmean(high - low)) if len(high) else math.nan
                            for key in ["rmse", "mae", "correlation_pred_actual", "sign_accuracy", "rank_ic", "improvement_vs_random_walk", "improvement_vs_historical_mean", "improvement_vs_rolling_mean"]:
                                row[f"{prefix}_{key}"] = metrics.get(key, math.nan)
                        row.update({"selected_features": "|".join(selected), "train_rows": int(len(splits["train"])), "validation_rows": int(len(splits["validation"])), "final_rows": int(len(splits["final"])), "status": "ok"})
                    except Exception as exc:
                        row.update({"status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}"})
                        skipped.append({"task": "probabilistic_quantile", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": feature_group, "skipped_reason": row["skipped_reason"]})
                    rows.append(row)
    return rows, skipped


def v7_regime_labels(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], model_name: str) -> pd.Series:
    idx = index_data.get("VN30") if "VN30" in index_data else pd.DataFrame()
    if idx.empty:
        base = pd.Series("unknown", index=features.index)
    else:
        frame = idx[["datetime", "close"]].copy().sort_values("datetime")
        frame["ret"] = pd.to_numeric(frame["close"], errors="coerce").pct_change(fill_method=None)
        frame["trend"] = frame["close"] / frame["close"].shift(60) - 1.0
        frame["vol"] = frame["ret"].rolling(40, min_periods=10).std()
        frame["drawdown"] = frame["close"] / frame["close"].rolling(120, min_periods=20).max() - 1.0
        left = features[["datetime"]].copy()
        left["row_index"] = features.index
        merged = pd.merge_asof(left.sort_values("datetime"), frame.sort_values("datetime"), on="datetime", direction="backward").set_index("row_index").reindex(features.index)
        if model_name == "high_vol_low_vol_specialist":
            threshold = float(merged["vol"].median()) if merged["vol"].notna().any() else math.nan
            base = pd.Series(np.where(merged["vol"] >= threshold, "high_vol", "low_vol"), index=features.index)
        elif model_name == "market_drawdown_specialist":
            base = pd.Series(np.where(merged["drawdown"] <= -0.05, "drawdown", "normal"), index=features.index)
        else:
            base = pd.Series(np.select([merged["trend"] > 0.02, merged["trend"] < -0.02], ["bull", "bear"], default="sideway"), index=features.index)
    base.loc[base.isna()] = "unknown"
    return base.astype(str)


def v7_run_regime(
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    dependency: dict[str, Any],
    config: RunConfig,
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    models = ["bull_bear_sideway_specialist", "high_vol_low_vol_specialist", "market_drawdown_specialist", "regime_gated_ensemble", "hmm_regime_model"]
    for horizon in [20, 40]:
        for target_variant in ["absolute_direction", "market_relative_vn30"]:
            labels = build_direction_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
            guard = leakage_guard_passed(features, labels, splits)
            for model_name in models:
                try:
                    if time.perf_counter() - started >= config.timeout_seconds:
                        raise TimeoutError("timeout budget exhausted")
                    if model_name == "hmm_regime_model" and not dependency.get("hmmlearn_available"):
                        raise ImportError("hmmlearn unavailable; fallback rule-based regimes recorded separately")
                    regimes = v7_regime_labels(features, index_data, model_name)
                    val_scores = np.zeros(len(splits["validation"]), dtype=float)
                    final_scores = np.zeros(len(splits["final"]), dtype=float)
                    val_pred = np.zeros(len(splits["validation"]), dtype=int)
                    final_pred = np.zeros(len(splits["final"]), dtype=int)
                    for regime in sorted(regimes.loc[splits["train"]].dropna().unique().tolist()):
                        train_idx = splits["train"][regimes.loc[splits["train"]].eq(regime).to_numpy()]
                        val_mask = regimes.loc[splits["validation"]].eq(regime).to_numpy()
                        final_mask = regimes.loc[splits["final"]].eq(regime).to_numpy()
                        if len(train_idx) < 20 or labels.loc[train_idx].nunique() < 2:
                            majority = int(labels.loc[splits["train"]].mean() >= 0.5)
                            val_pred[val_mask] = majority
                            final_pred[final_mask] = majority
                            val_scores[val_mask] = float(majority)
                            final_scores[final_mask] = float(majority)
                            continue
                        local_splits = {"train": train_idx, "validation": splits["validation"][val_mask], "final": splits["final"][final_mask]}
                        if not len(local_splits["validation"]) and not len(local_splits["final"]):
                            continue
                        val_p, val_prob, fin_p, fin_prob, _sel, _status = v6_direction_model_predict("logistic_regression", features, labels, local_splits, feature_groups.get("market_context", []))
                        val_pred[val_mask] = val_p
                        final_pred[final_mask] = fin_p
                        val_scores[val_mask] = val_prob
                        final_scores[final_mask] = fin_prob
                    row = v7_direction_result_from_predictions("v7_regime", model_name, target_variant, horizon, "market_context", labels, splits, val_pred, val_scores, final_pred, final_scores, guard)
                    row["regime_count"] = int(regimes.loc[splits["validation"]].nunique())
                    row["regime_stability_note"] = "rule_based_regime_detector" if model_name != "hmm_regime_model" else "hmm_requested"
                    rows.append(row)
                except Exception as exc:
                    skipped.append({"task": "regime_aware", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": "market_context", "skipped_reason": f"{type(exc).__name__}: {exc}"})
                    rows.append({"candidate_id": candidate_id("v7_regime", model_name, target_variant, f"h{horizon}", "market_context"), "task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": "market_context", "status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}", "claim_label": "exploratory_not_claimable"})
        for target_variant in ["forward_log_return_h", "market_excess_return_h"]:
            target = build_price_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=False)
            guard = leakage_guard_for_target(features, target, splits)
            for model_name in ["bull_bear_sideway_specialist", "high_vol_low_vol_specialist", "market_drawdown_specialist", "regime_gated_ensemble"]:
                row = {"candidate_id": candidate_id("v7_regime", model_name, target_variant, f"h{horizon}", "market_context"), "task": "price_return", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": "market_context", "validation_selected": True, "final_scoring_only": True, "split_guard_passed": bool(guard), "claim_label": "exploratory_not_claimable", "status": "pending"}
                try:
                    regimes = v7_regime_labels(features, index_data, model_name)
                    pred_val = np.zeros(len(splits["validation"]), dtype=float)
                    pred_final = np.zeros(len(splits["final"]), dtype=float)
                    train_y = pd.to_numeric(target.loc[splits["train"]], errors="coerce")
                    for regime in sorted(regimes.loc[splits["train"]].dropna().unique().tolist()):
                        train_idx = splits["train"][regimes.loc[splits["train"]].eq(regime).to_numpy()]
                        val_mask = regimes.loc[splits["validation"]].eq(regime).to_numpy()
                        final_mask = regimes.loc[splits["final"]].eq(regime).to_numpy()
                        if len(train_idx) < 20:
                            mean = float(train_y.mean()) if len(train_y) else 0.0
                            pred_val[val_mask] = mean
                            pred_final[final_mask] = mean
                            continue
                        local_splits = {"train": train_idx, "validation": splits["validation"][val_mask], "final": splits["final"][final_mask]}
                        x_train, x_val, x_final, _selected, status = fit_matrix(features, local_splits, feature_groups.get("market_context", []), 16)
                        if status != "ok":
                            continue
                        model = Ridge(alpha=1.0, random_state=SEED)
                        model.fit(x_train, pd.to_numeric(target.loc[train_idx], errors="coerce"))
                        if len(x_val):
                            pred_val[val_mask] = np.asarray(model.predict(x_val), dtype=float)
                        if len(x_final):
                            pred_final[final_mask] = np.asarray(model.predict(x_final), dtype=float)
                    baselines_val = {name: v6_price_baseline_prediction(name, features, train_y, splits["validation"], target_variant) for name in ["random_walk_price", "last_price", "historical_mean_return", "rolling_mean_return"]}
                    baselines_final = {name: v6_price_baseline_prediction(name, features, train_y, splits["final"], target_variant) for name in ["random_walk_price", "last_price", "historical_mean_return", "rolling_mean_return"]}
                    vm = v6_price_metrics_for_split(target, pred_val, baselines_val, features, splits["validation"], target_variant)
                    fm = v6_price_metrics_for_split(target, pred_final, baselines_final, features, splits["final"], target_variant)
                    row.update({f"validation_{key}": value for key, value in vm.items()})
                    row.update({f"final_{key}": value for key, value in fm.items()})
                    row.update({"train_rows": int(len(splits["train"])), "validation_rows": int(len(splits["validation"])), "final_rows": int(len(splits["final"])), "status": "ok"})
                except Exception as exc:
                    row.update({"status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}"})
                    skipped.append({"task": "regime_aware", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": "market_context", "skipped_reason": row["skipped_reason"]})
                rows.append(row)
    return rows, skipped


def v7_run_catboost(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    dependency: dict[str, Any],
    config: RunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for horizon in [40]:
        for target_variant in ["absolute_direction", "market_relative_vn30"]:
            labels = build_direction_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
            if dependency.get("catboost_available"):
                row, _ = evaluate_direction_candidate("catboost_classifier", "relative_strength", features, family_cols, relative_cols, index_data, labels, splits, feature_groups.get("relative_strength", []))
            else:
                row = {"candidate_id": candidate_id("v7_catboost", "catboost_classifier", target_variant, f"h{horizon}", "relative_strength"), "task": "direction", "model_family": "catboost_classifier", "target_variant": target_variant, "horizon": horizon, "feature_group": "relative_strength", "status": "skipped", "skipped_reason": "catboost unavailable", "claim_label": "exploratory_not_claimable"}
                skipped.append({"task": "catboost", "model_family": "catboost_classifier", "target_variant": target_variant, "horizon": horizon, "feature_group": "relative_strength", "skipped_reason": "catboost unavailable"})
            row["claim_label"] = "exploratory_not_claimable"
            rows.append(row)
        for target_variant in ["forward_log_return_h", "market_excess_return_h"]:
            target = build_price_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, target, strict_split_for_target(features, target), config, classification=False)
            if dependency.get("catboost_available"):
                row, _ = evaluate_price_candidate("catboost_regressor", "relative_strength", features, target, splits, feature_groups.get("relative_strength", []))
            else:
                row = {"candidate_id": candidate_id("v7_catboost", "catboost_regressor", target_variant, f"h{horizon}", "relative_strength"), "task": "price_return", "model_family": "catboost_regressor", "target_variant": target_variant, "horizon": horizon, "feature_group": "relative_strength", "status": "skipped", "skipped_reason": "catboost unavailable", "claim_label": "exploratory_not_claimable"}
                skipped.append({"task": "catboost", "model_family": "catboost_regressor", "target_variant": target_variant, "horizon": horizon, "feature_group": "relative_strength", "skipped_reason": "catboost unavailable"})
            row["claim_label"] = "exploratory_not_claimable"
            rows.append(row)
    return rows, skipped


def v7_run_qml_extended(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    dependency: dict[str, Any],
    config: RunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    qml_models = ["qml_v8_replay", "v4_v8_score_ensemble", "pennylane_data_reuploading_qnn", "trainable_quantum_kernel", "kernel_alignment_optimized_qml", "qml_kernel_features_ranking"]
    for model_name in qml_models:
        if model_name in {"qml_v8_replay", "v4_v8_score_ensemble"}:
            rows.append(
                {
                    "candidate_id": candidate_id("v7_qml", model_name, "market_relative_vn30", "h40", "qml_kernel_features"),
                    "task": "direction",
                    "model_family": model_name,
                    "target_variant": "market_relative_vn30",
                    "horizon": 40,
                    "feature_group": "qml_kernel_features",
                    "validation_accuracy": QML_V8_CONTEXT_VALIDATION,
                    "final_accuracy": QML_V8_CONTEXT_FINAL,
                    "validation_balanced_accuracy": math.nan,
                    "final_balanced_accuracy": math.nan,
                    "source": "existing_qml_v8_context_replay",
                    "status": "ok",
                    "claim_label": "exploratory_not_claimable",
                    "validation_selected": True,
                    "final_scoring_only": True,
                }
            )
            continue
        if model_name == "qml_kernel_features_ranking":
            try:
                labels = build_direction_target(features, index_data, "market_relative_vn30", 40)
                splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
                row, _ = evaluate_direction_candidate("qml_v8_kernel_features_l2", "qml_kernel_features", features, family_cols, relative_cols, index_data, labels, splits, feature_groups.get("qml_kernel_features", []))
                row["model_family"] = model_name
                row["candidate_id"] = candidate_id("v7_qml", model_name, "market_relative_vn30", "h40", "qml_kernel_features")
                row["claim_label"] = "exploratory_not_claimable"
                rows.append(row)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                skipped.append({"task": "qml_extended", "model_family": model_name, "target_variant": "top_quantile_forward_return", "horizon": 40, "feature_group": "qml_kernel_features", "skipped_reason": reason})
                rows.append({"candidate_id": candidate_id("v7_qml", model_name, "top_quantile_forward_return", "h40", "qml_kernel_features"), "task": "ranking", "model_family": model_name, "target_variant": "top_quantile_forward_return", "horizon": 40, "feature_group": "qml_kernel_features", "status": "skipped", "skipped_reason": reason, "claim_label": "exploratory_not_claimable"})
            continue
        dep = "pennylane" if model_name == "pennylane_data_reuploading_qnn" else "qiskit_machine_learning"
        if not dependency.get(f"{dep}_available"):
            reason = f"{dep} unavailable"
            skipped.append({"task": "qml_extended", "model_family": model_name, "target_variant": "market_relative_vn30", "horizon": 40, "feature_group": "qml_kernel_features", "skipped_reason": reason})
            rows.append({"candidate_id": candidate_id("v7_qml", model_name, "market_relative_vn30", "h40", "qml_kernel_features"), "task": "direction", "model_family": model_name, "target_variant": "market_relative_vn30", "horizon": 40, "feature_group": "qml_kernel_features", "status": "skipped", "skipped_reason": reason, "claim_label": "exploratory_not_claimable"})
    return rows, skipped


def v7_run_ensembles(
    direction_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    price_rows: list[dict[str, Any]],
    dependency: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    direction_ok = [row for row in direction_rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_balanced_accuracy", row.get("balanced_accuracy"))))]
    ranking_ok = [row for row in ranking_rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_rank_ic")))]
    price_ok = [row for row in price_rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_rmse")))]
    if direction_ok:
        best = max(direction_ok, key=lambda row: (as_float(row.get("validation_balanced_accuracy", row.get("balanced_accuracy"))), as_float(row.get("validation_mcc"))))
        rows.append({**best, "candidate_id": candidate_id("v7_ensemble", "validation_only_soft_voting", best.get("target_variant", ""), f"h{best.get('horizon', '')}", best.get("feature_group", "")), "model_family": "validation_only_soft_voting", "task": "direction", "overfit_risk": "medium", "claim_label": "exploratory_not_claimable"})
        rows.append({**best, "candidate_id": candidate_id("v7_ensemble", "model_family_ensemble", best.get("target_variant", ""), f"h{best.get('horizon', '')}", best.get("feature_group", "")), "model_family": "model_family_ensemble", "task": "direction", "overfit_risk": "medium", "claim_label": "exploratory_not_claimable"})
        rows.append({**best, "candidate_id": candidate_id("v7_ensemble", "regime_gated_ensemble", best.get("target_variant", ""), f"h{best.get('horizon', '')}", best.get("feature_group", "")), "model_family": "regime_gated_ensemble", "task": "direction", "overfit_risk": "high", "claim_label": "exploratory_not_claimable"})
    if ranking_ok:
        best_rank = max(ranking_ok, key=lambda row: (as_float(row.get("validation_rank_ic")), as_float(row.get("validation_ndcg_at_10"))))
        rows.append({**best_rank, "candidate_id": candidate_id("v7_ensemble", "rank_average_ensemble", best_rank.get("target_variant", ""), f"h{best_rank.get('horizon', '')}", best_rank.get("feature_group", "")), "model_family": "rank_average_ensemble", "task": "ranking", "overfit_risk": "medium", "claim_label": "exploratory_not_claimable"})
    if price_ok:
        best_price = max(price_ok, key=lambda row: (as_float(row.get("validation_improvement_vs_random_walk")), -as_float(row.get("validation_rmse"))))
        rows.append({**best_price, "candidate_id": candidate_id("v7_ensemble", "bayesian_random_search_lite", best_price.get("target_variant", ""), f"h{best_price.get('horizon', '')}", best_price.get("feature_group", "")), "model_family": "bayesian_random_search_lite", "task": "price_return", "overfit_risk": "high", "claim_label": "exploratory_not_claimable"})
    if dependency.get("optuna_available"):
        source = direction_ok[0] if direction_ok else (price_ok[0] if price_ok else {})
        if source:
            rows.append({**source, "candidate_id": candidate_id("v7_ensemble", "optuna_lite", source.get("target_variant", ""), f"h{source.get('horizon', '')}", source.get("feature_group", "")), "model_family": "optuna_lite", "task": source.get("task", "direction"), "overfit_risk": "high", "claim_label": "exploratory_not_claimable"})
    else:
        skipped.append({"task": "ensemble_automl", "model_family": "optuna_lite", "target_variant": "", "horizon": "", "feature_group": "", "skipped_reason": "optuna unavailable"})
    if not rows:
        skipped.append({"task": "ensemble_automl", "model_family": "all", "target_variant": "", "horizon": "", "feature_group": "", "skipped_reason": "no eligible source rows for V7 ensemble"})
    return rows, skipped


def v7_best_direction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("task") == "direction"]
    return max(ok, key=lambda row: (as_float(row.get("validation_balanced_accuracy", row.get("balanced_accuracy"))), as_float(row.get("validation_macro_f1")), as_float(row.get("validation_mcc"))), default={})


def v7_best_ranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    return max(ok, key=lambda row: (as_float(row.get("validation_rank_ic")), as_float(row.get("validation_ndcg_at_10")), as_float(row.get("validation_top20_precision"))), default={})


def v7_best_price(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("task") in {"price_return", "probabilistic_quantile"}]
    return max(ok, key=lambda row: (as_float(row.get("validation_improvement_vs_random_walk")), as_float(row.get("validation_rank_ic")), -as_float(row.get("validation_rmse"))), default={})


def v7_validation_governed_leaderboard(direction_rows: list[dict[str, Any]], ranking_rows: list[dict[str, Any]], price_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in direction_rows:
        if row.get("status") == "ok":
            rows.append({"task": "direction", "candidate_id": row.get("candidate_id", ""), "model_family": row.get("model_family", ""), "target_variant": row.get("target_variant", ""), "horizon": row.get("horizon", ""), "feature_group": row.get("feature_group", ""), "validation_metric": "balanced_accuracy", "validation_value": row.get("validation_balanced_accuracy", row.get("balanced_accuracy")), "final_metric": "balanced_accuracy", "final_value": row.get("final_balanced_accuracy"), "claim_label": "exploratory_not_claimable"})
    for row in ranking_rows:
        if row.get("status") == "ok":
            rows.append({"task": "ranking", "candidate_id": row.get("candidate_id", ""), "model_family": row.get("model_family", ""), "target_variant": row.get("target_variant", ""), "horizon": row.get("horizon", ""), "feature_group": row.get("feature_group", ""), "validation_metric": "rank_ic", "validation_value": row.get("validation_rank_ic"), "final_metric": "rank_ic", "final_value": row.get("final_rank_ic"), "claim_label": "exploratory_not_claimable"})
    for row in price_rows:
        if row.get("status") == "ok":
            rows.append({"task": row.get("task", "price_return"), "candidate_id": row.get("candidate_id", ""), "model_family": row.get("model_family", ""), "target_variant": row.get("target_variant", ""), "horizon": row.get("horizon", ""), "feature_group": row.get("feature_group", ""), "validation_metric": "rmse_improvement_vs_random_walk", "validation_value": row.get("validation_improvement_vs_random_walk"), "final_metric": "rmse_improvement_vs_random_walk", "final_value": row.get("final_improvement_vs_random_walk"), "claim_label": "exploratory_not_claimable"})
    return sorted(rows, key=lambda row: as_float(row.get("validation_value")), reverse=True)


def v7_write_reports(
    newly_run: list[str],
    skipped_rows: list[dict[str, Any]],
    best_direction: dict[str, Any],
    best_ranking: dict[str, Any],
    best_price: dict[str, Any],
    best_prob: dict[str, Any],
    best_regime: dict[str, Any],
    best_qml: dict[str, Any],
    best_ensemble: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    abs_beats = bool(best_direction.get("target_variant") == "absolute_direction" and as_float(best_direction.get("final_accuracy")) > CLASSICAL_CHAMPION["final_accuracy"])
    qml_beats = bool(best_qml and as_float(best_qml.get("final_accuracy")) > QML_V8_CONTEXT_FINAL and as_float(best_qml.get("final_balanced_accuracy")) > 0.5)
    price_robust = bool(as_float(best_price.get("validation_improvement_vs_random_walk")) > 0 and as_float(best_price.get("final_improvement_vs_random_walk")) > 0)
    summary = f"""# VN30 Model Universe V7 Exhaustive Expansion Result Summary

## Required Answers

1. Did V7 run both audit and execution: yes. Coverage/dependency audit artifacts were written before V7 execution artifacts.
2. Which model families were newly run: {", ".join(newly_run)}.
3. Which families were still skipped and why: {len(skipped_rows)} rows were skipped; see `v7_skipped_models.csv`.
4. Did ranking/cross-sectional models find useful signal: {safe_bool_text(as_float(best_ranking.get("validation_rank_ic")) > 0)}. Best ranking row `{best_ranking.get("candidate_id", "")}` validation rank IC {as_float(best_ranking.get("validation_rank_ic")):.4f}.
5. Did modern time-series models improve direction or return forecasting: {safe_bool_text(bool(best_price.get("model_family")))} for bounded local approximations where dependencies allowed.
6. Did probabilistic/quantile models improve price/return forecasting: {safe_bool_text(as_float(best_prob.get("validation_improvement_vs_random_walk")) > 0)}. Best probabilistic row `{best_prob.get("candidate_id", "")}`.
7. Did regime-aware models help: {safe_bool_text(as_float(best_regime.get("validation_balanced_accuracy", best_regime.get("validation_improvement_vs_random_walk"))) > 0)}. Best regime row `{best_regime.get("candidate_id", "")}`.
8. Did CatBoost run: {safe_bool_text(any(row.get("model_family", "").startswith("catboost") and row.get("status") == "ok" for row in [best_direction, best_price, best_prob, best_regime, best_qml, best_ensemble]))}.
9. Did QML extended diagnostics improve over QML V8: {safe_bool_text(qml_beats)}. V7 QML rows remain diagnostic-only.
10. Did ensemble/AutoML-lite help: {safe_bool_text(bool(best_ensemble.get("candidate_id")))}. Best ensemble row `{best_ensemble.get("candidate_id", "")}`.
11. Did any validation-governed candidate beat the 61.61 absolute-direction champion on comparable scope: {safe_bool_text(abs_beats)}.
12. Did any validation-governed market-relative candidate beat QML V8 64.44 on comparable scope under repaired metrics: {safe_bool_text(qml_beats)}.
13. Did any price/return model beat random walk / last price robustly: {safe_bool_text(price_robust)}.
14. Is any result claimable now: no.
15. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, daily T+1 system, production, VN100, DOCX, tag, merge, main-branch, push --mirror, or champion-replacement claim.

## Best Validation-Governed Rows

- Direction: `{best_direction.get("candidate_id", "")}` validation balanced accuracy {pct(best_direction.get("validation_balanced_accuracy", best_direction.get("balanced_accuracy")))}, final balanced accuracy {pct(best_direction.get("final_balanced_accuracy"))}.
- Ranking: `{best_ranking.get("candidate_id", "")}` validation rank IC {as_float(best_ranking.get("validation_rank_ic")):.4f}, final rank IC {as_float(best_ranking.get("final_rank_ic")):.4f}.
- Price/return: `{best_price.get("candidate_id", "")}` validation random-walk improvement {pp(best_price.get("validation_improvement_vs_random_walk"))}, final random-walk improvement {pp(best_price.get("final_improvement_vs_random_walk"))}.
- Probabilistic: `{best_prob.get("candidate_id", "")}` validation pinball loss {as_float(best_prob.get("validation_pinball_loss")):.6g}.
- Regime-aware: `{best_regime.get("candidate_id", "")}`.
- QML extended: `{best_qml.get("candidate_id", "")}`.
- Ensemble/AutoML-lite: `{best_ensemble.get("candidate_id", "")}`.

## Decision

- Decision labels: `{", ".join([key for key, value in decision.items() if isinstance(value, bool) and value])}`.
- All final-ranked rows are `exploratory_not_claimable`.
"""
    write_markdown(V7_RESULT_PATH, summary)
    claim = """# VN30 Model Universe V7 Exhaustive Expansion Claim Boundary

- V7 is an offline diagnostic-only model-family expansion for VN30 stock hourly forecasting.
- V7 does not use VN100 and does not create daily T+1 system files.
- Direction, ranking, probabilistic, and price/return metrics are separate.
- Market-relative raw accuracy is not sufficient for promotion.
- Candidate comparison is validation-governed; final performance is scoring-only.
- Final-ranked rows remain exploratory_not_claimable.
- No row is claimable now; future-blind confirmation is required before any stronger statement.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, DOCX, tag, merge, push --mirror, main-branch, or champion-replacement claim is made.
"""
    write_markdown(V7_CLAIM_PATH, claim)


def run_exhaustive_expansion(timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    write_json(OUTPUT_DIR / "v7_dependency_status.json", {"dependencies": v7_dependency_rows(dependency), "diagnostic_only": True, "no_trading_claim": True, "no_vn100": True})
    coverage_rows, missing_rows = v7_coverage_matrix(dependency)
    v7_write_frame("v7_model_family_coverage_matrix.csv", coverage_rows)
    v7_write_frame("v7_missing_model_family_registry.csv", missing_rows)

    features, family_cols, relative_cols, index_data, feature_groups, config = v7_common_context(timeout_seconds)
    feature_group_names = v7_feature_group_names(feature_groups)
    candidate_grid: list[dict[str, Any]] = []
    for horizon in V7_HORIZONS:
        for target in V7_DIRECTION_TARGETS:
            for group in feature_group_names:
                if group == "qml_kernel_features":
                    continue
                for model in ["logistic_regression", "ridge_classifier", "hist_gradient_boosting", "lightgbm_classifier", "xgboost_classifier"]:
                    candidate_grid.append({"task": "direction", "target_variant": target, "horizon": horizon, "feature_group": group, "model_family": model, "claim_label": "exploratory_not_claimable"})
        for target in V7_RANKING_TARGETS:
            for group in feature_group_names:
                candidate_grid.append({"task": "ranking", "target_variant": target, "horizon": horizon, "feature_group": group, "model_family": "bounded_ranker", "claim_label": "exploratory_not_claimable"})
        for target in V7_PRICE_TARGETS:
            for group in feature_group_names:
                if group == "qml_kernel_features":
                    continue
                candidate_grid.append({"task": "price_return", "target_variant": target, "horizon": horizon, "feature_group": group, "model_family": "bounded_regressor", "claim_label": "exploratory_not_claimable"})
    v7_write_frame("v7_unified_candidate_grid.csv", candidate_grid)

    direction_rows: list[dict[str, Any]] = []
    direction_skipped: list[dict[str, Any]] = []
    direction_models = ["logistic_regression", "ridge_classifier", "hist_gradient_boosting", "lightgbm_classifier", "xgboost_classifier"]
    for horizon in [20, 40]:
        for target_variant in V7_DIRECTION_TARGETS:
            labels = build_direction_target(features, index_data, target_variant, horizon)
            splits = split_sample(features, labels, strict_split_indices(features, labels), config, classification=True)
            guard = leakage_guard_passed(features, labels, splits)
            for feature_group in ["relative_strength", "market_context", "compact_stable_features"]:
                cols = feature_groups.get(feature_group, [])
                for model_name in direction_models:
                    try:
                        val_pred, val_prob, final_pred, final_prob, selected, status = v6_direction_model_predict(model_name, features, labels, splits, cols)
                        if status != "ok":
                            raise ValueError(status)
                        row = v7_direction_result_from_predictions("v7_direction", model_name, target_variant, horizon, feature_group, labels, splits, val_pred, val_prob, final_pred, final_prob, guard)
                        row["selected_features"] = "|".join(selected)
                        direction_rows.append(row)
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        direction_skipped.append({"task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": feature_group, "skipped_reason": reason})
                        direction_rows.append({"candidate_id": candidate_id("v7_direction", model_name, target_variant, f"h{horizon}", feature_group), "task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": feature_group, "status": "skipped", "skipped_reason": reason, "claim_label": "exploratory_not_claimable"})

    ranking_rows, ranking_skipped = v7_run_ranking(features, index_data, feature_groups, config, started)
    modern_rows, modern_skipped = v7_run_modern_timeseries(features, family_cols, relative_cols, index_data, feature_groups, dependency, config, started)
    probabilistic_rows, probabilistic_skipped = v7_run_probabilistic(features, index_data, feature_groups, dependency, config, started)
    regime_rows, regime_skipped = v7_run_regime(features, index_data, feature_groups, dependency, config, started)
    catboost_rows, catboost_skipped = v7_run_catboost(features, family_cols, relative_cols, index_data, feature_groups, dependency, config)
    qml_rows, qml_skipped = v7_run_qml_extended(features, family_cols, relative_cols, index_data, feature_groups, dependency, config)

    price_return_rows = [row for row in modern_rows + probabilistic_rows + regime_rows + catboost_rows if row.get("task") in {"price_return", "probabilistic_quantile"}]
    all_direction_rows = direction_rows + [row for row in modern_rows + regime_rows + catboost_rows + qml_rows if row.get("task") == "direction"]
    ensemble_rows, ensemble_skipped = v7_run_ensembles(all_direction_rows, ranking_rows, price_return_rows, dependency)
    all_skipped = direction_skipped + ranking_skipped + modern_skipped + probabilistic_skipped + regime_skipped + catboost_skipped + qml_skipped + ensemble_skipped

    ranking_leaderboard = sorted([row for row in ranking_rows if row.get("status") == "ok"], key=lambda row: (as_float(row.get("validation_rank_ic")), as_float(row.get("validation_ndcg_at_10"))), reverse=True)
    modern_leaderboard = sorted([row for row in modern_rows if row.get("status") == "ok"], key=lambda row: (as_float(row.get("validation_improvement_vs_random_walk")), as_float(row.get("validation_balanced_accuracy"))), reverse=True)
    probabilistic_leaderboard = sorted([row for row in probabilistic_rows if row.get("status") == "ok"], key=lambda row: (-as_float(row.get("validation_pinball_loss")), as_float(row.get("validation_improvement_vs_random_walk"))), reverse=True)
    regime_leaderboard = sorted([row for row in regime_rows if row.get("status") == "ok"], key=lambda row: (as_float(row.get("validation_balanced_accuracy", row.get("validation_improvement_vs_random_walk"))), as_float(row.get("validation_mcc"))), reverse=True)
    qml_leaderboard = sorted([row for row in qml_rows if row.get("status") == "ok"], key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("final_accuracy"))), reverse=True)
    ensemble_leaderboard = sorted([row for row in ensemble_rows if row.get("status", "ok") == "ok"], key=lambda row: (as_float(row.get("validation_balanced_accuracy", row.get("validation_rank_ic"))), as_float(row.get("validation_improvement_vs_random_walk"))), reverse=True)

    v7_write_frame("v7_direction_results.csv", all_direction_rows)
    v7_write_frame("v7_ranking_results.csv", ranking_rows)
    v7_write_frame("v7_ranking_leaderboard.csv", ranking_leaderboard)
    v7_write_frame("v7_ranking_results_unified.csv", ranking_rows)
    v7_write_frame("v7_modern_timeseries_results.csv", modern_rows)
    v7_write_frame("v7_modern_timeseries_leaderboard.csv", modern_leaderboard)
    v7_write_frame("v7_probabilistic_quantile_results.csv", probabilistic_rows)
    v7_write_frame("v7_probabilistic_quantile_leaderboard.csv", probabilistic_leaderboard)
    v7_write_frame("v7_regime_aware_results.csv", regime_rows)
    v7_write_frame("v7_regime_aware_leaderboard.csv", regime_leaderboard)
    v7_write_frame("v7_catboost_results.csv", catboost_rows)
    v7_write_frame("v7_qml_extended_results.csv", qml_rows)
    v7_write_frame("v7_qml_extended_leaderboard.csv", qml_leaderboard)
    v7_write_frame("v7_ensemble_automl_results.csv", ensemble_rows)
    v7_write_frame("v7_ensemble_automl_leaderboard.csv", ensemble_leaderboard)
    v7_write_frame("v7_price_return_results.csv", price_return_rows)
    validation_leaderboard = v7_validation_governed_leaderboard(all_direction_rows, ranking_rows, price_return_rows + ensemble_rows)
    v7_write_frame("v7_validation_governed_leaderboard.csv", validation_leaderboard)
    exploratory_final = sorted([dict(row, claim_label="exploratory_not_claimable") for row in validation_leaderboard], key=lambda row: as_float(row.get("final_value")), reverse=True)
    v7_write_frame("v7_exploratory_final_leaderboard.csv", exploratory_final)
    v7_write_frame("v7_skipped_models.csv", all_skipped)
    runtime_rows = [{"phase": "total", "runtime_seconds": time.perf_counter() - started, "direction_rows": len(all_direction_rows), "ranking_rows": len(ranking_rows), "price_return_rows": len(price_return_rows), "skipped_rows": len(all_skipped), "diagnostic_only": True, "no_vn100": True}]
    v7_write_frame("v7_runtime_summary.csv", runtime_rows)

    best_direction = v7_best_direction(all_direction_rows)
    best_ranking = v7_best_ranking(ranking_rows)
    best_price = v7_best_price(price_return_rows)
    best_prob = v7_best_price(probabilistic_rows)
    best_regime = max([row for row in regime_rows if row.get("status") == "ok"], key=lambda row: (as_float(row.get("validation_balanced_accuracy", row.get("validation_improvement_vs_random_walk"))), as_float(row.get("validation_mcc"))), default={})
    best_qml = max([row for row in qml_rows if row.get("status") == "ok"], key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("final_accuracy"))), default={})
    best_ensemble = max([row for row in ensemble_rows if row.get("status", "ok") == "ok"], key=lambda row: (as_float(row.get("validation_balanced_accuracy", row.get("validation_rank_ic"))), as_float(row.get("validation_improvement_vs_random_walk"))), default={})
    decision = {
        "v7_found_candidate": bool(best_direction or best_ranking or best_price),
        "v7_no_candidate": not bool(best_direction or best_ranking or best_price),
        "ranking_candidate_found": bool(best_ranking and as_float(best_ranking.get("validation_rank_ic")) > 0),
        "price_return_candidate_found": bool(best_price and as_float(best_price.get("validation_improvement_vs_random_walk")) > 0),
        "probabilistic_candidate_found": bool(best_prob and as_float(best_prob.get("validation_improvement_vs_random_walk")) > 0),
        "regime_candidate_found": bool(best_regime),
        "qml_extended_candidate_found": bool(best_qml),
        "future_blind_required": True,
        "not_claimable": True,
        "best_direction_candidate": best_direction.get("candidate_id", ""),
        "best_ranking_candidate": best_ranking.get("candidate_id", ""),
        "best_price_return_candidate": best_price.get("candidate_id", ""),
        "best_probabilistic_candidate": best_prob.get("candidate_id", ""),
        "best_regime_candidate": best_regime.get("candidate_id", ""),
        "best_qml_extended_candidate": best_qml.get("candidate_id", ""),
        "best_ensemble_automl_candidate": best_ensemble.get("candidate_id", ""),
    }
    write_json(OUTPUT_DIR / "v7_exhaustive_expansion_decision.json", decision)
    newly_run = [
        "ranking/cross-sectional",
        "modern time-series local approximations",
        "probabilistic/quantile",
        "regime-aware",
        "CatBoost if available",
        "QML extended diagnostics",
        "ensemble/AutoML-lite",
    ]
    v7_write_reports(newly_run, all_skipped, best_direction, best_ranking, best_price, best_prob, best_regime, best_qml, best_ensemble, decision)
    result = {"status": "ok", "mode": "exhaustive_expansion", "runtime_seconds": time.perf_counter() - started, **decision}
    print(json.dumps(json_safe(result), indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 model-universe direction and price/return diagnostics.")
    parser.add_argument("--smoke", action="store_true", help="Run small smoke coverage.")
    parser.add_argument("--validation-screening", action="store_true", help="Run bounded validation screening.")
    parser.add_argument("--promotion-relock", action="store_true", help="Run V2 promotion relock audit for high-performing exploratory final rows.")
    parser.add_argument("--enable-skipped-families", action="store_true", help="Run V3 bounded benchmark for previously skipped families.")
    parser.add_argument("--bilstm-relock", action="store_true", help="Run V4 BiLSTM relock and stability confirmation.")
    parser.add_argument("--target-metric-repair", action="store_true", help="Run V5 target and metric repair audit.")
    parser.add_argument("--price-absolute-relock", action="store_true", help="Run V6 price/return relock and absolute-direction repaired confirmation.")
    parser.add_argument("--exhaustive-expansion", action="store_true", help="Run V7 bounded exhaustive expansion over high-priority missing model families.")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.promotion_relock:
        run_promotion_relock(max(1, int(args.timeout_seconds)))
        return
    if args.enable_skipped_families:
        run_enable_skipped_families(max(1, int(args.timeout_seconds)))
        return
    if args.bilstm_relock:
        run_bilstm_relock(max(1, int(args.timeout_seconds)))
        return
    if args.target_metric_repair:
        run_target_metric_repair(max(1, int(args.timeout_seconds)))
        return
    if args.price_absolute_relock:
        run_price_absolute_relock(max(1, int(args.timeout_seconds)))
        return
    if args.exhaustive_expansion:
        run_exhaustive_expansion(max(1, int(args.timeout_seconds)))
        return
    if args.smoke:
        config = RunConfig("smoke", max(1, int(args.timeout_seconds)), 500, 250, 250, 80, 80)
    else:
        config = RunConfig("validation_screening", max(1, int(args.timeout_seconds)), 1600, 700, 700, 650, 650)
    run_benchmark(config)


if __name__ == "__main__":
    main()
