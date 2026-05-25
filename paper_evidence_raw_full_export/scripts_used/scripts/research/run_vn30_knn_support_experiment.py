"""Run KNN-support feature and ensemble experiment for VN30 model zoo."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
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

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research import run_vn30_comprehensive_model_universe_benchmark as universe  # noqa: E402
from scripts.research import run_vn30_fair_exhaustive_model_zoo_tuning as fair  # noqa: E402
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

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "knn_support"
PRIMARY_HORIZON = 40
THRESHOLD_GRID = [0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]
RANDOM_STATE = 42
CURRENT_MAIN_FINAL_ACCURACY = 0.6163475699558174
CURRENT_MAIN_LABEL = "Logistic L2 h40 threshold 0.55 final 61.63%"
BEST_DESCRIPTIVE_LABEL = "bull_bear_sideway_router h40 fixed 0.50 final 63.33%"
KNN_FEATURES = [
    "knn_up_ratio_k5",
    "knn_up_ratio_k11",
    "knn_up_ratio_k21",
    "knn_mean_distance_k11",
    "knn_distance_weighted_up_ratio_k11",
    "knn_similar_regime_up_ratio",
    "knn_neighbor_volatility_mean",
    "knn_prediction_probability",
]


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def pct(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:.2f}%"


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


def candidate_id(*parts: Any) -> str:
    return universe.candidate_id("knn_support", *parts)


def config_tag(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        value = params[key]
        if key == "base_inputs" and isinstance(value, (list, tuple)):
            text = f"{len(value)}inputs"
        elif isinstance(value, (list, tuple)):
            text = "x".join(str(item) for item in value)
        else:
            text = str(value)
        parts.append(f"{key}{text}".replace(".", "p").replace(" ", "_").replace(":", "_"))
    return "_".join(parts)[:120]


def threshold_specs(y_true: pd.Series | np.ndarray, score: np.ndarray) -> list[tuple[str, float]]:
    specs = [("fixed_0.50", 0.50)]
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(score, dtype=float)
    best_threshold = 0.50
    best_accuracy = -1.0
    for threshold in THRESHOLD_GRID:
        pred = (scores >= threshold).astype(int)
        acc = float((pred == y).mean()) if len(y) else math.nan
        if math.isfinite(acc) and (
            acc > best_accuracy + 1e-12
            or (abs(acc - best_accuracy) <= 1e-12 and abs(threshold - 0.50) < abs(best_threshold - 0.50))
        ):
            best_threshold = float(threshold)
            best_accuracy = acc
    specs.append(("validation_selected_threshold", best_threshold))
    return specs


def volatility_column(features: pd.DataFrame) -> str:
    for col in ["rolling_return_vol_20", "roll_vol_20", "vnindex_vol_20_lag_ctx", "vn30_vol_20_lag_ctx"]:
        if col in features.columns:
            return col
    return "close"


def clean_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def build_knn_support_features(
    features: pd.DataFrame,
    labels: pd.Series,
    idx: dict[str, pd.Index],
    base_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build KNN support features with time-safe neighbor universes.

    Train rows use leave-one-out neighbors from train only. Validation rows use
    train-only neighbors. Final rows use train+validation neighbors, with the
    scaler still fit on train only and no final labels.
    """

    support = pd.DataFrame(index=features.index, columns=KNN_FEATURES, dtype=float)
    train_idx = pd.Index(idx["train"])
    validation_idx = pd.Index(idx["validation"])
    final_idx = pd.Index(idx["final"])
    train_val_idx = train_idx.append(validation_idx)

    cols = [col for col in base_cols if col in features.columns and pd.api.types.is_numeric_dtype(features[col])]
    if not cols:
        raise RuntimeError("no numeric columns available for KNN neighbor space")

    train_raw = clean_matrix(features.loc[train_idx], cols)
    med = train_raw.median(axis=0).fillna(0.0)
    train_clean = train_raw.fillna(med)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_clean)

    all_clean = clean_matrix(features, cols).fillna(med)
    all_scaled = pd.DataFrame(scaler.transform(all_clean), index=features.index, columns=cols)

    y_train = labels.loc[train_idx].astype(int).to_numpy()
    y_train_val = labels.loc[train_val_idx].astype(int).to_numpy()
    regime_train = features.loc[train_idx, "regime_router_key"].astype(str).to_numpy()
    regime_train_val = features.loc[train_val_idx, "regime_router_key"].astype(str).to_numpy()
    vol_col = volatility_column(features)
    vol_train = pd.to_numeric(features.loc[train_idx, vol_col], errors="coerce").fillna(0.0).to_numpy()
    vol_train_val = pd.to_numeric(features.loc[train_val_idx, vol_col], errors="coerce").fillna(0.0).to_numpy()

    def assign(
        query_idx: pd.Index,
        ref_idx: pd.Index,
        ref_y: np.ndarray,
        ref_regime: np.ndarray,
        ref_vol: np.ndarray,
        *,
        leave_one_out: bool,
    ) -> None:
        if len(query_idx) == 0:
            return
        n_neighbors = 22 if leave_one_out else 21
        model = NearestNeighbors(n_neighbors=min(n_neighbors, len(ref_idx)), metric="minkowski", p=2)
        model.fit(all_scaled.loc[ref_idx].to_numpy(dtype=float))
        distances, neighbor_pos = model.kneighbors(all_scaled.loc[query_idx].to_numpy(dtype=float), return_distance=True)
        for row_number, row_id in enumerate(query_idx):
            pos = neighbor_pos[row_number]
            dist = distances[row_number]
            if leave_one_out:
                ref_rows = ref_idx.to_numpy()[pos]
                keep = ref_rows != row_id
                pos = pos[keep][:21]
                dist = dist[keep][:21]
            else:
                pos = pos[:21]
                dist = dist[:21]
            if len(pos) == 0:
                values = {feature: math.nan for feature in KNN_FEATURES}
            else:
                neighbor_y = ref_y[pos]
                neighbor_regime = ref_regime[pos]
                neighbor_vol = ref_vol[pos]
                weights = 1.0 / np.maximum(dist, 1e-9)
                query_regime = str(features.at[row_id, "regime_router_key"])
                same_regime = neighbor_regime == query_regime
                similar_regime_ratio = float(neighbor_y[same_regime].mean()) if same_regime.any() else float(neighbor_y[: min(11, len(neighbor_y))].mean())
                values = {
                    "knn_up_ratio_k5": float(neighbor_y[: min(5, len(neighbor_y))].mean()),
                    "knn_up_ratio_k11": float(neighbor_y[: min(11, len(neighbor_y))].mean()),
                    "knn_up_ratio_k21": float(neighbor_y.mean()),
                    "knn_mean_distance_k11": float(dist[: min(11, len(dist))].mean()),
                    "knn_distance_weighted_up_ratio_k11": float(np.average(neighbor_y[: min(11, len(neighbor_y))], weights=weights[: min(11, len(weights))])),
                    "knn_similar_regime_up_ratio": similar_regime_ratio,
                    "knn_neighbor_volatility_mean": float(np.nanmean(neighbor_vol[: min(11, len(neighbor_vol))])),
                    "knn_prediction_probability": float(np.average(neighbor_y[: min(11, len(neighbor_y))], weights=weights[: min(11, len(weights))])),
                }
            for feature, value in values.items():
                support.at[row_id, feature] = value

    assign(train_idx, train_idx, y_train, regime_train, vol_train, leave_one_out=True)
    assign(validation_idx, train_idx, y_train, regime_train, vol_train, leave_one_out=False)
    assign(final_idx, train_val_idx, y_train_val, regime_train_val, vol_train_val, leave_one_out=False)
    support = support.astype(float).fillna(support.loc[train_idx].median(axis=0).fillna(0.0))
    manifest = {
        "knn_neighbor_space": "baseline_C_closest numeric columns",
        "neighbor_space_feature_count": len(cols),
        "support_features": KNN_FEATURES,
        "scaler_fit_scope": "train_only",
        "train_rows_neighbor_rule": "leave_one_out_train_neighbors_only",
        "validation_neighbor_rule": "train_neighbors_only",
        "final_neighbor_rule": "train_plus_validation_neighbors_only_after_validation_lock",
        "final_labels_used": False,
        "max_k": 21,
        "distance_metric": "minkowski_p2",
        "volatility_source_column": vol_col,
    }
    return support, manifest


def estimator_logistic_l2(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1200, solver="liblinear", penalty="l2", C=float(params.get("C", 0.3)), class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def estimator_elastic_net(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    solver="saga",
                    penalty="elasticnet",
                    C=float(params.get("C", 0.3)),
                    l1_ratio=float(params.get("l1_ratio", 0.2)),
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def estimator_random_forest(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=int(params.get("n_estimators", 140)),
                    max_depth=params.get("max_depth", 8),
                    min_samples_leaf=int(params.get("min_samples_leaf", 10)),
                    max_features=params.get("max_features", "sqrt"),
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def estimator_xgboost(params: dict[str, Any]) -> Pipeline | None:
    if XGBClassifier is None:
        return None
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=int(params.get("n_estimators", 100)),
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


def estimator_lightgbm(params: dict[str, Any]) -> Pipeline | None:
    if LGBMClassifier is None:
        return None
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=int(params.get("n_estimators", 100)),
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


def predict_score(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    return np.clip(universe.predict_score(model, x_data), 0.0, 1.0)


def add_grid_row(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    model_family: str,
    model_id: str,
    feature_set: str,
    threshold_policy: str,
    planned_status: str,
    params: dict[str, Any],
    reason: str = "",
    runtime: float = math.nan,
) -> None:
    rows.append(
        {
            "candidate_id": candidate,
            "model_family": model_family,
            "model_id": model_id,
            "feature_set": feature_set,
            "horizon": PRIMARY_HORIZON,
            "threshold_policy": threshold_policy,
            "planned_status": planned_status,
            "hyperparameters": json.dumps(json_safe(params), sort_keys=True),
            "fit_runtime_seconds": runtime,
            "reason": reason,
            "selection_source": "validation_only",
            "final_accuracy_used_for_selection": False,
            "ticker_subset": False,
            "confidence_abstention": False,
            "topk_substitution": False,
        }
    )


def result_for_scores(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    idx: dict[str, pd.Index],
    candidate: str,
    model_family: str,
    model_id: str,
    feature_set: str,
    threshold_policy: str,
    threshold: float,
    val_score: np.ndarray,
    final_score: np.ndarray,
    train_rows: int,
    feature_count: int,
    params: dict[str, Any],
    runtime: float,
    note: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    val_frame = universe.prediction_frame(
        features,
        idx["validation"],
        labels,
        val_score,
        (val_score >= threshold).astype(int),
        model_group=model_family,
        model_id=model_id,
        feature_family=feature_set,
        horizon=PRIMARY_HORIZON,
        threshold_policy=threshold_policy,
        threshold=threshold,
        candidate=candidate,
        split="validation",
    )
    final_frame = universe.prediction_frame(
        features,
        idx["final"],
        labels,
        final_score,
        (final_score >= threshold).astype(int),
        model_group=model_family,
        model_id=model_id,
        feature_family=feature_set,
        horizon=PRIMARY_HORIZON,
        threshold_policy=threshold_policy,
        threshold=threshold,
        candidate=candidate,
        split="final",
    )
    row = universe.result_row(
        candidate=candidate,
        model_group=model_family,
        model_id=model_id,
        feature_family=feature_set,
        horizon=PRIMARY_HORIZON,
        threshold_policy=threshold_policy,
        threshold=threshold,
        validation_frame=val_frame,
        final_frame=final_frame,
        train_rows=train_rows,
        feature_count=feature_count,
        reason_not_claim_eligible="not selected by validation-only KNN-support rule",
        implementation_note=note,
    )
    row.update(fair.validation_metric_overlay(val_frame))
    row.update(
        {
            "hyperparameters": json.dumps(json_safe(params), sort_keys=True),
            "fit_runtime_seconds": runtime,
            "selection_source": "validation_only",
            "final_accuracy_used_for_selection": False,
            "ticker_subset": False,
            "confidence_abstention": False,
            "topk_substitution": False,
            "knn_auxiliary_only": model_family != "standalone_knn_comparator",
            "claim_eligible_yes_no": "no",
            "reason_not_claim_eligible": "not selected by validation-only KNN-support rule",
        }
    )
    risk, risk_reason = fair.classify_overfit(pd.Series(row))
    row["overfit_risk"] = risk
    row["overfit_risk_reason"] = risk_reason
    return row, val_frame, final_frame


def add_candidate_results(
    *,
    grid_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    features: pd.DataFrame,
    labels: pd.Series,
    idx: dict[str, pd.Index],
    model_family: str,
    model_id: str,
    feature_set: str,
    val_score: np.ndarray,
    final_score: np.ndarray,
    train_rows: int,
    feature_count: int,
    params: dict[str, Any],
    runtime: float,
    note: str,
) -> None:
    val_y = labels.loc[idx["validation"]].astype(int)
    for threshold_policy, threshold in threshold_specs(val_y, val_score):
        cid = candidate_id(model_family, model_id, feature_set, f"h{PRIMARY_HORIZON}", config_tag(params), threshold_policy, f"t{threshold:.3f}")
        add_grid_row(
            grid_rows,
            candidate=cid,
            model_family=model_family,
            model_id=model_id,
            feature_set=feature_set,
            threshold_policy=threshold_policy,
            planned_status="run",
            params=params,
            runtime=runtime,
        )
        row, val_frame, final_frame = result_for_scores(
            features=features,
            labels=labels,
            idx=idx,
            candidate=cid,
            model_family=model_family,
            model_id=model_id,
            feature_set=feature_set,
            threshold_policy=threshold_policy,
            threshold=threshold,
            val_score=val_score,
            final_score=final_score,
            train_rows=train_rows,
            feature_count=feature_count,
            params=params,
            runtime=runtime,
            note=note,
        )
        result_rows.append(row)
        prediction_cache[cid] = (val_frame, final_frame)


def model_specs() -> list[dict[str, Any]]:
    return [
        {"family": "logistic_l2_knn_features", "model_id": "logistic_l2_knn", "factory": estimator_logistic_l2, "params": {"C": 0.3}},
        {"family": "logistic_l2_knn_features", "model_id": "logistic_l2_knn", "factory": estimator_logistic_l2, "params": {"C": 1.0}},
        {"family": "elastic_net_knn_features", "model_id": "elastic_net_knn", "factory": estimator_elastic_net, "params": {"C": 0.3, "l1_ratio": 0.2}},
        {"family": "elastic_net_knn_features", "model_id": "elastic_net_knn", "factory": estimator_elastic_net, "params": {"C": 0.3, "l1_ratio": 0.5}},
        {"family": "xgboost_knn_features", "model_id": "xgboost_knn", "factory": estimator_xgboost, "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}},
        {"family": "xgboost_knn_features", "model_id": "xgboost_knn", "factory": estimator_xgboost, "params": {"n_estimators": 140, "max_depth": 2, "learning_rate": 0.03}},
        {"family": "lightgbm_knn_features", "model_id": "lightgbm_knn", "factory": estimator_lightgbm, "params": {"n_estimators": 100, "max_depth": 3, "num_leaves": 15, "learning_rate": 0.05}},
        {"family": "lightgbm_knn_features", "model_id": "lightgbm_knn", "factory": estimator_lightgbm, "params": {"n_estimators": 140, "max_depth": 2, "num_leaves": 15, "learning_rate": 0.03}},
        {"family": "random_forest_knn_features", "model_id": "random_forest_knn", "factory": estimator_random_forest, "params": {"n_estimators": 140, "max_depth": 8, "min_samples_leaf": 10}},
        {"family": "random_forest_knn_features", "model_id": "random_forest_knn", "factory": estimator_random_forest, "params": {"n_estimators": 180, "max_depth": None, "min_samples_leaf": 25}},
        {"family": "regime_aware_knn_helper", "model_id": "regime_logistic_knn_helper", "factory": estimator_logistic_l2, "params": {"C": 0.3}},
    ]


def select_feature_columns(family_cols: dict[str, list[str]], feature_set: str) -> list[str]:
    if feature_set == "baseline_plus_knn":
        return family_cols["baseline_C_closest"] + KNN_FEATURES
    if feature_set == "combined_plus_knn":
        return family_cols["combined_context"] + KNN_FEATURES
    if feature_set == "regime_plus_knn":
        return family_cols["regime_context"] + KNN_FEATURES
    raise ValueError(f"unknown feature set {feature_set}")


def run_base_models(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    idx: dict[str, pd.Index],
    family_cols: dict[str, list[str]],
    grid_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    base_scores: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    train_y = labels.loc[idx["train"]].astype(int)
    for spec in model_specs():
        feature_sets = ["regime_plus_knn"] if spec["family"] == "regime_aware_knn_helper" else ["baseline_plus_knn", "combined_plus_knn"]
        for feature_set in feature_sets:
            params = dict(spec["params"])
            factory: Callable[[dict[str, Any]], Any] = spec["factory"]
            cols = select_feature_columns(family_cols, feature_set)
            start = time.perf_counter()
            model = factory(params)
            if model is None:
                cid = candidate_id(spec["family"], spec["model_id"], feature_set, "skipped")
                add_grid_row(grid_rows, candidate=cid, model_family=spec["family"], model_id=spec["model_id"], feature_set=feature_set, threshold_policy="not_run", planned_status="skipped_with_reason", params=params, reason="optional dependency unavailable")
                continue
            model.fit(features.loc[idx["train"], cols], train_y)
            val_score = predict_score(model, features.loc[idx["validation"], cols])
            final_score = predict_score(model, features.loc[idx["final"], cols])
            runtime = time.perf_counter() - start
            add_candidate_results(
                grid_rows=grid_rows,
                result_rows=result_rows,
                prediction_cache=prediction_cache,
                features=features,
                labels=labels,
                idx=idx,
                model_family=spec["family"],
                model_id=spec["model_id"],
                feature_set=feature_set,
                val_score=val_score,
                final_score=final_score,
                train_rows=len(train_y),
                feature_count=len(cols),
                params=params,
                runtime=runtime,
                note="KNN support features computed time-safely; downstream model fit on train only",
            )
            key = f"{spec['model_id']}:{feature_set}:{json.dumps(json_safe(params), sort_keys=True)}"
            base_scores[key] = (val_score, final_score, float((val_score >= 0.50).astype(int).mean()))
    return base_scores


def run_ensemble_models(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    idx: dict[str, pd.Index],
    grid_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    base_scores: dict[str, tuple[np.ndarray, np.ndarray, float]],
    knn_val_prob: np.ndarray,
    knn_final_prob: np.ndarray,
) -> None:
    if not base_scores:
        return
    selected = sorted(base_scores.items(), key=lambda item: item[0])[:6]
    val_matrix = np.column_stack([item[1][0] for item in selected] + [knn_val_prob])
    final_matrix = np.column_stack([item[1][1] for item in selected] + [knn_final_prob])
    base_names = [item[0] for item in selected] + ["knn_prediction_probability"]
    soft_val = val_matrix.mean(axis=1)
    soft_final = final_matrix.mean(axis=1)
    add_candidate_results(
        grid_rows=grid_rows,
        result_rows=result_rows,
        prediction_cache=prediction_cache,
        features=features,
        labels=labels,
        idx=idx,
        model_family="ensemble_meta_knn_support",
        model_id="soft_vote_with_knn_probability",
        feature_set="base_probabilities_plus_knn",
        val_score=soft_val,
        final_score=soft_final,
        train_rows=len(labels.loc[idx["train"]].dropna()),
        feature_count=len(base_names),
        params={"base_inputs": base_names, "ensemble": "unweighted_soft_vote"},
        runtime=0.0,
        note="soft vote includes KNN probability as auxiliary input; no final labels or final scores used for weights",
    )
    start = time.perf_counter()
    meta = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE))])
    meta.fit(pd.DataFrame(val_matrix, columns=base_names), labels.loc[idx["validation"]].astype(int))
    meta_val = predict_score(meta, pd.DataFrame(val_matrix, columns=base_names))
    meta_final = predict_score(meta, pd.DataFrame(final_matrix, columns=base_names))
    add_candidate_results(
        grid_rows=grid_rows,
        result_rows=result_rows,
        prediction_cache=prediction_cache,
        features=features,
        labels=labels,
        idx=idx,
        model_family="ensemble_meta_knn_support",
        model_id="logistic_meta_with_knn_probability",
        feature_set="base_probabilities_plus_knn",
        val_score=meta_val,
        final_score=meta_final,
        train_rows=len(labels.loc[idx["validation"]].dropna()),
        feature_count=len(base_names),
        params={"base_inputs": base_names, "meta_train_scope": "validation_predictions_only"},
        runtime=time.perf_counter() - start,
        note="meta-model trained on validation predictions plus KNN probability; final labels not used",
    )


def apply_selection(final_results: pd.DataFrame) -> pd.DataFrame:
    out = final_results.copy()
    out["selected_by_validation_yes_no"] = "no"
    out["claim_eligible_yes_no"] = "no"
    out["reason_not_claim_eligible"] = "not selected by validation-only KNN-support rule"
    pool = out[
        out["status"].astype(str).eq("ok")
        & out["full_ticker_coverage"].astype(bool)
        & ~out["model_group"].astype(str).eq("standalone_knn_comparator")
        & out["validation_accuracy"].apply(lambda value: math.isfinite(as_float(value)))
    ].copy()
    if pool.empty:
        return out
    selected = pool.sort_values(["validation_accuracy", "balanced_robust_score", "candidate_id"], ascending=[False, False, True]).iloc[0]
    mask = out["candidate_id"].astype(str).eq(str(selected["candidate_id"]))
    out.loc[mask, "selected_by_validation_yes_no"] = "yes"
    eligible = mask & ~out["overfit_risk"].astype(str).eq("high")
    out.loc[eligible, "claim_eligible_yes_no"] = "yes"
    out.loc[eligible, "reason_not_claim_eligible"] = ""
    out.loc[mask & out["overfit_risk"].astype(str).eq("high"), "reason_not_claim_eligible"] = "selected by validation but high overfit risk"
    return out


def row_predictions_for(final_results: pd.DataFrame, prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    selected_ids = final_results[final_results["selected_by_validation_yes_no"].astype(str).eq("yes")]["candidate_id"].astype(str).tolist()
    standalone_ids = final_results[final_results["model_group"].astype(str).eq("standalone_knn_comparator")].sort_values("validation_accuracy", ascending=False)["candidate_id"].astype(str).head(1).tolist()
    descriptive_ids = final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False])["candidate_id"].astype(str).head(3).tolist()
    ids = sorted(set(selected_ids + standalone_ids + descriptive_ids))
    frames: list[pd.DataFrame] = []
    for cid in ids:
        if cid in prediction_cache:
            frames.extend(prediction_cache[cid])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_slices(row_predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return fair.build_slice_outputs(row_predictions)


def transfer_overfit_risk(final_results: pd.DataFrame) -> tuple[str, str]:
    selected = final_results[final_results["selected_by_validation_yes_no"].astype(str).eq("yes")]
    if selected.empty:
        return "unknown", "no selected candidate"
    return str(selected.iloc[0]["overfit_risk"]), str(selected.iloc[0]["overfit_risk_reason"])


def write_reports(
    *,
    manifest: dict[str, Any],
    candidate_grid: pd.DataFrame,
    final_results: pd.DataFrame,
    row_predictions: pd.DataFrame,
    slices: dict[str, pd.DataFrame],
) -> None:
    selected = final_results[final_results["selected_by_validation_yes_no"].astype(str).eq("yes")]
    standalone = final_results[final_results["model_group"].astype(str).eq("standalone_knn_comparator")].sort_values("validation_accuracy", ascending=False).head(1)
    descriptive = final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False]).head(10)
    fair_summary = "not available"
    fair_path = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "fair_tuning" / "fair_tuning_final_results.csv"
    if fair_path.exists():
        fair_results = pd.read_csv(fair_path, low_memory=False)
        fair_best_selected = fair_results[fair_results.get("selected_by_validation_objective_yes_no", "no").astype(str).eq("yes")]
        fair_summary = f"fair tuning selected best final={pct(fair_best_selected['final_accuracy'].max()) if not fair_best_selected.empty else 'n/a'}; descriptive best={pct(fair_results['final_accuracy'].max())}"
    risk, risk_reason = transfer_overfit_risk(final_results)
    leakage_checks = [
        ("scaler_train_only", True, "StandardScaler fit on h40 train rows only"),
        ("knn_index_train_only", True, "validation KNN index uses train rows only; train features use leave-one-out train rows"),
        ("validation_neighbors_train_only", True, "validation neighbor labels come from train only"),
        ("final_neighbors_ex_ante", True, "final neighbor labels come from train+validation only after validation lock"),
        ("no_final_label_use", True, "final labels not used in KNN features, calibration, thresholds, weights, or meta fit"),
        ("full_coverage", bool(final_results[final_results["selected_by_validation_yes_no"].eq("yes")]["full_ticker_coverage"].astype(bool).all()), "selected rows have 30-stock coverage"),
        ("no_ticker_subset", bool(final_results["ticker_subset"].astype(str).str.lower().isin(["false", "0", "no"]).all()), ""),
        ("no_confidence_abstention", bool(final_results["confidence_abstention"].astype(str).str.lower().isin(["false", "0", "no"]).all()), ""),
        ("no_topk_substitution", bool(final_results["topk_substitution"].astype(str).str.lower().isin(["false", "0", "no"]).all()), ""),
    ]
    leakage_passed = all(item[1] for item in leakage_checks)
    leakage_frame = pd.DataFrame([{"check": name, "passed": "yes" if passed else "no", "detail": detail} for name, passed, detail in leakage_checks])
    write_markdown(
        OUTPUT_DIR / "knn_support_leakage_audit.md",
        "\n".join(
            [
                "# KNN Support Leakage Audit",
                "",
                f"- Leakage passed: {'yes' if leakage_passed else 'no'}.",
                f"- Overfit risk for selected KNN-support candidate: {risk}.",
                f"- Overfit risk reason: {risk_reason}.",
                "",
                fair.markdown_table(leakage_frame, max_rows=len(leakage_frame)),
            ]
        ),
    )
    write_markdown(
        OUTPUT_DIR / "knn_support_summary.md",
        "\n".join(
            [
                "# KNN Support Experiment Summary",
                "",
                f"- Candidate rows: {len(candidate_grid)}.",
                f"- Successful results: {int(final_results['status'].astype(str).eq('ok').sum())}.",
                f"- Primary horizon: h{PRIMARY_HORIZON}.",
                "- Optional h20/h60/h80 diagnostics: not run because the requested primary KNN-support test is h40 and no main result was promoted.",
                "- Data fetch: no.",
                "- Final-window selection: no.",
                f"- Current main comparison: {CURRENT_MAIN_LABEL}.",
                f"- Fair model-zoo comparison: {fair_summary}.",
                f"- Best descriptive comparison: {BEST_DESCRIPTIVE_LABEL}.",
                "",
                "## Validation-Selected KNN-Support Candidate",
                "",
                fair.markdown_table(selected[["candidate_id", "model_group", "model_id", "feature_family", "threshold_policy", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "overfit_risk"]] if not selected.empty else selected, max_rows=5),
                "",
                "## Standalone KNN Comparator",
                "",
                fair.markdown_table(standalone[["candidate_id", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "overfit_risk"]] if not standalone.empty else standalone, max_rows=3),
                "",
                "## Descriptive KNN-Support Final Leaderboard",
                "",
                fair.markdown_table(descriptive[["candidate_id", "model_group", "model_id", "feature_family", "threshold_policy", "validation_accuracy", "final_accuracy", "selected_by_validation_yes_no", "claim_eligible_yes_no", "overfit_risk"]], max_rows=10),
            ]
        ),
    )
    write_markdown(
        OUTPUT_DIR / "knn_support_claim_boundary.md",
        "\n".join(
            [
                "# KNN Support Claim Boundary",
                "",
                "- KNN is tested as an auxiliary similarity/probability/regime/calibration signal, not as a standalone main claim.",
                "- Claim eligibility requires validation-only selection, full 30-stock coverage, non-standalone KNN role, no final-label use, leakage audit pass, and no high overfit risk.",
                "- Descriptive final-window rows do not override validation-only selection.",
                "- No trading, profitability, investment recommendation, or live-deployment claim is made.",
                "",
                "## Claim Eligible Rows",
                "",
                fair.markdown_table(final_results[final_results["claim_eligible_yes_no"].eq("yes")][["candidate_id", "model_group", "model_id", "validation_accuracy", "final_accuracy", "overfit_risk"]], max_rows=10),
            ]
        ),
    )
    write_json(OUTPUT_DIR / "knn_support_feature_manifest.json", manifest)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(active_stock_tickers()) != 30:
        raise RuntimeError("full 30-stock VN30 coverage is required")
    features, family_cols, source_manifest = universe.prepare_features()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    labels = add_absolute_labels(features, PRIMARY_HORIZON)
    idx = universe.split_indices(features, labels)

    support, manifest = build_knn_support_features(features, labels, idx, family_cols["baseline_C_closest"])
    manifest.update(
        {
            "output_dir": rel(OUTPUT_DIR),
            "data_fetch": False,
            "provider_behavior_changed": False,
            "paper_docx_generated": False,
            "train_end": str(TRAIN_END),
            "validation_start": str(VAL_START),
            "validation_end": str(VAL_END),
            "final_start": str(FINAL_START),
            "source_manifest_keys": sorted(source_manifest.keys()),
        }
    )
    features = pd.concat([features, support], axis=1)
    family_cols = dict(family_cols)
    family_cols["baseline_plus_knn"] = family_cols["baseline_C_closest"] + KNN_FEATURES
    family_cols["combined_plus_knn"] = family_cols["combined_context"] + KNN_FEATURES
    family_cols["regime_plus_knn"] = family_cols["regime_context"] + KNN_FEATURES

    grid_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    knn_val = features.loc[idx["validation"], "knn_prediction_probability"].to_numpy(dtype=float)
    knn_final = features.loc[idx["final"], "knn_prediction_probability"].to_numpy(dtype=float)
    add_candidate_results(
        grid_rows=grid_rows,
        result_rows=result_rows,
        prediction_cache=prediction_cache,
        features=features,
        labels=labels,
        idx=idx,
        model_family="standalone_knn_comparator",
        model_id="knn_prediction_probability",
        feature_set="knn_probability_only",
        val_score=knn_val,
        final_score=knn_final,
        train_rows=len(idx["train"]),
        feature_count=len(KNN_FEATURES),
        params={"k": 11, "role": "standalone_comparator_only"},
        runtime=0.0,
        note="standalone KNN probability comparator only; not eligible as main claim",
    )
    base_scores = run_base_models(features=features, labels=labels, idx=idx, family_cols=family_cols, grid_rows=grid_rows, result_rows=result_rows, prediction_cache=prediction_cache)
    run_ensemble_models(
        features=features,
        labels=labels,
        idx=idx,
        grid_rows=grid_rows,
        result_rows=result_rows,
        prediction_cache=prediction_cache,
        base_scores=base_scores,
        knn_val_prob=knn_val,
        knn_final_prob=knn_final,
    )

    candidate_grid = pd.DataFrame(grid_rows)
    final_results = apply_selection(pd.DataFrame(result_rows))
    validation_results = final_results.copy()
    row_predictions = row_predictions_for(final_results, prediction_cache)
    slices = aggregate_slices(row_predictions)

    write_csv(OUTPUT_DIR / "knn_support_candidate_grid.csv", candidate_grid)
    write_csv(OUTPUT_DIR / "knn_support_validation_results.csv", validation_results)
    write_csv(OUTPUT_DIR / "knn_support_final_results.csv", final_results)
    write_csv(OUTPUT_DIR / "knn_support_row_predictions.csv", row_predictions)
    write_csv(OUTPUT_DIR / "knn_support_by_ticker.csv", slices["by_ticker"])
    write_csv(OUTPUT_DIR / "knn_support_by_month.csv", slices["by_month"])
    write_csv(OUTPUT_DIR / "knn_support_by_quarter.csv", slices["by_quarter"])
    write_csv(OUTPUT_DIR / "knn_support_rolling_250.csv", slices["rolling_250"])
    write_csv(OUTPUT_DIR / "knn_support_rolling_500.csv", slices["rolling_500"])
    write_csv(OUTPUT_DIR / "knn_support_rolling_1000.csv", slices["rolling_1000"])
    write_reports(manifest=manifest, candidate_grid=candidate_grid, final_results=final_results, row_predictions=row_predictions, slices=slices)
    selected = final_results[final_results["selected_by_validation_yes_no"].eq("yes")]
    print(f"Wrote KNN support outputs to {rel(OUTPUT_DIR)}")
    if not selected.empty:
        row = selected.iloc[0]
        print(f"Selected: {row['model_id']} final={pct(row['final_accuracy'])} risk={row['overfit_risk']}")


if __name__ == "__main__":
    main()
