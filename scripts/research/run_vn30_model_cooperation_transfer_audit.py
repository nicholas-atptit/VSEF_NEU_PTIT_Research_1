"""Run VN30 model cooperation and transfer-audit experiment."""

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "model_cooperation_transfer_audit"
FIGURE_DIR = OUTPUT_DIR / "figures"
PRIMARY_HORIZON = 40
THRESHOLD_GRID = [0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]
RANDOM_STATE = 42
CURRENT_MAIN_FINAL_ACCURACY = 0.6163475699558174
BEST_DESCRIPTIVE_FINAL_ACCURACY = 0.6332842415316642
CURRENT_MAIN_LABEL = "Logistic L2, baseline_C_closest, h40, validation-selected threshold 0.55, final accuracy 61.63%"
BEST_DESCRIPTIVE_LABEL = "bull_bear_sideway_router, h40, fixed 0.50, final accuracy 63.33%, not claim-eligible"


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
    return universe.candidate_id("coop", *parts)


def config_tag(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, (list, tuple)):
            text = f"{len(value)}items" if key in {"base_models", "features"} else "x".join(str(item) for item in value)
        else:
            text = str(value)
        parts.append(f"{key}{text}".replace(".", "p").replace(" ", "_").replace(":", "_"))
    return "_".join(parts)[:100]


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
            best_accuracy = acc
            best_threshold = float(threshold)
    specs.append(("validation_selected_threshold", best_threshold))
    return specs


def predict_score(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    return np.clip(universe.predict_score(model, x_data), 0.0, 1.0)


def estimator(model_id: str, params: dict[str, Any]) -> Any | None:
    if model_id == "logistic_l2":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(max_iter=1200, solver="liblinear", penalty="l2", C=float(params.get("C", 0.3)), class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
    if model_id == "elastic_net":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(max_iter=2500, solver="saga", penalty="elasticnet", C=float(params.get("C", 0.3)), l1_ratio=float(params.get("l1_ratio", 0.2)), class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
    if model_id == "linear_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearSVC(C=float(params.get("C", 0.3)), class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000, dual="auto")),
            ]
        )
    if model_id == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=int(params.get("n_estimators", 140)), max_depth=params.get("max_depth", 8), min_samples_leaf=int(params.get("min_samples_leaf", 10)), max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
    if model_id == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=int(params.get("n_estimators", 140)), max_depth=params.get("max_depth", 8), min_samples_leaf=int(params.get("min_samples_leaf", 10)), max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
    if model_id == "xgboost" and XGBClassifier is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(n_estimators=int(params.get("n_estimators", 100)), max_depth=int(params.get("max_depth", 3)), learning_rate=float(params.get("learning_rate", 0.05)), min_child_weight=8, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=2)),
            ]
        )
    if model_id == "lightgbm" and LGBMClassifier is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMClassifier(n_estimators=int(params.get("n_estimators", 100)), num_leaves=int(params.get("num_leaves", 15)), max_depth=int(params.get("max_depth", 3)), learning_rate=float(params.get("learning_rate", 0.05)), min_child_samples=35, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, verbose=-1, n_jobs=2)),
            ]
        )
    if model_id == "catboost" and CatBoostClassifier is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostClassifier(iterations=int(params.get("iterations", 100)), depth=int(params.get("depth", 4)), learning_rate=float(params.get("learning_rate", 0.05)), loss_function="Logloss", random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)),
            ]
        )
    return None


def base_model_specs() -> list[tuple[str, dict[str, Any]]]:
    specs = [
        ("logistic_l2", {"C": 0.3}),
        ("elastic_net", {"C": 0.3, "l1_ratio": 0.2}),
        ("linear_svm", {"C": 0.3}),
        ("random_forest", {"n_estimators": 140, "max_depth": 8, "min_samples_leaf": 10}),
        ("extra_trees", {"n_estimators": 140, "max_depth": 8, "min_samples_leaf": 10}),
        ("xgboost", {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}),
        ("lightgbm", {"n_estimators": 100, "max_depth": 3, "num_leaves": 15, "learning_rate": 0.05}),
    ]
    if CatBoostClassifier is not None:
        specs.append(("catboost", {"iterations": 100, "depth": 4, "learning_rate": 0.05}))
    return specs


def compute_base_predictions(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    labels: pd.Series,
    idx: dict[str, pd.Index],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    val_scores: dict[str, np.ndarray] = {}
    final_scores: dict[str, np.ndarray] = {}
    train_y = labels.loc[idx["train"]].astype(int)
    cols = family_cols["baseline_C_closest"]
    for model_id, params in base_model_specs():
        model = estimator(model_id, params)
        if model is None:
            continue
        start = time.perf_counter()
        model.fit(features.loc[idx["train"], cols], train_y)
        runtime = time.perf_counter() - start
        val_score = predict_score(model, features.loc[idx["validation"], cols])
        final_score = predict_score(model, features.loc[idx["final"], cols])
        key = model_id
        val_scores[key] = val_score
        final_scores[key] = final_score
        val_pred = (val_score >= 0.50).astype(int)
        y_val = labels.loc[idx["validation"]].astype(int).to_numpy()
        val_frame = universe.prediction_frame(features, idx["validation"], labels, val_score, val_pred, model_group="base_model_pool", model_id=model_id, feature_family="baseline_C_closest", horizon=PRIMARY_HORIZON, threshold_policy="fixed_0.50", threshold=0.50, candidate=f"base_{model_id}", split="validation")
        metrics = fair.validation_metric_overlay(val_frame)
        rows.append(
            {
                "base_model": model_id,
                "validation_accuracy": float((val_pred == y_val).mean()),
                "validation_balanced_accuracy": metrics["validation_balanced_accuracy"],
                "validation_lift_over_majority": metrics["validation_lift_over_majority"],
                "validation_rolling_stability": metrics["validation_rolling_stability"],
                "validation_instability": metrics["validation_instability"],
                "runtime_seconds": runtime,
            }
        )
    # Add validation-safe router and technical-rule signals to the cooperation pool.
    cols_base = family_cols["baseline_C_closest"]
    try:
        val_score, final_score = universe.fit_router_models(features, labels, idx, cols_base, "market_direction_regime")
        val_scores["regime_router"] = np.clip(val_score, 0.0, 1.0)
        final_scores["regime_router"] = np.clip(final_score, 0.0, 1.0)
        val_frame = universe.prediction_frame(features, idx["validation"], labels, val_scores["regime_router"], (val_scores["regime_router"] >= 0.50).astype(int), model_group="base_model_pool", model_id="regime_router", feature_family="baseline_C_closest", horizon=PRIMARY_HORIZON, threshold_policy="fixed_0.50", threshold=0.50, candidate="base_regime_router", split="validation")
        metrics = fair.validation_metric_overlay(val_frame)
        rows.append({"base_model": "regime_router", "validation_accuracy": float(val_frame["correct"].mean()), "validation_balanced_accuracy": metrics["validation_balanced_accuracy"], "validation_lift_over_majority": metrics["validation_lift_over_majority"], "validation_rolling_stability": metrics["validation_rolling_stability"], "validation_instability": metrics["validation_instability"], "runtime_seconds": 0.0})
    except Exception:
        pass
    try:
        train_majority = int(float(train_y.mean()) >= 0.5)
        score = pd.Series(universe.score_series_rule(features, labels, PRIMARY_HORIZON, "macd_rule", train_majority), index=features.index).fillna(float(train_majority)).clip(0.0, 1.0)
        val_scores["technical_rule_macd"] = score.loc[idx["validation"]].to_numpy(dtype=float)
        final_scores["technical_rule_macd"] = score.loc[idx["final"]].to_numpy(dtype=float)
        val_frame = universe.prediction_frame(features, idx["validation"], labels, val_scores["technical_rule_macd"], (val_scores["technical_rule_macd"] >= 0.50).astype(int), model_group="base_model_pool", model_id="technical_rule_macd", feature_family="ex_ante_rule", horizon=PRIMARY_HORIZON, threshold_policy="fixed_0.50", threshold=0.50, candidate="base_technical_rule_macd", split="validation")
        metrics = fair.validation_metric_overlay(val_frame)
        rows.append({"base_model": "technical_rule_macd", "validation_accuracy": float(val_frame["correct"].mean()), "validation_balanced_accuracy": metrics["validation_balanced_accuracy"], "validation_lift_over_majority": metrics["validation_lift_over_majority"], "validation_rolling_stability": metrics["validation_rolling_stability"], "validation_instability": metrics["validation_instability"], "runtime_seconds": 0.0})
    except Exception:
        pass
    knn_path = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "knn_support" / "knn_support_row_predictions.csv"
    if knn_path.exists():
        try:
            knn = pd.read_csv(knn_path, usecols=["datetime", "ticker", "split", "y_score_or_probability"], low_memory=False)
            for split_name, target_idx, target in [("validation", idx["validation"], val_scores), ("final", idx["final"], final_scores)]:
                base = features.loc[target_idx, ["datetime", "ticker"]].copy()
                base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce").astype(str)
                part = knn[knn["split"].astype(str).eq(split_name)].drop_duplicates(["datetime", "ticker"])
                part["datetime"] = pd.to_datetime(part["datetime"], errors="coerce").astype(str)
                merged = base.merge(part, on=["datetime", "ticker"], how="left")
                target["knn_probability"] = pd.to_numeric(merged["y_score_or_probability"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
            rows.append({"base_model": "knn_probability", "validation_accuracy": float(((val_scores["knn_probability"] >= 0.50).astype(int) == labels.loc[idx["validation"]].astype(int).to_numpy()).mean()), "validation_balanced_accuracy": math.nan, "validation_lift_over_majority": math.nan, "validation_rolling_stability": math.nan, "validation_instability": math.nan, "runtime_seconds": 0.0})
        except Exception:
            pass
    return pd.DataFrame(rows), pd.DataFrame(val_scores), pd.DataFrame(final_scores)


def add_grid_row(grid_rows: list[dict[str, Any]], row: dict[str, Any], status: str = "run", reason: str = "") -> None:
    grid_rows.append(
        {
            "track": row["track"],
            "candidate_id": row["candidate_id"],
            "base_models_used": row.get("base_models_used", ""),
            "meta_model_if_any": row.get("meta_model_if_any", ""),
            "feature_family": row.get("feature_family", ""),
            "horizon": PRIMARY_HORIZON,
            "threshold_policy": row.get("threshold_policy", ""),
            "planned_status": status,
            "selection_source": "validation_only",
            "final_accuracy_used_for_selection": False,
            "ticker_subset": False,
            "confidence_abstention": False,
            "topk_substitution": False,
            "reason": reason,
        }
    )


def add_result(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    idx: dict[str, pd.Index],
    result_rows: list[dict[str, Any]],
    prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    grid_rows: list[dict[str, Any]],
    track: str,
    model_id: str,
    feature_family: str,
    val_score: np.ndarray,
    final_score: np.ndarray,
    base_models_used: list[str],
    meta_model_if_any: str = "",
    params: dict[str, Any] | None = None,
    runtime: float = 0.0,
    note: str = "",
) -> None:
    val_y = labels.loc[idx["validation"]].astype(int)
    for threshold_policy, threshold in threshold_specs(val_y, val_score):
        cid = candidate_id(track, model_id, feature_family, config_tag(params or {}), threshold_policy, f"t{threshold:.3f}")
        val_frame = universe.prediction_frame(features, idx["validation"], labels, val_score, (val_score >= threshold).astype(int), model_group=track, model_id=model_id, feature_family=feature_family, horizon=PRIMARY_HORIZON, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="validation")
        final_frame = universe.prediction_frame(features, idx["final"], labels, final_score, (final_score >= threshold).astype(int), model_group=track, model_id=model_id, feature_family=feature_family, horizon=PRIMARY_HORIZON, threshold_policy=threshold_policy, threshold=threshold, candidate=cid, split="final")
        row = universe.result_row(candidate=cid, model_group=track, model_id=model_id, feature_family=feature_family, horizon=PRIMARY_HORIZON, threshold_policy=threshold_policy, threshold=threshold, validation_frame=val_frame, final_frame=final_frame, train_rows=len(idx["train"]), feature_count=len(base_models_used), reason_not_claim_eligible="not selected by validation-only cooperation objective", implementation_note=note)
        row.update(fair.validation_metric_overlay(val_frame))
        row["track"] = track
        row["base_models_used"] = ",".join(base_models_used)
        row["meta_model_if_any"] = meta_model_if_any
        row["beats_61_63_yes_no"] = "yes" if as_float(row["final_accuracy"]) > CURRENT_MAIN_FINAL_ACCURACY else "no"
        row["beats_63_33_yes_no"] = "yes" if as_float(row["final_accuracy"]) > BEST_DESCRIPTIVE_FINAL_ACCURACY else "no"
        row["selected_by_validation_yes_no"] = "no"
        row["claim_eligible_yes_no"] = "no"
        row["reason_not_claim_eligible"] = "not selected by validation-only cooperation objective"
        row["fit_runtime_seconds"] = runtime
        row["selection_source"] = "validation_only"
        row["final_accuracy_used_for_selection"] = False
        row["ticker_subset"] = False
        row["confidence_abstention"] = False
        row["topk_substitution"] = False
        row["hyperparameters"] = json.dumps(json_safe(params or {}), sort_keys=True)
        risk, risk_reason = fair.classify_overfit(pd.Series(row))
        row["overfit_risk"] = risk
        row["overfit_risk_reason"] = risk_reason
        result_rows.append(row)
        prediction_cache[cid] = (val_frame, final_frame)
        add_grid_row(grid_rows, row)


def validation_weight_table(base_metrics: pd.DataFrame) -> pd.DataFrame:
    work = base_metrics.copy()
    if work.empty:
        return work
    acc = work["validation_accuracy"].fillna(0.0).clip(lower=0.0)
    lift = work["validation_lift_over_majority"].fillna(0.0).clip(lower=0.0)
    stability = work["validation_rolling_stability"].fillna(work["validation_accuracy"]).clip(lower=0.0)
    conservative = acc * stability / (1.0 + work["validation_instability"].fillna(0.0).clip(lower=0.0))
    for name, values in [("validation_accuracy_weight", acc), ("validation_lift_weight", lift), ("validation_stability_weight", stability), ("conservative_weight", conservative)]:
        denom = float(values.sum())
        work[name] = values / denom if denom > 0 else 1.0 / len(work)
    work["unweighted_weight"] = 1.0 / len(work)
    return work


def weighted_vote(matrix: pd.DataFrame, weights: pd.Series) -> np.ndarray:
    cols = list(matrix.columns)
    aligned = weights.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    if aligned.sum() <= 0:
        aligned = np.repeat(1.0 / len(cols), len(cols))
    else:
        aligned = aligned / aligned.sum()
    return np.clip(matrix[cols].to_numpy(dtype=float) @ aligned, 0.0, 1.0)


def run_soft_voting(features: pd.DataFrame, labels: pd.Series, idx: dict[str, pd.Index], base_metrics: pd.DataFrame, val_scores: pd.DataFrame, final_scores: pd.DataFrame, result_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    weights = validation_weight_table(base_metrics).set_index("base_model")
    variants = {
        "unweighted_soft_vote": "unweighted_weight",
        "validation_accuracy_weighted_soft_vote": "validation_accuracy_weight",
        "validation_lift_weighted_soft_vote": "validation_lift_weight",
        "validation_stability_weighted_soft_vote": "validation_stability_weight",
        "conservative_soft_vote": "conservative_weight",
    }
    for model_id, weight_col in variants.items():
        val = weighted_vote(val_scores, weights[weight_col])
        final = weighted_vote(final_scores, weights[weight_col])
        add_result(features=features, labels=labels, idx=idx, result_rows=result_rows, prediction_cache=prediction_cache, grid_rows=grid_rows, track="soft_voting", model_id=model_id, feature_family="base_model_probabilities", val_score=val, final_score=final, base_models_used=list(val_scores.columns), params={"weight_column": weight_col}, note="ensemble weights selected from validation metrics only")
    return weights.reset_index()


def run_model_as_feature(features: pd.DataFrame, labels: pd.Series, idx: dict[str, pd.Index], val_scores: pd.DataFrame, final_scores: pd.DataFrame, result_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> dict[str, Any]:
    meta_specs: list[tuple[str, Any | None]] = [
        ("logistic_meta", estimator("logistic_l2", {"C": 0.3})),
        ("elastic_net_meta", estimator("elastic_net", {"C": 0.3, "l1_ratio": 0.2})),
        ("lightgbm_meta", estimator("lightgbm", {"n_estimators": 60, "max_depth": 2, "num_leaves": 7, "learning_rate": 0.05})),
        ("xgboost_meta", estimator("xgboost", {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.05})),
        ("random_forest_meta", estimator("random_forest", {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 20})),
    ]
    y_val = labels.loc[idx["validation"]].astype(int)
    manifest = {"auxiliary_prediction_features": list(val_scores.columns), "meta_train_scope": "validation_predictions_only", "final_labels_used": False, "meta_models": []}
    for model_id, model in meta_specs:
        if model is None:
            continue
        start = time.perf_counter()
        model.fit(val_scores, y_val)
        runtime = time.perf_counter() - start
        val = predict_score(model, val_scores)
        final = predict_score(model, final_scores)
        manifest["meta_models"].append(model_id)
        add_result(features=features, labels=labels, idx=idx, result_rows=result_rows, prediction_cache=prediction_cache, grid_rows=grid_rows, track="model_as_feature", model_id=model_id, feature_family="validation_base_predictions", val_score=val, final_score=final, base_models_used=list(val_scores.columns), meta_model_if_any=model_id, params={"meta_train_scope": "validation_predictions_only"}, runtime=runtime, note="meta-model trained only on validation base predictions; final labels not used")
    return manifest


def score_with_group_thresholds(score: np.ndarray, groups: pd.Series, thresholds: dict[str, float], default: float = 0.50) -> np.ndarray:
    return np.asarray([int(float(s) >= thresholds.get(str(g), default)) for s, g in zip(score, groups.astype(str).to_numpy())], dtype=float)


def group_thresholds(y: pd.Series, score: np.ndarray, groups: pd.Series) -> dict[str, float]:
    work = pd.DataFrame({"y": y.astype(int).to_numpy(), "score": score, "group": groups.astype(str).to_numpy()})
    out: dict[str, float] = {}
    for group, frame in work.groupby("group"):
        if len(frame) < 30:
            continue
        best = 0.50
        best_acc = -1.0
        for threshold in THRESHOLD_GRID:
            acc = float(((frame["score"].to_numpy() >= threshold).astype(int) == frame["y"].to_numpy()).mean())
            if acc > best_acc:
                best_acc = acc
                best = threshold
        out[str(group)] = float(best)
    return out


def run_calibration(features: pd.DataFrame, labels: pd.Series, idx: dict[str, pd.Index], base_logistic_val: np.ndarray, base_logistic_final: np.ndarray, result_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    y_val = labels.loc[idx["validation"]].astype(int)
    # Calibration is fit on the first half of validation diagnostics, then applied to the full validation/final scores.
    order = features.loc[idx["validation"]].sort_values(["datetime", "ticker"]).index
    cal_ids = order[: max(100, len(order) // 2)]
    cal_pos = pd.Index(idx["validation"]).get_indexer(cal_ids)
    cal_score = base_logistic_val[cal_pos]
    cal_y = labels.loc[cal_ids].astype(int)
    variants: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    platt = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    platt.fit(cal_score.reshape(-1, 1), cal_y)
    variants["platt_calibration"] = (platt.predict_proba(base_logistic_val.reshape(-1, 1))[:, 1], platt.predict_proba(base_logistic_final.reshape(-1, 1))[:, 1], {"calibration": "platt", "fit_scope": "first_half_validation"})
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(cal_score, cal_y.to_numpy())
    variants["isotonic_calibration"] = (isotonic.predict(base_logistic_val), isotonic.predict(base_logistic_final), {"calibration": "isotonic", "fit_scope": "first_half_validation"})
    variants["validation_selected_global_threshold"] = (base_logistic_val, base_logistic_final, {"threshold_scope": "global_validation"})
    for col, name in [("ticker", "ticker_specific_threshold"), ("market_direction_regime", "regime_specific_threshold"), ("volatility_regime", "volatility_specific_threshold")]:
        thresholds = group_thresholds(y_val, base_logistic_val, features.loc[idx["validation"], col])
        val_binary = score_with_group_thresholds(base_logistic_val, features.loc[idx["validation"], col], thresholds)
        final_binary = score_with_group_thresholds(base_logistic_final, features.loc[idx["final"], col], thresholds)
        variants[name] = (val_binary, final_binary, {"threshold_scope": col, "thresholds": thresholds})
    for model_id, (val, final, params) in variants.items():
        add_result(features=features, labels=labels, idx=idx, result_rows=result_rows, prediction_cache=prediction_cache, grid_rows=grid_rows, track="calibration_threshold_cooperation", model_id=model_id, feature_family="logistic_l2_probability", val_score=np.asarray(val, dtype=float), final_score=np.asarray(final, dtype=float), base_models_used=["logistic_l2"], params=params, note="calibration/threshold cooperation selected on validation only; no final labels used")
        rows.append({"model_id": model_id, **params})
    return pd.DataFrame(rows)


def run_error_correction(features: pd.DataFrame, labels: pd.Series, idx: dict[str, pd.Index], val_scores: pd.DataFrame, final_scores: pd.DataFrame, result_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    if "logistic_l2" not in val_scores.columns:
        return pd.DataFrame()
    y_val = labels.loc[idx["validation"]].astype(int)
    base_val = val_scores["logistic_l2"].to_numpy(dtype=float)
    base_final = final_scores["logistic_l2"].to_numpy(dtype=float)
    base_val_pred = (base_val >= 0.55).astype(int)
    wrong = (base_val_pred != y_val.to_numpy()).astype(int)
    diag_val = val_scores.copy()
    diag_final = final_scores.copy()
    for col in ["rolling_return_vol_20", "vnindex_vol_20_lag_ctx", "vn30_vol_20_lag_ctx"]:
        if col in features.columns:
            diag_val[col] = pd.to_numeric(features.loc[idx["validation"], col], errors="coerce").fillna(0.0).to_numpy()
            diag_final[col] = pd.to_numeric(features.loc[idx["final"], col], errors="coerce").fillna(0.0).to_numpy()
            break
    summary_rows = []
    specs = [
        ("error_logistic", estimator("logistic_l2", {"C": 0.3})),
        ("error_random_forest", estimator("random_forest", {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 20})),
        ("error_lightgbm", estimator("lightgbm", {"n_estimators": 60, "max_depth": 2, "num_leaves": 7, "learning_rate": 0.05})),
        ("error_xgboost", estimator("xgboost", {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.05})),
    ]
    alt_model = "regime_router" if "regime_router" in val_scores.columns else val_scores.columns[0]
    for model_id, model in specs:
        if model is None:
            continue
        model.fit(diag_val, wrong)
        err_val = predict_score(model, diag_val)
        err_final = predict_score(model, diag_final)
        for rule in ["threshold_adjustment", "probability_shrinkage", "fallback_to_conservative_prediction", "fallback_to_validation_selected_model"]:
            if rule == "threshold_adjustment":
                val = base_val - 0.08 * err_val
                final = base_final - 0.08 * err_final
            elif rule == "probability_shrinkage":
                val = 0.50 + (base_val - 0.50) * (1.0 - err_val)
                final = 0.50 + (base_final - 0.50) * (1.0 - err_final)
            elif rule == "fallback_to_conservative_prediction":
                val = np.where(err_val >= 0.60, 0.50, base_val)
                final = np.where(err_final >= 0.60, 0.50, base_final)
            else:
                val = np.where(err_val >= 0.60, val_scores[alt_model].to_numpy(dtype=float), base_val)
                final = np.where(err_final >= 0.60, final_scores[alt_model].to_numpy(dtype=float), base_final)
            add_result(features=features, labels=labels, idx=idx, result_rows=result_rows, prediction_cache=prediction_cache, grid_rows=grid_rows, track="error_correction", model_id=f"{model_id}_{rule}", feature_family="validation_diagnostics", val_score=np.clip(val, 0.0, 1.0), final_score=np.clip(final, 0.0, 1.0), base_models_used=["logistic_l2", alt_model], meta_model_if_any=model_id, params={"rule": rule, "error_train_scope": "validation_only"}, note="error model trained only on validation diagnostics; final labels not used")
            summary_rows.append({"error_model": model_id, "rule": rule, "base_model": "logistic_l2", "fallback_model": alt_model})
    return pd.DataFrame(summary_rows)


def run_mixture(features: pd.DataFrame, labels: pd.Series, idx: dict[str, pd.Index], val_scores: pd.DataFrame, final_scores: pd.DataFrame, result_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    y_val = labels.loc[idx["validation"]].astype(int)
    val_acc = {col: float(((val_scores[col].to_numpy() >= 0.50).astype(int) == y_val.to_numpy()).mean()) for col in val_scores.columns}
    best_global = max(val_acc, key=val_acc.get)
    rows = [{"router": "validation_selected_global_router", "selected": best_global}]
    add_result(features=features, labels=labels, idx=idx, result_rows=result_rows, prediction_cache=prediction_cache, grid_rows=grid_rows, track="mixture_of_experts", model_id="validation_selected_global_router", feature_family="base_model_probabilities", val_score=val_scores[best_global].to_numpy(dtype=float), final_score=final_scores[best_global].to_numpy(dtype=float), base_models_used=list(val_scores.columns), params={"selected_model": best_global}, note="global expert selected by validation accuracy only")
    for col, router_name in [("market_direction_regime", "regime_router"), ("volatility_regime", "volatility_router"), ("ticker", "ticker_group_router")]:
        mapping = {}
        val_out = np.zeros(len(idx["validation"]), dtype=float)
        final_out = np.zeros(len(idx["final"]), dtype=float)
        for group, frame in features.loc[idx["validation"]].groupby(col, sort=True):
            positions = pd.Index(idx["validation"]).get_indexer(frame.index)
            if len(positions) < 40:
                selected = "logistic_l2" if "logistic_l2" in val_scores.columns else best_global
            else:
                scores = {model: float(((val_scores[model].to_numpy()[positions] >= 0.50).astype(int) == y_val.to_numpy()[positions]).mean()) for model in val_scores.columns}
                selected = max(scores, key=scores.get)
            mapping[str(group)] = selected
        for i, group in enumerate(features.loc[idx["validation"], col].astype(str).to_numpy()):
            val_out[i] = val_scores[mapping.get(group, best_global)].iloc[i]
        for i, group in enumerate(features.loc[idx["final"], col].astype(str).to_numpy()):
            final_out[i] = final_scores[mapping.get(group, best_global)].iloc[i]
        add_result(features=features, labels=labels, idx=idx, result_rows=result_rows, prediction_cache=prediction_cache, grid_rows=grid_rows, track="mixture_of_experts", model_id=router_name, feature_family=col, val_score=val_out, final_score=final_out, base_models_used=list(val_scores.columns), params={"router_column": col, "mapping": mapping}, note="router mapping selected on validation only using ex-ante labels")
        rows.append({"router": router_name, "selected": json.dumps(mapping, sort_keys=True)})
    return pd.DataFrame(rows)


def select_compact_features(features: pd.DataFrame, family_cols: dict[str, list[str]], labels: pd.Series, idx: dict[str, pd.Index]) -> list[str]:
    train_y = labels.loc[idx["train"]].astype(int)
    cols = [col for col in family_cols["combined_context"] if col in features.columns and pd.api.types.is_numeric_dtype(features[col])]
    train = features.loc[idx["train"], cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_array = train_y.to_numpy(dtype=float)
    corr = train.apply(lambda col: abs(pd.Series(col.to_numpy(dtype=float)).corr(pd.Series(y_array))), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rf = RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train, train_y)
    imp = pd.Series(rf.feature_importances_, index=cols)
    score = corr.rank(pct=True) + imp.rank(pct=True)
    return score.sort_values(ascending=False).head(24).index.tolist()


def run_feature_selection(features: pd.DataFrame, family_cols: dict[str, list[str]], labels: pd.Series, idx: dict[str, pd.Index], result_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, list[str]]:
    compact = select_compact_features(features, family_cols, labels, idx)
    train_y = labels.loc[idx["train"]].astype(int)
    specs = ["logistic_l2", "elastic_net", "xgboost", "lightgbm"]
    for model_id in specs:
        model = estimator(model_id, {"C": 0.3, "l1_ratio": 0.2, "n_estimators": 90, "max_depth": 3, "learning_rate": 0.05, "num_leaves": 15})
        if model is None:
            continue
        start = time.perf_counter()
        model.fit(features.loc[idx["train"], compact], train_y)
        runtime = time.perf_counter() - start
        val = predict_score(model, features.loc[idx["validation"], compact])
        final = predict_score(model, features.loc[idx["final"], compact])
        add_result(features=features, labels=labels, idx=idx, result_rows=result_rows, prediction_cache=prediction_cache, grid_rows=grid_rows, track="feature_selection_cooperation", model_id=f"{model_id}_compact", feature_family="compact_robust_feature_set", val_score=val, final_score=final, base_models_used=[model_id], params={"features": compact}, runtime=runtime, note="compact features selected using train correlation and train tree importance only; no final feature selection")
    return pd.DataFrame({"selected_feature": compact}), compact


def run_base_candidate(features: pd.DataFrame, labels: pd.Series, idx: dict[str, pd.Index], val_scores: pd.DataFrame, final_scores: pd.DataFrame, result_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    if "logistic_l2" in val_scores.columns:
        add_result(features=features, labels=labels, idx=idx, result_rows=result_rows, prediction_cache=prediction_cache, grid_rows=grid_rows, track="reference_base", model_id="logistic_l2_reference_threshold_0p55", feature_family="baseline_C_closest", val_score=val_scores["logistic_l2"].to_numpy(dtype=float), final_score=final_scores["logistic_l2"].to_numpy(dtype=float), base_models_used=["logistic_l2"], params={"threshold_reference": 0.55}, note="reference base model included for cooperation effect comparison")


def refresh_selected(final_results: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    key_cols = ["candidate_id", "selection_objective", "selection_metric", "selection_metric_value", "selection_scope"]
    keep = selected[[col for col in key_cols if col in selected.columns]].copy()
    return keep.merge(final_results, on="candidate_id", how="left")


def select_by_objectives(final_results: pd.DataFrame) -> pd.DataFrame:
    pool = final_results[
        final_results["status"].astype(str).eq("ok")
        & final_results["full_ticker_coverage"].astype(bool)
        & ~final_results["track"].astype(str).eq("reference_base")
        & final_results["validation_accuracy"].apply(lambda value: math.isfinite(as_float(value)))
    ].copy()
    if pool.empty:
        return pd.DataFrame()
    pool["balanced_transfer_score"] = compute_balanced_transfer_score(pool)
    objectives = [
        ("max_validation_accuracy", "validation_accuracy", False),
        ("max_validation_balanced_accuracy", "validation_balanced_accuracy", False),
        ("max_validation_lift_over_majority", "validation_lift_over_majority", False),
        ("max_validation_rolling_stability", "validation_rolling_stability", False),
        ("max_validation_monthly_stability", "validation_monthly_stability", False),
        ("max_validation_ticker_stability", "validation_ticker_stability", False),
        ("min_validation_instability", "validation_instability", True),
        ("balanced_transfer_score", "balanced_transfer_score", False),
    ]
    rows = []
    for objective, metric, ascending in objectives:
        work = pool[pool[metric].apply(lambda value: math.isfinite(as_float(value)))].copy()
        if work.empty:
            continue
        selected = work.sort_values([metric, "validation_accuracy", "candidate_id"], ascending=[ascending, False, True]).iloc[0].copy()
        selected["selection_objective"] = objective
        selected["selection_metric"] = metric
        selected["selection_metric_value"] = selected.get(metric)
        selected["selection_scope"] = "h40_validation_only"
        rows.append(dict(selected))
    return pd.DataFrame(rows)


def compute_balanced_transfer_score(frame: pd.DataFrame) -> pd.Series:
    work = frame.copy()
    return (
        0.20 * work["validation_accuracy"].fillna(0.0)
        + 0.20 * work["validation_balanced_accuracy"].fillna(work["validation_accuracy"])
        + 0.15 * (0.50 + work["validation_lift_over_majority"].fillna(0.0))
        + 0.15 * work["validation_rolling_stability"].fillna(work["validation_accuracy"])
        + 0.10 * work["validation_monthly_stability"].fillna(work["validation_accuracy"])
        + 0.10 * work["validation_ticker_stability"].fillna(work["validation_accuracy"])
        - 0.10 * work["validation_instability"].fillna(0.0)
    )


def apply_selection(final_results: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = final_results.copy()
    if "balanced_transfer_score" not in out.columns:
        out["balanced_transfer_score"] = np.nan
    out["selected_by_validation_yes_no"] = "no"
    out["selection_objectives_won"] = ""
    if not selected.empty:
        mapping = selected.groupby("candidate_id")["selection_objective"].apply(lambda values: ";".join(sorted(set(map(str, values))))).to_dict()
        for cid, objectives in mapping.items():
            mask = out["candidate_id"].astype(str).eq(str(cid))
            out.loc[mask, "selected_by_validation_yes_no"] = "yes"
            out.loc[mask, "selection_objectives_won"] = objectives
    out["claim_eligible_yes_no"] = "no"
    out["reason_not_claim_eligible"] = "not selected by validation-only cooperation objective"
    eligible = out["selected_by_validation_yes_no"].eq("yes") & out["full_ticker_coverage"].astype(bool) & ~out["overfit_risk"].astype(str).eq("high")
    out.loc[eligible, "claim_eligible_yes_no"] = "yes"
    out.loc[eligible, "reason_not_claim_eligible"] = ""
    out.loc[out["selected_by_validation_yes_no"].eq("yes") & out["overfit_risk"].astype(str).eq("high"), "reason_not_claim_eligible"] = "selected by validation objective but high overfit risk"
    return out


def aggregate_outputs(final_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ok = final_results[final_results["status"].astype(str).eq("ok")].copy()
    agg = {
        "candidates": ("candidate_id", "nunique"),
        "mean_validation_accuracy": ("validation_accuracy", "mean"),
        "mean_final_accuracy": ("final_accuracy", "mean"),
        "best_final_accuracy": ("final_accuracy", "max"),
        "mean_validation_final_gap": ("validation_final_gap", "mean"),
        "mean_rolling_250": ("rolling_250_mean", "mean"),
    }
    by_track = ok.groupby("track").agg(**agg).reset_index() if not ok.empty else pd.DataFrame()
    by_family = ok.groupby("model_id").agg(**agg).reset_index() if not ok.empty else pd.DataFrame()
    transfer = by_track.copy()
    if not transfer.empty:
        gap = ok.assign(abs_gap=ok["validation_final_gap"].abs()).groupby("track")["abs_gap"].median().reset_index(name="median_abs_validation_final_gap")
        transfer = transfer.merge(gap, on="track", how="left")
        transfer["transfer_quality_score"] = transfer["mean_final_accuracy"] - transfer["median_abs_validation_final_gap"].fillna(0.0)
    runtime = ok.groupby(["track", "model_id"], dropna=False).agg(candidates=("candidate_id", "nunique"), mean_runtime_seconds=("fit_runtime_seconds", "mean"), best_final_accuracy=("final_accuracy", "max")).reset_index()
    return {"by_track": by_track, "by_model_family": by_family, "transfer": transfer, "runtime": runtime}


def row_predictions_for(final_results: pd.DataFrame, prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    ids = set(final_results[final_results["selected_by_validation_yes_no"].eq("yes")]["candidate_id"].astype(str))
    ids.update(final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False])["candidate_id"].astype(str).head(8))
    frames: list[pd.DataFrame] = []
    for cid in sorted(ids):
        if cid in prediction_cache:
            frames.extend(prediction_cache[cid])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_figures(final_results: pd.DataFrame, selected: pd.DataFrame, soft_weights: pd.DataFrame, aggregates: dict[str, pd.DataFrame]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    def safe_plot(name: str, func: Callable[[], None]) -> None:
        plt.figure(figsize=(10, 6))
        try:
            func()
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / name, dpi=160)
        finally:
            plt.close()

    safe_plot("fig_cooperation_track_final_accuracy.png", lambda: final_results.groupby("track")["final_accuracy"].max().sort_values().plot(kind="barh", title="Best Final Accuracy by Cooperation Track", xlabel="final accuracy"))
    safe_plot("fig_validation_vs_final_cooperation.png", lambda: (plt.scatter(final_results["validation_accuracy"], final_results["final_accuracy"], s=18, alpha=0.6), plt.xlabel("validation accuracy"), plt.ylabel("final accuracy"), plt.title("Validation vs Final Cooperation Candidates")))
    safe_plot("fig_claim_eligible_vs_descriptive_cooperation.png", lambda: (plt.bar(["claim eligible", "descriptive"], [final_results[final_results["claim_eligible_yes_no"].eq("yes")]["final_accuracy"].max(), final_results["final_accuracy"].max()]), plt.axhline(CURRENT_MAIN_FINAL_ACCURACY, color="red", linestyle="--"), plt.ylabel("final accuracy"), plt.title("Claim Eligible vs Descriptive Cooperation")))
    safe_plot("fig_soft_vote_weights.png", lambda: soft_weights.set_index("base_model")["conservative_weight"].sort_values().plot(kind="barh", title="Conservative Soft Vote Weights", xlabel="weight"))
    for track, name in [
        ("model_as_feature", "fig_model_as_feature_transfer.png"),
        ("error_correction", "fig_error_correction_effect.png"),
        ("mixture_of_experts", "fig_mixture_of_experts_comparison.png"),
        ("calibration_threshold_cooperation", "fig_calibration_cooperation_effect.png"),
        ("feature_selection_cooperation", "fig_feature_selection_cooperation_effect.png"),
    ]:
        safe_plot(name, lambda track=track: final_results[final_results["track"].eq(track)].sort_values("final_accuracy").plot(x="model_id", y="final_accuracy", kind="barh", legend=False, title=f"{track} Final Accuracy", xlabel="final accuracy"))
    safe_plot("fig_current_main_vs_best_cooperation.png", lambda: (plt.bar(["current main", "best cooperation"], [CURRENT_MAIN_FINAL_ACCURACY, final_results["final_accuracy"].max()]), plt.ylabel("final accuracy"), plt.title("Current Main vs Best Cooperation")))
    safe_plot("fig_overfit_risk_cooperation.png", lambda: final_results.groupby(["track", "overfit_risk"]).size().unstack(fill_value=0).plot(kind="bar", stacked=True, title="Overfit Risk by Track", ylabel="candidates"))


def write_reports(final_results: pd.DataFrame, selected: pd.DataFrame, aggregates: dict[str, pd.DataFrame], soft_weights: pd.DataFrame, error_summary: pd.DataFrame, mixture_summary: pd.DataFrame, calibration_summary: pd.DataFrame, feature_summary: pd.DataFrame, compact_features: list[str]) -> None:
    descriptive = final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False]).head(12)
    claim = final_results[final_results["claim_eligible_yes_no"].eq("yes")].sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False])
    lines = [
        "# VN30 Model Cooperation Transfer Audit Summary",
        "",
        f"- Candidate rows: {len(final_results)}.",
        f"- Tracks run: {', '.join(sorted(final_results['track'].astype(str).unique()))}.",
        "- Selection: validation-only.",
        "- Final-window selection: no.",
        "- Data fetch: no.",
        f"- Current main result: {CURRENT_MAIN_LABEL}.",
        f"- Descriptive prior context: {BEST_DESCRIPTIVE_LABEL}.",
        "",
        "## Selected By Objective",
        "",
        fair.markdown_table(selected[["selection_objective", "candidate_id", "track", "model_id", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "overfit_risk"]] if not selected.empty else selected, max_rows=20),
        "",
        "## Descriptive Final Leaderboard",
        "",
        fair.markdown_table(descriptive[["candidate_id", "track", "model_id", "validation_accuracy", "final_accuracy", "selected_by_validation_yes_no", "claim_eligible_yes_no", "overfit_risk"]], max_rows=12),
    ]
    write_markdown(OUTPUT_DIR / "cooperation_summary.md", "\n".join(lines))
    write_markdown(OUTPUT_DIR / "cooperation_claim_boundary.md", "\n".join(["# Cooperation Claim Boundary", "", "- Claim eligibility requires validation-only selection, full 30-stock coverage, leakage audit pass, stability audit, and no high overfit risk.", "- Descriptive final leaderboard rows do not override validation-only selection.", "- No trading, profitability, investment recommendation, or live-deployment claim is made.", "", "## Claim Eligible Rows", "", fair.markdown_table(claim[["candidate_id", "track", "model_id", "validation_accuracy", "final_accuracy", "overfit_risk"]] if not claim.empty else claim, max_rows=20)]))
    write_csv(OUTPUT_DIR / "soft_vote_weights.csv", soft_weights)
    write_csv(OUTPUT_DIR / "error_correction_summary.csv", error_summary)
    write_csv(OUTPUT_DIR / "mixture_of_experts_summary.csv", mixture_summary)
    write_csv(OUTPUT_DIR / "calibration_cooperation_summary.csv", calibration_summary)
    write_csv(OUTPUT_DIR / "feature_selection_cooperation_summary.csv", feature_summary)
    write_json(OUTPUT_DIR / "cooperation_protocol_manifest.json", {"data_fetch": False, "provider_behavior_changed": False, "paper_docx_generated": False, "primary_horizon": PRIMARY_HORIZON, "train_end": str(TRAIN_END), "validation_start": str(VAL_START), "validation_end": str(VAL_END), "final_start": str(FINAL_START), "selection": "validation_only", "compact_feature_count": len(compact_features), "compact_features": compact_features})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if len(active_stock_tickers()) != 30:
        raise RuntimeError("full 30-stock coverage is required")
    features, family_cols, _manifest = universe.prepare_features()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    labels = add_absolute_labels(features, PRIMARY_HORIZON)
    idx = universe.split_indices(features, labels)
    base_metrics, val_scores, final_scores = compute_base_predictions(features, family_cols, labels, idx)

    result_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    prediction_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    run_base_candidate(features, labels, idx, val_scores, final_scores, result_rows, grid_rows, prediction_cache)
    soft_weights = run_soft_voting(features, labels, idx, base_metrics, val_scores, final_scores, result_rows, grid_rows, prediction_cache)
    model_as_feature_manifest = run_model_as_feature(features, labels, idx, val_scores, final_scores, result_rows, grid_rows, prediction_cache)
    error_summary = run_error_correction(features, labels, idx, val_scores, final_scores, result_rows, grid_rows, prediction_cache)
    mixture_summary = run_mixture(features, labels, idx, val_scores, final_scores, result_rows, grid_rows, prediction_cache)
    calibration_summary = run_calibration(features, labels, idx, val_scores["logistic_l2"].to_numpy(dtype=float), final_scores["logistic_l2"].to_numpy(dtype=float), result_rows, grid_rows, prediction_cache)
    feature_summary, compact_features = run_feature_selection(features, family_cols, labels, idx, result_rows, grid_rows, prediction_cache)

    final_results = pd.DataFrame(result_rows)
    final_results["balanced_transfer_score"] = compute_balanced_transfer_score(final_results)
    selected = select_by_objectives(final_results)
    final_results = apply_selection(final_results, selected)
    selected = refresh_selected(final_results, selected)
    validation_results = final_results.copy()
    row_predictions = row_predictions_for(final_results, prediction_cache)
    slices = fair.build_slice_outputs(row_predictions)
    aggregates = aggregate_outputs(final_results)
    overfit = final_results[(final_results["selected_by_validation_yes_no"].eq("yes")) | (final_results["beats_61_63_yes_no"].eq("yes")) | (final_results["beats_63_33_yes_no"].eq("yes"))].copy()

    write_csv(OUTPUT_DIR / "cooperation_candidate_grid.csv", pd.DataFrame(grid_rows))
    write_csv(OUTPUT_DIR / "cooperation_validation_results.csv", validation_results)
    write_csv(OUTPUT_DIR / "cooperation_selected_by_objective.csv", selected)
    write_csv(OUTPUT_DIR / "cooperation_final_results.csv", final_results)
    write_csv(OUTPUT_DIR / "cooperation_row_predictions.csv", row_predictions)
    write_csv(OUTPUT_DIR / "cooperation_by_track.csv", aggregates["by_track"])
    write_csv(OUTPUT_DIR / "cooperation_by_model_family.csv", aggregates["by_model_family"])
    write_csv(OUTPUT_DIR / "cooperation_by_ticker.csv", slices["by_ticker"])
    write_csv(OUTPUT_DIR / "cooperation_by_month.csv", slices["by_month"])
    write_csv(OUTPUT_DIR / "cooperation_by_quarter.csv", slices["by_quarter"])
    write_csv(OUTPUT_DIR / "cooperation_rolling_250.csv", slices["rolling_250"])
    write_csv(OUTPUT_DIR / "cooperation_rolling_500.csv", slices["rolling_500"])
    write_csv(OUTPUT_DIR / "cooperation_rolling_1000.csv", slices["rolling_1000"])
    write_csv(OUTPUT_DIR / "cooperation_transfer_quality.csv", aggregates["transfer"])
    write_csv(OUTPUT_DIR / "cooperation_overfit_risk.csv", overfit)
    write_csv(OUTPUT_DIR / "cooperation_runtime_summary.csv", aggregates["runtime"])
    write_json(OUTPUT_DIR / "model_as_feature_manifest.json", model_as_feature_manifest)
    write_reports(final_results, selected, aggregates, soft_weights, error_summary, mixture_summary, calibration_summary, feature_summary, compact_features)
    make_figures(final_results, selected, soft_weights, aggregates)
    best = selected[selected["selection_objective"].eq("max_validation_accuracy")]
    print(f"Wrote cooperation outputs to {rel(OUTPUT_DIR)}")
    if not best.empty:
        row = best.iloc[0]
        print(f"Best validation-selected: {row['model_id']} final={pct(row['final_accuracy'])}")


if __name__ == "__main__":
    main()
