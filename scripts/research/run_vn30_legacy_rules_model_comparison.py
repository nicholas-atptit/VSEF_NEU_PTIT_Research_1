"""Run legacy-compatible VN30 h40 model comparison.

This script intentionally uses the old feature-timestamp split rule so the
61.51% h40 reference and newer model families are compared on the same row
construction.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

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

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_legacy_rules_reference_and_stacking" / "model_comparison"
HORIZON = 40
THRESHOLDS = [0.45, 0.50, 0.55]
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
DEEP_MODELS = ["lstm", "gru", "tcn"]
SEQUENCE_LENGTH = 16
RANDOM_STATE = 42


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
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:+.2f} pp"


def accuracy(y_true: pd.Series | np.ndarray, pred: np.ndarray | pd.Series) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(pred, dtype=int)).mean())


def majority_accuracy(y_true: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    rate = float(np.asarray(y_true, dtype=int).mean())
    return max(rate, 1.0 - rate)


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
        return model.predict_proba(x_data)[:, 1]
    return np.asarray(model.predict(x_data), dtype=float)


def candidate_id(*parts: Any) -> str:
    return "__".join(str(part).replace(".", "p").replace(" ", "_") for part in parts)


def select_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_accuracy = -1.0
    for threshold in THRESHOLDS:
        acc = accuracy(y_true, (np.asarray(probability) >= threshold).astype(int))
        if acc > best_accuracy + 1e-12 or (abs(acc - best_accuracy) <= 1e-12 and abs(threshold - 0.50) < abs(best_threshold - 0.50)):
            best_accuracy = acc
            best_threshold = threshold
    return float(best_threshold), float(best_accuracy)


def load_features() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    features, family_cols, manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    combined = sorted(
        {
            col
            for cols in family_cols.values()
            for col in cols
            if col in features.columns and pd.api.types.is_numeric_dtype(features[col])
        }
    )
    family_cols = {name: [col for col in cols if col in features.columns and pd.api.types.is_numeric_dtype(features[col])] for name, cols in family_cols.items()}
    family_cols["combined_context"] = combined
    manifest["legacy_rules"] = {
        "split_rule": "feature timestamp split with non-null h-step labels",
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
    }
    return features, family_cols, manifest


def label_df_from_series(labels: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"y": labels}, index=labels.index)


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    probability: np.ndarray,
    threshold: float,
    split: str,
    row: dict[str, Any],
) -> pd.DataFrame:
    out = features.reindex(idx)[["datetime", "ticker"]].copy()
    out["experiment_group"] = row["experiment_group"]
    out["method_group"] = row["experiment_group"]
    out["model"] = row["model"]
    out["feature_family"] = row["feature_family"]
    out["horizon"] = row["horizon"]
    out["threshold_policy"] = row["threshold_policy"]
    out["threshold"] = threshold
    out["candidate_id"] = row["candidate_id"]
    out["split"] = split
    out["y_true"] = labels.reindex(idx).astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(probability, dtype=float)
    out["y_pred"] = (out["y_score_or_probability"].to_numpy(dtype=float) >= threshold).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def rolling_means(frame: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    ordered = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    for window in (250, 500, 1000):
        rolling = ordered["correct"].astype(float).rolling(window=window, min_periods=window).mean().dropna()
        out[f"rolling_{window}_mean"] = float(rolling.mean()) if not rolling.empty else math.nan
    return out


def make_result_row(
    model_name: str,
    feature_family: str,
    threshold_policy: str,
    threshold: float,
    feature_count: int,
    train_y: pd.Series,
    val_y: pd.Series,
    final_y: pd.Series,
    val_prob: np.ndarray,
    final_prob: np.ndarray,
    candidate: str,
    experiment_group: str = "single_model",
    status: str = "ok",
) -> dict[str, Any]:
    val_pred = (np.asarray(val_prob) >= threshold).astype(int)
    final_pred = (np.asarray(final_prob) >= threshold).astype(int)
    val_acc = accuracy(val_y, val_pred)
    final_acc = accuracy(final_y, final_pred)
    return {
        "candidate_id": candidate,
        "experiment_group": experiment_group,
        "model": model_name,
        "feature_family": feature_family,
        "horizon": HORIZON,
        "threshold_policy": threshold_policy,
        "threshold": threshold,
        "status": status,
        "selection_source": "validation_only",
        "final_window_role": "scoring_only",
        "final_accuracy_used_for_selection": False,
        "feature_count": feature_count,
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "final_rows": int(len(final_y)),
        "validation_accuracy": val_acc,
        "final_accuracy": final_acc,
        "delta_vs_61_51": final_acc - REFERENCE_FINAL_ACCURACY,
        "validation_majority_baseline": majority_accuracy(val_y),
        "final_majority_baseline": majority_accuracy(final_y),
        "validation_lift_vs_majority": val_acc - majority_accuracy(val_y),
        "final_lift_vs_majority": final_acc - majority_accuracy(final_y),
        "validation_final_gap": final_acc - val_acc,
        "ticker_coverage": 30,
        "full_ticker_coverage": True,
        "leakage_status": "passed_legacy_rules",
    }


def run_classical(features: pd.DataFrame, family_cols: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    labels = add_absolute_labels(features, HORIZON)
    idx = split_indices(features, labels)
    train_y = labels.reindex(idx["train"]).astype(int)
    val_y = labels.reindex(idx["validation"]).astype(int)
    final_y = labels.reindex(idx["final"]).astype(int)
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    grid_rows: list[dict[str, Any]] = []
    for feature_family in FEATURE_FAMILIES:
        cols = family_cols.get(feature_family, [])
        if not cols:
            continue
        x_train = features.reindex(idx["train"])[cols]
        x_val = features.reindex(idx["validation"])[cols]
        x_final = features.reindex(idx["final"])[cols]
        for model_name in CLASSICAL_MODELS:
            grid_rows.append({"model": model_name, "feature_family": feature_family, "horizon": HORIZON, "feature_count": len(cols), "threshold_policies": "fixed_0.50;validation_selected_threshold"})
            model = make_model(model_name)
            if model is None:
                rows.append({"candidate_id": candidate_id("legacy_single", model_name, feature_family, "h40", "skipped"), "experiment_group": "single_model", "model": model_name, "feature_family": feature_family, "horizon": HORIZON, "threshold_policy": "not_run", "status": "skipped_with_reason", "skip_reason": "optional dependency missing"})
                continue
            try:
                model.fit(x_train, train_y)
                val_prob = predict_probability(model, x_val)
                final_prob = predict_probability(model, x_final)
            except Exception as exc:
                rows.append({"candidate_id": candidate_id("legacy_single", model_name, feature_family, "h40", "failed"), "experiment_group": "single_model", "model": model_name, "feature_family": feature_family, "horizon": HORIZON, "threshold_policy": "not_run", "status": "failed", "error": str(exc)[:300]})
                continue
            threshold_specs = [("fixed_0.50", 0.50), ("validation_selected_threshold", select_threshold(val_y, val_prob)[0])]
            for threshold_policy, threshold in threshold_specs:
                cid = candidate_id("legacy_single", model_name, feature_family, "h40", threshold_policy, f"t{threshold:.3f}")
                row = make_result_row(model_name, feature_family, threshold_policy, threshold, len(cols), train_y, val_y, final_y, val_prob, final_prob, cid)
                rows.append(row)
                payloads[cid] = {
                    "row": row,
                    "idx": idx,
                    "labels": labels,
                    "val_prob": val_prob,
                    "final_prob": final_prob,
                    "threshold": threshold,
                }
    return pd.DataFrame(rows), payloads, grid_rows


def run_deep(features: pd.DataFrame, family_cols: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    grid_rows: list[dict[str, Any]] = []
    if torch is None:
        for model_name in DEEP_MODELS:
            rows.append({"candidate_id": candidate_id("legacy_deep", model_name, "h40", "skipped"), "experiment_group": "deep_model", "model": model_name, "feature_family": "baseline_C_closest_sequence32", "horizon": HORIZON, "threshold_policy": "fixed_0.50", "status": "skipped_with_reason", "skip_reason": "torch not importable"})
        return pd.DataFrame(rows), payloads, grid_rows
    feature_cols = select_deep_feature_cols(features, family_cols)
    labels = add_absolute_labels(features, HORIZON)
    label_df = label_df_from_series(labels)
    idx = split_indices(features, labels)
    matrix = standardize_feature_matrix(features, feature_cols, idx["train"])
    x_train, y_train, _train_rows = build_sequences(features, matrix, label_df, idx["train"], SEQUENCE_LENGTH)
    x_val, y_val, val_rows = build_sequences(features, matrix, label_df, idx["validation"], SEQUENCE_LENGTH)
    x_final, y_final, final_rows = build_sequences(features, matrix, label_df, idx["final"], SEQUENCE_LENGTH)
    if len(y_train) == 0 or len(y_val) == 0 or len(y_final) == 0 or len(np.unique(y_train)) < 2:
        for model_name in DEEP_MODELS:
            rows.append({"candidate_id": candidate_id("legacy_deep", model_name, "h40", "skipped"), "experiment_group": "deep_model", "model": model_name, "feature_family": "baseline_C_closest_sequence32", "horizon": HORIZON, "threshold_policy": "fixed_0.50", "status": "skipped_with_reason", "skip_reason": "invalid sequence data shape"})
        return pd.DataFrame(rows), payloads, grid_rows
    train_y = pd.Series(y_train.astype(int))
    val_y = pd.Series(y_val.astype(int))
    final_y = pd.Series(y_final.astype(int))
    for model_name in DEEP_MODELS:
        grid_rows.append({"model": model_name, "feature_family": "baseline_C_closest_sequence32", "horizon": HORIZON, "feature_count": len(feature_cols), "threshold_policies": "fixed_0.50"})
        cid = candidate_id("legacy_deep", model_name, "baseline_C_sequence32", "h40", f"seq{SEQUENCE_LENGTH}", "fixed_0.50")
        try:
            model, meta, val_prob = fit_deep_model(model_name, x_train, y_train, x_val, y_val)
            final_prob = predict_deep(model, x_final)
            row = make_result_row(model_name, "baseline_C_closest_sequence32", "fixed_0.50", 0.50, len(feature_cols), train_y, val_y, final_y, val_prob, final_prob, cid, experiment_group="deep_model")
            row["sequence_length"] = SEQUENCE_LENGTH
            row["best_epoch"] = int(meta.get("best_epoch", 0))
            row["validation_ticker_coverage"] = int(features.reindex(val_rows)["ticker"].nunique())
            row["ticker_coverage"] = int(features.reindex(final_rows)["ticker"].nunique())
            row["full_ticker_coverage"] = int(row["ticker_coverage"]) == 30
            if int(row["ticker_coverage"]) != 30:
                row["leakage_status"] = "not_headline_full_coverage"
            rows.append(row)
            payloads[cid] = {
                "row": row,
                "idx": {"validation": val_rows, "final": final_rows},
                "labels": labels,
                "val_prob": val_prob,
                "final_prob": final_prob,
                "threshold": 0.50,
            }
        except Exception as exc:
            rows.append({"candidate_id": cid, "experiment_group": "deep_model", "model": model_name, "feature_family": "baseline_C_closest_sequence32", "horizon": HORIZON, "threshold_policy": "fixed_0.50", "status": "skipped_with_reason", "skip_reason": str(exc)[:300]})
        gc.collect()
    return pd.DataFrame(rows), payloads, grid_rows


def select_prediction_payloads(results: pd.DataFrame, payloads: dict[str, dict[str, Any]]) -> set[str]:
    selected: set[str] = set()
    fixed = results[(results["status"].eq("ok")) & (results["threshold_policy"].eq("fixed_0.50"))].copy()
    for model_name, group in fixed.groupby("model", sort=True):
        row = group.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0]
        selected.add(str(row["candidate_id"]))
    exact_ref = "legacy_single__logistic_l2__baseline_C_closest__h40__fixed_0p50__t0p500"
    if exact_ref in payloads:
        selected.add(exact_ref)
    ok = results[results["status"].eq("ok")].copy()
    if not ok.empty:
        selected.add(str(ok.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0]["candidate_id"]))
    return selected


def build_row_predictions(features: pd.DataFrame, selected_ids: set[str], payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cid in sorted(selected_ids):
        payload = payloads.get(cid)
        if not payload:
            continue
        row = payload["row"]
        threshold = float(payload["threshold"])
        frames.append(prediction_frame(features, payload["idx"]["validation"], payload["labels"], payload["val_prob"], threshold, "validation", row))
        frames.append(prediction_frame(features, payload["idx"]["final"], payload["labels"], payload["final_prob"], threshold, "final", row))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def slice_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = predictions[predictions["split"].eq("final")].copy()
    if final.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    final["datetime"] = pd.to_datetime(final["datetime"], errors="coerce")
    final["month"] = final["datetime"].dt.to_period("M").astype(str)
    final["quarter"] = final["datetime"].dt.to_period("Q").astype(str)
    by_ticker = final.groupby(["candidate_id", "model", "feature_family", "threshold_policy", "ticker"])["correct"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "accuracy", "count": "rows"})
    by_month = final.groupby(["candidate_id", "model", "feature_family", "threshold_policy", "month"])["correct"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "accuracy", "count": "rows"})
    by_quarter = final.groupby(["candidate_id", "model", "feature_family", "threshold_policy", "quarter"])["correct"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "accuracy", "count": "rows"})
    return by_ticker, by_month, by_quarter


def add_rolling_to_results(results: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    if predictions.empty or out.empty:
        return out
    for cid, group in predictions[predictions["split"].eq("final")].groupby("candidate_id", sort=True):
        stats = rolling_means(group)
        for key, value in stats.items():
            out.loc[out["candidate_id"].eq(cid), key] = value
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, family_cols, manifest = load_features()
    if len(active_stock_tickers()) != 30:
        raise ValueError("legacy comparison requires 30 active tickers")
    classical_results, classical_payloads, classical_grid = run_classical(features, family_cols)
    deep_results, deep_payloads, deep_grid = run_deep(features, family_cols)
    results = pd.concat([classical_results, deep_results], ignore_index=True, sort=False)
    payloads = {**classical_payloads, **deep_payloads}
    selected_ids = select_prediction_payloads(results, payloads)
    predictions = build_row_predictions(features, selected_ids, payloads)
    results = add_rolling_to_results(results, predictions)
    by_ticker, by_month, by_quarter = slice_outputs(predictions)
    grid = pd.DataFrame(classical_grid + deep_grid)
    validation_results = results.copy()
    final_results = results[results["status"].eq("ok")].copy()
    grid.to_csv(OUTPUT_DIR / "legacy_model_candidate_grid.csv", index=False)
    validation_results.to_csv(OUTPUT_DIR / "legacy_model_validation_results.csv", index=False)
    final_results.to_csv(OUTPUT_DIR / "legacy_model_final_results.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "legacy_model_row_predictions.csv", index=False)
    by_ticker.to_csv(OUTPUT_DIR / "legacy_model_by_ticker.csv", index=False)
    by_month.to_csv(OUTPUT_DIR / "legacy_model_by_month.csv", index=False)
    by_quarter.to_csv(OUTPUT_DIR / "legacy_model_by_quarter.csv", index=False)
    write_json(OUTPUT_DIR / "legacy_model_manifest.json", {"data_fetch": False, "provider_behavior_changed": False, "model_training": True, "model_selection": "validation_only", "final_window_role": "scoring_only", "split_rule": "legacy_feature_timestamp_split", "feature_manifest": manifest, "row_prediction_candidate_ids": sorted(selected_ids)})
    best = final_results.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0].to_dict() if not final_results.empty else {}
    lines = [
        "# VN30 Legacy Rules Model Comparison",
        "",
        f"- Candidate grid rows: {len(grid)}.",
        f"- Scored candidate rows: {len(final_results)}.",
        f"- Row-prediction candidate count: {len(selected_ids)}.",
        f"- Best validation-selected single/deep model: `{best.get('candidate_id', '')}`.",
        f"- Validation accuracy: {pct(best.get('validation_accuracy'))}.",
        f"- Final accuracy: {pct(best.get('final_accuracy'))}.",
        f"- Delta vs 61.51%: {pp(best.get('delta_vs_61_51'))}.",
        "- Data fetched: no.",
        "- Final score used for selection: no.",
    ]
    (OUTPUT_DIR / "legacy_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"legacy_model_comparison_complete best={best.get('candidate_id', '')} final={pct(best.get('final_accuracy'))} output_dir={rel(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
