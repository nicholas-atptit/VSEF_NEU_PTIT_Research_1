"""Run legacy-rule validation-only soft-vote and stacking ensembles."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import REFERENCE_FINAL_ACCURACY  # noqa: E402
from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, rel  # noqa: E402

ROOT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_legacy_rules_reference_and_stacking"
MODEL_DIR = ROOT_DIR / "model_comparison"
OUTPUT_DIR = ROOT_DIR / "stacking"
RANDOM_STATE = 42


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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


def accuracy(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(pred, dtype=int)).mean())


def add_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S")
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["row_key"] = out["datetime"].astype(str) + "|" + out["ticker"].astype(str) + "|" + out["horizon"].astype(str)
    return out


def load_base_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = MODEL_DIR / "legacy_model_row_predictions.csv"
    final_results_path = MODEL_DIR / "legacy_model_final_results.csv"
    if not path.exists() or not final_results_path.exists():
        raise FileNotFoundError("model comparison outputs are required before stacking")
    predictions = pd.read_csv(path, low_memory=False)
    final_results = pd.read_csv(final_results_path)
    predictions = predictions[predictions["threshold_policy"].astype(str).eq("fixed_0.50")].copy()
    predictions = add_key(predictions)
    validation = predictions[predictions["split"].eq("validation")].copy()
    final = predictions[predictions["split"].eq("final")].copy()
    return validation, final, final_results


def select_base_candidates(validation: pd.DataFrame, final: pd.DataFrame, final_results: pd.DataFrame) -> list[str]:
    scored = final_results[(final_results["status"].eq("ok")) & (final_results["threshold_policy"].eq("fixed_0.50"))].copy()
    available = set(validation["candidate_id"].astype(str).unique())
    full_ids: set[str] = set()
    for cid, group in final[final["candidate_id"].isin(available)].groupby("candidate_id", sort=True):
        if len(group) == 4074 and int(group["ticker"].nunique()) == 30:
            full_ids.add(str(cid))
    selected: list[str] = []
    for model_name, group in scored[scored["candidate_id"].isin(full_ids)].groupby("model", sort=True):
        row = group.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0]
        selected.append(str(row["candidate_id"]))
    return sorted(set(selected))


def wide_prob(frame: pd.DataFrame, candidate_ids: list[str]) -> pd.DataFrame:
    scoped = frame[frame["candidate_id"].isin(candidate_ids)].copy()
    return scoped.pivot_table(index="row_key", columns="candidate_id", values="y_score_or_probability", aggfunc="first")


def truth_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("row_key", as_index=False).agg(
        datetime=("datetime", "first"),
        ticker=("ticker", "first"),
        horizon=("horizon", "first"),
        y_true=("y_true", "first"),
    )


def validation_weights(validation: pd.DataFrame, candidate_ids: list[str]) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {
        "validation_accuracy_weighted_soft_vote": {},
        "validation_lift_weighted_soft_vote": {},
        "stability_weighted_soft_vote": {},
    }
    for cid in candidate_ids:
        scoped = validation[validation["candidate_id"].eq(cid)].copy()
        acc = float(scoped["correct"].astype(int).mean()) if not scoped.empty else 0.0
        majority = max(float(scoped["y_true"].astype(int).mean()), 1.0 - float(scoped["y_true"].astype(int).mean())) if not scoped.empty else 0.0
        scoped["month"] = pd.to_datetime(scoped["datetime"], errors="coerce").dt.to_period("M").astype(str)
        monthly = scoped.groupby("month")["correct"].mean()
        stability = float(monthly.min()) if not monthly.empty else 0.0
        weights["validation_accuracy_weighted_soft_vote"][cid] = max(acc, 0.0)
        weights["validation_lift_weighted_soft_vote"][cid] = max(acc - majority, 0.0)
        weights["stability_weighted_soft_vote"][cid] = max(stability - 0.45, 0.0)
    return weights


def score_from_weights(prob: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    if weights is None:
        return prob.mean(axis=1)
    total = sum(max(float(weights.get(col, 0.0)), 0.0) for col in prob.columns)
    if total <= 0.0:
        return prob.mean(axis=1)
    score = pd.Series(0.0, index=prob.index)
    for col in prob.columns:
        score += prob[col].astype(float) * max(float(weights.get(col, 0.0)), 0.0)
    return score / total


def make_prediction_frame(split_frame: pd.DataFrame, prob: pd.DataFrame, score: pd.Series, method: str, split: str) -> pd.DataFrame:
    truth = truth_frame(split_frame).set_index("row_key")
    common = truth.index.intersection(score.dropna().index)
    out = truth.loc[common].reset_index()
    out["experiment_group"] = "stacking_ensemble"
    out["model"] = method
    out["feature_family"] = "validation_selected_base_models"
    out["threshold_policy"] = "fixed_0.50"
    out["threshold"] = 0.50
    out["candidate_id"] = "legacy_stack__" + method
    out["split"] = split
    out["y_score_or_probability"] = score.loc[common].to_numpy(dtype=float)
    out["y_pred"] = (out["y_score_or_probability"].to_numpy(dtype=float) >= 0.50).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def fit_meta(method: str, val_prob: pd.DataFrame, validation: pd.DataFrame, final_prob: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    truth = truth_frame(validation).set_index("row_key")
    common = val_prob.dropna().index.intersection(truth.index)
    if len(common) < 100:
        return None
    y = truth.loc[common, "y_true"].astype(int)
    if y.nunique() < 2:
        return None
    if method == "meta_logistic_stacking":
        estimator: Any = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    elif method == "meta_lightgbm_stacking" and LGBMClassifier is not None:
        estimator = LGBMClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, min_child_samples=40, random_state=RANDOM_STATE, verbose=-1, n_jobs=2)
    else:
        return None
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    pipeline.fit(val_prob.loc[common], y)
    val_score = pd.Series(pipeline.predict_proba(val_prob)[:, 1], index=val_prob.index)
    final_score = pd.Series(pipeline.predict_proba(final_prob)[:, 1], index=final_prob.index)
    return val_score, final_score


def result_row(method: str, validation_frame: pd.DataFrame, final_frame: pd.DataFrame, selected: bool = False) -> dict[str, Any]:
    val_acc = float(validation_frame["correct"].astype(int).mean()) if not validation_frame.empty else math.nan
    final_acc = float(final_frame["correct"].astype(int).mean()) if not final_frame.empty else math.nan
    final_rows = int(len(final_frame))
    return {
        "experiment_group": "stacking_ensemble",
        "model": method,
        "feature_family": "validation_selected_base_models",
        "horizon": 40,
        "threshold_policy": "fixed_0.50",
        "candidate_id": "legacy_stack__" + method,
        "validation_accuracy": val_acc,
        "final_accuracy": final_acc,
        "delta_vs_61_51": final_acc - REFERENCE_FINAL_ACCURACY if math.isfinite(final_acc) else math.nan,
        "final_rows": final_rows,
        "ticker_coverage": int(final_frame["ticker"].nunique()) if not final_frame.empty else 0,
        "validation_final_gap": final_acc - val_acc if math.isfinite(final_acc) and math.isfinite(val_acc) else math.nan,
        "selected_on_validation": selected,
        "selection_source": "validation_only",
        "final_window_role": "scoring_only",
        "final_accuracy_used_for_selection": False,
        "full_ticker_coverage": int(final_frame["ticker"].nunique()) == 30 if not final_frame.empty else False,
        "leakage_status": "passed_validation_only_stacking",
    }


def slice_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = predictions[predictions["split"].eq("final")].copy()
    final["datetime"] = pd.to_datetime(final["datetime"], errors="coerce")
    final["month"] = final["datetime"].dt.to_period("M").astype(str)
    final["quarter"] = final["datetime"].dt.to_period("Q").astype(str)
    by_ticker = final.groupby(["candidate_id", "model", "ticker"])["correct"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "accuracy", "count": "rows"})
    by_month = final.groupby(["candidate_id", "model", "month"])["correct"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "accuracy", "count": "rows"})
    by_quarter = final.groupby(["candidate_id", "model", "quarter"])["correct"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "accuracy", "count": "rows"})
    return by_ticker, by_month, by_quarter


def rolling_table(predictions: pd.DataFrame, window: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    final = predictions[predictions["split"].eq("final")].copy()
    for cid, group in final.groupby("candidate_id", sort=True):
        ordered = group.sort_values(["datetime", "ticker"]).reset_index(drop=True)
        roll = ordered["correct"].astype(float).rolling(window=window, min_periods=window).mean()
        out = ordered[["datetime", "ticker"]].copy()
        out["candidate_id"] = cid
        out["model"] = str(group["model"].iloc[0])
        out["window"] = window
        out["rolling_accuracy"] = roll
        rows.append(out.dropna(subset=["rolling_accuracy"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation, final, final_results = load_base_predictions()
    candidate_ids = select_base_candidates(validation, final, final_results)
    if len(candidate_ids) < 2:
        raise RuntimeError("Need at least two selected base candidates for stacking")
    val_base = validation[validation["candidate_id"].isin(candidate_ids)].copy()
    final_base = final[final["candidate_id"].isin(candidate_ids)].copy()
    val_prob = wide_prob(val_base, candidate_ids).dropna()
    final_prob = wide_prob(final_base, candidate_ids).dropna()
    weights = validation_weights(val_base, candidate_ids)
    methods = [
        "unweighted_soft_vote",
        "validation_accuracy_weighted_soft_vote",
        "validation_lift_weighted_soft_vote",
        "stability_weighted_soft_vote",
        "meta_logistic_stacking",
        "meta_lightgbm_stacking",
    ]
    prediction_frames: list[pd.DataFrame] = []
    result_rows: list[dict[str, Any]] = []
    for method in methods:
        if method == "unweighted_soft_vote":
            val_score = score_from_weights(val_prob)
            final_score = score_from_weights(final_prob)
        elif method in weights:
            val_score = score_from_weights(val_prob, weights[method])
            final_score = score_from_weights(final_prob, weights[method])
        else:
            fitted = fit_meta(method, val_prob, val_base, final_prob)
            if fitted is None:
                result_rows.append({"experiment_group": "stacking_ensemble", "model": method, "feature_family": "validation_selected_base_models", "horizon": 40, "threshold_policy": "fixed_0.50", "candidate_id": "legacy_stack__" + method, "status": "skipped_with_reason", "skip_reason": "optional dependency missing or invalid meta training shape"})
                continue
            val_score, final_score = fitted
        val_frame = make_prediction_frame(val_base, val_prob, val_score, method, "validation")
        final_frame = make_prediction_frame(final_base, final_prob, final_score, method, "final")
        result_rows.append(result_row(method, val_frame, final_frame))
        prediction_frames.extend([val_frame, final_frame])
    results = pd.DataFrame(result_rows)
    ok = results[results.get("final_accuracy", pd.Series(dtype=float)).notna()].copy()
    if not ok.empty:
        selected_id = str(ok.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0]["candidate_id"])
        results.loc[results["candidate_id"].eq(selected_id), "selected_on_validation"] = True
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    by_ticker, by_month, by_quarter = slice_outputs(predictions)
    val_base.to_csv(OUTPUT_DIR / "stacking_base_validation_predictions.csv", index=False)
    final_base.to_csv(OUTPUT_DIR / "stacking_base_final_predictions.csv", index=False)
    write_json(
        OUTPUT_DIR / "stacking_meta_training_manifest.json",
        {
            "data_fetch": False,
            "provider_behavior_changed": False,
            "base_model_selection": "validation_only_from_legacy_model_comparison_fixed_threshold_candidates",
            "base_candidate_ids": candidate_ids,
            "ensemble_weights": weights,
            "meta_training_split": "validation",
            "final_labels_used_for_meta_training": False,
            "final_accuracy_used_for_selection": False,
            "confidence_abstention": False,
            "ticker_subset": False,
            "topk": False,
            "full_ticker_coverage_required": True,
        },
    )
    results.to_csv(OUTPUT_DIR / "stacking_validation_results.csv", index=False)
    results.to_csv(OUTPUT_DIR / "stacking_final_results.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "stacking_row_predictions.csv", index=False)
    by_ticker.to_csv(OUTPUT_DIR / "stacking_by_ticker.csv", index=False)
    by_month.to_csv(OUTPUT_DIR / "stacking_by_month.csv", index=False)
    by_quarter.to_csv(OUTPUT_DIR / "stacking_by_quarter.csv", index=False)
    for window in (250, 500, 1000):
        rolling_table(predictions, window).to_csv(OUTPUT_DIR / f"stacking_rolling_{window}.csv", index=False)
    best = ok.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0].to_dict() if not ok.empty else {}
    lines = [
        "# VN30 Legacy Rules Stacking Summary",
        "",
        f"- Base candidates: {len(candidate_ids)}.",
        f"- Stacking/ensemble methods evaluated: {len(ok)}.",
        f"- Best validation-selected method: `{best.get('model', '')}`.",
        f"- Validation accuracy: {pct(best.get('validation_accuracy'))}.",
        f"- Final accuracy: {pct(best.get('final_accuracy'))}.",
        f"- Delta vs 61.51%: {pp(best.get('delta_vs_61_51'))}.",
        "- Meta-model training source: validation base predictions only.",
        "- Final labels used for weights/meta training: no.",
        "- Data fetched: no.",
    ]
    (OUTPUT_DIR / "stacking_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"legacy_stacking_complete best={best.get('model', '')} final={pct(best.get('final_accuracy'))} output_dir={rel(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
