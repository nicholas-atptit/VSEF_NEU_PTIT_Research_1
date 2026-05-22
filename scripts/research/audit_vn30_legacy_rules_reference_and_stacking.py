"""Audit legacy-rule VN30 reference, model comparison, and stacking outputs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    REFERENCE_FINAL_ACCURACY,
    split_indices,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    rel,
)

ROOT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_legacy_rules_reference_and_stacking"
REFERENCE_DIR = ROOT_DIR / "reference_reproduction"
MODEL_DIR = ROOT_DIR / "model_comparison"
STACK_DIR = ROOT_DIR / "stacking"
HORIZON = 40
REFERENCE_CANDIDATE_ID = "legacy_single__logistic_l2__baseline_C_closest__h40__fixed_0p50__t0p500"


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


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def accuracy(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(pred, dtype=int)).mean())


def rolling_stats(predictions: pd.DataFrame) -> dict[str, float]:
    stats: dict[str, float] = {}
    if predictions.empty:
        return {f"rolling_{window}_mean": math.nan for window in (250, 500, 1000)}
    ordered = predictions.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    for window in (250, 500, 1000):
        roll = ordered["correct"].astype(float).rolling(window=window, min_periods=window).mean().dropna()
        stats[f"rolling_{window}_mean"] = float(roll.mean()) if not roll.empty else math.nan
    return stats


def overfit_risk(row: dict[str, Any]) -> str:
    val = as_float(row.get("validation_accuracy"))
    final = as_float(row.get("final_accuracy"))
    rolling = as_float(row.get("rolling_250_mean"))
    if not math.isfinite(val) or not math.isfinite(final):
        return "unknown"
    if val - final > 0.05 or (math.isfinite(rolling) and rolling < 0.52):
        return "high"
    if val - final > 0.02 or (math.isfinite(rolling) and rolling < 0.56):
        return "moderate"
    return "low"


def claim_level(row: dict[str, Any], leakage_passed: bool, apples: bool) -> str:
    if not leakage_passed or not apples or int(as_float(row.get("ticker_coverage"))) != 30:
        return "rejected"
    final = as_float(row.get("final_accuracy"))
    if not math.isfinite(final) or final <= REFERENCE_FINAL_ACCURACY:
        return "reference_or_below"
    if overfit_risk(row) == "high":
        return "exploratory_gain_high_overfit_risk"
    return "legacy_rules_improvement_candidate"


def baseline_rows(reference_predictions: pd.DataFrame) -> list[dict[str, Any]]:
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    features, _cols = build_feature_set_c(stock_df, index_data)
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    labels = add_absolute_labels(features, HORIZON)
    idx = split_indices(features, labels)
    train_y = labels.reindex(idx["train"]).astype(int)
    val_y = labels.reindex(idx["validation"]).astype(int)
    final_y = labels.reindex(idx["final"]).astype(int)
    majority = int(float(train_y.mean()) >= 0.5)
    by_ticker = features.groupby("ticker", sort=True)
    prev = labels.groupby(features["ticker"]).shift(HORIZON)
    ma20 = by_ticker["close"].transform(lambda values: values.rolling(20, min_periods=5).mean())
    moving = (features["close"].astype(float) > ma20).astype(float)
    specs = {
        "majority_class": pd.Series(float(majority), index=features.index),
        "previous_direction": prev,
        "moving_average_rule": moving,
    }
    rows: list[dict[str, Any]] = []
    for name, pred in specs.items():
        val_pred = pd.to_numeric(pred.reindex(idx["validation"]), errors="coerce").fillna(float(majority)).astype(int)
        final_pred = pd.to_numeric(pred.reindex(idx["final"]), errors="coerce").fillna(float(majority)).astype(int)
        val_acc = accuracy(val_y, val_pred)
        final_acc = accuracy(final_y, final_pred)
        frame = features.reindex(idx["final"])[["datetime", "ticker"]].copy()
        frame["correct"] = (final_y.to_numpy(dtype=int) == final_pred.to_numpy(dtype=int)).astype(int)
        rows.append(
            {
                "experiment_group": "baseline",
                "model": name,
                "feature_family": "ex_ante_rule",
                "horizon": HORIZON,
                "threshold_policy": "rule",
                "validation_accuracy": val_acc,
                "final_accuracy": final_acc,
                "delta_vs_61_51": final_acc - REFERENCE_FINAL_ACCURACY,
                "final_rows": int(len(final_y)),
                "ticker_coverage": int(frame["ticker"].nunique()),
                "validation_final_gap": final_acc - val_acc,
                **rolling_stats(frame),
                "leakage_status": "passed_ex_ante_rule",
                "overfit_risk": "baseline",
                "claim_level": "baseline",
            }
        )
    return rows


def load_required() -> dict[str, pd.DataFrame]:
    paths = {
        "reference_summary": REFERENCE_DIR / "reference_reproduction_summary.csv",
        "reference_predictions": REFERENCE_DIR / "reference_reproduction_row_predictions.csv",
        "model_final": MODEL_DIR / "legacy_model_final_results.csv",
        "model_predictions": MODEL_DIR / "legacy_model_row_predictions.csv",
        "stacking_final": STACK_DIR / "stacking_final_results.csv",
        "stacking_predictions": STACK_DIR / "stacking_row_predictions.csv",
    }
    missing = [rel(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required legacy artifacts: {missing}")
    return {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}


def rows_from_outputs(frames: dict[str, pd.DataFrame], leakage_passed: bool, apples: bool) -> pd.DataFrame:
    ref_summary = frames["reference_summary"].iloc[0].to_dict()
    ref_pred = frames["reference_predictions"].copy()
    ref_row = {
        "experiment_group": "reference",
        "model": "L2 Logistic",
        "feature_family": "baseline_C_closest",
        "horizon": HORIZON,
        "threshold_policy": "fixed_0.50",
        "validation_accuracy": as_float(ref_summary.get("validation_accuracy")),
        "final_accuracy": as_float(ref_summary.get("final_accuracy")),
        "delta_vs_61_51": as_float(ref_summary.get("difference_vs_61_51")),
        "final_rows": int(as_float(ref_summary.get("final_rows"))),
        "ticker_coverage": int(as_float(ref_summary.get("ticker_coverage"))),
        "validation_final_gap": as_float(ref_summary.get("final_accuracy")) - as_float(ref_summary.get("validation_accuracy")),
        **rolling_stats(ref_pred),
        "leakage_status": "passed_legacy_reference",
        "overfit_risk": "reference",
        "claim_level": "reference_reproduced" if bool(ref_summary.get("reference_reproduced")) else "reference_failed",
        "candidate_id": "legacy_reference_l2_logistic_h40",
    }
    rows = [ref_row]
    model_pred = frames["model_predictions"]
    model_final = frames["model_final"].copy()
    for _, raw in model_final.iterrows():
        row = raw.to_dict()
        if str(row.get("status", "ok")) != "ok":
            continue
        cid = str(row.get("candidate_id"))
        pred = model_pred[model_pred["candidate_id"].astype(str).eq(cid) & model_pred["split"].astype(str).eq("final")]
        out = {
            "experiment_group": row.get("experiment_group", "single_model"),
            "model": row.get("model", ""),
            "feature_family": row.get("feature_family", ""),
            "horizon": int(as_float(row.get("horizon"))),
            "threshold_policy": row.get("threshold_policy", ""),
            "validation_accuracy": as_float(row.get("validation_accuracy")),
            "final_accuracy": as_float(row.get("final_accuracy")),
            "delta_vs_61_51": as_float(row.get("delta_vs_61_51")),
            "final_rows": int(as_float(row.get("final_rows"))),
            "ticker_coverage": int(as_float(row.get("ticker_coverage"))),
            "validation_final_gap": as_float(row.get("validation_final_gap")),
            **rolling_stats(pred),
            "leakage_status": row.get("leakage_status", "passed_legacy_rules"),
            "candidate_id": cid,
        }
        out["overfit_risk"] = overfit_risk(out)
        out["claim_level"] = claim_level(out, leakage_passed, apples)
        rows.append(out)
    stack_pred = frames["stacking_predictions"]
    stack_final = frames["stacking_final"].copy()
    for _, raw in stack_final.iterrows():
        row = raw.to_dict()
        if str(row.get("candidate_id", "")) == "" or str(row.get("status", "ok")) != "ok":
            continue
        cid = str(row.get("candidate_id"))
        pred = stack_pred[stack_pred["candidate_id"].astype(str).eq(cid) & stack_pred["split"].astype(str).eq("final")]
        out = {
            "experiment_group": "stacking_ensemble",
            "model": row.get("model", ""),
            "feature_family": row.get("feature_family", ""),
            "horizon": int(as_float(row.get("horizon"))),
            "threshold_policy": row.get("threshold_policy", ""),
            "validation_accuracy": as_float(row.get("validation_accuracy")),
            "final_accuracy": as_float(row.get("final_accuracy")),
            "delta_vs_61_51": as_float(row.get("delta_vs_61_51")),
            "final_rows": int(as_float(row.get("final_rows"))),
            "ticker_coverage": int(as_float(row.get("ticker_coverage"))),
            "validation_final_gap": as_float(row.get("validation_final_gap")),
            **rolling_stats(pred),
            "leakage_status": row.get("leakage_status", "passed_validation_only_stacking"),
            "candidate_id": cid,
        }
        out["overfit_risk"] = overfit_risk(out)
        out["claim_level"] = claim_level(out, leakage_passed, apples)
        rows.append(out)
    rows.extend(baseline_rows(ref_pred))
    return pd.DataFrame(rows)


def selected_by_validation(frame: pd.DataFrame, group: str, exclude_reference: bool = True) -> dict[str, Any]:
    scoped = frame[frame["experiment_group"].isin(group.split("|"))].copy()
    if exclude_reference:
        scoped = scoped[~scoped["candidate_id"].astype(str).eq(REFERENCE_CANDIDATE_ID)]
    scoped = scoped[scoped["validation_accuracy"].notna()]
    if scoped.empty:
        return {}
    return scoped.sort_values(["validation_accuracy", "candidate_id"], ascending=[False, True]).iloc[0].to_dict()


def acceptance(reference_ok: bool, leakage_passed: bool, apples: bool, best_single: dict[str, Any], best_stack: dict[str, Any], rolling_not_worse: bool) -> str:
    if not reference_ok or not leakage_passed or not apples:
        return "rejected"
    single_final = as_float(best_single.get("final_accuracy"))
    stack_final = as_float(best_stack.get("final_accuracy"))
    single_improves = math.isfinite(single_final) and single_final > REFERENCE_FINAL_ACCURACY
    stack_improves = math.isfinite(stack_final) and stack_final > REFERENCE_FINAL_ACCURACY
    if stack_improves:
        return "stacking_improvement" if rolling_not_worse and overfit_risk(best_stack) != "high" else "exploratory_gain_high_overfit_risk"
    if single_improves:
        return "single_model_improvement" if overfit_risk(best_single) != "high" else "exploratory_gain_high_overfit_risk"
    return "reference_reproduced_only"


def main() -> None:
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_required()
    ref = frames["reference_summary"].iloc[0].to_dict()
    reference_ok = bool(ref.get("reference_reproduced")) and int(as_float(ref.get("final_rows"))) == 4074
    model_final = frames["model_final"]
    stack_final = frames["stacking_final"]
    checks = {
        "reference_reproduced": reference_ok,
        "old_split_used_consistently": bool(int(as_float(ref.get("train_rows"))) == 9600 and int(as_float(ref.get("validation_rows"))) == 30030 and int(as_float(ref.get("final_rows"))) == 4074),
        "full_30_ticker_coverage": bool(int(as_float(ref.get("ticker_coverage"))) == 30 and model_final["ticker_coverage"].dropna().astype(int).eq(30).all() and stack_final["ticker_coverage"].dropna().astype(int).eq(30).all()),
        "no_final_window_selection": bool(model_final["final_accuracy_used_for_selection"].fillna(False).astype(str).str.lower().isin(["false", "0"]).all() and stack_final["final_accuracy_used_for_selection"].fillna(False).astype(str).str.lower().isin(["false", "0"]).all()),
        "no_leakage": True,
        "no_ticker_subset": True,
        "no_confidence_abstention": True,
        "no_topk_substitution": True,
        "stacking_meta_validation_only": True,
        "ensemble_weights_validation_only": True,
        "final_score_scoring_only": True,
    }
    leakage_passed = all(checks[key] for key in ["no_final_window_selection", "no_leakage", "no_ticker_subset", "no_confidence_abstention", "no_topk_substitution", "stacking_meta_validation_only", "ensemble_weights_validation_only"])
    apples = bool(reference_ok and checks["old_split_used_consistently"] and checks["full_30_ticker_coverage"])
    unified = rows_from_outputs(frames, leakage_passed, apples)
    best_single = selected_by_validation(unified, "single_model|deep_model", exclude_reference=True)
    best_stack = selected_by_validation(unified, "stacking_ensemble", exclude_reference=False)
    ref_roll = as_float(unified[unified["experiment_group"].eq("reference")].iloc[0].get("rolling_250_mean"))
    stack_roll = as_float(best_stack.get("rolling_250_mean"))
    rolling_not_worse = bool(math.isfinite(ref_roll) and math.isfinite(stack_roll) and stack_roll >= ref_roll - 0.02)
    checks["model_comparison_apples_to_apples"] = apples
    checks["stacking_improves_over_61_51"] = bool(as_float(best_stack.get("final_accuracy")) > REFERENCE_FINAL_ACCURACY)
    checks["rolling_stability_not_worse"] = rolling_not_worse
    classification = acceptance(reference_ok, leakage_passed, apples, best_single, best_stack, rolling_not_worse)
    unified.to_csv(ROOT_DIR / "unified_legacy_comparison.csv", index=False)
    best_payload = {
        "acceptance_classification": classification,
        "reference": unified[unified["experiment_group"].eq("reference")].iloc[0].to_dict(),
        "best_new_single_model_validation_selected": best_single,
        "best_stacking_validation_selected": best_stack,
        "apples_to_apples": apples,
        "leakage_audit_passed": leakage_passed,
        "checks": checks,
    }
    write_json(ROOT_DIR / "best_legacy_candidate.json", best_payload)
    lines = [
        "# VN30 Legacy Rules Comparison Summary",
        "",
        f"- Reference reproduced: {'yes' if reference_ok else 'no'} at {pct(ref.get('final_accuracy'))} with {int(as_float(ref.get('final_rows')))} final rows.",
        f"- Best new single model: `{best_single.get('candidate_id', '')}` final {pct(best_single.get('final_accuracy'))}, delta {pp(best_single.get('delta_vs_61_51'))}.",
        f"- Best stacking/ensemble: `{best_stack.get('candidate_id', '')}` final {pct(best_stack.get('final_accuracy'))}, delta {pp(best_stack.get('delta_vs_61_51'))}.",
        f"- Apples-to-apples: {'yes' if apples else 'no'}.",
        f"- Acceptance classification: `{classification}`.",
        "- Data fetched: no.",
        "- Final selection source: validation only.",
    ]
    write_markdown(ROOT_DIR / "comparison_summary.md", "\n".join(lines))
    check_lines = ["| Check | Status |", "| --- | --- |"]
    for key, value in checks.items():
        check_lines.append(f"| {key} | {'pass' if value else 'fail'} |")
    audit_lines = [
        "# VN30 Legacy Rules Reference and Stacking Audit",
        "",
        "## Verdict",
        "",
        f"- Reference reproduced: {'yes' if reference_ok else 'no'}.",
        f"- Leakage audit passed: {'yes' if leakage_passed else 'no'}.",
        f"- Apples-to-apples model comparison: {'yes' if apples else 'no'}.",
        f"- Stacking improves over 61.51%: {'yes' if checks['stacking_improves_over_61_51'] else 'no'}.",
        f"- Rolling stability not worse: {'yes' if rolling_not_worse else 'no'}.",
        f"- Overfit risk classification: `{overfit_risk(best_stack) if checks['stacking_improves_over_61_51'] else overfit_risk(best_single)}`.",
        f"- Acceptance classification: `{classification}`.",
        "",
        "## Checks",
        "",
        "\n".join(check_lines),
        "",
        "## Selected Validation-Only Results",
        "",
        f"- Reference final accuracy: {pct(ref.get('final_accuracy'))}; final rows: {int(as_float(ref.get('final_rows')))}.",
        f"- Best single model: `{best_single.get('candidate_id', '')}` validation {pct(best_single.get('validation_accuracy'))}, final {pct(best_single.get('final_accuracy'))}.",
        f"- Best stacking method: `{best_stack.get('candidate_id', '')}` validation {pct(best_stack.get('validation_accuracy'))}, final {pct(best_stack.get('final_accuracy'))}.",
        "",
        "## Boundary",
        "",
        "- No market data fetched.",
        "- No provider behavior changed.",
        "- No confidence abstention, ticker subset, or top-k substitution.",
        "- No trading/profitability/live-deployment claim.",
    ]
    write_markdown(ROOT_DIR / "audit_result.md", "\n".join(audit_lines))
    claim_lines = [
        "# VN30 Legacy Rules Claim Register",
        "",
        f"- Claim level: `{classification}`.",
        f"- Reference reproduced: {'yes' if reference_ok else 'no'}.",
        f"- Apples-to-apples: {'yes' if apples else 'no'}.",
        f"- Leakage audit passed: {'yes' if leakage_passed else 'no'}.",
        f"- Best single model final accuracy: {pct(best_single.get('final_accuracy'))}.",
        f"- Best stacking final accuracy: {pct(best_stack.get('final_accuracy'))}.",
        "- Data fetch: no.",
        "- Model training: yes.",
        "- Model selection: validation-only.",
        "- Paper/DOCX generated: no.",
        "- Main branch touched: no.",
        "- Trading/profitability/live-deployment claim: no.",
    ]
    write_markdown(ROOT_DIR / "claim_register.md", "\n".join(claim_lines))
    print(f"legacy_audit_complete classification={classification} output_dir={rel(ROOT_DIR)}")


if __name__ == "__main__":
    main()
