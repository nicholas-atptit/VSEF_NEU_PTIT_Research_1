"""Audit VN30 hourly validation-safe improvement track artifacts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    LOCKED_RF_H60,
    REFERENCE_FINAL_ACCURACY,
    REFERENCE_MAJORITY_BASELINE,
    REFERENCE_VALIDATION_FINAL_GAP,
)
from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, rel  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_validation_safe_improvement_tracks"
REQUIRED_FILES = [
    "candidate_grid.csv",
    "validation_results.csv",
    "selected_candidate.json",
    "final_scoring_results.csv",
    "row_predictions_selected.csv",
    "by_ticker.csv",
    "by_month.csv",
    "by_quarter.csv",
    "rolling_250.csv",
    "rolling_500.csv",
    "rolling_1000.csv",
    "rolling_summary.csv",
    "feature_ablation_summary.csv",
    "model_family_summary.csv",
    "improvement_summary.md",
    "leakage_audit.md",
    "claim_boundary.md",
    "feature_family_manifest.json",
    "run_config.json",
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
    return f"{number * 100.0:+.2f} percentage points"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {rel(path)}")
    return payload


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def check_required_files() -> tuple[bool, list[str]]:
    missing = [name for name in REQUIRED_FILES if not (REPORT_DIR / name).exists()]
    return not missing, missing


def validation_final_gap(selected: dict[str, Any]) -> float:
    return as_float(selected.get("final_accuracy")) - as_float(selected.get("validation_accuracy"))


def rolling_summary_table(rolling_summary: pd.DataFrame) -> str:
    lines = [
        "| Window | Min Acc | Mean Acc | End Acc | Windows <60% | Min Lift | Mean Lift |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in rolling_summary.iterrows():
        lines.append(
            f"| {int(row['window_rows'])} | {pct(row['rolling_min_accuracy'])} | "
            f"{pct(row['rolling_mean_accuracy'])} | {pct(row['final_endpoint_rolling_accuracy'])} | "
            f"{int(row['windows_below_60'])} | {pp(row['rolling_min_lift_vs_majority'])} | "
            f"{pp(row['rolling_mean_lift_vs_majority'])} |"
        )
    return "\n".join(lines)


def classify_overfit(selected: dict[str, Any]) -> str:
    stored = str(selected.get("overfit_risk_classification", "")).strip()
    if stored:
        return stored
    penalty = as_float(selected.get("overfit_risk_penalty"))
    train_val_gap = as_float(selected.get("train_accuracy")) - as_float(selected.get("validation_accuracy"))
    val_final_gap = validation_final_gap(selected)
    if penalty >= 0.12 or train_val_gap > 0.15 or abs(val_final_gap) > 0.14:
        return "high"
    if penalty >= 0.06 or train_val_gap > 0.08 or abs(val_final_gap) > 0.10:
        return "medium"
    return "low"


def recompute_acceptance(selected: dict[str, Any], checks: dict[str, bool]) -> str:
    final_acc = as_float(selected.get("final_accuracy"))
    gap_ok = abs(validation_final_gap(selected)) <= abs(REFERENCE_VALIDATION_FINAL_GAP) + 0.02
    if (
        final_acc >= 0.65
        and checks["validation_selected"]
        and checks["full_30_stock_coverage"]
        and checks["leakage_audit_passed"]
        and checks["rolling_stability_not_worse"]
        and gap_ok
    ):
        return "final65_candidate"
    if (
        final_acc > REFERENCE_FINAL_ACCURACY
        and checks["validation_selected"]
        and checks["full_30_stock_coverage"]
        and checks["leakage_audit_passed"]
        and checks["rolling_stability_not_worse"]
        and gap_ok
    ):
        return "stronger_baseline60_candidate"
    if final_acc > REFERENCE_FINAL_ACCURACY and checks["validation_selected"] and checks["full_30_stock_coverage"] and checks["leakage_audit_passed"]:
        return "weak_improvement"
    return "failed_improvement"


def main() -> None:
    required_ok, missing = check_required_files()
    if not required_ok:
        raise FileNotFoundError(f"missing required improvement artifacts: {missing}")

    selected = read_json(REPORT_DIR / "selected_candidate.json")
    manifest = read_json(REPORT_DIR / "feature_family_manifest.json")
    config = read_json(REPORT_DIR / "run_config.json")
    validation_results = pd.read_csv(REPORT_DIR / "validation_results.csv")
    final_results = pd.read_csv(REPORT_DIR / "final_scoring_results.csv")
    predictions = pd.read_csv(REPORT_DIR / "row_predictions_selected.csv")
    by_ticker = pd.read_csv(REPORT_DIR / "by_ticker.csv")
    by_month = pd.read_csv(REPORT_DIR / "by_month.csv")
    by_quarter = pd.read_csv(REPORT_DIR / "by_quarter.csv")
    rolling_summary = pd.read_csv(REPORT_DIR / "rolling_summary.csv")

    candidate_id = str(selected.get("candidate_id"))
    selected_validation_row = validation_results[validation_results["candidate_id"].astype(str) == candidate_id]
    selected_final_row = final_results[final_results["candidate_id"].astype(str) == candidate_id]
    family = str(selected.get("feature_family"))
    family_manifest = manifest.get("feature_families", {}).get(family, {})

    predictions["datetime"] = pd.to_datetime(predictions["datetime"], errors="coerce")
    predictions["ticker"] = predictions["ticker"].astype(str).str.upper()
    predictions["y_true"] = pd.to_numeric(predictions["y_true"], errors="coerce")
    predictions["y_pred"] = pd.to_numeric(predictions["y_pred"], errors="coerce")
    predictions["correct"] = pd.to_numeric(predictions["correct"], errors="coerce")

    checks = {
        "required_files_present": required_ok,
        "final_window_not_used_in_selection": bool(config.get("final_accuracy_used_for_selection") is False)
        and "final_accuracy" not in validation_results.columns,
        "validation_selected": bool(selected.get("selected_by_validation_only")) and not selected_validation_row.empty,
        "selected_final_scored": not selected_final_row.empty,
        "no_future_features": not bool(family_manifest.get("future_return_features")) and not bool(family_manifest.get("future_regime_labels")),
        "no_target_leakage": not bool(family_manifest.get("target_leakage_features")),
        "no_same_row_leakage": not bool(family_manifest.get("same_row_target_leakage")),
        "no_confidence_abstention": bool(config.get("confidence_abstention") is False) and predictions["y_pred"].notna().all(),
        "no_ticker_subset": bool(config.get("ticker_subset") is False) and int(predictions["ticker"].nunique()) == 30,
        "no_topk_ranking_substitution": bool(config.get("topk") is False),
        "full_30_stock_coverage": int(predictions["ticker"].nunique()) == 30 and int(selected.get("active_ticker_count", 0)) == 30,
        "validation_final_gap_recorded": math.isfinite(validation_final_gap(selected)),
        "ticker_stability_available": len(by_ticker) == 30 and by_ticker["rows"].min() > 0,
        "monthly_stability_available": not by_month.empty,
        "quarterly_stability_available": not by_quarter.empty,
        "rolling_250_available": (REPORT_DIR / "rolling_250.csv").exists(),
        "rolling_500_available": (REPORT_DIR / "rolling_500.csv").exists(),
        "rolling_1000_available": (REPORT_DIR / "rolling_1000.csv").exists(),
        "lift_over_majority_global": as_float(selected.get("final_lift_vs_majority")) > 0,
        "comparison_with_61_51_reference": math.isfinite(as_float(selected.get("delta_vs_61_51_reference"))),
        "rolling_stability_not_worse": bool(selected.get("rolling_stability_not_worse_than_reference")),
    }
    checks["leakage_audit_passed"] = all(
        checks[name]
        for name in [
            "final_window_not_used_in_selection",
            "no_future_features",
            "no_target_leakage",
            "no_same_row_leakage",
            "no_confidence_abstention",
            "no_ticker_subset",
            "no_topk_ranking_substitution",
            "full_30_stock_coverage",
        ]
    )
    acceptance = recompute_acceptance(selected, checks)
    overfit_risk = classify_overfit(selected)

    audit_lines = [
        "# VN30 Hourly Validation-Safe Improvement Tracks Audit Result",
        "",
        "## Selected Candidate",
        "",
        f"- Candidate ID: `{candidate_id}`.",
        f"- Feature family: `{family}`.",
        f"- Model: `{selected.get('model')}`.",
        f"- Horizon: h={int(as_float(selected.get('horizon')))}.",
        f"- Validation accuracy: {pct(selected.get('validation_accuracy'))}.",
        f"- Final accuracy: {pct(selected.get('final_accuracy'))}.",
        f"- Delta vs 61.51% reference: {pp(selected.get('delta_vs_61_51_reference'))}.",
        f"- Delta vs 50.44% majority reference: {pp(selected.get('delta_vs_reference_majority_50_44'))}.",
        f"- Historical RF h=60 reference: {pct(LOCKED_RF_H60)}.",
        f"- Final rows: {int(as_float(selected.get('final_rows')))}.",
        f"- Full 30-stock coverage: {'yes' if checks['full_30_stock_coverage'] else 'no'}.",
        f"- Validation-final gap: {pp(validation_final_gap(selected))}.",
        f"- Overfit risk classification: `{overfit_risk}`.",
        f"- Acceptance classification: `{acceptance}`.",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, passed in checks.items():
        audit_lines.append(f"| {name} | {'pass' if passed else 'fail'} |")
    audit_lines.extend(["", "## Rolling 250/500/1000 Summary", "", rolling_summary_table(rolling_summary)])
    audit_lines.extend(
        [
            "",
            "## Stability Notes",
            "",
            f"- Ticker accuracy range: {pct(by_ticker['accuracy'].min())} to {pct(by_ticker['accuracy'].max())}.",
            f"- Month accuracy range: {pct(by_month['accuracy'].min())} to {pct(by_month['accuracy'].max())}.",
            f"- Quarter accuracy range: {pct(by_quarter['accuracy'].min())} to {pct(by_quarter['accuracy'].max())}.",
            "",
            "## Boundary",
            "",
            "The final window was scoring-only. This audit does not create a trading, profitability, investment recommendation, live-deployment, or paper/DOCX claim.",
        ]
    )
    write_markdown(REPORT_DIR / "audit_result.md", "\n".join(audit_lines))

    claim_lines = [
        "# VN30 Hourly Validation-Safe Improvement Tracks Claim Register",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
        f"| Validation-safe improvement experiment ran | safe | `{rel(REPORT_DIR / 'validation_results.csv')}` |",
        f"| Selected candidate was validation-selected | {'safe' if checks['validation_selected'] else 'unsafe'} | `{rel(REPORT_DIR / 'selected_candidate.json')}` |",
        f"| Final accuracy was {pct(selected.get('final_accuracy'))} | safe as scoring-only result | `{rel(REPORT_DIR / 'final_scoring_results.csv')}` |",
        f"| Final accuracy beat 61.51% reference | {'safe' if as_float(selected.get('final_accuracy')) > REFERENCE_FINAL_ACCURACY else 'not supported'} | delta {pp(selected.get('delta_vs_61_51_reference'))} |",
        f"| Full 30-stock coverage | {'safe' if checks['full_30_stock_coverage'] else 'not supported'} | `{rel(REPORT_DIR / 'by_ticker.csv')}` |",
        f"| Leakage audit passed | {'safe' if checks['leakage_audit_passed'] else 'not supported'} | `{rel(REPORT_DIR / 'audit_result.md')}` |",
        f"| Final65 established | {'exploratory candidate only' if acceptance == 'final65_candidate' else 'not supported'} | acceptance `{acceptance}` |",
        "| Trading/profitability/live deployment | unsafe | outside experiment scope |",
        "| Confidence-filtered, ticker-subset, top-k, index-only, or joint-panel result as main claim | unsafe | prohibited by protocol |",
        "",
        "## Claim Level",
        "",
        f"- Acceptance classification: `{acceptance}`.",
        f"- Claim level: `{selected.get('claim_level')}`.",
        f"- Majority reference: {pct(REFERENCE_MAJORITY_BASELINE)}.",
        "",
        "All claims remain exploratory and bounded to this validation-safe stock-only benchmark experiment.",
    ]
    write_markdown(REPORT_DIR / "claim_register.md", "\n".join(claim_lines))
    print(f"Audit {acceptance}; leakage_passed={checks['leakage_audit_passed']}; final={pct(selected.get('final_accuracy'))}")


if __name__ == "__main__":
    main()
