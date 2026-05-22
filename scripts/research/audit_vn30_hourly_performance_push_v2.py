"""Audit VN30 hourly performance-push v2 artifacts."""

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

from scripts.research.run_vn30_hourly_performance_push_v2 import (  # noqa: E402
    REFERENCE_FINAL_ACCURACY,
    REFERENCE_MAJORITY_BASELINE,
)
from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, rel  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_performance_push_v2"
REQUIRED_FILES = [
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
    "run_config.json",
    "feature_set_manifest.json",
    "rolling_summary_by_policy.csv",
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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {rel(path)}")
    return payload


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def required_files_present() -> tuple[bool, list[str]]:
    missing = [name for name in REQUIRED_FILES if not (REPORT_DIR / name).exists()]
    return not missing, missing


def all_values_equal(frame: pd.DataFrame, column: str, expected: Any) -> bool:
    if frame.empty:
        return True
    if column not in frame.columns:
        return False
    return bool((frame[column].fillna(expected) == expected).all())


def policy_ticker_coverage(predictions: pd.DataFrame) -> bool:
    if predictions.empty:
        return False
    coverage = predictions.groupby("policy")["ticker"].nunique()
    return bool((coverage == 30).all())


def no_confidence_abstention(predictions: pd.DataFrame, final_results: pd.DataFrame) -> bool:
    if predictions.empty:
        return False
    if predictions[["y_true", "y_pred", "correct"]].isna().any().any():
        return False
    rows = predictions.groupby("policy").size().rename("prediction_rows").reset_index()
    merged = rows.merge(final_results[["policy", "final_rows"]], on="policy", how="left")
    return bool((merged["prediction_rows"].astype(int) == merged["final_rows"].astype(int)).all())


def manifest_leakage_safe(manifest: dict[str, Any]) -> bool:
    families = manifest.get("feature_families", {})
    if not isinstance(families, dict) or not families:
        return False
    for payload in families.values():
        if bool(payload.get("future_return_features")):
            return False
        if bool(payload.get("future_regime_labels")):
            return False
        if bool(payload.get("target_leakage_features")):
            return False
        if bool(payload.get("same_row_target_leakage")):
            return False
        if bool(payload.get("final_window_derived_features")):
            return False
        if bool(payload.get("uses_final_window")):
            return False
    return True


def rolling_table(rolling_summary: pd.DataFrame) -> str:
    lines = [
        "| Policy | Window | Min Acc | Mean Acc | End Acc | Windows <60% | Min Lift | Mean Lift |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in rolling_summary.iterrows():
        lines.append(
            f"| {row['policy']} | {int(row['window_rows'])} | {pct(row['rolling_min_accuracy'])} | "
            f"{pct(row['rolling_mean_accuracy'])} | {pct(row['final_endpoint_rolling_accuracy'])} | "
            f"{int(row['windows_below_60'])} | {pp(row['rolling_min_lift_vs_majority'])} | {pp(row['rolling_mean_lift_vs_majority'])} |"
        )
    return "\n".join(lines)


def policy_table(final_results: pd.DataFrame) -> str:
    lines = [
        "| Policy | Candidate | Val Acc | Final Acc | Delta Ref | Coverage | Risk | Claim Level |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in final_results.iterrows():
        lines.append(
            f"| {row['policy']} | `{row['candidate_id']}` | {pct(row['validation_accuracy'])} | "
            f"{pct(row['final_accuracy'])} | {pp(row['delta_vs_61_51_reference'])} | "
            f"{int(row['final_unique_tickers'])}/30 | {row['overfit_risk_classification']} | {row['claim_level']} |"
        )
    return "\n".join(lines)


def best_policy(final_results: pd.DataFrame) -> dict[str, Any]:
    if final_results.empty:
        return {}
    return final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=False).iloc[0].to_dict()


def main() -> None:
    files_ok, missing = required_files_present()
    if not files_ok:
        raise FileNotFoundError(f"missing performance-push artifacts: {missing}")

    config = read_json(REPORT_DIR / "run_config.json")
    manifest = read_json(REPORT_DIR / "feature_set_manifest.json")
    selected_json = read_json(REPORT_DIR / "selected_candidates.json")
    validation_scores = pd.read_csv(REPORT_DIR / "validation_scores_all.csv")
    selections = pd.read_csv(REPORT_DIR / "selection_policy_results.csv")
    final_results = pd.read_csv(REPORT_DIR / "final_scoring_results.csv")
    predictions = pd.read_csv(REPORT_DIR / "final_row_predictions_by_policy.csv")
    by_ticker = pd.read_csv(REPORT_DIR / "by_ticker_by_policy.csv")
    by_month = pd.read_csv(REPORT_DIR / "by_month_by_policy.csv")
    by_quarter = pd.read_csv(REPORT_DIR / "by_quarter_by_policy.csv")
    rolling_summary = pd.read_csv(REPORT_DIR / "rolling_summary_by_policy.csv")
    thresholds = pd.read_csv(REPORT_DIR / "per_ticker_thresholds.csv")
    ensembles = pd.read_csv(REPORT_DIR / "ensemble_weights.csv")
    calibrations = pd.read_csv(REPORT_DIR / "calibration_summary.csv")
    routers = pd.read_csv(REPORT_DIR / "router_summary.csv")

    predictions["ticker"] = predictions["ticker"].astype(str).str.upper()
    checks = {
        "required_files_present": files_ok,
        "no_final_window_selection": bool(config.get("final_accuracy_used_for_selection") is False)
        and "final_accuracy" not in validation_scores.columns
        and "delta_vs_61_51_reference" not in validation_scores.columns,
        "no_final_tuned_threshold": thresholds.empty or all_values_equal(thresholds, "selection_source", "validation_only"),
        "no_future_features": manifest_leakage_safe(manifest),
        "no_target_leakage": manifest_leakage_safe(manifest),
        "no_same_row_leakage": manifest_leakage_safe(manifest),
        "no_ticker_subset": bool(config.get("ticker_subset") is False) and policy_ticker_coverage(predictions),
        "no_confidence_abstention": bool(config.get("confidence_abstention") is False) and no_confidence_abstention(predictions, final_results),
        "no_topk_substitution": bool(config.get("topk") is False),
        "full_30_stock_coverage": bool((final_results["final_unique_tickers"].astype(int) == 30).all()),
        "all_policy_selections_validation_only": bool((selections["selected_by_policy_validation_only"] == True).all())  # noqa: E712
        and bool(selected_json.get("selection_boundary", {}).get("final_accuracy_used_for_candidate_selection") is False),
        "per_ticker_thresholds_validation_only": thresholds.empty or all_values_equal(thresholds, "selection_source", "validation_only"),
        "calibration_validation_only": calibrations.empty or all_values_equal(calibrations, "fit_window", "validation_only"),
        "ensemble_weights_validation_only": ensembles.empty or all_values_equal(ensembles, "weight_selection_source", "validation_only"),
        "router_validation_only": routers.empty or all_values_equal(routers, "selection_source", "validation_only"),
        "validation_final_gap_available": bool(final_results["validation_final_gap"].notna().all()),
        "rolling_stability_available": not rolling_summary.empty and set(rolling_summary["window_rows"].astype(int)) == {250, 500, 1000},
        "monthly_stability_available": not by_month.empty,
        "quarterly_stability_available": not by_quarter.empty,
        "ticker_stability_available": not by_ticker.empty and bool((by_ticker.groupby("policy")["ticker"].nunique() == 30).all()),
        "lift_over_majority_available": bool(final_results["final_lift_vs_majority"].notna().all()),
        "comparison_with_reference_available": bool(final_results["delta_vs_61_51_reference"].notna().all()),
    }
    leakage_passed = all(
        checks[name]
        for name in [
            "no_final_window_selection",
            "no_final_tuned_threshold",
            "no_future_features",
            "no_target_leakage",
            "no_same_row_leakage",
            "no_ticker_subset",
            "no_confidence_abstention",
            "no_topk_substitution",
            "full_30_stock_coverage",
            "all_policy_selections_validation_only",
            "per_ticker_thresholds_validation_only",
            "calibration_validation_only",
            "ensemble_weights_validation_only",
            "router_validation_only",
        ]
    )
    best = best_policy(final_results)
    any_gain = bool((final_results["final_accuracy"] > REFERENCE_FINAL_ACCURACY).any())
    final65 = bool((final_results["final_accuracy"] >= 0.65).any())
    overall_acceptance = "final65_candidate" if final65 and leakage_passed else ("stronger_candidate" if any_gain and leakage_passed else "failed_push")
    if any_gain and leakage_passed and bool((final_results.loc[final_results["final_accuracy"] > REFERENCE_FINAL_ACCURACY, "overfit_risk_classification"] == "high").all()):
        overall_acceptance = "likely_overfit"

    audit_lines = [
        "# VN30 Hourly Performance Push V2 Audit Result",
        "",
        "## Best Observed Policy",
        "",
        f"- Policy: `{best.get('policy', '')}`.",
        f"- Candidate: `{best.get('candidate_id', '')}`.",
        f"- Candidate family: `{best.get('candidate_family', '')}`.",
        f"- Feature set: `{best.get('feature_set', '')}`.",
        f"- Model: `{best.get('model', '')}`.",
        f"- Horizon: h={int(as_float(best.get('horizon')))}.",
        f"- Validation accuracy: {pct(best.get('validation_accuracy'))}.",
        f"- Final accuracy: {pct(best.get('final_accuracy'))}.",
        f"- Delta vs 61.51% reference: {pp(best.get('delta_vs_61_51_reference'))}.",
        f"- Delta vs 50.44% majority reference: {pp(best.get('delta_vs_reference_majority_50_44'))}.",
        f"- Final rows: {int(as_float(best.get('final_rows')))}.",
        f"- Full 30-stock coverage: {'yes' if int(as_float(best.get('final_unique_tickers'))) == 30 else 'no'}.",
        f"- Validation-final gap: {pp(best.get('validation_final_gap'))}.",
        f"- Per-ticker calibration used: {'yes' if bool(best.get('per_ticker_calibration_used')) else 'no'}.",
        f"- Ensemble used: {'yes' if bool(best.get('ensemble_used')) else 'no'}.",
        f"- Router used: {'yes' if bool(best.get('router_used')) else 'no'}.",
        f"- Overfit risk classification: `{best.get('overfit_risk_classification', '')}`.",
        f"- Claim level: `{best.get('claim_level', '')}`.",
        f"- Overall acceptance: `{overall_acceptance}`.",
        "",
        "## Audit Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, passed in checks.items():
        audit_lines.append(f"| {name} | {'pass' if passed else 'fail'} |")
    audit_lines.append(f"| leakage_audit_passed | {'pass' if leakage_passed else 'fail'} |")
    audit_lines.extend(["", "## Policy Results", "", policy_table(final_results), "", "## Rolling Stability", "", rolling_table(rolling_summary)])
    audit_lines.extend(
        [
            "",
            "## Stability Ranges",
            "",
            f"- Ticker accuracy range: {pct(by_ticker['accuracy'].min())} to {pct(by_ticker['accuracy'].max())}.",
            f"- Month accuracy range: {pct(by_month['accuracy'].min())} to {pct(by_month['accuracy'].max())}.",
            f"- Quarter accuracy range: {pct(by_quarter['accuracy'].min())} to {pct(by_quarter['accuracy'].max())}.",
            "",
            "## Boundary",
            "",
            "Final rows were scoring-only. This audit does not support trading, profitability, investment recommendation, live-deployment, ticker-subset, confidence-abstention, or top-k substitution claims.",
        ]
    )
    write_markdown(REPORT_DIR / "audit_result.md", "\n".join(audit_lines))

    claim_lines = [
        "# VN30 Hourly Performance Push V2 Claim Register",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
        f"| Performance-push v2 benchmark ran | safe | `{rel(REPORT_DIR / 'performance_push_summary.md')}` |",
        f"| Candidate/policy selections were validation-only | {'safe' if checks['all_policy_selections_validation_only'] else 'unsafe'} | `{rel(REPORT_DIR / 'selection_policy_results.csv')}` |",
        f"| Leakage audit passed | {'safe' if leakage_passed else 'not supported'} | `{rel(REPORT_DIR / 'audit_result.md')}` |",
        f"| Best observed final accuracy was {pct(best.get('final_accuracy'))} | safe as scoring-only result | `{rel(REPORT_DIR / 'final_scoring_results.csv')}` |",
        f"| Best observed result beat 61.51% reference | {'safe with classification boundary' if as_float(best.get('final_accuracy')) > REFERENCE_FINAL_ACCURACY else 'not supported'} | delta {pp(best.get('delta_vs_61_51_reference'))} |",
        f"| Full 30-stock coverage | {'safe' if checks['full_30_stock_coverage'] else 'not supported'} | `{rel(REPORT_DIR / 'by_ticker_by_policy.csv')}` |",
        f"| Final65 established | {'exploratory candidate only' if final65 else 'not supported'} | overall `{overall_acceptance}` |",
        "| Automatic paper claim upgrade | unsafe | protocol boundary |",
        "| Trading/profitability/live deployment | unsafe | outside experiment scope |",
        "| Confidence-filtered, ticker-subset, top-k, index-only, or joint-panel result as headline VN30 stock-only accuracy | unsafe | prohibited |",
        "",
        "## Classification Counts",
        "",
    ]
    for claim_level, count in final_results["claim_level"].value_counts().sort_index().items():
        claim_lines.append(f"- `{claim_level}`: {int(count)} policy result(s).")
    claim_lines.extend(
        [
            "",
            f"Reference majority baseline: {pct(REFERENCE_MAJORITY_BASELINE)}.",
            "All findings remain bounded to this audited performance-push experiment.",
        ]
    )
    write_markdown(REPORT_DIR / "claim_register.md", "\n".join(claim_lines))
    print(
        f"Audit overall={overall_acceptance}; leakage_passed={leakage_passed}; "
        f"best_policy={best.get('policy', '')}; best_final={pct(best.get('final_accuracy'))}"
    )


if __name__ == "__main__":
    main()
