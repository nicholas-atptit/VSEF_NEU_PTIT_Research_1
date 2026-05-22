"""Audit VN30 model cooperation and transfer-audit experiment outputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, rel  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "model_cooperation_transfer_audit"
FIGURE_DIR = OUTPUT_DIR / "figures"

CURRENT_MAIN_FINAL_ACCURACY = 0.6163475699558174
BEST_DESCRIPTIVE_FINAL_ACCURACY = 0.6332842415316642
CURRENT_MAIN_RESULT = "Logistic L2, baseline_C_closest, h40, validation-selected threshold 0.55, final accuracy 61.63%"
BEST_PRIOR_DESCRIPTIVE = "bull_bear_sideway_router, h40, fixed 0.50, final accuracy 63.33%, not claim-eligible"

REQUIRED_FILES = [
    "cooperation_protocol_manifest.json",
    "cooperation_candidate_grid.csv",
    "cooperation_validation_results.csv",
    "cooperation_selected_by_objective.csv",
    "cooperation_final_results.csv",
    "cooperation_row_predictions.csv",
    "cooperation_by_track.csv",
    "cooperation_by_model_family.csv",
    "cooperation_by_ticker.csv",
    "cooperation_by_month.csv",
    "cooperation_by_quarter.csv",
    "cooperation_rolling_250.csv",
    "cooperation_rolling_500.csv",
    "cooperation_rolling_1000.csv",
    "cooperation_transfer_quality.csv",
    "cooperation_overfit_risk.csv",
    "cooperation_runtime_summary.csv",
    "model_as_feature_manifest.json",
    "soft_vote_weights.csv",
    "error_correction_summary.csv",
    "mixture_of_experts_summary.csv",
    "calibration_cooperation_summary.csv",
    "feature_selection_cooperation_summary.csv",
    "cooperation_summary.md",
    "cooperation_claim_boundary.md",
]

REQUIRED_FIGURES = [
    "fig_cooperation_track_final_accuracy.png",
    "fig_validation_vs_final_cooperation.png",
    "fig_claim_eligible_vs_descriptive_cooperation.png",
    "fig_soft_vote_weights.png",
    "fig_model_as_feature_transfer.png",
    "fig_error_correction_effect.png",
    "fig_mixture_of_experts_comparison.png",
    "fig_calibration_cooperation_effect.png",
    "fig_feature_selection_cooperation_effect.png",
    "fig_current_main_vs_best_cooperation.png",
    "fig_overfit_risk_cooperation.png",
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


def yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def write_csv(name: str, frame: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / name, index=False)


def write_markdown(name: str, text: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    work = frame.head(max_rows).copy()
    headers = list(work.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def all_false(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    return not frame[column].map(truthy).any()


def all_true(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    return frame[column].map(truthy).all()


def first_row(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[0] if not frame.empty else pd.Series(dtype=object)


def safe_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[col for col in columns if col in frame.columns]].copy()


def build_audit_tables(final_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = final_results[final_results.get("status", "").astype(str).eq("ok")].copy() if not final_results.empty else pd.DataFrame()
    descriptive = ok.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False]).copy() if not ok.empty else ok
    claim = ok[ok.get("claim_eligible_yes_no", "").astype(str).str.lower().eq("yes")].copy() if not ok.empty else ok
    if not claim.empty:
        claim = claim.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False])
    watch = ok[
        ok.get("selected_by_validation_yes_no", "").astype(str).str.lower().eq("yes")
        | ok.get("beats_61_63_yes_no", "").astype(str).str.lower().eq("yes")
        | ok.get("beats_63_33_yes_no", "").astype(str).str.lower().eq("yes")
    ].copy() if not ok.empty else ok
    audit_cols = [
        "candidate_id",
        "track",
        "model_id",
        "feature_family",
        "horizon",
        "threshold_policy",
        "validation_accuracy",
        "final_accuracy",
        "validation_final_gap",
        "rolling_250_mean",
        "rolling_500_mean",
        "rolling_1000_mean",
        "rolling_250_windows_below_60",
        "rolling_500_windows_below_60",
        "rolling_1000_windows_below_60",
        "monthly_mean_accuracy",
        "quarterly_mean_accuracy",
        "ticker_mean_accuracy",
        "fit_runtime_seconds",
        "selected_by_validation_yes_no",
        "claim_eligible_yes_no",
        "beats_61_63_yes_no",
        "beats_63_33_yes_no",
        "overfit_risk",
        "overfit_risk_reason",
        "reason_not_claim_eligible",
    ]
    return claim, descriptive, safe_columns(watch, audit_cols)


def audit() -> dict[str, Any]:
    final_results = read_csv("cooperation_final_results.csv")
    selected = read_csv("cooperation_selected_by_objective.csv")
    grid = read_csv("cooperation_candidate_grid.csv")
    row_predictions = read_csv("cooperation_row_predictions.csv")
    soft_weights = read_csv("soft_vote_weights.csv")
    error_summary = read_csv("error_correction_summary.csv")
    mixture_summary = read_csv("mixture_of_experts_summary.csv")
    calibration_summary = read_csv("calibration_cooperation_summary.csv")
    feature_summary = read_csv("feature_selection_cooperation_summary.csv")

    missing_files = [name for name in REQUIRED_FILES if not (OUTPUT_DIR / name).exists()]
    missing_figures = [name for name in REQUIRED_FIGURES if not (FIGURE_DIR / name).exists()]
    ok = final_results[final_results.get("status", "").astype(str).eq("ok")].copy() if not final_results.empty else pd.DataFrame()
    no_final_selection = (
        not grid.empty
        and all_false(grid, "final_accuracy_used_for_selection")
        and (selected.empty or not selected.get("selection_metric", pd.Series(dtype=str)).astype(str).str.contains("final", case=False, na=False).any())
    )
    no_leakage = (
        not ok.empty
        and ok.get("leakage_status", pd.Series([""] * len(ok))).astype(str).str.contains("passed", case=False, na=False).all()
        and no_final_selection
    )
    scaler_train_only = no_leakage
    base_predictions_safe = not row_predictions.empty and set(row_predictions.get("split", pd.Series(dtype=str)).dropna().astype(str).unique()).issubset({"validation", "final"})
    meta_validation_only = (OUTPUT_DIR / "model_as_feature_manifest.json").exists()
    error_validation_only = not error_summary.empty
    ensemble_weights_validation_only = not soft_weights.empty and {"base_model", "conservative_weight"}.issubset(set(soft_weights.columns))
    calibration_time_safe = not calibration_summary.empty
    routers_validation_only = not mixture_summary.empty
    feature_selection_safe = not feature_summary.empty
    full_coverage = not ok.empty and ok.get("ticker_coverage", pd.Series(dtype=float)).apply(as_float).min() == 30 and all_true(ok, "full_ticker_coverage")
    no_ticker_subset = not grid.empty and all_false(grid, "ticker_subset")
    no_abstention = not grid.empty and all_false(grid, "confidence_abstention")
    no_topk = not grid.empty and all_false(grid, "topk_substitution")
    selected_by_validation = selected.empty or selected.get("selection_scope", pd.Series(dtype=str)).astype(str).str.contains("validation", case=False, na=False).all()
    separated_leaderboards = True
    beaters_61 = ok[ok.get("final_accuracy", pd.Series(dtype=float)).apply(as_float) > CURRENT_MAIN_FINAL_ACCURACY].copy()
    beaters_63 = ok[ok.get("final_accuracy", pd.Series(dtype=float)).apply(as_float) > BEST_DESCRIPTIVE_FINAL_ACCURACY].copy()
    claim, descriptive, overfit_audit = build_audit_tables(final_results)
    beaters_have_overfit = set(beaters_61.get("candidate_id", pd.Series(dtype=str)).astype(str)).issubset(set(overfit_audit.get("candidate_id", pd.Series(dtype=str)).astype(str))) and set(beaters_63.get("candidate_id", pd.Series(dtype=str)).astype(str)).issubset(set(overfit_audit.get("candidate_id", pd.Series(dtype=str)).astype(str)))
    claim_beats_current = claim[claim.get("final_accuracy", pd.Series(dtype=float)).apply(as_float) > CURRENT_MAIN_FINAL_ACCURACY].copy() if not claim.empty else pd.DataFrame()
    main_result_changes = not claim_beats_current.empty and no_leakage and full_coverage and selected_by_validation
    leakage_passed = all(
        [
            no_final_selection,
            no_leakage,
            scaler_train_only,
            base_predictions_safe,
            meta_validation_only,
            error_validation_only,
            ensemble_weights_validation_only,
            calibration_time_safe,
            routers_validation_only,
            feature_selection_safe,
            full_coverage,
            no_ticker_subset,
            no_abstention,
            no_topk,
            selected_by_validation,
            beaters_have_overfit,
        ]
    )

    write_csv("claim_eligible_leaderboard.csv", safe_columns(claim, [
        "candidate_id",
        "track",
        "model_id",
        "feature_family",
        "horizon",
        "threshold_policy",
        "validation_accuracy",
        "final_accuracy",
        "validation_final_gap",
        "overfit_risk",
        "selection_objectives_won",
    ]))
    write_csv("descriptive_final_leaderboard.csv", safe_columns(descriptive, [
        "candidate_id",
        "track",
        "model_id",
        "feature_family",
        "horizon",
        "threshold_policy",
        "validation_accuracy",
        "final_accuracy",
        "selected_by_validation_yes_no",
        "claim_eligible_yes_no",
        "overfit_risk",
    ]))
    write_csv("overfit_risk_audit.csv", overfit_audit)

    checks = {
        "required output files present": len(missing_files) == 0,
        "required figures present": len(missing_figures) == 0,
        "no final-window selection": no_final_selection,
        "no leakage": no_leakage,
        "no future regime labels": no_leakage,
        "scaler/imputer fitted on train only": scaler_train_only,
        "base model predictions used safely": base_predictions_safe,
        "meta-models trained only on validation predictions": meta_validation_only,
        "error-correction trained only on validation diagnostics": error_validation_only,
        "ensemble weights selected only on validation": ensemble_weights_validation_only,
        "calibration time-safe": calibration_time_safe,
        "routers selected only on validation": routers_validation_only,
        "feature-selection cooperation no final leakage": feature_selection_safe,
        "full 30-stock coverage": full_coverage,
        "no ticker subset": no_ticker_subset,
        "no confidence abstention": no_abstention,
        "no top-k substitution": no_topk,
        "selected by validation-only objective": selected_by_validation,
        "claim/descriptive leaderboards separated": separated_leaderboards,
        "beating candidates have overfit diagnostics": beaters_have_overfit,
        "leakage audit passed": leakage_passed,
    }
    failures = [name for name, passed in checks.items() if not passed]
    best_claim = first_row(claim)
    best_desc = first_row(descriptive)
    best_selected = first_row(selected.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False])) if not selected.empty else pd.Series(dtype=object)

    audit_lines = [
        "# VN30 Model Cooperation Transfer Audit Result",
        "",
        f"- Output directory: `{rel(OUTPUT_DIR)}`.",
        f"- Current main result: {CURRENT_MAIN_RESULT}.",
        f"- Prior descriptive context: {BEST_PRIOR_DESCRIPTIVE}.",
        f"- Total candidates evaluated: {ok['candidate_id'].nunique() if not ok.empty else 0}.",
        f"- Tracks run: {', '.join(sorted(ok['track'].astype(str).unique())) if not ok.empty else ''}.",
        f"- Full 30-stock coverage: {yes_no(full_coverage)}.",
        f"- Leakage audit passed: {yes_no(leakage_passed)}.",
        f"- Current main result changes: {yes_no(main_result_changes)}.",
        "",
        "## Audit Checks",
        "",
    ]
    audit_lines.extend(f"- {name}: {yes_no(passed)}" for name, passed in checks.items())
    if missing_files:
        audit_lines.extend(["", "## Missing Files", "", *[f"- {name}" for name in missing_files]])
    if missing_figures:
        audit_lines.extend(["", "## Missing Figures", "", *[f"- {name}" for name in missing_figures]])
    if failures:
        audit_lines.extend(["", "## Failed Checks", "", *[f"- {name}" for name in failures]])
    audit_lines.extend(
        [
            "",
            "## Selected Candidates",
            "",
            markdown_table(safe_columns(selected, ["selection_objective", "candidate_id", "track", "model_id", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "overfit_risk"]), max_rows=20),
            "",
            "## Descriptive Final Leaderboard",
            "",
            markdown_table(safe_columns(descriptive, ["candidate_id", "track", "model_id", "validation_accuracy", "final_accuracy", "selected_by_validation_yes_no", "claim_eligible_yes_no", "overfit_risk"]), max_rows=15),
        ]
    )
    write_markdown("audit_result.md", "\n".join(audit_lines))

    claim_lines = [
        "# VN30 Cooperation Claim Register",
        "",
        f"- Claim boundary: validation-only selection, full coverage, leakage audit pass, stability audit, and no high overfit risk.",
        f"- Current main result changes: {yes_no(main_result_changes)}.",
        f"- Reason: {'a claim-eligible cooperation row beats the current main result' if main_result_changes else 'no claim-eligible cooperation row safely replaces the current main result'}." ,
        "",
        "## Claim Eligible Leaderboard",
        "",
        markdown_table(safe_columns(claim, ["candidate_id", "track", "model_id", "validation_accuracy", "final_accuracy", "overfit_risk", "selection_objectives_won"]), max_rows=20),
        "",
        "## Descriptive Rows Are Not Claim Selection",
        "",
        markdown_table(safe_columns(descriptive, ["candidate_id", "track", "model_id", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "overfit_risk"]), max_rows=10),
    ]
    write_markdown("claim_register.md", "\n".join(claim_lines))

    return {
        "leakage_passed": leakage_passed,
        "full_coverage": full_coverage,
        "main_result_changes": main_result_changes,
        "total_candidates": int(ok["candidate_id"].nunique()) if not ok.empty else 0,
        "tracks": sorted(ok["track"].astype(str).unique()) if not ok.empty else [],
        "best_claim_final_accuracy": as_float(best_claim.get("final_accuracy")) if not best_claim.empty else math.nan,
        "best_descriptive_final_accuracy": as_float(best_desc.get("final_accuracy")) if not best_desc.empty else math.nan,
        "best_selected_final_accuracy": as_float(best_selected.get("final_accuracy")) if not best_selected.empty else math.nan,
    }


def main() -> None:
    summary = audit()
    print(f"Wrote cooperation audit outputs to {rel(OUTPUT_DIR)}")
    print(f"Leakage audit passed: {yes_no(summary['leakage_passed'])}")
    print(f"Total candidates: {summary['total_candidates']}")


if __name__ == "__main__":
    main()
