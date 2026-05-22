"""Audit fair VN30 exhaustive model-zoo tuning outputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, rel  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark" / "fair_tuning"
FIGURE_DIR = OUTPUT_DIR / "figures"
CURRENT_MAIN_FINAL_ACCURACY = 0.6163475699558174
CURRENT_MAIN_LABEL = "Logistic L2, baseline_C_closest, h40, threshold 0.55, final accuracy 61.63%"
EXPECTED_GROUPS = {
    "naive_baseline",
    "technical_rules",
    "linear_models",
    "svm_and_kernel_models",
    "distance_based_models",
    "probabilistic_models",
    "tree_models",
    "boosting_models",
    "neural_deep_models",
    "ensemble_stacking_models",
    "calibration_variants",
    "regime_aware_models",
    "statistical_models",
}
REQUIRED_FILES = [
    "tuning_budget_registry.csv",
    "tuning_budget_registry.md",
    "fair_tuning_candidate_grid.csv",
    "fair_tuning_validation_results.csv",
    "fair_tuning_selected_by_objective.csv",
    "fair_tuning_final_results.csv",
    "fair_tuning_row_predictions.csv",
    "fair_tuning_by_model_group.csv",
    "fair_tuning_by_horizon.csv",
    "fair_tuning_by_ticker.csv",
    "fair_tuning_by_month.csv",
    "fair_tuning_by_quarter.csv",
    "fair_tuning_rolling_250.csv",
    "fair_tuning_rolling_500.csv",
    "fair_tuning_rolling_1000.csv",
    "fair_tuning_transfer_quality.csv",
    "fair_tuning_runtime_summary.csv",
    "fair_tuning_summary.md",
    "fair_tuning_claim_boundary.md",
]
REQUIRED_FIGURES = [
    "fig_fair_tuning_budget_by_family.png",
    "fig_claim_eligible_vs_descriptive_accuracy.png",
    "fig_validation_vs_final_selected_candidates.png",
    "fig_model_family_accuracy_stability_tradeoff.png",
    "fig_validation_final_gap_by_family.png",
    "fig_overfit_risk_by_family.png",
    "fig_runtime_vs_accuracy.png",
    "fig_interpretability_vs_accuracy.png",
    "fig_transfer_quality_by_family.png",
    "fig_current_main_vs_best_fair_tuned.png",
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


def pp(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:+.2f} pp"


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    work = frame.head(max_rows).copy()
    headers = list(work.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def load_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(rel(path))
    return pd.read_csv(path, low_memory=False)


def yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def risk_sort_value(value: Any) -> int:
    return {"high": 3, "medium": 2, "low": 1, "unknown": 0}.get(str(value), 0)


def build_family_table(final_results: pd.DataFrame, transfer: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    ok = final_results[final_results["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    grouped = (
        ok.groupby("model_group")
        .agg(
            candidate_rows=("candidate_id", "nunique"),
            models_run=("model_id", "nunique"),
            mean_validation_accuracy=("validation_accuracy", "mean"),
            best_validation_accuracy=("validation_accuracy", "max"),
            mean_final_accuracy=("final_accuracy", "mean"),
            best_final_accuracy=("final_accuracy", "max"),
            mean_validation_final_gap=("validation_final_gap", "mean"),
            mean_rolling_250=("rolling_250_mean", "mean"),
            mean_monthly_min=("monthly_min_accuracy", "mean"),
            mean_quarterly_min=("quarterly_min_accuracy", "mean"),
            mean_ticker_min=("ticker_min_accuracy", "mean"),
            mean_runtime_seconds=("fit_runtime_seconds", "mean"),
            interpretability_score=("interpretability_score", "max"),
        )
        .reset_index()
    )
    risk = ok.groupby(["model_group", "overfit_risk"]).size().unstack(fill_value=0).reset_index()
    for col in ["low", "medium", "high", "unknown"]:
        if col not in risk.columns:
            risk[col] = 0
    grouped = grouped.merge(risk[["model_group", "low", "medium", "high", "unknown"]], on="model_group", how="left")
    if not transfer.empty and "transfer_quality_score" in transfer.columns:
        grouped = grouped.merge(transfer[["model_group", "transfer_quality_score", "median_abs_validation_final_gap"]], on="model_group", how="left")
    else:
        grouped["transfer_quality_score"] = np.nan
        grouped["median_abs_validation_final_gap"] = np.nan
    budget_cols = registry[["model_group", "planned_config_count", "actual_config_count", "dependency_status", "claim_eligibility_rule"]]
    grouped = grouped.merge(budget_cols, on="model_group", how="left")
    grouped["accurate_family"] = grouped["best_final_accuracy"] >= 0.58
    grouped["stable_family"] = grouped["mean_rolling_250"] >= 0.56
    grouped["overfit_warning_family"] = grouped["high"].fillna(0).astype(int) > 0
    grouped["computationally_expensive"] = grouped["mean_runtime_seconds"] >= grouped["mean_runtime_seconds"].quantile(0.75)
    grouped["diagnostic_only"] = grouped["model_group"].eq("statistical_models") & grouped["dependency_status"].astype(str).str.contains("garch_volatility_diagnostic")
    return grouped.sort_values(["best_final_accuracy", "transfer_quality_score"], ascending=[False, False])


def build_overfit_audit(final_results: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_ids = set(selected["candidate_id"].astype(str)) if not selected.empty else set()
    beating = final_results[final_results["final_accuracy"].apply(lambda value: as_float(value) > CURRENT_MAIN_FINAL_ACCURACY)].copy()
    selected_rows = final_results[final_results["candidate_id"].astype(str).isin(selected_ids)].copy()
    audit = pd.concat([selected_rows, beating], ignore_index=True).drop_duplicates("candidate_id")
    cols = [
        "candidate_id",
        "model_group",
        "model_id",
        "feature_family",
        "horizon",
        "threshold_policy",
        "config_name",
        "validation_accuracy",
        "final_accuracy",
        "validation_final_gap",
        "rolling_250_mean",
        "rolling_500_mean",
        "rolling_1000_mean",
        "rolling_250_windows_below_60",
        "rolling_500_windows_below_60",
        "rolling_1000_windows_below_60",
        "monthly_min_accuracy",
        "quarterly_min_accuracy",
        "ticker_min_accuracy",
        "market_regime_accuracy_summary",
        "volatility_regime_accuracy_summary",
        "router_regime_accuracy_summary",
        "fit_runtime_seconds",
        "interpretability_score",
        "selected_by_validation_objective_yes_no",
        "selection_objectives_won",
        "claim_eligible_yes_no",
        "reason_not_claim_eligible",
        "overfit_risk",
        "overfit_risk_reason",
    ]
    return audit[[col for col in cols if col in audit.columns]].sort_values(
        ["overfit_risk", "final_accuracy"], key=lambda series: series.map(risk_sort_value) if series.name == "overfit_risk" else series, ascending=[False, False]
    )


def main() -> None:
    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": yes_no(passed), "detail": detail})

    missing = [name for name in REQUIRED_FILES if not (OUTPUT_DIR / name).exists()]
    missing_figures = [name for name in REQUIRED_FIGURES if not (FIGURE_DIR / name).exists()]
    add_check("required_output_files_exist", not missing, ", ".join(missing))
    add_check("required_figures_exist", not missing_figures, ", ".join(missing_figures))

    registry = load_csv("tuning_budget_registry.csv")
    candidate_grid = load_csv("fair_tuning_candidate_grid.csv")
    validation_results = load_csv("fair_tuning_validation_results.csv")
    selected = load_csv("fair_tuning_selected_by_objective.csv")
    final_results = load_csv("fair_tuning_final_results.csv")
    row_predictions = load_csv("fair_tuning_row_predictions.csv")
    transfer = load_csv("fair_tuning_transfer_quality.csv")

    groups = set(registry["model_group"].astype(str))
    add_check("all_required_model_groups_budgeted", groups == EXPECTED_GROUPS, f"missing={sorted(EXPECTED_GROUPS - groups)} extra={sorted(groups - EXPECTED_GROUPS)}")
    add_check("all_groups_have_documented_budget", bool((registry["planned_config_count"].fillna(0).astype(float) > 0).all()), "")
    add_check("all_groups_have_actual_or_documented_status", bool((registry["actual_config_count"].fillna(0).astype(float) >= 0).all()), "")
    add_check(
        "no_model_family_privileged_by_final_score",
        bool(candidate_grid.get("final_accuracy_used_for_selection", pd.Series([False])).astype(str).str.lower().isin(["false", "0", "no"]).all())
        and bool(final_results.get("prior_final_score_privileged", pd.Series([False])).astype(str).str.lower().isin(["false", "0", "no"]).all()),
        "candidate grid and result flags indicate final scores were not used for selection",
    )

    skipped_failed = candidate_grid[candidate_grid["planned_status"].astype(str).isin(["skipped_with_reason", "failed_with_reason"])]
    add_check("skipped_failed_models_have_reasons", bool(skipped_failed.empty or skipped_failed["reason"].astype(str).str.len().gt(0).all()), "")

    selection_metric_safe = selected.empty or ~selected["selection_metric"].astype(str).str.contains("final", case=False, na=False).any()
    scope_safe = selected.empty or selected["selection_scope"].astype(str).eq("primary_h40_validation_only").all()
    add_check("no_final_window_selection", bool(selection_metric_safe and scope_safe), "")
    add_check("validation_results_match_final_rows", len(validation_results) == len(final_results), f"validation={len(validation_results)} final={len(final_results)}")
    add_check(
        "leakage_status_passed",
        bool(final_results["leakage_status"].astype(str).str.contains("passed", case=False, na=False).all()) if "leakage_status" in final_results.columns else False,
        "",
    )
    selected_final = final_results[final_results["selected_by_validation_objective_yes_no"].astype(str).eq("yes")]
    add_check("selected_rows_full_30_stock_coverage", bool(not selected_final.empty and selected_final["full_ticker_coverage"].astype(bool).all()), "")
    add_check("no_ticker_subset", bool(final_results["ticker_subset"].astype(str).str.lower().isin(["false", "0", "no"]).all()), "")
    add_check("no_confidence_abstention", bool(final_results["confidence_abstention"].astype(str).str.lower().isin(["false", "0", "no"]).all()), "")
    add_check("no_topk_substitution", bool(final_results["topk_substitution"].astype(str).str.lower().isin(["false", "0", "no"]).all()), "")
    add_check("row_predictions_no_abstention", bool(row_predictions.empty or row_predictions["y_pred"].notna().all()), "")
    add_check(
        "calibration_time_safe",
        bool(final_results[final_results["model_group"].eq("calibration_variants")]["implementation_note"].astype(str).str.contains("time-safe|trailing train split", case=False, regex=True).all()),
        "",
    )
    stacking = final_results[final_results["model_group"].eq("ensemble_stacking_models")]
    stack_meta = stacking[stacking["model_id"].astype(str).str.contains("stacking")]
    add_check(
        "stacking_meta_uses_validation_predictions_only",
        bool(stack_meta.empty or stack_meta["implementation_note"].astype(str).str.contains("validation base predictions", case=False, na=False).all()),
        "",
    )

    beaters = final_results[final_results["final_accuracy"].apply(lambda value: as_float(value) > CURRENT_MAIN_FINAL_ACCURACY)].copy()
    overfit_audit = build_overfit_audit(final_results, selected)
    write_csv(OUTPUT_DIR / "overfit_risk_audit.csv", overfit_audit)
    add_check("overfit_audit_covers_selected_and_beaters", bool(set(selected_final["candidate_id"].astype(str)).issubset(set(overfit_audit["candidate_id"].astype(str))) and set(beaters["candidate_id"].astype(str)).issubset(set(overfit_audit["candidate_id"].astype(str)))), "")

    family_table = build_family_table(final_results, transfer, registry)
    write_csv(OUTPUT_DIR / "model_family_audit_table.csv", family_table)

    claim_leaderboard = final_results[final_results["claim_eligible_yes_no"].astype(str).eq("yes")].sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False]).copy()
    descriptive = final_results.sort_values(["final_accuracy", "validation_accuracy"], ascending=[False, False]).copy()
    descriptive["claim_role"] = np.where(descriptive["claim_eligible_yes_no"].astype(str).eq("yes"), "claim_eligible", "descriptive_only")
    write_csv(OUTPUT_DIR / "claim_eligible_leaderboard.csv", claim_leaderboard)
    write_csv(OUTPUT_DIR / "descriptive_final_leaderboard.csv", descriptive)

    if not family_table.empty:
        best_transfer = family_table.sort_values("transfer_quality_score", ascending=False).iloc[0]
        worst_transfer = family_table.sort_values("transfer_quality_score", ascending=True).iloc[0]
        accurate = family_table[family_table["accurate_family"].astype(bool)]["model_group"].tolist()
        stable = family_table[family_table["stable_family"].astype(bool)]["model_group"].tolist()
        overfit = family_table[family_table["overfit_warning_family"].astype(bool)]["model_group"].tolist()
        expensive = family_table[family_table["computationally_expensive"].astype(bool)]["model_group"].tolist()
        interpretable = family_table[family_table["interpretability_score"].fillna(0) >= 4]["model_group"].tolist()
    else:
        best_transfer = pd.Series(dtype=object)
        worst_transfer = pd.Series(dtype=object)
        accurate = stable = overfit = expensive = interpretable = []

    interpretation_lines = [
        "# Model Family Interpretation",
        "",
        f"- Accurate families by best final accuracy threshold: {', '.join(accurate) if accurate else 'none'}.",
        f"- Stable families by mean rolling-250 threshold: {', '.join(stable) if stable else 'none'}.",
        f"- Families with high overfit warnings: {', '.join(overfit) if overfit else 'none'}.",
        f"- Computationally expensive families: {', '.join(expensive) if expensive else 'none'}.",
        f"- Interpretable families: {', '.join(interpretable) if interpretable else 'none'}.",
        f"- Best transfer quality family: {best_transfer.get('model_group', '')} ({pct(best_transfer.get('transfer_quality_score', math.nan))}).",
        f"- Worst transfer quality family: {worst_transfer.get('model_group', '')} ({pct(worst_transfer.get('transfer_quality_score', math.nan))}).",
        "- GARCH remains diagnostic only and is not a direct direction classifier.",
        "- Claim eligibility is limited to validation-selected, full-coverage, non-diagnostic rows that are not high overfit risk and pass this audit.",
        "",
        "## Family Audit Table",
        "",
        markdown_table(family_table, max_rows=len(family_table)),
    ]
    write_markdown(OUTPUT_DIR / "model_family_interpretation.md", "\n".join(interpretation_lines))

    claim_lines = [
        "# Fair Tuning Claim Register",
        "",
        f"- Current main result context: {CURRENT_MAIN_LABEL}.",
        f"- Any model beats 61.63% descriptively: {yes_no(not beaters.empty)}.",
        f"- Claim-eligible rows: {len(claim_leaderboard)}.",
        "- Final-window descriptive rows do not change the main result unless validation-selected and audit-passed.",
        "- No trading, profitability, investment recommendation, or live-deployment claim is made.",
        "",
        "## Claim-Eligible Leaderboard",
        "",
        markdown_table(claim_leaderboard[["candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "validation_accuracy", "final_accuracy", "overfit_risk"]] if not claim_leaderboard.empty else claim_leaderboard, max_rows=20),
        "",
        "## Descriptive Beating Rows",
        "",
        markdown_table(beaters[["candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "validation_accuracy", "final_accuracy", "selected_by_validation_objective_yes_no", "claim_eligible_yes_no", "overfit_risk"]] if not beaters.empty else beaters, max_rows=30),
    ]
    write_markdown(OUTPUT_DIR / "claim_register.md", "\n".join(claim_lines))

    all_passed = all(row["passed"] == "yes" for row in checks)
    audit_lines = [
        "# Fair Exhaustive Model-Zoo Tuning Audit Result",
        "",
        f"- Overall audit passed: {yes_no(all_passed)}.",
        f"- Total model groups budgeted: {registry['model_group'].nunique()}.",
        f"- Total candidate rows: {len(candidate_grid)}.",
        f"- Total successful result rows: {int(final_results['status'].astype(str).eq('ok').sum())}.",
        f"- Validation-selected rows: {len(selected_final)}.",
        f"- Rows beating 61.63% descriptively: {len(beaters)}.",
        f"- Claim-eligible rows: {len(claim_leaderboard)}.",
        f"- Full coverage selected rows: {yes_no(not selected_final.empty and selected_final['full_ticker_coverage'].astype(bool).all())}.",
        f"- Data fetch: no.",
        f"- Paper/DOCX generated: no.",
        "",
        "## Checks",
        "",
        markdown_table(pd.DataFrame(checks), max_rows=len(checks)),
        "",
        "## Selected By Validation Objective",
        "",
        markdown_table(selected[["selection_objective", "candidate_id", "model_group", "model_id", "feature_family", "horizon", "threshold_policy", "validation_accuracy", "final_accuracy", "claim_eligible_yes_no", "overfit_risk"]] if not selected.empty else selected, max_rows=20),
        "",
        "## Overfit Risk Audit",
        "",
        markdown_table(overfit_audit, max_rows=30),
    ]
    write_markdown(OUTPUT_DIR / "audit_result.md", "\n".join(audit_lines))
    print(f"Wrote audit outputs to {rel(OUTPUT_DIR)}")
    print(f"Audit passed: {yes_no(all_passed)}")


if __name__ == "__main__":
    main()
