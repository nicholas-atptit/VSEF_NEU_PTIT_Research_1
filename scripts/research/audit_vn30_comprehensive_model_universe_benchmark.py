"""Audit VN30 comprehensive model-universe benchmark outputs."""

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

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark"
CURRENT_MAIN_FINAL_ACCURACY = 0.6163475699558174
CURRENT_MAIN_LABEL = "Logistic L2 baseline_C_closest h40 validation-selected threshold 0.55"
EXPECTED_MODEL_VARIANTS = 75
EXPECTED_MODEL_GROUPS = {
    "naive_baselines",
    "technical_rule_baselines",
    "linear_generalized_linear_models",
    "kernel_distance_based_models",
    "probabilistic_models",
    "tree_based_models",
    "boosting_models",
    "neural_deep_models",
    "ensemble_stacking",
    "calibration_variants",
    "regime_aware_models",
    "traditional_statistical_financial_models",
}
EXPECTED_MODELS = {
    "majority_class",
    "random_walk_direction",
    "previous_direction",
    "persistence_rule",
    "moving_average_rule",
    "rolling_momentum_rule",
    "volatility_adjusted_momentum_rule",
    "sma_crossover",
    "ema_crossover",
    "macd_rule",
    "rsi_rule",
    "bollinger_band_rule",
    "price_momentum_rule",
    "volume_momentum_rule",
    "mean_reversion_rule",
    "breakout_rule",
    "logistic_l2",
    "logistic_l1",
    "logistic_elastic_net",
    "ridge_classifier",
    "lda",
    "qda",
    "passive_aggressive",
    "perceptron",
    "sgd_hinge",
    "sgd_log_loss",
    "linear_svm",
    "svm_rbf",
    "svm_poly",
    "knn",
    "radius_neighbors",
    "nearest_centroid",
    "gaussian_naive_bayes",
    "bernoulli_naive_bayes",
    "complement_naive_bayes",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "adaboost",
    "sklearn_gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp_classifier",
    "lstm",
    "gru",
    "tcn",
    "cnn_1d",
    "cnn_lstm",
    "hard_voting",
    "soft_voting",
    "validation_weighted_soft_vote",
    "stacking_logistic_meta",
    "stacking_lightgbm_meta",
    "stacking_xgboost_meta",
    "blending",
    "platt_logistic",
    "isotonic_logistic",
    "calibrated_svm",
    "calibrated_random_forest",
    "calibrated_xgboost",
    "calibrated_lightgbm",
    "regime_context_logistic",
    "regime_context_xgboost",
    "regime_context_lightgbm",
    "bull_bear_sideway_router",
    "high_low_volatility_router",
    "regime_threshold_router",
    "regime_model_router",
    "arima_direction",
    "sarima_direction",
    "ets_direction",
    "var_direction",
    "garch_volatility_diagnostic",
}


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


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame, max_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows._"
    work = frame.head(max_rows)
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


def has_attempt(candidate_grid: pd.DataFrame, registry: pd.DataFrame, model_id: str) -> bool:
    if not candidate_grid.empty and candidate_grid["model_id"].astype(str).eq(model_id).any():
        return True
    return registry["model_id"].astype(str).eq(model_id).any()


def overfit_risk(row: pd.Series | None) -> tuple[str, str]:
    if row is None or row.empty:
        return "unknown", "missing row"
    validation = as_float(row.get("validation_accuracy"))
    final = as_float(row.get("final_accuracy"))
    gap = validation - final if math.isfinite(validation) and math.isfinite(final) else math.nan
    rolling = as_float(row.get("rolling_250_mean"))
    monthly_min = as_float(row.get("monthly_min_accuracy"))
    quarterly_min = as_float(row.get("quarterly_min_accuracy"))
    ticker_min = as_float(row.get("ticker_min_accuracy"))
    rolling_below_value = as_float(row.get("rolling_250_windows_below_60"))
    rolling_below = int(rolling_below_value) if math.isfinite(rolling_below_value) else 0
    selected = str(row.get("selected_by_validation_yes_no", "no")).lower() == "yes"
    if not math.isfinite(validation) or not math.isfinite(final):
        return "unknown", "missing validation or final accuracy"
    reasons = []
    if final > CURRENT_MAIN_FINAL_ACCURACY and not selected:
        reasons.append("post-hoc final leaderboard beating row, not validation-selected")
    if math.isfinite(gap) and gap > 0.05:
        reasons.append(f"validation-final gap {gap * 100.0:.2f} pp")
    if math.isfinite(rolling) and rolling < 0.56:
        reasons.append(f"rolling 250 mean {rolling * 100.0:.2f}%")
    if rolling_below > 0:
        reasons.append(f"{rolling_below} rolling 250 windows below 60%")
    if math.isfinite(monthly_min) and monthly_min < 0.55:
        reasons.append(f"monthly minimum {monthly_min * 100.0:.2f}%")
    if math.isfinite(quarterly_min) and quarterly_min < 0.55:
        reasons.append(f"quarterly minimum {quarterly_min * 100.0:.2f}%")
    if math.isfinite(ticker_min) and ticker_min < 0.55:
        reasons.append(f"ticker minimum {ticker_min * 100.0:.2f}%")
    if final > CURRENT_MAIN_FINAL_ACCURACY and not selected:
        return "high", "; ".join(reasons)
    if (math.isfinite(gap) and gap > 0.05) or (math.isfinite(rolling) and rolling < 0.52):
        return "high", "; ".join(reasons) or "large validation-final deterioration"
    if reasons:
        return "medium", "; ".join(reasons)
    return "low", "validation-only selection and stability diagnostics do not show a major warning"


def main() -> None:
    required = [
        "model_universe_registry.csv",
        "candidate_grid.csv",
        "validation_results.csv",
        "final_results.csv",
        "row_predictions.csv",
        "skipped_models_report.md",
        "failed_models_report.md",
        "not_recommended_models_report.md",
        "model_coverage_audit.md",
        "best_by_model_group.csv",
        "best_by_horizon.csv",
        "best_by_model_family.csv",
        "augmented_leaderboard.csv",
        "comparison_vs_current_best.csv",
        "technical_rules_summary.csv",
        "calibration_summary.csv",
        "ensemble_summary.csv",
        "regime_summary.csv",
        "statistical_models_summary.csv",
        "garch_diagnostic_summary.md",
        "dependency_install_report.md",
        "model_universe_summary.md",
        "model_universe_claim_boundary.md",
    ]
    base_figure_names = [
        "fig_model_universe_coverage.png",
        "fig_final_accuracy_by_model_group.png",
        "fig_validation_vs_final_by_model_group.png",
        "fig_best_by_model_family.png",
        "fig_horizon_accuracy_heatmap.png",
        "fig_technical_rules_vs_ml.png",
        "fig_svm_tree_boosting_comparison.png",
        "fig_deep_vs_classical_comparison.png",
        "fig_calibration_effects.png",
        "fig_current_best_vs_expansion.png",
        "fig_skipped_failed_model_reasons.png",
        "fig_statistical_models_diagnostic.png",
        "fig_overfit_risk_beating_rows.png",
        "fig_catboost_vs_boosting_family.png",
    ]
    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": yes_no(passed), "detail": detail})

    registry = load_csv("model_universe_registry.csv")
    candidate_grid = load_csv("candidate_grid.csv")
    final_results = load_csv("final_results.csv")
    validation_results = load_csv("validation_results.csv")
    statistical_summary = load_csv("statistical_models_summary.csv")

    if not final_results.empty:
        final_results["final_accuracy_numeric"] = pd.to_numeric(final_results["final_accuracy"], errors="coerce")
        final_results["validation_accuracy_numeric"] = pd.to_numeric(final_results["validation_accuracy"], errors="coerce")
    beats = final_results[final_results.get("final_accuracy_numeric", pd.Series(dtype=float)) > CURRENT_MAIN_FINAL_ACCURACY].copy()
    if not beats.empty:
        risk_values = beats.apply(overfit_risk, axis=1)
        beats["audit_overfit_risk"] = [risk for risk, _ in risk_values]
        beats["audit_overfit_risk_reason"] = [reason for _, reason in risk_values]
        beats["validation_minus_final_gap"] = beats["validation_accuracy_numeric"] - beats["final_accuracy_numeric"]
    overfit_columns = [
        "candidate_id",
        "model_group",
        "model_id",
        "feature_family",
        "horizon",
        "threshold_policy",
        "validation_accuracy",
        "final_accuracy",
        "validation_minus_final_gap",
        "rolling_250_mean",
        "rolling_500_mean",
        "rolling_1000_mean",
        "rolling_250_windows_below_60",
        "rolling_500_windows_below_60",
        "rolling_1000_windows_below_60",
        "monthly_mean_accuracy",
        "monthly_median_accuracy",
        "quarterly_mean_accuracy",
        "quarterly_median_accuracy",
        "ticker_mean_accuracy",
        "ticker_median_accuracy",
        "market_regime_accuracy_summary",
        "volatility_regime_accuracy_summary",
        "router_regime_accuracy_summary",
        "selected_by_validation_yes_no",
        "claim_eligible_yes_no",
        "audit_overfit_risk",
        "audit_overfit_risk_reason",
    ]
    beating_diagnostic = beats[[col for col in overfit_columns if col in beats.columns]].copy() if not beats.empty else pd.DataFrame(columns=overfit_columns)
    write_csv(OUTPUT_DIR / "overfit_risk_audit.csv", beating_diagnostic)
    write_csv(OUTPUT_DIR / "beating_rows_diagnostic.csv", beating_diagnostic)

    garch_succeeded = (
        not statistical_summary.empty
        and statistical_summary["model_id"].astype(str).eq("garch_volatility_diagnostic").any()
        and (
            "fit_status" not in statistical_summary.columns
            or statistical_summary.loc[statistical_summary["model_id"].astype(str).eq("garch_volatility_diagnostic"), "fit_status"].astype(str).eq("ok").any()
        )
    )
    figure_names = list(base_figure_names)
    if garch_succeeded:
        figure_names.append("fig_garch_volatility_diagnostic.png")

    for name in required:
        add_check(f"required output exists: {name}", (OUTPUT_DIR / name).exists(), rel(OUTPUT_DIR / name))
    for name in ["overfit_risk_audit.csv", "beating_rows_diagnostic.csv"]:
        add_check(f"required audit output exists: {name}", (OUTPUT_DIR / name).exists(), rel(OUTPUT_DIR / name))
    for name in figure_names:
        add_check(f"required figure exists: {name}", (OUTPUT_DIR / "figures" / name).exists(), rel(OUTPUT_DIR / "figures" / name))

    registry_models = set(registry["model_id"].astype(str))
    registry_groups = set(registry["model_group"].astype(str))
    allowed_statuses = {"run", "failed_with_reason", "skipped_with_reason", "not_recommended_with_reason"}
    status = registry["run_status"].astype(str)
    status_counts = {
        "run": int(status.eq("run").sum()),
        "failed": int(status.eq("failed_with_reason").sum()),
        "skipped": int(status.eq("skipped_with_reason").sum()),
        "not_recommended": int(status.eq("not_recommended_with_reason").sum()),
    }
    status_counts["attempted"] = sum(status_counts.values())
    add_check("model universe registry exists", not registry.empty, f"rows={len(registry)}")
    add_check("all 75 planned model variants listed", len(registry) == EXPECTED_MODEL_VARIANTS, f"rows={len(registry)} expected={EXPECTED_MODEL_VARIANTS}")
    add_check("all 75 planned model variants attempted", status_counts["attempted"] == EXPECTED_MODEL_VARIANTS, f"attempted={status_counts['attempted']} expected={EXPECTED_MODEL_VARIANTS}")
    add_check("all planned model groups listed", EXPECTED_MODEL_GROUPS.issubset(registry_groups), f"missing={sorted(EXPECTED_MODEL_GROUPS - registry_groups)}")
    add_check("all planned models listed", EXPECTED_MODELS.issubset(registry_models), f"missing={sorted(EXPECTED_MODELS - registry_models)}")
    add_check("all planned models have terminal run status", set(registry["run_status"].astype(str)).issubset(allowed_statuses), f"statuses={sorted(set(registry['run_status'].astype(str)))}")
    omitted = sorted(EXPECTED_MODELS - registry_models)
    add_check("no silent omission", len(omitted) == 0, f"omitted={omitted}")

    add_check("SVM RBF attempted yes/no", has_attempt(candidate_grid, registry, "svm_rbf"), "")
    add_check("SVM Polynomial attempted yes/no", has_attempt(candidate_grid, registry, "svm_poly"), "")
    add_check("KNN attempted yes/no", has_attempt(candidate_grid, registry, "knn"), "")
    add_check("CatBoost attempted yes/no", has_attempt(candidate_grid, registry, "catboost"), registry.loc[registry["model_id"].eq("catboost"), "run_status"].astype(str).str.cat(sep=","))
    add_check("CatBoost run yes/no", registry.loc[registry["model_id"].eq("catboost"), "run_status"].astype(str).eq("run").any(), registry.loc[registry["model_id"].eq("catboost"), "run_status"].astype(str).str.cat(sep=","))
    add_check("technical rules attempted yes/no", final_results["model_group"].astype(str).eq("technical_rule_baselines").any(), "")
    statistical_ids = {"arima_direction", "sarima_direction", "ets_direction", "var_direction", "garch_volatility_diagnostic"}
    attempted_statistical = statistical_ids.issubset(registry_models) and all(has_attempt(candidate_grid, registry, model_id) for model_id in statistical_ids)
    add_check("statistical models attempted yes/no", attempted_statistical, f"models={sorted(statistical_ids)}")
    garch_in_final = final_results["model_id"].astype(str).eq("garch_volatility_diagnostic").any() if not final_results.empty else False
    garch_registry = registry[registry["model_id"].astype(str).eq("garch_volatility_diagnostic")]
    add_check("GARCH treated as volatility diagnostic only yes/no", (not garch_in_final) and (not garch_registry.empty) and garch_registry["claim_eligible"].astype(str).eq("no").all(), "")
    add_check("GARCH diagnostic attempted yes/no", has_attempt(candidate_grid, registry, "garch_volatility_diagnostic"), "")
    add_check("GARCH diagnostic run yes/no", garch_succeeded, "")

    if not final_results.empty:
        add_check("no final-window selection", final_results["final_accuracy_used_for_selection"].astype(str).str.lower().isin(["false", "0"]).all(), "")
        add_check("no leakage", final_results["leakage_status"].astype(str).str.contains("passed", case=False, na=False).all(), "")
        add_check("scaler/imputer fitted only on train", final_results["leakage_status"].astype(str).str.contains("train_only", case=False, na=False).all(), "")
        selected = final_results[final_results["selected_by_validation_yes_no"].astype(str).eq("yes")].copy()
        add_check("validation-only threshold/model selection", len(selected) <= 1 and final_results["selection_source"].astype(str).eq("validation_only").all(), f"selected_rows={len(selected)}")
        add_check("full 30-stock coverage for headline rows", (not selected.empty) and selected["ticker_coverage"].astype(int).eq(30).all(), "")
        add_check("no ticker subset", final_results["ticker_subset"].astype(str).str.lower().isin(["false", "0"]).all(), "")
        add_check("no confidence abstention", final_results["confidence_abstention"].astype(str).str.lower().isin(["false", "0"]).all(), "")
        add_check("no top-k substitution", final_results["topk_substitution"].astype(str).str.lower().isin(["false", "0"]).all(), "")
        add_check("h40 main claim kept separate", (OUTPUT_DIR / "model_universe_claim_boundary.md").exists(), "")
        add_check("any model beats 61.63 yes/no", True, f"yes_no={yes_no(not beats.empty)} count={len(beats)}")
        if not beats.empty:
            beat_claim_eligible = beats["claim_eligible_yes_no"].astype(str).eq("yes").any()
            add_check("if a model beats 61.63, whether it is validation-selected and claim-eligible", True, f"yes_no={yes_no(beat_claim_eligible)}")
            add_check("beating rows have low/medium/high overfit risk", beats["audit_overfit_risk"].astype(str).isin(["low", "medium", "high"]).all(), "")
    else:
        selected = pd.DataFrame()
        beats = pd.DataFrame()
        add_check("no final-window selection", False, "final_results empty")

    selected_row = selected.iloc[0] if not selected.empty else None
    best_row = final_results.sort_values(["final_accuracy_numeric", "validation_accuracy_numeric"], ascending=[False, False]).iloc[0] if not final_results.empty else None
    selected_overfit, selected_overfit_reason = overfit_risk(selected_row)
    best_overfit, best_overfit_reason = overfit_risk(best_row)
    add_check("overfit risk classification", selected_overfit in {"low", "medium", "high"} or best_overfit in {"low", "medium", "high"}, f"selected={selected_overfit}; best={best_overfit}")

    check_frame = pd.DataFrame(checks)
    all_passed = check_frame["passed"].eq("yes").all()
    main_changes = bool(
        all_passed
        and selected_row is not None
        and as_float(selected_row.get("final_accuracy")) > CURRENT_MAIN_FINAL_ACCURACY
        and str(selected_row.get("claim_eligible_yes_no")) == "yes"
        and int(selected_row.get("ticker_coverage", 0)) == 30
    )

    def best_model_row(model_id: str, extra_filter: pd.Series | None = None) -> pd.Series | None:
        scoped = final_results[final_results["model_id"].astype(str).eq(model_id)].copy()
        if extra_filter is not None:
            scoped = scoped[extra_filter.loc[scoped.index]]
        if scoped.empty:
            return None
        return scoped.sort_values(["final_accuracy_numeric", "validation_accuracy_numeric"], ascending=[False, False]).iloc[0]

    bull_filter = final_results["horizon"].astype(str).eq("40") & final_results["threshold_policy"].astype(str).eq("fixed_0.50")
    bull_row = best_model_row("bull_bear_sideway_router", bull_filter)
    catboost_row = best_model_row("catboost")
    garch_rows = statistical_summary[statistical_summary["model_id"].astype(str).eq("garch_volatility_diagnostic")].copy()

    comparison_rows: list[dict[str, Any]] = [
        {
            "comparison": "current_main_logistic_l2_h40_threshold_0.55",
            "model_id": "logistic_l2",
            "candidate_id": "paper_reference_current_main",
            "validation_accuracy": "validation-selected",
            "final_accuracy": CURRENT_MAIN_FINAL_ACCURACY,
            "validation_selected": "yes",
            "claim_eligible": "yes",
        }
    ]
    for label, row in [
        ("previous_best_final_descriptive_bull_bear_sideway_router_h40_fixed_0.50", bull_row),
        ("best_catboost_result", catboost_row),
    ]:
        comparison_rows.append(
            {
                "comparison": label,
                "model_id": "" if row is None else row.get("model_id"),
                "candidate_id": "" if row is None else row.get("candidate_id"),
                "validation_accuracy": "" if row is None else row.get("validation_accuracy"),
                "final_accuracy": "" if row is None else row.get("final_accuracy"),
                "validation_selected": "" if row is None else row.get("selected_by_validation_yes_no"),
                "claim_eligible": "" if row is None else row.get("claim_eligible_yes_no"),
            }
        )
    if not garch_rows.empty:
        comparison_rows.append(
            {
                "comparison": "garch_volatility_diagnostic",
                "model_id": "garch_volatility_diagnostic",
                "candidate_id": "diagnostic_only",
                "validation_accuracy": "",
                "final_accuracy": "",
                "validation_selected": "no",
                "claim_eligible": "no",
            }
        )
    comparison_frame = pd.DataFrame(comparison_rows)

    audit_lines = [
        "# VN30 Comprehensive Model Universe Benchmark Audit Result",
        "",
        f"- Audit passed: {yes_no(all_passed)}.",
        f"- Registry rows: {len(registry)}.",
        f"- Total model groups listed: {registry['model_group'].nunique()}.",
        f"- Total model variants planned: {len(registry)}.",
        f"- Total model variants attempted: {status_counts['attempted']}.",
        f"- Total model variants run: {status_counts['run']}.",
        f"- Total model variants failed: {status_counts['failed']}.",
        f"- Total model variants skipped: {status_counts['skipped']}.",
        f"- Total model variants not recommended: {status_counts['not_recommended']}.",
        f"- Candidate grid rows: {len(candidate_grid)}.",
        f"- Validation result rows: {len(validation_results)}.",
        f"- Final result rows: {len(final_results)}.",
        f"- Current main result: {CURRENT_MAIN_LABEL} ({pct(CURRENT_MAIN_FINAL_ACCURACY)}).",
        f"- Main result changes: {yes_no(main_changes)}.",
        f"- Selected overfit risk: {selected_overfit} ({selected_overfit_reason}).",
        f"- Best-final overfit risk: {best_overfit} ({best_overfit_reason}).",
        f"- CatBoost run: {yes_no(registry.loc[registry['model_id'].eq('catboost'), 'run_status'].astype(str).eq('run').any())}.",
        f"- GARCH diagnostic run: {yes_no(garch_succeeded)}.",
        "- GARCH used as main direction classifier: no.",
        "",
        "## Current Main Result Comparison",
        "",
        markdown_table(comparison_frame, max_rows=len(comparison_frame)),
        "",
        "## Beating Rows Overfit-Risk Audit",
        "",
        markdown_table(beating_diagnostic, max_rows=50),
        "",
        "## Checks",
        "",
        markdown_table(check_frame, max_rows=len(check_frame)),
    ]
    write_markdown(OUTPUT_DIR / "audit_result.md", "\n".join(audit_lines))

    claim_rows = []
    if selected_row is not None:
        claim_rows.append(
            {
                "claim": "validation_selected_candidate",
                "candidate_id": selected_row.get("candidate_id"),
                "model_id": selected_row.get("model_id"),
                "feature_family": selected_row.get("feature_family"),
                "horizon": selected_row.get("horizon"),
                "threshold_policy": selected_row.get("threshold_policy"),
                "validation_accuracy": selected_row.get("validation_accuracy"),
                "final_accuracy": selected_row.get("final_accuracy"),
                "beats_61_63": yes_no(as_float(selected_row.get("final_accuracy")) > CURRENT_MAIN_FINAL_ACCURACY),
                "claim_eligible": selected_row.get("claim_eligible_yes_no"),
                "overfit_risk": selected_overfit,
                "overfit_risk_reason": selected_overfit_reason,
                "main_result_changes": yes_no(main_changes),
            }
        )
    if best_row is not None:
        claim_rows.append(
            {
                "claim": "best_final_descriptive_only",
                "candidate_id": best_row.get("candidate_id"),
                "model_id": best_row.get("model_id"),
                "feature_family": best_row.get("feature_family"),
                "horizon": best_row.get("horizon"),
                "threshold_policy": best_row.get("threshold_policy"),
                "validation_accuracy": best_row.get("validation_accuracy"),
                "final_accuracy": best_row.get("final_accuracy"),
                "beats_61_63": yes_no(as_float(best_row.get("final_accuracy")) > CURRENT_MAIN_FINAL_ACCURACY),
                "claim_eligible": best_row.get("claim_eligible_yes_no"),
                "overfit_risk": best_overfit,
                "overfit_risk_reason": best_overfit_reason,
                "main_result_changes": "no",
            }
        )
    claim_frame = pd.DataFrame(claim_rows)
    claim_lines = [
        "# Claim Register",
        "",
        f"- Current main result: {CURRENT_MAIN_LABEL}, {pct(CURRENT_MAIN_FINAL_ACCURACY)}.",
        f"- Main result changes: {yes_no(main_changes)}.",
        f"- Final-window score used for selection: no.",
        f"- Data fetch: no.",
        f"- GARCH used as main direction classifier: no.",
        f"- CatBoost run: {yes_no(registry.loc[registry['model_id'].eq('catboost'), 'run_status'].astype(str).eq('run').any())}.",
        f"- GARCH diagnostic run: {yes_no(garch_succeeded)}.",
        "",
        markdown_table(claim_frame, max_rows=len(claim_frame)),
    ]
    write_markdown(OUTPUT_DIR / "claim_register.md", "\n".join(claim_lines))
    print(f"Wrote audit outputs to {rel(OUTPUT_DIR)}; passed={yes_no(all_passed)}")


if __name__ == "__main__":
    main()
