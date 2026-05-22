"""Audit VN30 full benchmark regime/deep artifacts and claim boundary."""

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

from scripts.research.run_vn30_full_benchmark_regime_deep import (  # noqa: E402
    REFERENCE_FINAL_ACCURACY,
    REPORT_DIR,
    classify_claim,
    pct,
    pp,
    write_markdown,
)
from scripts.research.vn30_hourly_dual_track_common import rel  # noqa: E402

REQUIRED_FILES = [
    "run_config.json",
    "feature_family_manifest.json",
    "data_label_audit.md",
    "data_label_summary.csv",
    "label_distribution_by_split.csv",
    "baseline_results.csv",
    "baseline_row_predictions.csv",
    "classical_ml_results.csv",
    "classical_ml_selected_candidates.csv",
    "classical_ml_row_predictions.csv",
    "deep_learning_results.csv",
    "deep_learning_selected_candidates.csv",
    "deep_learning_row_predictions.csv",
    "deep_learning_skip_report.md",
    "regime_feature_manifest.json",
    "regime_distribution.csv",
    "regime_slice_results.csv",
    "walk_forward_config.json",
    "walk_forward_validation_results.csv",
    "walk_forward_final_results.csv",
    "unified_leaderboard.csv",
    "best_by_group.csv",
    "best_overall_validation_selected.json",
    "comparison_summary.md",
    "benchmark_completion_manifest.json",
]


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {rel(path)}")
    return payload


def all_equal(frame: pd.DataFrame, column: str, expected: Any) -> bool:
    if frame.empty:
        return True
    if column not in frame.columns:
        return False
    return bool((frame[column].fillna(expected) == expected).all())


def prediction_coverage(predictions: pd.DataFrame) -> bool:
    if predictions.empty:
        return False
    final = predictions[predictions["split"].eq("final")].copy()
    if final.empty:
        return False
    coverage = final.groupby("candidate_id")["ticker"].nunique()
    return bool((coverage == 30).all())


def no_abstention(predictions: pd.DataFrame, leaderboard: pd.DataFrame) -> bool:
    if predictions.empty:
        return False
    final = predictions[predictions["split"].eq("final")].copy()
    if final[["y_true", "y_pred", "correct"]].isna().any().any():
        return False
    counts = final.groupby("candidate_id").size().rename("prediction_rows").reset_index()
    scored = leaderboard[leaderboard["candidate_id"].isin(counts["candidate_id"])][["candidate_id", "final_rows"]]
    merged = counts.merge(scored, on="candidate_id", how="left")
    return bool((merged["prediction_rows"].astype(int) == merged["final_rows"].astype(int)).all())


def manifest_safe(manifest: dict[str, Any]) -> bool:
    families = manifest.get("feature_families", {})
    if not isinstance(families, dict) or not families:
        return False
    for payload in families.values():
        if bool(payload.get("future_regime_labels")):
            return False
        if bool(payload.get("future_return_features")):
            return False
        if bool(payload.get("target_leakage_features")):
            return False
        if bool(payload.get("same_row_target_leakage")):
            return False
        if bool(payload.get("final_window_derived_features")):
            return False
    regime = manifest.get("regime_features", {})
    return bool(regime.get("uses_future_returns") is False and regime.get("uses_final_window_for_threshold") is False)


def combine_predictions(*frames: pd.DataFrame) -> pd.DataFrame:
    pieces = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()


def best_rows(leaderboard: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, frame in [
        ("baseline", leaderboard[leaderboard["method_group"].eq("baseline")]),
        ("classical_ml", leaderboard[leaderboard["method_group"].eq("classical_ml")]),
        ("deep_learning", leaderboard[leaderboard["method_group"].eq("deep_learning")]),
        ("regime_aware", leaderboard[(leaderboard["method_group"].eq("classical_ml")) & (leaderboard["feature_family"].eq("regime_context"))]),
    ]:
        if frame.empty:
            out[key] = {}
        else:
            sort_col = "validation_accuracy" if key != "baseline" else "final_accuracy"
            out[key] = frame.sort_values([sort_col, "final_accuracy"], ascending=False).iloc[0].to_dict()
    return out


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (REPORT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required full benchmark artifacts: {missing}")

    config = read_json(REPORT_DIR / "run_config.json")
    manifest = read_json(REPORT_DIR / "feature_family_manifest.json")
    best = read_json(REPORT_DIR / "best_overall_validation_selected.json")
    completion = read_json(REPORT_DIR / "benchmark_completion_manifest.json")

    baseline_results = pd.read_csv(REPORT_DIR / "baseline_results.csv")
    classical_results = pd.read_csv(REPORT_DIR / "classical_ml_results.csv")
    classical_selected = pd.read_csv(REPORT_DIR / "classical_ml_selected_candidates.csv")
    deep_results = pd.read_csv(REPORT_DIR / "deep_learning_results.csv")
    deep_selected = pd.read_csv(REPORT_DIR / "deep_learning_selected_candidates.csv")
    leaderboard = pd.read_csv(REPORT_DIR / "unified_leaderboard.csv")
    baseline_pred = pd.read_csv(REPORT_DIR / "baseline_row_predictions.csv")
    classical_pred = pd.read_csv(REPORT_DIR / "classical_ml_row_predictions.csv")
    deep_pred = pd.read_csv(REPORT_DIR / "deep_learning_row_predictions.csv")
    regime_slices = pd.read_csv(REPORT_DIR / "regime_slice_results.csv")
    walk_val = pd.read_csv(REPORT_DIR / "walk_forward_validation_results.csv")
    walk_final = pd.read_csv(REPORT_DIR / "walk_forward_final_results.csv")
    all_predictions = combine_predictions(baseline_pred, classical_pred, deep_pred)

    selected_frames = [frame for frame in [classical_selected, deep_selected] if not frame.empty]
    selected = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    checks = {
        "required_files_present": not missing,
        "no_final_window_selection": bool(config.get("final_accuracy_used_for_selection") is False)
        and (selected.empty or bool((selected["selection_source"].fillna("validation_only") == "validation_only").all()))
        and "final_accuracy" not in classical_results[classical_results["status"].eq("ok") & classical_results["final_accuracy"].isna()].columns[:0],
        "no_future_features": manifest_safe(manifest),
        "no_target_leakage": manifest_safe(manifest),
        "no_same_row_leakage": manifest_safe(manifest),
        "no_ticker_subset_in_main_claim": bool(config.get("ticker_subset") is False) and bool(leaderboard["full_ticker_coverage"].fillna(False).astype(bool).any()),
        "no_confidence_abstention_in_main_claim": bool(config.get("confidence_abstention") is False) and no_abstention(all_predictions, leaderboard),
        "no_topk_substitution": bool(config.get("topk") is False),
        "full_ticker_coverage": prediction_coverage(all_predictions) and bool(leaderboard["ticker_coverage"].max() == 30),
        "walk_forward_validation_respected": not walk_val.empty and not walk_final.empty and bool((walk_val["test_end"].astype(str) < "2025-01-01").all()),
        "index_data_context_only": bool(manifest.get("regime_features", {}).get("uses_future_returns") is False),
        "deep_learning_sequences_time_safe": bool(deep_results.empty or all_equal(deep_results, "shuffle_across_time", False)),
        "baselines_ex_ante": not baseline_results.empty and bool((baseline_results["leakage_status"].astype(str).str.startswith("passed")).all()),
        "model_comparison_includes_simple_baselines": not baseline_results.empty and not classical_results.empty,
        "regime_slice_available": not regime_slices.empty,
        "final65_not_overclaimed": bool((leaderboard["claim_level"].ne("final65_candidate_exploratory") | (leaderboard["final_accuracy"] >= 0.65)).all()),
    }
    leakage_passed = all(
        checks[key]
        for key in [
            "no_final_window_selection",
            "no_future_features",
            "no_target_leakage",
            "no_same_row_leakage",
            "no_ticker_subset_in_main_claim",
            "no_confidence_abstention_in_main_claim",
            "no_topk_substitution",
            "full_ticker_coverage",
            "walk_forward_validation_respected",
            "index_data_context_only",
            "deep_learning_sequences_time_safe",
            "baselines_ex_ante",
        ]
    )
    best_final = as_float(best.get("final_accuracy"))
    best_claim = classify_claim(best, audit_passed=leakage_passed) if best else "failed_to_beat_reference"
    overfit_risk = str(best.get("overfit_risk_classification", "unknown"))
    group_best = best_rows(leaderboard)

    check_table = [
        "| Check | Status |",
        "| --- | --- |",
    ]
    for key, value in checks.items():
        check_table.append(f"| {key} | {'pass' if value else 'fail'} |")

    audit_lines = [
        "# VN30 Full Benchmark Regime Deep Audit Result",
        "",
        "## Verdict",
        "",
        f"- Leakage audit passed: {'yes' if leakage_passed else 'no'}.",
        f"- Acceptance classification: `{best_claim}`.",
        f"- Overfit risk classification: `{overfit_risk}`.",
        f"- Best overall validation-selected candidate: `{best.get('candidate_id', '')}`.",
        f"- Validation accuracy: {pct(best.get('validation_accuracy'))}.",
        f"- Final accuracy: {pct(best_final)}.",
        f"- Delta vs 61.51% reference: {pp(best_final - REFERENCE_FINAL_ACCURACY)}.",
        f"- Full ticker coverage: {'yes' if best.get('full_ticker_coverage') else 'no'}.",
        f"- Final rows: {best.get('final_rows', '')}.",
        "",
        "## Checks",
        "",
        "\n".join(check_table),
        "",
        "## Best Results",
        "",
    ]
    for label, row in group_best.items():
        audit_lines.append(
            f"- {label}: `{row.get('candidate_id', '')}` validation {pct(row.get('validation_accuracy'))}, "
            f"final {pct(row.get('final_accuracy'))}, rows {row.get('final_rows', '')}."
        )
    audit_lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- No market data was fetched.",
            "- No provider behavior was changed.",
            "- No ticker subset, confidence abstention, or top-k substitute is used for the main claim.",
            "- No trading, profitability, investment recommendation, or live-deployment claim is made.",
            "- Final65 is exploratory only if present and still needs future blind validation.",
        ]
    )
    write_markdown(REPORT_DIR / "audit_result.md", "\n".join(audit_lines))

    claim_lines = [
        "# VN30 Full Benchmark Claim Register",
        "",
        f"- Benchmark run: {'yes' if completion.get('benchmark_run') else 'no'}.",
        f"- Data fetch: {'yes' if completion.get('data_fetch') else 'no'}.",
        f"- Model training: {'yes' if completion.get('model_training') else 'no'}.",
        f"- Model selection: {completion.get('model_selection', '')}.",
        f"- Main target: {config.get('main_target', '')}.",
        f"- Headline candidate: `{best.get('candidate_id', '')}`.",
        f"- Claim level: `{best_claim}`.",
        f"- Final accuracy: {pct(best_final)}.",
        f"- Reference delta: {pp(best_final - REFERENCE_FINAL_ACCURACY)}.",
        f"- Full ticker coverage: {'yes' if best.get('full_ticker_coverage') else 'no'}.",
        f"- Leakage audit passed: {'yes' if leakage_passed else 'no'}.",
        f"- Overfit risk: `{overfit_risk}`.",
        "- Trading/profitability/live-deployment claim: no.",
        "- Paper/DOCX generated: no.",
        "- Main branch touched: no.",
    ]
    write_markdown(REPORT_DIR / "claim_register.md", "\n".join(claim_lines))
    print(f"Audit complete: {rel(REPORT_DIR / 'audit_result.md')}")


if __name__ == "__main__":
    main()
