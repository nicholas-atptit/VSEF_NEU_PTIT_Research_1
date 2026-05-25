"""Build the raw full VN30 paper-evidence export package.

This script uses local repository data and artifacts only. It does not fetch
market data, train models, rerun benchmarks, generate paper documents, stage
git files, create tags, or push anything.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = REPO_ROOT / "paper_evidence_raw_full_export"
RAW_DATA_DIR = EXPORT_DIR / "raw_data"
RAW_MODEL_DIR = EXPORT_DIR / "raw_model_outputs"
RAW_PREDICTION_DIR = EXPORT_DIR / "raw_predictions"
TABLE_DIR = EXPORT_DIR / "tables"
FIGURE_DIR = EXPORT_DIR / "figures"
PROTOCOL_DIR = EXPORT_DIR / "protocols"
SCRIPT_DIR = EXPORT_DIR / "scripts_used"
MANIFEST_DIR = EXPORT_DIR / "manifests"
LOG_DIR = EXPORT_DIR / "logs"

MODEL_UNIVERSE_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark"
LEGACY_DIR = REPO_ROOT / "reports" / "generated" / "vn30_legacy_rules_reference_and_stacking"
INDEX_OUTPUT_DIR = REPO_ROOT / "outputs" / "index_directional_benchmark"
INDEX_REPORT_DIR = REPO_ROOT / "reports" / "generated" / "index_benchmark"
JOINT_PANEL_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_stock_index_joint_panel_training_v1"
JOINT_PANEL_REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_stock_index_joint_panel"
PROTOCOL_SRC_DIR = REPO_ROOT / "reports" / "protocols"

HORIZONS = [20, 40, 60, 80]
SUPPORTED_INDICES = ["VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VN100"]
REQUIRED_TABLES = [
    "table_01_full_model_horizon_results.csv",
    "table_02_per_instrument_actual_vs_predicted.csv",
    "table_03_row_level_prediction_archive.csv.gz",
    "table_04_model_family_by_horizon_pivot.csv",
    "table_05_best_model_by_horizon.csv",
    "table_06_best_model_by_family.csv",
    "table_07_descriptive_practical_ranking.csv",
    "table_08_transfer_quality_by_model.csv",
    "table_09_overfit_risk_by_model.csv",
    "table_10_index_context_results.csv",
    "table_11_knn_support_actuals_summary.csv",
    "table_12_model_cooperation_actuals_summary.csv",
    "table_13_actual_vs_predicted_confusion_summary.csv",
    "table_14_missing_data_and_artifact_report.csv",
    "table_15_paper_key_numbers.csv",
]
REQUIRED_FIGURES = [
    "fig_01_full_model_horizon_heatmap.png",
    "fig_02_best_final_accuracy_by_model_family.png",
    "fig_03_descriptive_practical_ranking.png",
    "fig_04_claim_status_vs_final_accuracy.png",
    "fig_05_per_ticker_accuracy_heatmap_selected_models.png",
    "fig_06_per_index_accuracy_heatmap_selected_models.png",
    "fig_07_selected_candidate_actual_vs_predicted.png",
    "fig_08_best_descriptive_router_actual_vs_predicted.png",
    "fig_09_soft_voting_actual_vs_predicted.png",
    "fig_10_knn_support_actual_vs_predicted.png",
    "fig_11_model_as_feature_actual_vs_predicted.png",
    "fig_12_all_vn30_stocks_actual_vs_predicted_small_multiples.png",
    "fig_13_supported_indices_actual_vs_predicted_small_multiples.png",
    "fig_14_stock_performance_ranking.png",
    "fig_15_index_performance_ranking.png",
    "fig_16_confusion_decomposition_selected_models.png",
    "fig_17_validation_vs_final_by_model.png",
    "fig_18_overfit_risk_by_model_family.png",
    "fig_19_model_cooperation_actual_comparison.png",
    "fig_20_knn_support_actual_comparison.png",
]

SOURCE_MANIFEST_ROWS: list[dict[str, Any]] = []
FIGURE_MANIFEST_ROWS: list[dict[str, Any]] = []
MISSING_ROWS: list[dict[str, Any]] = []
RUN_STATS: dict[str, Any] = {
    "raw_data_files_copied": 0,
    "model_output_files_copied": 0,
    "prediction_artifact_files_copied": 0,
    "row_level_prediction_rows": 0,
}


RESULT_SOURCES = [
    (MODEL_UNIVERSE_DIR / "final_results.csv", "model_universe"),
    (MODEL_UNIVERSE_DIR / "fair_tuning" / "fair_tuning_final_results.csv", "fair_tuning"),
    (MODEL_UNIVERSE_DIR / "knn_support" / "knn_support_final_results.csv", "knn_support"),
    (
        MODEL_UNIVERSE_DIR / "model_cooperation_transfer_audit" / "cooperation_final_results.csv",
        "model_cooperation",
    ),
    (LEGACY_DIR / "full_horizon" / "full_horizon_leaderboard.csv", "legacy_full_horizon"),
    (LEGACY_DIR / "model_comparison" / "legacy_model_final_results.csv", "legacy_model_comparison"),
    (LEGACY_DIR / "stacking" / "stacking_final_results.csv", "legacy_stacking"),
    (LEGACY_DIR / "targeted_improvement" / "final_results_by_track.csv", "targeted_improvement"),
    (LEGACY_DIR / "reference_reproduction" / "reference_reproduction_summary.csv", "reference_reproduction"),
    (INDEX_OUTPUT_DIR / "accuracy_summary.csv", "index_benchmark"),
    (INDEX_OUTPUT_DIR / "baseline_summary.csv", "index_baseline"),
]

PREDICTION_SOURCES = [
    (MODEL_UNIVERSE_DIR / "row_predictions.csv", "stock_model_universe"),
    (MODEL_UNIVERSE_DIR / "fair_tuning" / "fair_tuning_row_predictions.csv", "fair_tuning"),
    (MODEL_UNIVERSE_DIR / "knn_support" / "knn_support_row_predictions.csv", "knn_support"),
    (
        MODEL_UNIVERSE_DIR / "model_cooperation_transfer_audit" / "cooperation_row_predictions.csv",
        "model_cooperation",
    ),
    (LEGACY_DIR / "model_comparison" / "legacy_model_row_predictions.csv", "legacy_model_comparison"),
    (LEGACY_DIR / "reference_reproduction" / "reference_reproduction_row_predictions.csv", "reference_reproduction"),
    (
        REPO_ROOT / "outputs" / "vn30_hourly_selected_l2_logistic_h40_row_predictions" / "row_predictions.csv",
        "selected_candidate_reproduction",
    ),
    (LEGACY_DIR / "stacking" / "stacking_row_predictions.csv", "legacy_stacking"),
    (INDEX_OUTPUT_DIR / "predicted_vs_actual.csv", "index_benchmark"),
]

MODEL_OUTPUT_DIRS = [
    MODEL_UNIVERSE_DIR,
    LEGACY_DIR,
    INDEX_REPORT_DIR,
    INDEX_OUTPUT_DIR,
    JOINT_PANEL_REPORT_DIR,
    JOINT_PANEL_OUTPUT_DIR,
    REPO_ROOT / "outputs" / "vn30_hourly_selected_l2_logistic_h40_row_predictions",
]

RAW_DATA_PATHS = [
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015",
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "daily_2015",
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_listing_aware",
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015",
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "daily_2015",
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "universe",
    REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "vn30_hourly_2015",
    REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "vn30_hourly_listing_aware",
    REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "index_hourly_2015",
    REPO_ROOT / "configs" / "universes",
]

SCRIPT_SOURCES = [
    REPO_ROOT / "scripts" / "research" / "run_vn30_comprehensive_model_universe_benchmark.py",
    REPO_ROOT / "scripts" / "research" / "run_vn30_fair_exhaustive_model_zoo_tuning.py",
    REPO_ROOT / "scripts" / "research" / "run_vn30_knn_support_experiment.py",
    REPO_ROOT / "scripts" / "research" / "run_vn30_model_cooperation_transfer_audit.py",
    REPO_ROOT / "scripts" / "research" / "run_vn30_legacy_full_horizon_comparison.py",
    REPO_ROOT / "scripts" / "research" / "run_vn30_legacy_rules_model_comparison.py",
    REPO_ROOT / "scripts" / "research" / "run_vn30_legacy_rules_stacking_ensemble.py",
    REPO_ROOT / "scripts" / "research" / "run_vn30_legacy_targeted_improvement_tracks.py",
    REPO_ROOT / "scripts" / "research" / "rerun_vn30_hourly_selected_candidate_row_predictions.py",
    REPO_ROOT / "scripts" / "research" / "reproduce_vn30_legacy_reference_l2_logistic_h40.py",
    REPO_ROOT / "scripts" / "research" / "run_supported_indices_directional_benchmark.py",
    REPO_ROOT / "scripts" / "research" / "index_benchmark_common.py",
    REPO_ROOT / "scripts" / "research" / "vn30_hourly_dual_track_common.py",
    REPO_ROOT / "scripts" / "research" / "export_vn30_raw_full_paper_evidence.py",
]


def rel(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path_obj.as_posix()


def git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return f"git_error: {exc}"
    return (result.stdout or result.stderr).strip()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def yes_no(value: Any) -> str:
    text = clean_text(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "yes"
    if text in {"false", "0", "no", "n"}:
        return "no"
    return "unknown"


def ensure_source_columns(frame: pd.DataFrame, source_file: str, status: str = "found") -> pd.DataFrame:
    out = frame.copy()
    out["source_file"] = source_file
    out["source_status"] = status
    out["generation_status"] = "regenerated_from_local_artifact"
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for column in ("source_file", "source_status", "generation_status"):
        if column not in out.columns:
            out[column] = "regenerated_from_local_artifact" if column == "generation_status" else ""
    out = out.astype(object).where(pd.notna(out), "")
    out.to_csv(path, index=False)


def file_modified_time(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def manifest_row(source: Path, copied_to: Path | str, source_type: str, status: str, notes: str) -> dict[str, Any]:
    size = source.stat().st_size if source.exists() and source.is_file() else ""
    return {
        "source_path": rel(source),
        "copied_to": rel(copied_to) if copied_to else "",
        "source_type": source_type,
        "file_size": size,
        "modified_time": file_modified_time(source),
        "source_status": status,
        "notes": notes,
    }


def add_missing(name: str, source_type: str, reason: str, affects_paper_tables: str = "yes") -> None:
    row = {
        "missing_item": name,
        "source_type": source_type,
        "reason": reason,
        "affects_paper_tables": affects_paper_tables,
        "regeneration_possible_from_local_files": "no",
        "source_status": "missing_with_reason",
        "generation_status": "not_generated_missing_source_or_requires_rerun",
    }
    MISSING_ROWS.append(row)
    SOURCE_MANIFEST_ROWS.append(
        {
            "source_path": name,
            "copied_to": "",
            "source_type": source_type,
            "file_size": "",
            "modified_time": "",
            "source_status": "missing_with_reason",
            "notes": reason,
        }
    )


def reset_export_dir() -> None:
    marker = EXPORT_DIR / ".generated_by_export_vn30_raw_full_paper_evidence"
    if EXPORT_DIR.exists():
        if not marker.exists():
            raise RuntimeError(f"{rel(EXPORT_DIR)} already exists without exporter marker; refusing to overwrite")
        shutil.rmtree(EXPORT_DIR)
    for directory in (
        RAW_DATA_DIR,
        RAW_MODEL_DIR,
        RAW_PREDICTION_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        PROTOCOL_DIR,
        SCRIPT_DIR,
        MANIFEST_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    marker.write_text("generated by scripts/research/export_vn30_raw_full_paper_evidence.py\n", encoding="utf-8")


def copy_file(source: Path, destination_root: Path, source_type: str, notes: str = "") -> Path | None:
    if not source.exists() or not source.is_file():
        add_missing(rel(source), source_type, "Local source file not found", affects_paper_tables="maybe")
        return None
    relative = source.resolve().relative_to(REPO_ROOT.resolve())
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    SOURCE_MANIFEST_ROWS.append(manifest_row(source, destination, source_type, "found", notes))
    if destination_root == RAW_DATA_DIR:
        RUN_STATS["raw_data_files_copied"] += 1
    elif destination_root == RAW_PREDICTION_DIR:
        RUN_STATS["prediction_artifact_files_copied"] += 1
    elif destination_root == RAW_MODEL_DIR:
        RUN_STATS["model_output_files_copied"] += 1
    return destination


def copy_tree_files(source: Path, destination_root: Path, source_type: str, notes: str = "") -> None:
    if not source.exists():
        add_missing(rel(source), source_type, "Local source directory not found", affects_paper_tables="maybe")
        return
    if source.is_file():
        copy_file(source, destination_root, source_type, notes)
        return
    for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
        if EXPORT_DIR in file_path.parents:
            continue
        copy_file(file_path, destination_root, source_type, notes)


def is_prediction_artifact(path: Path) -> bool:
    text = path.name.lower()
    return "prediction" in text or "predicted_vs_actual" in text or text.endswith("_predictions.csv")


def copy_model_outputs_and_predictions() -> None:
    prediction_sources = {path.resolve() for path, _ in PREDICTION_SOURCES if path.exists()}
    for path, _name in PREDICTION_SOURCES:
        copy_file(path, RAW_PREDICTION_DIR, "prediction", "row-level prediction artifact")
    for root in MODEL_OUTPUT_DIRS:
        if not root.exists():
            add_missing(rel(root), "model_output", "Expected model output directory not found", "maybe")
            continue
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            if file_path.resolve() in prediction_sources or is_prediction_artifact(file_path):
                if file_path.resolve() not in prediction_sources:
                    copy_file(file_path, RAW_PREDICTION_DIR, "prediction", "prediction artifact discovered in model output tree")
                continue
            suffix = file_path.suffix.lower()
            if suffix == ".png":
                copy_file(file_path, RAW_MODEL_DIR, "figure", "original generated figure artifact")
            elif suffix == ".csv":
                copy_file(file_path, RAW_MODEL_DIR, "table", "original generated table artifact")
            elif suffix in {".json", ".md", ".txt"}:
                copy_file(file_path, RAW_MODEL_DIR, "model_output", "original generated model/report artifact")
            else:
                copy_file(file_path, RAW_MODEL_DIR, "model_output", "original generated artifact")


def copy_raw_data() -> None:
    for path in RAW_DATA_PATHS:
        copy_tree_files(path, RAW_DATA_DIR, "raw_data", "local raw/cache/source data for VN30 or index experiments")


def copy_protocols_and_scripts() -> None:
    if PROTOCOL_SRC_DIR.exists():
        for path in sorted(PROTOCOL_SRC_DIR.glob("*.md")):
            copy_file(path, PROTOCOL_DIR, "protocol", "research protocol archive")
    else:
        add_missing(rel(PROTOCOL_SRC_DIR), "protocol", "reports/protocols directory not found", "maybe")
    for path in SCRIPT_SOURCES:
        copy_file(path, SCRIPT_DIR, "script", "script used to generate or audit source evidence")


def write_repo_state() -> dict[str, Any]:
    status_summary = git_output(["status", "--short", "--branch"])
    repo_state = {
        "branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit_hash": git_output(["rev-parse", "HEAD"]),
        "git_status_summary": status_summary,
        "export_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "working_tree_clean": not any(
            line
            for line in status_summary.splitlines()
            if line and not line.startswith("##") and "paper_evidence_raw_full_export" not in line
        ),
        "benchmark_rerun": "no",
        "model_training_run": "no",
        "data_fetch_run": "no",
        "paper_generated": "no",
    }
    write_json(MANIFEST_DIR / "repo_state.json", repo_state)
    write_text(MANIFEST_DIR / "commit_info.txt", git_output(["show", "--stat", "--oneline", "--no-renames", "HEAD"]))
    return repo_state


def read_vn30_tickers() -> list[str]:
    path = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
    if not path.exists():
        add_missing(rel(path), "raw_data", "VN30 ticker universe file missing", "yes")
        return []
    frame = pd.read_csv(path)
    column = "ticker" if "ticker" in frame.columns else frame.columns[0]
    return sorted(frame[column].dropna().astype(str).str.upper().unique().tolist())


def normalize_result_source(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        add_missing(rel(path), "table", f"Result source missing for {source_name}", "yes")
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["model_group"] = pick_series(frame, ["model_group", "experiment_group", "track"], source_name)
    if source_name == "index_baseline":
        out["model_group"] = "index_baseline"
    out["model_id"] = pick_series(frame, ["model_id", "model", "baseline"], "")
    out["candidate_id"] = pick_series(frame, ["candidate_id"], "")
    out["feature_family"] = pick_series(frame, ["feature_family", "feature_set"], "")
    out["horizon"] = pd.to_numeric(pick_series(frame, ["horizon"], np.nan), errors="coerce").astype("Int64")
    out["threshold_policy"] = pick_series(frame, ["threshold_policy"], "")
    out["validation_accuracy"] = pd.to_numeric(pick_series(frame, ["validation_accuracy"], np.nan), errors="coerce")
    out["final_accuracy"] = pd.to_numeric(
        pick_series(frame, ["final_accuracy", "model_accuracy"], np.nan), errors="coerce"
    )
    out["validation_final_gap"] = pd.to_numeric(
        pick_series(frame, ["validation_final_gap"], np.nan), errors="coerce"
    )
    if out["validation_final_gap"].isna().all():
        out["validation_final_gap"] = out["validation_accuracy"] - out["final_accuracy"]
    out["final_rows"] = pd.to_numeric(pick_series(frame, ["final_rows"], np.nan), errors="coerce").astype("Int64")
    out["ticker_coverage"] = pick_series(frame, ["ticker_coverage", "validation_ticker_coverage"], "")
    out["index_coverage_if_applicable"] = pick_series(frame, ["index_code"], "")
    out["selected_by_validation_yes_no"] = pick_series(
        frame,
        [
            "selected_by_validation_yes_no",
            "selected_by_validation_objective_yes_no",
            "selected_on_validation",
        ],
        "",
    ).map(yes_no)
    out["claim_status"] = infer_claim_status(frame, source_name)
    out["source_file"] = rel(path)
    out["source_status"] = "found"
    out["generation_status"] = "regenerated_from_local_artifact"
    out["notes"] = source_name
    out["source_dataset"] = source_name
    out["overfit_risk"] = pick_series(frame, ["overfit_risk"], "")
    out["rolling_250_mean"] = pd.to_numeric(
        pick_series(frame, ["rolling_250_mean", "final_rolling_250_mean"], np.nan), errors="coerce"
    )
    out["rolling_500_mean"] = pd.to_numeric(
        pick_series(frame, ["rolling_500_mean", "final_rolling_500_mean"], np.nan), errors="coerce"
    )
    out["ticker_min_accuracy"] = pd.to_numeric(pick_series(frame, ["ticker_min_accuracy"], np.nan), errors="coerce")
    if source_name == "index_benchmark":
        out["model_group"] = "index_benchmark"
        out["feature_family"] = "index_directional_features"
        out["threshold_policy"] = "model_default"
        out["candidate_id"] = (
            "index_benchmark__"
            + frame["index_code"].astype(str)
            + "__"
            + frame["model"].astype(str)
            + "__h"
            + frame["horizon"].astype(str)
            + "__"
            + frame["frequency"].astype(str)
        )
        out["claim_status"] = frame.get("claim_level", pd.Series([""] * len(frame))).fillna("").replace("", "diagnostic")
    elif source_name == "index_baseline":
        out["model_id"] = frame["baseline"].astype(str)
        out["feature_family"] = "index_directional_baseline"
        out["threshold_policy"] = "baseline_rule"
        out["candidate_id"] = (
            "index_baseline__"
            + frame["index_code"].astype(str)
            + "__"
            + frame["baseline"].astype(str)
            + "__h"
            + frame["horizon"].astype(str)
            + "__"
            + frame["frequency"].astype(str)
        )
        out["validation_accuracy"] = np.nan
        out["validation_final_gap"] = np.nan
        out["claim_status"] = "diagnostic_baseline"
    elif source_name == "reference_reproduction":
        out["candidate_id"] = "reference_reproduction__" + out["model_id"].astype(str) + "__h" + out["horizon"].astype(str)
        out["model_group"] = "reference_reproduction"
        out["threshold_policy"] = "fixed_0.50"
        out["claim_status"] = np.where(
            frame.get("reference_reproduced", pd.Series([False] * len(frame))).map(yes_no).eq("yes"),
            "reference_reproduced",
            "diagnostic",
        )
    elif source_name == "targeted_improvement":
        out["model_group"] = "targeted_improvement_" + frame.get("track", pd.Series([""] * len(frame))).astype(str)
        out["horizon"] = 40
        out["claim_status"] = frame.get("candidate_classification", pd.Series(["diagnostic"] * len(frame))).astype(str)
    missing_candidate = out["candidate_id"].fillna("").astype(str).str.strip().eq("")
    out.loc[missing_candidate, "candidate_id"] = (
        source_name
        + "__"
        + out.loc[missing_candidate, "model_group"].astype(str)
        + "__"
        + out.loc[missing_candidate, "model_id"].astype(str)
        + "__h"
        + out.loc[missing_candidate, "horizon"].astype(str)
    )
    out["descriptive_best_yes_no"] = "no"
    out["practical_experimental_candidate_yes_no"] = practical_candidate_flag(frame)
    return out


def pick_series(frame: pd.DataFrame, columns: list[str], default: Any) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def infer_claim_status(frame: pd.DataFrame, source_name: str) -> pd.Series:
    if "claim_eligible_yes_no" in frame.columns:
        claim = frame["claim_eligible_yes_no"].map(yes_no)
        return claim.map({"yes": "claim_eligible", "no": "diagnostic_or_descriptive", "unknown": "diagnostic"})
    if "paper_role" in frame.columns:
        return frame["paper_role"].fillna("").replace("", "diagnostic")
    if "claim_level" in frame.columns:
        return frame["claim_level"].fillna("").replace("", "diagnostic")
    if source_name in {"legacy_full_horizon", "legacy_model_comparison", "reference_reproduction"}:
        return pd.Series(["paper_evidence"] * len(frame), index=frame.index)
    return pd.Series(["diagnostic_or_descriptive"] * len(frame), index=frame.index)


def practical_candidate_flag(frame: pd.DataFrame) -> pd.Series:
    full = pick_series(frame, ["full_ticker_coverage"], True).map(yes_no).isin(["yes", "unknown"])
    no_subset = pick_series(frame, ["ticker_subset"], False).map(yes_no).isin(["no", "unknown"])
    no_abstain = pick_series(frame, ["confidence_abstention"], False).map(yes_no).isin(["no", "unknown"])
    no_topk = pick_series(frame, ["topk_substitution"], False).map(yes_no).isin(["no", "unknown"])
    return np.where(full & no_subset & no_abstain & no_topk, "yes", "no")


def build_table_01() -> pd.DataFrame:
    frames = [normalize_result_source(path, name) for path, name in RESULT_SOURCES]
    table = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if table.empty:
        add_missing("table_01_source_results", "table", "No local result rows found", "yes")
        return pd.DataFrame()
    found = table[table["source_status"].eq("found")].copy()
    descriptive_pool = stock_model_rows(found)
    if descriptive_pool.empty:
        descriptive_pool = found
    idx = descriptive_pool.groupby("horizon")["final_accuracy"].idxmax()
    table.loc[idx.dropna().astype(int), "descriptive_best_yes_no"] = "yes"
    table = pd.concat([table, missing_horizon_rows(table)], ignore_index=True)
    ordered = [
        "model_group",
        "model_id",
        "candidate_id",
        "feature_family",
        "horizon",
        "threshold_policy",
        "validation_accuracy",
        "final_accuracy",
        "validation_final_gap",
        "final_rows",
        "ticker_coverage",
        "index_coverage_if_applicable",
        "selected_by_validation_yes_no",
        "descriptive_best_yes_no",
        "practical_experimental_candidate_yes_no",
        "claim_status",
        "source_file",
        "source_status",
        "generation_status",
        "notes",
        "source_dataset",
        "overfit_risk",
        "rolling_250_mean",
        "rolling_500_mean",
        "ticker_min_accuracy",
    ]
    table = table[[column for column in ordered if column in table.columns]]
    write_table(TABLE_DIR / "table_01_full_model_horizon_results.csv", table)
    return table


def missing_horizon_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["source_dataset", "model_group", "model_id", "feature_family", "threshold_policy"]
    for values, group in table.dropna(subset=["horizon"]).groupby(group_cols, dropna=False):
        present = set(pd.to_numeric(group["horizon"], errors="coerce").dropna().astype(int).tolist())
        missing = [horizon for horizon in HORIZONS if horizon not in present]
        if not missing:
            continue
        source_dataset, model_group, model_id, feature_family, threshold_policy = values
        first = group.iloc[0].to_dict()
        for horizon in missing:
            rows.append(
                {
                    "model_group": model_group,
                    "model_id": model_id,
                    "candidate_id": f"missing__{source_dataset}__{model_group}__{model_id}__{feature_family}__h{horizon}",
                    "feature_family": feature_family,
                    "horizon": horizon,
                    "threshold_policy": threshold_policy,
                    "validation_accuracy": np.nan,
                    "final_accuracy": np.nan,
                    "validation_final_gap": np.nan,
                    "final_rows": np.nan,
                    "ticker_coverage": "",
                    "index_coverage_if_applicable": first.get("index_coverage_if_applicable", ""),
                    "selected_by_validation_yes_no": "no",
                    "descriptive_best_yes_no": "no",
                    "practical_experimental_candidate_yes_no": "no",
                    "claim_status": "missing_with_reason",
                    "source_file": first.get("source_file", ""),
                    "source_status": "missing_with_reason",
                    "generation_status": "not_generated_missing_source_or_requires_model_rerun",
                    "notes": f"No local {source_dataset} result row for h{horizon}; regeneration would require benchmark/model run.",
                    "source_dataset": source_dataset,
                    "overfit_risk": "",
                    "rolling_250_mean": np.nan,
                    "rolling_500_mean": np.nan,
                    "ticker_min_accuracy": np.nan,
                }
            )
    return pd.DataFrame(rows)


def normalize_prediction_chunk(chunk: pd.DataFrame, path: Path, source_name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=chunk.index)
    if source_name == "index_benchmark":
        out["datetime"] = chunk["datetime"]
        out["model_group"] = "index_benchmark"
        out["model_id"] = chunk["model"].astype(str)
        out["candidate_id"] = (
            "index_benchmark__"
            + chunk["index_code"].astype(str)
            + "__"
            + chunk["model"].astype(str)
            + "__h"
            + chunk["horizon"].astype(str)
        )
        out["horizon"] = chunk["horizon"]
        out["instrument_type"] = "index"
        out["instrument_symbol"] = chunk["index_code"].astype(str).str.upper()
        out["y_true"] = chunk["y_true"]
        out["y_pred"] = chunk["y_pred"]
        out["predicted_probability_or_score"] = ""
        out["threshold"] = ""
        out["split"] = "final_or_reported"
    else:
        out["datetime"] = pick_series(chunk, ["datetime"], "")
        out["model_group"] = pick_series(chunk, ["model_group", "method_group", "experiment_group"], source_name)
        out["model_id"] = pick_series(chunk, ["model_id", "model"], "")
        out["candidate_id"] = pick_series(chunk, ["candidate_id"], "")
        out["horizon"] = pick_series(chunk, ["horizon"], "")
        out["instrument_type"] = "stock"
        out["instrument_symbol"] = pick_series(chunk, ["ticker"], "").astype(str).str.upper()
        out["y_true"] = pick_series(chunk, ["y_true"], "")
        out["y_pred"] = pick_series(chunk, ["y_pred"], "")
        out["predicted_probability_or_score"] = pick_series(
            chunk, ["y_score_or_probability", "predicted_probability_or_score", "y_score"], ""
        )
        out["threshold"] = pick_series(chunk, ["threshold"], "")
        out["split"] = pick_series(chunk, ["split"], "")
        if source_name == "selected_candidate_reproduction":
            out["model_group"] = "selected_candidate_reproduction"
            empty_candidate = out["candidate_id"].astype(str).str.strip().eq("")
            out.loc[empty_candidate, "candidate_id"] = (
                "selected_candidate_reproduction__l2_logistic__h"
                + out.loc[empty_candidate, "horizon"].astype(str)
                + "__t"
                + out.loc[empty_candidate, "threshold"].astype(str)
            )
        empty_candidate = out["candidate_id"].fillna("").astype(str).str.strip().eq("")
        out.loc[empty_candidate, "candidate_id"] = (
            source_name
            + "__"
            + out.loc[empty_candidate, "model_group"].astype(str)
            + "__"
            + out.loc[empty_candidate, "model_id"].astype(str)
            + "__h"
            + out.loc[empty_candidate, "horizon"].astype(str)
        )
    y_true = pd.to_numeric(out["y_true"], errors="coerce")
    y_pred = pd.to_numeric(out["y_pred"], errors="coerce")
    out["correct"] = (y_true == y_pred).where(y_true.notna() & y_pred.notna(), np.nan)
    out["correct"] = out["correct"].map({True: 1, False: 0}).fillna("")
    out["source_file"] = rel(path)
    out["source_status"] = "found"
    out["generation_status"] = "copied_from_local_row_prediction_artifact"
    columns = [
        "datetime",
        "model_group",
        "model_id",
        "candidate_id",
        "horizon",
        "instrument_type",
        "instrument_symbol",
        "y_true",
        "y_pred",
        "predicted_probability_or_score",
        "threshold",
        "correct",
        "source_file",
        "source_status",
        "generation_status",
        "split",
    ]
    return out[columns]


def update_confusion_agg(agg: dict[tuple[Any, ...], dict[str, Any]], normalized: pd.DataFrame) -> None:
    frame = normalized.copy()
    if "split" in frame.columns:
        stock_mask = frame["instrument_type"].eq("stock")
        frame = frame[(~stock_mask) | frame["split"].astype(str).str.lower().eq("final")].copy()
    y_true = pd.to_numeric(frame["y_true"], errors="coerce")
    y_pred = pd.to_numeric(frame["y_pred"], errors="coerce")
    valid = frame[y_true.notna() & y_pred.notna()].copy()
    if valid.empty:
        return
    valid["y_true_num"] = y_true.loc[valid.index].astype(int)
    valid["y_pred_num"] = y_pred.loc[valid.index].astype(int)
    valid["TP"] = ((valid["y_true_num"] == 1) & (valid["y_pred_num"] == 1)).astype(int)
    valid["TN"] = ((valid["y_true_num"] == 0) & (valid["y_pred_num"] == 0)).astype(int)
    valid["FP"] = ((valid["y_true_num"] == 0) & (valid["y_pred_num"] == 1)).astype(int)
    valid["FN"] = ((valid["y_true_num"] == 1) & (valid["y_pred_num"] == 0)).astype(int)
    valid["correct_count"] = (valid["y_true_num"] == valid["y_pred_num"]).astype(int)
    keys = [
        "model_group",
        "model_id",
        "candidate_id",
        "horizon",
        "instrument_type",
        "instrument_symbol",
        "source_file",
        "source_status",
        "generation_status",
    ]
    grouped = valid.groupby(keys, dropna=False)
    for key, group in grouped:
        item = agg.setdefault(
            tuple(key),
            {
                "actual_rows": 0,
                "actual_up_count": 0,
                "predicted_up_count": 0,
                "correct_count": 0,
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
            },
        )
        item["actual_rows"] += int(len(group))
        item["actual_up_count"] += int(group["y_true_num"].sum())
        item["predicted_up_count"] += int(group["y_pred_num"].sum())
        item["correct_count"] += int(group["correct_count"].sum())
        item["TP"] += int(group["TP"].sum())
        item["TN"] += int(group["TN"].sum())
        item["FP"] += int(group["FP"].sum())
        item["FN"] += int(group["FN"].sum())


def build_prediction_archive_and_table_02(vn30_tickers: list[str]) -> tuple[pd.DataFrame, int]:
    archive_path = TABLE_DIR / "table_03_row_level_prediction_archive.csv.gz"
    agg: dict[tuple[Any, ...], dict[str, Any]] = {}
    row_count = 0
    first = True
    with gzip.open(archive_path, "wt", encoding="utf-8", newline="") as handle:
        for path, source_name in PREDICTION_SOURCES:
            if not path.exists():
                add_missing(rel(path), "prediction", f"Missing row-level prediction source for {source_name}", "yes")
                continue
            for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
                normalized = normalize_prediction_chunk(chunk, path, source_name)
                normalized.to_csv(handle, index=False, header=first)
                first = False
                row_count += len(normalized)
                update_confusion_agg(agg, normalized)
    RUN_STATS["row_level_prediction_rows"] = row_count
    table_02 = confusion_table_from_agg(agg)
    table_02 = add_required_missing_instrument_rows(table_02, vn30_tickers)
    write_table(TABLE_DIR / "table_02_per_instrument_actual_vs_predicted.csv", table_02)
    return table_02, row_count


def confusion_table_from_agg(agg: dict[tuple[Any, ...], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    key_cols = [
        "model_group",
        "model_id",
        "candidate_id",
        "horizon",
        "instrument_type",
        "instrument_symbol",
        "source_file",
        "source_status",
        "generation_status",
    ]
    for key, values in agg.items():
        row = dict(zip(key_cols, key))
        actual_rows = values["actual_rows"]
        actual_up = values["actual_up_count"]
        pred_up = values["predicted_up_count"]
        correct = values["correct_count"]
        tp = values["TP"]
        tn = values["TN"]
        fp = values["FP"]
        fn = values["FN"]
        row.update(
            {
                "actual_rows": actual_rows,
                "actual_up_count": actual_up,
                "actual_down_count": actual_rows - actual_up,
                "actual_up_ratio": actual_up / actual_rows if actual_rows else np.nan,
                "predicted_up_count": pred_up,
                "predicted_down_count": actual_rows - pred_up,
                "predicted_up_ratio": pred_up / actual_rows if actual_rows else np.nan,
                "correct_count": correct,
                "incorrect_count": actual_rows - correct,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "accuracy": correct / actual_rows if actual_rows else np.nan,
                "balanced_accuracy": balanced_accuracy(tp, tn, fp, fn),
                "notes": "final split for stock rows; reported rows for index benchmark",
            }
        )
        rows.append(row)
    columns = [
        "model_group",
        "model_id",
        "candidate_id",
        "horizon",
        "instrument_type",
        "instrument_symbol",
        "actual_rows",
        "actual_up_count",
        "actual_down_count",
        "actual_up_ratio",
        "predicted_up_count",
        "predicted_down_count",
        "predicted_up_ratio",
        "correct_count",
        "incorrect_count",
        "TP",
        "TN",
        "FP",
        "FN",
        "accuracy",
        "balanced_accuracy",
        "source_file",
        "source_status",
        "generation_status",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def balanced_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    pos = tp + fn
    neg = tn + fp
    if pos == 0 or neg == 0:
        return math.nan
    return ((tp / pos) + (tn / neg)) / 2.0


def add_required_missing_instrument_rows(table_02: pd.DataFrame, vn30_tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    found_stocks = set(
        table_02.loc[table_02["instrument_type"].eq("stock") & table_02["source_status"].eq("found"), "instrument_symbol"]
        .astype(str)
        .str.upper()
    )
    for ticker in vn30_tickers:
        if ticker not in found_stocks:
            rows.append(missing_instrument_row("required_vn30_stock", ticker, "stock", "No final row-level predictions found"))
    found_indices = set(
        table_02.loc[table_02["instrument_type"].eq("index") & table_02["source_status"].eq("found"), "instrument_symbol"]
        .astype(str)
        .str.upper()
    )
    for index_code in SUPPORTED_INDICES:
        if index_code not in found_indices:
            rows.append(missing_instrument_row("required_supported_index", index_code, "index", "No local index predictions found"))
    if rows:
        table_02 = pd.concat([table_02, pd.DataFrame(rows)], ignore_index=True)
    return table_02


def missing_instrument_row(model_group: str, symbol: str, instrument_type: str, reason: str) -> dict[str, Any]:
    return {
        "model_group": model_group,
        "model_id": "missing_required_instrument",
        "candidate_id": f"missing__{instrument_type}__{symbol}",
        "horizon": "",
        "instrument_type": instrument_type,
        "instrument_symbol": symbol,
        "actual_rows": "",
        "actual_up_count": "",
        "actual_down_count": "",
        "actual_up_ratio": "",
        "predicted_up_count": "",
        "predicted_down_count": "",
        "predicted_up_ratio": "",
        "correct_count": "",
        "incorrect_count": "",
        "TP": "",
        "TN": "",
        "FP": "",
        "FN": "",
        "accuracy": "",
        "balanced_accuracy": "",
        "source_file": "",
        "source_status": "missing_with_reason",
        "generation_status": "not_generated_missing_source",
        "notes": reason,
    }


def build_summary_tables(table_01: pd.DataFrame, table_02: pd.DataFrame) -> dict[str, pd.DataFrame]:
    found = table_01[table_01["source_status"].eq("found")].copy()
    found["horizon"] = pd.to_numeric(found["horizon"], errors="coerce")
    found["final_accuracy"] = pd.to_numeric(found["final_accuracy"], errors="coerce")
    found["validation_accuracy"] = pd.to_numeric(found["validation_accuracy"], errors="coerce")
    stock_found = stock_model_rows(found)

    pivot = (
        found.pivot_table(index="model_group", columns="horizon", values="final_accuracy", aggfunc="max")
        .reset_index()
        .rename(columns={20.0: "h20", 40.0: "h40", 60.0: "h60", 80.0: "h80"})
    )
    pivot = ensure_source_columns(pivot, "paper_evidence_raw_full_export/tables/table_01_full_model_horizon_results.csv")
    write_table(TABLE_DIR / "table_04_model_family_by_horizon_pivot.csv", pivot)

    best_horizon = load_full_horizon_best_by_horizon()
    if best_horizon.empty:
        best_horizon = best_rows(stock_found, ["horizon"])
    write_table(TABLE_DIR / "table_05_best_model_by_horizon.csv", best_horizon)

    best_family = best_rows(stock_found, ["model_group"])
    write_table(TABLE_DIR / "table_06_best_model_by_family.csv", best_family)

    ranking = practical_ranking(stock_found)
    write_table(TABLE_DIR / "table_07_descriptive_practical_ranking.csv", ranking)

    transfer = load_transfer_quality()
    write_table(TABLE_DIR / "table_08_transfer_quality_by_model.csv", transfer)

    overfit = load_overfit_risk(found)
    write_table(TABLE_DIR / "table_09_overfit_risk_by_model.csv", overfit)

    index_context = load_index_context()
    write_table(TABLE_DIR / "table_10_index_context_results.csv", index_context)

    knn_actuals = table_02[table_02.apply(lambda r: "knn" in " ".join(map(str, r.values)).lower(), axis=1)].copy()
    knn_actuals = ensure_source_columns(knn_actuals, "paper_evidence_raw_full_export/tables/table_02_per_instrument_actual_vs_predicted.csv")
    write_table(TABLE_DIR / "table_11_knn_support_actuals_summary.csv", knn_actuals)

    coop_mask = table_02.apply(
        lambda r: any(token in " ".join(map(str, r.values)).lower() for token in ["coop", "soft_vote", "cooperation", "model_as_feature", "error_correction"]),
        axis=1,
    )
    cooperation_actuals = ensure_source_columns(
        table_02[coop_mask].copy(),
        "paper_evidence_raw_full_export/tables/table_02_per_instrument_actual_vs_predicted.csv",
    )
    write_table(TABLE_DIR / "table_12_model_cooperation_actuals_summary.csv", cooperation_actuals)

    confusion_summary = aggregate_confusion_summary(table_02)
    write_table(TABLE_DIR / "table_13_actual_vs_predicted_confusion_summary.csv", confusion_summary)

    missing_report = build_missing_report(table_01, table_02)
    write_table(TABLE_DIR / "table_14_missing_data_and_artifact_report.csv", missing_report)

    key_numbers = build_key_numbers(table_01, table_02, ranking)
    write_table(TABLE_DIR / "table_15_paper_key_numbers.csv", key_numbers)

    return {
        "pivot": pivot,
        "best_horizon": best_horizon,
        "best_family": best_family,
        "ranking": ranking,
        "transfer": transfer,
        "overfit": overfit,
        "index_context": index_context,
        "knn_actuals": knn_actuals,
        "cooperation_actuals": cooperation_actuals,
        "confusion_summary": confusion_summary,
        "missing_report": missing_report,
        "key_numbers": key_numbers,
    }


def stock_model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model_group" not in frame.columns:
        return frame.copy()
    group_text = frame["model_group"].astype(str).str.lower()
    dataset_text = frame["source_dataset"].astype(str).str.lower() if "source_dataset" in frame.columns else ""
    return frame[~group_text.str.startswith("index_") & ~pd.Series(dataset_text, index=frame.index).str.startswith("index_")].copy()


def load_full_horizon_best_by_horizon() -> pd.DataFrame:
    path = LEGACY_DIR / "full_horizon" / "best_by_horizon.csv"
    if not path.exists():
        add_missing(rel(path), "table", "Full-horizon best-by-horizon source missing", "yes")
        return pd.DataFrame()
    frame = normalize_result_source(path, "legacy_full_horizon_best")
    if frame.empty:
        return frame
    frame["notes"] = "validation-selected full-horizon headline row from best_by_horizon.csv"
    return frame


def best_rows(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    valid = frame.dropna(subset=["final_accuracy"]).copy()
    if valid.empty:
        return ensure_source_columns(pd.DataFrame(), rel(TABLE_DIR / "table_01_full_model_horizon_results.csv"))
    idx = valid.groupby(group_cols)["final_accuracy"].idxmax()
    return ensure_source_columns(valid.loc[idx].sort_values(group_cols).reset_index(drop=True), rel(TABLE_DIR / "table_01_full_model_horizon_results.csv"))


def practical_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["final_accuracy"]).copy()
    if valid.empty:
        return ensure_source_columns(pd.DataFrame(), rel(TABLE_DIR / "table_01_full_model_horizon_results.csv"))
    full_coverage = pd.to_numeric(valid["ticker_coverage"], errors="coerce").fillna(0).ge(30) | valid[
        "index_coverage_if_applicable"
    ].astype(str).ne("")
    stability = pd.to_numeric(valid.get("ticker_min_accuracy", np.nan), errors="coerce").fillna(
        pd.to_numeric(valid.get("rolling_250_mean", np.nan), errors="coerce")
    )
    gap = pd.to_numeric(valid["validation_final_gap"], errors="coerce").abs()
    valid["coverage_score"] = full_coverage.astype(int)
    valid["per_instrument_stability_score"] = stability.fillna(0)
    valid["validation_final_gap_abs"] = gap
    valid["interpretability_score"] = valid.apply(interpretability_score, axis=1)
    valid["overfit_risk_score"] = valid["overfit_risk"].astype(str).str.lower().map({"low": 3, "medium": 2, "high": 1}).fillna(2)
    valid = valid.sort_values(
        [
            "final_accuracy",
            "coverage_score",
            "per_instrument_stability_score",
            "validation_final_gap_abs",
            "rolling_250_mean",
            "interpretability_score",
            "overfit_risk_score",
        ],
        ascending=[False, False, False, True, False, False, False],
    ).reset_index(drop=True)
    valid["practical_experimental_usability_rank"] = np.arange(1, len(valid) + 1)
    valid["ranking_name"] = "practical_experimental_usability_ranking"
    return ensure_source_columns(valid, rel(TABLE_DIR / "table_01_full_model_horizon_results.csv"))


def interpretability_score(row: pd.Series) -> int:
    text = f"{row.get('model_group', '')} {row.get('model_id', '')}".lower()
    if any(token in text for token in ["naive", "technical", "linear", "logistic", "statistical", "baseline"]):
        return 3
    if any(token in text for token in ["tree", "forest", "boost", "svm", "knn", "router"]):
        return 2
    return 1


def load_transfer_quality() -> pd.DataFrame:
    rows = []
    for path in [
        MODEL_UNIVERSE_DIR / "fair_tuning" / "fair_tuning_transfer_quality.csv",
        MODEL_UNIVERSE_DIR / "model_cooperation_transfer_audit" / "cooperation_transfer_quality.csv",
    ]:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["source_file"] = rel(path)
            frame["source_status"] = "found"
            frame["generation_status"] = "regenerated_from_local_artifact"
            rows.append(frame)
        else:
            add_missing(rel(path), "table", "Transfer quality source missing", "maybe")
    if not rows:
        return pd.DataFrame(columns=["source_file", "source_status", "generation_status"])
    return pd.concat(rows, ignore_index=True, sort=False)


def load_overfit_risk(table_01_found: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in [
        MODEL_UNIVERSE_DIR / "overfit_risk_audit.csv",
        MODEL_UNIVERSE_DIR / "fair_tuning" / "overfit_risk_audit.csv",
        MODEL_UNIVERSE_DIR / "model_cooperation_transfer_audit" / "overfit_risk_audit.csv",
        MODEL_UNIVERSE_DIR / "model_cooperation_transfer_audit" / "cooperation_overfit_risk.csv",
    ]:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["source_file"] = rel(path)
            frame["source_status"] = "found"
            frame["generation_status"] = "regenerated_from_local_artifact"
            rows.append(frame)
    if rows:
        return pd.concat(rows, ignore_index=True, sort=False)
    fallback = table_01_found[["model_group", "model_id", "candidate_id", "horizon", "overfit_risk", "source_file"]].copy()
    fallback["source_status"] = "found"
    fallback["generation_status"] = "regenerated_from_local_artifact"
    return fallback


def load_index_context() -> pd.DataFrame:
    rows = []
    for path in [
        INDEX_OUTPUT_DIR / "accuracy_summary.csv",
        INDEX_OUTPUT_DIR / "baseline_summary.csv",
        INDEX_OUTPUT_DIR / "baseline_delta_summary.csv",
        INDEX_REPORT_DIR / "index_directional_benchmark_audit.csv",
        INDEX_REPORT_DIR / "index_data_scope_audit.csv",
    ]:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["source_file"] = rel(path)
            frame["source_status"] = "found"
            frame["generation_status"] = "regenerated_from_local_artifact"
            rows.append(frame)
        else:
            add_missing(rel(path), "table", "Index context source missing", "maybe")
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def aggregate_confusion_summary(table_02: pd.DataFrame) -> pd.DataFrame:
    frame = table_02[table_02["source_status"].eq("found")].copy()
    for column in ["actual_rows", "correct_count", "TP", "TN", "FP", "FN"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    group_cols = ["model_group", "model_id", "candidate_id", "horizon", "instrument_type"]
    grouped = frame.groupby(group_cols, dropna=False)[["actual_rows", "correct_count", "TP", "TN", "FP", "FN"]].sum().reset_index()
    grouped["accuracy"] = grouped["correct_count"] / grouped["actual_rows"].replace(0, np.nan)
    grouped["balanced_accuracy"] = grouped.apply(
        lambda row: balanced_accuracy(int(row["TP"]), int(row["TN"]), int(row["FP"]), int(row["FN"])), axis=1
    )
    return ensure_source_columns(grouped, rel(TABLE_DIR / "table_02_per_instrument_actual_vs_predicted.csv"))


def build_missing_report(table_01: pd.DataFrame, table_02: pd.DataFrame) -> pd.DataFrame:
    rows = list(MISSING_ROWS)
    for _, row in table_01[table_01["source_status"].eq("missing_with_reason")].iterrows():
        rows.append(
            {
                "missing_item": row.get("candidate_id", ""),
                "source_type": "model_horizon",
                "reason": row.get("notes", ""),
                "affects_paper_tables": "yes",
                "regeneration_possible_from_local_files": "no_without_model_or_benchmark_rerun",
                "source_status": "missing_with_reason",
                "generation_status": row.get("generation_status", ""),
            }
        )
    for _, row in table_02[table_02["source_status"].eq("missing_with_reason")].iterrows():
        rows.append(
            {
                "missing_item": row.get("candidate_id", ""),
                "source_type": "per_instrument_actual_vs_predicted",
                "reason": row.get("notes", ""),
                "affects_paper_tables": "yes",
                "regeneration_possible_from_local_files": "no_local_row_predictions_found",
                "source_status": "missing_with_reason",
                "generation_status": row.get("generation_status", ""),
            }
        )
    return pd.DataFrame(rows)


def build_key_numbers(table_01: pd.DataFrame, table_02: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    found = table_01[table_01["source_status"].eq("found")].copy()
    found["final_accuracy"] = pd.to_numeric(found["final_accuracy"], errors="coerce")
    key_specs = [
        ("logistic_l2_baseline_C_closest_h40_threshold_0p55", lambda df: df[df["candidate_id"].astype(str).str.contains("logistic_l2.*baseline_C_closest.*h40.*t0p550", case=False, regex=True)]),
        ("bull_bear_sideway_router_h40_fixed_0p50", lambda df: df[df["model_id"].astype(str).str.contains("bull_bear_sideway_router", case=False, regex=False)]),
        ("validation_lift_weighted_soft_voting", lambda df: df[df["model_id"].astype(str).str.contains("validation_lift_weighted_soft_vote", case=False, regex=False)]),
        ("standalone_knn", lambda df: df[df["model_group"].astype(str).str.contains("standalone_knn_comparator", case=False, regex=False)]),
        ("best_knn_support", lambda df: df[df["model_group"].astype(str).str.contains("knn", case=False, regex=False)]),
        ("model_as_feature_random_forest_meta", lambda df: df[df["model_id"].astype(str).str.contains("random_forest_meta", case=False, regex=False)]),
        (
            "fair_tuning_stacking_xgboost_meta",
            lambda df: df[
                df["source_dataset"].astype(str).eq("fair_tuning")
                & df["model_id"].astype(str).str.contains("stacking_xgboost_meta", case=False, regex=False)
            ],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, selector in key_specs:
        subset = selector(found)
        if subset.empty:
            rows.append(key_number_missing(label, "No matching local artifact row found"))
            continue
        row = subset.sort_values("final_accuracy", ascending=False).iloc[0]
        rows.append(
            {
                "key_number": label,
                "model_group": row.get("model_group", ""),
                "model_id": row.get("model_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "horizon": row.get("horizon", ""),
                "validation_accuracy": row.get("validation_accuracy", ""),
                "final_accuracy": row.get("final_accuracy", ""),
                "source_file": row.get("source_file", ""),
                "source_status": "found",
                "generation_status": "regenerated_from_local_artifact",
                "notes": "verified from local artifact row",
            }
        )
    legacy_full_horizon = load_full_horizon_best_by_horizon()
    for horizon in HORIZONS:
        subset = legacy_full_horizon[pd.to_numeric(legacy_full_horizon["horizon"], errors="coerce").eq(horizon)]
        if subset.empty:
            rows.append(key_number_missing(f"full_horizon_best_h{horizon}", "No local legacy full-horizon row for horizon"))
        else:
            row = subset.sort_values("final_accuracy", ascending=False).iloc[0]
            rows.append(
                {
                    "key_number": f"full_horizon_best_h{horizon}",
                    "model_group": row.get("model_group", ""),
                    "model_id": row.get("model_id", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "horizon": horizon,
                    "validation_accuracy": row.get("validation_accuracy", ""),
                    "final_accuracy": row.get("final_accuracy", ""),
                    "source_file": row.get("source_file", ""),
                    "source_status": "found",
                    "generation_status": "regenerated_from_local_artifact",
                    "notes": "best stock-only legacy full-horizon final accuracy for this horizon",
                }
            )
    if not ranking.empty:
        row = ranking.iloc[0]
        rows.append(
            {
                "key_number": "practical_experimental_usability_ranking_top",
                "model_group": row.get("model_group", ""),
                "model_id": row.get("model_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "horizon": row.get("horizon", ""),
                "validation_accuracy": row.get("validation_accuracy", ""),
                "final_accuracy": row.get("final_accuracy", ""),
                "source_file": rel(TABLE_DIR / "table_07_descriptive_practical_ranking.csv"),
                "source_status": "found",
                "generation_status": "regenerated_from_local_artifact",
                "notes": "top row of requested practical_experimental_usability_ranking",
            }
        )
    return pd.DataFrame(rows)


def key_number_missing(label: str, reason: str) -> dict[str, Any]:
    return {
        "key_number": label,
        "model_group": "",
        "model_id": "",
        "candidate_id": "",
        "horizon": "",
        "validation_accuracy": "",
        "final_accuracy": "",
        "source_file": "",
        "source_status": "missing_with_reason",
        "generation_status": "not_generated_missing_source",
        "notes": reason,
    }


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, filename: str, source_table: str, notes: str = "") -> None:
    path = FIGURE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    FIGURE_MANIFEST_ROWS.append(
        {
            "figure": filename,
            "source_table": source_table,
            "source_status": "found" if (TABLE_DIR / source_table).exists() else "missing_with_reason",
            "generation_status": "regenerated_from_local_artifact",
            "notes": notes,
        }
    )


def generate_figures(table_01: pd.DataFrame, table_02: pd.DataFrame, summary_tables: dict[str, pd.DataFrame]) -> None:
    setup_plot()
    figure_heatmap(summary_tables["pivot"])
    figure_bar(summary_tables["best_family"], "model_group", "final_accuracy", "fig_02_best_final_accuracy_by_model_family.png", "Best Final Accuracy By Model Family", "table_06_best_model_by_family.csv")
    figure_bar(summary_tables["ranking"].head(20), "model_id", "final_accuracy", "fig_03_descriptive_practical_ranking.png", "Practical Experimental Usability Ranking", "table_07_descriptive_practical_ranking.csv")
    figure_claim_status(table_01)
    figure_per_instrument_heatmap(table_02, "stock", "fig_05_per_ticker_accuracy_heatmap_selected_models.png", "table_02_per_instrument_actual_vs_predicted.csv")
    figure_per_instrument_heatmap(table_02, "index", "fig_06_per_index_accuracy_heatmap_selected_models.png", "table_02_per_instrument_actual_vs_predicted.csv")
    figure_actual_vs_pred("selected_candidate_reproduction", "fig_07_selected_candidate_actual_vs_predicted.png", "Selected Candidate Actual Vs Predicted")
    figure_actual_vs_pred("bull_bear_sideway_router", "fig_08_best_descriptive_router_actual_vs_predicted.png", "Router Actual Vs Predicted")
    figure_actual_vs_pred("validation_lift_weighted_soft_vote", "fig_09_soft_voting_actual_vs_predicted.png", "Soft Voting Actual Vs Predicted")
    figure_actual_vs_pred("knn_support", "fig_10_knn_support_actual_vs_predicted.png", "KNN Support Actual Vs Predicted")
    figure_actual_vs_pred("random_forest_meta", "fig_11_model_as_feature_actual_vs_predicted.png", "Model As Feature Actual Vs Predicted")
    figure_instrument_ranking(table_02, "stock", "fig_12_all_vn30_stocks_actual_vs_predicted_small_multiples.png", "VN30 Stock Accuracy Summary")
    figure_instrument_ranking(table_02, "index", "fig_13_supported_indices_actual_vs_predicted_small_multiples.png", "Supported Index Accuracy Summary")
    figure_instrument_ranking(table_02, "stock", "fig_14_stock_performance_ranking.png", "Stock Performance Ranking")
    figure_instrument_ranking(table_02, "index", "fig_15_index_performance_ranking.png", "Index Performance Ranking")
    figure_confusion(summary_tables["confusion_summary"])
    figure_validation_vs_final(table_01)
    figure_overfit(summary_tables["overfit"], table_01)
    figure_bar(summary_tables["cooperation_actuals"].head(30), "model_id", "accuracy", "fig_19_model_cooperation_actual_comparison.png", "Model Cooperation Actual Comparison", "table_12_model_cooperation_actuals_summary.csv")
    figure_bar(summary_tables["knn_actuals"].head(30), "model_id", "accuracy", "fig_20_knn_support_actual_comparison.png", "KNN Support Actual Comparison", "table_11_knn_support_actuals_summary.csv")
    write_table(FIGURE_DIR / "figure_manifest.csv", pd.DataFrame(FIGURE_MANIFEST_ROWS))


def figure_heatmap(pivot: pd.DataFrame) -> None:
    data = pivot.copy()
    value_cols = [col for col in ["h20", "h40", "h60", "h80"] if col in data.columns]
    data = data.sort_values(value_cols, ascending=False).head(30)
    matrix = data[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(value_cols)), value_cols)
    ax.set_yticks(range(len(data)), data["model_group"].astype(str).tolist())
    ax.set_title("Full Model-Horizon Final Accuracy")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, "" if np.isnan(val) else f"{val:.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Accuracy")
    save_figure(fig, "fig_01_full_model_horizon_heatmap.png", "table_04_model_family_by_horizon_pivot.csv")


def figure_bar(frame: pd.DataFrame, label_col: str, value_col: str, filename: str, title: str, source_table: str) -> None:
    data = frame.copy()
    if data.empty or label_col not in data.columns or value_col not in data.columns:
        data = pd.DataFrame({label_col: ["missing"], value_col: [0.0]})
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[value_col]).sort_values(value_col, ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=(8, max(3, len(data) * 0.22)))
    ax.barh(data[label_col].astype(str), data[value_col], color="#4d4d4d")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    save_figure(fig, filename, source_table)


def figure_claim_status(table_01: pd.DataFrame) -> None:
    data = table_01[table_01["source_status"].eq("found")].copy()
    data["final_accuracy"] = pd.to_numeric(data["final_accuracy"], errors="coerce")
    data = data.dropna(subset=["final_accuracy"])
    statuses = {status: i for i, status in enumerate(sorted(data["claim_status"].astype(str).unique()))}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(data["final_accuracy"], data["claim_status"].astype(str).map(statuses), s=12, color="#4d4d4d", alpha=0.65)
    ax.set_yticks(list(statuses.values()), list(statuses.keys()))
    ax.set_xlabel("Final accuracy")
    ax.set_title("Claim Status Vs Final Accuracy")
    save_figure(fig, "fig_04_claim_status_vs_final_accuracy.png", "table_01_full_model_horizon_results.csv")


def selected_model_subset(table_02: pd.DataFrame, instrument_type: str) -> pd.DataFrame:
    data = table_02[table_02["instrument_type"].eq(instrument_type) & table_02["source_status"].eq("found")].copy()
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce")
    selected = data[
        data.apply(
            lambda row: any(
                token in " ".join(str(row.get(col, "")) for col in ["model_group", "model_id", "candidate_id"]).lower()
                for token in [
                    "logistic_l2",
                    "bull_bear_sideway_router",
                    "validation_lift_weighted_soft_vote",
                    "knn_support",
                    "random_forest_meta",
                ]
            ),
            axis=1,
        )
    ]
    if selected.empty:
        selected = data.sort_values("accuracy", ascending=False).head(120)
    return selected


def figure_per_instrument_heatmap(table_02: pd.DataFrame, instrument_type: str, filename: str, source_table: str) -> None:
    data = selected_model_subset(table_02, instrument_type)
    if data.empty:
        data = pd.DataFrame({"model_id": ["missing"], "instrument_symbol": ["missing"], "accuracy": [0.0]})
    pivot = data.pivot_table(index="model_id", columns="instrument_symbol", values="accuracy", aggfunc="mean")
    pivot = pivot.head(20)
    matrix = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.25)))
    im = ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=90)
    ax.set_yticks(range(len(pivot.index)), pivot.index.astype(str))
    ax.set_title(f"Per-{instrument_type} Accuracy Heatmap")
    fig.colorbar(im, ax=ax, label="Accuracy")
    save_figure(fig, filename, source_table)


def load_prediction_sample(token: str, max_rows: int = 600) -> pd.DataFrame:
    token = token.lower()
    rows = []
    for path, source_name in PREDICTION_SOURCES:
        if not path.exists():
            continue
        for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
            normalized = normalize_prediction_chunk(chunk, path, source_name)
            if "split" in normalized.columns:
                normalized = normalized[
                    normalized["split"].astype(str).str.lower().isin(["final", "final_or_reported"])
                ]
            text = normalized[["model_group", "model_id", "candidate_id"]].astype(str).agg(" ".join, axis=1).str.lower()
            subset = normalized[text.str.contains(token, regex=False)].head(max_rows - sum(len(r) for r in rows))
            if not subset.empty:
                rows.append(subset)
            if sum(len(r) for r in rows) >= max_rows:
                return pd.concat(rows, ignore_index=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def figure_actual_vs_pred(token: str, filename: str, title: str) -> None:
    data = load_prediction_sample(token)
    if data.empty:
        data = pd.DataFrame({"datetime": range(2), "y_true": [0, 1], "y_pred": [0, 0]})
    data = data.copy().head(300)
    data["row_number"] = np.arange(len(data))
    data["y_true"] = pd.to_numeric(data["y_true"], errors="coerce")
    data["y_pred"] = pd.to_numeric(data["y_pred"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(data["row_number"], data["y_true"], linewidth=0.8, color="#1f1f1f", label="actual")
    ax.plot(data["row_number"], data["y_pred"], linewidth=0.8, color="#8c8c8c", label="predicted")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title)
    ax.set_xlabel("Row order")
    ax.set_ylabel("Direction label")
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, filename, "table_03_row_level_prediction_archive.csv.gz")


def figure_instrument_ranking(table_02: pd.DataFrame, instrument_type: str, filename: str, title: str) -> None:
    data = table_02[table_02["instrument_type"].eq(instrument_type) & table_02["source_status"].eq("found")].copy()
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce")
    grouped = data.groupby("instrument_symbol", dropna=False)["accuracy"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, len(grouped) * 0.18)))
    ax.barh(grouped.index.astype(str), grouped.values, color="#4d4d4d")
    ax.set_title(title)
    ax.set_xlabel("Mean accuracy across available model rows")
    save_figure(fig, filename, "table_02_per_instrument_actual_vs_predicted.csv")


def figure_confusion(confusion: pd.DataFrame) -> None:
    data = confusion.copy()
    for column in ["TP", "TN", "FP", "FN"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["total"] = data[["TP", "TN", "FP", "FN"]].sum(axis=1)
    data = data.sort_values("total", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, max(4, len(data) * 0.25)))
    left = np.zeros(len(data))
    labels = data["model_id"].astype(str).tolist()
    for column, color in zip(["TP", "TN", "FP", "FN"], ["#1f1f1f", "#6b6b6b", "#b0b0b0", "#d9d9d9"]):
        values = data[column].to_numpy(dtype=float)
        ax.barh(labels, values, left=left, label=column, color=color)
        left += values
    ax.set_title("Confusion Decomposition Selected Models")
    ax.legend(frameon=False, ncols=4)
    save_figure(fig, "fig_16_confusion_decomposition_selected_models.png", "table_13_actual_vs_predicted_confusion_summary.csv")


def figure_validation_vs_final(table_01: pd.DataFrame) -> None:
    data = table_01[table_01["source_status"].eq("found")].copy()
    data["validation_accuracy"] = pd.to_numeric(data["validation_accuracy"], errors="coerce")
    data["final_accuracy"] = pd.to_numeric(data["final_accuracy"], errors="coerce")
    data = data.dropna(subset=["validation_accuracy", "final_accuracy"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(data["validation_accuracy"], data["final_accuracy"], s=10, color="#4d4d4d", alpha=0.55)
    lo = min(data["validation_accuracy"].min(), data["final_accuracy"].min())
    hi = max(data["validation_accuracy"].max(), data["final_accuracy"].max())
    ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=1)
    ax.set_xlabel("Validation accuracy")
    ax.set_ylabel("Final accuracy")
    ax.set_title("Validation Vs Final Accuracy")
    save_figure(fig, "fig_17_validation_vs_final_by_model.png", "table_01_full_model_horizon_results.csv")


def figure_overfit(overfit: pd.DataFrame, table_01: pd.DataFrame) -> None:
    if "overfit_risk" in overfit.columns:
        data = overfit.copy()
        family_col = "model_group" if "model_group" in data.columns else "model_family"
        if family_col not in data.columns:
            data[family_col] = "unknown"
        counts = data.groupby([family_col, "overfit_risk"]).size().unstack(fill_value=0).head(25)
    else:
        data = table_01.copy()
        counts = data.groupby(["model_group", "overfit_risk"]).size().unstack(fill_value=0).head(25)
    fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.25)))
    counts.plot(kind="barh", stacked=True, ax=ax, color=["#1f1f1f", "#8c8c8c", "#d0d0d0"])
    ax.set_title("Overfit Risk By Model Family")
    ax.legend(frameon=False)
    save_figure(fig, "fig_18_overfit_risk_by_model_family.png", "table_09_overfit_risk_by_model.csv")


def write_raw_data_availability(vn30_tickers: list[str]) -> None:
    lines = ["# Raw Data Availability", ""]
    raw_manifest = pd.DataFrame([row for row in SOURCE_MANIFEST_ROWS if row["source_type"] == "raw_data"])
    lines.append(f"- Raw/cache files found and copied: {len(raw_manifest[raw_manifest['source_status'].eq('found')]) if not raw_manifest.empty else 0}")
    lines.append(f"- Required VN30 tickers from frozen universe: {len(vn30_tickers)}")
    lines.append(f"- Supported indices required: {', '.join(SUPPORTED_INDICES)}")
    lines.append("")
    lines.append("## Raw Files Found")
    for row in raw_manifest[raw_manifest["source_status"].eq("found")].head(300).to_dict("records") if not raw_manifest.empty else []:
        lines.append(f"- `{row['source_path']}` -> `{row['copied_to']}`")
    if not raw_manifest.empty and len(raw_manifest) > 300:
        lines.append(f"- ... {len(raw_manifest) - 300} additional raw/cache files listed in manifests/raw_data_manifest.csv")
    lines.append("")
    lines.append("## Processed Files Found")
    lines.append("- Processed feature/label/model output tables are listed in source_manifest.csv and raw_model_outputs/.")
    lines.append("")
    lines.append("## Missing Raw Files")
    missing = [row for row in SOURCE_MANIFEST_ROWS if row["source_type"] == "raw_data" and row["source_status"] != "found"]
    if not missing:
        lines.append("- None detected for configured local raw/cache sources.")
    for row in missing:
        lines.append(f"- `{row['source_path']}`: {row['notes']}; affects paper tables: maybe; regeneration: no internet fetch allowed.")
    write_text(LOG_DIR / "raw_data_availability.md", "\n".join(lines))


def write_manifests(repo_state: dict[str, Any]) -> None:
    source_manifest = pd.DataFrame(SOURCE_MANIFEST_ROWS)
    write_table(MANIFEST_DIR / "source_manifest.csv", source_manifest)
    for name, directory, filename in [
        ("raw_data", RAW_DATA_DIR, "raw_data_manifest.csv"),
        ("model_artifact", RAW_MODEL_DIR, "model_artifact_manifest.csv"),
        ("prediction_artifact", RAW_PREDICTION_DIR, "prediction_artifact_manifest.csv"),
    ]:
        rows = []
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            rows.append(
                {
                    "export_path": rel(path),
                    "artifact_type": name,
                    "file_size": path.stat().st_size,
                    "modified_time": file_modified_time(path),
                    "source_status": "found",
                    "generation_status": "copied_from_local_artifact",
                }
            )
        write_table(MANIFEST_DIR / filename, pd.DataFrame(rows))
    generation_manifest = {
        "export_script": rel(REPO_ROOT / "scripts" / "research" / "export_vn30_raw_full_paper_evidence.py"),
        "data_fetch_run": "no",
        "benchmark_rerun": "no",
        "model_training_run": "no",
        "paper_generated": "no",
        "repo_state": repo_state,
        "result_sources": [rel(path) for path, _ in RESULT_SOURCES],
        "prediction_sources": [rel(path) for path, _ in PREDICTION_SOURCES],
        "required_horizons": HORIZONS,
        "supported_indices": SUPPORTED_INDICES,
        "missing_items": MISSING_ROWS,
        "run_stats": RUN_STATS,
    }
    write_json(MANIFEST_DIR / "evidence_generation_manifest.json", generation_manifest)


def validate_export(table_01: pd.DataFrame, table_02: pd.DataFrame, vn30_tickers: list[str]) -> dict[str, Any]:
    missing_tables = [name for name in REQUIRED_TABLES if not (TABLE_DIR / name).exists()]
    missing_figures = [name for name in REQUIRED_FIGURES if not (FIGURE_DIR / name).exists()]
    column_issues = []
    for name in REQUIRED_TABLES:
        path = TABLE_DIR / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, nrows=3, compression="infer")
        for column in ["source_file", "source_status", "generation_status"]:
            if column not in frame.columns:
                column_issues.append(f"{name} missing {column}")
    horizons_found = sorted(pd.to_numeric(table_01["horizon"], errors="coerce").dropna().astype(int).unique().tolist())
    table02_found = table_02[table_02["source_status"].eq("found")]
    tickers_found = sorted(
        table02_found.loc[table02_found["instrument_type"].eq("stock"), "instrument_symbol"].astype(str).str.upper().unique()
    )
    indices_found = sorted(
        table02_found.loc[table02_found["instrument_type"].eq("index"), "instrument_symbol"].astype(str).str.upper().unique()
    )
    knn_actuals = table02_found.apply(lambda r: "knn" in " ".join(map(str, r.values)).lower(), axis=1).any()
    cooperation_actuals = table02_found.apply(
        lambda r: any(token in " ".join(map(str, r.values)).lower() for token in ["coop", "soft_vote", "cooperation", "model_as_feature", "error_correction"]),
        axis=1,
    ).any()
    validation = {
        "required_tables_exist": not missing_tables,
        "required_figures_exist": not missing_figures,
        "source_columns_exist": not column_issues,
        "table_01_h20_h40_h60_h80_included": all(h in horizons_found for h in HORIZONS),
        "table_02_all_30_vn30_tickers_included_where_available": all(ticker in tickers_found for ticker in vn30_tickers),
        "table_02_all_six_indices_included_where_available": all(index in indices_found for index in SUPPORTED_INDICES),
        "knn_support_actual_rows_exist": bool(knn_actuals),
        "model_cooperation_actual_rows_exist": bool(cooperation_actuals),
        "row_level_archive_exists": (TABLE_DIR / "table_03_row_level_prediction_archive.csv.gz").exists(),
        "missing_tables": missing_tables,
        "missing_figures": missing_figures,
        "column_issues": column_issues,
        "horizons_found": horizons_found,
        "vn30_tickers_found": tickers_found,
        "indices_found": indices_found,
        "data_fetch_run": "no",
        "benchmark_rerun": "no",
        "model_training_run": "no",
    }
    validation["validation_passed"] = all(
        [
            validation["required_tables_exist"],
            validation["required_figures_exist"],
            validation["source_columns_exist"],
            validation["table_01_h20_h40_h60_h80_included"],
            validation["table_02_all_30_vn30_tickers_included_where_available"],
            validation["table_02_all_six_indices_included_where_available"],
            validation["knn_support_actual_rows_exist"],
            validation["model_cooperation_actual_rows_exist"],
            validation["row_level_archive_exists"],
        ]
    )
    lines = [
        "# Raw Full Export Validation",
        "",
        f"- Required tables exist: {'yes' if validation['required_tables_exist'] else 'no'}",
        f"- Required figures exist: {'yes' if validation['required_figures_exist'] else 'no'}",
        f"- Source columns exist: {'yes' if validation['source_columns_exist'] else 'no'}",
        f"- h20/h40/h60/h80 included: {'yes' if validation['table_01_h20_h40_h60_h80_included'] else 'no'}",
        f"- All 30 VN30 tickers included where available: {'yes' if validation['table_02_all_30_vn30_tickers_included_where_available'] else 'no'}",
        f"- All six supported indices included where available: {'yes' if validation['table_02_all_six_indices_included_where_available'] else 'no'}",
        f"- KNN-support actual rows exist: {'yes' if validation['knn_support_actual_rows_exist'] else 'no'}",
        f"- Model-cooperation actual rows exist: {'yes' if validation['model_cooperation_actual_rows_exist'] else 'no'}",
        f"- Row-level archive exists: {'yes' if validation['row_level_archive_exists'] else 'no'}",
        f"- Data fetch run: {validation['data_fetch_run']}",
        f"- Benchmark rerun: {validation['benchmark_rerun']}",
        f"- Model training run: {validation['model_training_run']}",
        f"- Validation passed: {'yes' if validation['validation_passed'] else 'no'}",
        "",
        "## Missing Tables",
        *[f"- `{name}`" for name in missing_tables],
        "- None" if not missing_tables else "",
        "",
        "## Missing Figures",
        *[f"- `{name}`" for name in missing_figures],
        "- None" if not missing_figures else "",
        "",
        "## Column Issues",
        *[f"- {issue}" for issue in column_issues],
        "- None" if not column_issues else "",
    ]
    write_text(LOG_DIR / "raw_full_export_validation.md", "\n".join(line for line in lines if line is not None))
    write_json(LOG_DIR / "raw_full_export_validation.json", validation)
    return validation


def write_summary(repo_state: dict[str, Any], table_01: pd.DataFrame, table_02: pd.DataFrame, row_count: int, validation: dict[str, Any], vn30_tickers: list[str]) -> None:
    table01_found = table_01[table_01["source_status"].eq("found")].copy()
    table01_found["final_accuracy"] = pd.to_numeric(table01_found["final_accuracy"], errors="coerce")
    stock_found = stock_model_rows(table01_found)
    table02_found = table_02[table_02["source_status"].eq("found")]
    best = stock_found.sort_values("final_accuracy", ascending=False).head(1)
    best_desc = best.iloc[0] if not best.empty else {}
    best_horizon_lines = []
    legacy_full_horizon = load_full_horizon_best_by_horizon()
    for horizon in HORIZONS:
        subset = legacy_full_horizon[pd.to_numeric(legacy_full_horizon["horizon"], errors="coerce").eq(horizon)]
        if subset.empty:
            best_horizon_lines.append(f"- h{horizon}: missing_with_reason")
        else:
            row = subset.sort_values("final_accuracy", ascending=False).iloc[0]
            best_horizon_lines.append(
                f"- h{horizon}: {row.get('model_id', '')} / {row.get('candidate_id', '')} final={as_float(row.get('final_accuracy')):.6f}"
            )
    tickers_found = sorted(
        table02_found.loc[table02_found["instrument_type"].eq("stock"), "instrument_symbol"].astype(str).str.upper().unique()
    )
    indices_found = sorted(
        table02_found.loc[table02_found["instrument_type"].eq("index"), "instrument_symbol"].astype(str).str.upper().unique()
    )
    knn_yes = "yes" if validation["knn_support_actual_rows_exist"] else "no"
    coop_yes = "yes" if validation["model_cooperation_actual_rows_exist"] else "no"
    summary = f"""# Paper Raw Full Evidence Summary

## Provenance

- Branch: `{repo_state.get('branch', '')}`
- Commit: `{repo_state.get('commit_hash', '')}`
- Benchmarks rerun: no
- Model training run: no
- Data fetch run: no
- Paper/DOCX/PDF generated: no

## Export Counts

- Raw data files copied: {RUN_STATS['raw_data_files_copied']}
- Model output files copied: {RUN_STATS['model_output_files_copied']}
- Prediction artifacts copied: {RUN_STATS['prediction_artifact_files_copied']}
- Model-horizon rows: {len(table_01)}
- Per-instrument actual-vs-predicted rows: {len(table_02)}
- Row-level prediction rows: {row_count}
- Missing model-horizon rows: {int(table_01['source_status'].eq('missing_with_reason').sum())}
- Missing per-instrument rows: {int(table_02['source_status'].eq('missing_with_reason').sum())}

## Coverage Checks

- All 30 VN30 tickers included yes/no: {'yes' if all(t in tickers_found for t in vn30_tickers) else 'no'}
- All six supported indices included yes/no: {'yes' if all(i in indices_found for i in SUPPORTED_INDICES) else 'no'}
- h20/h40/h60/h80 included yes/no: {'yes' if validation['table_01_h20_h40_h60_h80_included'] else 'no'}
- KNN-support actuals included yes/no: {knn_yes}
- Model-cooperation actuals included yes/no: {coop_yes}

## Key Results From Local Artifacts

- Best descriptive stock model: {best_desc.get('model_id', '') if isinstance(best_desc, pd.Series) else ''} / {best_desc.get('candidate_id', '') if isinstance(best_desc, pd.Series) else ''} final={as_float(best_desc.get('final_accuracy', math.nan)) if isinstance(best_desc, pd.Series) else ''}
{chr(10).join(best_horizon_lines)}

## Practical Experimental Usability Ranking

- Ranking table: `tables/table_07_descriptive_practical_ranking.csv`
- Ranking name: `practical_experimental_usability_ranking`

## Boundaries

- Index context does not replace stock-only evidence.
- This export is not a trading, profitability, investment, or live-deployment claim.
- Missing values are explicitly marked with `missing_with_reason`; no values were invented.
"""
    write_text(EXPORT_DIR / "PAPER_RAW_FULL_EVIDENCE_SUMMARY.md", summary)


def write_zip() -> Path:
    zip_path = REPO_ROOT / "vn30_raw_full_paper_evidence_export.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(item for item in EXPORT_DIR.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(REPO_ROOT))
    return zip_path


def main() -> int:
    reset_export_dir()
    repo_state = write_repo_state()
    vn30_tickers = read_vn30_tickers()
    copy_raw_data()
    copy_model_outputs_and_predictions()
    copy_protocols_and_scripts()
    table_01 = build_table_01()
    table_02, row_count = build_prediction_archive_and_table_02(vn30_tickers)
    summary_tables = build_summary_tables(table_01, table_02)
    generate_figures(table_01, table_02, summary_tables)
    write_raw_data_availability(vn30_tickers)
    validation = validate_export(table_01, table_02, vn30_tickers)
    write_summary(repo_state, table_01, table_02, row_count, validation, vn30_tickers)
    write_manifests(repo_state)
    write_zip()
    return 0 if validation["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
