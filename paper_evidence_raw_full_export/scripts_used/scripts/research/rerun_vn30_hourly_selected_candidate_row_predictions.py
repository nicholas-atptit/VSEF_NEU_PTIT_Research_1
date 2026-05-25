"""Rerun the fixed VN30 hourly selected L2 Logistic h=40 candidate.

This script exists only to save row-level final-window predictions for the
already selected Track A canonical-like candidate. It does not perform model
selection, hyperparameter tuning, confidence abstention, ticker filtering, or
top-k ranking.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_track_a_target62_validation_safe import (  # noqa: E402
    BASELINE_LOGISTIC_H40,
    EVAL_START,
    OUTPUT_DIR as SELECTED_OUTPUT_DIR,
    REPORT_DIR as SELECTED_REPORT_DIR,
    THRESHOLDS,
    TRAIN_END,
    VAL_END,
    VAL_START,
    candidate_model,
    split_indices,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    INDEX_CACHE_DIR,
    LOCKED_RF_H60,
    REPO_ROOT,
    STOCK_CACHE_DIR,
    active_stock_tickers,
    add_absolute_labels,
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    rel,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_selected_l2_logistic_h40_row_predictions"
REQUIRED_OUTPUTS = [
    OUTPUT_DIR / "row_predictions.csv",
    OUTPUT_DIR / "selected_candidate_reproduction_summary.csv",
    OUTPUT_DIR / "selected_candidate_reproduction_manifest.json",
    OUTPUT_DIR / "selected_candidate_reproduction_log.md",
]

SOURCE_RUN_ID = "vn30_hourly_target62_selected_l2_logistic_h40_row_reproduction_v1"
FIXED_MODEL = "l2_logistic"
FIXED_MODEL_LABEL = "L2 Logistic"
FIXED_HORIZON = 40
FIXED_FEATURE_SET = "feature_set_C_closest"
FIXED_THRESHOLD = 0.50
KNOWN_FINAL_ACCURACY = 0.6151202749140894
KNOWN_FINAL_ROWS = 4074
REPRODUCTION_TOLERANCE_PP = 0.10

ROW_PREDICTION_COLUMNS = [
    "datetime",
    "ticker",
    "horizon",
    "model",
    "feature_set",
    "threshold",
    "split",
    "y_true",
    "y_pred",
    "y_score_or_probability",
    "correct",
    "source_run_id",
    "reproduction_flag",
]

SUMMARY_COLUMNS = [
    "existing_final_accuracy",
    "reproduced_final_accuracy",
    "absolute_difference_pp",
    "existing_final_rows",
    "reproduced_final_rows",
    "rows_difference",
    "existing_majority_baseline",
    "reproduced_majority_baseline",
    "existing_lift_vs_majority",
    "reproduced_lift_vs_majority",
    "reproduction_status",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def accuracy(y_true: pd.Series, prediction: np.ndarray | pd.Series) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((y_true.astype(int).to_numpy() == np.asarray(prediction).astype(int)).mean())


def majority_prediction(train_y: pd.Series) -> int:
    return int(float(train_y.mean()) >= 0.5)


def selected_candidate_path() -> Path:
    output_path = SELECTED_OUTPUT_DIR / "selected_candidate_summary.csv"
    report_path = SELECTED_REPORT_DIR / "selected_candidate_summary.csv"
    if output_path.exists():
        return output_path
    if report_path.exists():
        return report_path
    raise FileNotFoundError("selected_candidate_summary.csv not found in target62 validation-safe artifacts")


def run_config_path() -> Path:
    output_path = SELECTED_OUTPUT_DIR / "run_config.json"
    report_path = SELECTED_REPORT_DIR / "run_config.json"
    if output_path.exists():
        return output_path
    if report_path.exists():
        return report_path
    raise FileNotFoundError("run_config.json not found in target62 validation-safe artifacts")


def load_selected_candidate() -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    selected_path = selected_candidate_path()
    selected_frame = pd.read_csv(selected_path)
    if selected_frame.empty:
        raise ValueError(f"selected candidate file is empty: {rel(selected_path)}")
    selected = selected_frame.iloc[0].to_dict()

    config_path = run_config_path()
    config = read_json(config_path)

    selected_model = str(selected.get("model", ""))
    selected_horizon = int(as_float(selected.get("horizon")))
    selected_feature_set = str(selected.get("feature_set", ""))
    selected_threshold = as_float(selected.get("threshold"))

    if selected_model != FIXED_MODEL:
        raise ValueError(f"selected model mismatch: expected {FIXED_MODEL}, found {selected_model}")
    if selected_horizon != FIXED_HORIZON:
        raise ValueError(f"selected horizon mismatch: expected {FIXED_HORIZON}, found {selected_horizon}")
    if selected_feature_set != FIXED_FEATURE_SET:
        raise ValueError(f"selected feature set mismatch: expected {FIXED_FEATURE_SET}, found {selected_feature_set}")
    if abs(selected_threshold - FIXED_THRESHOLD) > 1e-12:
        raise ValueError(f"selected threshold mismatch: expected {FIXED_THRESHOLD}, found {selected_threshold}")

    if bool(config.get("confidence_abstention")):
        raise ValueError("run_config has confidence_abstention=true")
    if bool(config.get("ticker_subset")):
        raise ValueError("run_config has ticker_subset=true")
    if bool(config.get("topk")):
        raise ValueError("run_config has topk=true")
    if bool(config.get("data_fetch")):
        raise ValueError("run_config has data_fetch=true")
    if FIXED_THRESHOLD not in [float(x) for x in THRESHOLDS]:
        raise ValueError("fixed threshold is not part of the original threshold set")

    return selected, selected_path, config, config_path


def ensure_required_inputs(tickers: list[str]) -> dict[str, Any]:
    if not STOCK_CACHE_DIR.exists():
        raise FileNotFoundError(f"stock cache directory missing: {rel(STOCK_CACHE_DIR)}")
    if not INDEX_CACHE_DIR.exists():
        raise FileNotFoundError(f"index cache directory missing: {rel(INDEX_CACHE_DIR)}")

    missing_stock_files = [ticker for ticker in tickers if not (STOCK_CACHE_DIR / f"{ticker}.csv").exists()]
    missing_index_files = [code for code in ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"] if not (INDEX_CACHE_DIR / f"{code}.csv").exists()]
    if missing_stock_files:
        raise FileNotFoundError(f"missing stock cache files for: {', '.join(missing_stock_files)}")
    if missing_index_files:
        raise FileNotFoundError(f"missing index cache files for: {', '.join(missing_index_files)}")

    return {
        "stock_cache_dir": rel(STOCK_CACHE_DIR),
        "index_cache_dir": rel(INDEX_CACHE_DIR),
        "ticker_count": len(tickers),
        "missing_stock_files": missing_stock_files,
        "missing_index_files": missing_index_files,
    }


def ensure_no_overwrite() -> None:
    existing = [path for path in REQUIRED_OUTPUTS if path.exists()]
    if existing:
        formatted = ", ".join(rel(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing reproduction evidence: {formatted}")


def build_fixed_candidate_predictions() -> tuple[pd.DataFrame, dict[str, Any]]:
    tickers = active_stock_tickers()
    input_manifest = ensure_required_inputs(tickers)

    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    if stock_df.empty:
        raise ValueError("stock data loaded empty")
    if not index_data:
        raise ValueError("index data loaded empty")

    features, feature_cols = build_feature_set_c(stock_df, index_data)
    if len(feature_cols) == 0:
        raise ValueError("feature_set_C_closest produced no feature columns")

    labels = add_absolute_labels(features, FIXED_HORIZON)
    idx = split_indices(features, labels)
    train_y = labels.reindex(idx["train"]).astype(int)
    val_y = labels.reindex(idx["validation"]).astype(int)
    final_y = labels.reindex(idx["final"]).astype(int)

    if train_y.empty or val_y.empty or final_y.empty:
        raise ValueError("one or more required splits are empty")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", candidate_model(FIXED_MODEL)),
        ]
    )
    model.fit(features.reindex(idx["train"])[feature_cols], train_y)

    val_prob = model.predict_proba(features.reindex(idx["validation"])[feature_cols])[:, 1]
    final_prob = model.predict_proba(features.reindex(idx["final"])[feature_cols])[:, 1]
    val_pred = (val_prob >= FIXED_THRESHOLD).astype(int)
    final_pred = (final_prob >= FIXED_THRESHOLD).astype(int)

    majority = majority_prediction(train_y)
    val_base = accuracy(val_y, np.full(len(val_y), majority))
    final_base = accuracy(final_y, np.full(len(final_y), majority))
    val_acc = accuracy(val_y, val_pred)
    final_acc = accuracy(final_y, final_pred)

    rows = features.reindex(idx["final"])[["datetime", "ticker"]].copy()
    rows["horizon"] = FIXED_HORIZON
    rows["model"] = FIXED_MODEL_LABEL
    rows["feature_set"] = FIXED_FEATURE_SET
    rows["threshold"] = FIXED_THRESHOLD
    rows["split"] = "final"
    rows["y_true"] = final_y.astype(int).to_numpy()
    rows["y_pred"] = final_pred.astype(int)
    rows["y_score_or_probability"] = final_prob.astype(float)
    rows["correct"] = (rows["y_true"].astype(int) == rows["y_pred"].astype(int)).astype(int)
    rows["source_run_id"] = SOURCE_RUN_ID

    metrics = {
        **input_manifest,
        "model": FIXED_MODEL,
        "model_label": FIXED_MODEL_LABEL,
        "horizon": FIXED_HORIZON,
        "feature_set": FIXED_FEATURE_SET,
        "threshold": FIXED_THRESHOLD,
        "feature_count": len(feature_cols),
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "final_rows": int(len(final_y)),
        "validation_accuracy": val_acc,
        "final_accuracy": final_acc,
        "validation_majority_baseline": val_base,
        "final_majority_baseline": final_base,
        "validation_lift_vs_majority": val_acc - val_base,
        "final_lift_vs_majority": final_acc - final_base,
        "majority_prediction": majority,
        "split_train_end": str(TRAIN_END),
        "split_validation_start": str(VAL_START),
        "split_validation_end": str(VAL_END),
        "split_final_start": str(EVAL_START),
    }
    return rows, metrics


def reproduction_summary(selected: dict[str, Any], metrics: dict[str, Any]) -> tuple[dict[str, Any], str]:
    existing_final_accuracy = as_float(selected.get("final_accuracy"))
    existing_final_rows = int(as_float(selected.get("final_rows")))
    existing_majority_baseline = as_float(selected.get("final_baseline_accuracy"))
    existing_lift_vs_majority = as_float(selected.get("final_delta_vs_baseline"))

    reproduced_final_accuracy = as_float(metrics.get("final_accuracy"))
    reproduced_final_rows = int(as_float(metrics.get("final_rows")))
    reproduced_majority_baseline = as_float(metrics.get("final_majority_baseline"))
    reproduced_lift_vs_majority = as_float(metrics.get("final_lift_vs_majority"))

    absolute_difference_pp = abs(reproduced_final_accuracy - existing_final_accuracy) * 100.0
    rows_difference = reproduced_final_rows - existing_final_rows

    status = "passed"
    reasons: list[str] = []
    if not math.isfinite(absolute_difference_pp) or absolute_difference_pp > REPRODUCTION_TOLERANCE_PP:
        status = "failed_or_mismatch"
        reasons.append(f"accuracy difference {absolute_difference_pp:.6f}pp exceeds {REPRODUCTION_TOLERANCE_PP:.2f}pp tolerance")
    if rows_difference != 0:
        status = "failed_or_mismatch"
        reasons.append(f"row difference {rows_difference} is material")
    if abs(existing_final_accuracy - KNOWN_FINAL_ACCURACY) * 100.0 > REPRODUCTION_TOLERANCE_PP:
        status = "failed_or_mismatch"
        reasons.append("selected artifact does not match known 61.51% final accuracy")
    if existing_final_rows != KNOWN_FINAL_ROWS:
        status = "failed_or_mismatch"
        reasons.append("selected artifact does not match known 4,074 final rows")

    row = {
        "existing_final_accuracy": existing_final_accuracy,
        "reproduced_final_accuracy": reproduced_final_accuracy,
        "absolute_difference_pp": absolute_difference_pp,
        "existing_final_rows": existing_final_rows,
        "reproduced_final_rows": reproduced_final_rows,
        "rows_difference": rows_difference,
        "existing_majority_baseline": existing_majority_baseline,
        "reproduced_majority_baseline": reproduced_majority_baseline,
        "existing_lift_vs_majority": existing_lift_vs_majority,
        "reproduced_lift_vs_majority": reproduced_lift_vs_majority,
        "reproduction_status": status,
    }
    return row, "; ".join(reasons) if reasons else "within tolerance"


def pct(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:+.4f}pp"


def write_outputs(
    rows: pd.DataFrame,
    summary: dict[str, Any],
    selected: dict[str, Any],
    selected_path: Path,
    config: dict[str, Any],
    config_path: Path,
    metrics: dict[str, Any],
    status_reason: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reproduction_status = str(summary["reproduction_status"])
    output_rows = rows.copy()
    output_rows["reproduction_flag"] = reproduction_status
    output_rows = output_rows.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    output_rows[ROW_PREDICTION_COLUMNS].to_csv(OUTPUT_DIR / "row_predictions.csv", index=False)
    pd.DataFrame([summary], columns=SUMMARY_COLUMNS).to_csv(OUTPUT_DIR / "selected_candidate_reproduction_summary.csv", index=False)

    manifest = {
        "source_run_id": SOURCE_RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "save row-level predictions for already selected candidate only",
        "not_model_selection": True,
        "not_new_tuning": True,
        "broad_benchmark_run": False,
        "data_fetch": False,
        "confidence_abstention": False,
        "ticker_subset": False,
        "topk_ranking": False,
        "paper_docx_edited": False,
        "selected_candidate_changed": False,
        "selected_candidate_source": rel(selected_path),
        "run_config_source": rel(config_path),
        "selected_candidate": {
            "model": selected.get("model"),
            "horizon": int(as_float(selected.get("horizon"))),
            "feature_set": selected.get("feature_set"),
            "threshold": as_float(selected.get("threshold")),
            "existing_final_accuracy": as_float(selected.get("final_accuracy")),
            "existing_final_rows": int(as_float(selected.get("final_rows"))),
        },
        "fixed_rerun": {
            "model": FIXED_MODEL,
            "model_label": FIXED_MODEL_LABEL,
            "horizon": FIXED_HORIZON,
            "feature_set": FIXED_FEATURE_SET,
            "threshold": FIXED_THRESHOLD,
            "same_split_constants": True,
            "same_feature_builder": "scripts.research.vn30_hourly_dual_track_common.build_feature_set_c",
            "same_label_builder": "scripts.research.vn30_hourly_dual_track_common.add_absolute_labels",
            "same_model_factory": "scripts.research.run_vn30_hourly_track_a_target62_validation_safe.candidate_model",
        },
        "original_run_config_selection_rule": config.get("selection_rule", {}),
        "baselines": {
            "baseline_logistic_h40": BASELINE_LOGISTIC_H40,
            "historical_rf_h60": LOCKED_RF_H60,
        },
        "metrics": metrics,
        "summary": summary,
        "status_reason": status_reason,
        "outputs": [rel(path) for path in REQUIRED_OUTPUTS],
    }
    write_json(OUTPUT_DIR / "selected_candidate_reproduction_manifest.json", manifest)

    log_lines = [
        "# Selected VN30 Hourly L2 Logistic h40 Row-Level Reproduction Log",
        "",
        "- Purpose: reproduce the already selected candidate only to save row-level predictions.",
        "- New model selection: no.",
        "- New tuning: no.",
        "- Broad benchmark sweep: no.",
        "- Data fetch: no.",
        "- Confidence abstention: no.",
        "- Ticker subset: no.",
        "- Top-k/ranking: no.",
        "- DOCX/paper rewrite: no.",
        "- Trading/profitability/live-deployment claim: no.",
        f"- Source run id: `{SOURCE_RUN_ID}`.",
        f"- Selected candidate source: `{rel(selected_path)}`.",
        f"- Run config source: `{rel(config_path)}`.",
        f"- Candidate: `{FIXED_MODEL}` h={FIXED_HORIZON} `{FIXED_FEATURE_SET}` threshold={FIXED_THRESHOLD:.2f}.",
        f"- Train rows: {metrics['train_rows']}.",
        f"- Validation rows: {metrics['validation_rows']}.",
        f"- Final rows: {metrics['final_rows']}.",
        f"- Existing final accuracy: {pct(summary['existing_final_accuracy'])}.",
        f"- Reproduced final accuracy: {pct(summary['reproduced_final_accuracy'])}.",
        f"- Absolute difference: {summary['absolute_difference_pp']:.6f} percentage points.",
        f"- Existing majority baseline: {pct(summary['existing_majority_baseline'])}.",
        f"- Reproduced majority baseline: {pct(summary['reproduced_majority_baseline'])}.",
        f"- Existing lift vs majority: {pp(summary['existing_lift_vs_majority'] * 100.0)}.",
        f"- Reproduced lift vs majority: {pp(summary['reproduced_lift_vs_majority'] * 100.0)}.",
        f"- Reproduction status: `{reproduction_status}`.",
        f"- Status reason: {status_reason}.",
        "",
    ]
    (OUTPUT_DIR / "selected_candidate_reproduction_log.md").write_text("\n".join(log_lines), encoding="utf-8")


def main() -> int:
    ensure_no_overwrite()
    selected, selected_path, config, config_path = load_selected_candidate()
    rows, metrics = build_fixed_candidate_predictions()
    summary, status_reason = reproduction_summary(selected, metrics)
    write_outputs(rows, summary, selected, selected_path, config, config_path, metrics, status_reason)
    print(
        "selected_candidate_reproduction_status="
        f"{summary['reproduction_status']} final_accuracy={summary['reproduced_final_accuracy']:.12f} "
        f"rows={summary['reproduced_final_rows']} diff_pp={summary['absolute_difference_pp']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
