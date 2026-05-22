"""Reproduce the legacy VN30 L2 Logistic h40 61.51% reference."""

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
    EVAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    candidate_model,
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

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_legacy_rules_reference_and_stacking" / "reference_reproduction"
REFERENCE_ACCURACY = 0.6151202749140894
REFERENCE_ROWS = 4074
REFERENCE_VALIDATION_ACCURACY = 0.5188145188145188
EXPECTED_TRAIN_ROWS = 9600
EXPECTED_VALIDATION_ROWS = 30030
EXPECTED_FEATURE_COUNT = 99
MODEL_NAME = "l2_logistic"
MODEL_LABEL = "L2 Logistic"
HORIZON = 40
FEATURE_FAMILY = "baseline_C_closest"
LEGACY_FEATURE_SET = "feature_set_C_closest"
THRESHOLD = 0.50
TOLERANCE = 0.001


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


def accuracy(y_true: pd.Series, prediction: np.ndarray | pd.Series) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((y_true.astype(int).to_numpy() == np.asarray(prediction).astype(int)).mean())


def majority_prediction(train_y: pd.Series) -> int:
    return int(float(train_y.mean()) >= 0.5)


def prediction_frame(features: pd.DataFrame, idx: pd.Index, labels: pd.Series, probability: np.ndarray, split: str) -> pd.DataFrame:
    out = features.reindex(idx)[["datetime", "ticker"]].copy()
    out["horizon"] = HORIZON
    out["model"] = MODEL_LABEL
    out["feature_family"] = FEATURE_FAMILY
    out["feature_set"] = LEGACY_FEATURE_SET
    out["threshold_policy"] = "fixed_0.50"
    out["threshold"] = THRESHOLD
    out["split"] = split
    out["y_true"] = labels.reindex(idx).astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(probability, dtype=float)
    out["y_pred"] = (out["y_score_or_probability"].to_numpy(dtype=float) >= THRESHOLD).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    out["source_run_id"] = "vn30_legacy_reference_l2_logistic_h40_v1"
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def run_reference() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = active_stock_tickers()
    if len(tickers) != 30:
        raise ValueError(f"expected 30 VN30 tickers, got {len(tickers)}")
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    features, feature_cols = build_feature_set_c(stock_df, index_data)
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    labels = add_absolute_labels(features, HORIZON)
    idx = split_indices(features, labels)
    train_y = labels.reindex(idx["train"]).astype(int)
    val_y = labels.reindex(idx["validation"]).astype(int)
    final_y = labels.reindex(idx["final"]).astype(int)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", candidate_model(MODEL_NAME)),
        ]
    )
    model.fit(features.reindex(idx["train"])[feature_cols], train_y)
    val_prob = model.predict_proba(features.reindex(idx["validation"])[feature_cols])[:, 1]
    final_prob = model.predict_proba(features.reindex(idx["final"])[feature_cols])[:, 1]
    val_pred = (val_prob >= THRESHOLD).astype(int)
    final_pred = (final_prob >= THRESHOLD).astype(int)
    majority = majority_prediction(train_y)
    val_base = accuracy(val_y, np.full(len(val_y), majority))
    final_base = accuracy(final_y, np.full(len(final_y), majority))
    val_acc = accuracy(val_y, val_pred)
    final_acc = accuracy(final_y, final_pred)
    final_frame = prediction_frame(features, idx["final"], labels, final_prob, "final")
    summary = {
        "model": MODEL_LABEL,
        "model_id": MODEL_NAME,
        "horizon": HORIZON,
        "feature_family": FEATURE_FAMILY,
        "feature_set": LEGACY_FEATURE_SET,
        "threshold": THRESHOLD,
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "final_rows": int(len(final_y)),
        "feature_count": int(len(feature_cols)),
        "ticker_coverage": int(final_frame["ticker"].nunique()),
        "validation_accuracy": val_acc,
        "validation_majority_baseline": val_base,
        "final_accuracy": final_acc,
        "final_majority_baseline": final_base,
        "lift_vs_majority": final_acc - final_base,
        "difference_vs_61_51": final_acc - REFERENCE_ACCURACY,
        "difference_vs_expected_validation": val_acc - REFERENCE_VALIDATION_ACCURACY,
        "expected_train_rows": EXPECTED_TRAIN_ROWS,
        "expected_validation_rows": EXPECTED_VALIDATION_ROWS,
        "expected_final_rows": REFERENCE_ROWS,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "reference_reproduced": abs(final_acc - REFERENCE_ACCURACY) <= TOLERANCE and int(len(final_y)) == REFERENCE_ROWS,
        "row_count_matched": int(len(final_y)) == REFERENCE_ROWS,
        "feature_count_matched": int(len(feature_cols)) == EXPECTED_FEATURE_COUNT,
        "full_ticker_coverage": int(final_frame["ticker"].nunique()) == 30,
    }
    manifest = {
        "run_id": "vn30_legacy_reference_l2_logistic_h40_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_fetch": False,
        "provider_behavior_changed": False,
        "model_training": True,
        "model_selection": "fixed_reference_candidate_only",
        "final_window_role": "scoring_only",
        "confidence_abstention": False,
        "ticker_subset": False,
        "topk": False,
        "stock_only": True,
        "split_rule": "legacy_feature_timestamp_split_with_non_null_horizon_label",
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(EVAL_START),
        "feature_builder": "scripts.research.vn30_hourly_dual_track_common.build_feature_set_c",
        "label_builder": "scripts.research.vn30_hourly_dual_track_common.add_absolute_labels",
        "model_factory": "scripts.research.run_vn30_hourly_track_a_target62_validation_safe.candidate_model",
        "summary": summary,
        "input_paths": {
            "stock_cache": "data/market_cache/vnstock_data/vn30/hourly_2015",
            "index_cache": "data/market_cache/vnstock_data/indices/hourly_2015",
        },
    }
    return pd.DataFrame([summary]), final_frame, manifest


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, predictions, manifest = run_reference()
    summary.to_csv(OUTPUT_DIR / "reference_reproduction_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "reference_reproduction_row_predictions.csv", index=False)
    write_json(OUTPUT_DIR / "reference_reproduction_manifest.json", manifest)
    row = summary.iloc[0].to_dict()
    lines = [
        "# VN30 Legacy Reference Reproduction",
        "",
        f"- Model: {row['model']}.",
        f"- Horizon: h{int(row['horizon'])}.",
        f"- Feature family: `{row['feature_family']}` / `{row['feature_set']}`.",
        f"- Threshold: {float(row['threshold']):.2f}.",
        f"- Train rows: {int(row['train_rows'])}.",
        f"- Validation rows: {int(row['validation_rows'])}.",
        f"- Final rows: {int(row['final_rows'])}.",
        f"- Feature count: {int(row['feature_count'])}.",
        f"- Ticker coverage: {int(row['ticker_coverage'])}/30.",
        f"- Validation accuracy: {pct(row['validation_accuracy'])}.",
        f"- Reproduced final accuracy: {pct(row['final_accuracy'])}.",
        f"- Difference vs 61.51% reference: {pp(row['difference_vs_61_51'])}.",
        f"- Reproduction passed: {'yes' if bool(row['reference_reproduced']) else 'no'}.",
        "",
        "## Boundary",
        "",
        "- Data fetched: no.",
        "- Provider behavior changed: no.",
        "- Final window role: scoring-only.",
        "- Trading/profitability/live-deployment claim: no.",
    ]
    write_markdown(OUTPUT_DIR / "reference_reproduction_report.md", "\n".join(lines))
    print(f"reference_reproduced={bool(row['reference_reproduced'])} final_accuracy={pct(row['final_accuracy'])} output_dir={rel(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
