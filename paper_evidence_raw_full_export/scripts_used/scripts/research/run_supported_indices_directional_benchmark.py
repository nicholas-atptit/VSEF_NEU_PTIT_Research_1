"""Run index-only directional benchmarks for supported Vietnamese indices."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.index_benchmark_common import (
    DAILY_CACHE,
    HOURLY_CACHE,
    INDEX_CODES,
    OUTPUT_DIR,
    TARGET60,
    accuracy,
    add_labels,
    baseline_predictions,
    build_features,
    read_index_frame,
    rel,
    split_frame,
    validate_ohlcv,
    write_csv,
)

DAILY_HORIZONS = (1, 5, 10, 20, 40, 60)
HOURLY_HORIZONS = (1, 4, 8, 20, 40, 60)
SEED = 42


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def train_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series):
    if y_train.nunique(dropna=True) < 2:
        return ("constant", int(y_train.dropna().iloc[0]) if len(y_train.dropna()) else 0)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=SEED,
            class_weight="balanced",
            n_jobs=1,
        ).fit(x_train, y_train)
    if model_name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=1,
        ).fit(x_train, y_train)
    if model_name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=180,
            num_leaves=15,
            max_depth=4,
            learning_rate=0.04,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
        ).fit(x_train, y_train)
    raise ValueError(f"unknown model: {model_name}")


def predict_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    if isinstance(model, tuple) and model[0] == "constant":
        return np.full(len(x), int(model[1]), dtype=int)
    return model.predict(x).astype(int)


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def claim_level(final_accuracy: float, final_rows: int, coverage: float) -> str:
    if math.isfinite(final_accuracy) and final_accuracy >= TARGET60 and coverage >= 0.95 and final_rows >= 100:
        return "baseline60_index_directional"
    return "failed"


def load_usable_inputs() -> list[tuple[str, str, pd.DataFrame]]:
    inputs: list[tuple[str, str, pd.DataFrame]] = []
    for frequency, cache_root in (("1D", DAILY_CACHE), ("1H", HOURLY_CACHE)):
        for code in INDEX_CODES:
            path = cache_root / f"{code}.csv"
            frame = read_index_frame(path, code, frequency)
            valid, _ = validate_ohlcv(frame, frequency)
            if not valid:
                continue
            splits = split_frame(frame, frequency)
            if len(splits["train"]) and len(splits["validation"]) and len(splits["final"]):
                inputs.append((code, frequency, frame))
    return inputs


def run_one(code: str, frequency: str, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    horizons = DAILY_HORIZONS if frequency == "1D" else HOURLY_HORIZONS
    accuracy_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    feature_frame, features = build_features(frame)
    for horizon in horizons:
        labeled = add_labels(feature_frame, horizon)
        splits = split_frame(labeled, frequency)
        train = splits["train"].dropna(subset=["y_true"]).copy()
        validation = splits["validation"].dropna(subset=["y_true"]).copy()
        final = splits["final"].dropna(subset=["y_true"]).copy()
        if len(train) < 50 or len(validation) < 20 or len(final) < 20:
            continue
        x_train = clean_features(train[features])
        y_train = train["y_true"].astype(int)
        x_val = clean_features(validation[features])
        y_val = validation["y_true"].astype(int)
        x_final = clean_features(final[features])
        y_final = final["y_true"].astype(int)

        baseline_scores: dict[str, float] = {}
        for baseline_name, preds in baseline_predictions(final, seed=SEED + horizon).items():
            score = accuracy(y_final, preds)
            baseline_scores[baseline_name] = float(score["accuracy"])
            baseline_rows.append(
                {
                    "index_code": code,
                    "frequency": frequency,
                    "horizon": horizon,
                    "baseline": baseline_name,
                    "final_accuracy": score["accuracy"],
                    "final_rows": score["rows"],
                    "final_coverage": 1.0,
                }
            )
        best_baseline_name = max(baseline_scores, key=baseline_scores.get)
        best_baseline_accuracy = baseline_scores[best_baseline_name]

        for model_name in ("random_forest", "xgboost", "lightgbm"):
            model = train_model(model_name, x_train, y_train)
            val_pred = predict_model(model, x_val)
            final_pred = predict_model(model, x_final)
            val_score = accuracy(y_val, val_pred)
            final_score = accuracy(y_final, final_pred)
            final_accuracy = float(final_score["accuracy"])
            final_rows = int(final_score["rows"])
            coverage = 1.0
            pass_60 = bool(final_accuracy >= TARGET60 and coverage >= 0.95)
            row = {
                "index_code": code,
                "frequency": frequency,
                "model": model_name,
                "horizon": horizon,
                "validation_accuracy": val_score["accuracy"],
                "validation_rows": val_score["rows"],
                "final_accuracy": final_accuracy,
                "final_rows": final_rows,
                "final_coverage": coverage,
                "baseline_name": best_baseline_name,
                "baseline_accuracy": best_baseline_accuracy,
                "delta_vs_baseline": final_accuracy - best_baseline_accuracy,
                "pass_60": pass_60,
                "claim_level": claim_level(final_accuracy, final_rows, coverage),
            }
            accuracy_rows.append(row)
            for i, (_, raw) in enumerate(final.iterrows()):
                prediction_rows.append(
                    {
                        "index_code": code,
                        "frequency": frequency,
                        "model": model_name,
                        "horizon": horizon,
                        "datetime": raw["datetime"],
                        "y_true": int(raw["y_true"]),
                        "y_pred": int(final_pred[i]),
                    }
                )
    return accuracy_rows, prediction_rows, baseline_rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_usable_inputs()
    run_config = {
        "created_at_utc": now_utc(),
        "scope": "index_only",
        "indices": list(INDEX_CODES),
        "daily_horizons": list(DAILY_HORIZONS),
        "hourly_horizons": list(HOURLY_HORIZONS),
        "models": ["random_forest", "xgboost", "lightgbm"],
        "baselines": ["majority_class", "previous_direction", "always_up", "random_seeded_direction", "moving_average_signal"],
        "no_resampling": True,
        "no_stock_claims": True,
        "no_trading_claims": True,
    }
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    all_accuracy: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_baselines: list[dict[str, Any]] = []
    for code, frequency, frame in inputs:
        print(f"Running {code} {frequency}: {len(frame)} rows")
        accuracy_rows, prediction_rows, baseline_rows = run_one(code, frequency, frame)
        all_accuracy.extend(accuracy_rows)
        all_predictions.extend(prediction_rows)
        all_baselines.extend(baseline_rows)

    accuracy_fields = [
        "index_code",
        "frequency",
        "model",
        "horizon",
        "validation_accuracy",
        "validation_rows",
        "final_accuracy",
        "final_rows",
        "final_coverage",
        "baseline_name",
        "baseline_accuracy",
        "delta_vs_baseline",
        "pass_60",
        "claim_level",
    ]
    baseline_fields = ["index_code", "frequency", "horizon", "baseline", "final_accuracy", "final_rows", "final_coverage"]
    prediction_fields = ["index_code", "frequency", "model", "horizon", "datetime", "y_true", "y_pred"]
    write_csv(OUTPUT_DIR / "accuracy_summary.csv", all_accuracy, accuracy_fields)
    write_csv(OUTPUT_DIR / "baseline_summary.csv", all_baselines, baseline_fields)
    write_csv(OUTPUT_DIR / "predicted_vs_actual.csv", all_predictions, prediction_fields)
    baseline_delta = [
        {
            "index_code": row["index_code"],
            "frequency": row["frequency"],
            "model": row["model"],
            "horizon": row["horizon"],
            "model_accuracy": row["final_accuracy"],
            "baseline_name": row["baseline_name"],
            "baseline_accuracy": row["baseline_accuracy"],
            "delta_vs_baseline": row["delta_vs_baseline"],
        }
        for row in all_accuracy
    ]
    write_csv(OUTPUT_DIR / "baseline_delta_summary.csv", baseline_delta)
    best_daily = max((row for row in all_accuracy if row["frequency"] == "1D"), key=lambda r: r["final_accuracy"], default={})
    best_hourly = max((row for row in all_accuracy if row["frequency"] == "1H"), key=lambda r: r["final_accuracy"], default={})
    passed = [row for row in all_accuracy if row["pass_60"]]
    manifest = {
        "created_at_utc": now_utc(),
        "input_combinations": [{"index_code": code, "frequency": frequency, "rows": int(len(frame))} for code, frequency, frame in inputs],
        "results": len(all_accuracy),
        "predicted_rows": len(all_predictions),
        "any_pass_60": bool(passed),
        "pass_60_count": len(passed),
        "best_daily": best_daily,
        "best_hourly": best_hourly,
        "daily_to_hourly_resampling_used": False,
        "hourly_to_daily_resampling_used": False,
        "stock_claims_made": False,
        "trading_profitability_claims_made": False,
    }
    (OUTPUT_DIR / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "benchmark_summary.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Index benchmark written: {rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
