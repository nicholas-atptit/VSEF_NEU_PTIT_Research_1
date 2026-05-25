"""Robustness diagnostics for the fixed VN30 strict replay champion."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_champion_rescue_tuning import (  # noqa: E402
    REPLAY_CANDIDATE_ID,
    REPLAY_HORIZON,
    REPLAY_THRESHOLD,
    accuracy,
    build_feature_frame,
    clean_matrix,
    make_l2_old_candidate,
    pct,
    pp,
    prediction_frame,
    split_indices,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    add_absolute_labels,
    rel,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_strict_replay_robustness"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_SELECTED_CANDIDATE_STRICT_REPLAY_ROBUSTNESS_RESULT.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_SELECTED_CANDIDATE_STRICT_REPLAY_ROBUSTNESS_CLAIM_BOUNDARY.md"

THRESHOLDS = [0.49, 0.50, 0.51]
HORIZONS = [35, 40, 45]
BASELINES = ["always_up", "always_down", "lag1_direction", "vnindex_direction_lag1"]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def required_baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    rows = len(frame)
    out: dict[str, np.ndarray] = {
        "always_up": np.ones(rows, dtype=int),
        "always_down": np.zeros(rows, dtype=int),
    }
    if "return_1_lag_1" in frame.columns:
        out["lag1_direction"] = (pd.to_numeric(frame["return_1_lag_1"], errors="coerce") > 0.0).fillna(False).astype(int).to_numpy()
    else:
        out["lag1_direction"] = np.zeros(rows, dtype=int)
    if "vnindex_lag_1" in frame.columns:
        out["vnindex_direction_lag1"] = (pd.to_numeric(frame["vnindex_lag_1"], errors="coerce") > 0.0).fillna(False).astype(int).to_numpy()
    else:
        out["vnindex_direction_lag1"] = np.zeros(rows, dtype=int)
    return out


def fit_champion_for_horizon(features: pd.DataFrame, feature_cols: list[str], horizon: int) -> tuple[pd.Series, dict[str, pd.Index], Any, np.ndarray]:
    labels = add_absolute_labels(features, horizon)
    splits = split_indices(features, labels)
    train_y = labels.reindex(splits["train"]).astype(int)
    model = make_l2_old_candidate()
    model.fit(clean_matrix(features.loc[splits["train"]], feature_cols), train_y)
    final_prob = model.predict_proba(clean_matrix(features.loc[splits["final"]], feature_cols))[:, 1]
    return labels, splits, model, final_prob


def add_period_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["quarter"] = pd.to_datetime(out["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    return out


def attach_baseline_feature_columns(features: pd.DataFrame, idx: pd.Index, frame: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ("return_1_lag_1", "vnindex_lag_1") if col in features.columns]
    if not cols:
        return frame
    feature_slice = features.loc[idx, ["datetime", "ticker", *cols]].copy()
    return frame.merge(feature_slice, on=["datetime", "ticker"], how="left")


def strongest_baseline_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["y_true"].astype(int).to_numpy()
    baseline_preds = required_baseline_predictions(frame)
    rows = []
    for name in BASELINES:
        pred = baseline_preds[name]
        rows.append({"baseline_name": name, "baseline_accuracy": accuracy(y_true, pred)})
    return sorted(rows, key=lambda row: (float(row["baseline_accuracy"]), row["baseline_name"]), reverse=True)[0]


def slice_rows(frame: pd.DataFrame, group_col: str | None, slice_type: str) -> list[dict[str, Any]]:
    work = frame.copy()
    groups = [("overall", work)] if group_col is None else list(work.groupby(group_col, sort=True))
    rows: list[dict[str, Any]] = []
    for key, group in groups:
        y_true = group["y_true"].astype(int).to_numpy()
        model_acc = float(group["correct"].mean()) if len(group) else math.nan
        pred_up = float(group["y_pred"].astype(int).mean()) if len(group) else math.nan
        baseline_preds = required_baseline_predictions(group)
        baseline_accs = {name: accuracy(y_true, baseline_preds[name]) for name in BASELINES}
        strongest_name = max(BASELINES, key=lambda name: baseline_accs[name])
        row = {
            "slice_type": slice_type,
            "slice_value": str(key),
            "rows": int(len(group)),
            "model_accuracy": model_acc,
            "prediction_up_ratio": pred_up,
            "prediction_down_ratio": 1.0 - pred_up if math.isfinite(pred_up) else math.nan,
            "strongest_baseline_name": strongest_name,
            "strongest_baseline_accuracy": baseline_accs[strongest_name],
            "lift_over_strongest_baseline": model_acc - baseline_accs[strongest_name],
        }
        for name in BASELINES:
            row[f"{name}_accuracy"] = baseline_accs[name]
            row[f"lift_vs_{name}"] = model_acc - baseline_accs[name]
        rows.append(row)
    return rows


def confusion_rows(frame: pd.DataFrame, group_col: str | None, slice_type: str) -> list[dict[str, Any]]:
    work = frame.copy()
    groups = [("overall", work)] if group_col is None else list(work.groupby(group_col, sort=True))
    rows: list[dict[str, Any]] = []
    for key, group in groups:
        y_true = group["y_true"].astype(int)
        y_pred = group["y_pred"].astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": str(key),
                "rows": int(len(group)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": (tp + tn) / len(group) if len(group) else math.nan,
                "predicted_up": int((y_pred == 1).sum()),
                "predicted_down": int((y_pred == 0).sum()),
                "actual_up": int((y_true == 1).sum()),
                "actual_down": int((y_true == 0).sum()),
            }
        )
    return rows


def rolling_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for window in (250, 500, 1000):
        rolling = work["correct"].astype(float).rolling(window, min_periods=window).mean().dropna()
        rows.append(
            {
                "window": window,
                "count": int(len(rolling)),
                "min": float(rolling.min()) if len(rolling) else math.nan,
                "p10": float(rolling.quantile(0.10)) if len(rolling) else math.nan,
                "median": float(rolling.median()) if len(rolling) else math.nan,
                "p90": float(rolling.quantile(0.90)) if len(rolling) else math.nan,
                "max": float(rolling.max()) if len(rolling) else math.nan,
                "mean": float(rolling.mean()) if len(rolling) else math.nan,
                "windows_below_50": int((rolling < 0.50).sum()) if len(rolling) else 0,
                "windows_below_60": int((rolling < 0.60).sum()) if len(rolling) else 0,
            }
        )
    return pd.DataFrame(rows)


def prediction_balance(frame: pd.DataFrame) -> pd.DataFrame:
    work = add_period_columns(frame)
    rows: list[dict[str, Any]] = []
    for slice_type, group_col in (("overall", None), ("quarter", "quarter"), ("ticker", "ticker")):
        groups = [("overall", work)] if group_col is None else list(work.groupby(group_col, sort=True))
        for key, group in groups:
            up_ratio = float(group["y_pred"].astype(int).mean()) if len(group) else math.nan
            rows.append(
                {
                    "slice_type": slice_type,
                    "slice_value": str(key),
                    "rows": int(len(group)),
                    "prediction_up_ratio": up_ratio,
                    "prediction_down_ratio": 1.0 - up_ratio if math.isfinite(up_ratio) else math.nan,
                    "actual_up_ratio": float(group["y_true"].astype(int).mean()) if len(group) else math.nan,
                    "accuracy": float(group["correct"].mean()) if len(group) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def threshold_sensitivity(final_frame: pd.DataFrame, final_prob: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_source = final_frame.copy()
    sorted_prob = base_source["y_score_or_probability"].to_numpy(dtype=float)
    for threshold in THRESHOLDS:
        frame = base_source.copy()
        frame["threshold"] = threshold
        frame["y_pred"] = (sorted_prob >= threshold).astype(int)
        frame["correct"] = (frame["y_true"].astype(int).to_numpy() == frame["y_pred"].astype(int).to_numpy()).astype(int)
        strongest = strongest_baseline_for_frame(frame)
        acc = float(frame["correct"].mean())
        rows.append(
            {
                "threshold": threshold,
                "rows": int(len(frame)),
                "accuracy": acc,
                "strongest_baseline_name": strongest["baseline_name"],
                "strongest_baseline_accuracy": strongest["baseline_accuracy"],
                "lift_over_strongest_baseline": acc - float(strongest["baseline_accuracy"]),
                "prediction_up_ratio": float(frame["y_pred"].astype(int).mean()),
            }
        )
    return pd.DataFrame(rows)


def horizon_sensitivity(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        labels, splits, _model, final_prob = fit_champion_for_horizon(features, feature_cols, horizon)
        frame = prediction_frame(
            features,
            splits["final"],
            labels,
            final_prob,
            REPLAY_THRESHOLD,
            f"{REPLAY_CANDIDATE_ID}__horizon_sensitivity_h{horizon}",
            "final",
        )
        frame = attach_baseline_feature_columns(features, splits["final"], frame)
        strongest = strongest_baseline_for_frame(frame)
        acc = float(frame["correct"].mean())
        rows.append(
            {
                "horizon": horizon,
                "threshold": REPLAY_THRESHOLD,
                "rows": int(len(frame)),
                "accuracy": acc,
                "strongest_baseline_name": strongest["baseline_name"],
                "strongest_baseline_accuracy": strongest["baseline_accuracy"],
                "lift_over_strongest_baseline": acc - float(strongest["baseline_accuracy"]),
                "prediction_up_ratio": float(frame["y_pred"].astype(int).mean()),
            }
        )
    return pd.DataFrame(rows)


def leave_one_ticker_out(frame: pd.DataFrame) -> pd.DataFrame:
    full_strongest = strongest_baseline_for_frame(frame)
    full_accuracy = float(frame["correct"].mean())
    rows: list[dict[str, Any]] = []
    for ticker in sorted(frame["ticker"].astype(str).unique()):
        subset = frame[~frame["ticker"].astype(str).eq(ticker)].copy()
        strongest = strongest_baseline_for_frame(subset)
        acc = float(subset["correct"].mean())
        rows.append(
            {
                "scenario": "leave_one_ticker_out",
                "removed_ticker": ticker,
                "rows": int(len(subset)),
                "accuracy": acc,
                "accuracy_delta_vs_full": acc - full_accuracy,
                "strongest_baseline_name": strongest["baseline_name"],
                "strongest_baseline_accuracy": strongest["baseline_accuracy"],
                "lift_over_strongest_baseline": acc - float(strongest["baseline_accuracy"]),
                "full_accuracy": full_accuracy,
                "full_strongest_baseline_name": full_strongest["baseline_name"],
                "full_lift_over_strongest_baseline": full_accuracy - float(full_strongest["baseline_accuracy"]),
            }
        )
    ticker = pd.DataFrame(slice_rows(frame, "ticker", "ticker"))
    best_ticker = str(ticker.sort_values("model_accuracy", ascending=False).iloc[0]["slice_value"])
    worst_ticker = str(ticker.sort_values("model_accuracy", ascending=True).iloc[0]["slice_value"])
    for scenario, removed_ticker in (("remove_best_ticker", best_ticker), ("remove_worst_ticker", worst_ticker)):
        subset = frame[~frame["ticker"].astype(str).eq(removed_ticker)].copy()
        strongest = strongest_baseline_for_frame(subset)
        acc = float(subset["correct"].mean())
        rows.append(
            {
                "scenario": scenario,
                "removed_ticker": removed_ticker,
                "rows": int(len(subset)),
                "accuracy": acc,
                "accuracy_delta_vs_full": acc - full_accuracy,
                "strongest_baseline_name": strongest["baseline_name"],
                "strongest_baseline_accuracy": strongest["baseline_accuracy"],
                "lift_over_strongest_baseline": acc - float(strongest["baseline_accuracy"]),
                "full_accuracy": full_accuracy,
                "full_strongest_baseline_name": full_strongest["baseline_name"],
                "full_lift_over_strongest_baseline": full_accuracy - float(full_strongest["baseline_accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def build_reports(
    full_result: dict[str, Any],
    quarter: pd.DataFrame,
    ticker: pd.DataFrame,
    rolling: pd.DataFrame,
    balance: pd.DataFrame,
    leave_one: pd.DataFrame,
) -> None:
    strongest_quarter = quarter.sort_values("model_accuracy", ascending=False).iloc[0].to_dict()
    weakest_quarter = quarter.sort_values("model_accuracy", ascending=True).iloc[0].to_dict()
    strongest_ticker = ticker.sort_values("model_accuracy", ascending=False).iloc[0].to_dict()
    weakest_ticker = ticker.sort_values("model_accuracy", ascending=True).iloc[0].to_dict()
    overall_balance = balance[balance["slice_type"].eq("overall")].iloc[0].to_dict()
    rolling250 = rolling[rolling["window"].eq(250)].iloc[0].to_dict()
    rolling500 = rolling[rolling["window"].eq(500)].iloc[0].to_dict()
    rolling1000 = rolling[rolling["window"].eq(1000)].iloc[0].to_dict()
    ticker_below_50 = int((ticker["model_accuracy"] < 0.50).sum())
    quarter_below_50 = int((quarter["model_accuracy"] < 0.50).sum())
    severe_rolling = bool(float(rolling250["min"]) < 0.40 or float(rolling500["min"]) < 0.45)
    concentration = bool(ticker_below_50 > 0 or quarter_below_50 > 0)
    baseline60 = bool(
        abs(float(full_result["accuracy"]) - 0.6161021109474718) < 1e-12
        and abs(float(full_result["lift_over_strongest_baseline"]) - 0.10898379970544925) < 1e-12
    )

    result = f"""# VN30 Selected Candidate Strict Replay Robustness Result

## Fixed Champion

- Candidate: L2 Logistic, `feature_set_C_closest`, h40, threshold 0.50.
- Final accuracy: {pct(full_result["accuracy"])}.
- Strongest final baseline: {full_result["strongest_baseline_name"]} at {pct(full_result["strongest_baseline_accuracy"])}.
- Final lift over strongest baseline: {pp(full_result["lift_over_strongest_baseline"])}.
- Baseline60 diagnostic claim remains defensible: {str(baseline60).lower()}.
- Target62 and final65: not defensible.

## Quarter Stability

- Strongest quarter: {strongest_quarter["slice_value"]}, {pct(strongest_quarter["model_accuracy"])}, lift {pp(strongest_quarter["lift_over_strongest_baseline"])}.
- Weakest quarter: {weakest_quarter["slice_value"]}, {pct(weakest_quarter["model_accuracy"])}, lift {pp(weakest_quarter["lift_over_strongest_baseline"])}.
- Quarters below 50% accuracy: {quarter_below_50}.

## Ticker Stability

- Strongest ticker: {strongest_ticker["slice_value"]}, {pct(strongest_ticker["model_accuracy"])}, lift {pp(strongest_ticker["lift_over_strongest_baseline"])}.
- Weakest ticker: {weakest_ticker["slice_value"]}, {pct(weakest_ticker["model_accuracy"])}, lift {pp(weakest_ticker["lift_over_strongest_baseline"])}.
- Tickers below 50% accuracy: {ticker_below_50}.

## Rolling Stability

- Rolling250 min/p10/median/p90/max: {pct(rolling250["min"])} / {pct(rolling250["p10"])} / {pct(rolling250["median"])} / {pct(rolling250["p90"])} / {pct(rolling250["max"])}.
- Rolling500 min/p10/median/p90/max: {pct(rolling500["min"])} / {pct(rolling500["p10"])} / {pct(rolling500["median"])} / {pct(rolling500["p90"])} / {pct(rolling500["max"])}.
- Rolling1000 min/p10/median/p90/max: {pct(rolling1000["min"])} / {pct(rolling1000["p10"])} / {pct(rolling1000["median"])} / {pct(rolling1000["p90"])} / {pct(rolling1000["max"])}.
- Rolling instability remains severe: {str(severe_rolling).lower()}.

## Prediction Balance

- Overall up ratio: {pct(overall_balance["prediction_up_ratio"])}.
- Overall down ratio: {pct(overall_balance["prediction_down_ratio"])}.

## Concentration Disclosure

- Concentration/weak-slice risk present: {str(concentration).lower()}.
- Failed slices are retained in the CSV artifacts and not hidden.

No trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, paper, index-as-stock, or top-k-as-overall claim is made.
"""
    write_markdown(RESULT_PATH, result)

    claim = f"""# VN30 Selected Candidate Strict Replay Robustness Claim Boundary

- Claimable scope: fixed VN30 stock hourly strict replay champion robustness only.
- Candidate: L2 Logistic, feature_set_C_closest, h40, threshold 0.50.
- No new model selection, final optimization, or scope change was performed.
- Full final accuracy remains: {pct(full_result["accuracy"])}.
- Strongest final baseline remains: {full_result["strongest_baseline_name"]} at {pct(full_result["strongest_baseline_accuracy"])}.
- Final lift remains: {pp(full_result["lift_over_strongest_baseline"])}.
- Baseline60 diagnostic claim remains defensible: {str(baseline60).lower()}.
- Target62 claim: false.
- Final65 claim: false.
- Robustness caveat: rolling instability is severe, and weak ticker/quarter slices exist.
- No trading, profitability, BUY/SELL, investment recommendation, live deployment, VN100, DOCX, paper, index-as-stock, or top-k-as-overall claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, feature_groups, feature_manifest = build_feature_frame()
    feature_cols = feature_groups["feature_set_C_closest"]
    labels, splits, _model, final_prob = fit_champion_for_horizon(features, feature_cols, REPLAY_HORIZON)
    final_frame = prediction_frame(features, splits["final"], labels, final_prob, REPLAY_THRESHOLD, REPLAY_CANDIDATE_ID, "final")
    final_frame = attach_baseline_feature_columns(features, splits["final"], final_frame)
    final_frame = add_period_columns(final_frame)

    full_slice = slice_rows(final_frame, None, "overall")[0]
    quarter = pd.DataFrame(slice_rows(final_frame, "quarter", "quarter"))
    ticker = pd.DataFrame(slice_rows(final_frame, "ticker", "ticker"))
    rolling = rolling_distribution(final_frame)
    balance = prediction_balance(final_frame)
    confusion = pd.DataFrame(confusion_rows(final_frame, None, "overall") + confusion_rows(final_frame, "quarter", "quarter"))
    thresholds = threshold_sensitivity(final_frame, final_prob)
    horizons = horizon_sensitivity(features, feature_cols)
    leave_one = leave_one_ticker_out(final_frame)
    baseline_by_slice = pd.DataFrame(
        slice_rows(final_frame, None, "overall")
        + slice_rows(final_frame, "quarter", "quarter")
        + slice_rows(final_frame, "ticker", "ticker")
    )

    write_frame(OUTPUT_DIR / "quarter_stability.csv", quarter)
    write_frame(OUTPUT_DIR / "ticker_stability.csv", ticker)
    write_frame(OUTPUT_DIR / "rolling_distribution.csv", rolling)
    write_frame(OUTPUT_DIR / "prediction_balance.csv", balance)
    write_frame(OUTPUT_DIR / "confusion_matrix.csv", confusion)
    write_frame(OUTPUT_DIR / "threshold_sensitivity.csv", thresholds)
    write_frame(OUTPUT_DIR / "horizon_sensitivity.csv", horizons)
    write_frame(OUTPUT_DIR / "leave_one_ticker_out.csv", leave_one)
    write_frame(OUTPUT_DIR / "baseline_comparison_by_slice.csv", baseline_by_slice)

    strongest_quarter = quarter.sort_values("model_accuracy", ascending=False).iloc[0].to_dict()
    weakest_quarter = quarter.sort_values("model_accuracy", ascending=True).iloc[0].to_dict()
    strongest_ticker = ticker.sort_values("model_accuracy", ascending=False).iloc[0].to_dict()
    weakest_ticker = ticker.sort_values("model_accuracy", ascending=True).iloc[0].to_dict()
    manifest = {
        "created_at_utc": now_utc(),
        "scope": "VN30 stock hourly strict replay champion robustness",
        "candidate_id": REPLAY_CANDIDATE_ID,
        "model": "L2 Logistic Regression",
        "feature_group": "feature_set_C_closest",
        "horizon": REPLAY_HORIZON,
        "threshold": REPLAY_THRESHOLD,
        "final_rows": int(len(final_frame)),
        "full_result": full_slice,
        "strongest_quarter": strongest_quarter,
        "weakest_quarter": weakest_quarter,
        "strongest_ticker": strongest_ticker,
        "weakest_ticker": weakest_ticker,
        "rolling_distribution": rolling.to_dict("records"),
        "baseline60_remains_defensible": bool(
            abs(float(full_slice["model_accuracy"]) - 0.6161021109474718) < 1e-12
            and abs(float(full_slice["lift_over_strongest_baseline"]) - 0.10898379970544925) < 1e-12
        ),
        "target62_defensible": False,
        "final65_defensible": False,
        "feature_manifest": feature_manifest,
        "artifacts": {
            "quarter_stability": rel(OUTPUT_DIR / "quarter_stability.csv"),
            "ticker_stability": rel(OUTPUT_DIR / "ticker_stability.csv"),
            "rolling_distribution": rel(OUTPUT_DIR / "rolling_distribution.csv"),
            "prediction_balance": rel(OUTPUT_DIR / "prediction_balance.csv"),
            "confusion_matrix": rel(OUTPUT_DIR / "confusion_matrix.csv"),
            "threshold_sensitivity": rel(OUTPUT_DIR / "threshold_sensitivity.csv"),
            "horizon_sensitivity": rel(OUTPUT_DIR / "horizon_sensitivity.csv"),
            "leave_one_ticker_out": rel(OUTPUT_DIR / "leave_one_ticker_out.csv"),
            "baseline_comparison_by_slice": rel(OUTPUT_DIR / "baseline_comparison_by_slice.csv"),
        },
    }
    write_json(OUTPUT_DIR / "robustness_manifest.json", manifest)
    build_reports(
        {
            "accuracy": full_slice["model_accuracy"],
            "strongest_baseline_name": full_slice["strongest_baseline_name"],
            "strongest_baseline_accuracy": full_slice["strongest_baseline_accuracy"],
            "lift_over_strongest_baseline": full_slice["lift_over_strongest_baseline"],
        },
        quarter,
        ticker,
        rolling,
        balance,
        leave_one,
    )
    print(f"Robustness artifacts written: {rel(OUTPUT_DIR)}")
    print(f"Full accuracy: {pct(full_slice['model_accuracy'])}")
    print(f"Lift over strongest baseline: {pp(full_slice['lift_over_strongest_baseline'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
