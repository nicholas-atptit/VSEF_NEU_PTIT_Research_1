"""Run VN30 selected-candidate strict replay and local champion rescue tuning."""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    FINAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    assert_strict_target_boundaries,
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    rel,
    strict_target_split_indices,
    target_timestamp_from_labels,
)

REPLAY_DIR = REPO_ROOT / "reports" / "generated" / "vn30_selected_candidate_strict_replay"
RESCUE_DIR = REPO_ROOT / "reports" / "generated" / "vn30_champion_rescue_tuning"
REPLAY_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_SELECTED_CANDIDATE_STRICT_REPLAY_RESULT.md"
REPLAY_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_SELECTED_CANDIDATE_STRICT_REPLAY_CLAIM_BOUNDARY.md"
RESCUE_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_CHAMPION_RESCUE_TUNING_RESULT_SUMMARY.md"
RESCUE_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_CHAMPION_RESCUE_TUNING_CLAIM_BOUNDARY.md"

SEED = 42
LEGACY_REPLAY_INDEX_CODES = ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]
RESCUE_MARKET_CODES = ["VNINDEX", "VN30"]
REPLAY_CANDIDATE_ID = "old_selected_l2_logistic__feature_set_C_closest__h40__t0p50"
REPLAY_HORIZON = 40
REPLAY_THRESHOLD = 0.50
LOCAL_SEARCH_HORIZONS = [35, 40, 45, 50]
LOCAL_SEARCH_THRESHOLDS = [0.48, 0.49, 0.50, 0.51, 0.52, 0.53]
LOCAL_SEARCH_C_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0]
LOCAL_SEARCH_PENALTIES = ["l1", "l2", "elasticnet"]
LOCAL_SEARCH_CLASS_WEIGHTS = [None, "balanced"]
FEATURE_GROUP_ORDER = [
    "feature_set_C_closest",
    "feature_set_C_closest_plus_market",
    "feature_set_C_closest_plus_relative_strength",
    "compact_stable_features",
]


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


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number * 100.0:.2f}%" if math.isfinite(number) else ""


def pp(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number * 100.0:+.2f} pp" if math.isfinite(number) else ""


def accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(y_pred, dtype=int)).mean())


def majority_value(y_true: pd.Series | np.ndarray) -> int:
    if len(y_true) == 0:
        return 1
    return int(float(np.asarray(y_true, dtype=int).mean()) >= 0.5)


def make_l2_old_candidate() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    penalty="l2",
                    C=0.3,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_rescue_logistic(penalty: str, c_value: float, class_weight: str | None) -> Pipeline:
    if penalty == "elasticnet":
        model = LogisticRegression(
            max_iter=1000,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            C=float(c_value),
            class_weight=class_weight,
            random_state=SEED,
            tol=1e-3,
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            penalty=penalty,
            C=float(c_value),
            class_weight=class_weight,
            random_state=SEED,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])


def clean_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols].replace([np.inf, -np.inf], np.nan)


def build_market_addons(base: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    out = base.copy()
    cols: list[str] = []
    for code in RESCUE_MARKET_CODES:
        if code not in index_data:
            continue
        idx_df = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
        close = pd.to_numeric(idx_df["close"], errors="coerce")
        ret = close.pct_change(fill_method=None)
        prefix = code.lower()
        local_cols: list[str] = []
        for lag in (1, 2, 3, 5, 10):
            col = f"{prefix}_rescue_ret_lag_{lag}"
            idx_df[col] = ret.shift(lag)
            local_cols.append(col)
        for window in (5, 10, 20):
            ret_col = f"{prefix}_rescue_roll_return_{window}_lag"
            vol_col = f"{prefix}_rescue_roll_vol_{window}_lag"
            idx_df[ret_col] = (close / close.shift(window) - 1.0).shift(1)
            idx_df[vol_col] = ret.rolling(window, min_periods=max(3, window // 2)).std().shift(1)
            local_cols.extend([ret_col, vol_col])
        direction_col = f"{prefix}_rescue_direction_lag_1"
        idx_df[direction_col] = (ret.shift(1) > 0.0).astype(float)
        idx_df.loc[ret.shift(1).isna(), direction_col] = np.nan
        local_cols.append(direction_col)
        out = out.merge(idx_df[["datetime", *local_cols]], on="datetime", how="left")
        cols.extend(local_cols)
    return out, [col for col in cols if col in out.columns]


def build_relative_strength_addons(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = base.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    cols: list[str] = []
    for _ticker, group in out.groupby("ticker", sort=True):
        idx = group.index
        stock_ret = pd.to_numeric(group["return_1_lag_1"], errors="coerce") if "return_1_lag_1" in group else pd.Series(np.nan, index=idx)
        for code in ("vnindex", "vn30"):
            market_col = f"{code}_lag_1"
            if market_col not in out.columns:
                continue
            market_ret = pd.to_numeric(out.loc[idx, market_col], errors="coerce")
            rel = stock_ret.to_numpy(dtype=float) - market_ret.to_numpy(dtype=float)
            rel_series = pd.Series(rel, index=idx)
            col = f"rescue_relative_strength_vs_{code}_lag_1"
            out.loc[idx, col] = rel_series
            cols.append(col)
            for window in (5, 10, 20):
                roll_col = f"rescue_relative_strength_vs_{code}_{window}_lag"
                out.loc[idx, roll_col] = rel_series.rolling(window, min_periods=max(3, window // 2)).mean()
                cols.append(roll_col)
    return out, sorted({col for col in cols if col in out.columns})


def build_feature_frame() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers)
    all_index_data = load_index_data()
    legacy_index_data = {code: all_index_data[code] for code in LEGACY_REPLAY_INDEX_CODES if code in all_index_data}
    if len(tickers) != 30:
        raise ValueError(f"expected 30 VN30 tickers, got {len(tickers)}")
    base, base_cols = build_feature_set_c(stock_df, legacy_index_data)
    with_market, market_cols = build_market_addons(base, all_index_data)
    with_relative, relative_cols = build_relative_strength_addons(with_market)
    compact = [
        "return_1_lag_1",
        "return_1_lag_2",
        "return_1_lag_3",
        "return_1_lag_5",
        "return_1_lag_10",
        "rolling_return_mean_5",
        "rolling_return_vol_5",
        "momentum_5",
        "rolling_return_mean_10",
        "rolling_return_vol_10",
        "momentum_10",
        "rolling_return_mean_20",
        "rolling_return_vol_20",
        "momentum_20",
        "volume_shock_20",
        "high_low_range",
        "open_close_spread",
        "close_position_in_range",
        "vnindex_lag_1",
        "vn30_lag_1",
        "rescue_relative_strength_vs_vnindex_lag_1",
        "rescue_relative_strength_vs_vn30_lag_1",
        "day_of_week",
        "month",
        "hour",
    ]
    feature_groups = {
        "feature_set_C_closest": [col for col in base_cols if col in with_relative.columns],
        "feature_set_C_closest_plus_market": sorted(set(base_cols).union(market_cols)),
        "feature_set_C_closest_plus_relative_strength": sorted(set(base_cols).union(relative_cols)),
        "compact_stable_features": [col for col in compact if col in with_relative.columns],
    }
    all_cols = sorted({col for cols in feature_groups.values() for col in cols})
    with_relative[all_cols] = with_relative[all_cols].replace([np.inf, -np.inf], np.nan)
    manifest = {
        "stock_ticker_count": len(tickers),
        "stock_tickers": tickers,
        "stock_rows": int(len(stock_df)),
        "legacy_replay_index_codes": sorted(legacy_index_data.keys()),
        "rescue_market_addon_codes": [code for code in RESCUE_MARKET_CODES if code in all_index_data],
        "feature_groups": {name: {"feature_count": len(cols), "columns": cols} for name, cols in feature_groups.items()},
        "feature_set_C_closest_replay_count_matches_old_artifact": len(feature_groups["feature_set_C_closest"]) == 99,
    }
    return with_relative, feature_groups, manifest


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    splits = strict_target_split_indices(features, labels, TRAIN_END, VAL_START, VAL_END, FINAL_START)
    assert_strict_target_boundaries(features, labels, splits, TRAIN_END, VAL_START, VAL_END, FINAL_START)
    return splits


def split_guard_audit(features: pd.DataFrame, labels: pd.Series, splits: dict[str, pd.Index]) -> pd.DataFrame:
    target_timestamp = target_timestamp_from_labels(labels).reindex(features.index)
    rows: list[dict[str, Any]] = []
    for split, idx in splits.items():
        feature_ts = pd.to_datetime(features.loc[idx, "datetime"], errors="coerce") if len(idx) else pd.Series(dtype="datetime64[ns]")
        target_ts = target_timestamp.loc[idx] if len(idx) else pd.Series(dtype="datetime64[ns]")
        if split == "train":
            passes = bool(len(idx) == 0 or ((feature_ts <= TRAIN_END).all() and (target_ts <= TRAIN_END).all()))
            rule = "feature_timestamp <= train_end and target_timestamp <= train_end"
        elif split == "validation":
            passes = bool(len(idx) == 0 or (feature_ts.between(VAL_START, VAL_END).all() and target_ts.between(VAL_START, VAL_END).all()))
            rule = "feature_timestamp and target_timestamp inside validation period"
        else:
            passes = bool(len(idx) == 0 or ((feature_ts >= FINAL_START).all() and (target_ts >= FINAL_START).all()))
            rule = "feature_timestamp and target_timestamp inside final period"
        rows.append(
            {
                "split": split,
                "rows": int(len(idx)),
                "feature_timestamp_min": str(feature_ts.min()) if len(idx) else "",
                "feature_timestamp_max": str(feature_ts.max()) if len(idx) else "",
                "target_timestamp_min": str(target_ts.min()) if len(idx) else "",
                "target_timestamp_max": str(target_ts.max()) if len(idx) else "",
                "rule": rule,
                "passes": passes,
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    score: np.ndarray,
    threshold: float,
    candidate_id: str,
    split: str,
) -> pd.DataFrame:
    out = features.loc[idx, ["datetime", "ticker"]].copy()
    out["target_timestamp"] = target_timestamp_from_labels(labels).reindex(idx).to_numpy()
    out["y_true"] = labels.reindex(idx).astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(score, dtype=float)
    out["threshold"] = float(threshold)
    out["y_pred"] = (out["y_score_or_probability"].to_numpy(dtype=float) >= float(threshold)).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    out["candidate_id"] = candidate_id
    out["split"] = split
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def baseline_arrays(features: pd.DataFrame, labels: pd.Series, idx: pd.Index, train_y: pd.Series) -> dict[str, np.ndarray]:
    selected = features.loc[idx]
    majority = majority_value(train_y)
    raw: dict[str, pd.Series | np.ndarray] = {
        "majority_class": np.full(len(idx), majority, dtype=float),
        "always_up": np.ones(len(idx), dtype=float),
    }
    if "return_1_lag_1" in selected.columns:
        raw["lag1_direction"] = (pd.to_numeric(selected["return_1_lag_1"], errors="coerce") > 0.0).astype(float)
    if "momentum_20" in selected.columns:
        raw["rolling_momentum_20"] = (pd.to_numeric(selected["momentum_20"], errors="coerce") > 0.0).astype(float)
    if "vnindex_lag_1" in selected.columns:
        raw["vnindex_direction_lag1"] = (pd.to_numeric(selected["vnindex_lag_1"], errors="coerce") > 0.0).astype(float)
    out: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        series = pd.Series(values, index=idx) if not isinstance(values, pd.Series) else values.reindex(idx)
        out[name] = pd.to_numeric(series, errors="coerce").fillna(float(majority)).round().clip(0, 1).astype(int).to_numpy()
    return out


def baseline_frames_for_split(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    split: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    idx = splits[split]
    train_y = labels.reindex(splits["train"]).dropna().astype(int)
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    y_true = labels.reindex(idx).dropna().astype(int)
    for baseline_name, pred in baseline_arrays(features, labels, idx, train_y).items():
        frame = prediction_frame(features, idx, labels, pred.astype(float), 0.50, f"baseline__{baseline_name}__h{horizon}", split)
        frame["baseline_name"] = baseline_name
        rows.append(
            {
                "split": split,
                "horizon": horizon,
                "baseline_name": baseline_name,
                "accuracy": accuracy(y_true, pred),
                "rows": int(len(y_true)),
                "prediction_up_ratio": float(np.asarray(pred, dtype=int).mean()) if len(pred) else math.nan,
            }
        )
        frames.append(frame)
    summary = pd.DataFrame(rows)
    strongest = summary.sort_values(["accuracy", "baseline_name"], ascending=[False, True]).iloc[0].to_dict()
    return summary, pd.concat(frames, ignore_index=True), strongest


def group_accuracy(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        frame.groupby(group_col, sort=True)
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"), prediction_up_ratio=("y_pred", "mean"), target_up_ratio=("y_true", "mean"))
        .reset_index()
    )


def rolling_stability_frame(frame: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    work = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True).copy()
    work["row_number"] = np.arange(1, len(work) + 1)
    work[f"rolling{window}_accuracy"] = work["correct"].astype(float).rolling(window, min_periods=window).mean()
    return work.dropna(subset=[f"rolling{window}_accuracy"])[["split", "candidate_id", "row_number", "datetime", "ticker", f"rolling{window}_accuracy"]]


def frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "accuracy": math.nan,
            "prediction_up_ratio": math.nan,
            "ticker_median_accuracy": math.nan,
            "ticker_min_accuracy": math.nan,
            "quarter_min_accuracy": math.nan,
            "rolling250_min_accuracy": math.nan,
        }
    work = frame.copy()
    work["quarter"] = pd.to_datetime(work["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    ticker = group_accuracy(work, "ticker")
    quarter = group_accuracy(work, "quarter")
    rolling = rolling_stability_frame(work)
    return {
        "rows": int(len(work)),
        "accuracy": float(work["correct"].mean()),
        "prediction_up_ratio": float(work["y_pred"].astype(int).mean()),
        "ticker_median_accuracy": float(ticker["accuracy"].median()) if not ticker.empty else math.nan,
        "ticker_min_accuracy": float(ticker["accuracy"].min()) if not ticker.empty else math.nan,
        "quarter_min_accuracy": float(quarter["accuracy"].min()) if not quarter.empty else math.nan,
        "quarter_mean_accuracy": float(quarter["accuracy"].mean()) if not quarter.empty else math.nan,
        "quarters_below_45": int((quarter["accuracy"] < 0.45).sum()) if not quarter.empty else 0,
        "rolling250_min_accuracy": float(rolling["rolling250_accuracy"].min()) if not rolling.empty else math.nan,
        "rolling250_mean_accuracy": float(rolling["rolling250_accuracy"].mean()) if not rolling.empty else math.nan,
    }


def acceptance_label(final_accuracy: float, final_lift: float, stability_passes: bool) -> str:
    if not math.isfinite(final_accuracy) or not math.isfinite(final_lift) or final_lift <= 0:
        return "not_claimable"
    if final_accuracy >= 0.62 and stability_passes:
        return "target62_candidate"
    if final_accuracy >= 0.60:
        return "baseline60_candidate"
    return "diagnostic_only"


def run_strict_replay(features: pd.DataFrame, feature_groups: dict[str, list[str]], feature_manifest: dict[str, Any]) -> dict[str, Any]:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = feature_groups["feature_set_C_closest"]
    labels = add_absolute_labels(features, REPLAY_HORIZON)
    splits = split_indices(features, labels)
    guard = split_guard_audit(features, labels, splits)
    write_frame(REPLAY_DIR / "old_candidate_split_guard_audit.csv", guard)

    train_y = labels.reindex(splits["train"]).astype(int)
    val_y = labels.reindex(splits["validation"]).astype(int)
    final_y = labels.reindex(splits["final"]).astype(int)
    model = make_l2_old_candidate()
    model.fit(clean_matrix(features.loc[splits["train"]], feature_cols), train_y)
    val_prob = model.predict_proba(clean_matrix(features.loc[splits["validation"]], feature_cols))[:, 1]
    final_prob = model.predict_proba(clean_matrix(features.loc[splits["final"]], feature_cols))[:, 1]
    val_frame = prediction_frame(features, splits["validation"], labels, val_prob, REPLAY_THRESHOLD, REPLAY_CANDIDATE_ID, "validation")
    final_frame = prediction_frame(features, splits["final"], labels, final_prob, REPLAY_THRESHOLD, REPLAY_CANDIDATE_ID, "final")
    validation_baselines, _validation_baseline_frames, strongest_validation = baseline_frames_for_split(features, labels, splits, "validation", REPLAY_HORIZON)
    final_baselines, final_baseline_frames, strongest_final = baseline_frames_for_split(features, labels, splits, "final", REPLAY_HORIZON)
    baseline_comparison = pd.concat([validation_baselines, final_baselines], ignore_index=True)

    validation_summary = frame_summary(val_frame)
    final_summary = frame_summary(final_frame)
    validation_accuracy = float(validation_summary["accuracy"])
    final_accuracy = float(final_summary["accuracy"])
    final_lift = final_accuracy - float(strongest_final["accuracy"])
    stability_passes = bool(
        final_summary["quarter_min_accuracy"] >= 0.45
        and final_summary["ticker_median_accuracy"] >= 0.50
        and 0.35 <= final_summary["prediction_up_ratio"] <= 0.65
    )
    result = {
        "candidate_id": REPLAY_CANDIDATE_ID,
        "model": "l2_logistic",
        "penalty": "l2",
        "solver": "liblinear",
        "C": 0.3,
        "class_weight": "balanced",
        "feature_group": "feature_set_C_closest",
        "feature_count": len(feature_cols),
        "horizon": REPLAY_HORIZON,
        "threshold": REPLAY_THRESHOLD,
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "final_rows": int(len(final_y)),
        "validation_accuracy": validation_accuracy,
        "validation_strongest_baseline_name": strongest_validation["baseline_name"],
        "validation_strongest_baseline_accuracy": float(strongest_validation["accuracy"]),
        "validation_lift_over_strongest_baseline": validation_accuracy - float(strongest_validation["accuracy"]),
        "final_accuracy": final_accuracy,
        "final_strongest_baseline_name": strongest_final["baseline_name"],
        "final_strongest_baseline_accuracy": float(strongest_final["accuracy"]),
        "final_lift_over_strongest_baseline": final_lift,
        "strict_split_guard_passed": bool(guard["passes"].all()),
        "survives_strict_replay": bool(final_accuracy >= 0.60 and final_lift > 0 and guard["passes"].all()),
        "acceptance_label": acceptance_label(final_accuracy, final_lift, stability_passes),
        **{f"final_{key}": value for key, value in final_summary.items()},
    }

    config = {
        "created_at_utc": now_utc(),
        "candidate_id": REPLAY_CANDIDATE_ID,
        "scope": "VN30 stock hourly selected-candidate strict replay",
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
        "model": "LogisticRegression(C=0.3, penalty=l2, solver=liblinear, class_weight=balanced)",
        "feature_group": "feature_set_C_closest",
        "feature_count": len(feature_cols),
        "horizon": REPLAY_HORIZON,
        "threshold": REPLAY_THRESHOLD,
        "selection_uses_final": False,
        "final_role": "strict_replay_scoring",
        "feature_manifest": feature_manifest,
    }
    write_json(REPLAY_DIR / "old_candidate_replay_config.json", config)
    write_frame(REPLAY_DIR / "old_candidate_strict_final_result.csv", pd.DataFrame([result]))
    write_frame(REPLAY_DIR / "old_candidate_baseline_comparison.csv", baseline_comparison)
    write_frame(REPLAY_DIR / "old_candidate_rolling_stability.csv", rolling_stability_frame(final_frame))
    write_frame(REPLAY_DIR / "old_candidate_ticker_stability.csv", group_accuracy(final_frame, "ticker"))
    manifest = {
        **config,
        "result": result,
        "baseline_rows": int(len(baseline_comparison)),
        "rolling_rows": int(len(rolling_stability_frame(final_frame))),
        "ticker_rows": int(final_frame["ticker"].nunique()),
    }
    write_json(REPLAY_DIR / "old_candidate_replay_manifest.json", manifest)

    replay_report = f"""# VN30 Selected Candidate Strict Replay Result

## Replay Candidate

- Candidate: `L2 Logistic`, `feature_set_C_closest`, h40, threshold 0.50.
- Strict split guard passed: {str(result["strict_split_guard_passed"]).lower()}.
- Train rows: {result["train_rows"]}.
- Validation rows: {result["validation_rows"]}.
- Final rows: {result["final_rows"]}.

## Result

- Validation accuracy: {pct(result["validation_accuracy"])}.
- Strongest validation baseline: {result["validation_strongest_baseline_name"]} at {pct(result["validation_strongest_baseline_accuracy"])}.
- Validation lift: {pp(result["validation_lift_over_strongest_baseline"])}.
- Final accuracy: {pct(result["final_accuracy"])}.
- Strongest final baseline: {result["final_strongest_baseline_name"]} at {pct(result["final_strongest_baseline_accuracy"])}.
- Final lift over strongest baseline: {pp(result["final_lift_over_strongest_baseline"])}.
- Rolling250 min accuracy: {pct(result["final_rolling250_min_accuracy"])}.
- Ticker median accuracy: {pct(result["final_ticker_median_accuracy"])}.
- Quarter min accuracy: {pct(result["final_quarter_min_accuracy"])}.
- Survives strict replay as baseline60 diagnostic candidate: {str(result["survives_strict_replay"]).lower()}.
- Acceptance label: {result["acceptance_label"]}.

No trading, profitability, BUY/SELL, recommendation, live-deployment, VN100, top-k-as-overall-accuracy, DOCX, or paper claim is made.
"""
    write_markdown(REPLAY_RESULT_PATH, replay_report)
    replay_claim = f"""# VN30 Selected Candidate Strict Replay Claim Boundary

- Claimable scope: VN30 stock hourly selected-candidate strict replay only.
- Candidate: L2 Logistic, feature_set_C_closest, h40, threshold 0.50.
- Strict target_timestamp split discipline: {str(result["strict_split_guard_passed"]).lower()}.
- Final accuracy: {pct(result["final_accuracy"])}.
- Strongest final baseline: {result["final_strongest_baseline_name"]} at {pct(result["final_strongest_baseline_accuracy"])}.
- Final lift over strongest baseline: {pp(result["final_lift_over_strongest_baseline"])}.
- Baseline60 diagnostic candidate: {str(result["survives_strict_replay"]).lower()}.
- Target62 diagnostic candidate: false.
- Final65 remains not defensible.
- No trading, profitability, BUY/SELL, investment recommendation, live deployment, VN100, DOCX, paper, index-as-stock, or top-k-as-overall claim is made.
"""
    write_markdown(REPLAY_CLAIM_PATH, replay_claim)
    return {
        "result": result,
        "features": features,
        "feature_groups": feature_groups,
        "feature_manifest": feature_manifest,
        "baseline_frames_final": final_baseline_frames,
    }


def candidate_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seq = 0
    for feature_group, horizon, penalty, c_value, class_weight in itertools.product(
        FEATURE_GROUP_ORDER,
        LOCAL_SEARCH_HORIZONS,
        LOCAL_SEARCH_PENALTIES,
        LOCAL_SEARCH_C_VALUES,
        LOCAL_SEARCH_CLASS_WEIGHTS,
    ):
        seq += 1
        solver = "saga" if penalty == "elasticnet" else "liblinear"
        for threshold in LOCAL_SEARCH_THRESHOLDS:
            rows.append(
                {
                    "candidate_id": f"rescue_{seq:04d}__t{threshold:.2f}".replace(".", "p"),
                    "fit_id": f"rescue_{seq:04d}",
                    "model": "logistic_regression",
                    "feature_group": feature_group,
                    "horizon": horizon,
                    "penalty": penalty,
                    "solver": solver,
                    "C": c_value,
                    "class_weight": "" if class_weight is None else class_weight,
                    "threshold": threshold,
                    "selection_source": "validation_only",
                }
            )
    return pd.DataFrame(rows)


def quarterly_lift(candidate_frame: pd.DataFrame, baseline_frame: pd.DataFrame) -> pd.DataFrame:
    cand = candidate_frame.copy()
    base = baseline_frame.copy()
    cand["quarter"] = pd.to_datetime(cand["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    base["quarter"] = pd.to_datetime(base["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    cand_q = group_accuracy(cand, "quarter").rename(columns={"accuracy": "candidate_accuracy", "rows": "candidate_rows"})
    base_q = group_accuracy(base, "quarter").rename(columns={"accuracy": "baseline_accuracy", "rows": "baseline_rows"})
    out = cand_q.merge(base_q[["quarter", "baseline_accuracy", "baseline_rows"]], on="quarter", how="left")
    out["lift_over_baseline"] = out["candidate_accuracy"] - out["baseline_accuracy"]
    return out


def validation_ticker_metrics(candidate_frame: pd.DataFrame, baseline_frame: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    cand = group_accuracy(candidate_frame, "ticker").rename(columns={"accuracy": "candidate_accuracy"})
    base = group_accuracy(baseline_frame, "ticker").rename(columns={"accuracy": "baseline_accuracy"})
    out = cand.merge(base[["ticker", "baseline_accuracy"]], on="ticker", how="left")
    out["lift_over_baseline"] = out["candidate_accuracy"] - out["baseline_accuracy"]
    return out, float(out["candidate_accuracy"].median()), float(out["baseline_accuracy"].median())


def run_rescue_search(features: pd.DataFrame, feature_groups: dict[str, list[str]], feature_manifest: dict[str, Any]) -> dict[str, Any]:
    RESCUE_DIR.mkdir(parents=True, exist_ok=True)
    grid = candidate_grid()
    write_frame(RESCUE_DIR / "candidate_grid.csv", grid)
    run_config = {
        "created_at_utc": now_utc(),
        "scope": "VN30 stock hourly champion rescue tuning",
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
        "models": ["logistic_regression"],
        "penalties": LOCAL_SEARCH_PENALTIES,
        "C": LOCAL_SEARCH_C_VALUES,
        "class_weight": ["None", "balanced"],
        "horizons": LOCAL_SEARCH_HORIZONS,
        "thresholds": LOCAL_SEARCH_THRESHOLDS,
        "feature_groups": FEATURE_GROUP_ORDER,
        "selection_uses_final": False,
        "feature_manifest": feature_manifest,
    }
    write_json(RESCUE_DIR / "run_config.json", run_config)

    label_cache: dict[int, tuple[pd.Series, dict[str, pd.Index], pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    baseline_comparison_rows: list[pd.DataFrame] = []
    for horizon in LOCAL_SEARCH_HORIZONS:
        labels = add_absolute_labels(features, horizon)
        splits = split_indices(features, labels)
        validation_baselines, validation_baseline_frames, strongest_validation = baseline_frames_for_split(features, labels, splits, "validation", horizon)
        label_cache[horizon] = (labels, splits, validation_baselines, validation_baseline_frames, strongest_validation)
        baseline_comparison_rows.append(validation_baselines)

    validation_rows: list[dict[str, Any]] = []
    quarterly_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    fit_grid = grid.drop_duplicates(["fit_id", "feature_group", "horizon", "penalty", "solver", "C", "class_weight"]).copy()
    for fit_row in fit_grid.to_dict("records"):
        horizon = int(fit_row["horizon"])
        feature_group = str(fit_row["feature_group"])
        feature_cols = [col for col in feature_groups[feature_group] if col in features.columns]
        labels, splits, _validation_baselines, validation_baseline_frames, strongest_validation = label_cache[horizon]
        train_y = labels.reindex(splits["train"]).astype(int)
        validation_y = labels.reindex(splits["validation"]).astype(int)
        if train_y.nunique() < 2 or len(train_y) < 100 or len(validation_y) < 100:
            continue
        penalty = str(fit_row["penalty"])
        c_value = float(fit_row["C"])
        class_weight_text = str(fit_row["class_weight"])
        class_weight = None if class_weight_text == "" else class_weight_text
        model = make_rescue_logistic(penalty, c_value, class_weight)
        try:
            model.fit(clean_matrix(features.loc[splits["train"]], feature_cols), train_y)
            validation_prob = model.predict_proba(clean_matrix(features.loc[splits["validation"]], feature_cols))[:, 1]
        except Exception as exc:
            validation_rows.append(
                {
                    "candidate_id": f"{fit_row['fit_id']}__failed",
                    "fit_id": fit_row["fit_id"],
                    "status": "failed",
                    "failure_reason": str(exc)[:300],
                    "feature_group": feature_group,
                    "horizon": horizon,
                    "penalty": penalty,
                    "C": c_value,
                    "class_weight": class_weight_text,
                }
            )
            continue
        strongest_baseline_frame = validation_baseline_frames[
            validation_baseline_frames["baseline_name"].eq(strongest_validation["baseline_name"])
        ].copy()
        for threshold in LOCAL_SEARCH_THRESHOLDS:
            candidate_id = f"{fit_row['fit_id']}__t{threshold:.2f}".replace(".", "p")
            val_frame = prediction_frame(features, splits["validation"], labels, validation_prob, threshold, candidate_id, "validation")
            summary = frame_summary(val_frame)
            q = quarterly_lift(val_frame, strongest_baseline_frame)
            ticker_detail, ticker_median, baseline_ticker_median = validation_ticker_metrics(val_frame, strongest_baseline_frame)
            positive_quarters = int((q["lift_over_baseline"] > 0).sum())
            prediction_up_ratio = float(summary["prediction_up_ratio"])
            prediction_balance_passes = bool(0.35 <= prediction_up_ratio <= 0.65)
            validation_lift = float(summary["accuracy"]) - float(strongest_validation["accuracy"])
            validation_eligible = bool(
                validation_lift > 0.0
                and int(summary["rows"]) >= 3500
                and positive_quarters >= 3
                and float(summary["quarter_min_accuracy"]) >= 0.45
                and ticker_median >= baseline_ticker_median
                and prediction_balance_passes
            )
            validation_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fit_id": fit_row["fit_id"],
                    "status": "ok",
                    "model": "logistic_regression",
                    "feature_group": feature_group,
                    "feature_count": len(feature_cols),
                    "horizon": horizon,
                    "penalty": penalty,
                    "solver": fit_row["solver"],
                    "C": c_value,
                    "class_weight": class_weight_text,
                    "threshold": threshold,
                    "validation_accuracy": float(summary["accuracy"]),
                    "validation_rows": int(summary["rows"]),
                    "validation_strongest_baseline_name": strongest_validation["baseline_name"],
                    "validation_strongest_baseline_accuracy": float(strongest_validation["accuracy"]),
                    "validation_lift_over_strongest_baseline": validation_lift,
                    "validation_quarters_positive_lift": positive_quarters,
                    "validation_quarter_min_accuracy": float(summary["quarter_min_accuracy"]),
                    "validation_ticker_median_accuracy": ticker_median,
                    "validation_baseline_ticker_median_accuracy": baseline_ticker_median,
                    "validation_ticker_median_lift": ticker_median - baseline_ticker_median,
                    "validation_rolling250_min_accuracy": float(summary["rolling250_min_accuracy"]),
                    "prediction_up_ratio": prediction_up_ratio,
                    "prediction_balance_passes": prediction_balance_passes,
                    "validation_constraints_passed": validation_eligible,
                    "selection_source": "validation_only",
                    "final_accuracy_used_for_selection": False,
                    "model_complexity_rank": 0 if penalty == "l2" else (1 if penalty == "l1" else 2),
                    "feature_group_rank": FEATURE_GROUP_ORDER.index(feature_group),
                    "acceptance_label": "diagnostic_only" if validation_lift > 0 else "not_claimable",
                }
            )
            for row in q.to_dict("records"):
                quarterly_rows.append({"candidate_id": candidate_id, **row})
            balance_rows.append(
                {
                    "candidate_id": candidate_id,
                    "prediction_up_ratio": prediction_up_ratio,
                    "passes_35_65_band": prediction_balance_passes,
                    "justification": "" if prediction_balance_passes else "not eligible; no exception used",
                }
            )
            payloads[candidate_id] = {
                "model": model,
                "labels": labels,
                "splits": splits,
                "feature_cols": feature_cols,
                "threshold": threshold,
                "strongest_validation_baseline": strongest_validation,
            }

    validation = pd.DataFrame(validation_rows)
    quarterly = pd.DataFrame(quarterly_rows)
    balance = pd.DataFrame(balance_rows)
    write_frame(RESCUE_DIR / "validation_results.csv", validation)
    write_frame(RESCUE_DIR / "validation_quarterly_stability.csv", quarterly)
    write_frame(RESCUE_DIR / "validation_prediction_balance.csv", balance)

    valid = validation[validation["status"].eq("ok")].copy()
    eligible = valid[valid["validation_constraints_passed"].astype(bool)].copy()
    selection_pool = eligible if not eligible.empty else valid
    if selection_pool.empty:
        raise ValueError("no rescue candidates completed")
    selection_pool = selection_pool.sort_values(
        by=[
            "validation_lift_over_strongest_baseline",
            "validation_accuracy",
            "validation_quarters_positive_lift",
            "validation_ticker_median_lift",
            "validation_rolling250_min_accuracy",
            "validation_rows",
            "model_complexity_rank",
            "feature_group_rank",
        ],
        ascending=[False, False, False, False, False, False, True, True],
    )
    locked = selection_pool.iloc[0].to_dict()
    locked["locked_at_utc"] = now_utc()
    locked["selection_rule"] = (
        "validation-only: lift over strongest baseline, validation accuracy, quarterly stability, "
        "ticker stability, rolling stability, row count, simpler model"
    )
    locked["selected_from_constraint_passing_shortlist"] = bool(not eligible.empty)
    write_json(RESCUE_DIR / "locked_candidate.json", locked)

    payload = payloads[str(locked["candidate_id"])]
    labels = payload["labels"]
    splits = payload["splits"]
    final_prob = payload["model"].predict_proba(clean_matrix(features.loc[splits["final"]], payload["feature_cols"]))[:, 1]
    final_frame = prediction_frame(features, splits["final"], labels, final_prob, float(locked["threshold"]), str(locked["candidate_id"]), "final")
    final_baselines, final_baseline_frames, strongest_final = baseline_frames_for_split(features, labels, splits, "final", int(locked["horizon"]))
    baseline_comparison = pd.concat([*baseline_comparison_rows, final_baselines], ignore_index=True)
    final_summary = frame_summary(final_frame)
    final_accuracy = float(final_summary["accuracy"])
    final_lift = final_accuracy - float(strongest_final["accuracy"])
    final_stability_passes = bool(
        float(final_summary["quarter_min_accuracy"]) >= 0.45
        and float(final_summary["ticker_median_accuracy"]) >= float(locked["validation_baseline_ticker_median_accuracy"])
        and 0.35 <= float(final_summary["prediction_up_ratio"]) <= 0.65
    )
    final_result = {
        "candidate_id": locked["candidate_id"],
        "model": "logistic_regression",
        "feature_group": locked["feature_group"],
        "feature_count": int(locked["feature_count"]),
        "horizon": int(locked["horizon"]),
        "penalty": locked["penalty"],
        "solver": locked["solver"],
        "C": float(locked["C"]),
        "class_weight": locked["class_weight"],
        "threshold": float(locked["threshold"]),
        "selected_from_constraint_passing_shortlist": bool(locked["selected_from_constraint_passing_shortlist"]),
        "validation_accuracy": float(locked["validation_accuracy"]),
        "validation_lift_over_strongest_baseline": float(locked["validation_lift_over_strongest_baseline"]),
        "validation_constraints_passed": bool(locked["validation_constraints_passed"]),
        "final_accuracy": final_accuracy,
        "final_strongest_baseline_name": strongest_final["baseline_name"],
        "final_strongest_baseline_accuracy": float(strongest_final["accuracy"]),
        "final_lift_over_strongest_baseline": final_lift,
        "validation_final_gap": final_accuracy - float(locked["validation_accuracy"]),
        "final_rows": int(final_summary["rows"]),
        "final_prediction_up_ratio": float(final_summary["prediction_up_ratio"]),
        "final_quarter_min_accuracy": float(final_summary["quarter_min_accuracy"]),
        "final_ticker_median_accuracy": float(final_summary["ticker_median_accuracy"]),
        "final_rolling250_min_accuracy": float(final_summary["rolling250_min_accuracy"]),
        "acceptance_label": acceptance_label(final_accuracy, final_lift, final_stability_passes),
        "baseline60_defensible": bool(final_accuracy >= 0.60 and final_lift > 0.0),
        "target62_defensible": bool(final_accuracy >= 0.62 and final_lift > 0.0 and final_stability_passes),
        "final65_defensible": False,
        "final_role": "locked_candidate_scoring_once",
    }
    write_frame(RESCUE_DIR / "final_once_result.csv", pd.DataFrame([final_result]))
    write_frame(RESCUE_DIR / "baseline_comparison.csv", baseline_comparison)
    write_frame(RESCUE_DIR / "rolling_stability.csv", rolling_stability_frame(final_frame))
    write_frame(RESCUE_DIR / "ticker_stability.csv", group_accuracy(final_frame, "ticker"))
    tuning_manifest = {
        **run_config,
        "candidate_grid_rows": int(len(grid)),
        "completed_validation_rows": int(len(validation)),
        "constraint_passing_candidates": int(len(eligible)),
        "locked_candidate": locked,
        "final_once_result": final_result,
    }
    write_json(RESCUE_DIR / "tuning_manifest.json", tuning_manifest)
    write_rescue_reports(locked, final_result)
    return {"locked": locked, "final_result": final_result}


def write_rescue_reports(locked: dict[str, Any], final_result: dict[str, Any]) -> None:
    result = f"""# VN30 Champion Rescue Tuning Result Summary

## Best Validation Rescue Candidate

- Candidate: `{locked["candidate_id"]}`.
- Model: logistic regression.
- Feature group: {locked["feature_group"]}.
- Horizon: {int(locked["horizon"])}.
- Penalty/C/class weight: {locked["penalty"]} / {locked["C"]} / {locked["class_weight"] or "None"}.
- Threshold: {float(locked["threshold"]):.2f}.
- Validation accuracy: {pct(locked["validation_accuracy"])}.
- Validation lift over strongest baseline: {pp(locked["validation_lift_over_strongest_baseline"])}.
- Constraint-passing shortlist: {str(bool(locked["selected_from_constraint_passing_shortlist"])).lower()}.
- Validation positive-lift quarters: {int(locked["validation_quarters_positive_lift"])}.
- Validation ticker median lift: {pp(locked["validation_ticker_median_lift"])}.

## Locked Final Result

- Final accuracy: {pct(final_result["final_accuracy"])}.
- Strongest final baseline: {final_result["final_strongest_baseline_name"]} at {pct(final_result["final_strongest_baseline_accuracy"])}.
- Final lift over strongest baseline: {pp(final_result["final_lift_over_strongest_baseline"])}.
- Validation-final gap: {pp(final_result["validation_final_gap"])}.
- Final rows: {int(final_result["final_rows"])}.
- Rolling250 min accuracy: {pct(final_result["final_rolling250_min_accuracy"])}.
- Ticker median accuracy: {pct(final_result["final_ticker_median_accuracy"])}.
- Quarter min accuracy: {pct(final_result["final_quarter_min_accuracy"])}.
- Acceptance label: {final_result["acceptance_label"]}.

## Claim Boundary

- Baseline60 defensible: {str(bool(final_result["baseline60_defensible"])).lower()}.
- Target62 defensible: {str(bool(final_result["target62_defensible"])).lower()}.
- Final65 defensible: false.

Paper-safe wording:

> In a validation-locked VN30 stock hourly champion-rescue diagnostic benchmark, the selected logistic candidate reached {pct(final_result["final_accuracy"])} final pooled directional accuracy over {int(final_result["final_rows"])} rows, {pp(final_result["final_lift_over_strongest_baseline"])} versus the strongest same-horizon simple baseline. This supports only the stated diagnostic benchmark scope and does not support trading, profitability, live deployment, VN100, top-k-as-overall-accuracy, or final65 claims.
"""
    write_markdown(RESCUE_RESULT_PATH, result)
    claim = f"""# VN30 Champion Rescue Tuning Claim Boundary

- Claimable scope: VN30 stock hourly champion-rescue diagnostic benchmark only.
- Selection source: validation only.
- Final role: one-time locked-candidate scoring after `locked_candidate.json`.
- Final accuracy: {pct(final_result["final_accuracy"])}.
- Strongest final baseline: {final_result["final_strongest_baseline_name"]} at {pct(final_result["final_strongest_baseline_accuracy"])}.
- Final lift over strongest baseline: {pp(final_result["final_lift_over_strongest_baseline"])}.
- Baseline60 diagnostic candidate: {str(bool(final_result["baseline60_defensible"])).lower()}.
- Target62 diagnostic candidate: {str(bool(final_result["target62_defensible"])).lower()}.
- Final65 remains not defensible.
- No trading, profitability, BUY/SELL, investment recommendation, live deployment, VN100, DOCX, paper, index-as-stock, or top-k-as-overall claim is made.
"""
    write_markdown(RESCUE_CLAIM_PATH, claim)


def main() -> int:
    features, feature_groups, feature_manifest = build_feature_frame()
    replay = run_strict_replay(features, feature_groups, feature_manifest)
    if not bool(replay["result"]["strict_split_guard_passed"]):
        raise RuntimeError("strict replay split guard failed; local rescue search not run")
    rescue = run_rescue_search(features, feature_groups, feature_manifest)
    print(f"Strict replay final accuracy: {pct(replay['result']['final_accuracy'])}")
    print(f"Strict replay lift over strongest baseline: {pp(replay['result']['final_lift_over_strongest_baseline'])}")
    print(f"Rescue locked candidate: {rescue['locked']['candidate_id']}")
    print(f"Rescue final accuracy: {pct(rescue['final_result']['final_accuracy'])}")
    print(f"Rescue final lift: {pp(rescue['final_result']['final_lift_over_strongest_baseline'])}")
    print(f"Artifacts: {rel(REPLAY_DIR)} and {rel(RESCUE_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
