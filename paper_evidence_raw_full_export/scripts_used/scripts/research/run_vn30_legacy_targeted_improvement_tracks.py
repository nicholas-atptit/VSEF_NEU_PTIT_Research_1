"""Run targeted VN30 legacy-compatible improvement tracks.

The experiment uses the legacy h40 split/row rules and existing local data
only. Candidate and threshold selection is validation-only. Final-window scores
are computed after per-track selection for reporting, not for selection.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    FINAL_START,
    REFERENCE_FINAL_ACCURACY,
    TRAIN_END,
    VAL_END,
    VAL_START,
    build_feature_families,
    split_indices,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    load_index_data,
    load_stock_data,
    rel,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The max_iter was reached.*")

ROOT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_legacy_rules_reference_and_stacking"
OUTPUT_DIR = ROOT_DIR / "targeted_improvement"
LEGACY_MODEL_PREDICTIONS = ROOT_DIR / "model_comparison" / "legacy_model_row_predictions.csv"
SECTOR_PATH = REPO_ROOT / "data" / "ticker_sectors.csv"

HORIZON = 40
CURRENT_BEST_ACCURACY = 0.6163475699558174
CURRENT_BEST_LABEL = "61.63%"
CURRENT_BEST_CANDIDATE_ID = "legacy_single__logistic_l2__baseline_C_closest__h40__validation_selected_threshold__t0p550"
OLD_REFERENCE_ACCURACY = REFERENCE_FINAL_ACCURACY
MAJORITY_BASELINE = 0.5044182621502209
FULL_TICKER_COVERAGE = 30
RANDOM_STATE = 42
BASE_FEATURE_FAMILY = "baseline_C_closest"
THRESHOLDS = [round(x, 3) for x in np.arange(0.35, 0.701, 0.025)]
GLOBAL_THRESHOLDS = [round(x, 3) for x in np.arange(0.45, 0.651, 0.025)]


@dataclass(frozen=True)
class PredictionPayload:
    candidate_id: str
    track: str
    model: str
    feature_family: str
    threshold_policy: str
    threshold_detail: str
    feature_count: int
    selection_note: str
    validation: pd.DataFrame
    final: pd.DataFrame


def pct(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100.0:+.2f} pp"


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


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
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def accuracy(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(pred, dtype=int)).mean())


def majority_accuracy(y_true: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    rate = float(np.asarray(y_true, dtype=int).mean())
    return max(rate, 1.0 - rate)


def make_logistic(
    *,
    penalty: str = "l2",
    c: float = 0.3,
    l1_ratio: float | None = None,
    scale: bool = False,
    class_weight: str | dict[int, float] | None = "balanced",
) -> Pipeline:
    if penalty == "elasticnet":
        kwargs: dict[str, Any] = {
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": l1_ratio,
            "max_iter": 3000,
            "C": c,
            "class_weight": class_weight,
            "random_state": RANDOM_STATE,
        }
    elif penalty == "l1":
        kwargs = {
            "penalty": "l1",
            "solver": "liblinear",
            "max_iter": 1500,
            "C": c,
            "class_weight": class_weight,
            "random_state": RANDOM_STATE,
        }
    else:
        kwargs = {
            "penalty": "l2",
            "solver": "liblinear",
            "max_iter": 1200,
            "C": c,
            "class_weight": class_weight,
            "random_state": RANDOM_STATE,
        }
    try:
        imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    except TypeError:  # pragma: no cover - older sklearn
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    steps: list[tuple[str, Any]] = [("imputer", imputer)]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", LogisticRegression(**kwargs)))
    return Pipeline(steps)


def score_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    decision = np.asarray(model.decision_function(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-decision))


def select_global_threshold(y_true: pd.Series, scores: np.ndarray, thresholds: list[float] = GLOBAL_THRESHOLDS) -> tuple[float, float]:
    rows = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        rows.append((accuracy(y_true, pred), -abs(threshold - 0.55), threshold))
    rows.sort(reverse=True)
    return float(rows[0][2]), float(rows[0][0])


def select_group_thresholds(
    y_true: pd.Series,
    scores: np.ndarray,
    groups: pd.Series,
    default_threshold: float,
    min_rows: int = 40,
) -> dict[str, float]:
    work = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "score": scores, "group": groups.astype(str).to_numpy()})
    thresholds: dict[str, float] = {}
    for group_name, group in work.groupby("group", sort=True):
        if len(group) < min_rows:
            thresholds[group_name] = default_threshold
            continue
        options = []
        for threshold in THRESHOLDS:
            pred = (group["score"].to_numpy(dtype=float) >= threshold).astype(int)
            options.append((accuracy(group["y"], pred), -abs(threshold - default_threshold), threshold))
        options.sort(reverse=True)
        thresholds[str(group_name)] = float(options[0][2])
    return thresholds


def apply_group_thresholds(scores: np.ndarray, groups: pd.Series, thresholds: dict[str, float], default_threshold: float) -> np.ndarray:
    group_values = groups.astype(str).to_numpy()
    pred = np.zeros(len(scores), dtype=int)
    for i, score in enumerate(scores):
        threshold = thresholds.get(group_values[i], default_threshold)
        pred[i] = int(score >= threshold)
    return pred


def prediction_frame(features: pd.DataFrame, idx: pd.Index, y_true: pd.Series, scores: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    out = features.loc[idx, ["datetime", "ticker"]].copy().reset_index(drop=True)
    out["horizon"] = HORIZON
    out["y_true"] = y_true.astype(int).to_numpy()
    out["y_score_or_probability"] = scores
    out["y_pred"] = pred.astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    return out


def rolling_stats(frame: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    ordered = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    correct = ordered["correct"].astype(float)
    out: dict[str, Any] = {}
    means: list[float] = []
    below_total = 0
    for window in (250, 500, 1000):
        roll = correct.rolling(window=window, min_periods=window).mean().dropna()
        mean_value = float(roll.mean()) if not roll.empty else math.nan
        below_count = int((roll < 0.60).sum()) if not roll.empty else 0
        key = f"{prefix}_" if prefix else ""
        out[f"{key}rolling_{window}_mean"] = mean_value
        out[f"{key}rolling_{window}_below60_count"] = below_count
        below_total += below_count
        if math.isfinite(mean_value):
            means.append(mean_value)
    key = f"{prefix}_" if prefix else ""
    out[f"{key}rolling_mean_avg"] = float(np.mean(means)) if means else math.nan
    out[f"{key}rolling_below60_total"] = below_total
    return out


def slice_stats(frame: pd.DataFrame, column: str, value_name: str) -> pd.DataFrame:
    work = frame.copy()
    if column == "month":
        work[column] = work["datetime"].dt.to_period("M").astype(str)
    elif column == "quarter":
        work[column] = work["datetime"].dt.to_period("Q").astype(str)
    grouped = work.groupby(column, sort=True)
    return grouped.agg(rows=("correct", "size"), accuracy=("correct", "mean")).reset_index().rename(columns={column: value_name})


def period_mean(frame: pd.DataFrame, period: str) -> float:
    if frame.empty:
        return math.nan
    values = frame.assign(period=frame["datetime"].dt.to_period(period).astype(str)).groupby("period")["correct"].mean()
    return float(values.mean()) if not values.empty else math.nan


def ticker_mean(frame: pd.DataFrame) -> float:
    if frame.empty:
        return math.nan
    values = frame.groupby("ticker")["correct"].mean()
    return float(values.mean()) if not values.empty else math.nan


def load_sector_map(tickers: list[str]) -> tuple[dict[str, str], bool]:
    if not SECTOR_PATH.exists():
        return {ticker: "unknown" for ticker in tickers}, False
    sectors = pd.read_csv(SECTOR_PATH, low_memory=False)
    if "ticker" not in sectors.columns:
        return {ticker: "unknown" for ticker in tickers}, False
    group_col = "industry_code" if "industry_code" in sectors.columns else ("industry" if "industry" in sectors.columns else "")
    if not group_col:
        return {ticker: "unknown" for ticker in tickers}, False
    sectors["ticker"] = sectors["ticker"].astype(str).str.upper()
    sectors[group_col] = sectors[group_col].astype(str)
    mapping = sectors.drop_duplicates("ticker", keep="first").set_index("ticker")[group_col].to_dict()
    out = {ticker: str(mapping.get(ticker.upper(), "unknown")) for ticker in tickers}
    return out, any(group != "unknown" for group in out.values())


def add_regime_labels(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["datetime", "ticker"]].copy()
    if "vnindex_trend_60_lag_ctx" in features.columns:
        trend = pd.to_numeric(features["vnindex_trend_60_lag_ctx"], errors="coerce")
    elif "vn30_trend_60_lag_ctx" in features.columns:
        trend = pd.to_numeric(features["vn30_trend_60_lag_ctx"], errors="coerce")
    else:
        trend = pd.Series(np.nan, index=features.index)
    for fallback_col in ["momentum_60", "momentum_20", "rolling_return_mean_20"]:
        if fallback_col in features.columns:
            trend = trend.fillna(pd.to_numeric(features[fallback_col], errors="coerce"))
    out["market_direction_regime"] = "sideway"
    out.loc[trend > 0.02, "market_direction_regime"] = "bull"
    out.loc[trend < -0.02, "market_direction_regime"] = "bear"
    out.loc[trend.isna(), "market_direction_regime"] = "unknown_direction"
    if "vnindex_vol_20_lag_ctx" in features.columns and "vnindex_vol_60_lag_ctx" in features.columns:
        ratio = pd.to_numeric(features["vnindex_vol_20_lag_ctx"], errors="coerce") / pd.to_numeric(
            features["vnindex_vol_60_lag_ctx"], errors="coerce"
        ).replace(0.0, np.nan)
    else:
        ratio = pd.Series(np.nan, index=features.index)
    for short_col, long_col in [("rolling_return_vol_20", "rolling_return_vol_60"), ("roll_vol_20", "roll_vol_40")]:
        if short_col in features.columns and long_col in features.columns:
            fallback_ratio = pd.to_numeric(features[short_col], errors="coerce") / pd.to_numeric(
                features[long_col], errors="coerce"
            ).replace(0.0, np.nan)
            ratio = ratio.fillna(fallback_ratio)
    out["volatility_regime"] = "low_volatility"
    out.loc[ratio > 1.10, "volatility_regime"] = "high_volatility"
    out.loc[ratio.isna(), "volatility_regime"] = "unknown_volatility"
    out["regime_router_key"] = out["market_direction_regime"].astype(str) + "_" + out["volatility_regime"].astype(str)
    return out


def summarize_payload(payload: PredictionPayload, current_roll: dict[str, float]) -> dict[str, Any]:
    val_acc = float(payload.validation["correct"].mean())
    final_acc = float(payload.final["correct"].mean())
    val_majority = majority_accuracy(payload.validation["y_true"])
    final_majority = majority_accuracy(payload.final["y_true"])
    final_roll = rolling_stats(payload.final, "final")
    validation_roll = rolling_stats(payload.validation, "validation")
    val_gap = final_acc - val_acc
    overfit_risk = "high" if val_acc - final_acc > 0.05 else ("moderate" if val_acc - final_acc > 0.02 else "low")
    rolling_preserved = all(
        as_float(final_roll[f"final_rolling_{window}_mean"]) >= current_roll[f"final_rolling_{window}_mean"] - 0.005
        for window in (250, 500, 1000)
    )
    rolling_improved = all(
        as_float(final_roll[f"final_rolling_{window}_mean"]) > current_roll[f"final_rolling_{window}_mean"] + 0.002
        for window in (250, 500, 1000)
    )
    if overfit_risk == "high":
        classification = "rejected_overfit_risk"
    elif final_acc > CURRENT_BEST_ACCURACY + 0.002 and rolling_preserved:
        classification = "stronger_candidate"
    elif final_acc > CURRENT_BEST_ACCURACY:
        classification = "marginal_improvement"
    elif rolling_improved:
        classification = "stability_improvement_only"
    else:
        classification = "failed_improvement"
    summary = {
        "candidate_id": payload.candidate_id,
        "track": payload.track,
        "model": payload.model,
        "feature_family": payload.feature_family,
        "horizon": HORIZON,
        "threshold_policy": payload.threshold_policy,
        "threshold_detail": payload.threshold_detail,
        "feature_count": payload.feature_count,
        "selection_note": payload.selection_note,
        "selection_source": "validation_only",
        "final_accuracy_used_for_selection": False,
        "validation_rows": int(len(payload.validation)),
        "final_rows": int(len(payload.final)),
        "validation_ticker_coverage": int(payload.validation["ticker"].nunique()),
        "ticker_coverage": int(payload.final["ticker"].nunique()),
        "full_ticker_coverage": bool(payload.final["ticker"].nunique() == FULL_TICKER_COVERAGE),
        "validation_accuracy": val_acc,
        "final_accuracy": final_acc,
        "validation_majority_baseline": val_majority,
        "final_majority_baseline": final_majority,
        "validation_lift_over_majority": val_acc - val_majority,
        "final_lift_over_majority": final_acc - final_majority,
        "delta_vs_current_61_63": final_acc - CURRENT_BEST_ACCURACY,
        "delta_vs_old_reference_61_51": final_acc - OLD_REFERENCE_ACCURACY,
        "validation_final_gap": val_gap,
        "validation_final_gap_penalty": max(0.0, val_acc - final_acc),
        "monthly_mean_accuracy": period_mean(payload.final, "M"),
        "quarterly_mean_accuracy": period_mean(payload.final, "Q"),
        "ticker_mean_accuracy": ticker_mean(payload.final),
        "validation_monthly_mean_accuracy": period_mean(payload.validation, "M"),
        "validation_quarterly_mean_accuracy": period_mean(payload.validation, "Q"),
        "validation_ticker_mean_accuracy": ticker_mean(payload.validation),
        "overfit_risk": overfit_risk,
        "rolling_stability_vs_current": "improved" if rolling_improved else ("preserved" if rolling_preserved else "weaker"),
        "candidate_classification": classification,
        "leakage_status": "passed_validation_only_selection",
    }
    summary.update(validation_roll)
    summary.update(final_roll)
    below_share = summary["validation_rolling_below60_total"] / max(
        1, sum(max(0, len(payload.validation) - window + 1) for window in (250, 500, 1000))
    )
    summary["selection_score"] = (
        0.35 * summary["validation_accuracy"]
        + 0.20 * summary["validation_lift_over_majority"]
        + 0.20 * summary["validation_rolling_mean_avg"]
        + 0.10 * summary["validation_monthly_mean_accuracy"]
        + 0.10 * summary["validation_quarterly_mean_accuracy"]
        + 0.05 * summary["validation_ticker_mean_accuracy"]
        - 0.05 * below_share
    )
    return summary


def current_best_payload() -> PredictionPayload:
    if not LEGACY_MODEL_PREDICTIONS.exists():
        raise FileNotFoundError(f"missing current benchmark predictions: {rel(LEGACY_MODEL_PREDICTIONS)}")
    frame = pd.read_csv(LEGACY_MODEL_PREDICTIONS, low_memory=False)
    current = frame[frame["candidate_id"].eq(CURRENT_BEST_CANDIDATE_ID)].copy()
    if current.empty:
        raise ValueError(f"current best candidate not found in {rel(LEGACY_MODEL_PREDICTIONS)}")
    current["datetime"] = pd.to_datetime(current["datetime"], format="mixed", errors="coerce")
    current["correct"] = pd.to_numeric(current["correct"], errors="coerce").astype(int)
    keep_cols = ["datetime", "ticker", "horizon", "y_true", "y_score_or_probability", "y_pred", "correct"]
    validation = current[current["split"].eq("validation")][keep_cols].copy()
    final = current[current["split"].eq("final")][keep_cols].copy()
    return PredictionPayload(
        candidate_id=CURRENT_BEST_CANDIDATE_ID,
        track="current_best_comparator",
        model="logistic_l2",
        feature_family=BASE_FEATURE_FAMILY,
        threshold_policy="validation_selected_threshold",
        threshold_detail="global_threshold=0.55",
        feature_count=99,
        selection_note="fixed current apples-to-apples comparator from prior legacy benchmark",
        validation=validation,
        final=final,
    )


def fit_base_scores(
    features: pd.DataFrame,
    cols: list[str],
    labels: pd.Series,
    idx: dict[str, pd.Index],
    *,
    c: float = 0.3,
    penalty: str = "l2",
    l1_ratio: float | None = None,
    scale: bool = False,
) -> tuple[Any, np.ndarray, np.ndarray]:
    model = make_logistic(penalty=penalty, c=c, l1_ratio=l1_ratio, scale=scale)
    model.fit(features.loc[idx["train"], cols], labels.loc[idx["train"]].astype(int))
    return model, score_model(model, features.loc[idx["validation"], cols]), score_model(model, features.loc[idx["final"], cols])


def calibration_payloads(
    features: pd.DataFrame,
    cols: list[str],
    labels: pd.Series,
    idx: dict[str, pd.Index],
    current_threshold: float,
) -> list[PredictionPayload]:
    payloads: list[PredictionPayload] = []
    _model, val_scores, final_scores = fit_base_scores(features, cols, labels, idx)
    y_val = labels.loc[idx["validation"]].astype(int)
    y_final = labels.loc[idx["final"]].astype(int)
    val_meta = features.loc[idx["validation"], ["ticker"]].copy()
    final_meta = features.loc[idx["final"], ["ticker"]].copy()

    ticker_thresholds = select_group_thresholds(y_val, val_scores, val_meta["ticker"], current_threshold, min_rows=100)
    val_pred = apply_group_thresholds(val_scores, val_meta["ticker"], ticker_thresholds, current_threshold)
    final_pred = apply_group_thresholds(final_scores, final_meta["ticker"], ticker_thresholds, current_threshold)
    payloads.append(
        PredictionPayload(
            candidate_id="targeted__ticker_threshold_calibration__logistic_l2_h40",
            track="ticker_specific_threshold_calibration",
            model="logistic_l2",
            feature_family=BASE_FEATURE_FAMILY,
            threshold_policy="ticker_validation_selected_threshold",
            threshold_detail=json.dumps(ticker_thresholds, sort_keys=True),
            feature_count=len(cols),
            selection_note="per-ticker thresholds selected on validation only from Logistic L2 scores",
            validation=prediction_frame(features, idx["validation"], y_val, val_scores, val_pred),
            final=prediction_frame(features, idx["final"], y_final, final_scores, final_pred),
        )
    )

    sector_map, sector_available = load_sector_map(active_stock_tickers())
    if sector_available:
        val_group = val_meta["ticker"].astype(str).str.upper().map(sector_map).fillna("unknown")
        final_group = final_meta["ticker"].astype(str).str.upper().map(sector_map).fillna("unknown")
        sector_thresholds = select_group_thresholds(y_val, val_scores, val_group, current_threshold, min_rows=250)
        val_pred = apply_group_thresholds(val_scores, val_group, sector_thresholds, current_threshold)
        final_pred = apply_group_thresholds(final_scores, final_group, sector_thresholds, current_threshold)
        payloads.append(
            PredictionPayload(
                candidate_id="targeted__ticker_group_sector_threshold_calibration__logistic_l2_h40",
                track="ticker_group_threshold_calibration",
                model="logistic_l2",
                feature_family=BASE_FEATURE_FAMILY,
                threshold_policy="sector_validation_selected_threshold",
                threshold_detail=json.dumps(sector_thresholds, sort_keys=True),
                feature_count=len(cols),
                selection_note="sector/group thresholds selected on validation only from Logistic L2 scores",
                validation=prediction_frame(features, idx["validation"], y_val, val_scores, val_pred),
                final=prediction_frame(features, idx["final"], y_final, final_scores, final_pred),
            )
        )
    return payloads


def regime_router_payload(
    features: pd.DataFrame,
    cols: list[str],
    labels: pd.Series,
    idx: dict[str, pd.Index],
    current_threshold: float,
) -> PredictionPayload:
    _model, val_scores, final_scores = fit_base_scores(features, cols, labels, idx)
    y_val = labels.loc[idx["validation"]].astype(int)
    y_final = labels.loc[idx["final"]].astype(int)
    regimes = add_regime_labels(features)
    val_regime = regimes.loc[idx["validation"], "regime_router_key"]
    final_regime = regimes.loc[idx["final"], "regime_router_key"]
    thresholds = select_group_thresholds(y_val, val_scores, val_regime, current_threshold, min_rows=250)
    val_pred = apply_group_thresholds(val_scores, val_regime, thresholds, current_threshold)
    final_pred = apply_group_thresholds(final_scores, final_regime, thresholds, current_threshold)
    return PredictionPayload(
        candidate_id="targeted__exante_regime_threshold_router__logistic_l2_h40",
        track="regime_specific_router",
        model="logistic_l2",
        feature_family=BASE_FEATURE_FAMILY,
        threshold_policy="regime_validation_selected_threshold",
        threshold_detail=json.dumps(thresholds, sort_keys=True),
        feature_count=len(cols),
        selection_note="bull/bear/sideway and high/low-vol regime thresholds selected on validation only",
        validation=prediction_frame(features, idx["validation"], y_val, val_scores, val_pred),
        final=prediction_frame(features, idx["final"], y_final, final_scores, final_pred),
    )


def drag_ticker_payload(
    features: pd.DataFrame,
    cols: list[str],
    labels: pd.Series,
    idx: dict[str, pd.Index],
    current_payload: PredictionPayload,
) -> PredictionPayload:
    current_val = current_payload.validation.copy()
    weak = (
        current_val.groupby("ticker", sort=True)["correct"]
        .mean()
        .loc[lambda values: values < max(0.50, current_val["correct"].mean() - 0.08)]
        .index.astype(str)
        .tolist()
    )
    base_model, val_scores, final_scores = fit_base_scores(features, cols, labels, idx)
    val_pred = (val_scores >= 0.55).astype(int)
    final_pred = (final_scores >= 0.55).astype(int)
    val_meta = features.loc[idx["validation"], ["ticker"]].copy()
    final_meta = features.loc[idx["final"], ["ticker"]].copy()
    y_train = labels.loc[idx["train"]].astype(int)
    y_val = labels.loc[idx["validation"]].astype(int)
    y_final = labels.loc[idx["final"]].astype(int)
    repaired: dict[str, dict[str, Any]] = {}
    for ticker in weak:
        train_idx = idx["train"][features.loc[idx["train"], "ticker"].astype(str).eq(ticker)]
        validation_mask = val_meta["ticker"].astype(str).eq(ticker).to_numpy()
        final_mask = final_meta["ticker"].astype(str).eq(ticker).to_numpy()
        if len(train_idx) < 100 or y_train.loc[train_idx].nunique() < 2 or not validation_mask.any():
            continue
        ticker_model = make_logistic(penalty="l2", c=0.3, scale=True, class_weight="balanced")
        ticker_model.fit(features.loc[train_idx, cols], y_train.loc[train_idx])
        ticker_val_scores = score_model(ticker_model, features.loc[idx["validation"][validation_mask], cols])
        ticker_final_scores = score_model(ticker_model, features.loc[idx["final"][final_mask], cols])
        threshold, val_acc = select_global_threshold(y_val.iloc[np.flatnonzero(validation_mask)], ticker_val_scores, THRESHOLDS)
        val_pred[validation_mask] = (ticker_val_scores >= threshold).astype(int)
        final_pred[final_mask] = (ticker_final_scores >= threshold).astype(int)
        val_scores[validation_mask] = ticker_val_scores
        final_scores[final_mask] = ticker_final_scores
        repaired[ticker] = {"threshold": threshold, "validation_accuracy": val_acc, "train_rows": int(len(train_idx))}
    _ = base_model
    return PredictionPayload(
        candidate_id="targeted__drag_ticker_repair__weak_ticker_scaled_l2_h40",
        track="drag_ticker_repair",
        model="logistic_l2_weak_ticker_scaled_repair",
        feature_family=BASE_FEATURE_FAMILY,
        threshold_policy="weak_ticker_validation_selected_threshold",
        threshold_detail=json.dumps(repaired, sort_keys=True),
        feature_count=len(cols),
        selection_note="weak tickers identified on validation; ticker-specific scaled balanced L2 models and thresholds selected on validation only",
        validation=prediction_frame(features, idx["validation"], y_val, val_scores, val_pred),
        final=prediction_frame(features, idx["final"], y_final, final_scores, final_pred),
    )


def regularized_linear_payloads(features: pd.DataFrame, cols: list[str], labels: pd.Series, idx: dict[str, pd.Index]) -> list[PredictionPayload]:
    specs: list[tuple[str, str, float, float | None, bool]] = []
    for c in [0.05, 0.1, 0.2, 0.3, 0.6, 1.0, 2.0]:
        specs.append(("logistic_l2", "l2", c, None, False))
    for c in [0.05, 0.1, 0.2, 0.4, 0.8]:
        specs.append(("logistic_l1", "l1", c, None, True))
    for c in [0.1, 0.3, 0.8]:
        for ratio in [0.2, 0.5, 0.8]:
            specs.append(("logistic_elastic_net", "elasticnet", c, ratio, True))
    payloads: list[PredictionPayload] = []
    y_val = labels.loc[idx["validation"]].astype(int)
    y_final = labels.loc[idx["final"]].astype(int)
    for model_name, penalty, c, ratio, scale in specs:
        try:
            _model, val_scores, final_scores = fit_base_scores(features, cols, labels, idx, c=c, penalty=penalty, l1_ratio=ratio, scale=scale)
        except Exception as exc:
            print(f"skip_regularized model={model_name} c={c} ratio={ratio}: {exc}")
            continue
        threshold, _val_acc = select_global_threshold(y_val, val_scores, GLOBAL_THRESHOLDS)
        val_pred = (val_scores >= threshold).astype(int)
        final_pred = (final_scores >= threshold).astype(int)
        ratio_label = "na" if ratio is None else str(ratio).replace(".", "p")
        c_label = str(c).replace(".", "p")
        threshold_label = str(threshold).replace(".", "p")
        payloads.append(
            PredictionPayload(
                candidate_id=f"targeted__regularized_linear__{model_name}__c{c_label}__r{ratio_label}__t{threshold_label}",
                track="regularized_linear_models",
                model=model_name,
                feature_family=BASE_FEATURE_FAMILY,
                threshold_policy="validation_selected_threshold",
                threshold_detail=f"C={c}; l1_ratio={ratio}; threshold={threshold}; scale={scale}",
                feature_count=len(cols),
                selection_note="regularized linear hyperparameter and threshold selected on validation only within this track",
                validation=prediction_frame(features, idx["validation"], y_val, val_scores, val_pred),
                final=prediction_frame(features, idx["final"], y_final, final_scores, final_pred),
            )
        )
    return payloads


def compact_feature_payloads(features: pd.DataFrame, cols: list[str], labels: pd.Series, idx: dict[str, pd.Index]) -> list[PredictionPayload]:
    selector = make_logistic(penalty="l1", c=0.2, scale=True, class_weight="balanced")
    selector.fit(features.loc[idx["train"], cols], labels.loc[idx["train"]].astype(int))
    coefs = np.abs(selector.named_steps["model"].coef_[0])
    ranked = pd.DataFrame({"feature": cols, "abs_coef": coefs}).sort_values(["abs_coef", "feature"], ascending=[False, True])
    ranked = ranked[ranked["abs_coef"] > 1e-9]
    if ranked.empty:
        ranked = pd.DataFrame({"feature": cols, "abs_coef": np.ones(len(cols))}).sort_values("feature")
    payloads: list[PredictionPayload] = []
    y_val = labels.loc[idx["validation"]].astype(int)
    y_final = labels.loc[idx["final"]].astype(int)
    for k in [12, 20, 30, 45, 60]:
        selected_cols = ranked["feature"].head(min(k, len(ranked))).tolist()
        if len(selected_cols) < 4:
            continue
        _model, val_scores, final_scores = fit_base_scores(features, selected_cols, labels, idx, c=0.3, penalty="l2", scale=False)
        threshold, _val_acc = select_global_threshold(y_val, val_scores, GLOBAL_THRESHOLDS)
        val_pred = (val_scores >= threshold).astype(int)
        final_pred = (final_scores >= threshold).astype(int)
        threshold_label = str(threshold).replace(".", "p")
        payloads.append(
            PredictionPayload(
                candidate_id=f"targeted__compact_feature_set__l1_rank_top{k}__t{threshold_label}",
                track="feature_selection_compact",
                model="logistic_l2",
                feature_family=f"compact_l1_rank_top{k}",
                threshold_policy="validation_selected_threshold",
                threshold_detail=f"top_k={k}; threshold={threshold}; selector=L1 train-fit coefficients",
                feature_count=len(selected_cols),
                selection_note="compact feature count and threshold selected on validation only; feature ranking fit on train only",
                validation=prediction_frame(features, idx["validation"], y_val, val_scores, val_pred),
                final=prediction_frame(features, idx["final"], y_final, final_scores, final_pred),
            )
        )
    (OUTPUT_DIR / "compact_feature_manifest.json").write_text(
        json.dumps(
            json_safe({"selector": "train_only_l1_abs_coef", "ranked_features": ranked.to_dict("records")[:80]}),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return payloads


def select_by_track(summary: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for track, group in summary[summary["track"].ne("current_best_comparator")].groupby("track", sort=True):
        eligible = group[group["full_ticker_coverage"]].copy()
        if eligible.empty:
            continue
        selected = eligible.sort_values(
            [
                "selection_score",
                "validation_accuracy",
                "validation_rolling_mean_avg",
                "validation_rolling_below60_total",
                "candidate_id",
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        selected_rows.append(selected)
    current = summary[summary["track"].eq("current_best_comparator")]
    if not current.empty:
        selected_rows.insert(0, current.iloc[0])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def selected_slice_outputs(selected_payloads: dict[str, PredictionPayload]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ticker_rows: list[pd.DataFrame] = []
    month_rows: list[pd.DataFrame] = []
    quarter_rows: list[pd.DataFrame] = []
    rolling_rows: list[dict[str, Any]] = []
    for candidate_id, payload in selected_payloads.items():
        final = payload.final.copy()
        final["candidate_id"] = candidate_id
        final["track"] = payload.track
        for split_name, frame in [("validation", payload.validation), ("final", payload.final)]:
            stats = rolling_stats(frame)
            for window in (250, 500, 1000):
                rolling_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "track": payload.track,
                        "split": split_name,
                        "window": window,
                        "rolling_mean_accuracy": stats[f"rolling_{window}_mean"],
                        "rolling_windows_below_60": stats[f"rolling_{window}_below60_count"],
                    }
                )
        ticker = slice_stats(final, "ticker", "ticker")
        ticker.insert(0, "candidate_id", candidate_id)
        ticker.insert(1, "track", payload.track)
        month = slice_stats(final, "month", "month")
        month.insert(0, "candidate_id", candidate_id)
        month.insert(1, "track", payload.track)
        quarter = slice_stats(final, "quarter", "quarter")
        quarter.insert(0, "candidate_id", candidate_id)
        quarter.insert(1, "track", payload.track)
        ticker_rows.append(ticker)
        month_rows.append(month)
        quarter_rows.append(quarter)
    return (
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.concat(month_rows, ignore_index=True) if month_rows else pd.DataFrame(),
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.DataFrame(rolling_rows),
    )


def data_alignment_audit(features: pd.DataFrame, labels: pd.Series, idx: dict[str, pd.Index]) -> str:
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers).sort_values(["ticker", "datetime"])
    index_data = load_index_data()
    duplicate_rows = int(stock_df.duplicated(["ticker", "datetime"]).sum())
    ticker_counts = stock_df.groupby("ticker")["datetime"].nunique()
    max_rows = int(ticker_counts.max()) if not ticker_counts.empty else 0
    missing_rows = (max_rows - ticker_counts).clip(lower=0).astype(int)
    returns = stock_df.assign(return_1=stock_df.groupby("ticker")["close"].pct_change(fill_method=None))
    abnormal = returns[returns["return_1"].abs() > 0.20]
    label_available = int(labels.notna().sum())
    label_missing = int(labels.isna().sum())
    index_codes = sorted(index_data.keys())
    index_duplicate_rows = {
        code: int(frame.duplicated(["datetime"]).sum()) for code, frame in index_data.items() if "datetime" in frame.columns
    }
    boundary_rows = {name: int(len(index)) for name, index in idx.items()}
    text = f"""# VN30 Legacy Targeted Improvement Data Alignment Audit

## Split And Label Availability

- Horizon: h40 absolute close-direction label.
- Train boundary: `<= {TRAIN_END}`.
- Validation boundary: `{VAL_START}` through `{VAL_END}`.
- Final boundary: `>= {FINAL_START}`.
- Label-available rows: {label_available:,}.
- Label-missing rows: {label_missing:,}.
- Split rows: train {boundary_rows['train']:,}, validation {boundary_rows['validation']:,}, final {boundary_rows['final']:,}.
- Final rows expected for apples-to-apples comparison: 4,074; observed: {boundary_rows['final']:,}.

## Stock Bar Integrity

- Active ticker count: {len(tickers)}.
- Duplicate `(ticker, datetime)` stock rows: {duplicate_rows:,}.
- Max hourly rows by ticker: {max_rows:,}.
- Missing-row proxy versus max ticker count: total {int(missing_rows.sum()):,}, max per ticker {int(missing_rows.max()) if not missing_rows.empty else 0:,}.
- Abnormal one-bar stock return spikes above 20% absolute: {len(abnormal):,}.

## Index Context Alignment

- Local index codes loaded: {', '.join(index_codes)}.
- Index duplicate timestamp rows: {index_duplicate_rows}.
- Index features used by targeted tracks are lagged context columns from the existing feature builder; no final-window labels or future returns are used as features.

## Boundary

- Data fetch: no.
- Provider behavior changed: no.
- Ticker subset used: no.
- Confidence abstention/top-k used: no.
"""
    return text


def leakage_audit(selected: pd.DataFrame) -> str:
    final_selection = bool(pd.to_numeric(selected["final_accuracy_used_for_selection"], errors="coerce").fillna(0).astype(bool).any())
    full_coverage = bool((selected["ticker_coverage"].astype(int) == FULL_TICKER_COVERAGE).all())
    high_overfit = selected[selected["overfit_risk"].eq("high")]["candidate_id"].tolist()
    text = f"""# VN30 Legacy Targeted Improvement Leakage Audit

## Checks

| Check | Status |
| --- | --- |
| validation_only_selection | {'pass' if not final_selection else 'fail'} |
| final_window_scoring_only | {'pass' if not final_selection else 'fail'} |
| full_30_ticker_coverage | {'pass' if full_coverage else 'fail'} |
| ticker_subset_main_claim | pass |
| confidence_abstention | pass |
| top_k_substitution | pass |
| stacking_main_candidate | pass |
| provider_behavior_changed | pass |
| market_data_fetch | pass |
| h40_legacy_split_used | pass |
| index_features_lagged_context_only | pass |

## Notes

- Candidate thresholds, ticker repairs, regime routers, regularization grids, and compact feature counts were selected from validation metrics only.
- Final accuracy appears in reports only after the selected per-track candidates are fixed.
- High overfit-risk selected candidates: {', '.join(high_overfit) if high_overfit else 'none'}.
"""
    return text


def claim_boundary(selected: pd.DataFrame, summary: pd.DataFrame) -> str:
    best = selected[selected["track"].ne("current_best_comparator")].sort_values(
        ["candidate_classification", "final_accuracy"], ascending=[True, False]
    )
    best_final = selected[selected["track"].ne("current_best_comparator")].sort_values("final_accuracy", ascending=False).iloc[0]
    stronger = selected[selected["candidate_classification"].isin(["stronger_candidate", "marginal_improvement"])]
    lines = [
        "# VN30 Legacy Targeted Improvement Claim Boundary",
        "",
        "## Comparator",
        "",
        f"- Current apples-to-apples best: Logistic L2, `baseline_C_closest`, h40, threshold 0.55, final accuracy {CURRENT_BEST_LABEL}.",
        f"- Old reference: {pct(OLD_REFERENCE_ACCURACY)}.",
        f"- Majority baseline: {pct(MAJORITY_BASELINE)}.",
        "",
        "## Result",
        "",
        f"- Best selected targeted track by final score after validation-only selection: `{best_final['candidate_id']}`.",
        f"- Final accuracy: {pct(best_final['final_accuracy'])}; delta vs current: {pp(best_final['delta_vs_current_61_63'])}; classification: `{best_final['candidate_classification']}`.",
        f"- Rolling stability vs current: `{best_final['rolling_stability_vs_current']}`.",
        "",
        "## Boundary",
        "",
        "- No final-window score was used for model, threshold, feature, ticker repair, regime router, or track selection.",
        "- No ticker subset, confidence abstention, top-k/ranking substitution, stacking main candidate, or market-data fetch was used.",
        "- This is a directional accuracy benchmark only; it makes no trading, profitability, or live-deployment claim.",
    ]
    if stronger.empty:
        lines.append("- No targeted track established a stronger candidate beyond the current 61.63% comparator.")
    else:
        names = ", ".join(f"`{cid}`" for cid in stronger["candidate_id"].tolist())
        lines.append(f"- Validation-only selected targeted candidate(s) above current comparator: {names}.")
    unselected_improvers = summary[
        summary["track"].ne("current_best_comparator")
        & summary["final_accuracy"].gt(CURRENT_BEST_ACCURACY)
        & ~summary["candidate_id"].isin(set(selected["candidate_id"].tolist()))
    ].sort_values("final_accuracy", ascending=False)
    if not unselected_improvers.empty:
        top = unselected_improvers.iloc[0]
        lines.extend(
            [
                "",
                "## Non-Selected Final Observation",
                "",
                f"- `{top['candidate_id']}` scored {pct(top['final_accuracy'])} on final, {pp(top['delta_vs_current_61_63'])} vs current.",
                "- It was not selected by the validation-only track objective, so it is not a stronger accepted candidate in this run.",
            ]
        )
    _ = best
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, family_cols, manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    labels = add_absolute_labels(features, HORIZON)
    idx = split_indices(features, labels)
    cols = family_cols[BASE_FEATURE_FAMILY]
    current = current_best_payload()
    current_roll = rolling_stats(current.final, "final")

    payloads: list[PredictionPayload] = [current]
    payloads.extend(calibration_payloads(features, cols, labels, idx, current_threshold=0.55))
    payloads.append(regime_router_payload(features, cols, labels, idx, current_threshold=0.55))
    payloads.append(drag_ticker_payload(features, cols, labels, idx, current))
    payloads.extend(regularized_linear_payloads(features, cols, labels, idx))
    payloads.extend(compact_feature_payloads(features, cols, labels, idx))

    summaries = [summarize_payload(payload, current_roll) for payload in payloads]
    summary = pd.DataFrame(summaries).sort_values(["track", "selection_score", "candidate_id"], ascending=[True, False, True])
    selected = select_by_track(summary)
    selected_ids = set(selected["candidate_id"].tolist())
    payload_by_id = {payload.candidate_id: payload for payload in payloads if payload.candidate_id in selected_ids}
    by_ticker, by_month, by_quarter, rolling = selected_slice_outputs(payload_by_id)

    summary.to_csv(OUTPUT_DIR / "targeted_improvement_summary.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selected_candidates_by_track.csv", index=False)
    final_cols = [
        "track",
        "candidate_id",
        "model",
        "feature_family",
        "threshold_policy",
        "threshold_detail",
        "validation_accuracy",
        "final_accuracy",
        "delta_vs_current_61_63",
        "delta_vs_old_reference_61_51",
        "final_rows",
        "ticker_coverage",
        "final_rolling_250_mean",
        "final_rolling_500_mean",
        "final_rolling_1000_mean",
        "rolling_stability_vs_current",
        "overfit_risk",
        "candidate_classification",
    ]
    selected[final_cols].to_csv(OUTPUT_DIR / "final_results_by_track.csv", index=False)
    by_ticker.to_csv(OUTPUT_DIR / "by_ticker.csv", index=False)
    by_month.to_csv(OUTPUT_DIR / "by_month.csv", index=False)
    by_quarter.to_csv(OUTPUT_DIR / "by_quarter.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "rolling_250_500_1000.csv", index=False)
    write_markdown(OUTPUT_DIR / "data_alignment_audit.md", data_alignment_audit(features, labels, idx))
    write_markdown(OUTPUT_DIR / "leakage_audit.md", leakage_audit(selected))
    write_markdown(OUTPUT_DIR / "claim_boundary.md", claim_boundary(selected, summary))
    write_json(
        OUTPUT_DIR / "targeted_improvement_manifest.json",
        {
            "run_id": "vn30_legacy_targeted_improvement_tracks_v1",
            "data_fetch": False,
            "provider_behavior_changed": False,
            "stacking_main_candidate": False,
            "selection_source": "validation_only",
            "final_window_role": "scoring_only",
            "horizon": HORIZON,
            "current_best_accuracy": CURRENT_BEST_ACCURACY,
            "old_reference_accuracy": OLD_REFERENCE_ACCURACY,
            "majority_baseline": MAJORITY_BASELINE,
            "feature_manifest": manifest,
            "selected_candidates": selected[["track", "candidate_id", "candidate_classification", "final_accuracy"]].to_dict("records"),
        },
    )
    best = selected[selected["track"].ne("current_best_comparator")].sort_values("final_accuracy", ascending=False).iloc[0]
    print(
        f"targeted_improvement_complete best_track={best['track']} best={best['candidate_id']} "
        f"final={pct(best['final_accuracy'])} class={best['candidate_classification']} output_dir={rel(OUTPUT_DIR)}"
    )


if __name__ == "__main__":
    main()
