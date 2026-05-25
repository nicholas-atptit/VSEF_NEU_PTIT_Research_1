"""Run VN30 V8 strategy target redesign diagnostics.

This runner redesigns the strategy target away from absolute-direction ranking
and toward market-relative, top-quantile, cost-adjusted, and neutral-removed
stock selection targets. It uses staged validation screening and keeps final
rankings exploratory.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_full_model_resurrection_index_pretrain import (  # noqa: E402
    FINAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    as_float,
    clean_feature_matrix,
    now_utc,
    pct,
    pp,
    predict_probability,
    rel,
    split_indices,
    write_frame,
    write_json,
    write_markdown,
)
from scripts.research.run_vn30_full_model_tuning_v3 import (  # noqa: E402
    build_v3_feature_frame,
    index_forward_return,
    make_model,
    stock_forward_return_and_timestamp,
)
from scripts.research.run_vn30_v4_promotion_queue_strategy import (  # noqa: E402
    baseline_buy_hold_index,
    baseline_equal_weight_basket,
    metrics_from_equity,
)
from scripts.research.run_vn30_v5_strategy_relock_risk_hardening import add_market_drawdown_feature  # noqa: E402
from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, load_index_data  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_v8_strategy_target_redesign"
PROTOCOL_PATH = REPO_ROOT / "reports" / "protocols" / "VN30_V8_STRATEGY_TARGET_REDESIGN_PROTOCOL.md"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_V8_STRATEGY_TARGET_REDESIGN_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_V8_STRATEGY_TARGET_REDESIGN_CLAIM_BOUNDARY.md"

SEED = 42
HORIZONS = [20, 30, 40, 50, 60]
COST_BPS = [0, 5, 10, 20, 30]
SLIPPAGE_BPS = [0, 5, 10, 20]
MAX_POSITIONS = [3, 5, 10]
MAX_EXPOSURE = [0.5, 0.7, 1.0]
VALIDATION_COST_BPS = 10
VALIDATION_SLIPPAGE_BPS = 5
V7_LOCKED_FINAL_RETURN = 0.22354893587697267


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def target_specs() -> list[dict[str, Any]]:
    specs = [
        {"target_variant": "market_relative_vn30", "family": "market_relative", "definition": "stock_forward_return_h > vn30_index_forward_return_h"},
        {"target_variant": "market_relative_vnindex", "family": "market_relative", "definition": "stock_forward_return_h > vnindex_forward_return_h"},
        {"target_variant": "cost_adjusted_direction_c0010", "family": "cost_adjusted_direction", "cost_threshold": 0.0010, "definition": "stock_forward_return_h > 0.0010 estimated roundtrip cost"},
    ]
    for q in [0.10, 0.20, 0.30]:
        specs.append(
            {
                "target_variant": f"top_quantile_forward_return_q{int(q * 100):02d}",
                "family": "top_quantile_forward_return",
                "top_quantile": q,
                "definition": f"stock is in top {int(q * 100)}% forward return among VN30 rows at same timestamp",
            }
        )
    for band in [0.001, 0.002, 0.005, 0.010]:
        specs.append(
            {
                "target_variant": f"neutral_removed_direction_b{int(band * 10000):04d}",
                "family": "neutral_removed_direction",
                "neutral_band": band,
                "definition": f"stock_forward_return_h > 0 after excluding abs(return) < {band}",
            }
        )
    return specs


def enrich_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for col in ["momentum_20", "momentum_10", "momentum_5", "return_1_lag_5", "relative_strength_vs_market_20"]:
        if col in out.columns:
            out[f"{col}_cs_rank"] = pd.to_numeric(out[col], errors="coerce").groupby(out["datetime"]).rank(pct=True)
    if "return_1_lag_1" in out.columns and "market_return_lag1" in out.columns:
        out["stock_minus_market_lag1"] = pd.to_numeric(out["return_1_lag_1"], errors="coerce") - pd.to_numeric(out["market_return_lag1"], errors="coerce")
    if "return_1_lag_5" in out.columns and "market_return_lag5" in out.columns:
        out["stock_minus_market_lag5"] = pd.to_numeric(out["return_1_lag_5"], errors="coerce") - pd.to_numeric(out["market_return_lag5"], errors="coerce")
    return out


def existing_cols(features: pd.DataFrame, candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    cols: list[str] = []
    for col in candidates:
        if col in features.columns and col not in seen:
            cols.append(col)
            seen.add(col)
    return cols


def build_feature_groups(features: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    groups: dict[str, list[str]] = {}
    groups["stock_lag_momentum"] = existing_cols(
        features,
        [
            "return_1_lag_1",
            "return_1_lag_3",
            "return_1_lag_5",
            "return_1_lag_10",
            "return_1_lag_20",
            "rolling_return_mean_5",
            "rolling_return_mean_10",
            "rolling_return_mean_20",
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "momentum_20_cs_rank",
            "return_1_lag_5_cs_rank",
        ],
    )
    groups["relative_strength"] = existing_cols(
        features,
        [
            "relative_strength_vs_market_lag1",
            "relative_strength_vs_market_5",
            "relative_strength_vs_market_20",
            "relative_strength_vs_vn30_lag1",
            "relative_strength_vs_vn30_5",
            "relative_strength_vs_vn30_20",
            "relative_strength_vs_vnindex_lag1",
            "relative_strength_vs_vnindex_5",
            "relative_strength_vs_vnindex_20",
            "stock_minus_market_lag1",
            "stock_minus_market_lag5",
            "relative_strength_vs_market_20_cs_rank",
        ],
    )
    groups["market_context"] = existing_cols(
        features,
        [
            "vn30_ctx_return_lag1",
            "vn30_ctx_return_lag5",
            "vnindex_ctx_return_lag1",
            "vnindex_ctx_return_lag5",
            "market_return_lag1",
            "market_return_lag5",
            "market_volatility_5",
            "market_volatility_20",
            "market_direction_lag1",
            "index_agreement_score",
            "cross_index_agreement_score",
            "risk_on_risk_off_state",
            "sideway_flag",
            "high_volatility_flag",
            "low_volatility_flag",
        ],
    )
    groups["volume_volatility"] = existing_cols(
        features,
        [
            "volume_shock_20",
            "volume_change_1",
            "stock_volume_zscore_5_lag",
            "stock_volume_zscore_10_lag",
            "stock_volume_zscore_20_lag",
            "stock_volume_shock_5_lag",
            "stock_volume_shock_10_lag",
            "stock_volume_shock_20_lag",
            "high_low_range",
            "roll_vol_8",
            "roll_vol_20",
            "roll_vol_40",
            "rolling_return_vol_5",
            "rolling_return_vol_10",
            "rolling_return_vol_20",
            "stock_atr_proxy_5_lag",
            "stock_atr_proxy_10_lag",
            "stock_atr_proxy_20_lag",
        ],
    )
    combined: list[str] = []
    for name in ["stock_lag_momentum", "relative_strength", "market_context", "volume_volatility"]:
        combined.extend(groups[name])
    groups["combined_strategy_features"] = list(dict.fromkeys(combined))
    audit = pd.DataFrame(
        [
            {
                "feature_group": name,
                "feature_count": len(cols),
                "columns": compact_json({"columns": cols}),
                "timestamp_safe": True,
            }
            for name, cols in groups.items()
        ]
    )
    return groups, audit


def build_labels(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], horizon: int, spec: dict[str, Any]) -> pd.Series:
    stock_return, target_ts = stock_forward_return_and_timestamp(features, horizon)
    family = str(spec["family"])
    if family == "market_relative":
        code = "VN30" if spec["target_variant"] == "market_relative_vn30" else "VNINDEX"
        market_return = index_forward_return(features, index_data, code, target_ts)
        labels = (stock_return > market_return).astype(float)
        labels.loc[market_return.isna()] = np.nan
    elif family == "top_quantile_forward_return":
        q = float(spec["top_quantile"])
        rank_pct = stock_return.groupby(features["datetime"]).rank(method="first", pct=True, ascending=False)
        labels = (rank_pct <= q).astype(float)
        labels.loc[rank_pct.isna()] = np.nan
    elif family == "cost_adjusted_direction":
        threshold = float(spec["cost_threshold"])
        labels = (stock_return > threshold).astype(float)
    elif family == "neutral_removed_direction":
        band = float(spec["neutral_band"])
        labels = (stock_return > 0.0).astype(float)
        labels.loc[stock_return.abs() < band] = np.nan
    else:
        raise ValueError(f"unknown target family {family}")
    labels.loc[stock_return.isna() | target_ts.isna()] = np.nan
    labels = pd.Series(labels.to_numpy(dtype=float), index=features.index)
    labels.attrs["target_timestamp"] = target_ts.reindex(features.index)
    labels.attrs["horizon"] = int(horizon)
    labels.attrs["target_variant"] = spec["target_variant"]
    labels.attrs["target_definition"] = spec["definition"]
    labels.attrs["label_cutoff_rule"] = "strict target_timestamp split discipline"
    return labels


def target_audit(features: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[tuple[str, int], tuple[pd.Series, dict[str, pd.Index]]]]:
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, int], tuple[pd.Series, dict[str, pd.Index]]] = {}
    for spec in target_specs():
        for horizon in HORIZONS:
            labels = build_labels(features, index_data, horizon, spec)
            splits = split_indices(features, labels)
            cache[(str(spec["target_variant"]), horizon)] = (labels, splits)
            rows.append(
                {
                    "target_variant": spec["target_variant"],
                    "target_family": spec["family"],
                    "horizon": horizon,
                    "definition": spec["definition"],
                    "rows_total": int(labels.notna().sum()),
                    "train_rows": int(len(splits["train"])),
                    "validation_rows": int(len(splits["validation"])),
                    "final_rows": int(len(splits["final"])),
                    "validation_positive_rate": float(labels.reindex(splits["validation"]).mean()) if len(splits["validation"]) else math.nan,
                    "final_positive_rate": float(labels.reindex(splits["final"]).mean()) if len(splits["final"]) else math.nan,
                    "split_guard_passed": True,
                }
            )
    return pd.DataFrame(rows), cache


def model_params(model_family: str) -> dict[str, Any]:
    params = {
        "logistic_regression": {"model_family": "logistic_regression", "penalty": "l2", "solver": "liblinear", "C": 0.3, "class_weight": "balanced"},
        "calibrated_logistic": {"model_family": "calibrated_logistic", "solver": "liblinear", "C": 0.3, "class_weight": "balanced"},
        "elasticnet_logistic": {"model_family": "elasticnet_logistic", "penalty": "elasticnet", "solver": "saga", "C": 0.1, "class_weight": "balanced", "l1_ratio": 0.5},
        "random_forest": {"model_family": "random_forest", "n_estimators": 100, "max_depth": 5, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": "balanced"},
        "extra_trees": {"model_family": "extra_trees", "n_estimators": 150, "max_depth": 5, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": "balanced"},
        "xgboost": {"model_family": "xgboost", "n_estimators": 80, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 10, "min_child_weight": 10},
        "lightgbm": {"model_family": "lightgbm", "n_estimators": 100, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 50, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 10},
    }
    return params[model_family]


def model_registry(feature_groups: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seq = 0
    specs = target_specs()
    for spec, horizon, feature_group in itertools.product(specs, HORIZONS, feature_groups.keys()):
        seq += 1
        family = "logistic_regression"
        rows.append(
            {
                "model_candidate_id": f"v8model_{seq:05d}",
                "screen_stage": "broad_logistic_target_feature_horizon_screen",
                "target_variant": spec["target_variant"],
                "target_family": spec["family"],
                "horizon": horizon,
                "feature_group": feature_group,
                "model_family": family,
                "model_params": compact_json(model_params(family)),
            }
        )
    for spec, model_family in itertools.product(specs, ["logistic_regression", "calibrated_logistic", "elasticnet_logistic", "random_forest", "extra_trees", "xgboost", "lightgbm"]):
        seq += 1
        rows.append(
            {
                "model_candidate_id": f"v8model_{seq:05d}",
                "screen_stage": "model_family_sweep_h40_combined",
                "target_variant": spec["target_variant"],
                "target_family": spec["family"],
                "horizon": 40,
                "feature_group": "combined_strategy_features",
                "model_family": model_family,
                "model_params": compact_json(model_params(model_family)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["target_variant", "horizon", "feature_group", "model_family", "model_params"]).reset_index(drop=True)


def add_rank_columns(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = split
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["target_timestamp"] = pd.to_datetime(out["target_timestamp"], errors="coerce")
    out["model_score"] = pd.to_numeric(out["model_score"], errors="coerce")
    rs_col = "relative_strength_vs_market_20" if "relative_strength_vs_market_20" in out.columns else "relative_strength_vs_market_lag1"
    out["relative_strength_score"] = pd.to_numeric(out.get(rs_col, np.nan), errors="coerce")
    out["model_score_pct_rank"] = out.groupby("datetime")["model_score"].rank(method="average", pct=True)
    out["relative_strength_pct_rank"] = out.groupby("datetime")["relative_strength_score"].rank(method="average", pct=True)
    out["score_decile"] = np.ceil(out["model_score_pct_rank"].fillna(0.0) * 10.0).clip(1, 10).astype(int)
    return out


def signal_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    model_score: np.ndarray,
    split: str,
    candidate: dict[str, Any],
) -> pd.DataFrame:
    stock_return, stock_target_ts = stock_forward_return_and_timestamp(features, int(candidate["horizon"]))
    cols = [
        "feature_timestamp",
        "datetime",
        "ticker",
        "close",
        "momentum_20",
        "relative_strength_vs_market_lag1",
        "relative_strength_vs_market_20",
        "risk_on_risk_off_state",
        "market_momentum_20",
    ]
    frame = features.loc[idx, [col for col in cols if col in features.columns]].copy()
    frame["target_timestamp"] = labels.attrs["target_timestamp"].reindex(idx).to_numpy()
    frame["absolute_target_timestamp"] = stock_target_ts.reindex(idx).to_numpy()
    frame["y_true"] = labels.reindex(idx).astype(int).to_numpy()
    frame["model_score"] = np.asarray(model_score, dtype=float)
    frame["stock_forward_return"] = stock_return.reindex(idx).to_numpy(dtype=float)
    frame["split"] = split
    frame["model_candidate_id"] = candidate["model_candidate_id"]
    frame["target_variant"] = candidate["target_variant"]
    frame["target_family"] = candidate["target_family"]
    frame["horizon"] = int(candidate["horizon"])
    frame["feature_group"] = candidate["feature_group"]
    frame["model_family"] = candidate["model_family"]
    return add_rank_columns(frame.dropna(subset=["target_timestamp", "stock_forward_return", "model_score"]), split)


def fit_validation_models(
    features: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    registry: pd.DataFrame,
    label_cache: dict[tuple[str, int], tuple[pd.Series, dict[str, pd.Index]]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for candidate in registry.to_dict("records"):
        labels, splits = label_cache[(str(candidate["target_variant"]), int(candidate["horizon"]))]
        train_idx = splits["train"]
        val_idx = splits["validation"]
        feature_cols = feature_groups[str(candidate["feature_group"])]
        if not feature_cols or len(train_idx) < 100 or len(val_idx) < 100:
            rows.append({**candidate, "status": "failed", "failure_reason": "empty feature columns or insufficient rows"})
            continue
        train_y = labels.reindex(train_idx).astype(int)
        if train_y.nunique() < 2:
            rows.append({**candidate, "status": "failed", "failure_reason": "single-class train labels"})
            continue
        model = make_model(str(candidate["model_family"]), json.loads(str(candidate["model_params"])))
        if model is None:
            rows.append({**candidate, "status": "failed", "failure_reason": "model unavailable"})
            continue
        try:
            model.fit(clean_feature_matrix(features.loc[train_idx], feature_cols), train_y)
            val_prob = predict_probability(model, clean_feature_matrix(features.loc[val_idx], feature_cols))
            val_frame = signal_frame(features, val_idx, labels, val_prob, "validation", candidate)
            corr = val_frame[["model_score_pct_rank", "stock_forward_return"]].corr(method="spearman").iloc[0, 1] if len(val_frame) > 1 else math.nan
            decile_mean = val_frame.groupby("score_decile")["stock_forward_return"].mean()
            top_minus_bottom = as_float(decile_mean.get(10, math.nan)) - as_float(decile_mean.get(1, math.nan))
            pred = (pd.to_numeric(val_frame["model_score"], errors="coerce") >= 0.5).astype(int)
            acc = float((pred.to_numpy() == val_frame["y_true"].astype(int).to_numpy()).mean()) if len(val_frame) else math.nan
            validation_model_score = 0.40 * (corr if pd.notna(corr) else 0.0) + 0.25 * (top_minus_bottom if pd.notna(top_minus_bottom) else 0.0) + 0.20 * (acc if pd.notna(acc) else 0.0) + 0.15 * min(1.0, len(val_frame) / 3500.0)
            rows.append(
                {
                    **candidate,
                    "status": "fit",
                    "validation_rows": int(len(val_frame)),
                    "validation_accuracy_at_0p5": acc,
                    "validation_spearman_rank_return": float(corr) if pd.notna(corr) else math.nan,
                    "validation_top_decile_minus_bottom": top_minus_bottom,
                    "validation_score": validation_model_score,
                    "prediction_mean": float(pd.to_numeric(val_frame["model_score"], errors="coerce").mean()),
                    "prediction_max": float(pd.to_numeric(val_frame["model_score"], errors="coerce").max()),
                }
            )
            payloads[str(candidate["model_candidate_id"])] = {
                "candidate": candidate,
                "model": model,
                "feature_cols": feature_cols,
                "labels": labels,
                "splits": splits,
                "validation_frame": val_frame,
            }
        except Exception as exc:  # pragma: no cover - diagnostic artifact path
            rows.append({**candidate, "status": "failed", "failure_reason": str(exc)[:500]})
    return pd.DataFrame(rows), payloads


def shortlist_models(model_results: pd.DataFrame) -> list[str]:
    fitted = model_results[model_results["status"].eq("fit")].copy()
    if fitted.empty:
        return []
    selected: list[str] = []
    for frame in [
        fitted.sort_values("validation_score", ascending=False).head(8),
        fitted.sort_values("validation_spearman_rank_return", ascending=False).groupby("target_variant", as_index=False).head(1),
        fitted.sort_values("validation_score", ascending=False).groupby("model_family", as_index=False).head(1),
    ]:
        for cid in frame["model_candidate_id"].astype(str).tolist():
            if cid not in selected:
                selected.append(cid)
    return selected[:14]


def strategy_family_key(row: dict[str, Any]) -> str:
    return (
        f"{row['model_candidate_id']}__{row['strategy_template']}__n{row.get('top_n', '')}"
        f"__q{row.get('top_quantile', '')}__wm{row.get('w_model', '')}__wrs{row.get('w_rs', '')}"
        f"__mp{row['max_positions']}__ex{row['max_exposure']}"
    ).replace(".", "p")


def strategy_grid(shortlist: list[str], payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seq = 0

    def add(model_id: str, template: str, max_positions: int, max_exposure: float, **params: Any) -> None:
        nonlocal seq
        candidate = payloads[model_id]["candidate"]
        seq += 1
        row = {
            "strategy_id": f"v8strat_{seq:05d}",
            "model_candidate_id": model_id,
            "model_family": candidate["model_family"],
            "target_variant": candidate["target_variant"],
            "target_family": candidate["target_family"],
            "feature_group": candidate["feature_group"],
            "horizon": int(candidate["horizon"]),
            "strategy_template": template,
            "top_n": params.get("top_n", ""),
            "top_quantile": params.get("top_quantile", ""),
            "w_model": params.get("w_model", ""),
            "w_rs": params.get("w_rs", ""),
            "market_regime_filter": params.get("market_regime_filter", "off"),
            "max_positions": int(max_positions),
            "max_exposure": float(max_exposure),
            "cost_bps": VALIDATION_COST_BPS,
            "slippage_bps": VALIDATION_SLIPPAGE_BPS,
        }
        row["strategy_family_key"] = strategy_family_key(row)
        rows.append(row)

    for model_id in shortlist:
        for n, exposure in itertools.product([3, 5, 10], MAX_EXPOSURE):
            add(model_id, "top_n_rotation", n, exposure, top_n=n)
        for q, n in itertools.product([0.10, 0.20, 0.30], [3, 5, 10]):
            add(model_id, "top_quantile_rotation", n, 0.7, top_quantile=q)
        for (w_model, w_rs), n in itertools.product([(1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5)], [3, 5, 10]):
            add(model_id, "relative_strength_model_blend", n, 0.7, top_n=n, w_model=w_model, w_rs=w_rs)
        for n, exposure in itertools.product([3, 5, 10], MAX_EXPOSURE):
            add(model_id, "regime_filtered_rotation", n, exposure, top_n=n, market_regime_filter="on")
        for n in [3, 5, 10]:
            add(model_id, "benchmark_relative_rotation", n, 0.7, top_n=n)
    return pd.DataFrame(rows)


def select_rows(group: pd.DataFrame, variant: dict[str, Any]) -> pd.DataFrame:
    work = group.copy()
    template = str(variant["strategy_template"])
    if template == "regime_filtered_rotation":
        risk = pd.to_numeric(work.get("risk_on_risk_off_state", 0.0), errors="coerce").fillna(0.0)
        momentum = pd.to_numeric(work.get("market_momentum_20", 0.0), errors="coerce").fillna(0.0)
        work = work[(risk >= 0.0) & (momentum >= 0.0)].copy()
        template = "top_n_rotation"
    if work.empty:
        return work
    if template in {"top_n_rotation", "benchmark_relative_rotation"}:
        work["selection_score"] = pd.to_numeric(work["model_score_pct_rank"], errors="coerce")
        selected = work.sort_values(["selection_score", "model_score"], ascending=[False, False]).head(int(variant["top_n"]))
    elif template == "top_quantile_rotation":
        work["selection_score"] = pd.to_numeric(work["model_score_pct_rank"], errors="coerce")
        selected = work[work["selection_score"] >= 1.0 - float(variant["top_quantile"])].sort_values(["selection_score", "model_score"], ascending=[False, False])
    elif template == "relative_strength_model_blend":
        model_rank = pd.to_numeric(work["model_score_pct_rank"], errors="coerce").fillna(0.5)
        rs_rank = pd.to_numeric(work["relative_strength_pct_rank"], errors="coerce").fillna(0.5)
        work["selection_score"] = float(variant["w_model"]) * model_rank + float(variant["w_rs"]) * rs_rank
        selected = work.sort_values(["selection_score", "model_score"], ascending=[False, False]).head(int(variant["top_n"]))
    else:
        selected = work.iloc[0:0].copy()
    return selected.head(int(variant["max_positions"]))


def simulate_strategy(signal_rows: pd.DataFrame, variant: dict[str, Any], split: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows = signal_rows.copy()
    rows["datetime"] = pd.to_datetime(rows["datetime"], errors="coerce")
    rows["target_timestamp"] = pd.to_datetime(rows["target_timestamp"], errors="coerce")
    rows = rows.dropna(subset=["datetime", "target_timestamp", "stock_forward_return", "model_score"])
    if rows.empty:
        curve = pd.DataFrame([{"datetime": pd.NaT, "equity": 1.0, "cash": 1.0, "invested": 0.0, "open_positions": 0}])
        return metrics_from_equity(curve, pd.DataFrame(), [], 0.0), curve, pd.DataFrame()
    max_positions = int(variant["max_positions"])
    max_exposure = float(variant["max_exposure"])
    drag = 2.0 * (float(variant["cost_bps"]) + float(variant["slippage_bps"])) / 10000.0
    cash = 1.0
    open_positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exposure_values: list[float] = []
    turnover = 0.0
    for timestamp, group in rows.groupby("datetime", sort=True):
        still_open: list[dict[str, Any]] = []
        for pos in open_positions:
            if pos["exit_time"] <= timestamp:
                exit_value = pos["entry_value"] * (1.0 + pos["net_return"])
                cash += exit_value
                turnover += exit_value
                trades.append({**pos, "exit_value": exit_value, "realized_at": timestamp})
            else:
                still_open.append(pos)
        open_positions = still_open
        invested = sum(pos["entry_value"] for pos in open_positions)
        equity = cash + invested
        slots = max(0, max_positions - len(open_positions))
        if slots > 0 and cash > 1e-12 and equity > 0:
            selected = select_rows(group[~group["ticker"].isin({pos["ticker"] for pos in open_positions})], variant).head(slots)
            exposure_room = max(0.0, max_exposure - invested / equity) * equity
            allocatable = min(cash, exposure_room)
            if allocatable > 1e-12 and not selected.empty:
                slot_value = min(allocatable / len(selected), equity * max_exposure / max_positions)
                for _, row in selected.iterrows():
                    entry_value = min(slot_value, cash)
                    if entry_value <= 1e-12:
                        continue
                    cash -= entry_value
                    turnover += entry_value
                    raw_return = float(row["stock_forward_return"])
                    net_return = raw_return - drag
                    open_positions.append(
                        {
                            "strategy_id": variant["strategy_id"],
                            "split": split,
                            "ticker": row["ticker"],
                            "entry_time": timestamp,
                            "exit_time": row["target_timestamp"],
                            "entry_value": entry_value,
                            "raw_return": raw_return,
                            "net_return": net_return,
                            "model_score": row["model_score"],
                            "selection_score": row.get("selection_score", row.get("model_score_pct_rank", math.nan)),
                            "strategy_template": variant["strategy_template"],
                            "cost_bps": variant["cost_bps"],
                            "slippage_bps": variant["slippage_bps"],
                        }
                    )
        invested = sum(pos["entry_value"] for pos in open_positions)
        equity = cash + invested
        exposure_values.append(invested / equity if equity > 0 else 0.0)
        curve_rows.append({"datetime": timestamp, "equity": equity, "cash": cash, "invested": invested, "open_positions": len(open_positions)})
    for pos in sorted(open_positions, key=lambda item: item["exit_time"]):
        exit_value = pos["entry_value"] * (1.0 + pos["net_return"])
        cash += exit_value
        turnover += exit_value
        trades.append({**pos, "exit_value": exit_value, "realized_at": pos["exit_time"]})
        curve_rows.append({"datetime": pos["exit_time"], "equity": cash, "cash": cash, "invested": 0.0, "open_positions": 0})
    curve = pd.DataFrame(curve_rows).sort_values("datetime").reset_index(drop=True)
    trade_log = pd.DataFrame(trades)
    return metrics_from_equity(curve, trade_log, exposure_values, turnover), curve, trade_log


def run_strategies(grid: pd.DataFrame, frames: dict[str, pd.DataFrame], split: str, keep_strategy_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    for variant in grid.to_dict("records"):
        frame = frames.get(str(variant["model_candidate_id"]), pd.DataFrame())
        metrics, curve, trade_log = simulate_strategy(frame, variant, split)
        rows.append(
            {
                **variant,
                "split": split,
                **metrics,
                "claim_label": "diagnostic_only" if split == "validation" else "exploratory_not_claimable",
                "reason_not_claimable": "validation strategy diagnostic" if split == "validation" else "final-ranked row is exploratory",
            }
        )
        if keep_strategy_id is not None and str(variant["strategy_id"]) == str(keep_strategy_id):
            curve = curve.copy()
            curve.insert(0, "strategy_id", variant["strategy_id"])
            curves.append(curve)
            if not trade_log.empty:
                trade_log = trade_log.copy()
                trade_log["strategy_id"] = variant["strategy_id"]
                trades.append(trade_log)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(), pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()


def baseline_signal(features: pd.DataFrame, horizon: int, split: str, baseline_name: str, splits: dict[str, pd.Index]) -> pd.DataFrame:
    idx = splits[split]
    stock_return, target_ts = stock_forward_return_and_timestamp(features, horizon)
    frame = features.loc[idx, ["feature_timestamp", "datetime", "ticker", "close"]].copy()
    frame["target_timestamp"] = target_ts.reindex(idx).to_numpy()
    frame["stock_forward_return"] = stock_return.reindex(idx).to_numpy(dtype=float)
    if baseline_name == "simple_momentum":
        frame["model_score"] = pd.to_numeric(features.loc[idx, "momentum_20"], errors="coerce")
    elif baseline_name == "simple_relative_strength":
        col = "relative_strength_vs_market_20" if "relative_strength_vs_market_20" in features.columns else "relative_strength_vs_market_lag1"
        frame["model_score"] = pd.to_numeric(features.loc[idx, col], errors="coerce")
    elif baseline_name == "random_signal_same_turnover":
        rng = np.random.default_rng(SEED + horizon + (0 if split == "validation" else 1000))
        frame["model_score"] = rng.random(len(frame))
    else:
        frame["model_score"] = 1.0
    frame["relative_strength_score"] = pd.to_numeric(features.loc[idx, "relative_strength_vs_market_20"], errors="coerce") if "relative_strength_vs_market_20" in features.columns else np.nan
    return add_rank_columns(frame.dropna(subset=["target_timestamp", "stock_forward_return", "model_score"]), split)


def baseline_comparison(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], label_cache: dict[tuple[str, int], tuple[pd.Series, dict[str, pd.Index]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        # Reuse any target split for the same horizon because strict split boundaries depend on timestamps.
        splits = next(s for (target, h), (_labels, s) in label_cache.items() if h == horizon)
        for split in ["validation", "final"]:
            for cost_bps, slippage_bps in itertools.product(COST_BPS, SLIPPAGE_BPS):
                rows.append({"baseline_name": "cash_no_trade", "split": split, "horizon": horizon, "max_positions": math.nan, "cost_bps": cost_bps, "slippage_bps": slippage_bps, "total_return": 0.0})
                rows.append({**baseline_buy_hold_index(index_data, split, cost_bps, slippage_bps), "horizon": horizon, "max_positions": math.nan, "cost_bps": cost_bps, "slippage_bps": slippage_bps})
                rows.append({**baseline_equal_weight_basket(features, split, cost_bps, slippage_bps), "horizon": horizon, "max_positions": math.nan, "cost_bps": cost_bps, "slippage_bps": slippage_bps})
                for max_positions in MAX_POSITIONS:
                    for baseline_name in ["simple_momentum", "simple_relative_strength", "random_signal_same_turnover", "always_up_directional_baseline"]:
                        frame = baseline_signal(features, horizon, split, baseline_name, splits)
                        variant = {
                            "strategy_id": f"baseline_{baseline_name}",
                            "strategy_template": "top_n_rotation",
                            "top_n": max_positions,
                            "top_quantile": "",
                            "w_model": "",
                            "w_rs": "",
                            "max_positions": max_positions,
                            "max_exposure": 1.0,
                            "cost_bps": cost_bps,
                            "slippage_bps": slippage_bps,
                        }
                        metrics, _curve, _trades = simulate_strategy(frame, variant, split)
                        rows.append({"baseline_name": baseline_name, "split": split, "horizon": horizon, "max_positions": max_positions, "cost_bps": cost_bps, "slippage_bps": slippage_bps, **metrics})
    return pd.DataFrame(rows)


def add_baseline_delta(results: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out = out.drop(columns=["strongest_baseline", "strongest_baseline_return", "baseline_delta"], errors="ignore")
    rows: list[dict[str, Any]] = []
    for (split, horizon, cost_bps, slippage_bps, max_positions), _group in out.groupby(["split", "horizon", "cost_bps", "slippage_bps", "max_positions"], dropna=False):
        subset = baselines[
            baselines["split"].eq(split)
            & pd.to_numeric(baselines["horizon"], errors="coerce").eq(float(horizon))
            & pd.to_numeric(baselines["cost_bps"], errors="coerce").eq(float(cost_bps))
            & pd.to_numeric(baselines["slippage_bps"], errors="coerce").eq(float(slippage_bps))
        ]
        same = subset[subset["max_positions"].isna() | pd.to_numeric(subset["max_positions"], errors="coerce").eq(float(max_positions))]
        best_idx = pd.to_numeric(same["total_return"], errors="coerce").idxmax() if not same.empty else None
        rows.append(
            {
                "split": split,
                "horizon": horizon,
                "cost_bps": cost_bps,
                "slippage_bps": slippage_bps,
                "max_positions": max_positions,
                "strongest_baseline": str(same.loc[best_idx, "baseline_name"]) if best_idx is not None and pd.notna(best_idx) else "",
                "strongest_baseline_return": float(pd.to_numeric(same["total_return"], errors="coerce").max()) if not same.empty else 0.0,
            }
        )
    out = out.merge(pd.DataFrame(rows), on=["split", "horizon", "cost_bps", "slippage_bps", "max_positions"], how="left")
    out["baseline_delta"] = out["total_return"] - out["strongest_baseline_return"]
    return out


def add_strategy_scores(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["max_drawdown_score"] = (1.0 + pd.to_numeric(out["max_drawdown"], errors="coerce")).clip(0.0, 1.0)
    out["trade_count_score"] = (pd.to_numeric(out["trade_count"], errors="coerce") / 100.0).clip(0.0, 1.0)
    exposure = pd.to_numeric(out["exposure_ratio"], errors="coerce")
    out["exposure_score"] = (1.0 - (exposure - 0.5).abs()).clip(0.0, 1.0)
    out["turnover_score"] = 1.0 / (1.0 + pd.to_numeric(out["turnover"], errors="coerce").clip(lower=0.0))
    out["validation_strategy_score"] = (
        0.25 * pd.to_numeric(out["baseline_delta"], errors="coerce").fillna(-1.0)
        + 0.20 * pd.to_numeric(out["sharpe"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out["total_return"], errors="coerce").fillna(-1.0)
        + 0.15 * out["max_drawdown_score"].fillna(0.0)
        + 0.10 * out["trade_count_score"].fillna(0.0)
        + 0.05 * out["exposure_score"].fillna(0.0)
        + 0.05 * out["turnover_score"].fillna(0.0)
    )
    out["passes_trade_count"] = pd.to_numeric(out["trade_count"], errors="coerce") >= 20
    out["passes_exposure"] = exposure.between(0.10, 0.90, inclusive="both")
    out["passes_positive_after_cost"] = pd.to_numeric(out["total_return"], errors="coerce") > 0.0
    out["passes_drawdown_reported"] = pd.to_numeric(out["max_drawdown"], errors="coerce").notna()
    out["passes_baseline_preferred"] = pd.to_numeric(out["baseline_delta"], errors="coerce") > 0.0
    out["validation_selected_eligible"] = out["passes_trade_count"] & out["passes_exposure"] & out["passes_positive_after_cost"] & out["passes_drawdown_reported"]
    return out


def lock_strategy(validation: pd.DataFrame) -> dict[str, Any]:
    eligible = validation[validation["validation_selected_eligible"]].copy()
    if eligible.empty:
        selected = validation.sort_values(["validation_strategy_score", "total_return"], ascending=[False, False]).iloc[0].to_dict()
        status = "no_validation_strategy_passed_filters"
    else:
        selected = eligible.sort_values(["passes_baseline_preferred", "validation_strategy_score", "baseline_delta"], ascending=[False, False, False]).iloc[0].to_dict()
        status = "validation_selected_strategy"
    locked = {key: selected.get(key) for key in [
        "strategy_id",
        "model_candidate_id",
        "model_family",
        "target_variant",
        "target_family",
        "feature_group",
        "horizon",
        "strategy_template",
        "top_n",
        "top_quantile",
        "w_model",
        "w_rs",
        "market_regime_filter",
        "max_positions",
        "max_exposure",
        "cost_bps",
        "slippage_bps",
        "strategy_family_key",
        "total_return",
        "sharpe",
        "max_drawdown",
        "trade_count",
        "exposure_ratio",
        "turnover",
        "strongest_baseline",
        "baseline_delta",
        "validation_strategy_score",
    ]}
    locked["selection_status"] = status
    locked["selected_by"] = "validation_only_strategy_score"
    locked["final_performance_used"] = False
    locked["locked_at_utc"] = now_utc()
    locked["claim_label"] = "strategy_candidate"
    return locked


def build_final_frame(features: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    candidate = payload["candidate"]
    labels = payload["labels"]
    splits = payload["splits"]
    final_idx = splits["final"]
    prob = predict_probability(payload["model"], clean_feature_matrix(features.loc[final_idx], payload["feature_cols"]))
    return signal_frame(features, final_idx, labels, prob, "final", candidate)


def locked_cost_grid(locked: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cost_bps, slippage_bps in itertools.product(COST_BPS, SLIPPAGE_BPS):
        row = dict(locked)
        row["cost_bps"] = cost_bps
        row["slippage_bps"] = slippage_bps
        row["strategy_id"] = f"{locked['strategy_id']}__c{cost_bps}_s{slippage_bps}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_reports(
    validation: pd.DataFrame,
    final_locked: pd.DataFrame,
    exploratory: pd.DataFrame,
    locked: dict[str, Any],
    baselines: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> None:
    final_row = final_locked.iloc[0].to_dict() if not final_locked.empty else {}
    best_target = validation.sort_values("validation_strategy_score", ascending=False).iloc[0]["target_variant"] if not validation.empty else ""
    best_feature = validation.sort_values("validation_strategy_score", ascending=False).iloc[0]["feature_group"] if not validation.empty else ""
    best_model = validation.sort_values("validation_strategy_score", ascending=False).iloc[0]["model_family"] if not validation.empty else ""
    best_exp = exploratory.head(1).to_dict("records")
    locked_cost = as_float(final_row.get("cost_bps"))
    locked_slip = as_float(final_row.get("slippage_bps"))
    locked_horizon = as_float(final_row.get("horizon"))
    locked_mp = as_float(final_row.get("max_positions"))
    base = baselines[
        baselines["split"].eq("final")
        & pd.to_numeric(baselines["horizon"], errors="coerce").eq(locked_horizon)
        & pd.to_numeric(baselines["cost_bps"], errors="coerce").eq(locked_cost)
        & pd.to_numeric(baselines["slippage_bps"], errors="coerce").eq(locked_slip)
    ]
    buy_hold = base[base["baseline_name"].eq("buy_and_hold_vn30_index")]["total_return"].astype(float).max() if not base.empty else math.nan
    equal_weight = base[base["baseline_name"].eq("equal_weight_vn30_stock_basket")]["total_return"].astype(float).max() if not base.empty else math.nan
    random_same = base[base["baseline_name"].eq("random_signal_same_turnover") & pd.to_numeric(base["max_positions"], errors="coerce").eq(locked_mp)]["total_return"].astype(float).max() if not base.empty else math.nan
    improves_v7 = bool(as_float(final_row.get("total_return")) > V7_LOCKED_FINAL_RETURN)
    claimable = False

    protocol = f"""# VN30 V8 Strategy Target Redesign Protocol

## Scope

- VN30 stock hourly strategy diagnostics only.
- Target variants are market-relative, top-quantile, cost-adjusted, and neutral-removed labels.
- Main index data is lagged market context only; index benchmark results are not stock strategy claims.
- Validation-only model/target/feature/strategy selection; final-ranked rows are exploratory_not_claimable.
- No broad brute-force grid, trading recommendation, live deployment, DOCX, push, merge, or tag.

## Staged Search

- Model screening: broad logistic target/feature/horizon screen plus h40 combined-feature model-family sweep.
- Strategy grid: cross-sectional rank rotation templates on validation-shortlisted model candidates.
- Locked final: one validation-selected strategy evaluated once on final.
"""
    write_markdown(PROTOCOL_PATH, protocol)

    result = f"""# VN30 V8 Strategy Target Redesign Result Summary

## Locked Strategy

- Strategy: `{locked.get("strategy_id")}`.
- Target/feature/model: {locked.get("target_variant")} / {locked.get("feature_group")} / {locked.get("model_family")}.
- Horizon/template: h{locked.get("horizon")} / {locked.get("strategy_template")}.
- Validation return/Sharpe/drawdown: {pct(locked.get("total_return", math.nan))} / {locked.get("sharpe")} / {pct(locked.get("max_drawdown", math.nan))}.
- Validation trades/exposure/turnover: {locked.get("trade_count")} / {locked.get("exposure_ratio")} / {locked.get("turnover")}.
- Validation strongest baseline delta: {locked.get("strongest_baseline")} / {pp(locked.get("baseline_delta", math.nan))}.

## Final Locked Result

- Final return after cost: {pct(final_row.get("total_return", math.nan))}.
- Final Sharpe: {final_row.get("sharpe", "")}.
- Final max drawdown: {pct(final_row.get("max_drawdown", math.nan))}.
- Final trades/exposure/turnover: {final_row.get("trade_count", "")} / {final_row.get("exposure_ratio", "")} / {final_row.get("turnover", "")}.
- Final strongest baseline: {final_row.get("strongest_baseline", "")}; baseline delta: {pp(final_row.get("baseline_delta", math.nan))}.

## Required Answers

1. Best validation target variant: {best_target}.
2. Best validation feature group: {best_feature}.
3. Best validation model family: {best_model}.
4. Locked strategy: `{locked.get("strategy_id")}`.
5. Final total return after cost: {pct(final_row.get("total_return", math.nan))}.
6. Final Sharpe: {final_row.get("sharpe", "")}.
7. Final max drawdown: {pct(final_row.get("max_drawdown", math.nan))}.
8. Final trade count and exposure: {final_row.get("trade_count", "")} / {final_row.get("exposure_ratio", "")}.
9. Beats buy-and-hold VN30: {str(as_float(final_row.get("total_return")) > as_float(buy_hold)).lower()} (baseline={pct(buy_hold)}).
10. Beats equal-weight VN30: {str(as_float(final_row.get("total_return")) > as_float(equal_weight)).lower()} (baseline={pct(equal_weight)}).
11. Beats random same-turnover: {str(as_float(final_row.get("total_return")) > as_float(random_same)).lower()} (baseline={pct(random_same)}).
12. Best exploratory final strategy: {best_exp[:1]}.
13. Target redesign improves over V7 locked final return: {str(improves_v7).lower()}.
14. Claimable results: none as a strategy claim; locked strategy is validation-governed diagnostic only.
15. Paper-safe wording: offline diagnostic only; no BUY/SELL, profitability, investment advice, deployment, or claimable strategy claim.

Paper-safe wording:

> VN30 V8 redesigned the strategy target from absolute-direction ranking toward market-relative, top-quantile, cost-adjusted, and neutral-removed stock selection labels. The locked strategy was selected by validation-only diagnostics and evaluated once on final. Final-ranked strategy rows are exploratory only, and future-blind confirmation is required before any stronger strategy claim.
"""
    write_markdown(RESULT_PATH, result)

    claim = """# VN30 V8 Strategy Target Redesign Claim Boundary

- Offline diagnostic simulation only.
- No BUY/SELL recommendation.
- No live trading.
- No profitability guarantee.
- No investment advice.
- No deployment claim.
- Target variants are reported separately; absolute-direction, relative, top-quantile, cost-adjusted, and neutral-removed claims are not mixed.
- Final-ranked strategy rows are exploratory_not_claimable.
- No new strategy result is claimable without future-blind confirmation.
- Current claimable directional champion remains the 61.61% L2 Logistic baseline60_candidate.
- No DOCX, paper, push, merge, tag, VN100, or index-as-stock claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, _prior_groups, _index_audit, _index_manifest, feature_manifest = build_v3_feature_frame()
    index_data = load_index_data()
    features = enrich_features(add_market_drawdown_feature(features, index_data))
    feature_groups, feature_audit = build_feature_groups(features)
    target_audit_df, label_cache = target_audit(features, index_data)
    registry = model_registry(feature_groups)

    write_frame(OUTPUT_DIR / "target_variant_audit.csv", target_audit_df)
    write_frame(OUTPUT_DIR / "feature_group_audit.csv", feature_audit)
    write_frame(OUTPUT_DIR / "model_candidate_registry.csv", registry)

    print(f"Running V8 validation model screening: {len(registry)} candidates", flush=True)
    model_results, payloads = fit_validation_models(features, feature_groups, registry, label_cache)
    write_frame(OUTPUT_DIR / "validation_model_results.csv", model_results)
    shortlist = shortlist_models(model_results)
    if not shortlist:
        raise RuntimeError("no V8 model candidates fitted")
    grid = strategy_grid(shortlist, payloads)
    write_frame(OUTPUT_DIR / "strategy_grid.csv", grid)

    baselines = baseline_comparison(features, index_data, label_cache)
    write_frame(OUTPUT_DIR / "baseline_strategy_comparison.csv", baselines)

    validation_frames = {mid: payloads[mid]["validation_frame"] for mid in shortlist if mid in payloads}
    print(f"Running V8 validation strategies: {len(grid)} variants over {len(validation_frames)} model score frames", flush=True)
    validation_results, _curves, _trades = run_strategies(grid, validation_frames, "validation")
    validation_results = add_strategy_scores(add_baseline_delta(validation_results, baselines))
    validation_results.loc[validation_results["passes_baseline_preferred"], "claim_label"] = "strategy_outperforms_baseline"
    validation_results.loc[validation_results["validation_selected_eligible"] & ~validation_results["passes_baseline_preferred"], "claim_label"] = "strategy_candidate"
    leaderboard = validation_results.sort_values(["validation_selected_eligible", "passes_baseline_preferred", "validation_strategy_score"], ascending=[False, False, False]).reset_index(drop=True)
    locked = lock_strategy(validation_results)
    write_frame(OUTPUT_DIR / "validation_strategy_results.csv", validation_results)
    write_frame(OUTPUT_DIR / "validation_strategy_leaderboard.csv", leaderboard)
    write_json(OUTPUT_DIR / "locked_strategy.json", locked)

    locked_payload = payloads[str(locked["model_candidate_id"])]
    final_frame = build_final_frame(features, locked_payload)
    final_frames = {str(locked["model_candidate_id"]): final_frame}
    locked_grid = pd.DataFrame([locked])
    print("Running V8 locked final strategy once", flush=True)
    final_result, final_curve, final_trades = run_strategies(locked_grid, final_frames, "final", keep_strategy_id=str(locked["strategy_id"]))
    final_result = add_baseline_delta(final_result, baselines)
    final_result["claim_label"] = "strategy_candidate"
    final_result["reason_not_claimable"] = "offline diagnostic; future-blind confirmation required"
    write_frame(OUTPUT_DIR / "final_strategy_result.csv", final_result)
    write_frame(OUTPUT_DIR / "final_equity_curve.csv", final_curve)
    write_frame(OUTPUT_DIR / "final_trade_log.csv", final_trades)

    # Exploratory final leaderboard uses validation leaderboard rows only; ranking is explicitly non-claimable.
    exploratory_grid = leaderboard.head(80)[grid.columns].copy()
    final_frames_for_exploration: dict[str, pd.DataFrame] = {}
    for mid in exploratory_grid["model_candidate_id"].astype(str).unique():
        final_frames_for_exploration[mid] = build_final_frame(features, payloads[mid])
    print(f"Running V8 exploratory final strategies: {len(exploratory_grid)} variants", flush=True)
    exploratory, _e_curves, _e_trades = run_strategies(exploratory_grid, final_frames_for_exploration, "final")
    exploratory = add_baseline_delta(exploratory, baselines).sort_values(["total_return", "baseline_delta", "sharpe"], ascending=[False, False, False]).reset_index(drop=True)
    exploratory["claim_label"] = "exploratory_not_claimable"
    write_frame(OUTPUT_DIR / "exploratory_final_strategy_leaderboard.csv", exploratory)

    cost_grid = locked_cost_grid(locked)
    cost_final, _c_curves, _c_trades = run_strategies(cost_grid, final_frames, "final")
    cost_final = add_baseline_delta(cost_final, baselines)
    write_frame(OUTPUT_DIR / "cost_slippage_sensitivity.csv", cost_final)
    write_frame(OUTPUT_DIR / "random_same_turnover_comparison.csv", baselines[baselines["baseline_name"].eq("random_signal_same_turnover")])
    exposure = pd.concat([validation_results, final_result], ignore_index=True, sort=False)[["strategy_id", "split", "target_variant", "model_family", "strategy_template", "max_positions", "max_exposure", "cost_bps", "slippage_bps", "exposure_ratio", "turnover", "trade_count"]]
    drawdown = pd.concat([validation_results, final_result], ignore_index=True, sort=False)[["strategy_id", "split", "target_variant", "model_family", "strategy_template", "max_positions", "max_exposure", "cost_bps", "slippage_bps", "max_drawdown", "calmar", "total_return"]]
    write_frame(OUTPUT_DIR / "exposure_turnover_summary.csv", exposure)
    write_frame(OUTPUT_DIR / "drawdown_summary.csv", drawdown)

    run_config = {
        "created_at_utc": now_utc(),
        "scope": "VN30 V8 strategy target redesign diagnostics",
        "horizons": HORIZONS,
        "cost_bps": COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "max_positions": MAX_POSITIONS,
        "max_exposure": MAX_EXPOSURE,
        "model_registry_rows": int(len(registry)),
        "fitted_model_rows": int(model_results["status"].eq("fit").sum()),
        "strategy_grid_rows": int(len(grid)),
        "validation_rows": int(len(validation_results)),
        "exploratory_final_rows": int(len(exploratory)),
        "final_performance_used_for_claimable_selection": False,
        "broad_bruteforce_grid_run": False,
        "git_tags_created": False,
        "paper_docx_generated": False,
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)
    manifest = {
        **run_config,
        "locked_strategy": locked,
        "locked_final_result": final_result.to_dict("records"),
        "best_validation_target_variant": leaderboard.iloc[0]["target_variant"],
        "best_validation_feature_group": leaderboard.iloc[0]["feature_group"],
        "best_validation_model_family": leaderboard.iloc[0]["model_family"],
        "feature_manifest": feature_manifest,
        "claim_boundary": "offline diagnostic only; future-blind confirmation required",
    }
    write_json(OUTPUT_DIR / "v8_manifest.json", manifest)
    write_reports(validation_results, final_result, exploratory, locked, baselines, cost_final)
    print(f"VN30 V8 strategy target redesign complete: {rel(OUTPUT_DIR)}", flush=True)
    print(f"Locked strategy: {locked['strategy_id']} target={locked['target_variant']} model={locked['model_family']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
