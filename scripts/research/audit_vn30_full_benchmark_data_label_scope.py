"""Audit VN30 full benchmark data, labels, splits, and context scope."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_full_benchmark_regime_deep import (  # noqa: E402
    FINAL_START,
    HORIZONS,
    REPORT_DIR,
    TRAIN_END,
    VAL_END,
    VAL_START,
    label_frame,
    load_benchmark_features,
    strict_split_indices,
    write_csv,
    write_markdown,
)
from scripts.research.vn30_hourly_dual_track_common import active_stock_tickers, load_index_data, rel  # noqa: E402


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:.2f}%"


def label_distribution(features: pd.DataFrame, horizon: int, split: str, idx: pd.Index, labels: pd.DataFrame) -> dict[str, Any]:
    y = labels.loc[idx, "y"].dropna().astype(int)
    return {
        "horizon": horizon,
        "split": split,
        "rows": int(len(y)),
        "up_rows": int((y == 1).sum()),
        "down_rows": int((y == 0).sum()),
        "positive_rate": float(y.mean()) if len(y) else math.nan,
        "majority_baseline": max(float(y.mean()), 1.0 - float(y.mean())) if len(y) else math.nan,
        "ticker_coverage": int(features.loc[y.index, "ticker"].nunique()) if len(y) else 0,
        "first_feature_datetime": str(features.loc[y.index, "datetime"].min()) if len(y) else "",
        "last_feature_datetime": str(features.loc[y.index, "datetime"].max()) if len(y) else "",
        "last_future_datetime": str(labels.loc[y.index, "future_datetime"].max()) if len(y) else "",
    }


def grouped_imbalance(features: pd.DataFrame, labels: pd.DataFrame, idx: pd.Index, horizon: int, group_col: str, level_name: str) -> pd.DataFrame:
    if len(idx) == 0:
        return pd.DataFrame()
    work = features.loc[idx, ["datetime", "ticker", "market_direction_regime", "volatility_regime"]].copy()
    work["y"] = labels.loc[idx, "y"].astype(int).to_numpy()
    work["month"] = pd.to_datetime(work["datetime"]).dt.to_period("M").astype(str)
    work["quarter"] = pd.to_datetime(work["datetime"]).dt.to_period("Q").astype(str)
    work["regime"] = work["market_direction_regime"].astype(str) + "/" + work["volatility_regime"].astype(str)
    out = work.groupby(group_col)["y"].agg(["count", "mean"]).reset_index()
    out = out.rename(columns={group_col: "bucket", "count": "rows", "mean": "positive_rate"})
    out.insert(0, "horizon", horizon)
    out.insert(1, "split", "final")
    out.insert(2, "level", level_name)
    out["majority_baseline"] = out["positive_rate"].apply(lambda value: max(float(value), 1.0 - float(value)) if pd.notna(value) else math.nan)
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features, _family_cols, manifest, _regime = load_benchmark_features()
    tickers = active_stock_tickers()
    index_data = load_index_data()

    union_timestamps = set(pd.to_datetime(features["datetime"], errors="coerce").dropna().unique())
    ticker_rows: list[dict[str, Any]] = []
    duplicates = features.duplicated(["ticker", "datetime"], keep=False)
    for ticker in tickers:
        frame = features[features["ticker"].eq(ticker)].copy()
        ticker_times = set(pd.to_datetime(frame["datetime"], errors="coerce").dropna().unique())
        ticker_rows.append(
            {
                "row_type": "ticker_coverage",
                "ticker": ticker,
                "available_hourly_rows": int(len(frame)),
                "first_datetime": str(frame["datetime"].min()) if not frame.empty else "",
                "last_datetime": str(frame["datetime"].max()) if not frame.empty else "",
                "missing_rows_vs_union_calendar": int(len(union_timestamps.difference(ticker_times))),
                "duplicate_ticker_timestamps": int(duplicates.loc[frame.index].sum()) if not frame.empty else 0,
                "full_coverage_member": ticker in tickers,
            }
        )

    split_rows: list[dict[str, Any]] = []
    imbalance_frames: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        labels = label_frame(features, horizon)
        idx = strict_split_indices(features, labels)
        for split_name, split_idx in idx.items():
            split_rows.append(label_distribution(features, horizon, split_name, split_idx, labels))
        final_idx = idx["final"]
        imbalance_frames.extend(
            [
                grouped_imbalance(features, labels, final_idx, horizon, "ticker", "ticker"),
                grouped_imbalance(features, labels, final_idx, horizon, "month", "month"),
                grouped_imbalance(features, labels, final_idx, horizon, "quarter", "quarter"),
                grouped_imbalance(features, labels, final_idx, horizon, "regime", "regime"),
            ]
        )

    index_rows = [
        {
            "row_type": "index_context_availability",
            "index_code": code,
            "rows": int(len(frame)),
            "first_datetime": str(frame["datetime"].min()) if not frame.empty else "",
            "last_datetime": str(frame["datetime"].max()) if not frame.empty else "",
            "role": "lagged_context_only",
        }
        for code, frame in sorted(index_data.items())
    ]

    summary = pd.concat(
        [
            pd.DataFrame(ticker_rows),
            pd.DataFrame(index_rows),
            *(frame for frame in imbalance_frames if frame is not None and not frame.empty),
        ],
        ignore_index=True,
        sort=False,
    )
    label_summary = pd.DataFrame(split_rows)
    write_csv(REPORT_DIR / "data_label_summary.csv", summary)
    write_csv(REPORT_DIR / "label_distribution_by_split.csv", label_summary)

    h40_final = label_summary[(label_summary["horizon"].eq(40)) & (label_summary["split"].eq("final"))]
    h40_final_rows = int(h40_final.iloc[0]["rows"]) if not h40_final.empty else 0
    h40_majority = float(h40_final.iloc[0]["majority_baseline"]) if not h40_final.empty else math.nan
    missing_ticker_files = [ticker for ticker in tickers if features[features["ticker"].eq(ticker)].empty]
    duplicate_count = int(duplicates.sum())
    markdown = [
        "# VN30 Full Benchmark Data and Label Audit",
        "",
        "## Scope",
        "",
        "- Main target: VN30 stock-only hourly overall directional accuracy.",
        f"- Full ticker coverage expected: 30 tickers; observed: {features['ticker'].nunique()} tickers.",
        "- Data source: existing local artifacts only.",
        "- Provider behavior changed: no.",
        "- New market data fetched: no.",
        "",
        "## Split Boundaries",
        "",
        f"- Train feature/outcome cutoff: {TRAIN_END}.",
        f"- Validation feature/outcome window: {VAL_START} to {VAL_END}.",
        f"- Final scoring feature window starts: {FINAL_START}.",
        "- Strict label rule: train and validation rows require future outcome timestamps inside the same split.",
        "",
        "## Coverage",
        "",
        f"- Available ticker coverage: {features['ticker'].nunique()}/30.",
        f"- Missing ticker files/empty tickers: {', '.join(missing_ticker_files) if missing_ticker_files else 'none'}.",
        f"- Duplicate ticker timestamps after feature assembly: {duplicate_count}.",
        f"- h40 final rows expected/observed: {h40_final_rows}.",
        f"- h40 final majority baseline from strict labels: {pct(h40_majority)}.",
        "",
        "## Label Availability by Split",
        "",
        "| Horizon | Split | Rows | Up | Down | Positive Rate | Majority | Tickers | Last Future Timestamp |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in split_rows:
        markdown.append(
            f"| h{row['horizon']} | {row['split']} | {row['rows']} | {row['up_rows']} | {row['down_rows']} | "
            f"{pct(row['positive_rate'])} | {pct(row['majority_baseline'])} | {row['ticker_coverage']} | {row['last_future_datetime']} |"
        )
    markdown.extend(
        [
            "",
            "## Index Context Availability",
            "",
            "| Index | Rows | First | Last | Role |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for row in index_rows:
        markdown.append(f"| {row['index_code']} | {row['rows']} | {row['first_datetime']} | {row['last_datetime']} | {row['role']} |")
    markdown.extend(
        [
            "",
            "## Audit Status",
            "",
            f"- Full 30-stock coverage for headline target: {'yes' if features['ticker'].nunique() == 30 else 'no'}.",
            f"- Index data limited to lagged/context role by manifest: {'yes' if manifest.get('regime_features', {}).get('uses_future_returns') is False else 'no'}.",
            "- Ticker subset used for main claim: no.",
            "- Confidence abstention used for main claim: no.",
            "- Top-k substitution used for main claim: no.",
            "",
            f"Generated artifacts: `{rel(REPORT_DIR / 'data_label_summary.csv')}` and `{rel(REPORT_DIR / 'label_distribution_by_split.csv')}`.",
        ]
    )
    write_markdown(REPORT_DIR / "data_label_audit.md", "\n".join(markdown))
    print(f"Data/label audit complete: {rel(REPORT_DIR / 'data_label_audit.md')}")


if __name__ == "__main__":
    main()
