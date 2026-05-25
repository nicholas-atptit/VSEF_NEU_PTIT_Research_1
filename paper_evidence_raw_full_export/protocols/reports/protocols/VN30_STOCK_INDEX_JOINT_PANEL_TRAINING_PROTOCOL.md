# VN30 Stock + Index Joint Panel Training Protocol

## Purpose

This protocol defines a joint 36-instrument benchmark for the current research direction:

> So sánh hiệu quả các mô hình học máy trong dự báo xu hướng cổ phiếu VN30: Bằng chứng từ kiểm định walk-forward và benchmark chỉ số thị trường Việt Nam

The experiment tests whether a combined stock+index panel can improve overall directional accuracy while keeping stock-only and index-only results separate.

## Joint Universe

This is a joint 36-instrument benchmark.

- 30 instruments are active VN30 January 2025 stock tickers.
- 6 instruments are supported market indices.
- Indices are not merely side/context features. They are also prediction targets and rows in the panel.

Stocks:

- `ACB`, `BID`, `BCM`, `BVH`, `CTG`, `FPT`, `GAS`, `GVR`, `HDB`, `HPG`, `LPB`, `MBB`, `MSN`, `MWG`, `PLX`, `SAB`, `SHB`, `SSB`, `SSI`, `STB`, `TCB`, `TPB`, `VCB`, `VHM`, `VIB`, `VIC`, `VJC`, `VNM`, `VPB`, `VRE`

The prior joint-panel protocol accidentally listed `DGC` and `VPL` and omitted `BCM` and `BVH`. The active January 2025 VN30 joint-panel universe excludes `DGC` and `VPL`.

Indices:

- `VNINDEX`
- `VN30`
- `HNXINDEX`
- `HNX30`
- `UPCOMINDEX`
- `VN100`

## Metric And Targets

Main metric:

- Pooled overall directional accuracy across all eligible final rows in the 36-instrument panel.

Main target:

- Combined 36-instrument overall directional accuracy >= 60%.

Ambitious target:

- Combined 36-instrument overall directional accuracy >= 65%.
- Final65 can be claimed only if validation-safe, audited, full-coverage, and not selected by final-window score.

## Mandatory Reporting

Every run must report:

1. Combined 36-instrument accuracy.
2. Stock-only 30 VN30 accuracy.
3. Index-only 6-index accuracy.

The final claim must not confuse combined, stock-only, and index-only accuracy.

## Selection Rules

- No confidence abstention.
- No ticker or instrument subset for the main target.
- No top-k/ranking substitution for overall directional accuracy.
- Candidate selection must be validation-only.
- Final evaluation must be scoring-only.
- Final accuracy must not enter the candidate selection score.
- Because prior final windows have been inspected repeatedly, new improvements should be labeled exploratory unless later verified on future blind data.

## Feature And Leakage Rules

Allowed feature families:

- Own-instrument lagged returns, rolling returns, rolling volatility, rolling high-low range, rolling volume change, momentum, range shock, and volume shock.
- Lagged supported-index market context for every instrument.
- Relative stock/index spread features built from lagged values.
- Panel features including instrument type, instrument encoding, cross-sectional ranks from prior timestamp, and lagged market breadth.

Forbidden:

- Future own return.
- Future index return.
- Future market regime.
- Target direction leakage.
- Target timestamp leakage.
- Final-period hand-crafted filters.
- Using final accuracy in feature/model selection.

## Claim Boundary

- Combined 36-instrument result is not the same as stock-only result.
- Index-only performance cannot be used as VN30 stock performance.
- Stock-only performance must be reported separately.
- No trading, profitability, or live-deployment claim is allowed.
