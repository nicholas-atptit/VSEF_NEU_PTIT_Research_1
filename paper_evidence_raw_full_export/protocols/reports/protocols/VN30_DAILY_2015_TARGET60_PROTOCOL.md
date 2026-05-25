# VN30 Daily 2015 Target60 Protocol

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Branch: research/vn100-evidence-hardening-v1
- Previous tag: nckh/vn30-daily-2015-bcm-vib-recovery-v1

## Separation

- This is a **daily-only** track, separate from the hourly track.
- Daily data must **not** be resampled to hourly.
- Daily and hourly outputs must **not** be mixed.
- No hourly claims may be made from daily data.

## Current State

- Universe: VN30 January 2025 review (30/30 usable).
- Frequency: daily only.
- Previous best result: XGBoost h=1 at **55.84%** pooled overall directional accuracy (10,050 rows, 2025-01-01+).
- No experiment passed the 60% threshold.

## Target

- **>=60% pooled overall directional accuracy** on the full 30-ticker daily universe.
- Metric: pooled overall directional accuracy (correct / total valid predictions).
- No confidence abstention or ticker subset for the main target.

## Horizons

h=1, h=3, h=5, h=10, h=20, h=40, h=60 trading days.

## Data Discipline

- **Final evaluation labels (2025-01-01+) cannot be used for tuning.**
- Only pre-2025 validation data may be used for model/feature/hyperparameter selection.
- Validation preference: rolling validation across 2021, 2022, 2023, 2024.
- If rolling validation is not feasible, 2024 validation will be used with stated limitation.
- No future values in features.
- No label leakage.

## Models

- XGBoost
- LightGBM
- Random Forest
- Stacking only if time-series-safe (no shuffle, walk-forward or expanding window).

## Feature Sets

- Existing daily OHLCV features
- Lagged daily returns
- Rolling returns (various windows)
- Rolling volatility
- Rolling volume change
- Momentum features
- Market-relative features (if daily index data exists)
- No future values

## Selection Rule

- Best candidate is selected **solely on validation accuracy** (pre-2025).
- The selected candidate is then evaluated on the final period (2025-01-01+).
- If validation accuracy >=60%, the candidate is a "60 candidate".
- Final accuracy is reported without further tuning.

## Claim Boundary

- No trading-readiness claim.
- No profitability claim.
- No cost/slippage claim.
- No paper or DOCX generated.
