# VN30 Daily 2015 Target60 V2 Protocol

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Branch: research/vn100-evidence-hardening-v1
- Previous tag: nckh/vn30-daily-2015-target60-optimization-v1

## Separation

- This is a **daily-only** track, separate from the hourly track.
- Daily data must **not** be resampled to hourly.
- Daily and hourly outputs must **not** be mixed.
- No hourly claims may be made from daily data.

## Current State

- Universe: VN30 January 2025 review (30/30 usable).
- Frequency: daily only.
- Previous best result: LightGBM daily_cross h=40 at **57.58%** pooled overall directional accuracy (8,880 rows, 2025-01-01+).
- Target60 v1: 0 candidates passed 60%.

## Target

- **>=60% pooled overall directional accuracy** on the full 30-ticker daily universe.
- Metric: pooled overall directional accuracy (correct / total valid predictions).
- No confidence abstention or ticker subset for the main target.
- Every row must be predicted (no abstention).
- Full coverage required.

## Accuracy Drag Diagnosis (from v1)

- Worst ticker: VIC at 35.81% (class balance 90.2% positive).
- 8 tickers below 50% accuracy.
- Accuracy std across tickers: 0.1111.
- Weak periods: 2025-03 (46.51%), 2025-09 (41.83%), 2025-10 (41.88%), 2025-11 (46.33%).
- Model tends to predict positive more often (4995 predicted 1 vs 3885 predicted 0).
- Gap to 60%: 2.42 percentage points.

## Candidate Focus

### Models
1. LightGBM h=30, h=40, h=50, h=60 (primary)
2. XGBoost h=30, h=40, h=50, h=60 (comparison)
3. Random Forest h=40, h=60 (comparison)

### Feature Sets
1. **daily_cross**: Extended + cross-sectional rank (baseline, 108 features)
2. **daily_cross_plus_market**: daily_cross + daily index context (if daily index cache exists)
3. **daily_cross_interactions**: daily_cross + pairwise interaction features
4. **volatility_normalized**: Features normalized by rolling volatility
5. **momentum_volatility_interaction**: Momentum features interacted with volatility regime

### Decision Threshold
- Thresholds: 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65
- Selected on rolling validation only (2021-2024).
- Must apply to ALL final rows (no abstention).
- Default threshold is 0.50.

### Class Imbalance Handling
- class_weight / scale_pos_weight where supported.
- Threshold tuning by rolling validation only.

## Validation

- Rolling validation: 2021, 2022, 2023, 2024.
- Training: all data before each validation year.
- Selection: best candidate chosen on rolling validation stability_score (mean - std/2).
- Final evaluation: 2025-01-01 to latest available (used only for scoring, never for selection).

## Data Discipline

- **Final evaluation labels (2025-01-01+) cannot be used for tuning.**
- Only pre-2025 validation data may be used for model/feature/hyperparameter/threshold selection.
- No future values in features.
- No label leakage.

## Claim Boundary

- No trading-readiness claim.
- No profitability claim.
- No cost/slippage claim.
- No paper or DOCX generated.
