# VN30 Hourly 2015 - Top-K 75% Verification Protocol

## Purpose
Independent verification of the reported 75.54% precision@10 result.
Determine if the result is trustworthy or caused by:
- Overfitting
- Data leakage
- Metric definition error
- Target construction error
- Overly easy ranking metric

## 1. Metric Correctness

### precision@k Definition
- precision@k = overlap(predicted_top_k, true_top_k) / k
- Computed per timestamp/event group
- Averaged across event groups (macro average)
- Denominator must be exactly k unless fewer eligible stocks are documented
- Must verify: mean vs median vs event-weighted precision

### hit_rate@k Definition
- hit_rate@k = 1 if overlap(predicted_top_k, true_top_k) >= 1, else 0
- Averaged across event groups
- **Warning**: hit_rate@10 with 30 tickers is likely trivially easy
- Mark as weak/easy metric if hit_rate@10 > 95% across most configs

### Verification Steps
- Recompute precision@10 from raw scores if available
- Verify denominator is exactly k=10
- Verify overlap computation is correct
- Check if averaging method matches original

## 2. Grouping Correctness

### Requirements
- Predicted top-k and true top-k computed within same timestamp
- No cross-timestamp ranking
- No mixing ticker rows across different timestamps
- Each event group must disclose eligible stock count

### Verification Steps
- For each timestamp: verify all tickers belong to same datetime
- Verify predicted top-k count == k
- Verify true top-k count == k
- Report eligible stock count per timestamp
- Flag any timestamp with < k eligible stocks

## 3. Target Correctness

### Requirements
- True top-k based on future return for horizon h=40 (or h=120 if that's the actual best)
- Predicted top-k based on model scores available before target realization
- Future return must never be used as a feature

### Verification Steps
- Verify future return computation: (close[t+h] - close[t]) / close[t]
- Verify no future price/return in feature columns
- Verify model scores computed before target realization
- Check for any target leakage in features

## 4. Leakage Audit

### Forbidden Features
- future_return
- target
- actual
- label
- is_top_k
- rank_future
- Any column containing future target information
- Same-horizon future market return

### Verification Steps
- Inspect all feature columns used
- Check for any forward-looking information
- Verify no label-derived score
- Verify no final-eval label used in selection
- Verify no final-eval target leakage through normalization

## 5. Overfitting Audit

### Checks
- Compare train/validation/final precision@10
- Check if validation performance is close to final performance
- Check temporal stability by month/quarter
- Check if one small period drives the result
- Check ticker concentration

### Red Flags
- Validation precision >> final precision
- One month/quarter drives majority of correct predictions
- Top 5 tickers account for >50% of selections
- Fewer than 100 events in final evaluation

## 6. Baseline Audit

### Baselines for k=10
- Random top-10 baseline (1000 seeds)
- Previous-return top-10 baseline
- Momentum top-10 baseline
- Market-relative baseline if useful

### Metrics
- Model lift over random baseline
- Model delta over momentum baseline
- Statistical significance vs random

## 7. Null/Permutation Tests

### Tests
1. Random top-k baseline over 1000 seeds
2. Shuffle model scores within each timestamp
3. Shuffle true labels within each timestamp

### Outputs
- Observed precision@10
- Null mean/std
- Empirical p-value
- Whether observed result is statistically stronger than random

## 8. Decision Output

### Classification
- verified_strong: Passes all checks, robust, above baselines
- verified_but_metric_easy: Correct computation but metric is trivially easy
- suspicious_needs_fix: Some concerns but not invalid
- invalid_due_leakage: Evidence of data leakage
- invalid_due_metric_bug: Metric computation error found
- unresolved: Cannot determine due to missing artifacts

### Final Decision Labels
- use_as_primary_ranking_result
- use_with_caution
- appendix_only
- do_not_use_until_fixed
- invalid

## 9. Critical Observation from Initial Review

**IMPORTANT**: The original summary states "Best: lightgbm, h=40, k=10" but the final_topk_results.csv shows:
- lightgbm h=40 k=10: final_precision@10 = 38.68%
- lightgbm h=120 k=10: final_precision@10 = 75.54%

The 75.54% is from h=120, NOT h=40. This is a critical discrepancy that must be resolved.

Additionally:
- All 45 experiments pass hit_rate@k >= 65% threshold
- Only 3 experiments pass precision@k >= 65% (all at h=120, k=10)
- hit_rate@10 is 100% for most experiments - likely trivially easy
- Final eval events drop dramatically with horizon: 316 (h=20) -> 56 (h=120)
- With only 56 events, the 75.54% result may not be robust
