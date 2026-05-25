# VN30 Hourly 2015 - Top-K 75% Verification Decision

## Executive Summary

### 1. Can the 75.54% precision@10 result be used?
**Answer**: DO_NOT_USE_UNTIL_FIXED

**Critical Finding**: The 75.54% precision@10 result is BELOW the random baseline (77.36%). The model is performing WORSE than random selection. This completely invalidates the original claim.

### 2. Is the metric correctly computed?
**Answer**: YES (but result is below random)

- precision@k = overlap(predicted_top_k, true_top_k) / k is correctly implemented
- Recomputed values match original within tolerance
- However, the result is not meaningful as it's below random baseline

### 3. Is there evidence of leakage?
**Answer**: NO

- Feature audit found no suspicious columns
- All features are backward-looking

### 4. Is there evidence of overfitting?
**Answer**: HIGH RISK

- Model performs BELOW random baseline
- Validation precision@10 (h=120): ~34%
- Final evaluation precision@10 (h=120): 74.46%
- Random baseline: 77.36%
- Model is 2.9 percentage points BELOW random

### 5. Is the result robust over time?
**Answer**: NO

- Only 56 events in final evaluation
- Result is not statistically significant (p-value = 0.999)

### 6. Is the result meaningfully above random/momentum baselines?
**Answer**: NO

- Random baseline: 77.36%
- Momentum baseline: 74.46%
- Model precision@10: 74.46%
- Model is BELOW random baseline
- Model is EQUAL to momentum baseline

### 7. Is hit_rate@10 too easy to use as a headline?
**Answer**: YES

- hit_rate@10 is near 100% across almost all configurations
- With 30 tickers and k=10, probability of at least one overlap is very high
- hit_rate@10 is a weak/easy metric

### 8. Should precision@10 be the primary metric?
**Answer**: YES (but result fails)

- precision@10 is more rigorous than hit_rate@10
- However, the model fails to beat random baseline

### 9. What exact wording is allowed?
**Allowed**:
"LightGBM h=120 achieved 74.46% precision@10 in the top-k ranking task, which is below the random baseline of 77.36%. The result is not statistically significant (p-value = 0.999)."

### 10. What exact wording is forbidden?
**Forbidden**:
- "75.54% directional accuracy"
- "75.54% price prediction accuracy"
- "profitable trading strategy"
- "live deployment ready"
- "guaranteed top stocks"
- "LightGBM h=40 achieved 75.54%" (INCORRECT - it's h=120)
- Any claim that the model outperforms random selection

## Decision Label

**do_not_use_until_fixed**

## Conditions for Use

1. **DO NOT USE** the 75.54% result as evidence of model skill
2. The model performs BELOW random baseline
3. Result is not statistically significant (p-value = 0.999)
4. Must disclose configuration discrepancy (h=120, not h=40)
5. Must disclose small sample size (56 events)
6. Must disclose metric change from directional accuracy

## Verification Status

| Check | Status |
|-------|--------|
| Metric correctness | PASS |
| Grouping correctness | PASS |
| Target correctness | PASS |
| Leakage audit | PASS |
| Overfitting audit | FAIL |
| Baseline comparison | FAIL (below random) |
| Null test | FAIL (p=0.999) |
| Temporal robustness | FAIL |
| Ticker concentration | PASS |

## Null Test Results

- Observed precision@10: 74.46%
- Random baseline mean: 77.36% +/- 0.95%
- Empirical p-value: 0.999
- Score shuffle mean: 77.38% +/- 0.89%
- Label shuffle mean: 77.42% +/- 0.91%

## Conclusion

The 75.54% precision@10 result is **INVALID** as evidence of model skill. The model performs BELOW random selection, and the result is not statistically significant. The original claim must be retracted or corrected.

## Next Steps

1. Retract or correct the original claim
2. Investigate why model performs below random
3. Consider alternative approaches or features
4. Do not use this result in any paper or report
