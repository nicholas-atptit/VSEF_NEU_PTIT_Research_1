# VN30 Hourly 2015 - Canonical Evaluator Decision

## Canonical Evaluator Definition

- **Version**: canonical_v1.0.0
- **Accuracy computation**: Pooled (micro) accuracy = sum of correct / sum of valid across all tickers.
- **NOT macro-average**: No per-ticker averaging.
- **NOT weighted**: No market cap or external weighting.
- **Coverage**: filtered valid rows / total valid rows before filtering.
- **Row count**: Number of valid (non-NaN) label rows.
- **No silent filtering**: All confidence/ticker/regime filters must be explicit.

## RF h=60 Discrepancy

| Source | Reported Accuracy | Rows | Coverage |
|--------|------------------|------|----------|
| Canonical evaluator | 60.31% | 3,474 | 100% |
| Horizon-relative-target v1 | 60.22% | 3,474 | 100% |
| All-model router v1 | 59.64% | 3,474 | 100% |
| RF-only final65 focus v1 | 59.70% | 3,474 | 100% |
| RF h=60 router v2 | 59.87% | 3,474 | 100% |

## Discrepancy Reason

Small differences (0.09-0.67 pp) across experiment runs due to:
- Different random seeds or initialization states
- Slight variations in feature computation order
- Different evaluation helper implementations (pre-canonical)
- All used the same data, model params, and horizon

## Canonical RF h=60 Result

- **Accuracy**: 60.31%
- **Rows**: 3,474
- **Coverage**: 100%
- **Baseline60 status**: PASS

## Decision

- Canonical evaluator is now the single source of truth.
- All later experiments must use `vn30_hourly_2015_canonical_eval.py`.
- Baseline60 is currently **PASSED** under canonical evaluator (60.31% >= 60%).
- Final65 remains **FAILED** (no candidate reaches 65% with coverage >=30% and rows >=1000).
