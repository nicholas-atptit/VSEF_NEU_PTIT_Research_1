# VN30 Legacy Reference Reproduction

- Model: L2 Logistic.
- Horizon: h40.
- Feature family: `baseline_C_closest` / `feature_set_C_closest`.
- Threshold: 0.50.
- Train rows: 9600.
- Validation rows: 30030.
- Final rows: 4074.
- Feature count: 99.
- Ticker coverage: 30/30.
- Validation accuracy: 51.88%.
- Reproduced final accuracy: 61.51%.
- Difference vs 61.51% reference: +0.00 pp.
- Reproduction passed: yes.

## Boundary

- Data fetched: no.
- Provider behavior changed: no.
- Final window role: scoring-only.
- Trading/profitability/live-deployment claim: no.
