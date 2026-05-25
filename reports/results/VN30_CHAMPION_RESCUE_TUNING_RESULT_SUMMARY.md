# VN30 Champion Rescue Tuning Result Summary

## Best Validation Rescue Candidate

- Candidate: `rescue_0003__t0p49`.
- Model: logistic regression.
- Feature group: feature_set_C_closest.
- Horizon: 35.
- Penalty/C/class weight: l1 / 0.1 / None.
- Threshold: 0.49.
- Validation accuracy: 51.74%.
- Validation lift over strongest baseline: +1.40 pp.
- Constraint-passing shortlist: false.
- Validation positive-lift quarters: 1.
- Validation ticker median lift: +3.21 pp.

## Locked Final Result

- Final accuracy: 57.88%.
- Strongest final baseline: vnindex_direction_lag1 at 50.33%.
- Final lift over strongest baseline: +7.55 pp.
- Validation-final gap: +6.14 pp.
- Final rows: 4224.
- Rolling250 min accuracy: 7.20%.
- Ticker median accuracy: 61.84%.
- Quarter min accuracy: 25.14%.
- Acceptance label: diagnostic_only.

## Claim Boundary

- Baseline60 defensible: false.
- Target62 defensible: false.
- Final65 defensible: false.

Paper-safe wording:

> In a validation-locked VN30 stock hourly champion-rescue diagnostic benchmark, the selected logistic candidate reached 57.88% final pooled directional accuracy over 4224 rows, +7.55 pp versus the strongest same-horizon simple baseline. This supports only the stated diagnostic benchmark scope and does not support trading, profitability, live deployment, VN100, top-k-as-overall-accuracy, or final65 claims.
