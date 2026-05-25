# VN30 Legacy Targeted Improvement Claim Boundary

## Comparator

- Current apples-to-apples best: Logistic L2, `baseline_C_closest`, h40, threshold 0.55, final accuracy 61.63%.
- Old reference: 61.51%.
- Majority baseline: 50.44%.

## Result

- Best selected targeted track by final score after validation-only selection: `targeted__exante_regime_threshold_router__logistic_l2_h40`.
- Final accuracy: 61.36%; delta vs current: -0.27 pp; classification: `failed_improvement`.
- Rolling stability vs current: `preserved`.

## Boundary

- No final-window score was used for model, threshold, feature, ticker repair, regime router, or track selection.
- No ticker subset, confidence abstention, top-k/ranking substitution, stacking main candidate, or market-data fetch was used.
- This is a directional accuracy benchmark only; it makes no trading, profitability, or live-deployment claim.
- No targeted track established a stronger candidate beyond the current 61.63% comparator.

## Non-Selected Final Observation

- `targeted__regularized_linear__logistic_l2__c0p2__rna__t0p6` scored 61.71% on final, +0.07 pp vs current.
- It was not selected by the validation-only track objective, so it is not a stronger accepted candidate in this run.
