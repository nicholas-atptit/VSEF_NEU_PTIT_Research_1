# VN30 Hourly Validation-Safe Improvement Tracks Protocol

## Purpose

This protocol defines a validation-safe improvement attempt for VN30 stock-only hourly overall directional accuracy. The experiment uses only existing local data and artifacts. It does not fetch new market data and does not alter provider behavior.

## Reference Result

- Reference candidate: L2 Logistic, h=40, `feature_set_C_closest`, threshold 0.50.
- Reference final accuracy: 61.51%.
- Reference final rows: 4,074.
- Reference majority baseline: 50.44%.
- Reference lift over majority: +11.07 percentage points.
- Reproduction row-level difference: 0.000000 percentage points.
- Rolling stability: mixed.
- Claim level: exploratory improved baseline evidence.
- Final65: not established.

Final65 is aspirational only in this experiment. A Final65 candidate can be reported only if it is selected by validation-only rules, uses full 30-stock coverage, passes leakage and coverage checks, improves rolling/month/quarter/regime stability, and remains exploratory unless confirmed by a future blind test.

## Main Target

The main target is VN30 stock-only hourly overall directional accuracy across the full 30-stock active January 2025 VN30 universe. Full 30-stock coverage is required for the main claim.

Index data may be used only as lagged market-context features. Index-only results and stock-index joint-panel results are contextual diagnostics only and must not replace or upgrade stock-only claims.

## Prohibited Main-Claim Substitutions

The main claim must not use:

- Confidence abstention.
- Ticker subset filtering.
- Top-k or ranking metrics as a substitute for overall directional accuracy.
- Final-window score for feature, model, horizon, or threshold selection.
- Any trading, profitability, investment recommendation, or live-deployment claim.

## Candidate Selection

Candidate selection must use validation data only. The final window is scoring-only.

The main threshold is fixed at 0.50. Optional validation-selected threshold diagnostics may be reported separately, but the main claim prioritizes threshold 0.50 unless a validation-only rule is explicitly preregistered and justified.

The validation-only selection score is:

`validation_accuracy + lift_over_validation_majority + monthly_stability_score + quarterly_stability_score + regime_stability_score_if_available - instability_penalty - overfit_risk_penalty`

The final-window score must not contribute to the selected candidate, feature family, model family, horizon, or threshold.

## Feature Families

All added features must be lagged or otherwise ex-ante. Future regime labels, future returns, final-label-derived features, final-period manual filters, and same-row target leakage are prohibited.

Feature families:

- `baseline_C_closest`: existing `feature_set_C_closest`.
- `regime_context`: lagged VNINDEX/VN30 returns, lagged VNINDEX/VN30 volatility, lagged market direction, and ex-ante regime-like proxies only.
- `breadth_context`: percentage of VN30 stocks positive at the prior timestamp, lagged average VN30 return, lagged dispersion of VN30 returns, and lagged market breadth trend.
- `relative_strength`: stock return minus VNINDEX return, stock return minus VN30 index return, rolling relative momentum, and rolling relative volatility.
- `volatility_normalized`: return divided by rolling volatility, rolling z-score return, high-low range shock, and volume shock.
- `interaction_context`: stock momentum times lagged market direction, relative strength times lagged market volatility, and volatility-normalized momentum times breadth.

## Comparisons

All improvements must be compared against:

- 61.51% reproduced L2 Logistic h=40 reference.
- 50.44% majority baseline reference.
- Historical RF h=60 reference if available.
- Current rolling stability reference when available.

## Acceptance Classification

Classify the final result as:

- `stronger_baseline60_candidate`: final accuracy > 61.51%, validation-selected, full 30-stock coverage, no leakage, rolling stability not worse than reference, and validation-final gap not materially worse.
- `weak_improvement`: final accuracy improves but stability worsens materially.
- `failed_improvement`: final accuracy <= 61.51%, validation selection fails, or leakage/coverage rules fail.
- `final65_candidate`: final accuracy >= 65%, validation-selected, full 30-stock coverage, no leakage, stronger rolling/month/quarter/regime stability, not selected by final-window score, and still marked exploratory unless confirmed by a future blind test.

