# VN30 Hourly Performance Push V2 Protocol

## Purpose

This is an audited VN30 hourly stock-only performance-push experiment. It tests whether final overall directional accuracy can be improved beyond the current 61.51% reference while preserving explicit validation-only selection and audit boundaries.

This experiment is not automatically a paper claim upgrade. Any improvement must be classified by overfit risk and separated from strict validation-safe claims.

## Reference

- Current selected candidate: L2 Logistic, h=40, `feature_set_C_closest`, threshold 0.50.
- Reference final accuracy: 61.51%.
- Reference final rows: 4,074.
- Reference majority baseline: 50.44%.
- Reference lift over majority: +11.07 percentage points.
- Reference validation accuracy: 51.88%.
- Reference validation-final gap: +9.63 percentage points.
- Rolling stability: mixed.
- Previous strict validation-safe improvement search: no stronger strict candidate.
- Reference overfit risk: high.

## Hard Boundary

The final score remains scoring-only. Candidate choice must be based on validation-only rules. The final window must not be used to select model, feature set, horizon, threshold, ticker calibration, ensemble weights, per-ticker model, or router decision.

The headline result must use full 30-stock VN30 stock-only coverage. The main reported full-universe result must not use confidence abstention, ticker subset filtering, or top-k/ranking substitution.

No new market data may be fetched. Provider behavior must not change. No trading, profitability, investment recommendation, or live-deployment claim is allowed.

## Allowed Performance-Push Methods

The experiment may include:

- Fixed-threshold global models at threshold 0.50.
- Validation-selected threshold global models using the grid 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60.
- Probability calibration, including Platt and isotonic calibration fitted on validation rows only.
- Soft-vote ensembles using base predictions created without final leakage.
- Per-ticker calibration using ticker-specific thresholds selected on validation only.
- Per-ticker model families selected on validation only.
- Hybrid routers that choose between global and per-ticker candidates using validation ticker-level performance only.

Per-ticker models and per-ticker calibration are allowed only if all 30 tickers remain covered in final scoring.

Validation-selected thresholds must be reported separately from fixed-threshold 0.50 results.

## Candidate Families

Candidate families:

- Fixed-threshold global models: Logistic L2, Logistic Elastic Net, Random Forest, ExtraTrees, XGBoost, LightGBM, HistGradientBoosting.
- Validation-selected threshold global models.
- Probability calibrated models: Platt and isotonic if validation rows are sufficient.
- Soft-vote ensembles: unweighted average, validation-accuracy weighted, validation-lift weighted, and stability-weighted average probability.
- Per-ticker calibration.
- Per-ticker model family.
- Hybrid router.

Feature sets:

- `baseline_C_closest`
- `regime_context`
- `breadth_context`
- `relative_strength`
- `volatility_normalized`
- `interaction_context`
- `combined_context`
- `compact_top_features`

Horizons:

- h=20
- h=40
- h=60
- h=80

The primary target is h=40 absolute direction. Other horizon-specific results may be reported only as explicitly horizon-specific diagnostics.

## Selection Policies

All policies are validation-only:

- `max_validation_accuracy`
- `max_validation_lift_over_majority`
- `validation_accuracy_with_monthly_stability`
- `validation_accuracy_with_ticker_stability`
- `validation_accuracy_with_rolling_proxy_stability`
- `balanced_score`

For each policy, the selected candidate is saved first, then the final window is scored once for that selected candidate.

## Classification

Every policy result must be classified as one of:

- `strict_validation_safe`
- `exploratory_performance_push`
- `likely_overfit`
- `rejected_due_to_leakage_or_selection_risk`

Acceptance outcomes:

- `stronger_candidate`: final accuracy > 61.51%, full 30 coverage, no leakage, validation-only selection, and rolling stability not materially worse.
- `exploratory_accuracy_gain`: final accuracy > 61.51%, but validation-final gap or rolling stability worsens.
- `likely_overfit`: final accuracy improves but validation score is weak, validation-final gap increases materially, or rolling stability collapses.
- `failed_push`: no policy exceeds 61.51%.
- `final65_candidate`: final accuracy >= 65%, validation-only selected, full 30 coverage, no leakage, stability improves, not final-tuned, and still exploratory until future blind validation.

