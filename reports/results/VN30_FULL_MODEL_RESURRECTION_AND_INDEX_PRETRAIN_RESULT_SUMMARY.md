# VN30 Full Model Resurrection And Index-Pretrain Result Summary

## Historical 63% Candidate

- Found old around-63% candidate: `universe__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__fixed_0p50__t0p500`.
- Source: `reports/generated/vn30_model_universe_benchmark/final_results.csv`.
- Old model/feature/horizon/threshold: bull_bear_sideway_router / baseline_C_closest / h40 / 0.5.
- Old accuracy: 63.33%.
- Strict replay status: survives as baseline60 but not at the old 63% level.
- Strict replay accuracy: 61.61%.
- Strict replay retained old 63% level: false.

## Validation-Governed Result

- Locked candidate: `grid_090055__t0p450`.
- Selected from strict shortlist: true.
- Best validation-governed candidate: `grid_090055__t0p450`.
- Best validation-governed final accuracy: 51.48%.
- Best validation-governed final lift: -0.64 pp.
- Locked final accuracy: 51.48%.
- Locked final lift: -0.64 pp.
- Locked final rows: 4674.

## Exploratory Final Result

- Best exploratory final candidate: `grid_334693__t0p525`.
- Best exploratory final accuracy: 64.76%.
- Best exploratory final lift: +11.18 pp.
- Exploratory final candidate exceeded 62%: true.
- Exploratory final candidate exceeded 63%: true.

## Champion Decision

- Current champion: L2 Logistic, feature_set_C_closest, h40, threshold 0.50, 61.61% final accuracy, +10.90 pp lift, 4,074 rows.
- New champion replaces current champion: false.
- Reason: no validation-governed candidate beat 61.61% final accuracy; no validation-governed candidate beat +10.90 pp lift.

## Claim Boundary Answers

1. Old 63% candidate found: `universe__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__fixed_0p50__t0p500` at 63.33%.
2. Strict replay survival: survives as baseline60 but not at the old 63% level.
3. Any validation-governed candidate beat 61.61%: false.
4. Any candidate beat +10.90 pp lift: true.
5. Exploratory final exceeded 62% or 63%: 62=true, 63=true.
6. Claimable result: not_claimable.
7. Exploratory-only result: best final-ranked rows in `exploratory_final_leaderboard.csv`.
8. Baseline60 defensible: false; target62 defensible: false; final65 defensible: false.

Paper-safe wording:

> In a validation-governed VN30 stock hourly diagnostic benchmark with lagged market-index context features, the locked candidate reached 51.48% final pooled directional accuracy over 4674 rows, -0.64 pp versus the strongest same-horizon simple baseline. The current strict-replay L2 Logistic h40 champion at 61.61% and +10.90 pp is not replaced. Final-ranked rows outside the validation-governed lock are exploratory only and require re-locking or future-blind confirmation.
