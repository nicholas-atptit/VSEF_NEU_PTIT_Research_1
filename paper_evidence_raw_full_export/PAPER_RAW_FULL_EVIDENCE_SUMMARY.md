# Paper Raw Full Evidence Summary

## Provenance

- Branch: `research/vn30-legacy-rules-reference-and-stacking-v1`
- Commit: `35078143af530c18e0576d071c2471d2f923b186`
- Benchmarks rerun: no
- Model training run: no
- Data fetch run: no
- Paper/DOCX/PDF generated: no

## Export Counts

- Raw data files copied: 247
- Model output files copied: 235
- Prediction artifacts copied: 11
- Model-horizon rows: 4866
- Per-instrument actual-vs-predicted rows: 2538
- Row-level prediction rows: 2711484
- Missing model-horizon rows: 1514
- Missing per-instrument rows: 0

## Coverage Checks

- All 30 VN30 tickers included yes/no: yes
- All six supported indices included yes/no: yes
- h20/h40/h60/h80 included yes/no: yes
- KNN-support actuals included yes/no: yes
- Model-cooperation actuals included yes/no: yes

## Key Results From Local Artifacts

- Best descriptive stock model: bull_bear_sideway_router / universe__regime_aware_models__bull_bear_sideway_router__baseline_C_closest__h40__fixed_0p50__t0p500 final=0.6332842415316642
- h20: xgboost / fullhorizon__classical_ml__xgboost__baseline_C_closest__h20__validation_selected_threshold__t0p450 final=0.517116
- h40: logistic_l2 / fullhorizon__classical_ml__logistic_l2__baseline_C_closest__h40__validation_selected_threshold__t0p550 final=0.616348
- h60: logistic_elastic_net / fullhorizon__classical_ml__logistic_elastic_net__baseline_C_closest__h60__validation_selected_threshold__t0p550 final=0.560161
- h80: logistic_elastic_net / fullhorizon__classical_ml__logistic_elastic_net__baseline_C_closest__h80__validation_selected_threshold__t0p550 final=0.523312

## Practical Experimental Usability Ranking

- Ranking table: `tables/table_07_descriptive_practical_ranking.csv`
- Ranking name: `practical_experimental_usability_ranking`

## Boundaries

- Index context does not replace stock-only evidence.
- This export is not a trading, profitability, investment, or live-deployment claim.
- Missing values are explicitly marked with `missing_with_reason`; no values were invented.
