# VN30 Aggressive Model Tuning Protocol

## Scope

- Scope: VN30 stock hourly selected-candidate diagnostic benchmark using local stock hourly rows.
- Market-context scope: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30 only where local rows exist.
- Out of scope: VN100, trading, profitability, BUY/SELL, live deployment, DOCX/paper artifacts, git tags.

## Split Discipline

- Train rows require feature timestamp <= `2023-12-31 23:59:59` and target_timestamp <= `2023-12-31 23:59:59`.
- Validation rows require feature timestamp and target_timestamp between `2024-01-01 00:00:00` and `2024-12-31 23:59:59`.
- Final rows require feature timestamp and target_timestamp >= `2025-01-01 00:00:00`.
- Model, hyperparameter, threshold, and ensemble selection use validation only.
- Final is evaluated once after writing `locked_candidate.json`.

## Tuning

- Horizons: [20, 30, 40, 50, 60].
- Thresholds: 0.45 to 0.60 step 0.01.
- Requested model grids are enumerated in `candidate_grid.csv`.
- Fit budget per model family for this run: {'logistic_regression': 30, 'random_forest': 12, 'xgboost': 12, 'lightgbm': 12}.
- Baseline comparison uses the strongest simple baseline for the same split and horizon.

## Selection Rule

Validation-only rank order:

1. validation accuracy
2. positive lift over strongest validation baseline
3. row count
4. rolling stability
5. ticker stability
6. lower train-validation gap as a validation-final mismatch proxy
7. simpler model if tied
