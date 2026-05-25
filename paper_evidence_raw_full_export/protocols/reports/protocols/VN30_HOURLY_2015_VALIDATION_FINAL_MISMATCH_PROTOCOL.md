# VN30 Hourly 2015 - Validation/Final Mismatch Investigation Protocol

## Purpose
Investigate why validation accuracy (~50%) is much lower than final accuracy (~59-60%) and why rolling validation windows A/B failed in V2.

## 1. Data Availability
- Actual first timestamp per ticker
- Actual last timestamp per ticker
- Usable rows by year per ticker
- Whether the track truly has enough rows before 2022 and 2023
- Whether the "2015" design is nominal while actual hourly coverage begins later

## 2. Feature Warmup
- Rows lost by feature generation
- Rows lost by horizon label shift
- Rows lost by rolling windows
- Why Window A and B failed
- Whether warmup requirements are too aggressive

## 3. Split Design
- Window A: train 2015–2021, validation 2022
- Window B: train 2015–2022, validation 2023
- Window C: train 2015–2023, validation 2024
- Final: 2025–2026
- Verify each split has enough rows after feature/label construction

## 4. Label Construction
- Verify horizon h=40/h=60/h=80/h=120 labels
- Verify target_timestamp alignment
- Verify no off-by-one shift
- Verify actual_direction definition is consistent

## 5. Distribution Shift
Compare validation windows and final window:
- Class balance
- Return distribution
- Volatility distribution
- Ticker-level accuracy
- Market index trend
- Bull/bear/sideway proxy
- Hour/session distribution if available

## 6. Evaluator Consistency
- Verify canonical evaluator is used
- Verify pooled accuracy calculation
- Verify no macro/micro mismatch
- Verify full coverage denominator

## 7. Decision
Classify mismatch as:
- data_coverage_limitation
- feature_warmup_limitation
- validation_window_not_representative
- label_alignment_issue
- evaluator_issue
- unresolved
