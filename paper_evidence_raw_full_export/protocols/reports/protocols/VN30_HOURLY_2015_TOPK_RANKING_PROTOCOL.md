# VN30 Hourly 2015 - Top-K Ranking Protocol

## 1. Why Switch Metric
- Directional final65 failed after multiple valid optimization phases.
- Ranking may be more suitable for stock selection than full-universe binary direction.
- This changes the claim type and must be reported separately.

## 2. Ranking Target
For each timestamp and horizon:
- Compute future stock return.
- Compute cross-sectional rank among active VN30 tickers.
- Define top-k winners and optionally bottom-k losers.

### Candidate Labels
A. `top_k_outperformer`: stock is in top k future returns.
B. `above_median_outperformer`: stock future return above VN30 cross-sectional median.
C. `relative_to_vn30_top_k`: stock return minus VN30 index return, ranked cross-sectionally.
D. `top_bottom_spread`: classify top-k vs bottom-k only; middle ignored.

## 3. Candidate k
- k = 3
- k = 5
- k = 10

## 4. Metrics
- precision@k
- recall@k
- hit_rate@k
- mean future return of selected top-k
- rank IC if supported
- coverage / number of timestamp events
- number of selected stock-events

## 5. Success Target
- primary: precision@k >=65% on final evaluation
- secondary: hit_rate@k >=65%
- must disclose k, rows/events, and coverage
- no trading/profitability claim

## 6. Selection
- select model/horizon/k/policy using 2024 validation only
- final evaluation used only once for scoring

## 7. Forbidden
- final-label-selected k
- final-label-selected ticker list
- profitability/live trading claims
- comparing precision@k directly to full-universe directional accuracy without stating metric change

## 8. Boundary
- No trading-readiness, profitability, or live deployment claim.
- All selection on 2024 validation only, leakage-safe.
- Canonical evaluator v1.0.0 used for directional metrics; ranking metrics computed separately.
