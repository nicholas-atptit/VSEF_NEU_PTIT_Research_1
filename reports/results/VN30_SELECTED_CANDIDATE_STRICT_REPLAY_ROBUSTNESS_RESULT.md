# VN30 Selected Candidate Strict Replay Robustness Result

## Fixed Champion

- Candidate: L2 Logistic, `feature_set_C_closest`, h40, threshold 0.50.
- Final accuracy: 61.61%.
- Strongest final baseline: vnindex_direction_lag1 at 50.71%.
- Final lift over strongest baseline: +10.90 pp.
- Baseline60 diagnostic claim remains defensible: true.
- Target62 and final65: not defensible.

## Quarter Stability

- Strongest quarter: 2026Q1, 75.98%, lift +0.08 pp.
- Weakest quarter: 2025Q2, 47.54%, lift -47.54 pp.
- Quarters below 50% accuracy: 2.

## Ticker Stability

- Strongest ticker: MBB, 96.32%, lift +36.03 pp.
- Weakest ticker: VIC, 33.33%, lift -33.33 pp.
- Tickers below 50% accuracy: 7.

## Rolling Stability

- Rolling250 min/p10/median/p90/max: 7.60% / 46.40% / 63.60% / 83.20% / 92.80%.
- Rolling500 min/p10/median/p90/max: 31.40% / 40.80% / 62.40% / 84.60% / 89.40%.
- Rolling1000 min/p10/median/p90/max: 43.70% / 50.40% / 61.20% / 79.50% / 82.00%.
- Rolling instability remains severe: true.

## Prediction Balance

- Overall up ratio: 31.35%.
- Overall down ratio: 68.65%.

## Concentration Disclosure

- Concentration/weak-slice risk present: true.
- Failed slices are retained in the CSV artifacts and not hidden.

No trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, paper, index-as-stock, or top-k-as-overall claim is made.
