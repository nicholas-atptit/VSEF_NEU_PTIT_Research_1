# VN30 Legacy Targeted Improvement Data Alignment Audit

## Split And Label Availability

- Horizon: h40 absolute close-direction label.
- Train boundary: `<= 2023-12-31 23:59:59`.
- Validation boundary: `2024-01-01 00:00:00` through `2024-12-31 23:59:59`.
- Final boundary: `>= 2025-01-01 00:00:00`.
- Label-available rows: 43,704.
- Label-missing rows: 1,200.
- Split rows: train 9,600, validation 30,030, final 4,074.
- Final rows expected for apples-to-apples comparison: 4,074; observed: 4,074.

## Stock Bar Integrity

- Active ticker count: 30.
- Duplicate `(ticker, datetime)` stock rows: 0.
- Max hourly rows by ticker: 1,651.
- Missing-row proxy versus max ticker count: total 4,626, max per ticker 230.
- Abnormal one-bar stock return spikes above 20% absolute: 14.

## Index Context Alignment

- Local index codes loaded: HNXINDEX, UPCOMINDEX, VN30, VNINDEX.
- Index duplicate timestamp rows: {'VNINDEX': 0, 'VN30': 0, 'HNXINDEX': 0, 'UPCOMINDEX': 0}.
- Index features used by targeted tracks are lagged context columns from the existing feature builder; no final-window labels or future returns are used as features.

## Boundary

- Data fetch: no.
- Provider behavior changed: no.
- Ticker subset used: no.
- Confidence abstention/top-k used: no.
