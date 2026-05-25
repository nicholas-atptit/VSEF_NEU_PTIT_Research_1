# VN30 Hourly Track A Diagnostic 65 Row Audit Protocol

## Scope

- Setup: Track A canonical-like VN30 stock-only hourly setup.
- Diagnostic row: L2 Logistic h40, `regime_feature_v2`, `global_v2`.
- Diagnostic final accuracy: 65.71%.
- Prior clean baseline60 result: Logistic Regression h40 = 60.43%.
- Historical RF h60 reference: 60.31%.

## Claim Boundary

The L2 Logistic h40 65.71% row is diagnostic only. It is not a claim, not a final65 result, and not a replacement for the validation-selected Track A result.

The goal of this audit is to understand why validation did not select the row and whether the row looks like validation-selection mismatch, final-window luck, or a potentially valid candidate that would require a new pre-registered validation-only rule.

Any future use of this row, model class, or feature setup requires a new validation-only selection rule specified before final scoring or a future blind test.

## Controls

- No new model families.
- No final-label selection.
- No confidence abstention.
- No ticker subset.
- No top-k/ranking substitution.
- No data fetch.
- No paper/DOCX generation.
- No trading, profitability, investment-recommendation, or live-deployment claim.
