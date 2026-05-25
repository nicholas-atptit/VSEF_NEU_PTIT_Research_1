# VN30 V8 Strategy Target Redesign Protocol

## Scope

- VN30 stock hourly strategy diagnostics only.
- Target variants are market-relative, top-quantile, cost-adjusted, and neutral-removed labels.
- Main index data is lagged market context only; index benchmark results are not stock strategy claims.
- Validation-only model/target/feature/strategy selection; final-ranked rows are exploratory_not_claimable.
- No broad brute-force grid, trading recommendation, live deployment, DOCX, push, merge, or tag.

## Staged Search

- Model screening: broad logistic target/feature/horizon screen plus h40 combined-feature model-family sweep.
- Strategy grid: cross-sectional rank rotation templates on validation-shortlisted model candidates.
- Locked final: one validation-selected strategy evaluated once on final.
