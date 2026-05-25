# VN30 Stock + Index Joint Panel Data Source Recovery Protocol

## Purpose

This protocol governs an audit-only recovery pass for the VN30 stock + supported-index joint panel.

The previous joint-panel training run was correctly gated. It did not train models because the cached data available to the script did not satisfy validation-safe 36-instrument hourly readiness.

## Blocker

The blocker is data-source readiness, not model performance.

The current joint-panel script likely read an insufficient stock cache because prior stock-hourly baseline60 evidence exists for the VN30 active universe. The recovery task is to find the source behind that evidence and determine whether those source rows still exist locally.

Index files must be verified as true intraday hourly rows. Files with only midnight timestamps are daily-like rows and cannot be treated as validation-safe intraday-hourly inputs, even if a filename or metadata field says hourly.

## Joint Universe

- Stocks: 30 active VN30 January 2025 stock tickers.
- Indices: `VNINDEX`, `VN30`, `HNXINDEX`, `HNX30`, `UPCOMINDEX`, `VN100`.
- Indices must be prediction rows in the panel, not merely context features.

## Guardrails

- Do not run benchmark training in this pass.
- Do not fetch market data in this pass.
- Do not alter source data, benchmark outputs, historical summaries, labels, metrics, provider behavior, or research claims.
- Do not generate paper or DOCX artifacts.
- Do not create or push tags.

## Readiness Rule

No joint 36-instrument training should run until 36/36 readiness is established from true hourly stock and true intraday-hourly index inputs.

If local true intraday-hourly index data is unavailable for all six supported indices, the repair decision must recommend a non-training path such as rebuilding from existing raw data, writing a fetch protocol for approval, changing the joint-panel frequency, or keeping hourly stock-only and index benchmark tracks separate.
