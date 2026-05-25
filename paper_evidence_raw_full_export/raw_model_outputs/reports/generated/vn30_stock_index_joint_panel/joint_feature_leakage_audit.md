# Joint Panel Feature Leakage Audit

- Future own returns: not used.
- Future index returns: not used.
- Future market regime: not used.
- Target direction leakage: not used.
- Target timestamp leakage: not used.
- Final-period hand-crafted filters: not used.
- Final accuracy in model selection: forbidden by protocol; training script selection uses validation metrics only.

Index context features are lagged before as-of joining into instrument rows. Cross-sectional features use prior returns/volatility/volume shock at each timestamp.
