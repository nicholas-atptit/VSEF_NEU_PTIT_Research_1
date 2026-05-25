# VN30 Legacy Rules Comparison Summary

- Reference reproduced: yes at 61.51% with 4074 final rows.
- Best new single model: `legacy_single__logistic_l2__baseline_C_closest__h40__validation_selected_threshold__t0p550` final 61.63%, delta +0.12 pp.
- Best stacking/ensemble: `legacy_stack__meta_lightgbm_stacking` final 48.72%, delta -12.79 pp.
- Apples-to-apples: yes.
- Acceptance classification: `single_model_improvement`.
- Data fetched: no.
- Final selection source: validation only.
