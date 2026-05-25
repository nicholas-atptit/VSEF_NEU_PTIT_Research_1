# Joint Panel Baseline Report

Baselines are deterministic and do not train models.

- Best deterministic combined baseline: `majority_class` h=120 at 59.40%.
- Stock-only RF h=60 historical reference: 60.31%; reference only, not a replacement for joint-panel scoring.
- Existing index benchmark best results are reference-only and cannot replace combined 36-instrument scoring.

## Baseline Summary

| baseline | horizon | combined_accuracy | stock_only_accuracy | index_only_accuracy | combined_rows | stock_instrument_count | index_instrument_count | reference_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_class | 40 | 0.5365034965034965 | 0.5244946899623159 | 0.555475374346022 | 14300 | 30 | 6 | False |
| always_up | 40 | 0.5365034965034965 | 0.5244946899623159 | 0.555475374346022 | 14300 | 30 | 6 | False |
| previous_direction | 40 | 0.4935664335664336 | 0.4942331848806669 | 0.49251307955980517 | 14300 | 30 | 6 | False |
| moving_average_signal | 40 | 0.493986013986014 | 0.5018842069201781 | 0.481508208551326 | 14300 | 30 | 6 | False |
| majority_class | 60 | 0.5404775360271263 | 0.5279610976033345 | 0.5600652292081899 | 14156 | 30 | 6 | False |
| always_up | 60 | 0.5404775360271263 | 0.5279610976033345 | 0.5600652292081899 | 14156 | 30 | 6 | False |
| previous_direction | 60 | 0.4987990957897711 | 0.493226814866273 | 0.5075194781663345 | 14156 | 30 | 6 | False |
| moving_average_signal | 60 | 0.4923707261938401 | 0.4940372814634711 | 0.48976263815908677 | 14156 | 30 | 6 | False |
| majority_class | 80 | 0.5765058521267485 | 0.5786074909005519 | 0.5732484076433121 | 14012 | 30 | 6 | False |
| always_up | 80 | 0.5765058521267485 | 0.5786074909005519 | 0.5732484076433121 | 14012 | 30 | 6 | False |
| previous_direction | 80 | 0.49536111904082214 | 0.4837384055418575 | 0.513375796178344 | 14012 | 30 | 6 | False |
| moving_average_signal | 80 | 0.49386240365401085 | 0.4876130092755665 | 0.5035486806187444 | 14012 | 30 | 6 | False |
| majority_class | 100 | 0.5902076723391981 | 0.594855305466238 | 0.5830743922500456 | 13868 | 30 | 6 | False |
| always_up | 100 | 0.5902076723391981 | 0.594855305466238 | 0.5830743922500456 | 13868 | 30 | 6 | False |
| previous_direction | 100 | 0.4888231900778771 | 0.48005239966654756 | 0.5022847742643027 | 13868 | 30 | 6 | False |
| moving_average_signal | 100 | 0.4962503605422556 | 0.48934143146361797 | 0.5068543227929081 | 13868 | 30 | 6 | False |
| majority_class | 120 | 0.5939959195569805 | 0.5999758366557931 | 0.5849091242885992 | 13724 | 30 | 6 | False |
| always_up | 120 | 0.5939959195569805 | 0.5999758366557931 | 0.5849091242885992 | 13724 | 30 | 6 | False |
| previous_direction | 120 | 0.48877878169629846 | 0.47625951431678143 | 0.5078024600697632 | 13724 | 30 | 6 | False |
| moving_average_signal | 120 | 0.49169338385310407 | 0.4883411864202006 | 0.4967872223242152 | 13724 | 30 | 6 | False |
| existing_stock_hourly_rf_h60_reference | 60 | nan | 0.6031 | nan |  | 30 | 0 | True |
