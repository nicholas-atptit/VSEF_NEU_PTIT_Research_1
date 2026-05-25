# KNN Support Leakage Audit

- Leakage passed: yes.
- Overfit risk for selected KNN-support candidate: high.
- Overfit risk reason: rolling 250 mean 48.40%; 2333 rolling 250 windows below 60%; monthly minimum 20.04%; quarterly minimum 27.67%; ticker minimum 18.38%.

| check | passed | detail |
| --- | --- | --- |
| scaler_train_only | yes | StandardScaler fit on h40 train rows only |
| knn_index_train_only | yes | validation KNN index uses train rows only; train features use leave-one-out train rows |
| validation_neighbors_train_only | yes | validation neighbor labels come from train only |
| final_neighbors_ex_ante | yes | final neighbor labels come from train+validation only after validation lock |
| no_final_label_use | yes | final labels not used in KNN features, calibration, thresholds, weights, or meta fit |
| full_coverage | yes | selected rows have 30-stock coverage |
| no_ticker_subset | yes |  |
| no_confidence_abstention | yes |  |
| no_topk_substitution | yes |  |
