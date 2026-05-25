# Worktree Leftover Inventory

Initial audit branch: `research/vn30-quantum-ml-forecasting-v1`

Initial HEAD: `110a0b7b research: start VN30 quantum ML forecasting branch`

Initial counts:

- `git status --short`: 47 entries
- `git ls-files -o --exclude-standard`: 1376 files
- `git diff --name-only`: 6 files

## A. Safe VN30 Research Leftovers Likely Worth Committing

These files and directories appear to be VN30 research artifacts or protocols/results/claims related to completed leftover research work:

- Strict replay files:
  - `reports/claims/VN30_SELECTED_CANDIDATE_STRICT_REPLAY_CLAIM_BOUNDARY.md`
  - `reports/claims/VN30_SELECTED_CANDIDATE_STRICT_REPLAY_ROBUSTNESS_CLAIM_BOUNDARY.md`
  - `reports/results/VN30_SELECTED_CANDIDATE_STRICT_REPLAY_RESULT.md`
  - `reports/results/VN30_SELECTED_CANDIDATE_STRICT_REPLAY_ROBUSTNESS_RESULT.md`
  - `scripts/research/run_vn30_strict_replay_robustness.py`
  - `reports/generated/vn30_selected_candidate_strict_replay/`
  - `reports/generated/vn30_strict_replay_robustness/`
- Aggressive tuning files:
  - `reports/claims/VN30_AGGRESSIVE_MODEL_TUNING_CLAIM_BOUNDARY.md`
  - `reports/protocols/VN30_AGGRESSIVE_MODEL_TUNING_PROTOCOL.md`
  - `reports/results/VN30_AGGRESSIVE_MODEL_TUNING_RESULT_SUMMARY.md`
  - `scripts/research/run_vn30_aggressive_model_tuning.py`
  - `reports/generated/vn30_aggressive_model_tuning/`
- Champion rescue files:
  - `reports/claims/VN30_CHAMPION_RESCUE_TUNING_CLAIM_BOUNDARY.md`
  - `reports/results/VN30_CHAMPION_RESCUE_TUNING_RESULT_SUMMARY.md`
  - `scripts/research/run_vn30_champion_rescue_tuning.py`
  - `reports/generated/vn30_champion_rescue_tuning/`
- Full resurrection/index-pretrain files:
  - `reports/claims/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_CLAIM_BOUNDARY.md`
  - `reports/protocols/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_PROTOCOL.md`
  - `reports/results/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_RESULT_SUMMARY.md`
  - `scripts/research/run_vn30_full_model_resurrection_index_pretrain.py`
  - `reports/generated/vn30_full_model_resurrection/`
- V8 target redesign files:
  - `reports/claims/VN30_V8_STRATEGY_TARGET_REDESIGN_CLAIM_BOUNDARY.md`
  - `reports/protocols/VN30_V8_STRATEGY_TARGET_REDESIGN_PROTOCOL.md`
  - `reports/results/VN30_V8_STRATEGY_TARGET_REDESIGN_RESULT_SUMMARY.md`
  - `scripts/research/run_vn30_v8_strategy_target_redesign.py`
  - `reports/generated/vn30_v8_strategy_target_redesign/`
- Other VN30 generated research artifact directories needing review before staging:
  - `reports/generated/vn30_legacy_rules_reference_and_stacking/robustness_selection/`
  - `reports/generated/vn30_model_universe_benchmark/full_horizon_completion/`

## B. QML Branch Files

- Already committed on QML branch:
  - `reports/protocols/VN30_QML_FORECASTING_PROTOCOL.md`
- Future QML artifact directory, only if intentionally created:
  - `reports/generated/vn30_qml_forecasting/`

## C. Do NOT Commit

Blocked or excluded from commit:

- `paper_evidence_clean_final_export/`
- `paper_evidence_export/`
- `paper_evidence_raw_full_export/`
- `vn30_clean_final_paper_evidence_export.zip`
- `vn30_paper_evidence_export.zip`
- `vn30_raw_full_paper_evidence_export.zip`
- `reports/generated/paper_figure_data.zip`
- `reports/generated/paper_figures.zip`
- `scripts/research/export_vn30_clean_paper_tables_after_completion.py`
- `scripts/research/export_vn30_paper_evidence_package.py`
- `scripts/research/export_vn30_raw_full_paper_evidence.py`
- Ignored caches and machine-local files shown by `git status --ignored --short`, including `.pytest_cache/`, `.venv/`, `.vscode/`, `__pycache__/`, `node_modules/`, `tmp/`, `artifacts/`, `models/`, and ignored `outputs/`.
- Any `.env`, credential, token, virtual environment, local cache, temp, log, DOCX, or PPTX file.
- Data provider raw cache unless explicitly covered by tracked artifact policy.

## D. Needs Human Review

Tracked modified files not clearly tied to completed backfill artifacts:

- `scripts/research/index_benchmark_common.py`
- `scripts/research/run_vn30_daily_2015_benchmark.py`
- `scripts/research/run_vn30_hourly_performance_push_v2.py`
- `scripts/research/run_vn30_hourly_track_a_target62_validation_safe.py`
- `scripts/research/run_vn30_hourly_validation_safe_improvement_tracks.py`
- `tests/ml/test_directional_accuracy_metrics.py`

## Backfill Branch Inventory

Backfill branch: `research/vn30-leftover-artifact-backfill-v1`

Backfill base branch: `research/vn30-legacy-rules-reference-and-stacking-v1`

Backfill counts after stash/restore:

- Total modified tracked files: 6
- Total untracked files: 961
- Selected files staged for this backfill commit: 94
- Stash was used because the QML branch had non-QML dirty and untracked leftovers.
- Git saved the stashes, but some Windows unlink operations returned `Invalid argument`; remaining files were kept in place and staged explicitly only if in scope.

Files/groups that will be committed:

- `reports/cleanup/WORKTREE_LEFTOVER_INVENTORY.md`
- Aggressive tuning:
  - `scripts/research/run_vn30_aggressive_model_tuning.py`
  - `reports/claims/VN30_AGGRESSIVE_MODEL_TUNING_CLAIM_BOUNDARY.md`
  - `reports/protocols/VN30_AGGRESSIVE_MODEL_TUNING_PROTOCOL.md`
  - `reports/results/VN30_AGGRESSIVE_MODEL_TUNING_RESULT_SUMMARY.md`
  - `reports/generated/vn30_aggressive_model_tuning/`
- Champion rescue:
  - `scripts/research/run_vn30_champion_rescue_tuning.py`
  - `reports/claims/VN30_CHAMPION_RESCUE_TUNING_CLAIM_BOUNDARY.md`
  - `reports/results/VN30_CHAMPION_RESCUE_TUNING_RESULT_SUMMARY.md`
  - `reports/generated/vn30_champion_rescue_tuning/`
- Full resurrection/index-pretrain:
  - `scripts/research/run_vn30_full_model_resurrection_index_pretrain.py`
  - `reports/claims/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_CLAIM_BOUNDARY.md`
  - `reports/protocols/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_PROTOCOL.md`
  - `reports/results/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_RESULT_SUMMARY.md`
  - `reports/generated/vn30_full_model_resurrection/`
- Selected candidate strict replay and robustness:
  - `scripts/research/run_vn30_strict_replay_robustness.py`
  - `reports/claims/VN30_SELECTED_CANDIDATE_STRICT_REPLAY_CLAIM_BOUNDARY.md`
  - `reports/claims/VN30_SELECTED_CANDIDATE_STRICT_REPLAY_ROBUSTNESS_CLAIM_BOUNDARY.md`
  - `reports/results/VN30_SELECTED_CANDIDATE_STRICT_REPLAY_RESULT.md`
  - `reports/results/VN30_SELECTED_CANDIDATE_STRICT_REPLAY_ROBUSTNESS_RESULT.md`
  - `reports/generated/vn30_selected_candidate_strict_replay/`
  - `reports/generated/vn30_strict_replay_robustness/`
- V8 target redesign:
  - `scripts/research/run_vn30_v8_strategy_target_redesign.py`
  - `reports/claims/VN30_V8_STRATEGY_TARGET_REDESIGN_CLAIM_BOUNDARY.md`
  - `reports/protocols/VN30_V8_STRATEGY_TARGET_REDESIGN_PROTOCOL.md`
  - `reports/results/VN30_V8_STRATEGY_TARGET_REDESIGN_RESULT_SUMMARY.md`
  - `reports/generated/vn30_v8_strategy_target_redesign/`

Files/groups intentionally left uncommitted:

- `reports/generated/vn30_legacy_rules_reference_and_stacking/robustness_selection/`
- Stashed-only VN30 side work not selected for this backfill, including `scripts/research/run_vn30_legacy_robustness_selection.py`, `scripts/research/run_vn30_missing_full_horizon_completion.py`, and `reports/generated/vn30_model_universe_benchmark/full_horizon_completion/`.
- The six modified tracked files listed in "Needs Human Review".

Blocked from commit:

- `paper_evidence_clean_final_export/`
- `paper_evidence_export/`
- `paper_evidence_raw_full_export/`
- `vn30_clean_final_paper_evidence_export.zip`
- `vn30_paper_evidence_export.zip`
- `vn30_raw_full_paper_evidence_export.zip`
- `reports/generated/paper_figure_data.zip`
- `reports/generated/paper_figures.zip`
- `scripts/research/export_vn30_clean_paper_tables_after_completion.py`
- `scripts/research/export_vn30_paper_evidence_package.py`
- `scripts/research/export_vn30_raw_full_paper_evidence.py`
- Any `.env`, credential, token, virtual environment, local cache, temp, log, DOCX, PPTX, `__pycache__/`, `.pytest_cache/`, or data provider raw cache file.
