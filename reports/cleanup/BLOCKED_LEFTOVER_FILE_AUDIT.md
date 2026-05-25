# Blocked Leftover File Audit

Audit date: 2026-05-25

Starting branch: `research/vn30-quantum-ml-forecasting-v1`

Starting HEAD: `110a0b7b research: start VN30 quantum ML forecasting branch`

## Counts

- `git status --short`: 11 high-level entries
- `git ls-files -o --exclude-standard`: 908 files
- `git diff --name-only`: 0 files
- Current visible tracked modifications: 0
- Current visible `.zip`, `.docx`, `.pptx` files: 0
- Current visible `.env`, credential, secret, or token path matches: 0
- Current visible cache/temp path matches: 0
- Current visible binary-like extension matches: 113
- Current visible files larger than 25 MB: 29

## Pushed-Commit Coverage

- `d73b7002ebd5964b33b26f85492220290b1b9d97` covers VN30 V3-V7 scripts, protocols, reports, claims, and generated artifacts.
- `500272f42c2ddd420dc36bca3f1e2729b51fd9aa` covers the full-resurrection/index-pretrain and V8 files still visible as untracked on the QML checkout.
- `110a0b7b8f3fddd6dc110376a9ae4c805e7e2131` covers only `reports/protocols/VN30_QML_FORECASTING_PROTOCOL.md`.

## A. Should Remain Uncommitted

| Path | File count | Reason | Recommended action |
| --- | ---: | --- | --- |
| `paper_evidence_export/` | 271 | Paper/export package with figures, manifests, raw copied artifacts, and large row-prediction files. This is explicitly blocked from normal research commits. | leave |
| `paper_evidence_raw_full_export/` | 590 | Raw full paper export package with duplicated raw outputs, prediction archives, binary figures, and large CSV/GZ tables. This is explicitly blocked from normal research commits. | leave |
| `reports/claims/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_CLAIM_BOUNDARY.md` | 1 | Legitimate VN30 claim file, but already committed and pushed in `500272f4`; untracked only because the current checkout is QML. | leave |
| `reports/claims/VN30_V8_STRATEGY_TARGET_REDESIGN_CLAIM_BOUNDARY.md` | 1 | Legitimate VN30 claim file, but already committed and pushed in `500272f4`; untracked only because the current checkout is QML. | leave |
| `reports/protocols/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_PROTOCOL.md` | 1 | Legitimate VN30 protocol file, but already committed and pushed in `500272f4`; untracked only because the current checkout is QML. | leave |
| `reports/protocols/VN30_V8_STRATEGY_TARGET_REDESIGN_PROTOCOL.md` | 1 | Legitimate VN30 protocol file, but already committed and pushed in `500272f4`; untracked only because the current checkout is QML. | leave |
| `reports/results/VN30_FULL_MODEL_RESURRECTION_AND_INDEX_PRETRAIN_RESULT_SUMMARY.md` | 1 | Legitimate VN30 result file, but already committed and pushed in `500272f4`; untracked only because the current checkout is QML. | leave |
| `reports/results/VN30_V8_STRATEGY_TARGET_REDESIGN_RESULT_SUMMARY.md` | 1 | Legitimate VN30 result file, but already committed and pushed in `500272f4`; untracked only because the current checkout is QML. | leave |
| `reports/generated/vn30_full_model_resurrection/` | 16 | Legitimate VN30 generated artifact directory, but already committed and pushed in `500272f4`. Two files are larger than 25 MB and should not be recommitted from QML. | leave |
| `reports/generated/vn30_v8_strategy_target_redesign/` | 19 | Legitimate VN30 generated artifact directory, but already committed and pushed in `500272f4`; untracked only because the current checkout is QML. | leave |

Should-remain-uncommitted file count: 902

## B. Candidate For Commit

| Path | File count | Reason | Recommended action |
| --- | ---: | --- | --- |
| `reports/cleanup/BLOCKED_LEFTOVER_FILE_AUDIT.md` | 1 | Cleanup inventory documenting the blocked-leftover audit. This is not one of the blocked artifacts and is safe to preserve on a VN30 review branch. | commit |

Candidate file count: 1 audit file only. No original remaining untracked source/research file requires a new content backfill.

## C. Needs Human Review

| Path | File count | Reason | Recommended action |
| --- | ---: | --- | --- |
| `reports/generated/vn30_legacy_rules_reference_and_stacking/robustness_selection/` | 6 | VN30 generated folder that is not a paper/export package, but it is not covered by `d73b7002`, `500272f4`, or `110a0b7b`; no matching visible source/protocol/result bundle is present in the current worktree. | human_review |

Human-review file count: 6

## Large And Binary-Like File Notes

- Files larger than 25 MB: 29 total.
- Large-file grouping:
  - `paper_evidence_export/`: 13 files.
  - `paper_evidence_raw_full_export/`: 14 files.
  - `reports/generated/vn30_full_model_resurrection/`: 2 files, already covered by pushed commit `500272f4`.
- Binary-like extension matches: 113, all inside paper/export packages or generated figure/archive-style material.
- Current visible zip/DOCX/PPTX count is 0, but paper/export packages remain blocked by policy.

## Decision

No paper evidence package, export package, DOCX/PPTX, zip, cache, credential, `.env`, virtual environment, temp/log, or unclear generated artifact should be staged.

No QML branch commit should include these VN30 leftovers. The only safe commit from this audit is this inventory file on a VN30 review branch.
