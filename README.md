# VN Market Directional Benchmark Lab

A research repository for Vietnamese market directional forecasting benchmarks, provider-standardized OHLCV data access, evidence tracking, and claim governance.

## What This Repo Is

- A benchmark lab for stock and index directional forecasting experiments.
- A provider gateway/API adapter layer for governed VN OHLCV access.
- A data forensics and reproducibility workspace.
- A stock/index directional research workspace with preserved evidence artifacts.

## Active Tracks

- Stock hourly available-window benchmark.
- Stock daily 2015 benchmark.
- Index directional benchmark.
- Data forensics and provider diagnostics.
- Top-k ranking as a separate metric family.

Start with:

- `reports/_index/ACTIVE_EVIDENCE_INDEX.md`
- `reports/_index/ACTIVE_CODE_MAP.md`
- `docs/REPOSITORY_STRUCTURE.md`
- `reports/cleanup/REPO_RENAME_CLEANUP_INVENTORY.md`
- `reports/_index/REPO_CLEANUP_INVENTORY.md`
- `reports/cleanup/CODE_CLEANUP_CHANGES.md`

## Claim Boundary

- Stock hourly available-window has baseline60 evidence, but final65 is not established.
- Stock daily 2015 is 30/30 usable but below 60.
- Index benchmark has exact pass60 results, separate from stock.
- Top-k ranking is not overall directional accuracy.
- No trading, profitability, live-deployment, or investment-recommendation claim is made.

Current selected-candidate and paper-source files use explicit VN30 hourly names, including:

- `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_ROLLING_STABILITY_RESULT.md`
- `reports/claims/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md`
- `reports/claims/VN30_RESEARCH_CLAIM_REGISTER.md`
- `reports/paper/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md`
- `reports/results/VN30_DAILY_2015_RESULT_SUMMARY.md`
- `reports/results/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md`

## Provider/API Adapter

Normal fetch and benchmark code should go through:

- `src/data/providers/vn_price_gateway.py`
- `src/data/providers/vn_provider_contract.py`
- `src/data/adapters/vnstock_adapter.py`
- `scripts/check_provider_usage_policy.py`

Raw `vnstock` or `vnstock_data` imports are allowed only in approved adapter, probe, diagnostic, or test locations.

## Data And Artifact Policy

- `data/`, `outputs/`, `reports/generated/`, and `archive/generated_data_snapshots/` are preserved.
- Data and generated artifacts are tracked with Git LFS where applicable after the repository backup.
- Do not delete data, output artifacts, market cache, raw fetch data, or archive snapshots.
- Do not force-add secrets, raw `.env` files, credentials, tokens, virtual environments, or local caches.

## Validation

Preferred task runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dev_tasks.ps1 -Task validate-all
```

If `make` is available:

```powershell
make validate-all
```

Raw commands:

```powershell
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
<repo-approved-venv>\Scripts\python.exe scripts\check_runtime_preflight.py
<repo-approved-venv>\Scripts\python.exe scripts\check_provider_usage_policy.py
<repo-approved-venv>\Scripts\python.exe -m pytest tests\data\test_provider_usage_policy.py -q
<repo-approved-venv>\Scripts\python.exe -m pytest tests\data\test_vn_price_gateway_contract.py -q
<repo-approved-venv>\Scripts\python.exe -m pytest tests\ml\test_directional_accuracy_metrics.py -q
```

These commands are validation only. They do not run benchmarks, fetch market data, train models, or generate paper/DOCX artifacts.

ork on the current research branch unless explicitly directed otherwise.

See `docs/USAGE.md` and `docs/RESEARCH_WORKFLOW.md` for operating details.
