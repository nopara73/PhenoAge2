# autoresearch

This folder adapts the Autoresearch workflow to a BioAge biomarker-subset campaign on the frozen `nhanes3-bioage` benchmark.

Instead of one manually edited subset per run, the default `train.py` entrypoint now runs a resumable lane-aware campaign that searches biomarker subsets on the fixed development split while protecting the held-out `test` set.

## How it works

- `prepare.py` is the fixed harness and should not be modified during search.
- `train.py` runs the campaign controller, staged trainer, frontier search, and final validation lock-in.
- `program.md` documents the benchmark-safe campaign policy.

Candidate subsets are trained only on `development` participants and compared by development validation `C-index`. Held-out `test` is used only after the validation winner is locked.

The scorer family remains intentionally naive: standardize the selected raw inputs with train-fit mean/std, then feed them into one small MLP that emits a single risk score.

## Budget policy

The hard global ceiling remains `TIME_BUDGET = 10` in `prepare.py`.

`train.py` enforces lane-aware staged budgets beneath that ceiling:

- `with_age`: Tier A `1.0s`, Tier B `2.0s`, Tier C `10.0s`
- `without_age`: Tier A `0.75s`, Tier B `3.5s`, Tier C `10.0s`

Full `10s` runs are reserved for serious finalists, not routine screening.

## Fixed benchmark contract

- Inputs are limited to `HSAGEIR` plus the 57 candidate biomarkers already present in `nhanes3-bioage/cohort.csv`.
- Candidate subsets are compared on the same frozen fasting cohort split.
- Missing biomarkers must use one consistent benchmark-wide strategy.
- The held-out `test` split is frozen and must not be used during search.
- The headline benchmark metric is held-out `C-index`.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Sanity-check the fixed harness
uv run prepare.py --show-counts

# 3. Smoke-test the campaign implementation
uv run train.py --smoke-test

# 4. Run or resume the full campaign
uv run train.py

# 5. Re-evaluate the saved final candidate on the held-out test set
uv run ../evaluate_pa2.py
```

## Generated artifacts

- `results.tsv` — append-only experiment ledger
- `bioage_campaign_state.json` — resumable campaign state
- `bioage_campaign_status.json` — latest frontier snapshot
- `bioage_campaign_summary.json` — finalists and winner summary
- `campaign_artifacts/` — Tier C finalist artifacts
- `../bioage_test_result.json` — one-time held-out test result for the locked winner

## Project structure

```text
prepare.py      fixed NHANES harness and evaluation helpers
train.py        long-run search campaign controller
program.md      benchmark-safe research instructions
pyproject.toml  dependencies
```
