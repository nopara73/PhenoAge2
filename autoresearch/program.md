# BioAge Autoresearch

This folder now encodes a benchmark-safe long-run BioAge subset-search campaign.

## In-Scope Files

Read these files before running the campaign:

- `prepare.py` — frozen benchmark harness. Do not modify during search.
- `train.py` — the campaign controller and trainer.
- `../evaluation-protocol.md` — frozen benchmark rules and reporting contract.
- `../nhanes3-bioage/README.md` — frozen cohort definition and feature pool.

## Frozen Benchmark Rules

The benchmark is frozen and must remain unchanged throughout search.

- Inputs are limited to `HSAGEIR` plus the 57 biomarker candidates already present in `nhanes3-bioage/cohort.csv`.
- Candidate subsets must be compared on the same frozen fasting cohort split.
- No extra covariates, external data, pretrained models, or external labels.
- No leakage from the held-out `test` set.
- All preprocessing must be fit only on the training portion of the fixed development split.
- Missing biomarker values must use one benchmark-wide strategy on the frozen fasting cohort.
- The headline comparison metric is held-out `C-index`.

## Search Goal

The campaign should maximize development validation `C-index` while preserving the frozen benchmark contract.

This is a survival-ranking problem, not a classification problem. The primary search dimension is the biomarker subset, not benchmark redefinition.

The scorer family stays fixed to the naive MLP baseline:

```text
risk(x) = f((x - mu) / sigma)
```

where `x` is the selected raw input vector, `mu` and `sigma` are fit on the training split only, and `f` is a small MLP.

## Lane-Aware Budget Policy

`prepare.py` keeps the single hard global ceiling:

```text
TIME_BUDGET = 10
```

`train.py` must enforce shorter internal stage budgets beneath that ceiling.

- `with_age` Tier A: `1.0s`
- `with_age` Tier B: target `2.0s`, within the measured `1.5s` to `3.0s` confirmation regime
- `with_age` Tier C: `10.0s`
- `without_age` Tier A: target `0.75s`, within the measured `0.5s` to `1.0s` quick-screen regime
- `without_age` Tier B: `3.5s`
- `without_age` Tier C: `10.0s`

Full `10s` runs are reserved for serious finalists, not routine screening.

## Search Policy

`train.py` should run two parallel lanes from the start:

- `with_age`: `HSAGEIR + biomarkers`
- `without_age`: biomarkers only

The search should:

- seed each lane from the PhenoAge-9 subset, single-marker screens, biological-system groups, and random sparse subsets
- use Tier A for cheap screening
- require repeat Tier A checks for borderline gains
- use Tier B as the first frontier-confirmation budget
- use beam/frontier search instead of pure greedy forward selection
- reserve Tier C for a small diverse finalist set per lane

## Promotion Rules

Promotion and pruning must remain validation-only.

- No held-out `test` information may guide search.
- A single Tier A result is not enough to declare a lane leader.
- Tier A to Tier B requires improvement over the parent or lane reference at the same tier.
- Borderline Tier A gains must be repeated at the same tier before promotion.
- Tier B should be the first tier allowed to confirm frontier membership.
- Tier C should be used only for serious finalists and final ranking, not for routine rescue attempts.

## Logging Contract

Append one row per evaluation to `results.tsv` using the existing schema:

```text
commit	val_cindex	memory_gb	status	feature_count	selected_biomarkers	description
```

- `commit`: short git hash
- `val_cindex`: development validation C-index, use `0.000000` for crashes
- `memory_gb`: peak memory in GB, use `0.0` for crashes
- `status`: `keep`, `discard`, or `crash`
- `feature_count`: total selected input count including age when present
- `selected_biomarkers`: semicolon-separated biomarker codes
- `description`: parseable metadata payload

The `description` payload must record at least:

- `candidate_id`
- `lane`
- `tier`
- `requested_budget_s`
- `actual_training_s`
- `promotion`
- `parent_id`
- `operator`
- `seed_family`

## Status And State Artifacts

`train.py` should keep campaign state on disk so long runs remain resumable and auditable.

- `bioage_campaign_state.json` — resumable campaign state
- `bioage_campaign_status.json` — current frontier and best-per-lane snapshot
- `bioage_campaign_summary.json` — finalists and winner summary
- `campaign_artifacts/` — Tier C finalist artifacts

## Finalization Rules

After validation-only search completes:

1. Lock one winner per lane using validation evidence only.
2. Choose the overall primary winner from those locked lane winners.
3. Copy the winning Tier C artifact to the default candidate artifact path.
4. Evaluate the locked winner on held-out `test` exactly once.
5. Write the final benchmark report to `../bioage_test_result.json`.

If `test` is touched before the winner is locked, the primary benchmark is contaminated.

## How To Run

```bash
# Run or resume the full campaign
uv run train.py

# Start a fresh campaign state
uv run train.py --fresh

# Smoke-test the implementation without touching results.tsv or held-out test
uv run train.py --smoke-test
```

## Priority Order

1. Respect the frozen benchmark.
2. Preserve held-out test integrity.
3. Improve validation `C-index` through subset search first.
4. Prefer simpler and more stable subsets when performance is similar.
5. Keep the budget policy enforceable in code and logs.
