# PA2 Ageless Formula Restart

This folder is in an ageless-PhenoAge restart phase.

The intended mode is strict and simple: start from the exact ageless PhenoAge formula,
edit `train.py`, run one experiment, measure the result, keep it only if it helps,
otherwise restore the kept baseline and move on.

## In-Scope Files

Read these for context:

- `prepare.py` — fixed benchmark harness and evaluation. Do not modify.
- `train.py` — the only executable experiment file.
- `last_kept_train.py` — snapshot of the current kept baseline.
- `results.tsv` — active experiment ledger.
- `research_journal.md` — active notes for this clean restart.
- `../evaluation-protocol.md` — frozen benchmark rules.

Optional helpers:

- `manage_kept.py` — save/restore helper for the current kept baseline.
- `summarize_results.py` — convenience summary.
- `log_result.py` — convenience logger for `results.tsv`.

## Fixed Benchmark Rules

The benchmark is frozen. Do not redefine it.

- Inputs are limited to the 9 original PhenoAge biomarkers.
- `HSAGEIR` is not allowed as an input.
- No extra covariates, external data, or external labels.
- No leakage from the held-out `test` set.
- Everything during search must use only participants in the `development` split.
- The search metric is full-development `development_cindex`.
- There is no internal `development` train/validation split in the active workflow.
- `test` exists only for the final publish-time evaluation, not for model search.
- Use the existing interpreter at `C:\Users\user\Desktop\PhenoAge2\autoresearch\.venv\Scripts\python.exe`.
- Do not install dependencies or create a new environment.

## Restart Goal

Get the highest `development_cindex` on the full `development` split starting from the
exact ageless PhenoAge formula and using only deterministic biomarker scoring rules.

This restart is intentionally trying to answer a narrower question than the old search:
can formula-first, age-free scoring beat the ageless PhenoAge baseline without drifting
back into ML architecture churn?

## Allowed / Not Allowed

You may:

- modify only `train.py`
- use the repo-authoritative ageless PhenoAge biomarker formula as the baseline
- change deterministic preprocessing, transforms, clipping, score directions, and score weights
- add or remove simple hand-built arithmetic terms
- simplify the scoring rule if performance is preserved or improved

You may not:

- modify `prepare.py`
- modify `../evaluation-protocol.md`
- use `HSAGEIR` directly or indirectly
- use machine learning, gradient descent, learned weights, neural nets, residual towers, or ensembling
- use target-like columns such as `mortstat`, `time_months`, `permth_exm`, `ucod_leading`, or `aging_related_event` as inputs
- use the held-out `test` participants during search
- run parallel worker campaigns during this restart phase

## Output Contract

`train.py` must finish by printing a summary like:

```text
---
development_cindex:0.742100
training_seconds: 0.0
total_seconds:    0.3
peak_vram_mb:     0.0
num_steps:        0
num_params:       0
artifact_path:    ...candidate_pa2.pt
```

It must also save a scripted artifact at the printed `artifact_path`. The saved model must
accept a raw biomarker tensor of shape `[N, 9]` and return one risk score per participant.

## Results Ledger

Log each completed run to `results.tsv` using:

```text
commit	development_cindex	memory_gb	status	description
```

- `commit`: short git hash
- `development_cindex`: full-development C-index, or `0.000000` for crashes
- `memory_gb`: peak memory in GB, or `0.0` for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short description of the experiment

Prefix every restart-era description with `[restart-ageless]` so the new baseline era is
obvious in the ledger.

Higher `development_cindex` is better.

## Baseline Comparison Rule

During this restart, compare new experiments against the current kept ageless-formula
baseline, not against the age-including original PhenoAge score and not against older ML runs.

The initial kept baseline is the exact ageless PhenoAge formula.

## Experiment Selection

Use `results.tsv` to avoid exact repeats.

Run only one experiment at a time.

Each run must make one conceptual change. Do not spend time on tiny local tuning unless a
formula family first shows a real jump.

Prefer experiments that are:

1. clearly different from recent failures
2. simple to describe and evaluate
3. reversible if they do not help

## Recovery

If this clean restart has no prior run yet, keep the ledger and journal empty until the first new run finishes.

If a run crashes because of a small bug, fix it and rerun the same idea once. If the idea
itself seems broken, log the crash and move on.

Restore always means restore to `last_kept_train.py`, which should remain the current kept
ageless baseline.

## Experiment Loop

1. Review the current git state.
2. Optionally summarize recent results.
3. Start from the current kept ageless baseline, not an accidental leftover from a discarded run.
4. Modify only `train.py`.
5. Run `C:\Users\user\Desktop\PhenoAge2\autoresearch\.venv\Scripts\python.exe "C:\Users\user\Desktop\PhenoAge2\autoresearch\train.py" > "C:\Users\user\Desktop\PhenoAge2\autoresearch\run.log" 2>&1`.
6. Read `run.log`.
7. Log the result in `results.tsv` with a description prefixed by `[restart-ageless]`.
8. If the run is better, keep it.
9. If the run is equal or worse, restore the kept no-ML baseline.

## Priority Order

1. Respect the frozen benchmark.
2. Improve `development_cindex`.
3. Prefer simpler formulas when performance is close.
4. Avoid brittle hacks, local hyperparameter churn, and any return to anchor-based ML.
