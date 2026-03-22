# PA2 Autoresearch Restart

This folder is in a no-ML restart phase.

The intended mode is strict and simple: edit `train.py`, run one experiment, measure the
result, keep it only if it helps, otherwise restore the kept baseline and move on.

## In-Scope Files

Read these for context:

- `prepare.py` — fixed benchmark harness and evaluation. Do not modify.
- `train.py` — the only executable experiment file.
- `last_kept_train.py` — snapshot of the current kept no-ML baseline.
- `results.tsv` — minimal experiment ledger.
- `research_journal.md` — historical notes plus restart-era notes.
- `../evaluation-protocol.md` — frozen benchmark rules.

Optional helpers:

- `manage_kept.py` — save/restore helper for the current kept no-ML baseline.
- `summarize_results.py` — convenience summary.
- `log_result.py` — convenience logger for `results.tsv`.

## Fixed Benchmark Rules

The benchmark is frozen. Do not redefine it.

- Inputs are limited to the 9 original PhenoAge biomarkers.
- `HSAGEIR` is not allowed as an input.
- No extra covariates, external data, or external labels.
- No leakage from the held-out `test` set.
- All preprocessing must be fit only on the training portion of the development split.
- The search metric is development `val_cindex`.
- Use the existing interpreter at `C:\Users\user\Desktop\PhenoAge2\autoresearch\.venv\Scripts\python.exe`.
- Do not install dependencies or create a new environment.

## Restart Goal

Get the highest `val_cindex` on the fixed development validation split using only elementary,
hand-built biomarker scoring rules.

This restart is intentionally trying to answer a narrower question than the old search:
can simple arithmetic beat the current no-ML baseline without drifting back into anchors,
optimizers, or architecture churn?

## Allowed / Not Allowed

You may:

- modify only `train.py`
- change deterministic preprocessing, transforms, clipping, score directions, and score weights
- add or remove simple hand-built arithmetic terms
- simplify the scoring rule if performance is preserved or improved

You may not:

- modify `prepare.py`
- modify `../evaluation-protocol.md`
- use `HSAGEIR` directly or indirectly
- use `pheno_no_age_xb`, `compute_phenoage`, or original PhenoAge coefficients/constants
- use machine learning, gradient descent, learned weights, neural nets, residual towers, or ensembling
- use target-like columns such as `mortstat`, `time_months`, `permth_exm`, `ucod_leading`, or `aging_related_event` as inputs
- use the held-out `test` participants during search
- run parallel worker campaigns during this restart phase

## Output Contract

`train.py` must finish by printing a summary like:

```text
---
val_cindex:       0.742100
training_seconds: 0.0
total_seconds:    0.3
peak_vram_mb:     0.0
num_steps:        0
num_params:       0
best_step:        0
artifact_path:    ...candidate_pa2.pt
```

It must also save a scripted artifact at the printed `artifact_path`. The saved model must
accept a raw biomarker tensor of shape `[N, 9]` and return one risk score per participant.

## Results Ledger

Log each completed run to `results.tsv` using:

```text
commit	val_cindex	memory_gb	status	description
```

- `commit`: short git hash
- `val_cindex`: development validation C-index, or `0.000000` for crashes
- `memory_gb`: peak memory in GB, or `0.0` for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short description of the experiment

Prefix every restart-era description with `[restart-no-ml]` so the new baseline era is
obvious in the ledger.

Higher `val_cindex` is better.

## Baseline Comparison Rule

During this restart, compare new experiments against the current kept elementary baseline,
not against original age-including PhenoAge and not against old anchor-era ML runs.

If an authoritative no-age comparator is later documented, cite it explicitly in experiment
notes, but do not let that reintroduce the old anchor-plus-correction workflow.

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

If a previous run finished but was not logged yet, reconcile it before starting the next one.

If a run crashes because of a small bug, fix it and rerun the same idea once. If the idea
itself seems broken, log the crash and move on.

Restore always means restore to `last_kept_train.py`, which should remain the current kept
no-ML baseline.

## Experiment Loop

1. Review the current git state.
2. Optionally summarize recent results.
3. Start from the current kept no-ML baseline, not an accidental leftover from a discarded run.
4. Modify only `train.py`.
5. Run `C:\Users\user\Desktop\PhenoAge2\autoresearch\.venv\Scripts\python.exe "C:\Users\user\Desktop\PhenoAge2\autoresearch\train.py" > "C:\Users\user\Desktop\PhenoAge2\autoresearch\run.log" 2>&1`.
6. Read `run.log`.
7. Log the result in `results.tsv` with a description prefixed by `[restart-no-ml]`.
8. If the run is better, keep it.
9. If the run is equal or worse, restore the kept no-ML baseline.

## Priority Order

1. Respect the frozen benchmark.
2. Improve `val_cindex`.
3. Prefer simpler formulas when performance is close.
4. Avoid brittle hacks, local hyperparameter churn, and any return to anchor-based ML.
