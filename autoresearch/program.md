# PA2 Autoresearch

This folder adapts Karpathy-style autoresearch to the PhenoAge 2.0 benchmark.

The intended mode is simple: edit `train.py`, run one experiment, measure the result, keep it if it wins, otherwise revert and try something else.

## In-Scope Files

Read these for context:

- `prepare.py` — fixed benchmark harness and evaluation. Do not modify.
- `train.py` — the only file you edit.
- `results.tsv` — minimal experiment ledger.
- `../evaluation-protocol.md` — frozen benchmark rules.

Optional helpers:

- `research_journal.md` — human-readable notes. Use after runs if helpful, not as a required ideation input.
- `summarize_results.py` — convenience summary.
- `manage_kept.py` — convenience save/restore helper for the current kept `train.py`.
- `log_result.py` — convenience logger for `results.tsv`.

## Fixed Benchmark Rules

The benchmark is frozen. Do not redefine it.

- Inputs are limited to the 9 original PhenoAge biomarkers.
- `HSAGEIR` is not allowed as an input.
- No extra covariates, external data, or external labels.
- No leakage from the held-out `test` set.
- All preprocessing must be fit only on the training portion of the development split.
- The search metric is development `val_cindex`.
- Final success categories are defined in `../evaluation-protocol.md`.
- Use the existing interpreter at `C:\Users\user\Desktop\PhenoAge2\autoresearch\.venv\Scripts\python.exe`.
- Do not install dependencies or create a new environment.

## Goal

Get the highest `val_cindex` on the fixed development validation split while respecting the frozen benchmark.

This is a survival-ranking problem. The model should rank earlier aging-related deaths ahead of longer survivors.

## Allowed / Not Allowed

You may:

- modify `train.py`
- change architecture, optimizer, loss, and hyperparameters
- simplify the model if performance is preserved or improved

You may not:

- modify `prepare.py`
- modify `../evaluation-protocol.md`
- use `HSAGEIR` directly or indirectly
- use target-like columns such as `mortstat`, `time_months`, `permth_exm`, `ucod_leading`, or `aging_related_event` as inputs
- use the held-out `test` participants during search

## Output Contract

`train.py` must finish by printing a summary like:

```text
---
val_cindex:       0.742100
training_seconds: 300.0
total_seconds:    305.4
peak_vram_mb:     1234.5
num_steps:        412
num_params:       4513
best_step:        375
artifact_path:    ...candidate_pa2.pt
```

It must also save a scripted artifact at the printed `artifact_path`. The saved model must accept a raw biomarker tensor of shape `[N, 9]` and return one risk score per participant.

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

Higher `val_cindex` is better.

## Simplicity Criterion

All else equal, simpler is better.

Keep a change if it meaningfully improves `val_cindex`, or if it matches the current result with less complexity. Do not add brittle machinery for tiny gains.

## Experiment Selection

Use `results.tsv` to avoid exact repeats.

Do not require each new run to be a local tweak of the current best. Radical or disconnected ideas are allowed if they are benchmark-legal and not obvious repeats.

Prefer experiments that are:

1. clearly different from recent failures
2. simple to describe and evaluate
3. reversible if they do not help

## Recovery

If a previous run finished but was not logged yet, reconcile it before starting the next one.

If a run crashes because of a small bug, fix it and rerun the same idea once. If the idea itself seems broken, log the crash and move on.

## Experiment Loop

1. Review the current git state.
2. Optionally summarize recent results.
3. Start from the current kept `train.py` baseline, not an accidental leftover from a discarded run.
4. Modify only `train.py`.
5. Run `C:\Users\user\Desktop\PhenoAge2\autoresearch\.venv\Scripts\python.exe "C:\Users\user\Desktop\PhenoAge2\autoresearch\train.py" > "C:\Users\user\Desktop\PhenoAge2\autoresearch\run.log" 2>&1`.
6. Read `run.log`.
7. Log the result in `results.tsv`.
8. If the run is better, keep it.
9. If the run is equal or worse, revert to the kept baseline.

Convenience helpers such as `manage_kept.py`, `summarize_results.py`, and `log_result.py` may be used, but they are optional. The core loop should remain understandable without them.

## Priority Order

1. Respect the frozen benchmark.
2. Improve `val_cindex`.
3. Prefer simpler solutions when performance is close.
4. Avoid brittle hacks and repeated local churn.
