# PA2 Autoresearch

This folder adapts Karpathy's Autoresearch idea to the PhenoAge 2.0 benchmark.

## In-Scope Files

Read these files before starting:

- `prepare.py` — fixed harness. Loads the frozen NHANES benchmark, provides the development split, computes C-index, and defines the final comparison contract. Do not modify.
- `train.py` — the only file you edit. This is where candidate PA2 models are defined and trained.
- `results.tsv` — append-only experiment ledger. Read it before choosing the next experiment.
- `research_journal.md` — short natural-language memory of hypotheses, outcomes, and next steps.
- `summarize_results.py` — helper that prints the current leaderboard and recent history.
- `manage_kept.py` — helper that saves or restores the last kept `train.py`.
- `log_result.py` — helper that parses `run.log` and appends one row to `results.tsv`.
- `../evaluation-protocol.md` — the frozen benchmark rules.

## Fixed Benchmark Rules

The benchmark is frozen. Do not redefine it.

- Inputs for PA2 are limited to the 9 original PhenoAge biomarkers.
- `HSAGEIR` is not allowed as an input.
- No extra covariates, external data, or external labels.
- No leakage from the held-out `test` set.
- All preprocessing must be fit only on the training portion of each development-fold split.
- The headline comparison metric is held-out `C-index`.
- Final success categories are `superior`, `non-inferior`, or `inferior` as defined in `../evaluation-protocol.md`.

## Experimentation Goal

The goal is simple: get the highest `val_cindex` on the fixed development validation split while preserving the frozen benchmark constraints.

This is a survival-ranking problem, not a classification problem. Optimize the model to rank earlier aging-related deaths ahead of longer survivors.

## What You CAN Do

- Modify `train.py`
- Change model architecture
- Change optimizer and loss
- Change hyperparameters
- Simplify the model if it preserves or improves `val_cindex`

## What You CANNOT Do

- Modify `prepare.py`
- Modify `../evaluation-protocol.md`
- Use `HSAGEIR` directly or indirectly
- Use `mortstat`, `time_months`, `permth_exm`, `ucod_leading`, or `aging_related_event` as model inputs
- Touch the held-out `test` participants during search

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

It must also save a scripted candidate model artifact at the path printed in `artifact_path`. The saved model must accept a raw biomarker tensor of shape `[N, 9]` and return one risk score per participant.

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, not comma-separated).

Use these columns:

```text
commit	val_cindex	memory_gb	status	description
```

- `commit`: short git hash
- `val_cindex`: development validation C-index, use `0.000000` for crashes
- `memory_gb`: peak memory in GB, use `0.0` for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short description of the experiment

Higher `val_cindex` is better.

## Cross-Run Memory Policy

This search loop must build on previous runs instead of trying disconnected ideas.

Before editing `train.py`:

- Read `results.tsv` and `research_journal.md`.
- Run `summarize_results.py` to see the current best and the recent history.
- Start from the last kept version of `train.py`, not from a discarded one.
- Choose one experiment that follows from prior results.

When choosing the next run:

- Change only one conceptual thing at a time.
- Prefer local moves near the current best configuration.
- Do not repeat a discarded idea unless the journal explains why the retry is materially different.
- If two ideas are equally plausible, prefer the simpler one.

## Experiment Loop

1. Review the current git state.
2. Run `python summarize_results.py` and read `results.tsv` plus `research_journal.md`.
3. If needed, restore the last kept baseline with `python manage_kept.py restore`.
4. Modify only `train.py`.
5. Run `.venv\Scripts\python.exe "train.py" > "run.log" 2>&1`.
6. Read out the results from `run.log`.
7. Record the result in `results.tsv` with `python log_result.py --description "..." --status keep|discard|crash`.
8. Append a short entry to `research_journal.md` with the hypothesis, outcome, learning, and next move.
9. Keep the run only if `val_cindex` improves meaningfully or achieves the same result with less complexity.
10. If the run is kept, save it with `python manage_kept.py save`.
11. If the run is discarded or crashes, restore `train.py` from the last kept version with `python manage_kept.py restore`.

## Priority Order

1. Respect the frozen benchmark.
2. Improve `val_cindex`.
3. Prefer simpler models when performance is similar.
4. Avoid brittle hacks.
