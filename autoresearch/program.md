# PA2 Autoresearch

This folder adapts Karpathy's Autoresearch idea to the PhenoAge 2.0 benchmark.

## In-Scope Files

Read these files before starting:

- `prepare.py` — fixed harness. Loads the frozen NHANES benchmark, provides the development split, computes C-index, and defines the final comparison contract. Do not modify.
- `train.py` — the only file you edit. This is where candidate PA2 models are defined and trained.
- `../evaluation-protocol.md` — the frozen benchmark rules.

## Fixed Benchmark Rules

The benchmark is frozen. Do not redefine it.

- Inputs for PA2 are limited to chronological age plus the 9 original PhenoAge biomarkers.
- `HSAGEIR` is allowed as an input.
- No extra covariates, external data, or external labels.
- No leakage from the held-out `test` set.
- All preprocessing must be fit only on the training portion of each development-fold split.
- The headline comparison metric is held-out `C-index`.
- The only headline success category is `superior` as defined in `../evaluation-protocol.md`.

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
- Use covariates outside chronological age plus the 9 PhenoAge biomarkers
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

It must also save a scripted candidate model artifact at the path printed in `artifact_path`. The saved model must accept a raw feature tensor containing chronological age plus the 9 PhenoAge biomarkers and return one risk score per participant.

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

## Experiment Loop

1. Review the current git state.
2. Modify only `train.py`.
3. Commit the experiment.
4. Run `uv run train.py > run.log 2>&1`.
5. Read out the results from `run.log`.
6. If the run crashes, inspect the traceback, decide whether to fix or discard, and record the crash.
7. Record the result in `results.tsv` without committing the TSV file.
8. Keep the commit only if `val_cindex` improves meaningfully or achieves the same result with less complexity.
9. Otherwise revert to the previous good state and continue.

## Priority Order

1. Respect the frozen benchmark.
2. Improve `val_cindex`.
3. Prefer simpler models when performance is similar.
4. Avoid brittle hacks.
