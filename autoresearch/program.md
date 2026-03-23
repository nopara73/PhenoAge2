# BioAge Autoresearch

This folder adapts Karpathy's Autoresearch idea to a biomarker-subset search benchmark.

## In-Scope Files

Read these files before starting:

- `prepare.py` — fixed harness. Loads the frozen NHANES benchmark, provides the development split, computes C-index, and defines the final comparison contract. Do not modify during search.
- `train.py` — the file you edit during search. This is where the active biomarker subset and training settings live.
- `../evaluation-protocol.md` — the frozen benchmark rules.

## Fixed Benchmark Rules

The benchmark is frozen. Do not redefine it during search.

- Inputs are limited to `HSAGEIR` plus the 57 biomarker candidates already present in `nhanes3-bioage/cohort.csv`.
- Candidate subsets must be compared on the same frozen fasting cohort split.
- No extra covariates, external data, or external labels.
- No leakage from the held-out `test` set.
- All preprocessing must be fit only on the training portion of each development-fold split.
- The headline comparison metric is held-out `C-index`.

## Experimentation Goal

The goal is simple: get the highest `val_cindex` on the fixed development validation split while preserving the frozen benchmark constraints.

This is a survival-ranking problem, not a classification problem. Optimize the model to rank earlier aging-related deaths ahead of longer survivors by changing the selected biomarker subset before changing the scorer family.

The scorer family is the naive MLP baseline:

```text
risk(x) = f((x - mu) / sigma)
```

where `x` is the selected raw input vector, `mu` and `sigma` are fit on the training split only, and `f` is a small MLP.

The fixed harness budget is `10s` of training time per run unless the benchmark contract is deliberately changed.

## What You CAN Do

- Modify `train.py`
- Change the selected biomarker subset
- Change optimizer and loss only if needed after subset exploration stalls
- Change hyperparameters
- Simplify the model if it preserves or improves `val_cindex`

## What You CANNOT Do

- Modify `prepare.py`
- Modify `../evaluation-protocol.md`
- Use covariates outside `HSAGEIR` plus the 57 candidate biomarkers in `nhanes3-bioage/cohort.csv`
- Use `mortstat`, `time_months`, `permth_exm`, `ucod_leading`, or `aging_related_event` as model inputs
- Touch the held-out `test` participants during search
- Compare subsets on different participant sets

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

It must also save a scripted candidate model artifact at the path printed in `artifact_path` plus sidecar metadata describing the selected feature columns and train-fitted imputation values.

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, not comma-separated).

Use these columns:

```text
commit	val_cindex	memory_gb	status	feature_count	selected_biomarkers	description
```

- `commit`: short git hash
- `val_cindex`: development validation C-index, use `0.000000` for crashes
- `memory_gb`: peak memory in GB, use `0.0` for crashes
- `status`: `keep`, `discard`, or `crash`
- `feature_count`: total selected input count including age when present
- `selected_biomarkers`: semicolon-separated biomarker codes
- `description`: short description of the experiment

Higher `val_cindex` is better.

## Experiment Loop

1. Review the current git state.
2. Modify `train.py`, primarily by changing the active biomarker subset.
3. Commit the experiment.
4. Run `uv run train.py > run.log 2>&1`.
5. Read out the results from `run.log`.
6. If the run crashes, inspect the traceback, decide whether to fix or discard, and record the crash.
7. Record the result in `results.tsv` without committing the TSV file.
8. Keep the candidate artifact when `val_cindex` improves by a small exploratory margin or achieves the same result with less complexity.
9. Otherwise revert to the previous good state and continue.

## Priority Order

1. Respect the frozen benchmark.
2. Improve `val_cindex`.
3. Prefer input-subset exploration before changing the scorer family.
4. Prefer simpler models when performance is similar.
5. Avoid brittle hacks.
