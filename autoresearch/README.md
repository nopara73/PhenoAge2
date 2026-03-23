# autoresearch

This folder adapts the Autoresearch workflow to the PhenoAge 2.0 benchmark.

Instead of language-model pretraining, the fixed harness now operates on the frozen `nhanes3-phenoage` dataset and the frozen rules in `../evaluation-protocol.md`.

## How it works

The folder keeps the original Autoresearch shape:

- `prepare.py` is the fixed harness and should not be modified during search.
- `train.py` is the single agent-editable training file.
- `program.md` contains the human-authored research instructions.

Candidate models are trained on `development` participants only and scored with development validation `C-index`. Final comparison against original PhenoAge happens separately on the held-out `test` participants.

## Fixed benchmark contract

- Inputs are limited to chronological age plus the 9 original PhenoAge biomarkers.
- `HSAGEIR` is part of the allowed PA2 input set.
- The held-out `test` split is frozen and must not be used during model search.
- The headline benchmark metric is held-out `C-index`.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Sanity-check the fixed harness
uv run prepare.py --show-counts

# 3. Run one candidate training experiment
uv run train.py

# 4. Evaluate the saved candidate on the held-out test set
uv run ../evaluate_pa2.py
```

## Project structure

```text
prepare.py      fixed NHANES harness and evaluation helpers
train.py        agent-edited candidate trainer
program.md      autonomous research instructions
pyproject.toml  dependencies
```
