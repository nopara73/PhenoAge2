# autoresearch

This folder adapts the Autoresearch workflow to the PhenoAge 2.0 benchmark.

Instead of language-model pretraining, the fixed harness now operates on the frozen `nhanes3-phenoage` dataset and the frozen rules in `../evaluation-protocol.md`.

## How it works

The folder keeps the original Autoresearch shape:

- `prepare.py` is the fixed harness and should not be modified during search.
- `train.py` is the single agent-editable training file.
- `program.md` contains the human-authored research instructions.

Candidate models are developed and compared using only the full `development` split. The active search metric is `development_cindex`. The held-out `test` split is reserved for the final publish-time comparison only.

## Fixed benchmark contract

- Inputs are limited to the 9 original PhenoAge biomarkers.
- `HSAGEIR` is not allowed for PA2.
- The held-out `test` split is frozen and must not be used during model search.
- There is no internal development train/validation split in the active workflow.
- The active search metric is full-development `development_cindex`.
- The headline publish-time benchmark metric is held-out `C-index`.

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
