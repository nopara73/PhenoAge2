# autoresearch

This folder adapts the Autoresearch workflow to a BioAge-style biomarker-subset benchmark.

Instead of language-model pretraining, the fixed harness now operates on the frozen `nhanes3-bioage` dataset and the frozen rules in `../evaluation-protocol.md`.

## How it works

The folder keeps the original Autoresearch shape:

- `prepare.py` is the fixed harness and should not be modified during search.
- `train.py` is the agent-edited training file.
- `program.md` contains the human-authored research instructions.

Candidate models are trained on `development` participants only and scored with development validation `C-index`. Final evaluation happens separately on the held-out `test` participants.

## Fixed benchmark contract

- Inputs are limited to `HSAGEIR` plus the 57 candidate biomarkers already present in `nhanes3-bioage/cohort.csv`.
- Candidate subsets are compared on the same frozen fasting cohort split.
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
train.py        candidate subset trainer
program.md      autonomous research instructions
pyproject.toml  dependencies
```
