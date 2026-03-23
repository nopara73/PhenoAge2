# BioAge Long-Run Planning Prompt

You are entering an existing repo on branch `bioageautoresearch`.

Your task is **not to execute yet**. Your task is to produce a **complete plan** for a long-running, status-reporting autoresearch campaign whose goal is to find the most performant biological age model this branch can discover under the current benchmark contract.

You should assume the user wants a serious, long-horizon search plan from start to finish, including:

- how subset search should work
- what should be explored first
- what should be treated as frozen
- what should be logged and reported during the run
- how to decide which candidates are promoted
- how to handle failures, regressions, and budget waste
- how to preserve fairness and avoid benchmark contamination

Do **not** start executing. Produce the plan only.

## Current branch goal

This branch is no longer a PhenoAge-formula searcher. It is now a **BioAge biomarker-subset searcher**.

The search dimension is primarily the **selected biomarker subset**.

The scorer family is intentionally fixed to the naive formula:

```text
risk(x) = f((x - mu) / sigma)
```

where:

- `x` is the selected raw feature vector
- `mu` and `sigma` are fit on the training split only
- `f` is a small MLP

The current implementation uses one hidden layer and outputs a single risk score.

## Frozen benchmark contract

Read and respect these files first:

- `evaluation-protocol.md`
- `autoresearch/prepare.py`
- `autoresearch/program.md`
- `autoresearch/train.py`
- `nhanes3-bioage/README.md`

Current benchmark facts:

- Dataset: `nhanes3-bioage`
- Cohort: fasting NHANES III BioAge-style cohort
- Total participants: `10414`
- Aging-related deaths: `2930`
- Frozen split seed: `20260321`
- Development participants: `8331`
- Test participants: `2083`
- Development event counts: `2344` aging-related deaths, `5987` non-events
- Test event counts: `586` aging-related deaths, `1497` non-events
- Development train/val split used by the harness: `6665 / 1666`
- Primary metric: survival `C-index`
- Outcome: `time_months` / `aging_related_event`
- Held-out `test` must not guide search

Important fairness rule:

- **Do not compare subsets on different participant sets**
- **Do not use per-subset complete-case cohorts**
- Missing biomarkers must be handled by one consistent benchmark-wide strategy on the frozen fasting cohort

The strict complete-case BioAge files are **not** the main benchmark because they are death-only by construction.

## Allowed feature pool

Inputs are limited to:

- `HSAGEIR`
- the 57 candidate biomarkers already present in `nhanes3-bioage/cohort.csv`

No extra covariates, external data, pretrained models, or external labels.

## Current scorer

The current branch requirement is that the scorer must remain the naive standardized-input MLP family.

Do **not** propose replacing it with a more structured architecture unless you explicitly label that as a later optional extension outside the primary search plan.

The main search should assume:

- subset changes first
- scorer family fixed
- training budget fixed by harness defaults

## Current default budget

The fixed harness training budget has been changed to:

- `TIME_BUDGET = 10`

This was done because the naive MLP reaches near-formula performance very quickly.

## Empirical calibration already established

Using the PhenoAge-9 biomarker subset on the BioAge benchmark:

### Held-out test C-indices

- Naive MLP with age: `0.876725`
- PhenoAge formula with age: `0.874749`
- Naive MLP without age: `0.766521`
- PhenoAge formula without age: `0.735001`

### Validation C-indices

- PhenoAge formula with age: `0.862685`
- PhenoAge formula without age: `0.741810`
- Naive MLP with age baseline: `0.863655`
- Naive MLP without age baseline: `0.767929`

### Diminishing returns observations

For the naive MLP on the PhenoAge-9 subset:

- with age: within `0.01` C-index of the PhenoAge formula after about `1s`
- with age: diminishing returns begin around `1.5s`
- without age: within `0.01` C-index of the age-stripped PhenoAge formula after about `0.5s`
- without age: diminishing returns begin around `3.5s` to `6s`

This strongly suggests multi-fidelity subset search is appropriate.

## Current artifact state

The branch currently has a naive-MLP BioAge candidate artifact and metadata for the reference PhenoAge-9 subset with age.

Relevant files:

- `autoresearch/candidate_bioage.pt`
- `autoresearch/candidate_bioage.metadata.json`
- `bioage_test_result.json`
- `autoresearch/results.tsv`

## What the plan should cover

Your plan should be concrete and operational, not vague. It should cover:

1. A subset-search strategy that is computationally realistic for 57 biomarkers.
2. A staged budget policy suitable for the new `10s` default.
3. How to use quick screens versus deeper evaluations.
4. Whether to search with age included, age excluded, or both in parallel.
5. How to seed the search:
   - PhenoAge-9 reference
   - single-marker screens
   - grouped biological systems
   - random sparse seeds
6. What search operators to use:
   - add
   - drop
   - swap
   - grouped moves
   - random perturbations
7. How to maintain fairness across subsets.
8. How to log experiments so progress remains interpretable.
9. How status reporting should work during a long run.
10. How to decide when diminishing returns justify stopping or changing strategy.
11. How to protect the held-out test set.
12. What the final deliverables of the long run should be.

## Strong preferences for the plan

- Prefer a **coarse-to-fine subset search**, not brute force.
- Prefer **multi-fidelity evaluation**, not full-budget runs on every candidate.
- Prefer **beam/frontier search** over pure greedy forward selection.
- Prefer **biologically diverse finalists**, not just one narrow winner.
- Prefer a plan that future AI is likely to follow because it is encoded in code paths, logs, and benchmark rules rather than just prose.

## Suggested direction

Unless you have a better reason, assume the long-run search should center on:

- fixed naive MLP scorer
- fixed fasting-cohort benchmark
- same frozen split for all subset comparisons
- subset search over 57 biomarkers, with age treated as an explicit toggle

## Important constraint

The user wants a **plan prompt**, not execution. Do not start running experiments yet.

Your output should be a plan for a long-running, status-reporting campaign that is realistic, disciplined, and benchmark-safe.
