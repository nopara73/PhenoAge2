# Evaluation Protocol

## Frozen Dataset Version

The benchmark dataset for this branch is the current `nhanes3-bioage` package and no other dataset variant may be used for headline BioAge subset-search comparisons.

- Cohort file: `nhanes3-bioage/cohort.csv`
- Outcomes file: `nhanes3-bioage/outcomes.csv`
- Cohort rows: `10414`
- Outcomes rows: `10414`
- Aging-related deaths: `2930`
- Cohort SHA-256: `3c256c21956741e0be89e0a3f19faa0dce738802df9f42f935816c844ff9b3a6`
- Outcomes SHA-256: `3f03a013417007a322e47594e781d5cf3ea11203819e30ad8ad90beb77dc3ab1`

This frozen dataset is defined by the inclusion rule already documented in `nhanes3-bioage/README.md` and `nhanes3-bioage/manifest.json`:

- `eligstat == 1`
- `HSAGEIR >= 18`
- `PHPFAST >= 8`
- fasting cohort retained even when candidate biomarkers are missing

Once model search begins, no cohort filtering, endpoint relabeling, or file replacement is allowed for the main benchmark.

## Frozen Prediction Task

The main benchmark task is to predict time to aging-related death from baseline measurements on the frozen `nhanes3-bioage` fasting cohort.

- Candidate inputs are drawn from chronological age (`HSAGEIR`) plus the 57 biomarker candidates in `nhanes3-bioage/cohort.csv`
- Follow-up time field: `time_months`
- Event field: `aging_related_event`
- Participant set: identical for all candidate biomarker subsets
- Non-aging deaths are not counted as events, consistent with the branch's aging-related mortality focus

The intended search dimension is the input subset. The base machine-learning scorer remains fixed up to width changes required by the selected biomarker subset.

## Frozen Final Test Set

A participant-level final test set is frozen before subset search begins.

- Split file: `nhanes3-bioage/frozen_split.csv`
- Split manifest: `nhanes3-bioage/frozen_split_manifest.json`
- Test fraction: `0.20`
- Random seed: `20260321`
- Split unit: participant (`SEQN`)
- Stratification field: `aging_related_event`

The `test` participants in `nhanes3-bioage/frozen_split.csv` are the untouched final benchmark set. They may be used only for the one-time final evaluation of the selected biological-age candidate.

All subset search, candidate comparison, preprocessing fitting, and hyperparameter tuning must be performed only on participants marked `development`. If the frozen test set is used to guide further model changes, it is no longer a valid untouched test set for the main benchmark.

## Frozen Primary Metric

The primary evaluation metric for the branch is the concordance index (`C-index`) on the untouched final test set.

- Primary metric: `C-index`
- Evaluation set: participants marked `test` in `nhanes3-bioage/frozen_split.csv`
- Comparison rule: all candidate subsets are searched on the same development/test partition

This metric is used only for model comparison, not to redefine the underlying prediction task. The prediction task remains time to aging-related death on the frozen fasting cohort.

## Frozen Comparison Rule

Candidate biomarker subsets are compared by held-out `C-index` using the same base scorer family and the same frozen participant split.

- Primary objective: maximize held-out `C-index`
- Candidate comparison surface: development validation `C-index`
- Final report surface: held-out `test` `C-index`
- Search dimension: biomarker subset choice, not benchmark redefinition

There is no per-subset complete-case benchmark. Candidate subsets may not switch to their own participant sets in order to claim improvement.

## Frozen Definition Of A Valid Biological-Age Candidate

For the main benchmark, a candidate model counts as valid only if all of the following conditions are satisfied.

- Allowed raw inputs are limited to `HSAGEIR` plus the 57 biomarker candidates already present in `nhanes3-bioage/cohort.csv`
- No extra covariates are allowed beyond age and those biomarker columns
- No external datasets, pretrained models, or external labels may be used for training, tuning, or feature construction
- The training participant set must come only from rows marked `development` in `nhanes3-bioage/frozen_split.csv`
- Any preprocessing, feature transformation, scaling, imputation, normalization, selection, or derived-feature construction must be fit only on the training portion of each development-fold split
- Missing biomarker values must be handled by one consistent benchmark-wide strategy on the frozen fasting cohort, rather than by switching to per-subset complete-case cohorts

The following are not allowed for the main benchmark:

- use of mortality follow-up fields as model inputs
- use of `mortstat`, `time_months`, `permth_exm`, `ucod_leading`, or `aging_related_event` as model inputs
- leakage from held-out `test` participants into training, preprocessing, subset selection, or feature engineering
- replacing the fasting benchmark with `cohort_all_fields_required.csv` / `outcomes_all_fields_required.csv`
- comparing subsets on different participant sets

In short, a valid candidate must learn only from the frozen fasting cohort, the documented age-plus-biomarker pool, and the branch's fixed survival objective.

## Frozen Reporting Template

Any headline benchmark report for this branch must report the following items together.

- benchmark dataset identity: `nhanes3-bioage`
- cohort size and aging-related death count
- evaluation split identity: held-out `test` participants from `nhanes3-bioage/frozen_split.csv`
- primary metric: held-out `C-index`
- selected biomarker subset
- input feature count
- whether chronological age was included in the selected inputs
- the fixed missing-data strategy used during search and final evaluation

If reported, secondary analyses, exploratory findings, subgroup results, or alternative metrics must be clearly labeled as secondary and must not replace the frozen primary comparison.

The headline claim for the branch must be based on this frozen reporting template rather than on selective reporting of auxiliary results.
