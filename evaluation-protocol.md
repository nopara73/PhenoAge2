# Evaluation Protocol

## Frozen Dataset Version

The benchmark dataset for this project is the current `nhanes3-phenoage` package and no other dataset variant may be used for headline PhenoAge vs PhenoAge 2.0 comparisons.

- Cohort file: `nhanes3-phenoage/cohort.csv`
- Outcomes file: `nhanes3-phenoage/outcomes.csv`
- Cohort rows: `9358`
- Outcomes rows: `9358`
- Aging-related deaths: `2710`
- Cohort SHA-256: `a8dbf1cd650346c2e4c46db8aa4fa2ad57837e7bbf0901c2475f06f189118312`
- Outcomes SHA-256: `64e8328a6b3175c590c40927b356bacc7faf040aa5d07d186bdabf5be1a15809`

This frozen dataset is defined by the inclusion rule already documented in `nhanes3-phenoage/README.md` and `nhanes3-phenoage/manifest.json`:

- `eligstat == 1`
- `HSAGEIR >= 20`
- `PHPFAST >= 8`
- complete-case filtering on `HSAGEIR` plus the 9 PhenoAge biomarkers

Once model search begins, no cohort filtering, endpoint relabeling, or file replacement is allowed for the main benchmark.

## Frozen Prediction Task

The main benchmark task is to predict time to aging-related death from baseline measurements on the frozen `nhanes3-phenoage` cohort.

- Inputs for original `PhenoAge`: chronological age plus the 9 PhenoAge biomarkers
- Inputs for `PhenoAge 2.0`: chronological age plus the same 9 PhenoAge biomarkers
- Follow-up time field: `time_months`
- Event field: `aging_related_event`
- Participant set: identical for both models
- Non-aging deaths are not counted as events, consistent with the original PhenoAge focus on aging-related mortality

In other words, both models must be evaluated on the same participants, using the same follow-up information and the same aging-related mortality time-to-event outcome. The intended difference is the modeling approach used to turn the same age-plus-biomarker inputs into a survival risk score.

## Frozen Final Test Set

A participant-level final test set is frozen before model search begins.

- Split file: `nhanes3-phenoage/frozen_split.csv`
- Split manifest: `nhanes3-phenoage/frozen_split_manifest.json`
- Test fraction: `0.20`
- Random seed: `20260321`
- Split unit: participant (`SEQN`)
- Stratification field: `aging_related_event`

The `test` participants in `nhanes3-phenoage/frozen_split.csv` are the untouched final benchmark set. They may be used only for the one-time final comparison between original `PhenoAge` and the final selected `PhenoAge 2.0` model.

All model search, candidate comparison, feature engineering, preprocessing fitting, and hyperparameter tuning must be performed only on participants marked `development`. If the frozen test set is used to guide further model changes, it is no longer a valid untouched test set for the main benchmark.

## Frozen Primary Metric

The primary evaluation metric for headline original `PhenoAge` vs `PhenoAge 2.0` comparisons is the concordance index (`C-index`) on the untouched final test set.

- Primary metric: `C-index`
- Evaluation set: participants marked `test` in `nhanes3-phenoage/frozen_split.csv`
- Comparison rule: both models are scored on the same held-out participants

This metric is used only for model comparison, not to redefine the underlying prediction task. The prediction task remains time to aging-related death on the frozen cohort.

The original `PhenoAge` formulation remains Gompertz-based, but the benchmark comparison between original `PhenoAge` and `PhenoAge 2.0` is frozen to `C-index` so both methods can be judged with the same survival-aware discrimination metric.

## Frozen Comparison Rule

The comparison between original `PhenoAge` and `PhenoAge 2.0` is paired on the same untouched `test` participants from `nhanes3-phenoage/frozen_split.csv`.

- Primary win condition: superiority
- Primary comparison metric: held-out `C-index`
- Difference definition: `Delta = C-index(PhenoAge 2.0) - C-index(PhenoAge)`
- Meaningful superiority threshold: `Delta >= +0.01`

`PhenoAge 2.0` is considered superior if its held-out test-set `C-index` exceeds the held-out test-set `C-index` of original `PhenoAge` by at least `0.01`.

If superiority is not achieved, the headline benchmark result is not a success.

This rule is frozen before the final `PhenoAge 2.0` result is evaluated so that the benchmark cannot be redefined after seeing the outcome.

## Frozen Definition Of A Valid PhenoAge 2.0 Model

For the main benchmark, a candidate model counts as a valid `PhenoAge 2.0` model only if all of the following conditions are satisfied.

- Allowed input features are limited to chronological age (`HSAGEIR`) plus the 9 original PhenoAge biomarkers: `AMP`, `CEP`, `SGP`, `CRP`, `LMPPCNT`, `MVPSI`, `RWP`, `APPSI`, and `WCP`
- No extra covariates are allowed beyond chronological age and the 9 biomarkers
- No external datasets, pretrained models, or external labels may be used for training, tuning, or feature construction
- The training participant set must come only from rows marked `development` in `nhanes3-phenoage/frozen_split.csv`
- Any preprocessing, feature transformation, scaling, imputation, normalization, selection, or derived-feature construction must be fit only on the training portion of each development-fold split

The following are not allowed for the main benchmark:

- use of covariates beyond `HSAGEIR` and the 9 PhenoAge biomarkers
- use of mortality follow-up fields as model inputs
- use of `mortstat`, `time_months`, `permth_exm`, `ucod_leading`, or `aging_related_event` as model inputs
- leakage from held-out `test` participants into training, preprocessing, model selection, or feature engineering

In short, a valid `PhenoAge 2.0` model must learn only from chronological age plus the same 9 biomarkers used by original `PhenoAge`, without any extra auxiliary information.

## Frozen Reporting Template

Any headline benchmark comparison between original `PhenoAge` and `PhenoAge 2.0` must report the following items together.

- benchmark dataset identity: `nhanes3-phenoage`
- cohort size and aging-related death count
- evaluation split identity: held-out `test` participants from `nhanes3-phenoage/frozen_split.csv`
- primary metric: held-out `C-index`
- original `PhenoAge` held-out `C-index`
- `PhenoAge 2.0` held-out `C-index`
- primary difference definition: `Delta = C-index(PhenoAge 2.0) - C-index(PhenoAge)`
- superiority threshold: `Delta >= +0.01`
- final interpretation category: superior or inferior
- whether chronological age was included in `PhenoAge 2.0` inputs

If reported, secondary analyses, exploratory findings, subgroup results, or alternative metrics must be clearly labeled as secondary and must not replace the frozen primary comparison.

The headline claim for the project must be based on this frozen reporting template rather than on selective reporting of auxiliary results.
