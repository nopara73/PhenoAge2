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
- Inputs for `PhenoAge 2.0`: the same 9 PhenoAge biomarkers, but not chronological age
- Follow-up time field: `time_months`
- Event field: `aging_related_event`
- Participant set: identical for both models
- Non-aging deaths are not counted as events, consistent with the original PhenoAge focus on aging-related mortality

In other words, both models must be evaluated on the same participants, using the same follow-up information and the same aging-related mortality time-to-event outcome. The only intended input difference is that original `PhenoAge` is allowed to use age, while `PhenoAge 2.0` is not.

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
