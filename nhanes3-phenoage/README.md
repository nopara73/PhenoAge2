# nhanes3-phenoage

This folder is a data-only NHANES III package for strict PhenoAge-style downstream work.

## Files

- `README.md`
- `cohort.csv`
- `outcomes.csv`
- `data_dictionary.md`
- `manifest.json`

`cohort.csv` and `outcomes.csv` are perfectly joinable by `SEQN` and contain the same exact participant set.

## Source Files

- `nhanes3/1a/adult.dat`
- `nhanes3/1a/adult.sas`
- `nhanes3/1a/lab.dat`
- `nhanes3/1a/lab.sas`
- `nhanes3/mortality/NHANES_III_MORT_2019_PUBLIC.dat`
- `nhanes3/mortality/SAS_ReadInProgramAllSurveys.sas`

## Inclusion Rules

The canonical analysis cohort is defined by the following ordered filters:

1. Keep mortality-eligible participants with `eligstat == 1`.
2. Keep adults with `HSAGEIR >= 20`.
3. Keep participants with `PHPFAST >= 8`.
4. Apply one complete-case filter for `HSAGEIR`, `AMP`, `CEP`, `SGP`, `CRP`, `LMPPCNT`, `MVPSI`, `RWP`, `APPSI`, and `WCP`.
5. Write both `cohort.csv` and `outcomes.csv` from that same final cohort.

This means the complete-case filter is applied after the adult, eligibility, and fasting filters and before writing both CSVs.

## Artifact Schemas

### `cohort.csv`

- `SEQN`
- `HSAGEIR`
- `AMP`
- `CEP`
- `SGP`
- `CRP`
- `LMPPCNT`
- `MVPSI`
- `RWP`
- `APPSI`
- `WCP`

### `outcomes.csv`

- `SEQN`
- `permth_exm`
- `time_months`
- `mortstat`
- `ucod_leading`
- `aging_related_event`

`time_months` is copied directly from `permth_exm`.

## Endpoint Definition

`aging_related_event` is derived from the raw mortality fields:

- if `mortstat == 1` and `ucod_leading` is in the approved aging-related set, then `aging_related_event = 1`
- if `mortstat == 1` and `ucod_leading` is not in the approved aging-related set, then `aging_related_event = 0`
- if `mortstat == 0`, then `aging_related_event = 0`

Included UCOD buckets:

- `UCOD_001` diseases of heart
- `UCOD_002` malignant neoplasms
- `UCOD_003` chronic lower respiratory diseases
- `UCOD_005` cerebrovascular diseases
- `UCOD_006` Alzheimer's disease
- `UCOD_007` diabetes mellitus
- `UCOD_009` nephritis, nephrotic syndrome and nephrosis

Excluded UCOD buckets:

- `UCOD_004` accidents (unintentional injuries)
- `UCOD_008` influenza and pneumonia
- `UCOD_010` all other causes

`ucod_leading` is only meaningful when `mortstat == 1`.

## Counts

- Final cohort rows in `cohort.csv`: 9358
- Final cohort rows in `outcomes.csv`: 9358
- All-cause deaths in final cohort: 3797
- Aging-related deaths in final cohort: 2710

## Notes

- `HSAGEIR` is retained so original PhenoAge can be reproduced fairly as a baseline.
- `CRP` is stored raw in `cohort.csv`; any log transform is applied downstream when reproducing original PhenoAge.
- `permth_int` may be retained in future versions if needed for auditability, but is not part of the required package contract.
