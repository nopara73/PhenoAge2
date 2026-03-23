# nhanes3-bioage

This folder is a data-only NHANES III package for fasting BioAge-style downstream work.

## Files

- `cohort.csv`
- `outcomes.csv`
- `cohort_all_fields_required.csv`
- `outcomes_all_fields_required.csv`
- `data_dictionary.md`
- `manifest.json`

`cohort.csv` and `outcomes.csv` are perfectly joinable by `SEQN` and contain the same exact fasting participant set.

`cohort_all_fields_required.csv` and `outcomes_all_fields_required.csv` are perfectly joinable by `SEQN` and contain the same exact strict complete-case participant set.

## Source Files

- `nhanes3/1a/adult.dat`
- `nhanes3/1a/adult.sas`
- `nhanes3/1a/lab.dat`
- `nhanes3/1a/lab.sas`
- `nhanes3/mortality/NHANES_III_MORT_2019_PUBLIC.dat`
- `nhanes3/mortality/SAS_ReadInProgramAllSurveys.sas`

## Inclusion Rules

The package is built from the same NHANES III adult + lab + mortality source system as the existing analytic table.

1. Build the merged NHANES III adult + lab + mortality table.
2. Keep mortality-eligible adults with `eligstat == 1` and `HSAGEIR >= 18`.
3. Keep participants with `PHPFAST >= 8`.
4. Write `cohort.csv` and `outcomes.csv` from that same fasting cohort.
5. Apply one complete-case filter across `HSAGEIR`, the 57 cohort biomarkers, plus `SEQN`, `permth_exm`, `time_months`, `mortstat`, `ucod_leading`, and `aging_related_event`.
6. Write `cohort_all_fields_required.csv` and `outcomes_all_fields_required.csv` from that same strict cohort.

Because `ucod_leading` is only populated for decedents, the strict all-fields-required pair is death-only by construction.

## Artifact Schemas

### `cohort.csv`

- `SEQN`
- `HSAGEIR`
- `ACP`
- `AMP`
- `APPSI`
- `ASPSI`
- `ATPSI`
- `BCP`
- `BUP`
- `BXP`
- `C1P`
- `C3PSI`
- `CAPSI`
- `CEP`
- `CLPSI`
- `CRP`
- `DWP`
- `FEP`
- `FOP`
- `FRP`
- `GHP`
- `GRP`
- `GRPPCNT`
- `HDP`
- `HGP`
- `HTP`
- `I1P`
- `LDPSI`
- `LMP`
- `LMPPCNT`
- `LUP`
- `LYP`
- `MCPSI`
- `MHP`
- `MOP`
- `MOPPCNT`
- `MVPSI`
- `NAPSI`
- `PBP`
- `PLP`
- `PSP`
- `PVPSI`
- `PXP`
- `RCP`
- `RWP`
- `SCP`
- `SEP`
- `SGP`
- `SKPSI`
- `TBP`
- `TCP`
- `TGP`
- `TIP`
- `TPP`
- `UAP`
- `VAP`
- `VCP`
- `VEP`
- `WCP`

### `outcomes.csv`

- `SEQN`
- `permth_exm`
- `time_months`
- `mortstat`
- `ucod_leading`
- `aging_related_event`

### `cohort_all_fields_required.csv`

Same schema as `cohort.csv`.

### `outcomes_all_fields_required.csv`

Same schema as `outcomes.csv`.

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

- Fasting cohort rows in `cohort.csv`: 10414
- Fasting cohort rows in `outcomes.csv`: 10414
- Strict all-fields-required rows in `cohort_all_fields_required.csv`: 3432
- Strict all-fields-required rows in `outcomes_all_fields_required.csv`: 3432
- All-cause deaths in fasting cohort: 4116
- Aging-related deaths in fasting cohort: 2930
- All-cause deaths in strict all-fields-required cohort: 3432
- Aging-related deaths in strict all-fields-required cohort: 2450

## Notes

- This package keeps the fasting threshold you requested and does not add the `HSAGEIR >= 20` PhenoAge-specific restriction.
- `HSAGEIR` is retained in the cohort files so downstream age-aware software can run without additional joins.
- The strict all-fields-required pair is substantially smaller because requiring `ucod_leading` excludes participants who were alive at end of follow-up.
