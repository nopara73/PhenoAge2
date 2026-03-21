# Data Dictionary

| Variable | Clinical Meaning | Units | Notes |
| --- | --- | --- | --- |
| `SEQN` | NHANES participant identifier | none | Stable join key across `cohort.csv` and `outcomes.csv`. |
| `HSAGEIR` | Age at interview | years | Retained to support fair reproduction of original PhenoAge. |
| `AMP` | Serum albumin | g/dL | Raw source measurement. |
| `CEP` | Serum creatinine | mg/dL | Raw source measurement. |
| `SGP` | Serum glucose | mg/dL | Raw source measurement. |
| `CRP` | Serum C-reactive protein | mg/dL | Stored raw in `cohort.csv`; any log transform is applied downstream when reproducing original PhenoAge. |
| `LMPPCNT` | Lymphocyte percent (Coulter) | percent | Raw source measurement. |
| `MVPSI` | Mean cell volume | fL | SI label in source. |
| `RWP` | Red cell distribution width | percent | Raw source measurement. |
| `APPSI` | Serum alkaline phosphatase | U/L | SI label in source. |
| `WCP` | White blood cell count | not stated in source label | Raw source measurement; source label does not state units explicitly. |
| `permth_exm` | Person-months of follow-up from exam | months | Raw mortality linkage field. |
| `time_months` | Analysis follow-up time | months | Copied directly from `permth_exm`. |
| `mortstat` | Final mortality status | 0/1 | `0` assumed alive, `1` assumed deceased. |
| `ucod_leading` | Underlying leading cause of death recode | UCOD bucket | Meaningful only when `mortstat == 1`. |
| `aging_related_event` | Aging-related mortality endpoint | 0/1 | Derived from `mortstat` and `ucod_leading` using the documented bucket mapping. |
