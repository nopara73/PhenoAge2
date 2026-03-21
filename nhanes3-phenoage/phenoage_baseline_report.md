# PhenoAge Baseline Validation Report

## Cohort

- Input file: `C:\Users\user\Desktop\PhenoAge2\nhanes3-phenoage\phenoage_baseline.csv`
- Participants: 9358
- All-cause deaths: 3797
- Aging-related deaths: 2710
- Mean follow-up (months): 267.0778
- Median follow-up (months): 316.0000

## Aging-Related Mortality Discrimination

| Metric | ROC AUC | Mean if event=1 | Median if event=1 | Mean if event=0 | Median if event=0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PhenoAge | 0.8534 | 67.2244 | 68.6720 | 39.9689 | 35.5914 |
| PhenoAge Advance | 0.6342 | 3.4392 | 1.7615 | 0.0649 | -0.8943 |

## Time-To-Event Sanity Check

These correlations are computed only among participants with `aging_related_event = 1`. More positive score should generally correspond to shorter observed follow-up, so a negative correlation is directionally sensible.

| Metric | Pearson corr(score, time_months) among aging-related deaths |
| --- | ---: |
| PhenoAge | -0.5551 |
| PhenoAge Advance | -0.2511 |

## Decile Contrast

This compares the aging-related event rate in the highest-score decile versus the lowest-score decile.

| Metric | Bottom decile cutoff | Bottom decile event rate | Top decile cutoff | Top decile event rate |
| --- | ---: | ---: | ---: | ---: |
| PhenoAge | 22.5092 | 0.0288 | 79.8244 | 0.7340 |
| PhenoAge Advance | -5.8105 | 0.1859 | 8.8614 | 0.5128 |

## Notes

- This report benchmarks the original PhenoAge baseline against the repo's `aging_related_event` endpoint.
- ROC AUC treats the endpoint as binary and ignores censoring.
- The time-to-event correlation is only a quick directional check, not a full survival-model evaluation.
