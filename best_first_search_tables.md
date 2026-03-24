# Best-By-Size Search Status

Snapshot timestamp: `2026-03-24T14:43:35.713761+00:00`

Current run status:

- `stage`: `search`
- `evaluation_count`: `7684`
- `cache_hit_count`: `582`
- `frontier_pop_count`: `4960`
- `frontier_size`: `7686`
- `search_tier`: `B`

## With Age

`Total inputs` includes chronological age.


| Total Inputs | Biomarker Count | C-index  | Biomarkers                                                                                                                                                                    |
| ------------ | --------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2            | 1               | 0.864243 | vitamin C                                                                                                                                                                     |
| 3            | 2               | 0.867190 | vitamin C, white blood cell count                                                                                                                                             |
| 4            | 3               | 0.868774 | alpha carotene, vitamin C, white blood cell count                                                                                                                             |
| 5            | 4               | 0.870205 | HDL cholesterol, mean cell hemoglobin concentration, vitamin C, white blood cell count                                                                                        |
| 6            | 5               | 0.872251 | alpha carotene, HDL cholesterol, mean cell hemoglobin concentration, uric acid, white blood cell count                                                                        |
| 7            | 6               | 0.872325 | alpha carotene, HDL cholesterol, mean cell hemoglobin concentration, uric acid, vitamin C, white blood cell count                                                             |
| 8            | 7               | 0.872153 | alpha carotene, glycated hemoglobin, granulocyte number, lead, uric acid, vitamin C, white blood cell count                                                                   |
| 9            | 8               | 0.871479 | alpha carotene, blood urea nitrogen, HDL cholesterol, lead, uric acid, vitamin A, vitamin C, white blood cell count                                                           |
| 10           | 9               | 0.872778 | alpha carotene, blood urea nitrogen, granulocyte number, HDL cholesterol, lutein/zeaxanthin, mean cell hemoglobin concentration, uric acid, vitamin C, white blood cell count |


## Without Age

Here `inputs = biomarker count`.


| Inputs | C-index  | Biomarkers                                                                                                                               |
| ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | 0.682399 | glycated hemoglobin                                                                                                                      |
| 2      | 0.723624 | blood urea nitrogen, glycated hemoglobin                                                                                                 |
| 3      | 0.752300 | glycated hemoglobin, lycopene, vitamin E                                                                                                 |
| 4      | 0.772208 | blood urea nitrogen, glycated hemoglobin, lycopene, cholesterol                                                                          |
| 5      | 0.783507 | blood urea nitrogen, glycated hemoglobin, lycopene, mean cell volume, cholesterol                                                        |
| 6      | 0.790976 | albumin, blood urea nitrogen, glycated hemoglobin, lycopene, mean cell volume, cholesterol                                               |
| 7      | 0.797939 | albumin, blood urea nitrogen, glycated hemoglobin, lycopene, mean cell volume, lead, vitamin E                                           |
| 8      | 0.801644 | albumin, blood urea nitrogen, glycated hemoglobin, lycopene, mean cell volume, lead, red cell distribution width, vitamin E              |
| 9      | 0.800413 | albumin, blood urea nitrogen, glycated hemoglobin, lycopene, mean cell volume, lead, red cell distribution width, glucose, vitamin E     |
| 10     | 0.802306 | albumin, blood urea nitrogen, creatinine, folate, glycated hemoglobin, lycopene, lead, red cell distribution width, glucose, cholesterol |


