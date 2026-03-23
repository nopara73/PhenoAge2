# Data Dictionary

| Variable | Clinical Meaning | Units | Notes |
| --- | --- | --- | --- |
| `SEQN` | NHANES participant identifier | none | Stable join key across all cohort and outcomes files. |
| `HSAGEIR` | Age at interview | years | Retained in cohort files for age-aware downstream modeling. |
| `ACP` | alpha carotene | ug/dL | Serum laboratory measurement for serum alpha carotene. |
| `AMP` | albumin | g/dL | Serum laboratory measurement for albumin. |
| `APPSI` | alkaline phosphatase | U/L | Serum laboratory measurement for alkaline phosphatase (SI). |
| `ASPSI` | Aspartate aminotransferase | U/L | Same underlying measurement as ASP, but reported in SI units. |
| `ATPSI` | Alanine aminotransferase | U/L | Same underlying measurement as ATP, but reported in SI units. |
| `BCP` | beta carotene | ug/dL | Serum laboratory measurement for serum beta carotene. |
| `BUP` | blood urea nitrogen | mg/dL | Serum laboratory measurement for blood urea nitrogen (BUN). |
| `BXP` | beta cryptoxanthin | ug/dL | Serum laboratory measurement for serum beta cryptoxanthin. |
| `C1P` | C-peptide | pmol/mL | Serum laboratory measurement for serum c-peptide. |
| `C3PSI` | bicarbonate | mmol/L | Serum laboratory measurement for Serum bicarbonate: (SI, mmol/L) (SI units). |
| `CAPSI` | total calcium | mmol/L | Serum laboratory measurement for Serum total calcium: (SI, mmol/L) (SI units). |
| `CEP` | creatinine | mg/dL | Serum laboratory measurement for creatinine. |
| `CLPSI` | chloride | mmol/L | Serum laboratory measurement for Serum chloride: (SI, mmol/L) (SI units). |
| `CRP` | C-reactive protein | mg/dL | Serum laboratory measurement for C-reactive protein. |
| `DWP` | Platelet distribution width | percent | NHANES III variable capturing platelet distribution width. |
| `FEP` | iron | ug/dL | Serum laboratory measurement for serum iron. |
| `FOP` | folate | ng/mL | Serum laboratory measurement for serum folate. |
| `FRP` | ferritin | ng/mL | Serum laboratory measurement for serum ferritin. |
| `GHP` | Glycated hemoglobin | percent | NHANES III variable capturing glycohemoglobin / HbA1c-related measure. |
| `GRP` | Granulocyte number | Coulter | NHANES III variable capturing granulocyte number. |
| `GRPPCNT` | Granulocyte percent | Coulter | Laboratory percentage measurement for granulocyte percent. |
| `HDP` | HDL cholesterol | mg/dL | Serum laboratory measurement for HDL cholesterol. |
| `HGP` | Hemoglobin | g/dL | Blood-based laboratory measurement for hemoglobin. |
| `HTP` | Hematocrit | percent | NHANES III variable capturing hematocrit. |
| `I1P` | insulin | uU/mL | Serum laboratory measurement for serum insulin, first draw. |
| `LDPSI` | lactate dehydrogenase | U/L | Serum laboratory measurement for Serum lactate dehydrogenase: (SI, U/L) (SI units). |
| `LMP` | Lymphocyte number | Coulter | NHANES III variable capturing lymphocyte number. |
| `LMPPCNT` | Lymphocyte percent | Coulter | Laboratory percentage measurement for lymphocyte percent. |
| `LUP` | lutein/zeaxanthin | ug/dL | Serum laboratory measurement for serum lutein/zeaxanthin. |
| `LYP` | lycopene | ug/dL | Serum laboratory measurement for serum lycopene. |
| `MCPSI` | Mean cell hemoglobin | pg | Same underlying measurement as MCP, but reported in SI units. |
| `MHP` | Mean cell hemoglobin concentration | not stated in source label | NHANES III variable capturing mean corpuscular hemoglobin concentration (MCHC). |
| `MOP` | Mononuclear number | Coulter | NHANES III variable capturing mononuclear number. |
| `MOPPCNT` | Mononuclear percent | Coulter | Laboratory percentage measurement for mononuclear percent. |
| `MVPSI` | Mean cell volume | fL | Same underlying measurement as MVP, but reported in SI units. |
| `NAPSI` | sodium | mmol/L | Serum laboratory measurement for Serum sodium: (SI, mmol/L) (SI units). |
| `PBP` | Lead | ug/dL | NHANES III variable capturing lead. |
| `PLP` | Platelet count | not stated in source label | Laboratory count measurement for platelet count. |
| `PSP` | phosphorus | mg/dL | Serum laboratory measurement for serum phosphorus. |
| `PVPSI` | Mean platelet volume | fL | Same underlying measurement as PVP, but reported in SI units. |
| `PXP` | transferrin saturation | percent | Serum laboratory measurement for serum transferrin saturation. |
| `RCP` | Red blood cell count | not stated in source label | Laboratory count measurement for RBC count. |
| `RWP` | Red cell distribution width | percent | NHANES III variable capturing red cell distribution width (RDW). |
| `SCP` | total calcium | mg/dL | Serum laboratory measurement for serum total calcium. |
| `SEP` | selenium | ng/mL | Serum laboratory measurement for serum selenium. |
| `SGP` | glucose | mg/dL | Serum laboratory measurement for serum glucose. |
| `SKPSI` | potassium | mmol/L | Serum laboratory measurement for Serum potassium: (SI, mmol/L) (SI units). |
| `TBP` | total bilirubin | mg/dL | Serum laboratory measurement for total bilirubin. |
| `TCP` | cholesterol | mg/dL | Serum laboratory measurement for total cholesterol. |
| `TGP` | triglycerides | mg/dL | Serum laboratory measurement for triglycerides. |
| `TIP` | TIBC | ug/dL | Serum laboratory measurement for serum tibc. |
| `TPP` | total protein | g/dL | Serum laboratory measurement for total protein. |
| `UAP` | uric acid | mg/dL | Serum laboratory measurement for uric acid. |
| `VAP` | vitamin A | ug/dL | Serum laboratory measurement for serum vitamin a. |
| `VCP` | vitamin C | mg/dL | Serum laboratory measurement for serum vitamin c. |
| `VEP` | vitamin E | ug/dL | Serum laboratory measurement for serum vitamin e. |
| `WCP` | White blood cell count | not stated in source label | Laboratory count measurement for WBC count. |
| `permth_exm` | Person-months of follow-up from exam | months | Raw mortality linkage field. |
| `time_months` | Analysis follow-up time | months | Copied directly from `permth_exm`. |
| `mortstat` | Final mortality status | 0/1 | `0` assumed alive, `1` assumed deceased. |
| `ucod_leading` | Underlying leading cause of death recode | UCOD bucket | Meaningful only when `mortstat == 1`. |
| `aging_related_event` | Aging-related mortality endpoint | 0/1 | Derived from `mortstat` and `ucod_leading` using the documented bucket mapping. |
