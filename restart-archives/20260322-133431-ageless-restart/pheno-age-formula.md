# PhenoAge Formula

## Units Expected By This Formula

The original PhenoAge coefficients are used with these input units:

- Albumin: g/L
- Creatinine: umol/L
- Glucose: mmol/L
- CRP: mg/L before taking `ln(CRP)`
- Lymphocyte Percent: percent
- Mean Cell Volume: fL
- Red Cell Distribution Width: percent
- Alkaline Phosphatase: U/L
- White Blood Cell Count: `10^3 cells/uL`
- Chronological Age: years

For this repo's `nhanes3-phenoage/cohort.csv`, the raw cohort units differ for several biomarkers:

- `AMP` albumin is stored in g/dL, so convert to g/L by multiplying by 10.
- `CEP` creatinine is stored in mg/dL, so convert to umol/L by multiplying by 88.4.
- `SGP` glucose is stored in mg/dL, so convert to mmol/L by dividing by 18.0182.
- `CRP` is stored in mg/dL, so convert to mg/L by multiplying by 10 before taking the natural log.
- `LMPPCNT`, `MVPSI`, `RWP`, and `APPSI` are already on the needed scale.
- `WCP` appears to already be on the needed `10^3 cells/uL` scale.
- `HSAGEIR` is already in years.

```text
Phenotypic Age = 141.50 + ln(-0.00553 * ln(1 - M)) / 0.090165
```

Where:

```text
M = 1 - exp(-1.51714 * exp(xb) / 0.0076927)
```

and:

```text
xb = -19.907
     - 0.0336 * Albumin
     + 0.0095 * Creatinine
     + 0.1953 * Glucose
     + 0.0954 * ln(CRP)
     - 0.0120 * Lymphocyte Percent
     + 0.0268 * Mean Cell Volume
     + 0.3306 * Red Cell Distribution Width
     + 0.00188 * Alkaline Phosphatase
     + 0.0554 * White Blood Cell Count
     + 0.0804 * Chronological Age
```
