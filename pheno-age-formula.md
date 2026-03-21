# PhenoAge Formula

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
