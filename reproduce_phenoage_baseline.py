from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "nhanes3-phenoage"
DEFAULT_COHORT_PATH = DEFAULT_DATA_DIR / "cohort.csv"
DEFAULT_OUTCOMES_PATH = DEFAULT_DATA_DIR / "outcomes.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "phenoage_baseline.csv"

COHORT_REQUIRED_COLUMNS = (
    "SEQN",
    "HSAGEIR",
    "AMP",
    "CEP",
    "SGP",
    "CRP",
    "LMPPCNT",
    "MVPSI",
    "RWP",
    "APPSI",
    "WCP",
)

OUTCOMES_REQUIRED_COLUMNS = (
    "SEQN",
    "permth_exm",
    "time_months",
    "mortstat",
    "ucod_leading",
    "aging_related_event",
)

ALBUMIN_G_PER_DL_TO_G_PER_L = 10.0
CREATININE_MG_PER_DL_TO_UMOL_PER_L = 88.4
GLUCOSE_MG_PER_DL_TO_MMOL_PER_L = 1.0 / 18.0182
CRP_MG_PER_DL_TO_MG_PER_L = 10.0

# Original PhenoAge coefficients/constants as reproduced by the BioAge package.
XB_INTERCEPT = -19.90667
ALBUMIN_COEF = -0.03359355
CREATININE_COEF = 0.009506491
GLUCOSE_COEF = 0.1953192
LOG_CRP_COEF = 0.09536762
LYMPHOCYTE_PERCENT_COEF = -0.01199984
MEAN_CELL_VOLUME_COEF = 0.02676401
RDW_COEF = 0.3306156
ALKALINE_PHOSPHATASE_COEF = 0.001868778
WBC_COEF = 0.05542406
AGE_COEF = 0.08035356
MORTALITY_NUMERATOR_COEF = 1.51714
MORTALITY_DENOMINATOR = 0.007692696
PHENOAGE_LOG_COEF = 0.0055305
# Keep this denominator aligned with pheno-age-formula.md for this repo.
PHENOAGE_DENOMINATOR = 0.090165
PHENOAGE_INTERCEPT = 141.50225


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join nhanes3-phenoage cohort/outcomes by SEQN and compute "
            "the original PhenoAge score for each participant."
        )
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT_PATH,
        help=f"Path to cohort.csv (default: {DEFAULT_COHORT_PATH})",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=DEFAULT_OUTCOMES_PATH,
        help=f"Path to outcomes.csv (default: {DEFAULT_OUTCOMES_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the joined PhenoAge CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row.")
        return list(reader)


def validate_columns(path: Path, fieldnames: list[str], required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path} is missing required columns: {joined}")


def compute_original_phenoage(cohort_row: dict[str, str]) -> float:
    albumin_g_per_l = float(cohort_row["AMP"]) * ALBUMIN_G_PER_DL_TO_G_PER_L
    creatinine_umol_per_l = float(cohort_row["CEP"]) * CREATININE_MG_PER_DL_TO_UMOL_PER_L
    glucose_mmol_per_l = float(cohort_row["SGP"]) * GLUCOSE_MG_PER_DL_TO_MMOL_PER_L
    crp_mg_per_l = float(cohort_row["CRP"]) * CRP_MG_PER_DL_TO_MG_PER_L
    lymphocyte_percent = float(cohort_row["LMPPCNT"])
    mean_cell_volume = float(cohort_row["MVPSI"])
    red_cell_distribution_width = float(cohort_row["RWP"])
    alkaline_phosphatase = float(cohort_row["APPSI"])
    white_blood_cell_count = float(cohort_row["WCP"])
    chronological_age = float(cohort_row["HSAGEIR"])

    if crp_mg_per_l <= 0.0:
        seqn = cohort_row["SEQN"]
        raise ValueError(f"CRP must be > 0 for the PhenoAge log transform. Bad SEQN: {seqn}")

    xb = (
        XB_INTERCEPT
        + ALBUMIN_COEF * albumin_g_per_l
        + CREATININE_COEF * creatinine_umol_per_l
        + GLUCOSE_COEF * glucose_mmol_per_l
        + LOG_CRP_COEF * math.log(crp_mg_per_l)
        + LYMPHOCYTE_PERCENT_COEF * lymphocyte_percent
        + MEAN_CELL_VOLUME_COEF * mean_cell_volume
        + RDW_COEF * red_cell_distribution_width
        + ALKALINE_PHOSPHATASE_COEF * alkaline_phosphatase
        + WBC_COEF * white_blood_cell_count
        + AGE_COEF * chronological_age
    )

    # M = 1 - exp(-1.51714 * exp(xb) / 0.0076927)
    mortality_component = MORTALITY_NUMERATOR_COEF * math.exp(xb) / MORTALITY_DENOMINATOR

    # Phenotypic Age = 141.50 + ln(-0.00553 * ln(1 - M)) / 0.090165
    # Since 1 - M = exp(-mortality_component), ln(1 - M) = -mortality_component.
    phenoage = PHENOAGE_INTERCEPT + math.log(PHENOAGE_LOG_COEF * mortality_component) / PHENOAGE_DENOMINATOR
    return phenoage


def build_outcomes_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        seqn = row["SEQN"]
        if seqn in index:
            raise ValueError(f"Duplicate SEQN in outcomes.csv: {seqn}")
        index[seqn] = row
    return index


def join_and_score(
    cohort_rows: list[dict[str, str]],
    outcomes_index: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    joined_rows: list[dict[str, str]] = []

    for cohort_row in cohort_rows:
        seqn = cohort_row["SEQN"]
        if seqn not in outcomes_index:
            raise ValueError(f"SEQN present in cohort.csv but missing from outcomes.csv: {seqn}")

        outcomes_row = outcomes_index[seqn]
        phenoage = compute_original_phenoage(cohort_row)
        age_advance = phenoage - float(cohort_row["HSAGEIR"])

        joined_row = dict(cohort_row)
        for key, value in outcomes_row.items():
            if key != "SEQN":
                joined_row[key] = value
        joined_row["phenoage"] = f"{phenoage:.6f}"
        joined_row["phenoage_advance"] = f"{age_advance:.6f}"
        joined_rows.append(joined_row)

    if len(joined_rows) != len(outcomes_index):
        missing_from_cohort = sorted(set(outcomes_index) - {row["SEQN"] for row in cohort_rows})
        raise ValueError(
            "SEQN present in outcomes.csv but missing from cohort.csv: "
            + ", ".join(missing_from_cohort[:10])
        )

    return joined_rows


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No joined rows to write.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    phenoages = [float(row["phenoage"]) for row in rows]
    aging_related_events = sum(int(row["aging_related_event"]) for row in rows)

    print(f"Wrote {len(rows)} joined rows to: {output_path}")
    print(f"Aging-related events: {aging_related_events}")
    print(f"Mean PhenoAge: {statistics.fmean(phenoages):.4f}")
    print(f"Median PhenoAge: {statistics.median(phenoages):.4f}")
    print(f"Min PhenoAge: {min(phenoages):.4f}")
    print(f"Max PhenoAge: {max(phenoages):.4f}")


def main() -> None:
    args = parse_args()

    cohort_rows = read_csv_rows(args.cohort)
    outcomes_rows = read_csv_rows(args.outcomes)

    validate_columns(args.cohort, list(cohort_rows[0].keys()), COHORT_REQUIRED_COLUMNS)
    validate_columns(args.outcomes, list(outcomes_rows[0].keys()), OUTCOMES_REQUIRED_COLUMNS)

    outcomes_index = build_outcomes_index(outcomes_rows)
    joined_rows = join_and_score(cohort_rows, outcomes_index)

    write_output(args.output, joined_rows)
    print_summary(joined_rows, args.output)


if __name__ == "__main__":
    main()
