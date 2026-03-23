"""Fixed PA2 autoresearch harness.

This file owns the immutable benchmark contract for PhenoAge 2.0 search:
- frozen NHANES III package
- frozen development/test split
- allowed input features
- original PhenoAge baseline scorer
- development-only validation split helper
- held-out C-index evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Frozen benchmark configuration (do not modify in the search loop)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300
DEV_VAL_FRACTION = 0.20
DEV_VAL_SEED = 20260321
SUPERIORITY_THRESHOLD = 0.01

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "nhanes3-phenoage"
COHORT_PATH = DATA_DIR / "cohort.csv"
OUTCOMES_PATH = DATA_DIR / "outcomes.csv"
SPLIT_PATH = DATA_DIR / "frozen_split.csv"
DEFAULT_CANDIDATE_MODEL_PATH = Path(__file__).resolve().parent / "candidate_pa2.pt"
DEFAULT_RESULT_PATH = REPO_ROOT / "pa2_test_result.json"

FEATURE_COLUMNS = (
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
AGE_COLUMN = "HSAGEIR"
TIME_COLUMN = "time_months"
EVENT_COLUMN = "aging_related_event"
ALLOWED_SPLITS = {"development", "test"}

# ---------------------------------------------------------------------------
# Original PhenoAge constants
# ---------------------------------------------------------------------------

ALBUMIN_G_PER_DL_TO_G_PER_L = 10.0
CREATININE_MG_PER_DL_TO_UMOL_PER_L = 88.4
GLUCOSE_MG_PER_DL_TO_MMOL_PER_L = 1.0 / 18.0182
CRP_MG_PER_DL_TO_MG_PER_L = 10.0

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
PHENOAGE_DENOMINATOR = 0.090165
PHENOAGE_INTERCEPT = 141.50225


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return rows


def load_joined_rows() -> list[dict[str, str]]:
    cohort_rows = {row["SEQN"]: row for row in read_csv_rows(COHORT_PATH)}
    outcomes_rows = {row["SEQN"]: row for row in read_csv_rows(OUTCOMES_PATH)}
    split_rows = {row["SEQN"]: row for row in read_csv_rows(SPLIT_PATH)}

    if set(cohort_rows) != set(outcomes_rows) or set(cohort_rows) != set(split_rows):
        raise ValueError("cohort/outcomes/split participant sets do not match.")

    joined_rows: list[dict[str, str]] = []
    for seqn in sorted(cohort_rows, key=int):
        row = dict(cohort_rows[seqn])
        row.update({k: v for k, v in outcomes_rows[seqn].items() if k != "SEQN"})
        row["split"] = split_rows[seqn]["split"]
        if row["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"Unexpected split for SEQN {seqn}: {row['split']}")
        joined_rows.append(row)
    return joined_rows


def get_rows_for_split(rows: list[dict[str, str]], split_name: str) -> list[dict[str, str]]:
    if split_name not in ALLOWED_SPLITS:
        raise ValueError(f"Unexpected split name: {split_name}")
    return [row for row in rows if row["split"] == split_name]


def stratified_development_split(
    rows: list[dict[str, str]],
    val_fraction: float = DEV_VAL_FRACTION,
    seed: int = DEV_VAL_SEED,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    development_rows = get_rows_for_split(rows, "development")
    grouped: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
    for row in development_rows:
        grouped[row[EVENT_COLUMN]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for label, label_rows in grouped.items():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_fraction))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1
        val_seqn = {row["SEQN"] for row in shuffled[:val_count]}
        for row in label_rows:
            if row["SEQN"] in val_seqn:
                val_rows.append(row)
            else:
                train_rows.append(row)

    train_rows.sort(key=lambda row: int(row["SEQN"]))
    val_rows.sort(key=lambda row: int(row["SEQN"]))
    return train_rows, val_rows


def feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows],
        dtype=np.float32,
    )


def survival_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(row[TIME_COLUMN]) for row in rows], dtype=np.float64)
    events = np.asarray([int(row[EVENT_COLUMN]) for row in rows], dtype=np.int64)
    return times, events


def fit_standardizer(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    features = feature_matrix(rows)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def tensorize_features(rows: list[dict[str, str]], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(feature_matrix(rows), dtype=torch.float32, device=device)


def _compute_phenoage_xb(row: dict[str, str], include_age: bool) -> float:
    xb = (
        XB_INTERCEPT
        + ALBUMIN_COEF * (float(row["AMP"]) * ALBUMIN_G_PER_DL_TO_G_PER_L)
        + CREATININE_COEF * (float(row["CEP"]) * CREATININE_MG_PER_DL_TO_UMOL_PER_L)
        + GLUCOSE_COEF * (float(row["SGP"]) * GLUCOSE_MG_PER_DL_TO_MMOL_PER_L)
        + LOG_CRP_COEF * math.log(float(row["CRP"]) * CRP_MG_PER_DL_TO_MG_PER_L)
        + LYMPHOCYTE_PERCENT_COEF * float(row["LMPPCNT"])
        + MEAN_CELL_VOLUME_COEF * float(row["MVPSI"])
        + RDW_COEF * float(row["RWP"])
        + ALKALINE_PHOSPHATASE_COEF * float(row["APPSI"])
        + WBC_COEF * float(row["WCP"])
    )
    if include_age:
        xb += AGE_COEF * float(row[AGE_COLUMN])
    return xb


def compute_phenoage(row: dict[str, str], include_age: bool = True) -> float:
    xb = _compute_phenoage_xb(row, include_age=include_age)
    mortality_component = MORTALITY_NUMERATOR_COEF * math.exp(xb) / MORTALITY_DENOMINATOR
    return PHENOAGE_INTERCEPT + math.log(PHENOAGE_LOG_COEF * mortality_component) / PHENOAGE_DENOMINATOR


def compute_original_phenoage_scores(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([compute_phenoage(row, include_age=True) for row in rows], dtype=np.float64)


def compute_phenoage_without_age_scores(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([compute_phenoage(row, include_age=False) for row in rows], dtype=np.float64)


def harrell_c_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    concordant = 0.0
    tied = 0.0
    comparable = 0.0
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = times[i], times[j]
            ei, ej = events[i], events[j]
            si, sj = scores[i], scores[j]

            if ti == tj and ei == ej == 1:
                comparable += 1.0
                if si == sj:
                    tied += 1.0
                elif si > sj:
                    concordant += 1.0
                continue

            if ei == 1 and ti < tj:
                comparable += 1.0
                if si == sj:
                    tied += 1.0
                elif si > sj:
                    concordant += 1.0
            elif ej == 1 and tj < ti:
                comparable += 1.0
                if si == sj:
                    tied += 1.0
                elif sj > si:
                    concordant += 1.0

    if comparable == 0.0:
        raise ValueError("No comparable pairs available for C-index.")
    return (concordant + 0.5 * tied) / comparable


@torch.no_grad()
def evaluate_cindex(model: torch.nn.Module, rows: list[dict[str, str]], device: torch.device | str) -> float:
    model.eval()
    scores = model(tensorize_features(rows, device)).reshape(-1).detach().cpu().numpy()
    times, events = survival_arrays(rows)
    return harrell_c_index(times, events, scores)


@torch.no_grad()
def score_scripted_model(model_path: Path, rows: list[dict[str, str]], device: torch.device | str) -> np.ndarray:
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    features = tensorize_features(rows, device)
    return model(features).reshape(-1).detach().cpu().numpy()


def final_verdict(delta: float) -> str:
    if delta >= SUPERIORITY_THRESHOLD:
        return "superior"
    return "inferior"


def build_result_summary(pa2_c_index: float, phenoage_c_index: float) -> dict[str, float | str]:
    delta = pa2_c_index - phenoage_c_index
    return {
        "phenoage_c_index": phenoage_c_index,
        "pa2_c_index": pa2_c_index,
        "delta": delta,
        "verdict": final_verdict(delta),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the fixed PA2 autoresearch harness.")
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Print frozen split counts and baseline held-out C-index.",
    )
    args = parser.parse_args()

    rows = load_joined_rows()
    development_rows = get_rows_for_split(rows, "development")
    test_rows = get_rows_for_split(rows, "test")
    train_rows, val_rows = stratified_development_split(rows)
    print(f"Loaded {len(rows)} joined participants from {DATA_DIR}")
    print(f"Development participants: {len(development_rows)}")
    print(f"Test participants: {len(test_rows)}")
    print(f"Development train/val split: {len(train_rows)}/{len(val_rows)}")
    if args.show_counts:
        times, events = survival_arrays(test_rows)
        phenoage_c = harrell_c_index(times, events, compute_original_phenoage_scores(test_rows))
        print(f"Held-out PhenoAge C-index: {phenoage_c:.6f}")


if __name__ == "__main__":
    main()
