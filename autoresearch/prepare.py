"""Fixed BioAge autoresearch harness.

This file owns the immutable benchmark contract for biomarker-subset search:
- frozen NHANES III fasting BioAge package
- frozen development/test split
- allowed candidate biomarker pool
- development-only validation split helper
- held-out C-index evaluation
- saved-model metadata contract
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Frozen benchmark configuration (do not modify in the search loop)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60
DEV_VAL_FRACTION = 0.20
DEV_VAL_SEED = 20260321
SUPERIORITY_THRESHOLD = 0.01

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "nhanes3-bioage"
COHORT_PATH = DATA_DIR / "cohort.csv"
OUTCOMES_PATH = DATA_DIR / "outcomes.csv"
SPLIT_PATH = DATA_DIR / "frozen_split.csv"
DEFAULT_CANDIDATE_MODEL_PATH = Path(__file__).resolve().parent / "candidate_bioage.pt"
DEFAULT_CANDIDATE_METADATA_PATH = Path(__file__).resolve().parent / "candidate_bioage.metadata.json"
DEFAULT_RESULT_PATH = REPO_ROOT / "bioage_test_result.json"

AGE_COLUMN = "HSAGEIR"
REFERENCE_PHENOAGE_BIOMARKERS = (
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

CANDIDATE_BIOMARKER_COLUMNS = (
    "ACP",
    "AMP",
    "APPSI",
    "ASPSI",
    "ATPSI",
    "BCP",
    "BUP",
    "BXP",
    "C1P",
    "C3PSI",
    "CAPSI",
    "CEP",
    "CLPSI",
    "CRP",
    "DWP",
    "FEP",
    "FOP",
    "FRP",
    "GHP",
    "GRP",
    "GRPPCNT",
    "HDP",
    "HGP",
    "HTP",
    "I1P",
    "LDPSI",
    "LMP",
    "LMPPCNT",
    "LUP",
    "LYP",
    "MCPSI",
    "MHP",
    "MOP",
    "MOPPCNT",
    "MVPSI",
    "NAPSI",
    "PBP",
    "PLP",
    "PSP",
    "PVPSI",
    "PXP",
    "RCP",
    "RWP",
    "SCP",
    "SEP",
    "SGP",
    "SKPSI",
    "TBP",
    "TCP",
    "TGP",
    "TIP",
    "TPP",
    "UAP",
    "VAP",
    "VCP",
    "VEP",
    "WCP",
)

VALID_FEATURE_COLUMNS = (AGE_COLUMN, *CANDIDATE_BIOMARKER_COLUMNS)
TIME_COLUMN = "time_months"
EVENT_COLUMN = "aging_related_event"
ALLOWED_SPLITS = {"development", "test"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return rows


def validate_feature_columns(feature_columns: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    columns = tuple(feature_columns)
    if not columns:
        raise ValueError("feature_columns must contain at least one input column.")

    seen: set[str] = set()
    for column in columns:
        if column in seen:
            raise ValueError(f"Duplicate feature column requested: {column}")
        if column not in VALID_FEATURE_COLUMNS:
            raise ValueError(f"Unexpected feature column requested: {column}")
        seen.add(column)
    return columns


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


def _parse_feature_value(raw_value: str) -> float:
    stripped = raw_value.strip()
    if stripped == "":
        return float("nan")
    return float(stripped)


def feature_matrix(
    rows: list[dict[str, str]],
    feature_columns: tuple[str, ...] | list[str],
    imputation_values: np.ndarray | None = None,
) -> np.ndarray:
    columns = validate_feature_columns(feature_columns)
    matrix = np.asarray(
        [[_parse_feature_value(row[column]) for column in columns] for row in rows],
        dtype=np.float32,
    )
    if imputation_values is not None:
        matrix = impute_feature_matrix(matrix, imputation_values)
    return matrix


def survival_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(row[TIME_COLUMN]) for row in rows], dtype=np.float64)
    events = np.asarray([int(row[EVENT_COLUMN]) for row in rows], dtype=np.int64)
    return times, events


def fit_feature_imputer(
    rows: list[dict[str, str]],
    feature_columns: tuple[str, ...] | list[str],
) -> np.ndarray:
    matrix = feature_matrix(rows, feature_columns)
    with np.errstate(invalid="ignore"):
        imputation_values = np.nanmean(matrix, axis=0)
    imputation_values = np.where(np.isnan(imputation_values), 0.0, imputation_values)
    return imputation_values.astype(np.float32)


def impute_feature_matrix(matrix: np.ndarray, imputation_values: np.ndarray) -> np.ndarray:
    if matrix.shape[1] != len(imputation_values):
        raise ValueError("Imputation values do not match matrix width.")
    imputed = matrix.copy()
    missing = np.isnan(imputed)
    if np.any(missing):
        imputed[missing] = np.take(imputation_values, np.where(missing)[1])
    return imputed.astype(np.float32)


def tensorize_features(
    rows: list[dict[str, str]],
    device: torch.device | str,
    feature_columns: tuple[str, ...] | list[str],
    imputation_values: np.ndarray,
) -> torch.Tensor:
    return torch.tensor(
        feature_matrix(rows, feature_columns, imputation_values=imputation_values),
        dtype=torch.float32,
        device=device,
    )


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
def evaluate_cindex(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    device: torch.device | str,
    feature_columns: tuple[str, ...] | list[str],
    imputation_values: np.ndarray,
) -> float:
    model.eval()
    scores = model(tensorize_features(rows, device, feature_columns, imputation_values)).reshape(-1).detach().cpu().numpy()
    times, events = survival_arrays(rows)
    return harrell_c_index(times, events, scores)


def candidate_metadata_path_for_model(model_path: Path) -> Path:
    return model_path.with_suffix(".metadata.json")


def save_candidate_metadata(
    model_path: Path,
    feature_columns: tuple[str, ...] | list[str],
    imputation_values: np.ndarray,
) -> None:
    columns = validate_feature_columns(feature_columns)
    metadata = {
        "benchmark_dataset": "nhanes3-bioage",
        "feature_columns": list(columns),
        "input_feature_count": len(columns),
        "age_included_in_inputs": AGE_COLUMN in columns,
        "imputation_values": [float(value) for value in imputation_values],
    }
    metadata_path = candidate_metadata_path_for_model(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_candidate_metadata(model_path: Path) -> dict[str, object]:
    metadata_path = candidate_metadata_path_for_model(model_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Candidate metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = validate_feature_columns(payload["feature_columns"])
    if len(payload["imputation_values"]) != len(feature_columns):
        raise ValueError("Candidate metadata contains mismatched imputation values.")
    return payload


def candidate_feature_columns(model_path: Path) -> tuple[str, ...]:
    payload = load_candidate_metadata(model_path)
    return validate_feature_columns(payload["feature_columns"])


def candidate_imputation_values(model_path: Path) -> np.ndarray:
    payload = load_candidate_metadata(model_path)
    return np.asarray(payload["imputation_values"], dtype=np.float32)


@torch.no_grad()
def score_scripted_model(model_path: Path, rows: list[dict[str, str]], device: torch.device | str) -> np.ndarray:
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    feature_columns = candidate_feature_columns(model_path)
    imputation_values = candidate_imputation_values(model_path)
    features = tensorize_features(rows, device, feature_columns, imputation_values)
    return model(features).reshape(-1).detach().cpu().numpy()


def build_result_summary(test_c_index: float) -> dict[str, float]:
    return {
        "test_c_index": test_c_index,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the fixed BioAge autoresearch harness.")
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Print frozen split counts for the BioAge subset-search benchmark.",
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
        _, development_events = survival_arrays(development_rows)
        _, test_events = survival_arrays(test_rows)
        print(f"Development aging-related deaths: {int(development_events.sum())}")
        print(f"Test aging-related deaths: {int(test_events.sum())}")


if __name__ == "__main__":
    main()
