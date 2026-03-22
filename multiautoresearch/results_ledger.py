from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.tsv"
HEADER = [
    "commit",
    "development_cindex",
    "baseline_cindex",
    "delta_cindex",
    "memory_gb",
    "status",
    "candidate_path",
    "description",
]
LEGACY_HEADER = ["commit", "development_cindex", "memory_gb", "status", "description"]


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {column: row.get(column, "") for column in HEADER}
    if not normalized["development_cindex"] and row.get("val_cindex"):
        normalized["development_cindex"] = row["val_cindex"]
    if (
        not normalized["delta_cindex"]
        and normalized["development_cindex"]
        and normalized["baseline_cindex"]
    ):
        development = float(normalized["development_cindex"])
        baseline = float(normalized["baseline_cindex"])
        normalized["delta_cindex"] = f"{development - baseline:.6f}"
    return normalized


def load_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return []
        rows = [row for row in reader if row.get("commit") != "commit"]
    return [_normalize_row(row) for row in rows]


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t")
            writer.writeheader()
        return

    with RESULTS_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        existing_header = reader.fieldnames or []
        rows = list(reader)
    if existing_header == HEADER:
        return
    if existing_header not in (LEGACY_HEADER, HEADER):
        raise ValueError(
            f"Unexpected results.tsv header: {existing_header}. Expected {HEADER}."
        )

    normalized_rows = [_normalize_row(row) for row in rows if row.get("commit") != "commit"]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(normalized_rows)


def append_result(
    *,
    commit: str,
    development_cindex: float,
    baseline_cindex: float | None,
    delta_cindex: float | None,
    memory_gb: float,
    status: str,
    candidate_path: str,
    description: str,
) -> None:
    ensure_results_file()
    row = {
        "commit": commit,
        "development_cindex": f"{development_cindex:.6f}",
        "baseline_cindex": "" if baseline_cindex is None else f"{baseline_cindex:.6f}",
        "delta_cindex": "" if delta_cindex is None else f"{delta_cindex:.6f}",
        "memory_gb": f"{memory_gb:.3f}",
        "status": status,
        "candidate_path": candidate_path,
        "description": description,
    }
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t")
        writer.writerow(row)
