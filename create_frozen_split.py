from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "nhanes3-bioage"
DEFAULT_OUTCOMES_PATH = DEFAULT_DATA_DIR / "outcomes.csv"
DEFAULT_SPLIT_PATH = DEFAULT_DATA_DIR / "frozen_split.csv"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "frozen_split_manifest.json"
DEFAULT_TEST_FRACTION = 0.20
DEFAULT_RANDOM_SEED = 20260321

REQUIRED_COLUMNS = ("SEQN", "aging_related_event")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a frozen participant-level development/test split for nhanes3-bioage."
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=DEFAULT_OUTCOMES_PATH,
        help=f"Path to outcomes.csv (default: {DEFAULT_OUTCOMES_PATH})",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help=f"Where to write the split CSV (default: {DEFAULT_SPLIT_PATH})",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Where to write the split manifest JSON (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=f"Fraction of participants to assign to the final test set (default: {DEFAULT_TEST_FRACTION})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for the frozen split (default: {DEFAULT_RANDOM_SEED})",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row.")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} does not contain any data rows.")
    return rows


def stratified_split(
    rows: list[dict[str, str]],
    test_fraction: float,
    seed: int,
) -> list[dict[str, str]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    groups: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
    seen_seqn: set[str] = set()

    for row in rows:
        seqn = row["SEQN"]
        if seqn in seen_seqn:
            raise ValueError(f"Duplicate SEQN in outcomes.csv: {seqn}")
        seen_seqn.add(seqn)

        label = row["aging_related_event"]
        if label not in groups:
            raise ValueError(f"Unexpected aging_related_event value for SEQN {seqn}: {label}")
        groups[label].append(row)

    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []

    for label, group_rows in groups.items():
        shuffled = list(group_rows)
        rng.shuffle(shuffled)
        test_count = round(len(shuffled) * test_fraction)
        if test_count <= 0 or test_count >= len(shuffled):
            raise ValueError(f"Invalid test_count for label {label}: {test_count}")

        test_seqn = {row["SEQN"] for row in shuffled[:test_count]}

        for row in group_rows:
            split_rows.append(
                {
                    "SEQN": row["SEQN"],
                    "aging_related_event": label,
                    "split": "test" if row["SEQN"] in test_seqn else "development",
                }
            )

    split_rows.sort(key=lambda row: int(row["SEQN"]))
    return split_rows


def build_manifest(
    split_rows: list[dict[str, str]],
    outcomes_path: Path,
    test_fraction: float,
    seed: int,
) -> dict[str, object]:
    counts_by_split = {"development": 0, "test": 0}
    counts_by_label = {
        "development": {"0": 0, "1": 0},
        "test": {"0": 0, "1": 0},
    }

    for row in split_rows:
        split = row["split"]
        label = row["aging_related_event"]
        counts_by_split[split] += 1
        counts_by_label[split][label] += 1

    return {
        "benchmark_dataset": "nhanes3-bioage",
        "source_outcomes": str(outcomes_path),
        "test_fraction": test_fraction,
        "random_seed": seed,
        "participant_count": len(split_rows),
        "counts_by_split": counts_by_split,
        "aging_related_event_counts": counts_by_label,
    }


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("SEQN", "aging_related_event", "split"))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_json(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.outcomes)
    split_rows = stratified_split(rows, args.test_fraction, args.seed)
    manifest = build_manifest(split_rows, args.outcomes, args.test_fraction, args.seed)
    write_split_csv(args.split_output, split_rows)
    write_manifest_json(args.manifest_output, manifest)
    print(f"Wrote split CSV to: {args.split_output}")
    print(f"Wrote split manifest to: {args.manifest_output}")
    print(f"Development participants: {manifest['counts_by_split']['development']}")
    print(f"Test participants: {manifest['counts_by_split']['test']}")


if __name__ == "__main__":
    main()
