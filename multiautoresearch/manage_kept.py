from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare import SUPERIORITY_THRESHOLD
from results_ledger import append_result, load_rows


HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "train.py"
KEPT_PATH = HERE / "last_kept_train.py"
SAVED_CANDIDATES_DIR = HERE / "saved_candidates"
REPO_ROOT = HERE.parent


def _require_kept_snapshot() -> None:
    if KEPT_PATH.exists():
        return
    raise FileNotFoundError(
        f"No kept baseline snapshot exists at {KEPT_PATH}. Restore requires an existing baseline."
    )


def _candidate_relative_path(path: Path) -> str:
    return path.relative_to(HERE).as_posix()


def _candidate_absolute_path(candidate_ref: str) -> Path:
    candidate_path = Path(candidate_ref)
    if candidate_path.is_absolute():
        return candidate_path.resolve()
    direct_path = (HERE / candidate_path).resolve()
    if direct_path.exists():
        return direct_path
    return (SAVED_CANDIDATES_DIR / candidate_path).resolve()


def _find_candidate_row(candidate_path: Path) -> dict[str, str] | None:
    candidate_ref = _candidate_relative_path(candidate_path)
    candidate_name = candidate_path.name
    for row in reversed(load_rows()):
        row_candidate = row.get("candidate_path", "")
        if row_candidate in {candidate_ref, candidate_name}:
            return row
    return None


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def snapshot_candidate(commit: str, development_cindex: float) -> str:
    SAVED_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{commit}_{development_cindex:.6f}.py"
    candidate_path = SAVED_CANDIDATES_DIR / filename
    shutil.copyfile(TRAIN_PATH, candidate_path)
    return _candidate_relative_path(candidate_path)


def restore() -> None:
    _require_kept_snapshot()
    shutil.copyfile(KEPT_PATH, TRAIN_PATH)
    print(f"Restored train.py from the kept ageless baseline at {KEPT_PATH}")


def promote(candidate_ref: str, description: str | None = None) -> None:
    candidate_path = _candidate_absolute_path(candidate_ref)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Saved candidate not found: {candidate_path}")
    if SAVED_CANDIDATES_DIR.resolve() not in candidate_path.parents:
        raise ValueError("Promote only accepts files stored under saved_candidates/.")

    row = _find_candidate_row(candidate_path)
    if row is None:
        raise ValueError("Cannot promote a candidate that is not recorded in results.tsv.")
    if row["status"] != "saved":
        raise ValueError("Only rows recorded as `saved` can be promoted.")
    if not row["delta_cindex"]:
        raise ValueError("Saved candidate row is missing delta_cindex.")
    if float(row["delta_cindex"]) < SUPERIORITY_THRESHOLD:
        raise ValueError(
            f"Saved candidate delta must be at least {SUPERIORITY_THRESHOLD:.2f}."
        )

    shutil.copyfile(candidate_path, TRAIN_PATH)
    shutil.copyfile(candidate_path, KEPT_PATH)
    append_result(
        commit=row["commit"] or _get_git_commit(),
        development_cindex=float(row["development_cindex"]),
        baseline_cindex=float(row["baseline_cindex"]) if row["baseline_cindex"] else None,
        delta_cindex=float(row["delta_cindex"]),
        memory_gb=float(row["memory_gb"] or 0.0),
        status="keep",
        candidate_path=_candidate_relative_path(candidate_path),
        description=description or f"{row['description']} [promoted]",
    )
    print(
        "Promoted saved candidate to the kept baseline: "
        f"{_candidate_relative_path(candidate_path)}"
    )


def list_candidates() -> None:
    rows = [row for row in load_rows() if row["status"] == "saved" and row["candidate_path"]]
    if not rows:
        print("No saved candidates found.")
        return
    for row in rows:
        print(
            f"{row['candidate_path']} | development={row['development_cindex']} "
            f"| baseline={row['baseline_cindex'] or 'n/a'} | delta={row['delta_cindex'] or 'n/a'} "
            f"| commit={row['commit']} | {row['description']}"
        )


def status() -> None:
    train_matches_kept = (
        TRAIN_PATH.exists()
        and KEPT_PATH.exists()
        and TRAIN_PATH.read_bytes() == KEPT_PATH.read_bytes()
    )
    saved_count = len(
        [row for row in load_rows() if row["status"] == "saved" and row["candidate_path"]]
    )
    print(f"train.py:             {TRAIN_PATH}")
    print(f"last_kept_train.py:   {KEPT_PATH}")
    print(f"saved_candidates_dir: {SAVED_CANDIDATES_DIR}")
    print(f"snapshot_exists:      {KEPT_PATH.exists()}")
    print(f"train_matches_kept:   {train_matches_kept}")
    print(f"saved_candidates:     {saved_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the kept baseline or explicitly promote a qualifying saved candidate."
    )
    parser.add_argument("command", choices=("restore", "promote", "list", "status"))
    parser.add_argument("candidate", nargs="?", help="Saved candidate path or filename for promote.")
    parser.add_argument(
        "--description",
        help="Optional description override for the keep row written during promote.",
    )
    args = parser.parse_args()

    if args.command == "restore":
        restore()
    elif args.command == "promote":
        if not args.candidate:
            raise SystemExit("promote requires a saved candidate path or filename.")
        promote(args.candidate, description=args.description)
    elif args.command == "list":
        list_candidates()
    else:
        status()


if __name__ == "__main__":
    main()
