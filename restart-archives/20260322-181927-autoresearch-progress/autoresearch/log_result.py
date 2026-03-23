from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_LOG_PATH = HERE / "run.log"
RESULTS_PATH = HERE / "results.tsv"
HEADER = ["commit", "development_cindex", "memory_gb", "status", "description"]


def parse_metric(text: str, name: str) -> float:
    pattern = rf"^{re.escape(name)}:\s*([0-9.+-]+)$"
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"Could not find `{name}` in run log.")
    return float(matches[-1])


def get_git_commit() -> str:
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


def ensure_results_file() -> None:
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(HEADER)


def append_result(commit: str, development_cindex: float, memory_gb: float, status: str, description: str) -> None:
    ensure_results_file()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                commit,
                f"{development_cindex:.6f}",
                f"{memory_gb:.3f}",
                status,
                description,
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse run.log and append one row to results.tsv.")
    parser.add_argument("--description", required=True, help="Short description of the experiment.")
    parser.add_argument("--status", required=True, choices=("keep", "discard", "crash"))
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH), help="Path to the run log.")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = HERE / log_path
    if not log_path.exists():
        raise FileNotFoundError(f"Run log not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    if args.status == "crash":
        development_cindex = 0.0
        memory_gb = 0.0
    else:
        development_cindex = parse_metric(text, "development_cindex")
        peak_vram_mb = parse_metric(text, "peak_vram_mb")
        memory_gb = peak_vram_mb / 1024.0

    commit = get_git_commit()
    append_result(
        commit=commit,
        development_cindex=development_cindex,
        memory_gb=memory_gb,
        status=args.status,
        description=args.description,
    )
    print(
        f"Logged {args.status} result: commit={commit} "
        f"development_cindex={development_cindex:.6f} memory_gb={memory_gb:.3f}"
    )


if __name__ == "__main__":
    main()
