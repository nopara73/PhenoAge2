from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import prepare
from manage_kept import KEPT_PATH, restore, snapshot_candidate
from prepare import SUPERIORITY_THRESHOLD
from results_ledger import append_result


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_LOG_PATH = HERE / "run.log"


def parse_metric(text: str, name: str) -> float:
    pattern = rf"^{re.escape(name)}:\s+([0-9.+-]+)$"
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


def evaluate_snapshot_development_cindex(snapshot_path: Path) -> float:
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Baseline snapshot not found: {snapshot_path}")

    module_name = f"_multiautoresearch_snapshot_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, snapshot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load snapshot module from {snapshot_path}")

    module = importlib.util.module_from_spec(spec)
    stdout = io.StringIO()
    with tempfile.TemporaryDirectory() as temp_dir:
        original_artifact_path = prepare.DEFAULT_CANDIDATE_MODEL_PATH
        prepare.DEFAULT_CANDIDATE_MODEL_PATH = Path(temp_dir) / "candidate_pa2.pt"
        sys.modules[module_name] = module
        try:
            with contextlib.redirect_stdout(stdout):
                spec.loader.exec_module(module)
                if not hasattr(module, "main"):
                    raise AttributeError(f"{snapshot_path} does not define main()")
                module.main()
        finally:
            prepare.DEFAULT_CANDIDATE_MODEL_PATH = original_artifact_path
            sys.modules.pop(module_name, None)
    return parse_metric(stdout.getvalue(), "development_cindex")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse run.log, compare it to the kept baseline, and append one row to results.tsv."
    )
    parser.add_argument("--description", required=True, help="Short description of the experiment.")
    parser.add_argument(
        "--status",
        default="auto",
        choices=("auto", "crash"),
        help="Use `auto` for normal runs or `crash` if the experiment failed before producing metrics.",
    )
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH), help="Path to the run log.")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = HERE / log_path
    if not log_path.exists():
        raise FileNotFoundError(f"Run log not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    commit = get_git_commit()
    candidate_path = ""
    baseline_cindex: float | None = None
    delta_cindex: float | None = None

    try:
        if args.status == "crash":
            development_cindex = 0.0
            memory_gb = 0.0
            status = "crash"
        else:
            development_cindex = parse_metric(text, "development_cindex")
            peak_vram_mb = parse_metric(text, "peak_vram_mb")
            memory_gb = peak_vram_mb / 1024.0
            baseline_cindex = evaluate_snapshot_development_cindex(KEPT_PATH)
            delta_cindex = development_cindex - baseline_cindex
            if delta_cindex >= SUPERIORITY_THRESHOLD:
                candidate_path = snapshot_candidate(commit, development_cindex)
                status = "saved"
            else:
                status = "discard"

        append_result(
            commit=commit,
            development_cindex=development_cindex,
            baseline_cindex=baseline_cindex,
            delta_cindex=delta_cindex,
            memory_gb=memory_gb,
            status=status,
            candidate_path=candidate_path,
            description=args.description,
        )
        print(
            f"Logged {status} result: commit={commit} "
            f"development_cindex={development_cindex:.6f} "
            f"baseline_cindex={'n/a' if baseline_cindex is None else f'{baseline_cindex:.6f}'} "
            f"delta_cindex={'n/a' if delta_cindex is None else f'{delta_cindex:.6f}'} "
            f"memory_gb={memory_gb:.3f}"
        )
        if candidate_path:
            print(f"Saved candidate snapshot: {candidate_path}")
    finally:
        restore()


if __name__ == "__main__":
    main()
