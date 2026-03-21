"""Orchestrate PA2 autoresearch until a handoff stop rule fires.

Edits only train.py for experiments (same contract as the manual loop).
Stop when: 10h elapsed, 10000 runs, or 30 consecutive discards.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
TRAIN = HERE / "train.py"
KEPT = HERE / "last_kept_train.py"
RUNLOG = HERE / "run.log"
JOURNAL = HERE / "research_journal.md"
RESULTS = HERE / "results.tsv"
STOP_REPORT = HERE / "loop_stop_report.txt"

MAX_SECONDS = 10 * 3600
MAX_RUNS = 10_000
MAX_CONSECUTIVE_DISCARDS = 30


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(HERE), **kwargs)


def parse_metric(text: str, name: str) -> float:
    pat = rf"^{re.escape(name)}:\s+([0-9.+-]+)$"
    m = re.findall(pat, text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"missing {name} in log")
    return float(m[-1])


def parse_best_step(text: str) -> int:
    return int(parse_metric(text, "best_step"))


def best_kept_cindex() -> float:
    if not RESULTS.exists():
        return 0.0
    best = 0.0
    for line in RESULTS.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            best = max(best, float(parts[1]))
    return best


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def sub_hp(text: str, name: str, value_repr: str) -> str:
    return re.sub(rf"^{name} = .+$", f"{name} = {value_repr}", text, flags=re.MULTILINE)


def parse_float_line(text: str, name: str) -> float:
    m = re.search(rf"^{name} = ([0-9.e+-]+)$", text, re.MULTILINE)
    if not m:
        raise ValueError(f"cannot parse {name}")
    return float(m.group(1))


def parse_int_line(text: str, name: str) -> int:
    m = re.search(rf"^{name} = (\d+)$", text, re.MULTILINE)
    if not m:
        raise ValueError(f"cannot parse {name}")
    return int(m.group(1))


def fingerprint_from_text(t: str) -> tuple[float, float, float, int, tuple[int, int], float]:
    hid_m = re.search(r"^HIDDEN_SIZES = \((\d+), (\d+)\)$", t, re.MULTILINE)
    hidden = (int(hid_m.group(1)), int(hid_m.group(2))) if hid_m else (32, 16)
    res_m = re.search(
        r"self\.residual_scale = nn\.Parameter\(torch\.tensor\(([0-9.]+), dtype=torch\.float32\)\)",
        t,
    )
    res_scale = float(res_m.group(1)) if res_m else 0.1
    return (
        parse_float_line(t, "LEARNING_RATE"),
        parse_float_line(t, "WEIGHT_DECAY"),
        parse_float_line(t, "DROPOUT"),
        parse_int_line(t, "EVAL_EVERY"),
        hidden,
        res_scale,
    )


def build_trial_queue(base: str) -> list[tuple[str, str]]:
    """Return (description, patched_full_text) pairs branching from baseline text."""
    lr = parse_float_line(base, "LEARNING_RATE")
    wd = parse_float_line(base, "WEIGHT_DECAY")
    do = parse_float_line(base, "DROPOUT")
    ev = parse_int_line(base, "EVAL_EVERY")

    trials: list[tuple[str, str]] = []

    def add(desc: str, t: str) -> None:
        trials.append((desc, t))

    for nlr in (0.0019, 0.00195, 0.00205, 0.0021, 0.00185, 0.00215):
        if abs(nlr - lr) < 1e-12:
            continue
        t = sub_hp(base, "LEARNING_RATE", repr(nlr))
        add(f"LEARNING_RATE {lr} -> {nlr}", t)

    for nwd in (wd * 0.96, wd * 1.04, wd * 0.92, wd * 1.08):
        nwd_r = round(nwd, 8)
        if abs(nwd_r - wd) < 1e-15:
            continue
        t = sub_hp(base, "WEIGHT_DECAY", repr(nwd_r))
        add(f"WEIGHT_DECAY {wd} -> {nwd_r}", t)

    for ndo in (do - 0.005, do + 0.005, do - 0.01, do + 0.01):
        if ndo < 0 or ndo > 0.5:
            continue
        ndo_r = round(ndo, 3)
        if abs(ndo_r - do) < 1e-12:
            continue
        t = sub_hp(base, "DROPOUT", repr(ndo_r))
        add(f"DROPOUT {do} -> {ndo_r}", t)

    for nev in (ev - 5, ev + 5, ev - 10, ev + 10):
        if nev < 25:
            continue
        if nev == ev:
            continue
        t = sub_hp(base, "EVAL_EVERY", str(int(nev)))
        add(f"EVAL_EVERY {ev} -> {int(nev)}", t)

    # Hidden width (single conceptual knob): one step wider then narrower residual MLP.
    if "HIDDEN_SIZES = (32, 16)" in base:
        t = base.replace("HIDDEN_SIZES = (32, 16)", "HIDDEN_SIZES = (40, 20)")
        add("HIDDEN_SIZES (32,16) -> (40,20)", t)
        t = base.replace("HIDDEN_SIZES = (32, 16)", "HIDDEN_SIZES = (24, 12)")
        add("HIDDEN_SIZES (32,16) -> (24,12)", t)

    # residual_scale initial value
    if "self.residual_scale = nn.Parameter(torch.tensor(0.1" in base:
        t = base.replace(
            "self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))",
            "self.residual_scale = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))",
        )
        add("residual_scale init 0.1 -> 0.12", t)
        t = base.replace(
            "self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))",
            "self.residual_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))",
        )
        add("residual_scale init 0.1 -> 0.08", t)

    # Deduplicate by hyperparameter fingerprint
    seen: set[tuple[float, float, float, int]] = set()
    out: list[tuple[str, str]] = []
    for desc, t in trials:
        try:
            fp = fingerprint_from_text(t)
        except ValueError:
            continue
        if fp in seen:
            continue
        seen.add(fp)
        out.append((desc, t))
    return out


def next_run_number() -> int:
    if not JOURNAL.exists():
        return 1
    return len(re.findall(r"^## Run \d+", JOURNAL.read_text(encoding="utf-8"), flags=re.MULTILINE)) + 1


def append_journal(
    run_n: int,
    hypothesis: str,
    change: str,
    result: str,
    decision: str,
    learning: str,
    next_move: str,
) -> None:
    block = f"""
## Run {run_n}
- Hypothesis: {hypothesis}
- Change: {change}
- Result: {result}
- Decision: {decision}
- Learning: {learning}
- Next: {next_move}
"""
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(block)


def main() -> int:
    if not PY.is_file():
        print("Missing venv python:", PY, file=sys.stderr)
        return 1
    if not KEPT.is_file():
        print("Missing last_kept_train.py; run manage_kept.py save first.", file=sys.stderr)
        return 1

    t0 = time.time()
    consecutive_discards = 0
    total_runs = 0
    best = best_kept_cindex()
    seen_fp: set[tuple] = set()

    while True:
        elapsed = time.time() - t0
        if elapsed >= MAX_SECONDS:
            msg = f"stop: time limit {MAX_SECONDS}s elapsed (ran {total_runs} experiments)."
            STOP_REPORT.write_text(msg + "\n", encoding="utf-8")
            print(msg)
            return 0
        if total_runs >= MAX_RUNS:
            msg = f"stop: max runs {MAX_RUNS} reached."
            STOP_REPORT.write_text(msg + "\n", encoding="utf-8")
            print(msg)
            return 0
        if consecutive_discards >= MAX_CONSECUTIVE_DISCARDS:
            msg = (
                f"stop: {MAX_CONSECUTIVE_DISCARDS} consecutive discards. "
                f"Best kept val_cindex={best:.6f}. Local hyperparameter moves exhausted or noisy."
            )
            STOP_REPORT.write_text(msg + "\n", encoding="utf-8")
            print(msg)
            return 0

        run([str(PY), str(HERE / "summarize_results.py")], check=False)
        base = read_text(KEPT)
        full_queue = build_trial_queue(base)
        unseen = [(d, p) for d, p in full_queue if fingerprint_from_text(p) not in seen_fp]
        if not unseen:
            seen_fp.clear()
            unseen = list(full_queue)
        if not unseen:
            msg = "stop: empty trial queue from current kept baseline."
            STOP_REPORT.write_text(msg + "\n", encoding="utf-8")
            print(msg)
            return 0

        desc, patched = unseen[0]
        seen_fp.add(fingerprint_from_text(patched))

        TRAIN.write_text(patched, encoding="utf-8")
        with RUNLOG.open("w", encoding="utf-8") as logf:
            rc = run([str(PY), str(TRAIN)], stdout=logf, stderr=subprocess.STDOUT)
        total_runs += 1

        log_text = read_text(RUNLOG)
        if rc.returncode != 0:
            run(
                [
                    str(PY),
                    str(HERE / "log_result.py"),
                    "--description",
                    f"crash: {desc}",
                    "--status",
                    "crash",
                ],
                check=True,
            )
            run([str(PY), str(HERE / "manage_kept.py"), "restore"], check=True)
            consecutive_discards += 1
            append_journal(
                next_run_number(),
                "Training completes without error.",
                desc,
                "crash / non-zero exit",
                "discard (restore)",
                "Investigate run.log",
                "Continue queue",
            )
            continue

        try:
            val = parse_metric(log_text, "val_cindex")
            bstep = parse_best_step(log_text)
        except ValueError as e:
            run(
                [
                    str(PY),
                    str(HERE / "log_result.py"),
                    "--description",
                    f"crash parse: {desc} ({e})",
                    "--status",
                    "crash",
                ],
                check=True,
            )
            run([str(PY), str(HERE / "manage_kept.py"), "restore"], check=True)
            consecutive_discards += 1
            continue

        detail = f"{desc}; val_cindex {val:.6f} best_step {bstep}"
        if val > best:
            run(
                [
                    str(PY),
                    str(HERE / "log_result.py"),
                    "--description",
                    detail,
                    "--status",
                    "keep",
                ],
                check=True,
            )
            run([str(PY), str(HERE / "manage_kept.py"), "save"], check=True)
            best = val
            consecutive_discards = 0
            seen_fp.clear()
            append_journal(
                next_run_number(),
                "Single local hyperparameter move may improve val_cindex.",
                desc,
                f"val_cindex **{val:.6f}** best_step **{bstep}**",
                "**keep**",
                "Improved vs prior best kept snapshot.",
                "Continue local search from new baseline.",
            )
        else:
            run(
                [
                    str(PY),
                    str(HERE / "log_result.py"),
                    "--description",
                    detail,
                    "--status",
                    "discard",
                ],
                check=True,
            )
            run([str(PY), str(HERE / "manage_kept.py"), "restore"], check=True)
            consecutive_discards += 1
            append_journal(
                next_run_number(),
                "Single local move unlikely to beat current best.",
                desc,
                f"val_cindex **{val:.6f}** (best kept **{best:.6f}**)",
                "**discard** (restored)",
                "No improvement vs best kept.",
                "Try next queued neighbor.",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
