"""Resumable PA2 autoresearch supervisor.

This loop edits only ``train.py`` for candidate-model changes and stores its
own recovery state in sidecar files inside ``autoresearch``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
TRAIN = HERE / "train.py"
KEPT = HERE / "last_kept_train.py"
RUNLOG = HERE / "run.log"
JOURNAL = HERE / "research_journal.md"
RESULTS = HERE / "results.tsv"
STATE = HERE / "loop_state.json"
PENDING = HERE / "pending_run.json"
STATUS = HERE / "loop_status.txt"
STOP_REPORT = HERE / "loop_stop_report.txt"

MAX_SECONDS = 10 * 3600
MAX_RUNS = 10_000
MAX_CONSECUTIVE_DISCARDS = 30
MAX_EXPLORATION_BURST = 6
MAX_LOCAL_DISCARDS_BEFORE_EXPLORATION = 2
MEANINGFUL_IMPROVEMENT = 0.0005
SIMILARITY_TOLERANCE = 0.00005
RUN_POLL_SECONDS = 5.0


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(cmd: list[str], *, check: bool = False, stdout=None, stderr=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(HERE),
        check=check,
        stdout=stdout,
        stderr=stderr,
        text=stdout is subprocess.PIPE or stderr is subprocess.PIPE,
    )


def pid_is_running(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, SystemError, ValueError):
        return False
    return True


def parse_metric(text: str, name: str) -> float:
    matches = re.findall(rf"^{re.escape(name)}:\s+([0-9.+-]+)$", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"missing {name} in log")
    return float(matches[-1])


def parse_summary(text: str) -> dict[str, Any]:
    return {
        "val_cindex": parse_metric(text, "val_cindex"),
        "training_seconds": parse_metric(text, "training_seconds"),
        "total_seconds": parse_metric(text, "total_seconds"),
        "peak_vram_mb": parse_metric(text, "peak_vram_mb"),
        "num_steps": int(parse_metric(text, "num_steps")),
        "num_params": int(parse_metric(text, "num_params")),
        "best_step": int(parse_metric(text, "best_step")),
        "stop_reason": re.findall(r"^stop_reason:\s+(.+)$", text, flags=re.MULTILINE)[-1],
    }


def has_completed_summary(text: str) -> bool:
    required = ("val_cindex", "training_seconds", "total_seconds", "peak_vram_mb", "best_step", "artifact_path")
    return all(re.search(rf"^{name}:\s+.+$", text, flags=re.MULTILINE) for name in required)


def load_results_rows() -> list[dict[str, str]]:
    if not RESULTS.exists():
        return []
    with RESULTS.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def is_local_description(description: str) -> bool:
    normalized = description.strip()
    if normalized.startswith("crash: "):
        normalized = normalized[7:]
    prefixes = ("LEARNING_RATE ", "WEIGHT_DECAY ", "DROPOUT ", "EVAL_EVERY ")
    return normalized.startswith(prefixes)


def compute_history(rows: list[dict[str, str]]) -> dict[str, Any]:
    best_keep = 0.0
    consecutive_discards = 0
    plateau_streak = 0
    completed_runs = 0
    for row in rows:
        status = row.get("status", "")
        val = float(row.get("val_cindex", "0") or 0.0)
        if status in {"keep", "discard"}:
            completed_runs += 1
            if val >= best_keep + MEANINGFUL_IMPROVEMENT:
                plateau_streak = 0
            else:
                plateau_streak += 1
        if status == "keep":
            consecutive_discards = 0
            if val > best_keep:
                best_keep = val
        elif status == "discard":
            consecutive_discards += 1
    return {
        "best_keep": best_keep,
        "consecutive_discards": consecutive_discards,
        "plateau_streak": plateau_streak,
        "completed_runs": completed_runs,
        "total_rows": len(rows),
    }


def compute_local_discards_since_last_keep(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in reversed(rows):
        status = row.get("status", "")
        if status == "keep":
            break
        if status in {"discard", "crash"} and is_local_description(row.get("description", "")):
            count += 1
    return count


def next_run_number() -> int:
    if not JOURNAL.exists():
        return 1
    return len(re.findall(r"^## Run \d+", JOURNAL.read_text(encoding="utf-8"), flags=re.MULTILINE)) + 1


def append_journal(
    *,
    hypothesis: str,
    change: str,
    result: str,
    decision: str,
    learning: str,
    next_move: str,
) -> None:
    block = (
        f"\n## Run {next_run_number()}\n"
        f"- Hypothesis: {hypothesis}\n"
        f"- Change: {change}\n"
        f"- Result: {result}\n"
        f"- Decision: {decision}\n"
        f"- Learning: {learning}\n"
        f"- Next: {next_move}\n"
    )
    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write(block)


def update_status_file(state: dict[str, Any], history: dict[str, Any], note: str) -> None:
    pending = load_json(PENDING, {})
    lines = [
        f"updated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"supervisor_pid: {state.get('supervisor_pid', 0)}",
        f"started_at_epoch: {state.get('started_at_epoch', 0.0)}",
        f"phase: {state.get('phase', 'local')}",
        f"best_kept_val_cindex: {history['best_keep']:.6f}",
        f"plateau_streak: {history['plateau_streak']}",
        f"consecutive_discards: {history['consecutive_discards']}",
        f"completed_runs: {history['completed_runs']}",
        f"pending_description: {pending.get('description', 'none')}",
        f"pending_family: {pending.get('family', 'none')}",
        f"pending_pid: {pending.get('child_pid', 0)}",
        f"note: {note}",
    ]
    atomic_write_text(STATUS, "\n".join(lines) + "\n")


def record_stop_report(message: str) -> None:
    atomic_write_text(STOP_REPORT, message.rstrip() + "\n")


def file_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def replace_assignment(text: str, name: str, value_repr: str) -> str:
    updated = re.sub(rf"^{name} = .+$", f"{name} = {value_repr}", text, flags=re.MULTILINE)
    if updated == text:
        raise ValueError(f"could not replace assignment for {name}")
    return updated


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise ValueError(f"missing block for {label}")
    return text.replace(old, new, 1)


def add_retry_tag(text: str, tag: str) -> str:
    return text.rstrip() + f"\n\n# supervisor_retry_tag: {tag}\n"


def local_experiment(
    base_text: str,
    *,
    description: str,
    hypothesis: str,
    change: str,
    patch_fn,
) -> dict[str, Any]:
    patched = patch_fn(base_text)
    return {
        "kind": "local",
        "family": "local_tuning",
        "description": description,
        "hypothesis": hypothesis,
        "change": change,
        "train_text": patched,
        "train_hash": file_hash(patched),
        "prefer_simpler_on_tie": False,
    }


def exploration_experiment(
    base_text: str,
    *,
    family: str,
    description: str,
    hypothesis: str,
    change: str,
    patch_fn,
    prefer_simpler_on_tie: bool = False,
    retry_tag: str | None = None,
) -> dict[str, Any]:
    patched = patch_fn(base_text)
    if retry_tag:
        patched = add_retry_tag(patched, retry_tag)
    return {
        "kind": "explore",
        "family": family,
        "description": description,
        "hypothesis": hypothesis,
        "change": change,
        "train_text": patched,
        "train_hash": file_hash(patched),
        "prefer_simpler_on_tie": prefer_simpler_on_tie,
    }


def build_local_queue(base_text: str) -> list[dict[str, Any]]:
    lr = re.search(r"^LEARNING_RATE = ([0-9.e+-]+)$", base_text, re.MULTILINE)
    wd = re.search(r"^WEIGHT_DECAY = ([0-9.e+-]+)$", base_text, re.MULTILINE)
    do = re.search(r"^DROPOUT = ([0-9.e+-]+)$", base_text, re.MULTILINE)
    ev = re.search(r"^EVAL_EVERY = (\d+)$", base_text, re.MULTILINE)
    if not all((lr, wd, do, ev)):
        raise ValueError("could not parse local hyperparameters from kept train.py")
    lr_v = float(lr.group(1))
    wd_v = float(wd.group(1))
    do_v = float(do.group(1))
    ev_v = int(ev.group(1))

    queue = [
        local_experiment(
            base_text,
            description=f"LEARNING_RATE {lr_v} -> {lr_v - 0.000025:.6f}",
            hypothesis="A slightly smaller step size may refine the same early optimum without leaving the current local basin.",
            change=f"`LEARNING_RATE` `{lr_v}` -> `{lr_v - 0.000025:.6f}`.",
            patch_fn=lambda t: replace_assignment(t, "LEARNING_RATE", repr(round(lr_v - 0.000025, 8))),
        ),
        local_experiment(
            base_text,
            description=f"LEARNING_RATE {lr_v} -> {lr_v + 0.000025:.6f}",
            hypothesis="A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.",
            change=f"`LEARNING_RATE` `{lr_v}` -> `{lr_v + 0.000025:.6f}`.",
            patch_fn=lambda t: replace_assignment(t, "LEARNING_RATE", repr(round(lr_v + 0.000025, 8))),
        ),
        local_experiment(
            base_text,
            description=f"WEIGHT_DECAY {wd_v} -> {wd_v - 0.000005:.8f}",
            hypothesis="A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.",
            change=f"`WEIGHT_DECAY` `{wd_v}` -> `{wd_v - 0.000005:.8f}`.",
            patch_fn=lambda t: replace_assignment(t, "WEIGHT_DECAY", repr(round(wd_v - 0.000005, 8))),
        ),
        local_experiment(
            base_text,
            description=f"WEIGHT_DECAY {wd_v} -> {wd_v + 0.000005:.8f}",
            hypothesis="A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.",
            change=f"`WEIGHT_DECAY` `{wd_v}` -> `{wd_v + 0.000005:.8f}`.",
            patch_fn=lambda t: replace_assignment(t, "WEIGHT_DECAY", repr(round(wd_v + 0.000005, 8))),
        ),
        local_experiment(
            base_text,
            description=f"DROPOUT {do_v} -> {do_v - 0.005:.3f}",
            hypothesis="A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.",
            change=f"`DROPOUT` `{do_v}` -> `{do_v - 0.005:.3f}`.",
            patch_fn=lambda t: replace_assignment(t, "DROPOUT", repr(round(do_v - 0.005, 3))),
        ),
        local_experiment(
            base_text,
            description=f"EVAL_EVERY {ev_v} -> {ev_v + 5}",
            hypothesis="A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.",
            change=f"`EVAL_EVERY` `{ev_v}` -> `{ev_v + 5}`.",
            patch_fn=lambda t: replace_assignment(t, "EVAL_EVERY", str(ev_v + 5)),
        ),
    ]

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_hash = file_hash(base_text)
    for experiment in queue:
        train_hash = experiment["train_hash"]
        if train_hash == base_hash or train_hash in seen:
            continue
        seen.add(train_hash)
        deduped.append(experiment)
    return deduped


def make_weighted_cox_text(base_text: str) -> str:
    old = """def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_scores = risk_scores[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_scores, dim=0)
    event_count = ordered_events.sum().clamp_min(1.0)
    losses = -(ordered_scores - log_risk) * ordered_events
    return losses.sum() / event_count
"""
    new = """def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_scores = risk_scores[order]
    ordered_times = times[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_scores, dim=0)
    base_losses = -(ordered_scores - log_risk) * ordered_events
    safe_times = ordered_times.clamp_min(1.0)
    event_weights = torch.where(
        ordered_events > 0,
        (safe_times.mean() / safe_times).pow(0.35),
        torch.ones_like(safe_times),
    )
    weighted_events = ordered_events * event_weights
    normalizer = weighted_events.sum().clamp_min(1.0)
    return (base_losses * event_weights).sum() / normalizer
"""
    return replace_once(base_text, old, new, "weighted_cox_loss")


def make_pruned_encoder_text(base_text: str) -> str:
    text = replace_once(base_text, "        self.output_dim = 16", "        self.output_dim = 10", "encoder_output_dim")
    old = """        return torch.stack(
            (
                amp,
                cep,
                sgp,
                log_crp,
                lymph,
                mcv,
                rdw,
                alk,
                wbc,
                pheno_no_age_xb,
                amp * rdw,
                sgp * log_crp,
                wbc * log_crp,
                lymph * rdw,
                alk * rdw,
                cep * log_crp,
            ),
            dim=1,
        )
"""
    new = """        return torch.stack(
            (
                amp,
                cep,
                sgp,
                log_crp,
                lymph,
                mcv,
                rdw,
                alk,
                wbc,
                pheno_no_age_xb,
            ),
            dim=1,
        )
"""
    return replace_once(text, old, new, "pruned_encoder")


def make_linear_residual_text(base_text: str) -> str:
    old = """        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.residual_head = nn.Sequential(*layers)
"""
    new = """        self.residual_head = nn.Linear(input_dim, 1)
"""
    return replace_once(base_text, old, new, "linear_residual_head")


def make_smaller_residual_mlp_text(base_text: str) -> str:
    return replace_once(
        base_text,
        "HIDDEN_SIZES = (32, 16)",
        "HIDDEN_SIZES = (24, 12)",
        "smaller_residual_mlp",
    )


def make_lower_residual_scale_text(base_text: str) -> str:
    return replace_once(
        base_text,
        '        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))',
        '        self.residual_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))',
        "lower_residual_scale",
    )


def make_higher_residual_scale_text(base_text: str) -> str:
    return replace_once(
        base_text,
        '        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))',
        '        self.residual_scale = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))',
        "higher_residual_scale",
    )


def make_single_hidden_residual_text(base_text: str) -> str:
    return replace_once(
        base_text,
        "HIDDEN_SIZES = (32, 16)",
        "HIDDEN_SIZES = (16,)",
        "single_hidden_residual",
    )


def make_silu_residual_text(base_text: str) -> str:
    return replace_once(
        base_text,
        "            layers.append(nn.GELU())",
        "            layers.append(nn.SiLU())",
        "silu_residual_activation",
    )


def make_longer_patience_text(base_text: str) -> str:
    return replace_assignment(base_text, "EARLY_STOP_PATIENCE_EVALS", "5")


def make_lower_beta2_text(base_text: str) -> str:
    return replace_once(
        base_text,
        '    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)',
        '    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))',
        "lower_beta2",
    )


def make_expanded_encoder_retry_text(base_text: str) -> str:
    text = replace_once(base_text, "        self.output_dim = 16", "        self.output_dim = 21", "expanded_encoder_output_dim")
    old = """        return torch.stack(
            (
                amp,
                cep,
                sgp,
                log_crp,
                lymph,
                mcv,
                rdw,
                alk,
                wbc,
                pheno_no_age_xb,
                amp * rdw,
                sgp * log_crp,
                wbc * log_crp,
                lymph * rdw,
                alk * rdw,
                cep * log_crp,
            ),
            dim=1,
        )
"""
    new = """        return torch.stack(
            (
                amp,
                cep,
                sgp,
                log_crp,
                lymph,
                mcv,
                rdw,
                alk,
                wbc,
                pheno_no_age_xb,
                amp * rdw,
                sgp * log_crp,
                wbc * log_crp,
                lymph * rdw,
                alk * rdw,
                cep * log_crp,
                amp * log_crp,
                sgp * rdw,
                alk * log_crp,
                sgp / amp.clamp_min(1e-6),
                torch.log1p(alk.clamp_min(0.0)),
            ),
            dim=1,
        )
"""
    return replace_once(text, old, new, "expanded_encoder_retry")


def make_linear_skip_correction_text(base_text: str) -> str:
    text = replace_once(
        base_text,
        '        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))',
        '        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))\n        self.linear_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))\n        self.linear_skip = nn.Linear(input_dim, 1)',
        "linear_skip_parameters",
    )
    return replace_once(
        text,
        "        return self.base_weight * base_score + self.residual_scale * residual",
        "        linear_skip = self.linear_skip(standardized).squeeze(-1)\n        return self.base_weight * base_score + self.residual_scale * residual + self.linear_scale * linear_skip",
        "linear_skip_forward",
    )


def make_fixed_anchor_weight_text(base_text: str) -> str:
    text = replace_once(
        base_text,
        '        self.base_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))',
        '        self.register_buffer("base_weight", torch.tensor(1.0, dtype=torch.float32))',
        "fixed_anchor_weight",
    )
    return replace_once(
        text,
        "        return self.base_weight * base_score + self.residual_scale * residual",
        "        return base_score + self.residual_scale * residual",
        "fixed_anchor_forward",
    )


def make_hybrid_pairwise_loss_text(base_text: str) -> str:
    text = replace_once(
        base_text,
        """def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_scores = risk_scores[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_scores, dim=0)
    event_count = ordered_events.sum().clamp_min(1.0)
    losses = -(ordered_scores - log_risk) * ordered_events
    return losses.sum() / event_count
""",
        """def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_scores = risk_scores[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_scores, dim=0)
    event_count = ordered_events.sum().clamp_min(1.0)
    losses = -(ordered_scores - log_risk) * ordered_events
    return losses.sum() / event_count


def pairwise_ranking_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    event_idx = torch.nonzero(events > 0, as_tuple=False).squeeze(1)
    if event_idx.numel() == 0:
        return risk_scores.new_tensor(0.0)
    if event_idx.numel() > 256:
        perm = torch.randperm(event_idx.numel(), device=event_idx.device)
        event_idx = event_idx.index_select(0, perm[:256])
    pair_losses: list[torch.Tensor] = []
    for idx in event_idx:
        later_idx = torch.nonzero(times > times[idx], as_tuple=False).squeeze(1)
        if later_idx.numel() == 0:
            continue
        if later_idx.numel() > 256:
            perm = torch.randperm(later_idx.numel(), device=later_idx.device)
            later_idx = later_idx.index_select(0, perm[:256])
        score_diff = risk_scores[idx] - risk_scores.index_select(0, later_idx)
        pair_losses.append(torch.nn.functional.softplus(-score_diff).mean())
    if not pair_losses:
        return risk_scores.new_tensor(0.0)
    return torch.stack(pair_losses).mean()
""",
        "hybrid_pairwise_loss_block",
    )
    return replace_once(
        text,
        "        loss = cox_partial_loss(risk_scores, train_times, train_events)",
        "        loss = cox_partial_loss(risk_scores, train_times, train_events) + 0.2 * pairwise_ranking_loss(risk_scores, train_times, train_events)",
        "hybrid_pairwise_loss_train_step",
    )


def make_feature_noise_text(base_text: str) -> str:
    return replace_once(
        base_text,
        "        standardized = (encoded - self.feature_mean) / self.feature_std",
        "        standardized = (encoded - self.feature_mean) / self.feature_std\n        if self.training:\n            standardized = standardized + 0.03 * torch.randn_like(standardized)",
        "feature_noise_regularization",
    )


def make_feature_gate_text(base_text: str) -> str:
    text = replace_once(
        base_text,
        '        self.linear_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))\n        self.linear_skip = nn.Linear(input_dim, 1)',
        '        self.linear_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))\n        self.linear_skip = nn.Linear(input_dim, 1)\n        self.feature_gate = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))',
        "feature_gate_parameters",
    )
    text = replace_once(
        text,
        "        standardized = (encoded - self.feature_mean) / self.feature_std",
        "        standardized = (encoded - self.feature_mean) / self.feature_std\n        gated = standardized * torch.sigmoid(self.feature_gate)",
        "feature_gate_forward_standardized",
    )
    text = replace_once(
        text,
        "        residual = self.residual_head(standardized).squeeze(-1)\n        linear_skip = self.linear_skip(standardized).squeeze(-1)",
        "        residual = self.residual_head(gated).squeeze(-1)\n        linear_skip = self.linear_skip(gated).squeeze(-1)",
        "feature_gate_forward_paths",
    )
    return add_retry_tag(text, "feature_gate_current")


def make_quadratic_skip_text(base_text: str) -> str:
    text = replace_once(
        base_text,
        '        self.linear_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))\n        self.linear_skip = nn.Linear(input_dim, 1)',
        '        self.linear_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))\n        self.linear_skip = nn.Linear(input_dim, 1)\n        self.quadratic_scale = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))\n        self.quadratic_skip = nn.Linear(input_dim, 1, bias=False)',
        "quadratic_skip_parameters",
    )
    return replace_once(
        text,
        "        linear_skip = self.linear_skip(standardized).squeeze(-1)\n        return self.base_weight * base_score + self.residual_scale * residual + self.linear_scale * linear_skip",
        "        linear_skip = self.linear_skip(standardized).squeeze(-1)\n        quadratic_skip = self.quadratic_skip(standardized.square()).squeeze(-1)\n        return self.base_weight * base_score + self.residual_scale * residual + self.linear_scale * linear_skip + self.quadratic_scale * quadratic_skip",
        "quadratic_skip_forward",
    )


def make_minibatch_cosine_retry_text(base_text: str) -> str:
    text = replace_once(
        base_text,
        '    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n',
        '    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n    batch_size = min(1024, len(train_rows))\n    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64, eta_min=LEARNING_RATE * 0.35)\n',
        "minibatch_cosine_setup",
    )
    return replace_once(
        text,
        """        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk_scores = model(train_x)
        loss = cox_partial_loss(risk_scores, train_times, train_events)
        loss.backward()
        optimizer.step()
""",
        """        model.train()
        batch_perm = torch.randperm(len(train_rows), device=device)
        batch_losses: list[torch.Tensor] = []
        for start in range(0, len(train_rows), batch_size):
            batch_idx = batch_perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            risk_scores = model(train_x.index_select(0, batch_idx))
            batch_times = train_times.index_select(0, batch_idx)
            batch_events = train_events.index_select(0, batch_idx)
            loss = cox_partial_loss(risk_scores, batch_times, batch_events)
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.detach())
        loss = torch.stack(batch_losses).mean()
""",
        "minibatch_cosine_train_step",
    )


def build_exploration_queue(base_text: str, history_text: str) -> list[dict[str, Any]]:
    lower = history_text.lower()
    queue: list[dict[str, Any]] = []
    weighted_cox_crashed = "crash: time-weighted cox loss to emphasize earlier aging deaths" in lower
    lean_encoder_crashed = "crash: lean encoder with raw biomarkers plus pheno_no_age_xb only" in lower
    linear_head_crashed = "crash: single linear residual head instead of hidden mlp" in lower
    hybrid_pairwise_crashed = "crash: hybrid cox + pairwise ranking loss on single-hidden baseline" in lower
    weighted_cox_retried = "retry time-weighted cox loss after supervisor fix" in lower
    lean_encoder_retried = "retry lean encoder after supervisor fix" in lower
    linear_head_retried = "retry single linear residual head after supervisor fix" in lower
    hybrid_pairwise_retried = "retry hybrid cox + sampled pairwise ranking loss after memory fix" in lower

    if weighted_cox_crashed and not weighted_cox_retried:
        queue.append(
            exploration_experiment(
                base_text,
                family="loss_design_retry",
                description="Retry time-weighted Cox loss after supervisor fix",
                hypothesis="The previous time-weighted Cox attempt failed before producing a completed summary block, so rerunning it under the fixed synchronous supervisor will reveal whether the loss family is genuinely promising or truly weak.",
                change="Retry the time-weighted Cox objective after fixing the supervisor persistence bug that invalidated the earlier attempt.",
                patch_fn=make_weighted_cox_text,
                retry_tag="retry_weighted_cox_after_supervisor_fix",
            )
        )
    elif "weighted cox" not in lower and "time-weighted cox" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="loss_design",
                description="Time-weighted Cox loss to emphasize earlier aging deaths",
                hypothesis="Local tuning has plateaued, so reweighting the Cox objective toward earlier events may better align the model with the ranking signal the validation split rewards.",
                change="Replace the uniform Cox partial likelihood with a time-weighted Cox loss that gives somewhat more weight to earlier observed events.",
                patch_fn=make_weighted_cox_text,
            )
        )
    if lean_encoder_crashed and not lean_encoder_retried:
        queue.append(
            exploration_experiment(
                base_text,
                family="feature_representation_retry",
                description="Retry lean encoder after supervisor fix",
                hypothesis="The lean encoder never produced a valid run because the earlier orchestration failure killed it before completion, so it deserves one clean retry before the feature-representation family is abandoned.",
                change="Retry the lean encoder variant after fixing the supervisor persistence bug that invalidated the earlier attempt.",
                patch_fn=make_pruned_encoder_text,
                prefer_simpler_on_tie=True,
                retry_tag="retry_lean_encoder_after_supervisor_fix",
            )
        )
    elif "lean encoder" not in lower and "pruned encoder" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="feature_representation",
                description="Lean encoder with raw biomarkers plus pheno_no_age_xb only",
                hypothesis="The failed feature-expansion run suggests extra interactions may overfit, so a leaner anchor-aligned encoder may generalize better than the current richer representation.",
                change="Prune the encoder to the nine transformed biomarkers plus `pheno_no_age_xb`, removing all hand-crafted interaction features.",
                patch_fn=make_pruned_encoder_text,
                prefer_simpler_on_tie=True,
            )
        )
    if linear_head_crashed and not linear_head_retried:
        queue.append(
            exploration_experiment(
                base_text,
                family="architecture_retry",
                description="Retry single linear residual head after supervisor fix",
                hypothesis="The single-linear residual head was never measured cleanly because the first attempt died before logging, so rerunning it under the repaired supervisor is a materially different test of that simpler architecture.",
                change="Retry the single linear residual head after fixing the supervisor persistence bug that invalidated the earlier attempt.",
                patch_fn=make_linear_residual_text,
                prefer_simpler_on_tie=True,
                retry_tag="retry_linear_head_after_supervisor_fix",
            )
        )
    elif "linear residual head" not in lower and "single linear residual head" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="architecture",
                description="Single linear residual head instead of hidden MLP",
                hypothesis="Broader capacity increases have not helped, so collapsing the residual path to a single linear layer may reduce overfitting while preserving the pheno-no-age anchor.",
                change="Replace the hidden residual MLP with a single linear correction layer on standardized encoded features.",
                patch_fn=make_linear_residual_text,
                prefer_simpler_on_tie=True,
            )
        )
    if "smaller residual mlp 24-12" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="architecture_width",
                description="Smaller residual MLP 24-12",
                hypothesis="The full 32-16 residual head may be slightly too flexible, so shrinking it without collapsing to a purely linear correction could preserve the useful nonlinear adjustment while reducing overfitting.",
                change="Reduce `HIDDEN_SIZES` from `(32, 16)` to `(24, 12)` while keeping the current training setup unchanged.",
                patch_fn=make_smaller_residual_mlp_text,
                prefer_simpler_on_tie=True,
            )
        )
    if "lower residual_scale init 0.1 -> 0.08" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="anchor_strength",
                description="Lower residual_scale init 0.1 -> 0.08",
                hypothesis="The model may generalize better if it starts slightly closer to the pheno-no-age anchor and learns a smaller correction path, rather than giving the residual head as much influence from the start.",
                change="Reduce the initial `residual_scale` from `0.1` to `0.08` while keeping the kept architecture and optimizer unchanged.",
                patch_fn=make_lower_residual_scale_text,
                prefer_simpler_on_tie=True,
            )
        )
    if "higher residual_scale init 0.1 -> 0.12" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="anchor_strength_boost",
                description="Higher residual_scale init 0.1 -> 0.12",
                hypothesis="Reducing the residual contribution hurt badly, which suggests the kept model may actually be under-correcting the pheno-no-age anchor and could benefit from a slightly stronger residual path.",
                change="Increase the initial `residual_scale` from `0.1` to `0.12` while keeping the kept architecture and optimizer unchanged.",
                patch_fn=make_higher_residual_scale_text,
            )
        )
    if "single hidden residual mlp 16" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="architecture_depth",
                description="Single hidden residual MLP 16",
                hypothesis="The two-layer 32-16 head may be using the wrong shape of capacity, so collapsing it to one modest hidden layer could preserve useful nonlinear correction while removing unnecessary depth.",
                change="Change `HIDDEN_SIZES` from `(32, 16)` to `(16,)`, keeping the residual MLP, optimizer, and encoder otherwise unchanged.",
                patch_fn=make_single_hidden_residual_text,
                prefer_simpler_on_tie=True,
            )
        )
    if "residual head gelu -> silu" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="activation_family",
                description="Residual head GELU -> SiLU",
                hypothesis="The search looks optimization-limited near the current basin, so a smoother residual nonlinearity may slightly improve ranking without changing model width or the anchor pathway.",
                change="Replace the residual head activation from `nn.GELU()` to `nn.SiLU()` while keeping the kept encoder, widths, and optimizer otherwise unchanged.",
                patch_fn=make_silu_residual_text,
            )
        )
    if "early_stop_patience_evals 3 -> 5 on current kept baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="training_dynamics_retry",
                description="EARLY_STOP_PATIENCE_EVALS 3 -> 5 on current kept baseline",
                hypothesis="The earlier patience increase was tested on a weaker baseline before the later LR, weight-decay, and eval-grid gains, so retrying it on the stronger current setup is materially different and may allow a later checkpoint to emerge.",
                change="Increase `EARLY_STOP_PATIENCE_EVALS` from `3` to `5` on the current kept baseline without changing the model form.",
                patch_fn=make_longer_patience_text,
            )
        )
    if "adamw beta2 0.999 -> 0.98" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="optimizer_momentum",
                description="AdamW beta2 0.999 -> 0.98",
                hypothesis="The best local LR move nearly ties but does not clear the keep bar, which suggests the current basin may need a slightly different optimizer memory rather than another scalar LR nudge.",
                change="Keep the current architecture and scalar hyperparameters, but set AdamW `betas=(0.9, 0.98)` instead of the default `beta2=0.999`.",
                patch_fn=make_lower_beta2_text,
            )
        )
    if "single-hidden residual head gelu -> silu" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="activation_family_retry_current",
                description="Single-hidden residual head GELU -> SiLU",
                hypothesis="The earlier SiLU run was the best overall discard and was tested before the winning switch to a shallower residual head, so re-running SiLU on the stronger current baseline is a materially different retry with real upside.",
                change="Replace the residual activation from `nn.GELU()` to `nn.SiLU()` on top of the kept single-hidden residual MLP baseline.",
                patch_fn=make_silu_residual_text,
            )
        )
    if "expanded encoder retry on single-hidden baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="feature_expansion_retry_current",
                description="Expanded encoder retry on single-hidden baseline",
                hypothesis="The earlier feature-expansion attempt was paired with a worse residual architecture, so retrying a richer interaction set on the stronger single-hidden baseline could recover signal that the deeper head previously overfit.",
                change="Add five extra biomarker interactions to the encoder while keeping the current single-hidden residual head and optimizer unchanged.",
                patch_fn=make_expanded_encoder_retry_text,
            )
        )
    if "linear skip correction on single-hidden baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="linear_skip_current",
                description="Linear skip correction on single-hidden baseline",
                hypothesis="The single-hidden residual head may still be too nonlinear for some stable ranking signal, so adding a small direct linear correction path could capture simple coefficient adjustments that the nonlinear branch misses.",
                change="Add a learned linear skip correction on standardized encoded features alongside the current single-hidden residual head.",
                patch_fn=make_linear_skip_correction_text,
            )
        )
    if "fix base anchor weight at 1.0 on single-hidden baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="anchor_lock_current",
                description="Fix base anchor weight at 1.0 on single-hidden baseline",
                hypothesis="The current model may be overfitting by re-scaling the pheno-no-age anchor itself, so forcing the anchor weight to stay fixed could let the residual branch learn cleaner corrections.",
                change="Replace the learned `base_weight` with a fixed weight of `1.0`, keeping the current single-hidden residual correction path unchanged.",
                patch_fn=make_fixed_anchor_weight_text,
                prefer_simpler_on_tie=True,
            )
        )
    if hybrid_pairwise_crashed and not hybrid_pairwise_retried:
        queue.append(
            exploration_experiment(
                base_text,
                family="loss_design_pairwise_retry_current",
                description="Retry hybrid Cox + sampled pairwise ranking loss after memory fix",
                hypothesis="The previous hybrid Cox plus pairwise attempt likely failed because the naive all-pairs construction was too memory-heavy, so retrying it with sampled comparable pairs is still a meaningfully different loss-design probe with real upside.",
                change="Keep the current single-hidden residual architecture, but optimize a hybrid loss: Cox partial likelihood plus a sampled pairwise ranking surrogate over comparable event-survival pairs.",
                patch_fn=make_hybrid_pairwise_loss_text,
                retry_tag="retry_hybrid_pairwise_after_memory_fix",
            )
        )
    elif "hybrid cox + pairwise ranking loss on single-hidden baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="loss_design_pairwise_current",
                description="Hybrid Cox + pairwise ranking loss on single-hidden baseline",
                hypothesis="The model may now be limited by the mismatch between Cox optimization and the exact ranking target, so adding a smooth pairwise concordance surrogate could improve discrimination more meaningfully than another scalar hyperparameter tweak.",
                change="Keep the current single-hidden residual architecture, but optimize a hybrid loss: Cox partial likelihood plus a small pairwise ranking surrogate over comparable event-survival pairs.",
                patch_fn=make_hybrid_pairwise_loss_text,
            )
        )
    if "encoded feature noise regularization on single-hidden baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="regularization_noise_current",
                description="Encoded feature noise regularization on single-hidden baseline",
                hypothesis="The current winner may still be overfitting stable quirks of the training split, so a small amount of noise on standardized encoded features during training could improve robustness without changing the inference-time formula.",
                change="Add small Gaussian noise to standardized encoded features during training while keeping the current single-hidden residual architecture and optimizer unchanged.",
                patch_fn=make_feature_noise_text,
            )
        )
    if "feature gate on standardized encoder on single-hidden baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="feature_gate_current",
                description="Feature gate on standardized encoder on single-hidden baseline",
                hypothesis="The current model treats every encoded biomarker channel as equally available to the correction paths, so a learned per-feature gate may suppress noisy corrections and focus capacity on the most stable encoded signals.",
                change="Add a learned sigmoid feature gate on standardized encoded features before both the residual MLP and the linear skip path.",
                patch_fn=make_feature_gate_text,
            )
        )
    if "quadratic skip correction on single-hidden baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="quadratic_skip_current",
                description="Quadratic skip correction on single-hidden baseline",
                hypothesis="The remaining error may come from simple curvature in the encoded biomarker space, so adding a lightweight quadratic skip path could capture second-order corrections without needing a much larger residual network.",
                change="Add a small quadratic skip correction on squared standardized encoded features alongside the current linear skip and residual paths.",
                patch_fn=make_quadratic_skip_text,
            )
        )
    if "mini-batch cox with cosine decay on single-hidden baseline" not in lower:
        queue.append(
            exploration_experiment(
                base_text,
                family="training_regime_minibatch_current",
                description="Mini-batch Cox with cosine decay on single-hidden baseline",
                hypothesis="The current winner may be trapped by deterministic full-batch optimization, so revisiting mini-batch Cox on the stronger single-hidden baseline could unlock a different basin than the earlier weaker-architecture attempt.",
                change="Train the current single-hidden residual model with mini-batch Cox updates (`batch_size=1024`) and cosine learning-rate decay instead of the current full-batch constant-LR regime.",
                patch_fn=make_minibatch_cosine_retry_text,
            )
        )
    return queue


def choose_next_experiment(state: dict[str, Any], history: dict[str, Any], base_text: str, history_text: str) -> dict[str, Any]:
    attempted = set(state.get("attempted_hashes", []))
    phase = state.get("phase", "local")
    queued = state.get("exploration_queue", [])
    local_discards_since_keep = int(state.get("local_discards_since_keep", 0))

    if phase == "explore" and queued:
        remaining = [item for item in queued if item["train_hash"] not in attempted]
        state["exploration_queue"] = remaining
        if remaining:
            return remaining[0]

    if history["plateau_streak"] >= 8 or local_discards_since_keep >= MAX_LOCAL_DISCARDS_BEFORE_EXPLORATION:
        queue = build_exploration_queue(base_text, history_text)
        remaining = [item for item in queue if item["train_hash"] not in attempted]
        if remaining:
            burst = remaining[:MAX_EXPLORATION_BURST]
            state["phase"] = "explore"
            state["exploration_queue"] = burst
            return burst[0]

    state["phase"] = "local"
    state["exploration_queue"] = []
    local_queue = build_local_queue(base_text)
    remaining_local = [item for item in local_queue if item["train_hash"] not in attempted]
    if remaining_local:
        return remaining_local[0]

    raise RuntimeError("no fresh experiments remain for the current kept baseline")


def archive_run_log(pending: dict[str, Any]) -> None:
    archive_path = Path(pending["archive_path"])
    if RUNLOG.exists() and not archive_path.exists():
        shutil.copyfile(RUNLOG, archive_path)


def run_and_log(description: str, status: str) -> None:
    run(
        [
            str(PY),
            str(HERE / "log_result.py"),
            "--description",
            description,
            "--status",
            status,
        ],
        check=True,
    )


def apply_keep_or_restore(status: str) -> None:
    command = "save" if status == "keep" else "restore"
    run([str(PY), str(HERE / "manage_kept.py"), command], check=True)


def reconcile_pending_run(state: dict[str, Any]) -> bool:
    pending = load_json(PENDING, {})
    if not pending:
        return False

    child_pid = pending.get("child_pid")
    if child_pid and pid_is_running(child_pid):
        update_status_file(state, compute_history(load_results_rows()), "Waiting for active child run to finish.")
        time.sleep(RUN_POLL_SECONDS)
        return True

    log_text = RUNLOG.read_text(encoding="utf-8", errors="replace") if RUNLOG.exists() else ""
    if has_completed_summary(log_text):
        summary = parse_summary(log_text)
        archive_run_log(pending)
        improvement = summary["val_cindex"] - float(pending.get("baseline_best", 0.0))
        should_keep = improvement >= MEANINGFUL_IMPROVEMENT
        if (
            not should_keep
            and pending.get("prefer_simpler_on_tie", False)
            and improvement >= -SIMILARITY_TOLERANCE
        ):
            should_keep = True
        status = "keep" if should_keep else "discard"
        result_text = (
            f"`val_cindex` **{summary['val_cindex']:.6f}** at `best_step` **{summary['best_step']}** "
            f"vs best kept **{float(pending.get('baseline_best', 0.0)):.6f}**."
        )
        if not pending.get("logged", False):
            description = (
                f"{pending['description']}; val_cindex {summary['val_cindex']:.6f} "
                f"best_step {summary['best_step']}"
            )
            run_and_log(description, status)
            pending["logged"] = True
            pending["status"] = status
            save_json(PENDING, pending)
        if not pending.get("journal_written", False):
            if status == "keep":
                decision = "**keep**"
                learning = "This broader change improved enough to replace the kept baseline."
                next_move = "Resume local tuning around the new kept baseline."
            else:
                decision = "**discard**; restored `train.py` from `last_kept_train.py`."
                learning = "This change did not improve the kept baseline enough to justify replacing it."
                next_move = "Restore the kept baseline and move to the next queued conceptual probe."
            append_journal(
                hypothesis=pending["hypothesis"],
                change=pending["change"],
                result=result_text,
                decision=decision,
                learning=learning,
                next_move=next_move,
            )
            pending["journal_written"] = True
            save_json(PENDING, pending)
        if not pending.get("baseline_action_done", False):
            apply_keep_or_restore(status)
            pending["baseline_action_done"] = True
            save_json(PENDING, pending)
        state.setdefault("attempted_hashes", [])
        if pending["train_hash"] not in state["attempted_hashes"]:
            state["attempted_hashes"].append(pending["train_hash"])
        if status == "keep":
            state["phase"] = "local"
            state["exploration_queue"] = []
            state["attempted_hashes"] = [file_hash(read_text(KEPT))]
            state["local_discards_since_keep"] = 0
        elif pending.get("kind") == "local":
            state["local_discards_since_keep"] = int(state.get("local_discards_since_keep", 0)) + 1
        PENDING.unlink(missing_ok=True)
        update_status_file(state, compute_history(load_results_rows()), f"Reconciled completed run: {pending['description']}")
        return True

    archive_run_log(pending)
    if not pending.get("logged", False):
        run_and_log(f"crash: {pending['description']}", "crash")
        pending["logged"] = True
        save_json(PENDING, pending)
    if not pending.get("journal_written", False):
        append_journal(
            hypothesis=pending["hypothesis"],
            change=pending["change"],
            result="crash / no completed summary block in `run.log`",
            decision="**crash**; restored `train.py` from `last_kept_train.py`.",
            learning="The candidate did not finish cleanly, so it cannot be compared against the kept baseline.",
            next_move="Continue from the kept baseline with the next queued experiment.",
        )
        pending["journal_written"] = True
        save_json(PENDING, pending)
    if not pending.get("baseline_action_done", False):
        apply_keep_or_restore("discard")
        pending["baseline_action_done"] = True
        save_json(PENDING, pending)
    state.setdefault("attempted_hashes", [])
    if pending["train_hash"] not in state["attempted_hashes"]:
        state["attempted_hashes"].append(pending["train_hash"])
    if pending.get("kind") == "local":
        state["local_discards_since_keep"] = int(state.get("local_discards_since_keep", 0)) + 1
    PENDING.unlink(missing_ok=True)
    update_status_file(state, compute_history(load_results_rows()), f"Reconciled crashed run: {pending['description']}")
    return True


def launch_experiment(state: dict[str, Any], history: dict[str, Any], experiment: dict[str, Any]) -> None:
    base_text = read_text(KEPT)
    if read_text(TRAIN) != base_text:
        TRAIN.write_text(base_text, encoding="utf-8")

    TRAIN.write_text(experiment["train_text"], encoding="utf-8")
    RUNLOG.write_text("", encoding="utf-8")
    run_number = history["total_rows"] + 1
    archive_name = f"run_{run_number:03d}_{experiment['family']}.log"
    pending = {
        "archive_path": str(HERE / archive_name),
        "baseline_action_done": False,
        "baseline_best": history["best_keep"],
        "change": experiment["change"],
        "child_pid": 0,
        "description": experiment["description"],
        "family": experiment["family"],
        "hypothesis": experiment["hypothesis"],
        "journal_written": False,
        "kind": experiment["kind"],
        "logged": False,
        "prefer_simpler_on_tie": experiment.get("prefer_simpler_on_tie", False),
        "started_at_epoch": time.time(),
        "train_hash": experiment["train_hash"],
    }

    save_json(PENDING, pending)
    update_status_file(state, history, f"Started: {experiment['description']}")

    proc = subprocess.Popen(
        [str(PY), str(TRAIN)],
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    pending["child_pid"] = proc.pid
    save_json(PENDING, pending)
    update_status_file(state, history, f"Spawned child run: {experiment['description']}")

    with RUNLOG.open("w", encoding="utf-8") as log_handle:
        if proc.stdout is not None:
            for line in proc.stdout:
                log_handle.write(line)
                log_handle.flush()
        returncode = proc.wait()

    pending["child_pid"] = 0
    pending["returncode"] = returncode
    save_json(PENDING, pending)
    update_status_file(state, history, f"Finished raw run: {experiment['description']}")


def ensure_kept_snapshot() -> None:
    if KEPT.exists():
        return
    if TRAIN.exists():
        run([str(PY), str(HERE / "manage_kept.py"), "save"], check=True)


def summarize_results() -> None:
    run([str(PY), str(HERE / "summarize_results.py")], check=False)


def initialize_state() -> dict[str, Any]:
    default = {
        "attempted_hashes": [],
        "exploration_queue": [],
        "local_discards_since_keep": 0,
        "phase": "local",
        "started_at_epoch": time.time(),
        "supervisor_pid": 0,
    }
    state = load_json(STATE, default)
    state.setdefault("attempted_hashes", [])
    state.setdefault("exploration_queue", [])
    state.setdefault("local_discards_since_keep", 0)
    state.setdefault("phase", "local")
    state.setdefault("started_at_epoch", time.time())
    state["supervisor_pid"] = os.getpid()
    return state


def check_for_existing_supervisor(state: dict[str, Any]) -> None:
    disk_state = load_json(STATE, {})
    other_pid = disk_state.get("supervisor_pid")
    if other_pid and other_pid != os.getpid() and pid_is_running(other_pid):
        raise RuntimeError(f"Another autoresearch supervisor is already running with pid {other_pid}.")
    save_json(STATE, state)


def main() -> int:
    if not PY.is_file():
        print(f"Missing venv python: {PY}", file=sys.stderr)
        return 1

    ensure_kept_snapshot()
    if not KEPT.is_file():
        print("Missing last_kept_train.py; unable to start loop.", file=sys.stderr)
        return 1

    state = initialize_state()
    try:
        check_for_existing_supervisor(state)
        history = compute_history(load_results_rows())
        update_status_file(state, history, "Supervisor started.")

        while True:
            rows = load_results_rows()
            history = compute_history(rows)
            state["local_discards_since_keep"] = max(
                int(state.get("local_discards_since_keep", 0)),
                compute_local_discards_since_last_keep(rows),
            )
            elapsed = time.time() - float(state.get("started_at_epoch", time.time()))
            if elapsed >= MAX_SECONDS:
                msg = (
                    f"stop: time limit {MAX_SECONDS}s elapsed. "
                    f"Best kept val_cindex={history['best_keep']:.6f}. "
                    f"Completed runs={history['completed_runs']}."
                )
                record_stop_report(msg)
                update_status_file(state, history, msg)
                return 0
            if history["completed_runs"] >= MAX_RUNS:
                msg = f"stop: completed experiment limit {MAX_RUNS} reached."
                record_stop_report(msg)
                update_status_file(state, history, msg)
                return 0

            if reconcile_pending_run(state):
                save_json(STATE, state)
                continue

            summarize_results()
            journal_text = read_text(JOURNAL) if JOURNAL.exists() else ""
            rows = load_results_rows()
            history = compute_history(rows)
            state["local_discards_since_keep"] = max(
                int(state.get("local_discards_since_keep", 0)),
                compute_local_discards_since_last_keep(rows),
            )
            results_descriptions = "\n".join(row.get("description", "") for row in rows)
            history_text = journal_text + "\n" + results_descriptions
            base_text = read_text(KEPT)
            TRAIN.write_text(base_text, encoding="utf-8")

            if history["consecutive_discards"] >= MAX_CONSECUTIVE_DISCARDS:
                attempted = set(state.get("attempted_hashes", []))
                remaining_explore = [
                    item
                    for item in build_exploration_queue(base_text, history_text)
                    if item["train_hash"] not in attempted
                ]
                if not remaining_explore:
                    msg = (
                        f"stop: {MAX_CONSECUTIVE_DISCARDS} consecutive discarded runs with no meaningful "
                        f"improvement and no fresh exploration families remaining. "
                        f"Best kept val_cindex={history['best_keep']:.6f}."
                    )
                    record_stop_report(msg)
                    update_status_file(state, history, msg)
                    return 0

            try:
                experiment = choose_next_experiment(state, history, base_text, history_text)
            except RuntimeError as exc:
                msg = f"stop: {exc}. Best kept val_cindex={history['best_keep']:.6f}."
                record_stop_report(msg)
                update_status_file(state, history, msg)
                return 0
            save_json(STATE, state)
            launch_experiment(state, history, experiment)
            save_json(STATE, state)
    except KeyboardInterrupt:
        msg = (
            "Loop stopped by user interruption. "
            f"Best kept val_cindex remains {compute_history(load_results_rows())['best_keep']:.6f}."
        )
        record_stop_report(msg)
        update_status_file(state, compute_history(load_results_rows()), msg)
        return 0
    except Exception:
        tb = traceback.format_exc()
        msg = "stop: supervisor failure\n\n" + tb
        record_stop_report(msg)
        update_status_file(state, compute_history(load_results_rows()), "Supervisor failed; see loop_stop_report.txt.")
        print(tb, file=sys.stderr)
        return 1
    finally:
        state["supervisor_pid"] = 0
        save_json(STATE, state)


if __name__ == "__main__":
    raise SystemExit(main())
