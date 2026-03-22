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
MAX_CONSECUTIVE_DISCARDS = 60
MEANINGFUL_IMPROVEMENT = 0.0003
SIMILARITY_TOLERANCE = 0.00005
RUN_POLL_SECONDS = 5.0

# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


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


def run_cmd(cmd: list[str], *, check: bool = False, stdout=None, stderr=None) -> subprocess.CompletedProcess:
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


# ---------------------------------------------------------------------------
# Parsing and history
# ---------------------------------------------------------------------------


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


def has_logged_result(*, description: str, status: str) -> bool:
    for row in load_results_rows():
        if row.get("description", "").strip() == description and row.get("status", "").strip() == status:
            return True
    return False


def compute_history(rows: list[dict[str, str]]) -> dict[str, Any]:
    best_keep = 0.0
    consecutive_discards = 0
    completed_runs = 0
    for row in rows:
        status = row.get("status", "")
        val = float(row.get("val_cindex", "0") or 0.0)
        if status == "keep":
            consecutive_discards = 0
            if val > best_keep:
                best_keep = val
        elif status == "discard":
            consecutive_discards += 1
        if status in {"keep", "discard"}:
            completed_runs += 1
    return {
        "best_keep": best_keep,
        "consecutive_discards": consecutive_discards,
        "completed_runs": completed_runs,
        "total_rows": len(rows),
    }


def get_tried_descriptions() -> set[str]:
    """Return normalized core descriptions of every experiment ever attempted."""
    rows = load_results_rows()
    tried: set[str] = set()
    for row in rows:
        desc = row.get("description", "").strip()
        if not desc:
            continue
        core = desc.split(";")[0].strip()
        if core.startswith("crash: "):
            core = core[7:]
        tried.add(core.lower())
    return tried


def desc_is_tried(description: str, tried: set[str]) -> bool:
    core = description.split(";")[0].strip()
    if core.startswith("crash: "):
        core = core[7:]
    return core.lower() in tried


# ---------------------------------------------------------------------------
# Journal / status helpers
# ---------------------------------------------------------------------------


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
        f"best_kept_val_cindex: {history['best_keep']:.6f}",
        f"consecutive_discards: {history['consecutive_discards']}",
        f"completed_runs: {history['completed_runs']}",
        f"pending_description: {pending.get('description', 'none')}",
        f"note: {note}",
    ]
    atomic_write_text(STATUS, "\n".join(lines) + "\n")


def record_stop_report(message: str) -> None:
    atomic_write_text(STOP_REPORT, message.rstrip() + "\n")


# ---------------------------------------------------------------------------
# String manipulation
# ---------------------------------------------------------------------------


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


def replace_section(text: str, start_marker: str, end_marker: str, new_section: str, label: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise ValueError(f"missing start marker for {label}")
    end = text.find(end_marker, start)
    if end == -1:
        raise ValueError(f"missing end marker for {label}")
    return text[:start] + new_section + text[end:]


# ---------------------------------------------------------------------------
# Experiment factory
# ---------------------------------------------------------------------------


def _make_experiment(
    base_text: str,
    *,
    kind: str,
    family: str,
    description: str,
    hypothesis: str,
    change: str,
    patch_fn,
    prefer_simpler: bool = False,
) -> dict[str, Any] | None:
    """Build an experiment dict.  Returns None if patch_fn raises or is a no-op."""
    try:
        patched = patch_fn(base_text)
    except (ValueError, KeyError, TypeError, IndexError):
        return None
    base_h = file_hash(base_text)
    train_h = file_hash(patched)
    if train_h == base_h:
        return None
    return {
        "kind": kind,
        "family": family,
        "description": description,
        "hypothesis": hypothesis,
        "change": change,
        "train_text": patched,
        "train_hash": train_h,
        "prefer_simpler_on_tie": prefer_simpler,
    }


# ---------------------------------------------------------------------------
# Parse current hyperparameters from base text
# ---------------------------------------------------------------------------


def _parse_float(text: str, name: str) -> float | None:
    m = re.search(rf"^{name}\s*=\s*([0-9eE.+-]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def _parse_int(text: str, name: str) -> int | None:
    m = re.search(rf"^{name}\s*=\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def _parse_tuple(text: str, name: str) -> str | None:
    m = re.search(rf"^{name}\s*=\s*(\(.+?\))", text, re.MULTILINE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Experiment generators — hyperparameter perturbations
# ---------------------------------------------------------------------------


def gen_hp_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    lr = _parse_float(base_text, "LEARNING_RATE")
    wd = _parse_float(base_text, "WEIGHT_DECAY")
    do = _parse_float(base_text, "DROPOUT")
    ev = _parse_int(base_text, "EVAL_EVERY")
    pat = _parse_int(base_text, "EARLY_STOP_PATIENCE_EVALS")
    seed = _parse_int(base_text, "SEED")

    if lr is not None:
        for factor in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
            new_lr = round(lr * factor, 8)
            if new_lr == lr or new_lr <= 0:
                continue
            desc = f"LEARNING_RATE {lr} -> {new_lr}"
            if desc_is_tried(desc, tried):
                continue
            exp = _make_experiment(
                base_text, kind="local", family="lr_tuning", description=desc,
                hypothesis=f"LR scaled by {factor}x may find a better training basin.",
                change=f"LEARNING_RATE {lr} -> {new_lr}",
                patch_fn=lambda t, v=new_lr: replace_assignment(t, "LEARNING_RATE", repr(v)),
            )
            if exp:
                experiments.append(exp)

    if wd is not None:
        for factor in [0.0, 0.3, 0.5, 0.7, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0]:
            new_wd = round(wd * factor, 8) if factor > 0 else 0.0
            if new_wd == wd:
                continue
            desc = f"WEIGHT_DECAY {wd} -> {new_wd}"
            if desc_is_tried(desc, tried):
                continue
            exp = _make_experiment(
                base_text, kind="local", family="wd_tuning", description=desc,
                hypothesis=f"Weight decay scaled by {factor}x may regularize better.",
                change=f"WEIGHT_DECAY {wd} -> {new_wd}",
                patch_fn=lambda t, v=new_wd: replace_assignment(t, "WEIGHT_DECAY", repr(v)),
            )
            if exp:
                experiments.append(exp)

    if do is not None:
        for new_do in [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2]:
            if new_do == do:
                continue
            desc = f"DROPOUT {do} -> {new_do}"
            if desc_is_tried(desc, tried):
                continue
            exp = _make_experiment(
                base_text, kind="local", family="dropout_tuning", description=desc,
                hypothesis=f"Dropout {new_do} may regularize the residual head differently.",
                change=f"DROPOUT {do} -> {new_do}",
                patch_fn=lambda t, v=new_do: replace_assignment(t, "DROPOUT", repr(v)),
            )
            if exp:
                experiments.append(exp)

    if ev is not None:
        for new_ev in [10, 15, 20, 25, 30, 35, 40, 45, 55, 60, 75, 100]:
            if new_ev == ev:
                continue
            desc = f"EVAL_EVERY {ev} -> {new_ev}"
            if desc_is_tried(desc, tried):
                continue
            exp = _make_experiment(
                base_text, kind="local", family="eval_grid", description=desc,
                hypothesis=f"Eval spacing {new_ev} may capture the true val peak better.",
                change=f"EVAL_EVERY {ev} -> {new_ev}",
                patch_fn=lambda t, v=new_ev: replace_assignment(t, "EVAL_EVERY", str(v)),
            )
            if exp:
                experiments.append(exp)

    if pat is not None:
        for new_pat in [2, 4, 5, 6, 8, 10]:
            if new_pat == pat:
                continue
            desc = f"EARLY_STOP_PATIENCE_EVALS {pat} -> {new_pat}"
            if desc_is_tried(desc, tried):
                continue
            exp = _make_experiment(
                base_text, kind="local", family="patience_tuning", description=desc,
                hypothesis=f"Patience {new_pat} may allow the model to train longer or stop sooner.",
                change=f"EARLY_STOP_PATIENCE_EVALS {pat} -> {new_pat}",
                patch_fn=lambda t, v=new_pat: replace_assignment(t, "EARLY_STOP_PATIENCE_EVALS", str(v)),
            )
            if exp:
                experiments.append(exp)

    if seed is not None:
        for new_seed in [43, 44, 7, 137, 2024, 123, 999, 314]:
            if new_seed == seed:
                continue
            desc = f"SEED {seed} -> {new_seed}"
            if desc_is_tried(desc, tried):
                continue
            exp = _make_experiment(
                base_text, kind="explore", family="seed_robustness", description=desc,
                hypothesis=f"Seed {new_seed} may find a materially different training basin.",
                change=f"SEED {seed} -> {new_seed}",
                patch_fn=lambda t, v=new_seed: replace_assignment(t, "SEED", str(v)),
            )
            if exp:
                experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — architecture changes
# ---------------------------------------------------------------------------


def gen_architecture_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    current_hs = _parse_tuple(base_text, "HIDDEN_SIZES")

    candidates = [
        ("(8,)", "Narrow residual MLP (8,)"),
        ("(24,)", "Wider residual MLP (24,)"),
        ("(32,)", "Wider residual MLP (32,)"),
        ("(48,)", "Wider residual MLP (48,)"),
        ("(64,)", "Wider residual MLP (64,)"),
        ("(128,)", "Wide residual MLP (128,)"),
        ("(32, 16)", "Two-layer residual MLP (32, 16)"),
        ("(24, 12)", "Two-layer residual MLP (24, 12)"),
        ("(16, 8)", "Two-layer residual MLP (16, 8)"),
        ("(24, 16)", "Two-layer residual MLP (24, 16)"),
        ("(32, 16, 8)", "Three-layer residual MLP (32, 16, 8)"),
        ("(48, 24)", "Two-layer residual MLP (48, 24)"),
        ("(64, 32)", "Two-layer residual MLP (64, 32)"),
    ]
    for new_hs, desc in candidates:
        if current_hs and new_hs == current_hs:
            continue
        if desc_is_tried(desc, tried):
            continue
        exp = _make_experiment(
            base_text, kind="explore", family="architecture_width",
            description=desc,
            hypothesis=f"Changing residual MLP to {new_hs} may improve capacity/regularization balance.",
            change=f"HIDDEN_SIZES {current_hs} -> {new_hs}",
            patch_fn=lambda t, v=new_hs: replace_assignment(t, "HIDDEN_SIZES", v),
        )
        if exp:
            experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators - radical model families
# ---------------------------------------------------------------------------


def make_raw_feature_encoder_text(base_text: str) -> str:
    raw_encoder = """class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
"""
    return replace_section(
        base_text,
        "class FeatureEncoder(nn.Module):",
        "\n\nclass RiskMLP(nn.Module):",
        raw_encoder,
        "raw_feature_encoder",
    )


def make_raw_mlp_text(base_text: str) -> str:
    text = make_raw_feature_encoder_text(base_text)
    raw_mlp = """class RiskMLP(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        self.encoder = FeatureEncoder()
        input_dim = self.encoder.output_dim
        self.register_buffer("feature_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(input_dim, dtype=torch.float32))

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.head = nn.Sequential(*layers)

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        standardized = (encoded - self.feature_mean) / self.feature_std
        return self.head(standardized).squeeze(-1)
"""
    return replace_section(
        text,
        "class RiskMLP(nn.Module):",
        "\n\ndef cox_partial_loss(",
        raw_mlp,
        "raw_mlp_risk_model",
    )


def make_raw_linear_text(base_text: str) -> str:
    text = make_raw_feature_encoder_text(base_text)
    raw_linear = """class RiskMLP(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        self.encoder = FeatureEncoder()
        input_dim = self.encoder.output_dim
        self.register_buffer("feature_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(input_dim, dtype=torch.float32))
        self.linear = nn.Linear(input_dim, 1)

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        standardized = (encoded - self.feature_mean) / self.feature_std
        return self.linear(standardized).squeeze(-1)
"""
    return replace_section(
        text,
        "class RiskMLP(nn.Module):",
        "\n\ndef cox_partial_loss(",
        raw_linear,
        "raw_linear_risk_model",
    )


def make_engineered_linear_text(base_text: str) -> str:
    engineered_linear = """class RiskMLP(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        self.encoder = FeatureEncoder()
        input_dim = self.encoder.output_dim
        self.register_buffer("feature_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(input_dim, dtype=torch.float32))
        self.linear = nn.Linear(input_dim, 1)

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        standardized = (encoded - self.feature_mean) / self.feature_std
        return self.linear(standardized).squeeze(-1)
"""
    return replace_section(
        base_text,
        "class RiskMLP(nn.Module):",
        "\n\ndef cox_partial_loss(",
        engineered_linear,
        "engineered_linear_risk_model",
    )


def make_additive_raw_text(base_text: str) -> str:
    text = make_raw_feature_encoder_text(base_text)
    additive_model = """class RiskMLP(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        self.encoder = FeatureEncoder()
        input_dim = self.encoder.output_dim
        self.register_buffer("feature_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(input_dim, dtype=torch.float32))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, 8),
                    nn.SiLU(),
                    nn.Linear(8, 1),
                )
                for _ in range(input_dim)
            ]
        )

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        standardized = (encoded - self.feature_mean) / self.feature_std
        parts = [branch(standardized[:, i : i + 1]).squeeze(-1) for i, branch in enumerate(self.branches)]
        return torch.stack(parts, dim=0).sum(dim=0)
"""
    return replace_section(
        text,
        "class RiskMLP(nn.Module):",
        "\n\ndef cox_partial_loss(",
        additive_model,
        "additive_raw_model",
    )


def make_dual_encoder_text(base_text: str) -> str:
    dual_encoder = """class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 25
        self.albumin_scale = float(ALBUMIN_G_PER_DL_TO_G_PER_L)
        self.creatinine_scale = float(CREATININE_MG_PER_DL_TO_UMOL_PER_L)
        self.glucose_scale = float(GLUCOSE_MG_PER_DL_TO_MMOL_PER_L)
        self.crp_scale = float(CRP_MG_PER_DL_TO_MG_PER_L)
        self.albumin_coef = float(ALBUMIN_COEF)
        self.creatinine_coef = float(CREATININE_COEF)
        self.glucose_coef = float(GLUCOSE_COEF)
        self.log_crp_coef = float(LOG_CRP_COEF)
        self.lymphocyte_coef = float(LYMPHOCYTE_PERCENT_COEF)
        self.mcv_coef = float(MEAN_CELL_VOLUME_COEF)
        self.rdw_coef = float(RDW_COEF)
        self.alk_coef = float(ALKALINE_PHOSPHATASE_COEF)
        self.wbc_coef = float(WBC_COEF)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amp = x[:, 0] * self.albumin_scale
        cep = x[:, 1] * self.creatinine_scale
        sgp = x[:, 2] * self.glucose_scale
        crp_mg_per_l = x[:, 3] * self.crp_scale
        log_crp = torch.log(crp_mg_per_l.clamp_min(1e-6))
        lymph = x[:, 4]
        mcv = x[:, 5]
        rdw = x[:, 6]
        alk = x[:, 7]
        wbc = x[:, 8]

        pheno_no_age_xb = (
            self.albumin_coef * amp
            + self.creatinine_coef * cep
            + self.glucose_coef * sgp
            + self.log_crp_coef * log_crp
            + self.lymphocyte_coef * lymph
            + self.mcv_coef * mcv
            + self.rdw_coef * rdw
            + self.alk_coef * alk
            + self.wbc_coef * wbc
        )

        engineered = torch.stack(
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
        return torch.cat((x, engineered), dim=1)
"""
    text = replace_section(
        base_text,
        "class FeatureEncoder(nn.Module):",
        "\n\nclass RiskMLP(nn.Module):",
        dual_encoder,
        "dual_encoder",
    )
    dual_mlp = """class RiskMLP(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        self.encoder = FeatureEncoder()
        input_dim = self.encoder.output_dim
        self.register_buffer("feature_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(input_dim, dtype=torch.float32))

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.head = nn.Sequential(*layers)

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        standardized = (encoded - self.feature_mean) / self.feature_std
        return self.head(standardized).squeeze(-1)
"""
    return replace_section(
        text,
        "class RiskMLP(nn.Module):",
        "\n\ndef cox_partial_loss(",
        dual_mlp,
        "dual_encoder_risk_model",
    )


def make_pairwise_only_text(base_text: str) -> str:
    old_loss = """def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_scores = risk_scores[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_scores, dim=0)
    event_count = ordered_events.sum().clamp_min(1.0)
    losses = -(ordered_scores - log_risk) * ordered_events
    return losses.sum() / event_count
"""
    new_loss = """def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    event_idx = torch.nonzero(events > 0, as_tuple=False).squeeze(1)
    if event_idx.numel() == 0:
        return risk_scores.new_tensor(0.0)
    if event_idx.numel() > 512:
        perm = torch.randperm(event_idx.numel(), device=event_idx.device)
        event_idx = event_idx.index_select(0, perm[:512])
    pair_losses: list[torch.Tensor] = []
    for idx in event_idx:
        later_idx = torch.nonzero(times > times[idx], as_tuple=False).squeeze(1)
        if later_idx.numel() == 0:
            continue
        if later_idx.numel() > 512:
            perm = torch.randperm(later_idx.numel(), device=later_idx.device)
            later_idx = later_idx.index_select(0, perm[:512])
        score_diff = risk_scores[idx] - risk_scores.index_select(0, later_idx)
        pair_losses.append(torch.nn.functional.softplus(-score_diff).mean())
    if not pair_losses:
        return risk_scores.new_tensor(0.0)
    return torch.stack(pair_losses).mean()
"""
    return replace_once(base_text, old_loss, new_loss, "pairwise_only_loss")


def gen_radical_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []

    specs: list[tuple[str, str, str, Any, bool]] = [
        (
            "Raw biomarker MLP (32, 16), no pheno anchor",
            "Pure raw-biomarker modeling may outperform the anchor-based design if the hand-crafted pheno pathway is constraining the optimum.",
            "Replace the anchor-aligned encoder/pathway with a raw 9-biomarker MLP using HIDDEN_SIZES=(32, 16).",
            lambda t: replace_assignment(replace_assignment(make_raw_mlp_text(t), "HIDDEN_SIZES", "(32, 16)"), "DROPOUT", "0.1"),
            False,
        ),
        (
            "Raw biomarker MLP (64, 32), no pheno anchor",
            "A wider anchor-free MLP could capture nonlinear structure that the anchored family cannot express.",
            "Replace the anchor-aligned encoder/pathway with a raw 9-biomarker MLP using HIDDEN_SIZES=(64, 32).",
            lambda t: replace_assignment(replace_assignment(make_raw_mlp_text(t), "HIDDEN_SIZES", "(64, 32)"), "DROPOUT", "0.1"),
            False,
        ),
        (
            "Raw biomarker linear Cox",
            "A completely linear age-free model on raw biomarkers tests whether the search is overcomplicating a mostly linear signal.",
            "Use a standardized 9-biomarker linear Cox model with no pheno anchor and no hidden layers.",
            make_raw_linear_text,
            True,
        ),
        (
            "Engineered-feature linear Cox",
            "The best signal may live in the engineered 16-feature representation but not require any nonlinear residual network.",
            "Use a standardized linear Cox model on the current 16 engineered biomarker features.",
            make_engineered_linear_text,
            True,
        ),
        (
            "Additive biomarker subnetworks",
            "A generalized-additive style model may capture smooth per-biomarker nonlinearities without relying on dense multivariate mixing.",
            "Use nine independent 1D biomarker subnetworks and sum their outputs.",
            make_additive_raw_text,
            False,
        ),
        (
            "Dual raw+engineered MLP (64, 32)",
            "Combining raw biomarkers with engineered pheno-style features may recover signal lost by committing to only one representation.",
            "Concatenate raw 9 biomarkers with engineered 16 features, then train a SiLU MLP with HIDDEN_SIZES=(64, 32).",
            lambda t: replace_assignment(replace_assignment(make_dual_encoder_text(t), "HIDDEN_SIZES", "(64, 32)"), "DROPOUT", "0.1"),
            False,
        ),
        (
            "Pure pairwise ranking loss on current architecture",
            "If Cox is misaligned with the actual ranking objective, optimizing pairwise concordance directly may help.",
            "Replace Cox loss with sampled pairwise logistic ranking loss while keeping the current architecture.",
            make_pairwise_only_text,
            False,
        ),
        (
            "Pure pairwise ranking + raw biomarker MLP",
            "A new model family and a directly aligned ranking loss together may discover a basin the anchored Cox models never reach.",
            "Use a raw 9-biomarker MLP with HIDDEN_SIZES=(32, 16) and pure pairwise ranking loss.",
            lambda t: make_pairwise_only_text(replace_assignment(replace_assignment(make_raw_mlp_text(t), "HIDDEN_SIZES", "(32, 16)"), "DROPOUT", "0.1")),
            False,
        ),
    ]

    for desc, hyp, change, patch_fn, prefer_simpler in specs:
        if desc_is_tried(desc, tried):
            continue
        exp = _make_experiment(
            base_text,
            kind="explore",
            family="radical_models",
            description=desc,
            hypothesis=hyp,
            change=change,
            patch_fn=patch_fn,
            prefer_simpler=prefer_simpler,
        )
        if exp:
            experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — activation functions
# ---------------------------------------------------------------------------


def gen_activation_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    if "nn.GELU()" not in base_text:
        return experiments

    activations = [
        ("nn.SiLU()", "SiLU"),
        ("nn.ELU()", "ELU"),
        ("nn.Mish()", "Mish"),
        ("nn.LeakyReLU()", "LeakyReLU"),
        ("nn.Tanh()", "Tanh"),
        ("nn.ReLU()", "ReLU"),
    ]
    for act_code, act_name in activations:
        desc = f"Residual activation GELU -> {act_name}"
        if desc_is_tried(desc, tried):
            continue
        exp = _make_experiment(
            base_text, kind="explore", family="activation",
            description=desc,
            hypothesis=f"{act_name} may provide better gradient flow for the residual correction path.",
            change=f"Replace nn.GELU() with {act_code} in residual head.",
            patch_fn=lambda t, a=act_code: replace_once(t, "nn.GELU()", a, "activation"),
        )
        if exp:
            experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — optimizer changes
# ---------------------------------------------------------------------------


def gen_optimizer_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    adamw_line = "    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)"
    if adamw_line not in base_text:
        return experiments

    opt_changes = [
        (
            "AdamW with amsgrad",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)",
            "amsgrad may help stabilize the training trajectory.",
        ),
        (
            "AdamW beta2=0.98",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))",
            "Faster second-moment decay may reduce over-adaptation to early noisy gradients.",
        ),
        (
            "AdamW beta2=0.95",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))",
            "Aggressive second-moment decay may help in the low-data regime.",
        ),
        (
            "SGD with momentum 0.9",
            "    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)",
            "SGD with momentum may generalize differently than adaptive methods.",
        ),
        (
            "AdamW beta1=0.85",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.85, 0.999))",
            "Lower beta1 may help escape local minima by reducing momentum smoothing.",
        ),
    ]

    for desc, new_line, hyp in opt_changes:
        if desc_is_tried(desc, tried):
            continue
        exp = _make_experiment(
            base_text, kind="explore", family="optimizer",
            description=desc, hypothesis=hyp,
            change=f"Replace optimizer line with: {desc}",
            patch_fn=lambda t, nl=new_line: replace_once(t, adamw_line, nl, "optimizer"),
        )
        if exp:
            experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — training dynamics
# ---------------------------------------------------------------------------


def gen_training_dynamics_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []

    # Gradient clipping
    backward_line = "        loss.backward()\n        optimizer.step()"
    if backward_line in base_text:
        desc = "Gradient clipping max_norm=1.0"
        if not desc_is_tried(desc, tried):
            exp = _make_experiment(
                base_text, kind="explore", family="training_dynamics",
                description=desc,
                hypothesis="Gradient clipping may prevent sharp early steps that overshoot the optimum.",
                change="Add torch.nn.utils.clip_grad_norm_ between backward and step.",
                patch_fn=lambda t: replace_once(
                    t, backward_line,
                    "        loss.backward()\n"
                    "        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
                    "        optimizer.step()",
                    "grad_clip",
                ),
            )
            if exp:
                experiments.append(exp)

        desc = "Gradient clipping max_norm=0.5"
        if not desc_is_tried(desc, tried):
            exp = _make_experiment(
                base_text, kind="explore", family="training_dynamics",
                description=desc,
                hypothesis="Tighter gradient clipping may further stabilize the early training phase.",
                change="Add torch.nn.utils.clip_grad_norm_(max_norm=0.5) between backward and step.",
                patch_fn=lambda t: replace_once(
                    t, backward_line,
                    "        loss.backward()\n"
                    "        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)\n"
                    "        optimizer.step()",
                    "grad_clip_tight",
                ),
            )
            if exp:
                experiments.append(exp)

    # LR warmup + cosine decay
    opt_create = "    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n"
    if opt_create in base_text:
        desc = "Cosine LR schedule"
        if not desc_is_tried(desc, tried):
            new_opt = (
                "    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n"
                "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=LEARNING_RATE * 0.1)\n"
            )
            step_old = "        optimizer.step()\n"
            step_new = "        optimizer.step()\n        scheduler.step()\n"

            def patch_cosine(t: str) -> str:
                t = replace_once(t, opt_create, new_opt, "cosine_opt")
                return replace_once(t, step_old, step_new, "cosine_step")

            exp = _make_experiment(
                base_text, kind="explore", family="lr_schedule",
                description=desc,
                hypothesis="Cosine decay may prevent overfitting by reducing LR as training progresses.",
                change="Add CosineAnnealingLR scheduler with T_max=800.",
                patch_fn=patch_cosine,
            )
            if exp:
                experiments.append(exp)

        desc = "Linear warmup 100 steps + cosine decay"
        if not desc_is_tried(desc, tried):
            new_opt = (
                "    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n"
                "    warmup_steps = 100\n"
                "    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=LEARNING_RATE * 0.1)\n"
                "    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)\n"
                "    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])\n"
            )
            step_old = "        optimizer.step()\n"
            step_new = "        optimizer.step()\n        scheduler.step()\n"

            def patch_warmup_cosine(t: str) -> str:
                t = replace_once(t, opt_create, new_opt, "warmup_cosine_opt")
                return replace_once(t, step_old, step_new, "warmup_cosine_step")

            exp = _make_experiment(
                base_text, kind="explore", family="lr_schedule",
                description=desc,
                hypothesis="Warmup prevents early instability; cosine decay prevents late overfitting.",
                change="Add LinearLR warmup for 100 steps, then CosineAnnealingLR.",
                patch_fn=patch_warmup_cosine,
            )
            if exp:
                experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — loss function
# ---------------------------------------------------------------------------


def gen_loss_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    cox_fn = (
        "def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:\n"
        "    order = torch.argsort(times, descending=True)\n"
        "    ordered_scores = risk_scores[order]\n"
        "    ordered_events = events[order]\n"
        "    log_risk = torch.logcumsumexp(ordered_scores, dim=0)\n"
        "    event_count = ordered_events.sum().clamp_min(1.0)\n"
        "    losses = -(ordered_scores - log_risk) * ordered_events\n"
        "    return losses.sum() / event_count\n"
    )
    if cox_fn not in base_text:
        return experiments

    # Time-weighted Cox
    desc = "Time-weighted Cox loss (power=0.35)"
    if not desc_is_tried(desc, tried):
        weighted_cox = (
            "def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:\n"
            "    order = torch.argsort(times, descending=True)\n"
            "    ordered_scores = risk_scores[order]\n"
            "    ordered_times = times[order]\n"
            "    ordered_events = events[order]\n"
            "    log_risk = torch.logcumsumexp(ordered_scores, dim=0)\n"
            "    base_losses = -(ordered_scores - log_risk) * ordered_events\n"
            "    safe_times = ordered_times.clamp_min(1.0)\n"
            "    event_weights = torch.where(\n"
            "        ordered_events > 0,\n"
            "        (safe_times.mean() / safe_times).pow(0.35),\n"
            "        torch.ones_like(safe_times),\n"
            "    )\n"
            "    weighted_events = ordered_events * event_weights\n"
            "    normalizer = weighted_events.sum().clamp_min(1.0)\n"
            "    return (base_losses * event_weights).sum() / normalizer\n"
        )
        exp = _make_experiment(
            base_text, kind="explore", family="loss_design",
            description=desc,
            hypothesis="Upweighting earlier events may improve ranking of the most informative survival pairs.",
            change="Replace uniform Cox loss with time-weighted variant.",
            patch_fn=lambda t: replace_once(t, cox_fn, weighted_cox, "weighted_cox"),
        )
        if exp:
            experiments.append(exp)

    # Cox + L1 regularization on residual correction
    desc = "Cox loss + L1 penalty on residual output"
    if not desc_is_tried(desc, tried):
        loss_line = "        loss = cox_partial_loss(risk_scores, train_times, train_events)"
        new_loss = (
            "        loss = cox_partial_loss(risk_scores, train_times, train_events)\n"
            "        loss = loss + 1e-4 * risk_scores.abs().mean()"
        )
        if loss_line in base_text:
            exp = _make_experiment(
                base_text, kind="explore", family="loss_design",
                description=desc,
                hypothesis="L1 penalty on risk scores may compress the learned scale and reduce overfitting.",
                change="Add L1 regularization on model output to Cox loss.",
                patch_fn=lambda t: replace_once(t, loss_line, new_loss, "l1_loss"),
            )
            if exp:
                experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — feature representation
# ---------------------------------------------------------------------------


def gen_feature_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []

    # Remove interaction features
    interaction_block = (
        "                amp * rdw,\n"
        "                sgp * log_crp,\n"
        "                wbc * log_crp,\n"
        "                lymph * rdw,\n"
        "                alk * rdw,\n"
        "                cep * log_crp,\n"
    )
    if interaction_block in base_text and "self.output_dim = 16" in base_text:
        desc = "Remove interaction features (10 inputs only)"
        if not desc_is_tried(desc, tried):
            def patch_remove_interactions(t: str) -> str:
                t = replace_once(t, "        self.output_dim = 16", "        self.output_dim = 10", "remove_int_dim")
                return replace_once(t, interaction_block, "", "remove_int_features")

            exp = _make_experiment(
                base_text, kind="explore", family="feature_representation",
                description=desc,
                hypothesis="Interaction features may be adding noise; raw biomarkers + pheno_xb may suffice.",
                change="Remove 6 interaction features, keep 9 raw + pheno_no_age_xb.",
                patch_fn=patch_remove_interactions,
                prefer_simpler=True,
            )
            if exp:
                experiments.append(exp)

    # Different interaction set: squares instead of products
    if interaction_block in base_text and "self.output_dim = 16" in base_text:
        desc = "Replace product interactions with squared features"
        if not desc_is_tried(desc, tried):
            square_block = (
                "                amp * amp,\n"
                "                log_crp * log_crp,\n"
                "                rdw * rdw,\n"
                "                wbc * wbc,\n"
                "                lymph * lymph,\n"
                "                alk * alk,\n"
            )

            exp = _make_experiment(
                base_text, kind="explore", family="feature_representation",
                description=desc,
                hypothesis="Squared terms may capture nonlinear biomarker effects more directly.",
                change="Replace 6 product interactions with 6 squared features.",
                patch_fn=lambda t: replace_once(t, interaction_block, square_block, "square_features"),
            )
            if exp:
                experiments.append(exp)

    # Add ratio features
    if interaction_block in base_text and "self.output_dim = 16" in base_text:
        desc = "Add ratio features (22 inputs total)"
        if not desc_is_tried(desc, tried):
            extended_block = (
                "                amp * rdw,\n"
                "                sgp * log_crp,\n"
                "                wbc * log_crp,\n"
                "                lymph * rdw,\n"
                "                alk * rdw,\n"
                "                cep * log_crp,\n"
                "                amp / cep.clamp_min(1e-6),\n"
                "                wbc / lymph.clamp_min(1e-6),\n"
                "                rdw / mcv.clamp_min(1e-6),\n"
                "                sgp / amp.clamp_min(1e-6),\n"
                "                alk / amp.clamp_min(1e-6),\n"
                "                torch.log1p(alk.clamp_min(0.0)),\n"
            )

            def patch_ratio(t: str) -> str:
                t = replace_once(t, "        self.output_dim = 16", "        self.output_dim = 22", "ratio_dim")
                return replace_once(t, interaction_block, extended_block, "ratio_features")

            exp = _make_experiment(
                base_text, kind="explore", family="feature_representation",
                description=desc,
                hypothesis="Biomarker ratios may capture clinically meaningful relationships.",
                change="Add 6 ratio/log features to existing 16 encoder features.",
                patch_fn=patch_ratio,
            )
            if exp:
                experiments.append(exp)

    # Remove linear skip path
    if "self.linear_skip = nn.Linear(input_dim, 1)" in base_text:
        desc = "Remove linear skip path"
        if not desc_is_tried(desc, tried):
            def patch_remove_skip(t: str) -> str:
                t = replace_once(
                    t,
                    "        self.linear_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))\n"
                    "        self.linear_skip = nn.Linear(input_dim, 1)",
                    "",
                    "remove_skip_params",
                )
                t = replace_once(
                    t,
                    "        linear_skip = self.linear_skip(standardized).squeeze(-1)\n"
                    "        return self.base_weight * base_score + self.residual_scale * residual + self.linear_scale * linear_skip",
                    "        return self.base_weight * base_score + self.residual_scale * residual",
                    "remove_skip_forward",
                )
                return t

            exp = _make_experiment(
                base_text, kind="explore", family="architecture_simplify",
                description=desc,
                hypothesis="The linear skip may be redundant with the MLP path; removing it simplifies the model.",
                change="Remove linear_skip and linear_scale from the model.",
                patch_fn=patch_remove_skip,
                prefer_simpler=True,
            )
            if exp:
                experiments.append(exp)

    # Add quadratic skip
    if ("self.linear_skip = nn.Linear(input_dim, 1)" in base_text
            and "quadratic" not in base_text):
        desc = "Add quadratic skip correction"
        if not desc_is_tried(desc, tried):
            def patch_quadratic(t: str) -> str:
                t = replace_once(
                    t,
                    "        self.linear_skip = nn.Linear(input_dim, 1)",
                    "        self.linear_skip = nn.Linear(input_dim, 1)\n"
                    "        self.quadratic_scale = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))\n"
                    "        self.quadratic_skip = nn.Linear(input_dim, 1, bias=False)",
                    "quad_params",
                )
                t = replace_once(
                    t,
                    "        return self.base_weight * base_score + self.residual_scale * residual + self.linear_scale * linear_skip",
                    "        quadratic_skip = self.quadratic_skip(standardized.square()).squeeze(-1)\n"
                    "        return self.base_weight * base_score + self.residual_scale * residual + self.linear_scale * linear_skip + self.quadratic_scale * quadratic_skip",
                    "quad_forward",
                )
                return t

            exp = _make_experiment(
                base_text, kind="explore", family="architecture_quadratic",
                description=desc,
                hypothesis="A quadratic path may capture second-order biomarker effects cheaply.",
                change="Add a lightweight quadratic skip correction path.",
                patch_fn=patch_quadratic,
            )
            if exp:
                experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — normalization
# ---------------------------------------------------------------------------


def gen_normalization_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    layers_block = "        layers: list[nn.Module] = []\n        last_dim = input_dim\n"

    if layers_block not in base_text:
        return experiments

    # LayerNorm on input
    desc = "LayerNorm on encoder features before residual head"
    if not desc_is_tried(desc, tried):
        new_block = (
            "        self.input_norm = nn.LayerNorm(input_dim)\n"
            "        layers: list[nn.Module] = []\n"
            "        last_dim = input_dim\n"
        )
        fwd_old = "        residual = self.residual_head(standardized).squeeze(-1)"
        fwd_new = "        normed = self.input_norm(standardized)\n        residual = self.residual_head(normed).squeeze(-1)"

        def patch_layernorm(t: str) -> str:
            t = replace_once(t, layers_block, new_block, "layernorm_init")
            return replace_once(t, fwd_old, fwd_new, "layernorm_forward")

        exp = _make_experiment(
            base_text, kind="explore", family="normalization",
            description=desc,
            hypothesis="LayerNorm may stabilize the residual head input distribution across biomarker scales.",
            change="Add LayerNorm before the residual MLP.",
            patch_fn=patch_layernorm,
        )
        if exp:
            experiments.append(exp)

    # BatchNorm after first hidden layer
    desc = "BatchNorm1d after first hidden layer"
    if not desc_is_tried(desc, tried):
        old_loop = (
            "            layers.append(nn.Linear(last_dim, hidden_dim))\n"
            "            layers.append(nn.GELU())\n"
        )
        new_loop = (
            "            layers.append(nn.Linear(last_dim, hidden_dim))\n"
            "            layers.append(nn.BatchNorm1d(hidden_dim))\n"
            "            layers.append(nn.GELU())\n"
        )
        if old_loop in base_text:
            exp = _make_experiment(
                base_text, kind="explore", family="normalization",
                description=desc,
                hypothesis="BatchNorm may reduce internal covariate shift and speed up convergence.",
                change="Add BatchNorm1d after each hidden Linear layer.",
                patch_fn=lambda t: replace_once(t, old_loop, new_loop, "batchnorm"),
            )
            if exp:
                experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — regularization
# ---------------------------------------------------------------------------


def gen_regularization_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []

    # Feature noise during training
    std_line = "        standardized = (encoded - self.feature_mean) / self.feature_std"
    if std_line in base_text and "randn_like" not in base_text:
        for noise_std in [0.01, 0.03, 0.05, 0.1]:
            desc = f"Feature noise std={noise_std} during training"
            if desc_is_tried(desc, tried):
                continue
            new_std = (
                f"        standardized = (encoded - self.feature_mean) / self.feature_std\n"
                f"        if self.training:\n"
                f"            standardized = standardized + {noise_std} * torch.randn_like(standardized)"
            )
            exp = _make_experiment(
                base_text, kind="explore", family="regularization",
                description=desc,
                hypothesis=f"Input noise (std={noise_std}) may regularize the model against overfitting to specific biomarker values.",
                change=f"Add Gaussian noise (std={noise_std}) to standardized features during training.",
                patch_fn=lambda t, ns=new_std: replace_once(t, std_line, ns, "feature_noise"),
            )
            if exp:
                experiments.append(exp)

    # Feature gate
    if "feature_gate" not in base_text and "self.linear_skip = nn.Linear(input_dim, 1)" in base_text:
        desc = "Learned feature gate (sigmoid)"
        if not desc_is_tried(desc, tried):
            def patch_gate(t: str) -> str:
                t = replace_once(
                    t,
                    "        self.linear_skip = nn.Linear(input_dim, 1)",
                    "        self.linear_skip = nn.Linear(input_dim, 1)\n"
                    "        self.feature_gate = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))",
                    "gate_param",
                )
                t = replace_once(
                    t,
                    "        standardized = (encoded - self.feature_mean) / self.feature_std\n"
                    "        residual = self.residual_head(standardized).squeeze(-1)\n"
                    "        linear_skip = self.linear_skip(standardized).squeeze(-1)",
                    "        standardized = (encoded - self.feature_mean) / self.feature_std\n"
                    "        gated = standardized * torch.sigmoid(self.feature_gate)\n"
                    "        residual = self.residual_head(gated).squeeze(-1)\n"
                    "        linear_skip = self.linear_skip(gated).squeeze(-1)",
                    "gate_forward",
                )
                return t

            exp = _make_experiment(
                base_text, kind="explore", family="regularization",
                description=desc,
                hypothesis="A learned feature gate may suppress noisy encoder channels.",
                change="Add sigmoid feature gate on standardized features.",
                patch_fn=patch_gate,
            )
            if exp:
                experiments.append(exp)

    # Fix anchor weight
    if "self.base_weight = nn.Parameter" in base_text:
        desc = "Fix base anchor weight at 1.0"
        if not desc_is_tried(desc, tried):
            def patch_fix_anchor(t: str) -> str:
                return replace_once(
                    t,
                    '        self.base_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))',
                    '        self.register_buffer("base_weight", torch.tensor(1.0, dtype=torch.float32))',
                    "fix_anchor",
                )

            exp = _make_experiment(
                base_text, kind="explore", family="anchor_design",
                description=desc,
                hypothesis="Fixing the anchor weight may prevent co-adaptation between base and residual.",
                change="Replace learned base_weight with fixed buffer of 1.0.",
                patch_fn=patch_fix_anchor,
                prefer_simpler=True,
            )
            if exp:
                experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — scale/init
# ---------------------------------------------------------------------------


def gen_scale_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []

    for param, values in [
        ("residual_scale", [0.01, 0.03, 0.05, 0.08, 0.15, 0.2, 0.3]),
        ("linear_scale", [0.01, 0.02, 0.08, 0.1, 0.15]),
    ]:
        old_pattern = f'self.{param} = nn.Parameter(torch.tensor('
        m = re.search(rf'self\.{param} = nn\.Parameter\(torch\.tensor\(([0-9.]+)', base_text)
        if not m:
            continue
        current = float(m.group(1))
        for new_val in values:
            if new_val == current:
                continue
            desc = f"Init {param} {current} -> {new_val}"
            if desc_is_tried(desc, tried):
                continue

            old_str = f'self.{param} = nn.Parameter(torch.tensor({current}, dtype=torch.float32))'
            new_str = f'self.{param} = nn.Parameter(torch.tensor({new_val}, dtype=torch.float32))'

            exp = _make_experiment(
                base_text, kind="explore", family="init_scale",
                description=desc,
                hypothesis=f"Changing {param} init from {current} to {new_val} may shift the balance between anchor and correction.",
                change=f"{param} initial value {current} -> {new_val}.",
                patch_fn=lambda t, o=old_str, n=new_str: replace_once(t, o, n, f"{param}_init"),
            )
            if exp:
                experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Experiment generators — combined promising changes
# ---------------------------------------------------------------------------


def gen_combined_experiments(base_text: str, tried: set[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []

    # SiLU + wider (32,)
    if "nn.GELU()" in base_text:
        desc = "SiLU activation + wider MLP (32,)"
        if not desc_is_tried(desc, tried):
            def patch_silu_wider(t: str) -> str:
                t = replace_once(t, "nn.GELU()", "nn.SiLU()", "silu_act")
                return replace_assignment(t, "HIDDEN_SIZES", "(32,)")

            exp = _make_experiment(
                base_text, kind="explore", family="combined",
                description=desc,
                hypothesis="SiLU was the best-scoring discard and wider MLPs help; combining them may exceed either alone.",
                change="Switch to SiLU and widen residual MLP to (32,).",
                patch_fn=patch_silu_wider,
            )
            if exp:
                experiments.append(exp)

        desc = "SiLU activation + wider MLP (48,)"
        if not desc_is_tried(desc, tried):
            def patch_silu_48(t: str) -> str:
                t = replace_once(t, "nn.GELU()", "nn.SiLU()", "silu_act")
                return replace_assignment(t, "HIDDEN_SIZES", "(48,)")

            exp = _make_experiment(
                base_text, kind="explore", family="combined",
                description=desc,
                hypothesis="SiLU + larger capacity may discover a stronger correction path.",
                change="Switch to SiLU and widen residual MLP to (48,).",
                patch_fn=patch_silu_48,
            )
            if exp:
                experiments.append(exp)

    # Wider MLP + higher dropout
    desc = "Wider MLP (32,) + dropout 0.1"
    if not desc_is_tried(desc, tried):
        def patch_wider_dropout(t: str) -> str:
            t = replace_assignment(t, "HIDDEN_SIZES", "(32,)")
            return replace_assignment(t, "DROPOUT", "0.1")

        exp = _make_experiment(
            base_text, kind="explore", family="combined",
            description=desc,
            hypothesis="More capacity with stronger regularization may improve generalization.",
            change="Widen to (32,) and increase dropout to 0.1.",
            patch_fn=patch_wider_dropout,
        )
        if exp:
            experiments.append(exp)

    desc = "Wider MLP (48,) + dropout 0.15"
    if not desc_is_tried(desc, tried):
        def patch_wider48_dropout(t: str) -> str:
            t = replace_assignment(t, "HIDDEN_SIZES", "(48,)")
            return replace_assignment(t, "DROPOUT", "0.15")

        exp = _make_experiment(
            base_text, kind="explore", family="combined",
            description=desc,
            hypothesis="Significantly more capacity with strong dropout may find better features.",
            change="Widen to (48,) and increase dropout to 0.15.",
            patch_fn=patch_wider48_dropout,
        )
        if exp:
            experiments.append(exp)

    # Lower LR + wider MLP
    desc = "Wider MLP (32,) + lower LR 0.001"
    if not desc_is_tried(desc, tried):
        def patch_wider_lower_lr(t: str) -> str:
            t = replace_assignment(t, "HIDDEN_SIZES", "(32,)")
            return replace_assignment(t, "LEARNING_RATE", "0.001")

        exp = _make_experiment(
            base_text, kind="explore", family="combined",
            description=desc,
            hypothesis="Wider network may need slower learning to converge properly.",
            change="Widen to (32,) and halve LR to 0.001.",
            patch_fn=patch_wider_lower_lr,
        )
        if exp:
            experiments.append(exp)

    # Remove interactions + wider MLP
    interaction_block = (
        "                amp * rdw,\n"
        "                sgp * log_crp,\n"
        "                wbc * log_crp,\n"
        "                lymph * rdw,\n"
        "                alk * rdw,\n"
        "                cep * log_crp,\n"
    )
    if interaction_block in base_text and "self.output_dim = 16" in base_text:
        desc = "Remove interactions + wider MLP (48,)"
        if not desc_is_tried(desc, tried):
            def patch_no_int_wide(t: str) -> str:
                t = replace_once(t, "        self.output_dim = 16", "        self.output_dim = 10", "no_int_dim")
                t = replace_once(t, interaction_block, "", "no_int_features")
                return replace_assignment(t, "HIDDEN_SIZES", "(48,)")

            exp = _make_experiment(
                base_text, kind="explore", family="combined",
                description=desc,
                hypothesis="Let the MLP learn its own feature interactions instead of hand-crafting them.",
                change="Remove 6 interaction features, widen MLP to (48,).",
                patch_fn=patch_no_int_wide,
            )
            if exp:
                experiments.append(exp)

        desc = "Remove interactions + two-layer MLP (32, 16)"
        if not desc_is_tried(desc, tried):
            def patch_no_int_deep(t: str) -> str:
                t = replace_once(t, "        self.output_dim = 16", "        self.output_dim = 10", "no_int_dim")
                t = replace_once(t, interaction_block, "", "no_int_features")
                return replace_assignment(t, "HIDDEN_SIZES", "(32, 16)")

            exp = _make_experiment(
                base_text, kind="explore", family="combined",
                description=desc,
                hypothesis="A deeper MLP on raw features may learn better interactions than hand-crafted ones.",
                change="Remove 6 interaction features, use (32, 16) MLP.",
                patch_fn=patch_no_int_deep,
            )
            if exp:
                experiments.append(exp)

    return experiments


# ---------------------------------------------------------------------------
# Queue building and experiment selection
# ---------------------------------------------------------------------------


def build_full_queue(base_text: str) -> list[dict[str, Any]]:
    """Build ALL candidate experiments, filtered for novelty. Exploration first."""
    tried = get_tried_descriptions()
    queue: list[dict[str, Any]] = []

    queue.extend(gen_radical_experiments(base_text, tried))
    queue.extend(gen_architecture_experiments(base_text, tried))
    queue.extend(gen_activation_experiments(base_text, tried))
    queue.extend(gen_combined_experiments(base_text, tried))
    queue.extend(gen_normalization_experiments(base_text, tried))
    queue.extend(gen_optimizer_experiments(base_text, tried))
    queue.extend(gen_training_dynamics_experiments(base_text, tried))
    queue.extend(gen_loss_experiments(base_text, tried))
    queue.extend(gen_feature_experiments(base_text, tried))
    queue.extend(gen_regularization_experiments(base_text, tried))
    queue.extend(gen_scale_experiments(base_text, tried))
    queue.extend(gen_hp_experiments(base_text, tried))

    seen_hashes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for exp in queue:
        h = exp["train_hash"]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append(exp)

    return deduped


def choose_next_experiment(state: dict[str, Any], base_text: str) -> dict[str, Any]:
    """Pick the next untried experiment from the full queue."""
    attempted = set(state.get("attempted_hashes", []))
    queue = build_full_queue(base_text)
    for exp in queue:
        if exp["train_hash"] not in attempted:
            return exp
    raise RuntimeError("all experiments exhausted for the current kept baseline")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def run_and_log(description: str, status: str) -> None:
    run_cmd(
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
    run_cmd([str(PY), str(HERE / "manage_kept.py"), command], check=True)


def archive_run_log(pending: dict[str, Any]) -> None:
    archive_path = Path(pending["archive_path"])
    if RUNLOG.exists() and not archive_path.exists():
        shutil.copyfile(RUNLOG, archive_path)


# ---------------------------------------------------------------------------
# Recovery: reconcile a pending run that finished or crashed
# ---------------------------------------------------------------------------


def reconcile_pending_run(state: dict[str, Any]) -> bool:
    pending = load_json(PENDING, {})
    if not pending:
        return False

    child_pid = pending.get("child_pid")
    if child_pid and pid_is_running(child_pid):
        update_status_file(state, compute_history(load_results_rows()), "Waiting for active child run.")
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
            if has_logged_result(description=description, status=status):
                pending["logged"] = True
                pending["status"] = status
                save_json(PENDING, pending)
            else:
                try:
                    run_and_log(description, status)
                except Exception as exc:
                    update_status_file(
                        state,
                        compute_history(load_results_rows()),
                        f"Retrying result logging for {pending['description']}: {exc}",
                    )
                    time.sleep(RUN_POLL_SECONDS)
                    return True
                pending["logged"] = True
                pending["status"] = status
                save_json(PENDING, pending)
        if not pending.get("journal_written", False):
            if status == "keep":
                decision = "**keep**"
                learning = "This change improved enough to replace the kept baseline."
                next_move = "Resume exploration from the new kept baseline."
            else:
                decision = "**discard**; restored `train.py` from `last_kept_train.py`."
                learning = "This change did not improve the kept baseline enough to justify replacing it."
                next_move = "Restore the kept baseline and try the next experiment."
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
            pass  # DO NOT clear attempted_hashes; global dedup persists across keeps
        PENDING.unlink(missing_ok=True)
        update_status_file(state, compute_history(load_results_rows()), f"Reconciled: {pending['description']}")
        return True

    archive_run_log(pending)
    if not pending.get("logged", False):
        description = f"crash: {pending['description']}"
        if has_logged_result(description=description, status="crash"):
            pending["logged"] = True
            save_json(PENDING, pending)
        else:
            try:
                run_and_log(description, "crash")
            except Exception as exc:
                update_status_file(
                    state,
                    compute_history(load_results_rows()),
                    f"Retrying crash logging for {pending['description']}: {exc}",
                )
                time.sleep(RUN_POLL_SECONDS)
                return True
            pending["logged"] = True
            save_json(PENDING, pending)
    if not pending.get("journal_written", False):
        append_journal(
            hypothesis=pending["hypothesis"],
            change=pending["change"],
            result="crash / no completed summary block in `run.log`",
            decision="**crash**; restored `train.py` from `last_kept_train.py`.",
            learning="The candidate did not finish cleanly.",
            next_move="Continue from the kept baseline with the next experiment.",
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
    PENDING.unlink(missing_ok=True)
    update_status_file(state, compute_history(load_results_rows()), f"Reconciled crash: {pending['description']}")
    return True


# ---------------------------------------------------------------------------
# Launch an experiment
# ---------------------------------------------------------------------------


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

    with RUNLOG.open("w", encoding="utf-8") as log_handle:
        if proc.stdout is not None:
            for line in proc.stdout:
                log_handle.write(line)
                log_handle.flush()
        returncode = proc.wait()

    pending["child_pid"] = 0
    pending["returncode"] = returncode
    save_json(PENDING, pending)


# ---------------------------------------------------------------------------
# Initialization and main loop
# ---------------------------------------------------------------------------


def ensure_kept_snapshot() -> None:
    if KEPT.exists():
        return
    if TRAIN.exists():
        run_cmd([str(PY), str(HERE / "manage_kept.py"), "save"], check=True)


def summarize_results() -> None:
    run_cmd([str(PY), str(HERE / "summarize_results.py")], check=False)


def initialize_state() -> dict[str, Any]:
    default: dict[str, Any] = {
        "attempted_hashes": [],
        "started_at_epoch": time.time(),
        "supervisor_pid": 0,
    }
    state = load_json(STATE, default)
    state.setdefault("attempted_hashes", [])
    state.setdefault("started_at_epoch", time.time())
    state["supervisor_pid"] = os.getpid()
    return state


def check_for_existing_supervisor(state: dict[str, Any]) -> None:
    disk_state = load_json(STATE, {})
    other_pid = disk_state.get("supervisor_pid")
    if other_pid and other_pid != os.getpid() and pid_is_running(other_pid):
        raise RuntimeError(f"Another supervisor is already running (pid {other_pid}).")
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
            elapsed = time.time() - float(state.get("started_at_epoch", time.time()))
            if elapsed >= MAX_SECONDS:
                msg = f"stop: time limit {MAX_SECONDS}s. Best val_cindex={history['best_keep']:.6f}. Runs={history['completed_runs']}."
                record_stop_report(msg)
                update_status_file(state, history, msg)
                return 0
            if history["completed_runs"] >= MAX_RUNS:
                msg = f"stop: experiment limit {MAX_RUNS} reached."
                record_stop_report(msg)
                update_status_file(state, history, msg)
                return 0

            if reconcile_pending_run(state):
                save_json(STATE, state)
                continue

            summarize_results()
            rows = load_results_rows()
            history = compute_history(rows)
            base_text = read_text(KEPT)
            TRAIN.write_text(base_text, encoding="utf-8")

            if history["consecutive_discards"] >= MAX_CONSECUTIVE_DISCARDS:
                try:
                    choose_next_experiment(state, base_text)
                except RuntimeError:
                    msg = (
                        f"stop: {MAX_CONSECUTIVE_DISCARDS} consecutive discards and no fresh experiments. "
                        f"Best val_cindex={history['best_keep']:.6f}."
                    )
                    record_stop_report(msg)
                    update_status_file(state, history, msg)
                    return 0

            try:
                experiment = choose_next_experiment(state, base_text)
            except RuntimeError as exc:
                msg = f"stop: {exc}. Best val_cindex={history['best_keep']:.6f}."
                record_stop_report(msg)
                update_status_file(state, history, msg)
                return 0
            save_json(STATE, state)
            launch_experiment(state, history, experiment)
            save_json(STATE, state)
    except KeyboardInterrupt:
        msg = f"Stopped by user. Best val_cindex={compute_history(load_results_rows())['best_keep']:.6f}."
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
