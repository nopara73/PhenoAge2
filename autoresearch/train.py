"""PA2 autoresearch: robust preprocessing (train-fit only) + simplicity-biased Cox head.

Winsorization quantiles, optional log1p on fixed biomarker indices, then robust (median/IQR)
or z-score scaling — all statistics fit on development train only. Head is linear or a
tiny MLP. No PhenoAge anchors or formula features.

Uses WORKER7_EXP=1..4 (not PA2_EXP) to avoid collisions with other parallel workers.

Presets:
  1 — winsor 5/95, robust IQR scale, linear head
  2 — winsor 1/99, z-score after winsor, linear head
  3 — log1p on SGP+CRP+WCP, winsor 5/95, robust scale, linear head
  4 — same preprocess as 3, MLP (9 -> 12 -> 1), tanh

Set WORKER7_SWEEP=1 to run presets 1–4 sequentially in one process; the sweep exports the
best run's checkpoint (no second retrain). Default WORKER7_EXP=4 (strongest in worker-7 sweep).

Worker 5 hybrid: PA2_PAIR_WEIGHT (default 0), PA2_NUM_PAIRS (default 512) adds a vectorized
pairwise rank term to Cox; best worker-5 combo on exp3 was weight 0.12 in a 5s budget.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from prepare import (
    DEFAULT_CANDIDATE_MODEL_PATH,
    DEV_VAL_SEED,
    FEATURE_COLUMNS,
    TIME_BUDGET,
    evaluate_cindex,
    load_joined_rows,
    stratified_development_split,
    survival_arrays,
    tensorize_features,
)

LEARNING_RATE = 0.004
WEIGHT_DECAY = 0.0012
DROPOUT = 0.08
EVAL_EVERY = 50
SEED = 42
EARLY_STOP_MIN_DELTA = float(os.environ.get("PA2_ES_MIN_DELTA", "1e-4"))
EARLY_STOP_PATIENCE_EVALS = int(os.environ.get("PA2_ES_PATIENCE", "3"))
EARLY_STOP_MIN_TRAIN_SECONDS = 20.0
HIDDEN_DIM = 12

PAIR_RANK_WEIGHT = float(os.environ.get("PA2_PAIR_WEIGHT", "0"))
NUM_PAIR_SAMPLES = int(os.environ.get("PA2_NUM_PAIRS", "512"))


def preset_config(exp_id: int) -> tuple[float, float, bool, tuple[int, ...], bool]:
    if exp_id == 1:
        return 0.05, 0.95, False, (), False
    if exp_id == 2:
        return 0.01, 0.99, True, (), False
    if exp_id == 3:
        return 0.05, 0.95, False, (2, 3, 8), False
    if exp_id == 4:
        return 0.05, 0.95, False, (2, 3, 8), True
    raise ValueError(f"WORKER7_EXP must be 1..4, got {exp_id}")


def feature_numpy(rows: list) -> np.ndarray:
    return tensorize_features(rows, "cpu").numpy().astype(np.float32, copy=False)


def fit_preprocess_stats(
    train_rows: list,
    winsor_low_q: float,
    winsor_high_q: float,
    log1p_indices: tuple[int, ...],
    use_zscore: bool,
) -> dict[str, np.ndarray]:
    x = feature_numpy(train_rows)
    if log1p_indices:
        x = x.copy()
        for j in log1p_indices:
            x[:, j] = np.log1p(np.clip(x[:, j], 0.0, None))
    wlow = np.quantile(x, winsor_low_q, axis=0).astype(np.float32)
    whigh = np.quantile(x, winsor_high_q, axis=0).astype(np.float32)
    xw = np.clip(x, wlow, whigh)
    if use_zscore:
        center = xw.mean(axis=0).astype(np.float32)
        scale = xw.std(axis=0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
    else:
        center = np.median(xw, axis=0).astype(np.float32)
        q25 = np.quantile(xw, 0.25, axis=0).astype(np.float32)
        q75 = np.quantile(xw, 0.75, axis=0).astype(np.float32)
        scale = (q75 - q25).astype(np.float32)
        scale[scale < 1e-6] = 1.0
    log_mask = np.zeros(9, dtype=np.float32)
    for j in log1p_indices:
        log_mask[j] = 1.0
    return {
        "wlow": wlow,
        "whigh": whigh,
        "center": center,
        "scale": scale,
        "log_mask": log_mask,
    }


class RobustPreprocessCox(nn.Module):
    """Raw [N,9] -> optional log1p -> winsor clip -> affine scale -> linear or small MLP."""

    def __init__(self, use_mlp: bool, hidden: int, dropout: float) -> None:
        super().__init__()
        self.register_buffer("wlow", torch.zeros(9))
        self.register_buffer("whigh", torch.zeros(9))
        self.register_buffer("center", torch.zeros(9))
        self.register_buffer("scale", torch.ones(9))
        self.register_buffer("log_mask", torch.zeros(9))
        self.use_mlp = use_mlp
        if use_mlp:
            d = dropout if dropout > 0 else 0.0
            layers: list[nn.Module] = [
                nn.Linear(9, hidden),
                nn.Tanh(),
            ]
            if d > 0:
                layers.append(nn.Dropout(d))
            layers.append(nn.Linear(hidden, 1))
            self.trunk = nn.Sequential(*layers)
        else:
            self.head = nn.Linear(9, 1, bias=True)
            self.trunk = None

    def set_preprocess(
        self,
        wlow: torch.Tensor,
        whigh: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        log_mask: torch.Tensor,
    ) -> None:
        self.wlow.copy_(wlow.view_as(self.wlow))
        self.whigh.copy_(whigh.view_as(self.whigh))
        self.center.copy_(center.view_as(self.center))
        self.scale.copy_(scale.view_as(self.scale))
        self.log_mask.copy_(log_mask.view_as(self.log_mask))

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        m = self.log_mask.view(1, -1)
        x = torch.where(m > 0.5, torch.log1p(torch.clamp(x, min=0.0)), x)
        x = torch.max(torch.min(x, self.whigh.view(1, -1)), self.wlow.view(1, -1))
        return (x - self.center.view(1, -1)) / self.scale.view(1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._transform(x)
        if self.use_mlp:
            return self.trunk(z)
        return self.head(z)

    def reset_head_init(self) -> None:
        if self.use_mlp:
            for m in self.trunk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            nn.init.normal_(self.head.weight, std=0.25)
            nn.init.zeros_(self.head.bias)


def build_model(use_mlp: bool, hidden: int, dropout: float) -> RobustPreprocessCox:
    return RobustPreprocessCox(use_mlp, hidden, dropout)


def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_scores = risk_scores[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_scores, dim=0)
    event_count = ordered_events.sum().clamp_min(1.0)
    losses = -(ordered_scores - log_risk) * ordered_events
    return losses.sum() / event_count


def pairwise_event_rank_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Event i vs uniform strictly-later time (sort order); softplus(r_j - r_i)."""
    r = risk_scores.reshape(-1)
    t = times.reshape(-1)
    e = events.reshape(-1)
    n = r.numel()
    device = r.device
    event_idx = torch.nonzero(e > 0.5, as_tuple=False).reshape(-1)
    if event_idx.numel() == 0 or num_samples <= 0:
        return r.sum() * 0.0

    order = torch.argsort(t)
    pos = torch.zeros(n, dtype=torch.long, device=device)
    pos[order] = torch.arange(n, device=device, dtype=torch.long)

    pick = torch.randint(0, event_idx.numel(), (num_samples,), device=device)
    i = event_idx[pick]
    pi = pos[i]
    gap = (n - 1) - pi
    valid = gap > 0
    if not valid.any():
        return r.sum() * 0.0
    rand_off = (torch.rand(num_samples, device=device) * gap.float()).floor().long()
    rp = (pi + 1 + rand_off).clamp(max=n - 1)
    j = order[rp]
    contrib = torch.nn.functional.softplus(r[j] - r[i])
    return contrib[valid].mean()


def save_traced_model(model: nn.Module, path_str: str, example_x: torch.Tensor) -> None:
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)
    m = model.cpu().eval()
    ex = example_x.cpu().reshape(1, 9).contiguous()
    traced = torch.jit.trace(m, ex)
    traced.save(path_str)


def train_one_experiment(
    exp_id: int,
    device: torch.device,
    train_rows: list,
    val_rows: list,
    return_export_bundle: bool,
) -> dict[str, Any]:
    wl, wh, use_z, log_ix, use_mlp = preset_config(exp_id)
    stats = fit_preprocess_stats(train_rows, wl, wh, log_ix, use_z)
    wlow = torch.tensor(stats["wlow"], dtype=torch.float32, device=device)
    whigh = torch.tensor(stats["whigh"], dtype=torch.float32, device=device)
    center = torch.tensor(stats["center"], dtype=torch.float32, device=device)
    scale = torch.tensor(stats["scale"], dtype=torch.float32, device=device)
    log_mask = torch.tensor(stats["log_mask"], dtype=torch.float32, device=device)

    train_x = tensorize_features(train_rows, device)
    val_x = tensorize_features(val_rows, device)
    train_times_np, train_events_np = survival_arrays(train_rows)
    train_times = torch.tensor(train_times_np, dtype=torch.float32, device=device)
    train_events = torch.tensor(train_events_np.astype("float32"), dtype=torch.float32, device=device)

    torch.manual_seed(SEED + exp_id)
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED + exp_id)

    model = build_model(use_mlp, HIDDEN_DIM, DROPOUT).to(device)
    model.set_preprocess(wlow, whigh, center, scale, log_mask)
    model.reset_head_init()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scale_name = "zscore" if use_z else "robust_iqr"
    log_tag = f"log{len(log_ix)}" if log_ix else "nolog"
    head_tag = f"mlp{HIDDEN_DIM}" if use_mlp else "linear"
    hyb = f"_hybpair{PAIR_RANK_WEIGHT}_k{NUM_PAIR_SAMPLES}" if PAIR_RANK_WEIGHT > 0 else ""
    arch_label = (
        f"robust_{scale_name}_w{wl:.2f}-{wh:.2f}_{log_tag}_{head_tag}_"
        f"d{DROPOUT}_worker7exp{exp_id}{hyb}"
    )
    print(f"Device:            {device}")
    print(f"Architecture:      {arch_label}")
    print(f"Feature count:     {len(FEATURE_COLUMNS)}")
    print(f"Train/val rows:    {len(train_rows)}/{len(val_rows)}")
    print(f"Time budget:       {TIME_BUDGET}s")

    best_state: dict[str, torch.Tensor] | None = None
    best_val_cindex = float("-inf")
    best_step = -1
    step = 0
    train_seconds = 0.0
    eval_count = 0
    evals_since_improvement = 0
    stop_reason = "time_budget"

    while True:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk_scores = model(train_x).reshape(-1)
        loss = cox_partial_loss(risk_scores, train_times, train_events)
        if PAIR_RANK_WEIGHT > 0.0:
            loss = loss + PAIR_RANK_WEIGHT * pairwise_event_rank_loss(
                risk_scores, train_times, train_events, NUM_PAIR_SAMPLES
            )
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.time() - t0

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            raise RuntimeError("Training diverged.")

        if step % EVAL_EVERY == 0:
            eval_count += 1
            model.eval()
            val_cindex = evaluate_cindex(model, val_rows, device)
            if val_cindex > best_val_cindex + EARLY_STOP_MIN_DELTA:
                best_val_cindex = val_cindex
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                evals_since_improvement = 0
            else:
                evals_since_improvement += 1

            remaining = max(0.0, TIME_BUDGET - train_seconds)
            print(
                f"step {step:05d} | loss: {loss.item():.6f} | "
                f"val_cindex: {val_cindex:.6f} | best: {best_val_cindex:.6f} | "
                f"remaining: {remaining:.1f}s"
            )

            if (
                train_seconds >= EARLY_STOP_MIN_TRAIN_SECONDS
                and eval_count >= EARLY_STOP_PATIENCE_EVALS + 1
                and evals_since_improvement >= EARLY_STOP_PATIENCE_EVALS
            ):
                stop_reason = "early_stop"
                break

        step += 1
        if train_seconds >= TIME_BUDGET:
            break

    if best_state is None:
        raise RuntimeError("No validation measurement was recorded.")

    num_params = sum(p.numel() for p in build_model(use_mlp, HIDDEN_DIM, DROPOUT).parameters())

    out: dict[str, Any] = {
        "exp_id": exp_id,
        "val_cindex": float(best_val_cindex),
        "training_seconds": train_seconds,
        "num_steps": step,
        "num_params": num_params,
        "best_step": best_step,
        "stop_reason": stop_reason,
        "arch_label": arch_label,
    }
    if return_export_bundle:
        out["export"] = {
            "best_state": best_state,
            "use_mlp": use_mlp,
            "wlow": wlow.cpu(),
            "whigh": whigh.cpu(),
            "center": center.cpu(),
            "scale": scale.cpu(),
            "log_mask": log_mask.cpu(),
            "example_x": val_x.cpu()[:1, :],
        }
    return out


def export_from_bundle(artifact_path: str, bundle: dict[str, Any], val_rows: list) -> float:
    use_mlp = bundle["use_mlp"]
    export_model = build_model(use_mlp, HIDDEN_DIM, DROPOUT)
    export_model.set_preprocess(
        bundle["wlow"], bundle["whigh"], bundle["center"], bundle["scale"], bundle["log_mask"]
    )
    export_model.load_state_dict(bundle["best_state"])
    save_traced_model(export_model, artifact_path, bundle["example_x"])
    return float(evaluate_cindex(torch.jit.load(artifact_path), val_rows, "cpu"))


def main() -> None:
    t_start = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows = load_joined_rows()
    train_rows, val_rows = stratified_development_split(rows, seed=DEV_VAL_SEED)
    artifact_path = str(DEFAULT_CANDIDATE_MODEL_PATH)

    sweep = os.environ.get("WORKER7_SWEEP", "").strip() == "1"
    if sweep:
        best_exp = -1
        best_c = float("-inf")
        best_bundle: dict[str, Any] | None = None
        best_metrics: dict[str, Any] | None = None
        total_train_seconds = 0.0
        for eid in (1, 2, 3, 4):
            print(f"========== WORKER7_SWEEP experiment {eid} ==========")
            m = train_one_experiment(eid, device, train_rows, val_rows, return_export_bundle=True)
            total_train_seconds += float(m["training_seconds"])
            c = float(m["val_cindex"])
            print(f"--- sweep exp{eid} val_cindex (best checkpoint): {c:.6f}")
            if c > best_c:
                best_c = c
                best_exp = eid
                best_bundle = m["export"]
                best_metrics = m
        assert best_bundle is not None and best_metrics is not None
        print(f"========== Exporting best exp {best_exp} (no retrain) ==========")
        val_cindex = export_from_bundle(artifact_path, best_bundle, val_rows)
        train_seconds = total_train_seconds
        step = int(best_metrics["num_steps"])
        best_step = int(best_metrics["best_step"])
        stop_reason = f"worker7_sweep_best_exp{best_exp}"
        num_params = int(best_metrics["num_params"])
    else:
        exp_id = int(os.environ.get("WORKER7_EXP", "4"))
        m = train_one_experiment(exp_id, device, train_rows, val_rows, return_export_bundle=True)
        assert "export" in m
        val_cindex = export_from_bundle(artifact_path, m["export"], val_rows)
        train_seconds = float(m["training_seconds"])
        step = int(m["num_steps"])
        best_step = int(m["best_step"])
        stop_reason = str(m["stop_reason"])
        num_params = int(m["num_params"])

    t_end = time.time()
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    )

    print("---")
    print(f"val_cindex:       {val_cindex:.6f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params}")
    print(f"best_step:        {best_step}")
    print(f"stop_reason:      {stop_reason}")
    print(f"artifact_path:    {artifact_path}")


if __name__ == "__main__":
    main()
