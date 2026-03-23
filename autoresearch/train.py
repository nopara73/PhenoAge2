"""BioAge subset-search training script.

This is the file the autoresearch loop edits. The benchmark contract lives in
prepare.py; this script should vary biomarker subsets and training settings
while keeping the scorer family simple and stable.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    AGE_COLUMN,
    CANDIDATE_BIOMARKER_COLUMNS,
    DEFAULT_CANDIDATE_MODEL_PATH,
    DEV_VAL_SEED,
    REFERENCE_PHENOAGE_BIOMARKERS,
    SUPERIORITY_THRESHOLD,
    TIME_BUDGET,
    fit_feature_imputer,
    harrell_c_index,
    load_joined_rows,
    save_candidate_metadata,
    score_scripted_model,
    stratified_development_split,
    survival_arrays,
    tensorize_features,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

UNIVARIATE_HIDDEN = 12
PAIR_HIDDEN = 8
DROPOUT = 0.05
LEARNING_RATE = 0.002
WEIGHT_DECAY = 2e-4
EVAL_EVERY = 20
RANKING_WEIGHT = 0.60
HARD_PAIR_BATCH = 4096
PAIR_OVERSAMPLE = 8
SEED = 42
KEEP_IMPROVEMENT = 0.0005

INCLUDE_AGE = True
SELECTED_BIOMARKERS = REFERENCE_PHENOAGE_BIOMARKERS


def selected_feature_columns() -> tuple[str, ...]:
    seen: set[str] = set()
    for biomarker in SELECTED_BIOMARKERS:
        if biomarker not in CANDIDATE_BIOMARKER_COLUMNS:
            raise ValueError(f"Unexpected biomarker in SELECTED_BIOMARKERS: {biomarker}")
        if biomarker in seen:
            raise ValueError(f"Duplicate biomarker in SELECTED_BIOMARKERS: {biomarker}")
        seen.add(biomarker)

    if not SELECTED_BIOMARKERS:
        raise ValueError("SELECTED_BIOMARKERS must contain at least one biomarker.")
    if INCLUDE_AGE:
        return (AGE_COLUMN, *SELECTED_BIOMARKERS)
    return tuple(SELECTED_BIOMARKERS)


FEATURE_COLUMNS = selected_feature_columns()
CRP_INDEX = FEATURE_COLUMNS.index("CRP") if "CRP" in FEATURE_COLUMNS else None


class FeatureNet(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PairNet(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(2, hidden_size),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SparseInteractionAdditiveRiskModel(nn.Module):
    def __init__(self, input_dim: int, univariate_hidden: int, pair_hidden: int, dropout: float):
        super().__init__()
        self.register_buffer("raw_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("raw_std", torch.ones(input_dim, dtype=torch.float32))
        self.crp_index = CRP_INDEX if CRP_INDEX is not None else -1
        self.use_age_pairs = INCLUDE_AGE

        self.feature_nets = nn.ModuleList([FeatureNet(univariate_hidden) for _ in range(input_dim)])
        pair_count = input_dim - 1 if self.use_age_pairs else 0
        self.pair_nets = nn.ModuleList([PairNet(pair_hidden, dropout) for _ in range(pair_count)])

    def set_standardizer(self, raw_mean: torch.Tensor, raw_std: torch.Tensor) -> None:
        self.raw_mean.copy_(raw_mean)
        self.raw_std.copy_(raw_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.clone()
        if self.crp_index >= 0:
            raw[:, self.crp_index] = torch.log(torch.clamp(raw[:, self.crp_index] * 10.0, min=1e-6))
        standardized = (raw - self.raw_mean) / self.raw_std

        additive = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for idx, feature_net in enumerate(self.feature_nets):
            additive = additive + feature_net(standardized[:, idx : idx + 1])

        pairwise = torch.zeros_like(additive)
        if self.use_age_pairs:
            for pair_idx, pair_net in enumerate(self.pair_nets):
                feature_idx = pair_idx + 1
                pairwise = pairwise + pair_net(standardized[:, [0, feature_idx]])

        return additive + pairwise


def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_scores = risk_scores[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_scores, dim=0)
    event_count = ordered_events.sum().clamp_min(1.0)
    losses = -(ordered_scores - log_risk) * ordered_events
    return losses.sum() / event_count


def hard_pair_ranking_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    hard_pair_batch: int,
    oversample: int,
) -> torch.Tensor:
    event_indices = torch.nonzero(events > 0.5, as_tuple=False).squeeze(1)
    if event_indices.numel() == 0:
        return risk_scores.sum() * 0.0

    sample_size = hard_pair_batch * oversample
    i = event_indices[torch.randint(event_indices.numel(), (sample_size,), device=risk_scores.device)]
    j = torch.randint(times.shape[0], (sample_size,), device=risk_scores.device)
    valid = times[i] < times[j]
    if not torch.any(valid):
        return risk_scores.sum() * 0.0

    i = i[valid]
    j = j[valid]
    margins = risk_scores[i] - risk_scores[j]
    pair_losses = F.softplus(-margins)
    if pair_losses.numel() > hard_pair_batch:
        pair_losses = torch.topk(pair_losses, k=hard_pair_batch).values
    return pair_losses.mean()


def fast_harrell_c_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    n = len(times)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    ti = times[:, None]
    tj = times[None, :]
    ei = events[:, None]
    ej = events[None, :]
    si = scores[:, None]
    sj = scores[None, :]

    tied_time_events = (ti == tj) & (ei == 1) & (ej == 1)
    i_event_first = (ei == 1) & (ti < tj)
    j_event_first = (ej == 1) & (tj < ti)
    comparable = upper & (tied_time_events | i_event_first | j_event_first)

    if not np.any(comparable):
        raise RuntimeError("No comparable pairs available for validation.")

    concordant = upper & (
        (tied_time_events & (si > sj))
        | (i_event_first & (si > sj))
        | (j_event_first & (sj > si))
    )
    tied_scores = comparable & (si == sj)
    return float((concordant.sum() + 0.5 * tied_scores.sum()) / comparable.sum())


def save_scripted_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu())
    scripted.save(str(path))


@torch.no_grad()
def fit_standardizer(train_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = train_x.clone()
    if CRP_INDEX is not None:
        raw[:, CRP_INDEX] = torch.log(torch.clamp(raw[:, CRP_INDEX] * 10.0, min=1e-6))
    raw_mean = raw.mean(dim=0)
    raw_std = raw.std(dim=0, unbiased=False)
    raw_std = torch.where(raw_std == 0.0, torch.ones_like(raw_std), raw_std)
    return raw_mean, raw_std


def evaluate_saved_model_cindex(model_path: Path, rows: list[dict[str, str]]) -> float:
    scores = score_scripted_model(model_path, rows, device="cpu")
    times, events = survival_arrays(rows)
    return harrell_c_index(times, events, scores)


@torch.no_grad()
def evaluate_cindex_fast(
    model: nn.Module,
    features: torch.Tensor,
    times: np.ndarray,
    events: np.ndarray,
) -> float:
    model.eval()
    scores = model(features).reshape(-1).detach().cpu().numpy()
    return fast_harrell_c_index(times, events, scores)


def main() -> None:
    t_start = time.time()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows = load_joined_rows()
    train_rows, val_rows = stratified_development_split(rows, seed=DEV_VAL_SEED)
    kept_val_cindex: float | None = None
    if DEFAULT_CANDIDATE_MODEL_PATH.exists():
        kept_val_cindex = evaluate_saved_model_cindex(DEFAULT_CANDIDATE_MODEL_PATH, val_rows)

    imputation_values = fit_feature_imputer(train_rows, FEATURE_COLUMNS)
    train_x = tensorize_features(train_rows, device, FEATURE_COLUMNS, imputation_values)
    val_x = tensorize_features(val_rows, device, FEATURE_COLUMNS, imputation_values)
    train_times_np, train_events_np = survival_arrays(train_rows)
    val_times_np, val_events_np = survival_arrays(val_rows)

    train_times = torch.tensor(train_times_np, dtype=torch.float32, device=device)
    train_events = torch.tensor(train_events_np.astype("float32"), dtype=torch.float32, device=device)

    model = SparseInteractionAdditiveRiskModel(
        len(FEATURE_COLUMNS),
        UNIVARIATE_HIDDEN,
        PAIR_HIDDEN,
        DROPOUT,
    ).to(device)
    raw_mean, raw_std = fit_standardizer(train_x)
    model.set_standardizer(raw_mean, raw_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Device:            {device}")
    print(f"Feature count:     {len(FEATURE_COLUMNS)}")
    print(f"Age included:      {str(INCLUDE_AGE).lower()}")
    print(f"Biomarker count:   {len(SELECTED_BIOMARKERS)}")
    print(f"Train/val rows:    {len(train_rows)}/{len(val_rows)}")
    print(f"Time budget:       {TIME_BUDGET}s")
    print(f"Pair count:        {len(model.pair_nets)}")
    print(f"Headline delta:    {SUPERIORITY_THRESHOLD:.6f}")
    print(f"Selected features: {', '.join(FEATURE_COLUMNS)}")
    if kept_val_cindex is None:
        print("Kept val_cindex:   none")
        print("Keep threshold:    first successful run")
    else:
        print(f"Kept val_cindex:   {kept_val_cindex:.6f}")
        print(f"Keep threshold:    {kept_val_cindex + KEEP_IMPROVEMENT:.6f}")

    best_state: dict[str, torch.Tensor] | None = None
    best_val_cindex = float("-inf")
    best_step = -1
    step = 0
    train_seconds = 0.0

    while True:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk_scores = model(train_x)
        cox_loss = cox_partial_loss(risk_scores, train_times, train_events)
        ranking_loss = hard_pair_ranking_loss(
            risk_scores,
            train_times,
            train_events,
            HARD_PAIR_BATCH,
            PAIR_OVERSAMPLE,
        )
        loss = cox_loss + RANKING_WEIGHT * ranking_loss
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.time() - t0

        if not torch.isfinite(loss):
            raise RuntimeError("Training diverged.")

        if step % EVAL_EVERY == 0:
            val_cindex = evaluate_cindex_fast(model, val_x, val_times_np, val_events_np)
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            remaining = max(0.0, TIME_BUDGET - train_seconds)
            print(
                f"step {step:05d} | loss: {loss.item():.6f} | "
                f"val_cindex: {val_cindex:.6f} | best: {best_val_cindex:.6f} | "
                f"remaining: {remaining:.1f}s"
            )

        step += 1
        if train_seconds >= TIME_BUDGET:
            break

    if best_state is None:
        raise RuntimeError("No validation measurement was recorded.")

    model.load_state_dict(best_state)
    model = model.to("cpu")
    final_val_cindex = evaluate_cindex_fast(model, val_x.cpu(), val_times_np, val_events_np)
    keep_candidate = kept_val_cindex is None or final_val_cindex >= kept_val_cindex + KEEP_IMPROVEMENT
    if keep_candidate:
        save_scripted_model(model, DEFAULT_CANDIDATE_MODEL_PATH)
        save_candidate_metadata(DEFAULT_CANDIDATE_MODEL_PATH, FEATURE_COLUMNS, imputation_values)
    t_end = time.time()
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0.0
    )
    num_params = sum(param.numel() for param in model.parameters())

    with torch.no_grad():
        _ = model(val_x.cpu()[: min(8, len(val_rows))])

    print("---")
    print(f"val_cindex:       {final_val_cindex:.6f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params}")
    print(f"best_step:        {best_step}")
    if kept_val_cindex is not None:
        print(f"previous_kept:    {kept_val_cindex:.6f}")
        print(f"required_cindex:  {kept_val_cindex + KEEP_IMPROVEMENT:.6f}")
    print(f"keep_candidate:   {str(keep_candidate).lower()}")
    print(f"artifact_path:    {DEFAULT_CANDIDATE_MODEL_PATH if keep_candidate else 'unchanged'}")


if __name__ == "__main__":
    main()
