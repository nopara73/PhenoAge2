"""PA2 autoresearch training script.

This is the only file the autonomous loop edits. It trains a PhenoAge 2.0
model on the frozen development split and reports validation C-index.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from prepare import (
    DEFAULT_CANDIDATE_MODEL_PATH,
    DEV_VAL_SEED,
    FEATURE_COLUMNS,
    TIME_BUDGET,
    evaluate_cindex,
    harrell_c_index,
    load_joined_rows,
    score_scripted_model,
    stratified_development_split,
    survival_arrays,
    tensorize_features,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

HIDDEN_SIZES = (16,)
DROPOUT = 0.0
LEARNING_RATE = 0.002
WEIGHT_DECAY = 3e-4
EVAL_EVERY = 500
SEED = 42
KEEP_IMPROVEMENT = 0.01


class RiskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
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
        self.network = nn.Sequential(*layers)

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        standardized = (x - self.feature_mean) / self.feature_std
        return self.network(standardized).squeeze(-1)


def cox_partial_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_scores = risk_scores[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_scores, dim=0)
    event_count = ordered_events.sum().clamp_min(1.0)
    losses = -(ordered_scores - log_risk) * ordered_events
    return losses.sum() / event_count


def save_scripted_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu())
    scripted.save(str(path))


@torch.no_grad()
def fit_standardizer_from_tensor(train_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean, std


def evaluate_saved_model_cindex(model_path: Path, rows: list[dict[str, str]]) -> float:
    scores = score_scripted_model(model_path, rows, device="cpu")
    times, events = survival_arrays(rows)
    return harrell_c_index(times, events, scores)


def main() -> None:
    t_start = time.time()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows = load_joined_rows()
    train_rows, val_rows = stratified_development_split(rows, seed=DEV_VAL_SEED)
    kept_val_cindex: float | None = None
    if DEFAULT_CANDIDATE_MODEL_PATH.exists():
        kept_val_cindex = evaluate_saved_model_cindex(DEFAULT_CANDIDATE_MODEL_PATH, val_rows)

    train_x = tensorize_features(train_rows, device)
    val_x = tensorize_features(val_rows, device)
    train_times_np, train_events_np = survival_arrays(train_rows)

    train_times = torch.tensor(train_times_np, dtype=torch.float32, device=device)
    train_events = torch.tensor(train_events_np.astype("float32"), dtype=torch.float32, device=device)

    model = RiskMLP(len(FEATURE_COLUMNS), HIDDEN_SIZES, DROPOUT).to(device)
    feature_mean, feature_std = fit_standardizer_from_tensor(train_x)
    model.set_standardizer(feature_mean, feature_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Device:            {device}")
    print(f"Feature count:     {len(FEATURE_COLUMNS)}")
    print(f"Train/val rows:    {len(train_rows)}/{len(val_rows)}")
    print(f"Time budget:       {TIME_BUDGET}s")
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
        loss = cox_partial_loss(risk_scores, train_times, train_events)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_seconds += dt

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            raise RuntimeError("Training diverged.")

        if step % EVAL_EVERY == 0:
            val_cindex = evaluate_cindex(model, val_rows, device)
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
    final_val_cindex = evaluate_cindex(model, val_rows, "cpu")
    keep_candidate = kept_val_cindex is None or final_val_cindex >= kept_val_cindex + KEEP_IMPROVEMENT
    if keep_candidate:
        save_scripted_model(model, DEFAULT_CANDIDATE_MODEL_PATH)
    t_end = time.time()
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0.0
    )
    num_params = sum(param.numel() for param in model.parameters())

    # Warm the scripted model path with a single forward pass.
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
