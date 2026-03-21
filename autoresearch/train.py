"""PA2 autoresearch training script.

This is the only file the autonomous loop edits. It trains an age-free biomarker
model on the frozen development split and reports validation C-index.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from prepare import (
    ALBUMIN_COEF,
    ALBUMIN_G_PER_DL_TO_G_PER_L,
    ALKALINE_PHOSPHATASE_COEF,
    CREATININE_COEF,
    CREATININE_MG_PER_DL_TO_UMOL_PER_L,
    CRP_MG_PER_DL_TO_MG_PER_L,
    DEFAULT_CANDIDATE_MODEL_PATH,
    DEV_VAL_SEED,
    FEATURE_COLUMNS,
    GLUCOSE_COEF,
    GLUCOSE_MG_PER_DL_TO_MMOL_PER_L,
    LOG_CRP_COEF,
    LYMPHOCYTE_PERCENT_COEF,
    MEAN_CELL_VOLUME_COEF,
    RDW_COEF,
    TIME_BUDGET,
    WBC_COEF,
    evaluate_cindex,
    load_joined_rows,
    stratified_development_split,
    survival_arrays,
    tensorize_features,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

HIDDEN_SIZES = (32, 16)
DROPOUT = 0.05
LEARNING_RATE = 0.00195
WEIGHT_DECAY = 0.0002392
EVAL_EVERY = 50
SEED = 42
EARLY_STOP_MIN_DELTA = 1e-4
EARLY_STOP_PATIENCE_EVALS = 3
EARLY_STOP_MIN_TRAIN_SECONDS = 20.0


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 16
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

        return torch.stack(
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


class RiskMLP(nn.Module):
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
        self.residual_head = nn.Sequential(*layers)
        self.base_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        base_score = encoded[:, 9]
        standardized = (encoded - self.feature_mean) / self.feature_std
        residual = self.residual_head(standardized).squeeze(-1)
        return self.base_weight * base_score + self.residual_scale * residual


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
def fit_standardizer_from_tensor(model: RiskMLP, train_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = model.encoder(train_x).detach()
    mean = encoded.mean(dim=0)
    std = encoded.std(dim=0, unbiased=False)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean, std


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

    train_x = tensorize_features(train_rows, device)
    val_x = tensorize_features(val_rows, device)
    train_times_np, train_events_np = survival_arrays(train_rows)

    train_times = torch.tensor(train_times_np, dtype=torch.float32, device=device)
    train_events = torch.tensor(train_events_np.astype("float32"), dtype=torch.float32, device=device)

    model = RiskMLP(HIDDEN_SIZES, DROPOUT).to(device)
    feature_mean, feature_std = fit_standardizer_from_tensor(model, train_x)
    model.set_standardizer(feature_mean, feature_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Device:            {device}")
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
            eval_count += 1
            val_cindex = evaluate_cindex(model, val_rows, device)
            if val_cindex > best_val_cindex + EARLY_STOP_MIN_DELTA:
                best_val_cindex = val_cindex
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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

    model.load_state_dict(best_state)
    model = model.to("cpu")
    save_scripted_model(model, DEFAULT_CANDIDATE_MODEL_PATH)

    final_val_cindex = evaluate_cindex(model, val_rows, "cpu")
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
    print(f"stop_reason:      {stop_reason}")
    print(f"artifact_path:    {DEFAULT_CANDIDATE_MODEL_PATH}")


if __name__ == "__main__":
    main()
