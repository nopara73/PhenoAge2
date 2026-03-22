"""PA2 autoresearch restart baseline: deterministic no-ML biomarker score.

This restart intentionally removes machine learning from the search loop. `train.py` now
defines a fixed, scripted scoring rule that:

- fits only preprocessing statistics on development train
- uses no learned weights, optimizer, or training loop
- uses no PhenoAge coefficients or `pheno_no_age_xb`
- emits a valid TorchScript artifact accepting raw `[N, 9]` biomarker tensors

The baseline score is a signed sum of train-standardized biomarkers, with CEP, CRP, and SGP
receiving elementary `log1p` transforms before standardization, plus one extra red-cell
heterogeneity term from MVPSI and RWP.
"""

from __future__ import annotations

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
    load_joined_rows,
    stratified_development_split,
    tensorize_features,
)

CEP_INDEX = 1
SGP_INDEX = 2
CRP_INDEX = 3
MVPSI_INDEX = 5
RWP_INDEX = 6
Z_CLIP = 3.0
RISK_DIRECTIONS = (-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0)


class ElementaryRiskScore(nn.Module):
    """Signed clipped z-score sum plus one extra MVPSI/RWP hematologic term."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(9, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(9, dtype=torch.float32))
        self.register_buffer("risk_direction", torch.tensor(RISK_DIRECTIONS, dtype=torch.float32))

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean.view_as(self.feature_mean))
        self.feature_std.copy_(std.view_as(self.feature_std))

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y[:, CEP_INDEX] = torch.log1p(torch.clamp(y[:, CEP_INDEX], min=0.0))
        y[:, SGP_INDEX] = torch.log1p(torch.clamp(y[:, SGP_INDEX], min=0.0))
        y[:, CRP_INDEX] = torch.log1p(torch.clamp(y[:, CRP_INDEX], min=0.0))
        z = (y - self.feature_mean) / self.feature_std
        return torch.clamp(z, min=-Z_CLIP, max=Z_CLIP)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._transform(x)
        signed = z * self.risk_direction.view(1, -1)
        hematology_term = 0.5 * (signed[:, MVPSI_INDEX] + signed[:, RWP_INDEX])
        total = signed.sum(dim=1) + hematology_term
        return (total / 10.0).unsqueeze(1)


@torch.no_grad()
def fit_standardizer(train_rows: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = tensorize_features(train_rows, "cpu").clone()
    x[:, CEP_INDEX] = torch.log1p(torch.clamp(x[:, CEP_INDEX], min=0.0))
    x[:, SGP_INDEX] = torch.log1p(torch.clamp(x[:, SGP_INDEX], min=0.0))
    x[:, CRP_INDEX] = torch.log1p(torch.clamp(x[:, CRP_INDEX], min=0.0))
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def save_scripted_model(model: nn.Module, path: Path, example_x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(model.cpu().eval(), example_x.cpu().reshape(1, 9).contiguous())
    scripted.save(str(path))


def main() -> None:
    t_start = time.time()
    rows = load_joined_rows()
    train_rows, val_rows = stratified_development_split(rows, seed=DEV_VAL_SEED)
    artifact_path = Path(DEFAULT_CANDIDATE_MODEL_PATH)

    model = ElementaryRiskScore()
    feature_mean, feature_std = fit_standardizer(train_rows)
    model.set_standardizer(feature_mean, feature_std)

    val_x = tensorize_features(val_rows, "cpu")
    save_scripted_model(model, artifact_path, val_x[:1, :])
    final_val_cindex = evaluate_cindex(torch.jit.load(str(artifact_path)), val_rows, "cpu")

    t_end = time.time()
    num_params = sum(param.numel() for param in model.parameters())

    print("Device:            cpu")
    print("Architecture:      elementary_no_ml_signed_zsum_log1pcep_log1psgp_log1pcrp_plus_mvpsi_rwp")
    print(f"Feature count:     {len(FEATURE_COLUMNS)}")
    print(f"Train/val rows:    {len(train_rows)}/{len(val_rows)}")
    print(f"Time budget:       {TIME_BUDGET}s")
    print("---")
    print(f"val_cindex:       {final_val_cindex:.6f}")
    print("training_seconds: 0.0")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print("peak_vram_mb:     0.0")
    print("num_steps:        0")
    print(f"num_params:       {num_params}")
    print("best_step:        0")
    print("stop_reason:      formula_only")
    print(f"artifact_path:    {artifact_path}")


if __name__ == "__main__":
    main()
