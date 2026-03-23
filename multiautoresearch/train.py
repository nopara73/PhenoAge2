"""PA2 ageless restart experiment: one-sided guards with smooth immune tails.

`train.py` now defines a fixed, scripted scoring rule that:

- uses the repo-authoritative PhenoAge biomarker constants from `pheno-age-formula.md`
- replaces the raw linear RDW term with a smooth lower floor so very low RDW
  stops receiving extra reward
- replaces the raw linear WBC term with a broader smooth lower floor so very low
  WBC stops receiving extra reward without a hard kink
- caps albumin benefit above `50 g/L`
- caps lymphocyte-percent benefit above `60%` and adds a mild smooth penalty
  for higher values
- excludes chronological age entirely
- uses no learned weights, optimizer, or training loop
- emits a valid TorchScript artifact accepting raw `[N, 9]` biomarker tensors
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    DEFAULT_CANDIDATE_MODEL_PATH,
    FEATURE_COLUMNS,
    TIME_BUDGET,
    evaluate_cindex,
    get_rows_for_split,
    load_joined_rows,
    tensorize_features,
)

AMP_INDEX = 0
CEP_INDEX = 1
SGP_INDEX = 2
CRP_INDEX = 3
LMPPCNT_INDEX = 4
MVPSI_INDEX = 5
RWP_INDEX = 6
APPSI_INDEX = 7
WCP_INDEX = 8

ALBUMIN_G_PER_DL_TO_G_PER_L = 10.0
CREATININE_MG_PER_DL_TO_UMOL_PER_L = 88.4
GLUCOSE_MG_PER_DL_TO_MMOL_PER_L = 1.0 / 18.0182
CRP_MG_PER_DL_TO_MG_PER_L = 10.0

XB_INTERCEPT = -19.90667
ALBUMIN_COEF = -0.03359355
CREATININE_COEF = 0.009506491
GLUCOSE_COEF = 0.1953192
LOG_CRP_COEF = 0.09536762
LYMPHOCYTE_PERCENT_COEF = -0.01199984
MEAN_CELL_VOLUME_COEF = 0.02676401
RDW_COEF = 0.3306156
ALKALINE_PHOSPHATASE_COEF = 0.001868778
WBC_COEF = 0.05542406
MORTALITY_NUMERATOR_COEF = 1.51714
MORTALITY_DENOMINATOR = 0.007692696
PHENOAGE_LOG_COEF = 0.0055305
PHENOAGE_DENOMINATOR = 0.090165
PHENOAGE_INTERCEPT = 141.50225
MIN_POSITIVE_CRP_MG_PER_L = 1e-6
RDW_LOW_RISK_FLOOR = 11.4
RDW_FLOOR_SMOOTHING = 0.25
WBC_LOW_RISK_FLOOR = 3.5
WBC_FLOOR_SMOOTHING = 0.8
ALBUMIN_BENEFIT_CAP_G_PER_L = 50.0
LYMPHOCYTE_PERCENT_BENEFIT_CAP = 60.0
LYMPHOCYTE_UPPER_TAIL_PENALTY = 0.012
LYMPHOCYTE_UPPER_TAIL_SMOOTHING = 3.0


def smooth_floor(value: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
    """Smooth approximation to max(value, threshold)."""
    return threshold + scale * F.softplus((value - threshold) / scale)


def smooth_excess(value: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
    """Smooth approximation to max(value - threshold, 0)."""
    return scale * F.softplus((value - threshold) / scale)


class AgelessPhenoAgeScore(nn.Module):
    """Ageless PhenoAge score with simple one-sided biomarker guards and smooth tails."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        albumin_g_per_l = x[:, AMP_INDEX] * ALBUMIN_G_PER_DL_TO_G_PER_L
        creatinine_umol_per_l = x[:, CEP_INDEX] * CREATININE_MG_PER_DL_TO_UMOL_PER_L
        glucose_mmol_per_l = x[:, SGP_INDEX] * GLUCOSE_MG_PER_DL_TO_MMOL_PER_L
        crp_mg_per_l = torch.clamp(x[:, CRP_INDEX] * CRP_MG_PER_DL_TO_MG_PER_L, min=MIN_POSITIVE_CRP_MG_PER_L)

        albumin_effective = torch.clamp(albumin_g_per_l, max=ALBUMIN_BENEFIT_CAP_G_PER_L)
        lymphocyte_effective = torch.clamp(x[:, LMPPCNT_INDEX], max=LYMPHOCYTE_PERCENT_BENEFIT_CAP)
        lymphocyte_upper_tail = smooth_excess(
            x[:, LMPPCNT_INDEX],
            LYMPHOCYTE_PERCENT_BENEFIT_CAP,
            LYMPHOCYTE_UPPER_TAIL_SMOOTHING,
        )
        rdw_effective = smooth_floor(x[:, RWP_INDEX], RDW_LOW_RISK_FLOOR, RDW_FLOOR_SMOOTHING)
        wbc_effective = smooth_floor(x[:, WCP_INDEX], WBC_LOW_RISK_FLOOR, WBC_FLOOR_SMOOTHING)

        xb = (
            XB_INTERCEPT
            + ALBUMIN_COEF * albumin_effective
            + CREATININE_COEF * creatinine_umol_per_l
            + GLUCOSE_COEF * glucose_mmol_per_l
            + LOG_CRP_COEF * torch.log(crp_mg_per_l)
            + LYMPHOCYTE_PERCENT_COEF * lymphocyte_effective
            + LYMPHOCYTE_UPPER_TAIL_PENALTY * lymphocyte_upper_tail
            + MEAN_CELL_VOLUME_COEF * x[:, MVPSI_INDEX]
            + RDW_COEF * rdw_effective
            + ALKALINE_PHOSPHATASE_COEF * x[:, APPSI_INDEX]
            + WBC_COEF * wbc_effective
        )
        mortality_component = MORTALITY_NUMERATOR_COEF * torch.exp(xb) / MORTALITY_DENOMINATOR
        phenoage = PHENOAGE_INTERCEPT + torch.log(PHENOAGE_LOG_COEF * mortality_component) / PHENOAGE_DENOMINATOR
        return phenoage.unsqueeze(1)


def save_scripted_model(model: nn.Module, path: Path, example_x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(model.cpu().eval(), example_x.cpu().reshape(1, 9).contiguous())
    scripted.save(str(path))


def main() -> None:
    t_start = time.time()
    rows = load_joined_rows()
    development_rows = get_rows_for_split(rows, "development")
    artifact_path = Path(DEFAULT_CANDIDATE_MODEL_PATH)

    model = AgelessPhenoAgeScore()
    development_x = tensorize_features(development_rows, "cpu")
    save_scripted_model(model, artifact_path, development_x[:1, :])
    development_cindex = evaluate_cindex(torch.jit.load(str(artifact_path)), development_rows, "cpu")

    t_end = time.time()
    num_params = sum(param.numel() for param in model.parameters())

    print("Device:            cpu")
    print("Architecture:      ageless_phenoage_with_smooth_immune_tails")
    print(f"Feature count:     {len(FEATURE_COLUMNS)}")
    print(f"Development rows:  {len(development_rows)}")
    print(f"Time budget:       {TIME_BUDGET}s")
    print("---")
    print(f"development_cindex:{development_cindex:.6f}")
    print("training_seconds: 0.0")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print("peak_vram_mb:     0.0")
    print("num_steps:        0")
    print(f"num_params:       {num_params}")
    print("stop_reason:      smooth_immune_tail_hypothesis")
    print(f"artifact_path:    {artifact_path}")


if __name__ == "__main__":
    main()
