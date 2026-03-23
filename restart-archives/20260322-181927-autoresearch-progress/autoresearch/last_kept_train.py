"""PA2 ageless restart experiment: use constructive pairwise witnesses.

`train.py` now defines a fixed, scripted scoring rule that:

- starts from the repo-authoritative ageless PhenoAge formula
- replaces hard threshold steps with short smooth transition curves
- keeps the same burden domains but lets risk rise gradually around each cutoff
- replaces the separate severe macrocytic step with one smooth macrocytic tail
- rewrites the kidney block as atomic clauses plus one conjunction bonus
- replaces the burden-count jump with a higher-order aggregation of pairwise
  domain relations
- treats each pairwise relation as limited by its weakest supporting witness
- adds a simple extra renal-metabolic burden when both glucose and creatinine
  are above clinically familiar thresholds
- adds a simple extra burden when creatinine is high and either albumin is low
  or lymphocyte percent is low, with broader low-reserve thresholds
- adds one more point when kidney dysfunction appears alongside both low albumin
  and clearly low lymphocyte reserve
- adds one more point when kidney dysfunction appears alongside low lymphocyte
  reserve, using a slightly broader threshold
- adds one more point when lymphocyte reserve is moderately-to-clearly low while
  creatinine remains high
- excludes chronological age entirely
- uses no learned weights, optimizer, or training loop
- caps smoothed red-cell frailty and total extra burden to reduce leverage
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
RDW_HIGH_THRESHOLD = 14.0
MCV_HIGH_THRESHOLD = 95.0
MCV_VERY_HIGH_THRESHOLD = 100.0
GLUCOSE_HIGH_THRESHOLD_MMOL_PER_L = 7.0
CREATININE_HIGH_THRESHOLD_UMOL_PER_L = 110.0
ALBUMIN_LOW_THRESHOLD_G_PER_L = 38.0
LYMPHOCYTE_LOW_THRESHOLD_PERCENT = 25.0
LYMPHOCYTE_PRIORITY_THRESHOLD_PERCENT = 25.0
LYMPHOCYTE_VERY_LOW_THRESHOLD_PERCENT = 20.0
LYMPHOCYTE_SEVERE_PRIORITY_THRESHOLD_PERCENT = 22.0
MAX_HEMATOLOGIC_FRAILTY = 4.0
MAX_EXTRA_BURDEN = 6.0
MAX_PAIRWISE_DOMAIN_RELATION = 3.0
RDW_EXCESS_SCALE = 1.0
MCV_EXCESS_SCALE = 4.0
RDW_STEP_SCALE = 1.0
MCV_STEP_SCALE = 4.0
GLUCOSE_STEP_SCALE = 1.0
CREATININE_STEP_SCALE = 20.0
ALBUMIN_STEP_SCALE = 2.0
LYMPHOCYTE_LOW_STEP_SCALE = 6.0
LYMPHOCYTE_VERY_LOW_STEP_SCALE = 4.0
LYMPHOCYTE_SEVERE_STEP_SCALE = 4.0
HEMATOLOGIC_ACTIVITY_SCALE = 2.0
BURDEN_COUNT_STEP_SCALE = 1.0


def smooth_excess(value: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
    return scale * F.softplus((value - threshold) / scale)


def smooth_above(value: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
    return torch.sigmoid((value - threshold) / scale)


def smooth_below(value: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
    return torch.sigmoid((threshold - value) / scale)


def smooth_or(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - left) * (1.0 - right)


def smooth_or3(first: torch.Tensor, second: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - first) * (1.0 - second) * (1.0 - third)


def smooth_cap(value: torch.Tensor, limit: float) -> torch.Tensor:
    return limit * torch.tanh(value / limit)


class AgelessPhenoAgeScore(nn.Module):
    """Ageless PhenoAge score with simple extra burden terms."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        albumin_g_per_l = x[:, AMP_INDEX] * ALBUMIN_G_PER_DL_TO_G_PER_L
        creatinine_umol_per_l = x[:, CEP_INDEX] * CREATININE_MG_PER_DL_TO_UMOL_PER_L
        glucose_mmol_per_l = x[:, SGP_INDEX] * GLUCOSE_MG_PER_DL_TO_MMOL_PER_L
        crp_mg_per_l = torch.clamp(x[:, CRP_INDEX] * CRP_MG_PER_DL_TO_MG_PER_L, min=MIN_POSITIVE_CRP_MG_PER_L)

        xb = (
            XB_INTERCEPT
            + ALBUMIN_COEF * albumin_g_per_l
            + CREATININE_COEF * creatinine_umol_per_l
            + GLUCOSE_COEF * glucose_mmol_per_l
            + LOG_CRP_COEF * torch.log(crp_mg_per_l)
            + LYMPHOCYTE_PERCENT_COEF * x[:, LMPPCNT_INDEX]
            + MEAN_CELL_VOLUME_COEF * x[:, MVPSI_INDEX]
            + RDW_COEF * x[:, RWP_INDEX]
            + ALKALINE_PHOSPHATASE_COEF * x[:, APPSI_INDEX]
            + WBC_COEF * x[:, WCP_INDEX]
        )
        mortality_component = MORTALITY_NUMERATOR_COEF * torch.exp(xb) / MORTALITY_DENOMINATOR
        phenoage = PHENOAGE_INTERCEPT + torch.log(PHENOAGE_LOG_COEF * mortality_component) / PHENOAGE_DENOMINATOR
        rdw_excess = smooth_excess(x[:, RWP_INDEX], RDW_HIGH_THRESHOLD, RDW_EXCESS_SCALE)
        mcv_excess = smooth_excess(x[:, MVPSI_INDEX], MCV_HIGH_THRESHOLD, MCV_EXCESS_SCALE)
        hematologic_frailty_raw = rdw_excess + mcv_excess
        hematologic_frailty = smooth_cap(hematologic_frailty_raw, MAX_HEMATOLOGIC_FRAILTY)
        severe_macrocytic_frailty = (
            smooth_above(x[:, RWP_INDEX], RDW_HIGH_THRESHOLD, RDW_STEP_SCALE)
            * smooth_excess(x[:, MVPSI_INDEX], MCV_VERY_HIGH_THRESHOLD, MCV_STEP_SCALE)
        )
        renal_metabolic_burden = (
            smooth_above(glucose_mmol_per_l, GLUCOSE_HIGH_THRESHOLD_MMOL_PER_L, GLUCOSE_STEP_SCALE)
            * smooth_above(creatinine_umol_per_l, CREATININE_HIGH_THRESHOLD_UMOL_PER_L, CREATININE_STEP_SCALE)
        )
        albumin_low_signal = smooth_below(
            albumin_g_per_l,
            ALBUMIN_LOW_THRESHOLD_G_PER_L,
            ALBUMIN_STEP_SCALE,
        )
        lymphocyte_low_signal = smooth_below(
            x[:, LMPPCNT_INDEX],
            LYMPHOCYTE_LOW_THRESHOLD_PERCENT,
            LYMPHOCYTE_LOW_STEP_SCALE,
        )
        kidney_activation = smooth_above(
            creatinine_umol_per_l,
            CREATININE_HIGH_THRESHOLD_UMOL_PER_L,
            CREATININE_STEP_SCALE,
        )
        kidney_low_reserve_burden = kidney_activation * (albumin_low_signal + lymphocyte_low_signal)
        hematologic_domain_active = smooth_above(
            hematologic_frailty_raw,
            0.0,
            HEMATOLOGIC_ACTIVITY_SCALE,
        )
        kidney_domain_active = smooth_above(
            kidney_low_reserve_burden,
            0.5,
            BURDEN_COUNT_STEP_SCALE,
        )
        pairwise_domain_relations = (
            torch.minimum(hematologic_domain_active, renal_metabolic_burden),
            torch.minimum(hematologic_domain_active, kidney_domain_active),
            torch.minimum(renal_metabolic_burden, kidney_domain_active),
        )
        burden_count_overlap = smooth_cap(
            pairwise_domain_relations[0]
            + pairwise_domain_relations[1]
            + pairwise_domain_relations[2],
            MAX_PAIRWISE_DOMAIN_RELATION,
        )
        combined_kidney_low_reserve_severity = (
            albumin_low_signal
            * smooth_below(
                x[:, LMPPCNT_INDEX],
                LYMPHOCYTE_VERY_LOW_THRESHOLD_PERCENT,
                LYMPHOCYTE_VERY_LOW_STEP_SCALE,
            )
            * kidney_activation
        )
        lymphocyte_priority_kidney_burden = (
            lymphocyte_low_signal * kidney_activation
        )
        severe_lymphocyte_priority_kidney_burden = (
            smooth_below(
                x[:, LMPPCNT_INDEX],
                LYMPHOCYTE_SEVERE_PRIORITY_THRESHOLD_PERCENT,
                LYMPHOCYTE_SEVERE_STEP_SCALE,
            )
            * kidney_activation
        )
        extra_burden = (
            hematologic_frailty
            + severe_macrocytic_frailty
            + renal_metabolic_burden
            + kidney_low_reserve_burden
            + burden_count_overlap
            + combined_kidney_low_reserve_severity
            + lymphocyte_priority_kidney_burden
            + severe_lymphocyte_priority_kidney_burden
        )
        capped_extra_burden = smooth_cap(extra_burden, MAX_EXTRA_BURDEN)
        return (phenoage + capped_extra_burden).unsqueeze(1)


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
    print("Architecture:      ageless_phenoage_with_constructive_pairwise_witnesses")
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
    print("stop_reason:      simple_constructive_pairwise_witness_hypothesis")
    print(f"artifact_path:    {artifact_path}")


if __name__ == "__main__":
    main()
