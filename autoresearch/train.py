"""BioAge subset-search campaign runner.

`prepare.py` owns the frozen benchmark contract. This script implements the
development-only search policy that the long-run campaign should follow:

- two explicit lanes (`with_age` and `without_age`)
- lane-aware Tier A / Tier B / Tier C training budgets under the fixed 10s cap
- frontier-based subset search with biological seed families
- append-only `results.tsv` logging with parseable stage metadata
- resumable campaign state and benchmark-safe final held-out evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    AGE_COLUMN,
    CANDIDATE_BIOMARKER_COLUMNS,
    DEFAULT_CANDIDATE_MODEL_PATH,
    DEFAULT_RESULT_PATH,
    DEV_VAL_SEED,
    REFERENCE_PHENOAGE_BIOMARKERS,
    TIME_BUDGET,
    build_result_summary,
    fit_feature_imputer,
    get_rows_for_split,
    harrell_c_index,
    load_joined_rows,
    save_candidate_metadata,
    score_scripted_model,
    stratified_development_split,
    survival_arrays,
    tensorize_features,
    write_json,
)

# ---------------------------------------------------------------------------
# Fixed scorer family hyperparameters
# ---------------------------------------------------------------------------

MLP_HIDDEN = 32
DROPOUT = 0.05
LEARNING_RATE = 0.002
WEIGHT_DECAY = 2e-4
EVAL_EVERY = 20
RANKING_WEIGHT = 0.60
HARD_PAIR_BATCH = 4096
PAIR_OVERSAMPLE = 8
SEED = 42

# ---------------------------------------------------------------------------
# Campaign policy constants
# ---------------------------------------------------------------------------

TIER_ORDER = ("A", "B", "C")
LANE_ORDER = ("with_age", "without_age")
STATUS_KEEP = "keep"
STATUS_DISCARD = "discard"
STATUS_CRASH = "crash"

AUTORESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = AUTORESEARCH_DIR / "results.tsv"
CAMPAIGN_STATE_PATH = AUTORESEARCH_DIR / "bioage_campaign_state.json"
CAMPAIGN_STATUS_PATH = AUTORESEARCH_DIR / "bioage_campaign_status.json"
CAMPAIGN_SUMMARY_PATH = AUTORESEARCH_DIR / "bioage_campaign_summary.json"
FINALIST_ARTIFACT_DIR = AUTORESEARCH_DIR / "campaign_artifacts"
STATE_VERSION = 1

MAX_TOTAL_EVALUATIONS = 1_000_000
MAX_EXPANSION_ROUNDS = 1_000_000
FRONTIER_BEAM_WIDTH = 8
FINALISTS_PER_LANE = 3
STATUS_EVERY = 12
MAX_INPUT_FEATURES = 10
OVERLAP_REDUNDANCY_THRESHOLD = 0.85
NEAR_TIE_DELTA = 0.0005
TIER_A_BORDERLINE_DELTA = 0.001
TIER_A_STRONG_DELTA = 0.003
TIER_A_NEAR_BEST_DELTA = 0.002
TIER_B_PROMOTION_DELTA = 0.001
LANE_PATIENCE_EVALUATIONS = 36
GROUP_MOVE_MAX_CHANGE = 3
BANDIT_ARMS = (
    "frontier_local",
    "size_winner_local",
    "root_seed_local",
    "random_restart",
    "elite_crossover",
)
BANDIT_EPSILON = 0.20
BANDIT_STALLED_EPSILON = 0.45
BANDIT_PULLS_PER_LANE_PER_ROUND = 2
BANDIT_PARENT_LIMIT = 2
BANDIT_RANDOM_RESTARTS_PER_PULL = 3
BANDIT_CROSSOVER_CHILDREN_PER_PULL = 3
BANDIT_CROSSOVER_PARENT_POOL = 8
BANDIT_EPSILON_FLOOR = 0.08
BANDIT_RECENT_WINDOW = 24
BANDIT_WEAK_ARM_RATIO = 0.55
BANDIT_VERY_WEAK_ARM_RATIO = 0.30
BANDIT_BREAKOUT_TARGET_GAP = 0.002
BANDIT_RECENT_REWARD_VERSION = 2

LANE_REFERENCE_TARGETS = {
    "with_age": 0.876725,
    "without_age": 0.766521,
}

BIOLOGICAL_GROUPS = {
    "liver_proteins": (
        "AMP",
        "APPSI",
        "ASPSI",
        "ATPSI",
        "TBP",
        "TPP",
        "LDPSI",
    ),
    "renal_metabolic": (
        "CEP",
        "BUP",
        "UAP",
        "C1P",
        "I1P",
        "GHP",
        "SGP",
    ),
    "lipids_energy": (
        "TCP",
        "TGP",
        "HDP",
        "PSP",
    ),
    "electrolytes_minerals": (
        "C3PSI",
        "CAPSI",
        "CLPSI",
        "NAPSI",
        "SKPSI",
        "SCP",
    ),
    "iron_hematinic": (
        "FEP",
        "FRP",
        "TIP",
        "PXP",
        "FOP",
    ),
    "cbc_red_cell": (
        "HGP",
        "HTP",
        "RCP",
        "RWP",
        "MCPSI",
        "MHP",
        "MVPSI",
        "PLP",
        "PVPSI",
        "DWP",
    ),
    "immune_cells": (
        "WCP",
        "GRP",
        "GRPPCNT",
        "LMP",
        "LMPPCNT",
        "MOP",
        "MOPPCNT",
    ),
    "micronutrients": (
        "ACP",
        "BCP",
        "BXP",
        "LUP",
        "LYP",
        "VAP",
        "VCP",
        "VEP",
        "SEP",
    ),
    "inflammation_toxicology": (
        "CRP",
        "PBP",
    ),
}

INPUT_DISPLAY_NAMES = {
    "HSAGEIR": "age",
    "ACP": "alpha carotene",
    "AMP": "albumin",
    "APPSI": "alkaline phosphatase",
    "ASPSI": "aspartate aminotransferase",
    "ATPSI": "alanine aminotransferase",
    "BCP": "beta carotene",
    "BUP": "blood urea nitrogen",
    "BXP": "beta cryptoxanthin",
    "C1P": "c-peptide",
    "C3PSI": "bicarbonate",
    "CAPSI": "total calcium (SI)",
    "CEP": "creatinine",
    "CLPSI": "chloride",
    "CRP": "c-reactive protein",
    "DWP": "platelet distribution width",
    "FEP": "iron",
    "FOP": "folate",
    "FRP": "ferritin",
    "GHP": "glycated hemoglobin",
    "GRP": "granulocyte number",
    "GRPPCNT": "granulocyte percent",
    "HDP": "HDL cholesterol",
    "HGP": "hemoglobin",
    "HTP": "hematocrit",
    "I1P": "insulin",
    "LDPSI": "lactate dehydrogenase",
    "LMP": "lymphocyte number",
    "LMPPCNT": "lymphocyte percent",
    "LUP": "lutein/zeaxanthin",
    "LYP": "lycopene",
    "MCPSI": "mean cell hemoglobin",
    "MHP": "mean cell hemoglobin concentration",
    "MOP": "mononuclear number",
    "MOPPCNT": "mononuclear percent",
    "MVPSI": "mean cell volume",
    "NAPSI": "sodium",
    "PBP": "lead",
    "PLP": "platelet count",
    "PSP": "phosphorus",
    "PVPSI": "mean platelet volume",
    "PXP": "transferrin saturation",
    "RCP": "red blood cell count",
    "RWP": "red cell distribution width",
    "SCP": "total calcium",
    "SEP": "selenium",
    "SGP": "glucose",
    "SKPSI": "potassium",
    "TBP": "total bilirubin",
    "TCP": "cholesterol",
    "TGP": "triglycerides",
    "TIP": "TIBC",
    "TPP": "total protein",
    "UAP": "uric acid",
    "VAP": "vitamin A",
    "VCP": "vitamin C",
    "VEP": "vitamin E",
    "WCP": "white blood cell count",
}


@dataclass(frozen=True)
class LaneConfig:
    include_age: bool
    tier_budgets: dict[str, float]
    tier_budget_window: dict[str, tuple[float, float]]


LANE_CONFIGS = {
    "with_age": LaneConfig(
        include_age=True,
        tier_budgets={"A": 1.0, "B": 2.0, "C": float(TIME_BUDGET)},
        tier_budget_window={"A": (1.0, 1.0), "B": (1.5, 3.0), "C": (float(TIME_BUDGET), float(TIME_BUDGET))},
    ),
    "without_age": LaneConfig(
        include_age=False,
        tier_budgets={"A": 0.75, "B": 3.5, "C": float(TIME_BUDGET)},
        tier_budget_window={"A": (0.5, 1.0), "B": (3.5, 3.5), "C": (float(TIME_BUDGET), float(TIME_BUDGET))},
    ),
}


@dataclass
class DatasetBundle:
    rows: list[dict[str, str]]
    train_rows: list[dict[str, str]]
    val_rows: list[dict[str, str]]
    test_rows: list[dict[str, str]]
    train_times: torch.Tensor
    train_events: torch.Tensor
    val_times_np: np.ndarray
    val_events_np: np.ndarray
    test_times_np: np.ndarray
    test_events_np: np.ndarray
    device: torch.device


@dataclass
class TrainingResult:
    val_cindex: float
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    num_steps: int
    num_params: int
    best_step: int
    requested_budget_s: float
    effective_budget_s: float
    feature_columns: tuple[str, ...]
    biomarker_columns: tuple[str, ...]
    age_included: bool
    artifact_path: str | None = None
    cache_hit: bool = False


@dataclass
class Candidate:
    candidate_id: str
    lane: str
    biomarkers: tuple[str, ...]
    parent_id: str | None
    operator: str
    seed_family: str
    proposal_arm: str
    rationale: str
    created_round: int
    tiers: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {tier: [] for tier in TIER_ORDER}
    )
    status: str = "pending"
    locked_finalist: bool = False
    locked_winner: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "lane": self.lane,
            "biomarkers": list(self.biomarkers),
            "parent_id": self.parent_id,
            "operator": self.operator,
            "seed_family": self.seed_family,
            "proposal_arm": self.proposal_arm,
            "rationale": self.rationale,
            "created_round": self.created_round,
            "tiers": self.tiers,
            "status": self.status,
            "locked_finalist": self.locked_finalist,
            "locked_winner": self.locked_winner,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Candidate":
        candidate = cls(
            candidate_id=payload["candidate_id"],
            lane=payload["lane"],
            biomarkers=tuple(payload["biomarkers"]),
            parent_id=payload["parent_id"],
            operator=payload["operator"],
            seed_family=payload["seed_family"],
            proposal_arm=payload.get("proposal_arm", "legacy"),
            rationale=payload["rationale"],
            created_round=payload["created_round"],
        )
        candidate.tiers = {tier: list(payload["tiers"].get(tier, [])) for tier in TIER_ORDER}
        candidate.status = payload.get("status", "pending")
        candidate.locked_finalist = bool(payload.get("locked_finalist", False))
        candidate.locked_winner = bool(payload.get("locked_winner", False))
        return candidate

    def feature_columns(self) -> tuple[str, ...]:
        config = LANE_CONFIGS[self.lane]
        return selected_feature_columns(config.include_age, self.biomarkers)

    def feature_count(self) -> int:
        return len(self.biomarkers) + (1 if LANE_CONFIGS[self.lane].include_age else 0)

    def signature(self) -> str:
        return candidate_signature(self.lane, self.biomarkers)

    def scores(self, tier: str) -> list[float]:
        return [float(entry["val_cindex"]) for entry in self.tiers[tier] if entry.get("status") != STATUS_CRASH]

    def mean_score(self, tier: str) -> float | None:
        values = self.scores(tier)
        if not values:
            return None
        return float(sum(values) / len(values))

    def best_score(self, tier: str) -> float | None:
        values = self.scores(tier)
        if not values:
            return None
        return float(max(values))

    def score_dispersion(self, tier: str) -> float:
        values = self.scores(tier)
        if len(values) < 2:
            return 0.0
        return float(np.std(np.asarray(values, dtype=np.float64)))

    def artifact_path(self, tier: str = "C") -> Path | None:
        for entry in reversed(self.tiers[tier]):
            artifact_path = entry.get("artifact_path")
            if artifact_path:
                return Path(artifact_path)
        return None


@dataclass
class CampaignState:
    version: int
    next_candidate_index: int = 1
    evaluation_count: int = 0
    round_index: int = 0
    candidates: dict[str, Candidate] = field(default_factory=dict)
    frontier_ids: dict[str, list[str]] = field(default_factory=lambda: {lane: [] for lane in LANE_ORDER})
    lane_winner_ids: dict[str, str | None] = field(default_factory=lambda: {lane: None for lane in LANE_ORDER})
    overall_winner_id: str | None = None
    test_evaluated: bool = False
    initialized_seeds: bool = False
    latest_status: dict[str, Any] = field(default_factory=dict)
    bandit_stats: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: {
            lane: {arm: {"pulls": 0.0, "total_reward": 0.0} for arm in BANDIT_ARMS}
            for lane in LANE_ORDER
        }
    )
    bandit_recent_reward_version: int = BANDIT_RECENT_REWARD_VERSION
    bandit_recent_rewards: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            lane: {arm: [] for arm in BANDIT_ARMS}
            for lane in LANE_ORDER
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "next_candidate_index": self.next_candidate_index,
            "evaluation_count": self.evaluation_count,
            "round_index": self.round_index,
            "candidates": {key: candidate.to_dict() for key, candidate in self.candidates.items()},
            "frontier_ids": self.frontier_ids,
            "lane_winner_ids": self.lane_winner_ids,
            "overall_winner_id": self.overall_winner_id,
            "test_evaluated": self.test_evaluated,
            "initialized_seeds": self.initialized_seeds,
            "latest_status": self.latest_status,
            "bandit_stats": self.bandit_stats,
            "bandit_recent_reward_version": self.bandit_recent_reward_version,
            "bandit_recent_rewards": self.bandit_recent_rewards,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CampaignState":
        state = cls(version=int(payload["version"]))
        state.next_candidate_index = int(payload["next_candidate_index"])
        state.evaluation_count = int(payload["evaluation_count"])
        state.round_index = int(payload["round_index"])
        state.candidates = {
            key: Candidate.from_dict(candidate_payload)
            for key, candidate_payload in payload["candidates"].items()
        }
        state.frontier_ids = {lane: list(payload["frontier_ids"].get(lane, [])) for lane in LANE_ORDER}
        state.lane_winner_ids = {
            lane: payload["lane_winner_ids"].get(lane) for lane in LANE_ORDER
        }
        state.overall_winner_id = payload.get("overall_winner_id")
        state.test_evaluated = bool(payload.get("test_evaluated", False))
        state.initialized_seeds = bool(payload.get("initialized_seeds", False))
        state.latest_status = dict(payload.get("latest_status", {}))
        raw_bandit_stats = payload.get("bandit_stats", {})
        raw_bandit_recent_reward_version = int(
            payload.get("bandit_recent_reward_version", BANDIT_RECENT_REWARD_VERSION - 1)
        )
        raw_bandit_recent_rewards = payload.get("bandit_recent_rewards", {})
        state.bandit_stats = {
            lane: {
                arm: {
                    "pulls": float(raw_bandit_stats.get(lane, {}).get(arm, {}).get("pulls", 0.0)),
                    "total_reward": float(raw_bandit_stats.get(lane, {}).get(arm, {}).get("total_reward", 0.0)),
                }
                for arm in BANDIT_ARMS
            }
            for lane in LANE_ORDER
        }
        state.bandit_recent_reward_version = BANDIT_RECENT_REWARD_VERSION
        if raw_bandit_recent_reward_version == BANDIT_RECENT_REWARD_VERSION:
            state.bandit_recent_rewards = {
                lane: {
                    arm: [
                        float(value)
                        for value in raw_bandit_recent_rewards.get(lane, {}).get(arm, [])[-BANDIT_RECENT_WINDOW:]
                    ]
                    for arm in BANDIT_ARMS
                }
                for lane in LANE_ORDER
            }
        else:
            state.bandit_recent_rewards = {
                lane: {arm: [] for arm in BANDIT_ARMS}
                for lane in LANE_ORDER
            }
        return state


class SimpleMLPRiskModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.register_buffer("raw_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("raw_std", torch.ones(input_dim, dtype=torch.float32))
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_size), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def set_standardizer(self, raw_mean: torch.Tensor, raw_std: torch.Tensor) -> None:
        self.raw_mean.copy_(raw_mean)
        self.raw_std.copy_(raw_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        standardized = (x - self.raw_mean) / self.raw_std
        return self.net(standardized).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BioAge subset-search campaign.")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh campaign state instead of resuming.")
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=MAX_TOTAL_EVALUATIONS,
        help=f"Maximum tier evaluations to execute in this invocation (default: {MAX_TOTAL_EVALUATIONS}).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_EXPANSION_ROUNDS,
        help=f"Maximum frontier-expansion rounds to run (default: {MAX_EXPANSION_ROUNDS}).",
    )
    parser.add_argument(
        "--finalists-per-lane",
        type=int,
        default=FINALISTS_PER_LANE,
        help=f"How many Tier C finalists to keep per lane (default: {FINALISTS_PER_LANE}).",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=STATUS_EVERY,
        help=f"Write a status snapshot after this many evaluations (default: {STATUS_EVERY}).",
    )
    parser.add_argument(
        "--no-results-log",
        action="store_true",
        help="Do not append rows to results.tsv. Useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Stop after locking the validation winner without touching the held-out test split.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny validation-only campaign for implementation checks.",
    )
    return parser.parse_args()


def validate_biological_groups() -> None:
    assigned = {biomarker for group in BIOLOGICAL_GROUPS.values() for biomarker in group}
    expected = set(CANDIDATE_BIOMARKER_COLUMNS)
    if assigned != expected:
        missing = sorted(expected - assigned)
        extra = sorted(assigned - expected)
        raise ValueError(
            f"BIOLOGICAL_GROUPS must partition the biomarker pool. Missing={missing}, extra={extra}"
        )


def candidate_signature(lane: str, biomarkers: tuple[str, ...] | list[str]) -> str:
    return f"{lane}::{';'.join(sorted(biomarkers))}"


def new_candidate_id(state: CampaignState) -> str:
    candidate_id = f"cand_{state.next_candidate_index:05d}"
    state.next_candidate_index += 1
    return candidate_id


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def size_band(candidate: Candidate) -> str:
    size = len(candidate.biomarkers)
    if size <= 4:
        return "s"
    if size <= 8:
        return "m"
    if size <= 12:
        return "l"
    return "xl"


def jaccard_overlap(a: Candidate, b: Candidate) -> float:
    set_a = set(a.biomarkers)
    set_b = set(b.biomarkers)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def max_biomarker_count(include_age: bool) -> int:
    return MAX_INPUT_FEATURES - (1 if include_age else 0)


def biomarkers_within_feature_cap(include_age: bool, biomarkers: tuple[str, ...] | list[str]) -> bool:
    return len(tuple(biomarkers)) <= max_biomarker_count(include_age)


def candidate_within_feature_cap(candidate: Candidate) -> bool:
    return candidate.feature_count() <= MAX_INPUT_FEATURES


def selected_feature_columns(include_age: bool, biomarkers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    selected = tuple(sorted(biomarkers))
    if not selected:
        raise ValueError("At least one biomarker must be selected.")
    if not biomarkers_within_feature_cap(include_age, selected):
        raise ValueError(
            f"Selected inputs exceed the search cap of {MAX_INPUT_FEATURES} total features."
        )
    for biomarker in selected:
        if biomarker not in CANDIDATE_BIOMARKER_COLUMNS:
            raise ValueError(f"Unexpected biomarker: {biomarker}")
        if biomarker in seen:
            raise ValueError(f"Duplicate biomarker: {biomarker}")
        seen.add(biomarker)
    if include_age:
        return (AGE_COLUMN, *selected)
    return selected


def parse_description_fields(description: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in description.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def result_cache_key(
    lane: str,
    biomarkers: tuple[str, ...] | list[str],
    tier: str,
    requested_budget_s: float,
) -> tuple[str, tuple[str, ...], str, str]:
    return (
        lane,
        tuple(sorted(biomarkers)),
        tier,
        f"{float(requested_budget_s):.3f}",
    )


def load_result_cache() -> dict[tuple[str, tuple[str, ...], str, str], TrainingResult]:
    if not RESULTS_PATH.exists():
        return {}

    cache: dict[tuple[str, tuple[str, ...], str, str], TrainingResult] = {}
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row or row.get("status") == STATUS_CRASH:
                continue
            description_fields = parse_description_fields(row.get("description", ""))
            lane = description_fields.get("lane")
            tier = description_fields.get("tier")
            requested_budget_raw = description_fields.get("requested_budget_s")
            actual_training_raw = description_fields.get("actual_training_s")
            if lane not in LANE_ORDER or tier not in TIER_ORDER:
                continue
            if requested_budget_raw is None or actual_training_raw is None:
                continue

            selected_raw = row.get("selected_biomarkers", "")
            biomarkers = tuple(sorted(value for value in selected_raw.split(";") if value))
            if not biomarkers:
                continue
            if not biomarkers_within_feature_cap(LANE_CONFIGS[lane].include_age, biomarkers):
                continue

            try:
                requested_budget_s = float(requested_budget_raw)
                actual_training_s = float(actual_training_raw)
                val_cindex = float(row["val_cindex"])
                peak_vram_mb = float(row["memory_gb"]) * 1024.0
            except (KeyError, TypeError, ValueError):
                continue

            feature_columns = selected_feature_columns(LANE_CONFIGS[lane].include_age, biomarkers)
            cache[result_cache_key(lane, biomarkers, tier, requested_budget_s)] = TrainingResult(
                val_cindex=val_cindex,
                training_seconds=actual_training_s,
                total_seconds=actual_training_s,
                peak_vram_mb=peak_vram_mb,
                num_steps=0,
                num_params=0,
                best_step=-1,
                requested_budget_s=requested_budget_s,
                effective_budget_s=requested_budget_s,
                feature_columns=feature_columns,
                biomarker_columns=biomarkers,
                age_included=LANE_CONFIGS[lane].include_age,
                artifact_path=None,
                cache_hit=True,
            )
    return cache


def max_result_candidate_index() -> int:
    if not RESULTS_PATH.exists():
        return 0

    max_index = 0
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            description_fields = parse_description_fields(row.get("description", ""))
            candidate_id = description_fields.get("candidate_id")
            if candidate_id is None or not candidate_id.startswith("cand_"):
                continue
            try:
                max_index = max(max_index, int(candidate_id.split("_", 1)[1]))
            except ValueError:
                continue
    return max_index


def recover_state_from_results() -> CampaignState:
    state = CampaignState(version=STATE_VERSION)
    if not RESULTS_PATH.exists():
        return state

    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            description_fields = parse_description_fields(row.get("description", ""))
            candidate_id = description_fields.get("candidate_id")
            lane = description_fields.get("lane")
            tier = description_fields.get("tier")
            if candidate_id is None or lane not in LANE_ORDER or tier not in TIER_ORDER:
                continue

            biomarkers = tuple(sorted(value for value in row.get("selected_biomarkers", "").split(";") if value))
            if not biomarkers or not biomarkers_within_feature_cap(LANE_CONFIGS[lane].include_age, biomarkers):
                continue

            parent_id = description_fields.get("parent_id")
            if parent_id in {None, "", "root"}:
                parent_id = None

            candidate = state.candidates.get(candidate_id)
            if candidate is None:
                candidate = Candidate(
                    candidate_id=candidate_id,
                    lane=lane,
                    biomarkers=biomarkers,
                    parent_id=parent_id,
                    operator=description_fields.get("operator", "recovered"),
                    seed_family=description_fields.get("seed_family", "recovered"),
                    proposal_arm=description_fields.get("proposal_arm", "legacy"),
                    rationale="Recovered from results log.",
                    created_round=0,
                )
                state.candidates[candidate_id] = candidate

            try:
                val_cindex = float(row.get("val_cindex", "0.0"))
                peak_vram_mb = float(row.get("memory_gb", "0.0")) * 1024.0
                requested_budget_s = float(description_fields.get("requested_budget_s", "0.0"))
                actual_training_s = float(description_fields.get("actual_training_s", "0.0"))
            except ValueError:
                continue

            candidate.tiers[tier].append(
                {
                    "timestamp": now_utc(),
                    "status": row.get("status", STATUS_KEEP),
                    "promotion": description_fields.get("promotion", "recovered"),
                    "requested_budget_s": requested_budget_s,
                    "actual_training_s": actual_training_s,
                    "val_cindex": val_cindex,
                    "peak_vram_mb": peak_vram_mb,
                    "num_steps": 0,
                    "num_params": 0,
                    "best_step": -1,
                    "artifact_path": None,
                    "cache_hit": str(description_fields.get("cache_hit", "false")).lower() == "true",
                    "error": description_fields.get("error"),
                }
            )
            state.evaluation_count += 1

    state.next_candidate_index = max_result_candidate_index() + 1
    state.initialized_seeds = False
    for lane in LANE_ORDER:
        refresh_frontier(state, lane)
        winner = choose_lane_winner(state, lane)
        state.lane_winner_ids[lane] = None if winner is None else winner.candidate_id
    overall_winner = choose_overall_winner(state)
    state.overall_winner_id = None if overall_winner is None else overall_winner.candidate_id
    return state


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
    raw_mean = train_x.mean(dim=0)
    raw_std = train_x.std(dim=0, unbiased=False)
    raw_std = torch.where(raw_std == 0.0, torch.ones_like(raw_std), raw_std)
    return raw_mean, raw_std


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


def build_dataset_bundle() -> DatasetBundle:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows = load_joined_rows()
    train_rows, val_rows = stratified_development_split(rows, seed=DEV_VAL_SEED)
    test_rows = get_rows_for_split(rows, "test")
    train_times_np, train_events_np = survival_arrays(train_rows)
    val_times_np, val_events_np = survival_arrays(val_rows)
    test_times_np, test_events_np = survival_arrays(test_rows)
    return DatasetBundle(
        rows=rows,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        train_times=torch.tensor(train_times_np, dtype=torch.float32, device=device),
        train_events=torch.tensor(train_events_np.astype("float32"), dtype=torch.float32, device=device),
        val_times_np=val_times_np,
        val_events_np=val_events_np,
        test_times_np=test_times_np,
        test_events_np=test_events_np,
        device=device,
    )


def train_subset(
    dataset: DatasetBundle,
    candidate: Candidate,
    tier: str,
    requested_budget_s: float,
    *,
    save_path: Path | None = None,
    verbose: bool = False,
) -> TrainingResult:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    effective_budget_s = min(float(requested_budget_s), float(TIME_BUDGET))
    feature_columns = candidate.feature_columns()
    imputation_values = fit_feature_imputer(dataset.train_rows, feature_columns)
    train_x = tensorize_features(dataset.train_rows, dataset.device, feature_columns, imputation_values)
    val_x = tensorize_features(dataset.val_rows, dataset.device, feature_columns, imputation_values)

    model = SimpleMLPRiskModel(len(feature_columns), MLP_HIDDEN, DROPOUT).to(dataset.device)
    raw_mean, raw_std = fit_standardizer(train_x)
    model.set_standardizer(raw_mean, raw_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_cindex = float("-inf")
    best_step = -1
    step = 0
    train_seconds = 0.0

    while train_seconds < effective_budget_s:
        if dataset.device.type == "cuda":
            torch.cuda.synchronize()
        step_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk_scores = model(train_x)
        cox_loss = cox_partial_loss(risk_scores, dataset.train_times, dataset.train_events)
        ranking_loss = hard_pair_ranking_loss(
            risk_scores,
            dataset.train_times,
            dataset.train_events,
            HARD_PAIR_BATCH,
            PAIR_OVERSAMPLE,
        )
        loss = cox_loss + RANKING_WEIGHT * ranking_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Training diverged.")
        loss.backward()
        optimizer.step()

        if dataset.device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.time() - step_start

        if step % EVAL_EVERY == 0:
            val_cindex = evaluate_cindex_fast(model, val_x, dataset.val_times_np, dataset.val_events_np)
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if verbose:
                remaining = max(0.0, effective_budget_s - train_seconds)
                print(
                    f"{candidate.candidate_id} tier={tier} step {step:05d} "
                    f"loss={loss.item():.6f} val_cindex={val_cindex:.6f} "
                    f"best={best_val_cindex:.6f} remaining={remaining:.2f}s"
                )
        step += 1

    if best_state is None:
        raise RuntimeError("No validation measurement was recorded.")

    model.load_state_dict(best_state)
    model_cpu = model.to("cpu")
    final_val_cindex = evaluate_cindex_fast(model_cpu, val_x.cpu(), dataset.val_times_np, dataset.val_events_np)
    if save_path is not None:
        save_scripted_model(model_cpu, save_path)
        save_candidate_metadata(save_path, feature_columns, imputation_values)

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    num_params = sum(param.numel() for param in model_cpu.parameters())
    return TrainingResult(
        val_cindex=float(final_val_cindex),
        training_seconds=float(train_seconds),
        total_seconds=float(t_end - t_start),
        peak_vram_mb=float(peak_vram_mb),
        num_steps=step,
        num_params=num_params,
        best_step=best_step,
        requested_budget_s=float(requested_budget_s),
        effective_budget_s=float(effective_budget_s),
        feature_columns=feature_columns,
        biomarker_columns=tuple(candidate.biomarkers),
        age_included=LANE_CONFIGS[candidate.lane].include_age,
        artifact_path=str(save_path) if save_path is not None else None,
    )


def load_or_create_state(fresh: bool) -> CampaignState:
    if fresh or not CAMPAIGN_STATE_PATH.exists():
        state = CampaignState(version=STATE_VERSION)
        state.next_candidate_index = max(state.next_candidate_index, max_result_candidate_index() + 1)
        return state
    payload = json.loads(CAMPAIGN_STATE_PATH.read_text(encoding="utf-8"))
    state = CampaignState.from_dict(payload)
    if state.version != STATE_VERSION:
        raise RuntimeError(
            f"Unsupported campaign state version {state.version}; expected {STATE_VERSION}."
        )
    state.next_candidate_index = max(state.next_candidate_index, max_result_candidate_index() + 1)
    return state


def save_state(state: CampaignState) -> None:
    CAMPAIGN_STATE_PATH.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")


def ensure_results_header() -> None:
    if RESULTS_PATH.exists():
        return
    RESULTS_PATH.write_text(
        "commit\tval_cindex\tmemory_gb\tstatus\tfeature_count\tselected_biomarkers\tdescription\n",
        encoding="utf-8",
    )


def append_result_row(
    commit_hash: str,
    candidate: Candidate,
    tier: str,
    status: str,
    promotion: str,
    requested_budget_s: float,
    actual_training_s: float,
    val_cindex: float,
    peak_vram_mb: float,
    *,
    write_results: bool,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    if not write_results:
        return

    ensure_results_header()
    description_fields = {
        "candidate_id": candidate.candidate_id,
        "lane": candidate.lane,
        "tier": tier,
        "requested_budget_s": f"{requested_budget_s:.3f}",
        "actual_training_s": f"{actual_training_s:.3f}",
        "promotion": promotion,
        "parent_id": candidate.parent_id or "root",
        "operator": candidate.operator,
        "seed_family": candidate.seed_family,
        "subset_size": str(len(candidate.biomarkers)),
    }
    if extra_fields:
        for key, value in extra_fields.items():
            description_fields[key] = str(value)
    description = " ".join(f"{key}={value}" for key, value in description_fields.items())
    memory_gb = peak_vram_mb / 1024.0
    selected = ";".join(candidate.biomarkers)
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                commit_hash,
                f"{val_cindex:.6f}",
                f"{memory_gb:.2f}",
                status,
                candidate.feature_count(),
                selected,
                description,
            ]
        )


def format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def format_inputs(candidate: Candidate) -> str:
    return ", ".join(INPUT_DISPLAY_NAMES.get(column, column) for column in candidate.feature_columns())


def print_evaluation_report(
    state: CampaignState,
    candidate: Candidate,
    tier: str,
    status: str,
    promotion: str,
    result: TrainingResult | None,
    *,
    error: str | None = None,
) -> None:
    print(
        f"c-index: {format_score(None if result is None else result.val_cindex)}, "
        f"subsets: {candidate.feature_count()}, "
        f"biomarkers: {format_inputs(candidate)}"
    )
    if error is not None:
        print(
            f"c-index: crash, "
            f"subsets: {candidate.feature_count()}, "
            f"biomarkers: {format_inputs(candidate)}"
        )


def record_evaluation(
    state: CampaignState,
    candidate: Candidate,
    tier: str,
    status: str,
    promotion: str,
    result: TrainingResult | None,
    *,
    error: str | None = None,
    write_results: bool,
    commit_hash: str,
) -> None:
    entry = {
        "timestamp": now_utc(),
        "status": status,
        "promotion": promotion,
        "requested_budget_s": None if result is None else result.requested_budget_s,
        "actual_training_s": 0.0 if result is None else result.training_seconds,
        "val_cindex": 0.0 if result is None else result.val_cindex,
        "peak_vram_mb": 0.0 if result is None else result.peak_vram_mb,
        "num_steps": 0 if result is None else result.num_steps,
        "num_params": 0 if result is None else result.num_params,
        "best_step": -1 if result is None else result.best_step,
        "artifact_path": None if result is None else result.artifact_path,
        "cache_hit": False if result is None else result.cache_hit,
        "error": error,
    }
    candidate.tiers[tier].append(entry)
    state.evaluation_count += 1
    print_evaluation_report(
        state,
        candidate,
        tier,
        status,
        promotion,
        result,
        error=error,
    )
    append_result_row(
        commit_hash=commit_hash,
        candidate=candidate,
        tier=tier,
        status=status,
        promotion=promotion,
        requested_budget_s=0.0 if result is None else result.requested_budget_s,
        actual_training_s=0.0 if result is None else result.training_seconds,
        val_cindex=0.0 if result is None else result.val_cindex,
        peak_vram_mb=0.0 if result is None else result.peak_vram_mb,
        write_results=write_results,
        extra_fields={
            "error": error or "none",
            "cache_hit": "false" if result is None else str(result.cache_hit).lower(),
            "proposal_arm": candidate.proposal_arm,
        },
    )


def budget_failure(result: TrainingResult) -> bool:
    overshoot_tolerance = max(0.15, result.requested_budget_s * 0.15)
    return result.training_seconds > result.requested_budget_s + overshoot_tolerance


def lane_best_score(
    state: CampaignState,
    lane: str,
    tier: str,
    *,
    exclude_candidate_id: str | None = None,
) -> float | None:
    values = [
        candidate.mean_score(tier)
        for candidate in state.candidates.values()
        if candidate.lane == lane
        and candidate_within_feature_cap(candidate)
        and candidate.candidate_id != exclude_candidate_id
        and candidate.mean_score(tier) is not None
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def candidate_score(
    candidate: Candidate,
    tier: str,
    provisional_scores: dict[str, float] | None = None,
) -> float:
    if provisional_scores is not None and candidate.candidate_id in provisional_scores:
        return float(provisional_scores[candidate.candidate_id])
    score = candidate.mean_score(tier)
    if score is None:
        return float("-inf")
    return float(score)


def preferred_candidate(
    a: Candidate,
    b: Candidate,
    tier: str,
    provisional_scores: dict[str, float] | None = None,
) -> Candidate:
    score_a = candidate_score(a, tier, provisional_scores)
    score_b = candidate_score(b, tier, provisional_scores)
    if score_a > score_b + NEAR_TIE_DELTA:
        return a
    if score_b > score_a + NEAR_TIE_DELTA:
        return b
    if a.feature_count() < b.feature_count():
        return a
    if b.feature_count() < a.feature_count():
        return b
    if a.score_dispersion(tier) < b.score_dispersion(tier):
        return a
    return a if a.candidate_id < b.candidate_id else b


def current_frontier(state: CampaignState, lane: str) -> list[Candidate]:
    return [
        state.candidates[candidate_id]
        for candidate_id in state.frontier_ids[lane]
        if candidate_id in state.candidates
        and candidate_within_feature_cap(state.candidates[candidate_id])
    ]


def refresh_frontier(state: CampaignState, lane: str) -> None:
    confirmed = [
        candidate
        for candidate in state.candidates.values()
        if candidate.lane == lane
        and candidate_within_feature_cap(candidate)
        and candidate.mean_score("B") is not None
        and candidate.status != "crashed"
    ]
    ranked = sorted(
        confirmed,
        key=lambda candidate: (
            -(candidate.mean_score("B") or float("-inf")),
            candidate.feature_count(),
            candidate.score_dispersion("B"),
        ),
    )

    frontier: list[Candidate] = []
    band_counts: dict[str, int] = defaultdict(int)
    for candidate in ranked:
        band = size_band(candidate)
        redundant = False
        for kept in frontier:
            if jaccard_overlap(candidate, kept) >= OVERLAP_REDUNDANCY_THRESHOLD:
                if preferred_candidate(kept, candidate, "B") is kept:
                    redundant = True
                    break
        if redundant:
            continue
        if band_counts[band] == 0 or len(frontier) < FRONTIER_BEAM_WIDTH:
            frontier.append(candidate)
            band_counts[band] += 1
        if len(frontier) >= FRONTIER_BEAM_WIDTH:
            break

    state.frontier_ids[lane] = [candidate.candidate_id for candidate in frontier]
    frontier_ids = set(state.frontier_ids[lane])
    for candidate in state.candidates.values():
        if candidate.lane != lane or candidate.status == "crashed":
            continue
        if not candidate_within_feature_cap(candidate):
            candidate.status = "discarded"
            continue
        if candidate.candidate_id in frontier_ids:
            candidate.status = "frontier"
        elif candidate.mean_score("B") is not None:
            candidate.status = "confirmed"


def choose_lane_winner(state: CampaignState, lane: str) -> Candidate | None:
    finalists = [
        candidate
        for candidate in state.candidates.values()
        if candidate.lane == lane and candidate_within_feature_cap(candidate) and candidate.mean_score("C") is not None
    ]
    if not finalists:
        finalists = current_frontier(state, lane)
    if not finalists:
        finalists = [
            candidate
            for candidate in state.candidates.values()
            if candidate.lane == lane and candidate_within_feature_cap(candidate) and candidate.mean_score("A") is not None
        ]
    if not finalists:
        return None
    winner = finalists[0]
    winner_tier = "C" if winner.mean_score("C") is not None else ("B" if winner.mean_score("B") is not None else "A")
    for candidate in finalists[1:]:
        tier = "C" if candidate.mean_score("C") is not None else ("B" if candidate.mean_score("B") is not None else "A")
        compare_tier = "C" if winner_tier == "C" and tier == "C" else "B" if winner.mean_score("B") is not None and candidate.mean_score("B") is not None else "A"
        winner = preferred_candidate(winner, candidate, compare_tier)
        winner_tier = compare_tier
    return winner


def choose_overall_winner(state: CampaignState) -> Candidate | None:
    lane_winners = [
        state.candidates[candidate_id]
        for candidate_id in state.lane_winner_ids.values()
        if candidate_id is not None
    ]
    if not lane_winners:
        return None
    winner = lane_winners[0]
    compare_tier = "C" if winner.mean_score("C") is not None else "B"
    for candidate in lane_winners[1:]:
        winner = preferred_candidate(winner, candidate, compare_tier)
    return winner


def build_seed_specs(rng: random.Random) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    specs.append(
        {
            "biomarkers": tuple(sorted(REFERENCE_PHENOAGE_BIOMARKERS)),
            "seed_family": "phenoage9",
            "operator": "seed",
            "rationale": "Reference PhenoAge-9 subset.",
        }
    )
    for biomarker in sorted(CANDIDATE_BIOMARKER_COLUMNS):
        specs.append(
            {
                "biomarkers": (biomarker,),
                "seed_family": "single_marker",
                "operator": "seed",
                "rationale": f"Single-marker screen for {biomarker}.",
            }
        )
    for group_name, biomarkers in BIOLOGICAL_GROUPS.items():
        specs.append(
            {
                "biomarkers": tuple(sorted(biomarkers)),
                "seed_family": "group",
                "operator": "seed",
                "rationale": f"Biological system seed: {group_name}.",
            }
        )
    for subset_size in (3, 5, 8, 10):
        for sample_index in range(3):
            sampled = tuple(sorted(rng.sample(list(CANDIDATE_BIOMARKER_COLUMNS), subset_size)))
            specs.append(
                {
                    "biomarkers": sampled,
                    "seed_family": "random_sparse",
                    "operator": "seed",
                    "rationale": f"Random sparse seed size={subset_size} sample={sample_index + 1}.",
                }
            )
    return specs


def build_pending_seed_queue(state: CampaignState, alternate_lanes: bool) -> list[Candidate]:
    pending = [
        candidate
        for candidate in state.candidates.values()
        if candidate.parent_id is None and candidate.mean_score("A") is None
    ]
    if not alternate_lanes:
        return sorted(
            pending,
            key=lambda candidate: (
                candidate.lane,
                candidate.seed_family,
                len(candidate.biomarkers),
                candidate.candidate_id,
            ),
        )

    lane_buckets: dict[str, list[Candidate]] = {}
    for lane in LANE_ORDER:
        lane_buckets[lane] = sorted(
            [candidate for candidate in pending if candidate.lane == lane],
            key=lambda candidate: (
                candidate.seed_family,
                len(candidate.biomarkers),
                candidate.candidate_id,
            ),
        )

    interleaved: list[Candidate] = []
    while any(lane_buckets[lane] for lane in LANE_ORDER):
        for lane in LANE_ORDER:
            if lane_buckets[lane]:
                interleaved.append(lane_buckets[lane].pop(0))
    return interleaved


def initialize_seeds(state: CampaignState, rng: random.Random) -> None:
    if state.initialized_seeds:
        return
    seen_signatures = {candidate.signature() for candidate in state.candidates.values()}
    for lane in LANE_ORDER:
        include_age = LANE_CONFIGS[lane].include_age
        for spec in build_seed_specs(rng):
            if not biomarkers_within_feature_cap(include_age, spec["biomarkers"]):
                continue
            signature = candidate_signature(lane, spec["biomarkers"])
            if signature in seen_signatures:
                continue
            candidate = Candidate(
                candidate_id=new_candidate_id(state),
                lane=lane,
                biomarkers=tuple(sorted(spec["biomarkers"])),
                parent_id=None,
                operator=spec["operator"],
                seed_family=spec["seed_family"],
                proposal_arm="seed_init",
                rationale=spec["rationale"],
                created_round=0,
            )
            state.candidates[candidate.candidate_id] = candidate
            seen_signatures.add(signature)
    state.initialized_seeds = True


def candidate_needs_tier(candidate: Candidate, tier: str) -> bool:
    return len(candidate.tiers[tier]) == 0


def best_candidate_for_feature_count(
    state: CampaignState,
    lane: str,
    feature_count: int,
) -> Candidate | None:
    contenders = [
        candidate
        for candidate in state.candidates.values()
        if candidate.lane == lane
        and candidate.feature_count() == feature_count
        and candidate_within_feature_cap(candidate)
        and (
            candidate.mean_score("C") is not None
            or candidate.mean_score("B") is not None
            or candidate.mean_score("A") is not None
        )
    ]
    if not contenders:
        return None
    winner = contenders[0]
    for candidate in contenders[1:]:
        compare_tier = "C" if winner.mean_score("C") is not None and candidate.mean_score("C") is not None else (
            "B" if winner.mean_score("B") is not None and candidate.mean_score("B") is not None else "A"
        )
        winner = preferred_candidate(winner, candidate, compare_tier)
    return winner


def feature_count_best_score(
    state: CampaignState,
    lane: str,
    tier: str,
    feature_count: int,
    *,
    exclude_candidate_id: str | None = None,
) -> float | None:
    values = [
        candidate.mean_score(tier)
        for candidate in state.candidates.values()
        if candidate.lane == lane
        and candidate_within_feature_cap(candidate)
        and candidate.feature_count() == feature_count
        and candidate.candidate_id != exclude_candidate_id
        and candidate.mean_score(tier) is not None
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def tier_budget(candidate: Candidate, tier: str) -> float:
    return LANE_CONFIGS[candidate.lane].tier_budgets[tier]


def evaluate_candidate_tier(
    dataset: DatasetBundle,
    state: CampaignState,
    candidate: Candidate,
    tier: str,
    *,
    result_cache: dict[tuple[str, tuple[str, ...], str, str], TrainingResult],
    write_results: bool,
    commit_hash: str,
    persist_artifact: bool = False,
) -> TrainingResult | None:
    requested_budget_s = tier_budget(candidate, tier)
    if not persist_artifact:
        cached_result = result_cache.get(result_cache_key(candidate.lane, candidate.biomarkers, tier, requested_budget_s))
        if cached_result is not None:
            return cached_result
    artifact_path: Path | None = None
    if persist_artifact:
        artifact_path = FINALIST_ARTIFACT_DIR / f"{candidate.candidate_id}.pt"
    try:
        result = train_subset(
            dataset,
            candidate,
            tier,
            requested_budget_s,
            save_path=artifact_path,
        )
        if budget_failure(result):
            raise RuntimeError(
                f"Budget-control failure: actual {result.training_seconds:.3f}s exceeded request {requested_budget_s:.3f}s."
            )
        return result
    except Exception as exc:
        record_evaluation(
            state,
            candidate,
            tier,
            STATUS_CRASH,
            "crash",
            None,
            error=str(exc),
            write_results=write_results,
            commit_hash=commit_hash,
        )
        candidate.status = "crashed"
        return None


def tier_a_reference_score(state: CampaignState, candidate: Candidate) -> float | None:
    scores: list[float] = []
    if candidate.parent_id is not None and candidate.parent_id in state.candidates:
        parent_score = state.candidates[candidate.parent_id].mean_score("A")
        if parent_score is not None:
            scores.append(parent_score)
    lane_score = lane_best_score(
        state,
        candidate.lane,
        "A",
        exclude_candidate_id=candidate.candidate_id,
    )
    if lane_score is not None:
        scores.append(lane_score)
    if not scores:
        return None
    return max(scores)


def tier_a_parent_score(state: CampaignState, candidate: Candidate) -> float | None:
    if candidate.parent_id is None or candidate.parent_id not in state.candidates:
        return None
    return state.candidates[candidate.parent_id].mean_score("A")


def tier_a_lane_best_score(state: CampaignState, candidate: Candidate) -> float | None:
    return lane_best_score(
        state,
        candidate.lane,
        "A",
        exclude_candidate_id=candidate.candidate_id,
    )


def tier_b_lane_best_score(state: CampaignState, candidate: Candidate) -> float | None:
    return lane_best_score(
        state,
        candidate.lane,
        "B",
        exclude_candidate_id=candidate.candidate_id,
    )


def tier_a_close_to_lane_frontier(
    score: float,
    lane_best_a: float | None,
    lane_best_b: float | None,
    size_best_a: float | None,
    size_best_b: float | None,
) -> bool:
    return (
        lane_best_a is not None and score >= lane_best_a - TIER_A_NEAR_BEST_DELTA
    ) or (
        lane_best_b is not None and score >= lane_best_b - TIER_A_NEAR_BEST_DELTA
    ) or (
        size_best_a is not None and score >= size_best_a - TIER_A_NEAR_BEST_DELTA
    ) or (
        size_best_b is not None and score >= size_best_b - TIER_A_NEAR_BEST_DELTA
    )


def tier_b_reference_score(state: CampaignState, candidate: Candidate) -> float | None:
    scores: list[float] = []
    if candidate.parent_id is not None and candidate.parent_id in state.candidates:
        parent_score = state.candidates[candidate.parent_id].mean_score("B")
        if parent_score is not None:
            scores.append(parent_score)
    frontier_score = lane_best_score(state, candidate.lane, "B")
    if frontier_score is not None:
        scores.append(frontier_score)
    if not scores:
        return None
    return max(scores)


def process_candidate(
    dataset: DatasetBundle,
    state: CampaignState,
    candidate: Candidate,
    *,
    result_cache: dict[tuple[str, tuple[str, ...], str, str], TrainingResult],
    evaluation_limit: int,
    write_results: bool,
    commit_hash: str,
) -> None:
    if state.evaluation_count >= evaluation_limit:
        return
    if not candidate_within_feature_cap(candidate):
        candidate.status = "discarded"
        return

    if candidate_needs_tier(candidate, "A"):
        result_a = evaluate_candidate_tier(
            dataset,
            state,
            candidate,
            "A",
            result_cache=result_cache,
            write_results=write_results,
            commit_hash=commit_hash,
        )
        if result_a is None:
            return
        parent_score = tier_a_parent_score(state, candidate)
        lane_best_a = tier_a_lane_best_score(state, candidate)
        lane_best_b = tier_b_lane_best_score(state, candidate)
        size_best_a = feature_count_best_score(
            state,
            candidate.lane,
            "A",
            candidate.feature_count(),
            exclude_candidate_id=candidate.candidate_id,
        )
        size_best_b = feature_count_best_score(
            state,
            candidate.lane,
            "B",
            candidate.feature_count(),
            exclude_candidate_id=candidate.candidate_id,
        )
        reference_score = tier_a_reference_score(state, candidate)
        delta = result_a.val_cindex if reference_score is None else result_a.val_cindex - reference_score
        close_to_lane_frontier = tier_a_close_to_lane_frontier(
            result_a.val_cindex,
            lane_best_a,
            lane_best_b,
            size_best_a,
            size_best_b,
        )
        parent_gain = None if parent_score is None else result_a.val_cindex - parent_score
        if reference_score is not None and result_a.val_cindex <= reference_score and not close_to_lane_frontier:
            promotion = "discard"
            candidate.status = "discarded"
        elif delta >= TIER_A_STRONG_DELTA or (
            parent_gain is not None and parent_gain >= TIER_A_STRONG_DELTA
        ):
            promotion = "promote_to_B"
            candidate.status = "promote_to_B"
        elif close_to_lane_frontier or delta >= TIER_A_BORDERLINE_DELTA:
            promotion = "retest_A"
            candidate.status = "retest_A"
        else:
            promotion = "discard"
            candidate.status = "discarded"
        record_evaluation(
            state,
            candidate,
            "A",
            STATUS_KEEP if promotion in {"promote_to_B", "retest_A"} else STATUS_DISCARD,
            promotion,
            result_a,
            write_results=write_results,
            commit_hash=commit_hash,
        )

    if candidate.status == "retest_A" and state.evaluation_count < evaluation_limit:
        result_a_repeat = evaluate_candidate_tier(
            dataset,
            state,
            candidate,
            "A",
            result_cache=result_cache,
            write_results=write_results,
            commit_hash=commit_hash,
        )
        if result_a_repeat is None:
            return
        mean_score = candidate.mean_score("A")
        lane_best_a = tier_a_lane_best_score(state, candidate)
        lane_best_b = tier_b_lane_best_score(state, candidate)
        size_best_a = feature_count_best_score(
            state,
            candidate.lane,
            "A",
            candidate.feature_count(),
            exclude_candidate_id=candidate.candidate_id,
        )
        size_best_b = feature_count_best_score(
            state,
            candidate.lane,
            "B",
            candidate.feature_count(),
            exclude_candidate_id=candidate.candidate_id,
        )
        reference_score = tier_a_reference_score(state, candidate)
        delta = 0.0 if reference_score is None or mean_score is None else mean_score - reference_score
        close_to_lane_frontier = False
        if mean_score is not None:
            close_to_lane_frontier = tier_a_close_to_lane_frontier(
                mean_score,
                lane_best_a,
                lane_best_b,
                size_best_a,
                size_best_b,
            )
        if mean_score is not None and reference_score is None:
            promotion = "promote_to_B"
            candidate.status = "promote_to_B"
        elif delta >= TIER_A_BORDERLINE_DELTA or close_to_lane_frontier:
            promotion = "promote_to_B"
            candidate.status = "promote_to_B"
        else:
            promotion = "discard"
            candidate.status = "discarded"
        record_evaluation(
            state,
            candidate,
            "A",
            STATUS_KEEP if promotion == "promote_to_B" else STATUS_DISCARD,
            promotion,
            result_a_repeat,
            write_results=write_results,
            commit_hash=commit_hash,
        )

    if candidate.status != "promote_to_B" or state.evaluation_count >= evaluation_limit:
        return

    if candidate_needs_tier(candidate, "B"):
        result_b = evaluate_candidate_tier(
            dataset,
            state,
            candidate,
            "B",
            result_cache=result_cache,
            write_results=write_results,
            commit_hash=commit_hash,
        )
        if result_b is None:
            return
        reference_score = tier_b_reference_score(state, candidate)
        mean_score = result_b.val_cindex
        if reference_score is None:
            keep_frontier = True
        elif mean_score >= reference_score + TIER_B_PROMOTION_DELTA:
            keep_frontier = True
        elif mean_score >= reference_score - NEAR_TIE_DELTA:
            keep_frontier = True
        else:
            keep_frontier = False
        redundant = False
        provisional_scores = {candidate.candidate_id: mean_score}
        for frontier_candidate in current_frontier(state, candidate.lane):
            if jaccard_overlap(candidate, frontier_candidate) >= OVERLAP_REDUNDANCY_THRESHOLD:
                if (
                    preferred_candidate(
                        frontier_candidate,
                        candidate,
                        "B",
                        provisional_scores=provisional_scores,
                    )
                    is frontier_candidate
                ):
                    redundant = True
                    break
        promotion = "keep_frontier" if keep_frontier and not redundant else "discard"
        candidate.status = "confirmed" if promotion == "keep_frontier" else "discarded"
        record_evaluation(
            state,
            candidate,
            "B",
            STATUS_KEEP if promotion == "keep_frontier" else STATUS_DISCARD,
            promotion,
            result_b,
            write_results=write_results,
            commit_hash=commit_hash,
        )
        if promotion == "keep_frontier":
            refresh_frontier(state, candidate.lane)


def lane_stalled(state: CampaignState, lane: str) -> bool:
    recent = [
        candidate
        for candidate in state.candidates.values()
        if candidate.lane == lane and candidate.mean_score("B") is not None
    ]
    if len(recent) < LANE_PATIENCE_EVALUATIONS:
        return False
    recent = sorted(recent, key=lambda candidate: candidate.created_round)[-LANE_PATIENCE_EVALUATIONS:]
    scores = [candidate.mean_score("B") or float("-inf") for candidate in recent]
    return max(scores) - min(scores) < TIER_A_BORDERLINE_DELTA


def lane_reference_target(lane: str) -> float | None:
    return LANE_REFERENCE_TARGETS.get(lane)


def lane_reference_gap(state: CampaignState, lane: str) -> float | None:
    target = lane_reference_target(lane)
    best_b = lane_best_score(state, lane, "B")
    if target is None or best_b is None:
        return None
    return target - best_b


def lane_scored_candidate_count(state: CampaignState, lane: str, tier: str = "B") -> int:
    return sum(
        1
        for candidate in state.candidates.values()
        if candidate.lane == lane
        and candidate_within_feature_cap(candidate)
        and candidate.mean_score(tier) is not None
    )


def lane_search_phase(state: CampaignState, lane: str) -> str:
    if lane_best_score(state, lane, "B") is None or lane_scored_candidate_count(state, lane, "B") < FRONTIER_BEAM_WIDTH:
        return "discovery"
    reference_gap = lane_reference_gap(state, lane)
    if lane_stalled(state, lane) or (reference_gap is not None and reference_gap <= BANDIT_BREAKOUT_TARGET_GAP):
        return "breakout"
    return "consolidation"


def bandit_arm_stats(state: CampaignState, lane: str, arm: str) -> dict[str, float]:
    return state.bandit_stats.setdefault(lane, {}).setdefault(
        arm,
        {"pulls": 0.0, "total_reward": 0.0},
    )


def bandit_recent_rewards(state: CampaignState, lane: str, arm: str) -> list[float]:
    return state.bandit_recent_rewards.setdefault(lane, {}).setdefault(arm, [])


def bandit_arm_mean_reward(state: CampaignState, lane: str, arm: str) -> float:
    stats = bandit_arm_stats(state, lane, arm)
    pulls = stats["pulls"]
    if pulls <= 0:
        return float("-inf")
    return stats["total_reward"] / pulls


def bandit_recent_mean_reward(state: CampaignState, lane: str, arm: str) -> float | None:
    rewards = bandit_recent_rewards(state, lane, arm)
    if not rewards:
        return None
    return float(sum(rewards) / len(rewards))


def bandit_arm_strength(state: CampaignState, lane: str, arm: str) -> float:
    recent_mean = bandit_recent_mean_reward(state, lane, arm)
    if recent_mean is not None:
        return recent_mean
    lifetime_mean = bandit_arm_mean_reward(state, lane, arm)
    return 0.0 if not math.isfinite(lifetime_mean) else lifetime_mean


def bandit_phase_base_weight(phase: str, arm: str) -> float:
    phase_weights = {
        "discovery": {
            "frontier_local": 1.00,
            "size_winner_local": 1.05,
            "root_seed_local": 1.00,
            "random_restart": 1.00,
            "elite_crossover": 0.90,
        },
        "consolidation": {
            "frontier_local": 1.20,
            "size_winner_local": 1.35,
            "root_seed_local": 0.80,
            "random_restart": 0.70,
            "elite_crossover": 0.75,
        },
        "breakout": {
            "frontier_local": 1.30,
            "size_winner_local": 1.45,
            "root_seed_local": 0.65,
            "random_restart": 0.40,
            "elite_crossover": 0.65,
        },
    }
    return phase_weights.get(phase, phase_weights["discovery"]).get(arm, 1.0)


def bandit_arm_selection_weight(state: CampaignState, lane: str, arm: str) -> float:
    phase = lane_search_phase(state, lane)
    weight = bandit_phase_base_weight(phase, arm)
    pulls = bandit_arm_stats(state, lane, arm)["pulls"]
    strengths = [bandit_arm_strength(state, lane, candidate_arm) for candidate_arm in BANDIT_ARMS]
    best_strength = max(strengths, default=0.0)
    arm_strength = bandit_arm_strength(state, lane, arm)
    if pulls <= 0:
        unseen_multiplier = 0.95 if phase == "discovery" else 0.60 if phase == "consolidation" else 0.50
        return max(0.01, weight * unseen_multiplier)
    if best_strength > 0.0:
        relative_strength = arm_strength / best_strength
        if relative_strength < BANDIT_VERY_WEAK_ARM_RATIO:
            weight *= 0.20
        elif relative_strength < BANDIT_WEAK_ARM_RATIO:
            weight *= 0.45
        elif relative_strength < 0.80:
            weight *= 0.70
    return max(0.01, weight)


def weighted_bandit_choice(
    rng: random.Random,
    weighted_arms: list[tuple[str, float]],
) -> str:
    total_weight = sum(max(0.0, weight) for _, weight in weighted_arms)
    if total_weight <= 0.0:
        return rng.choice([arm for arm, _ in weighted_arms])
    draw = rng.random() * total_weight
    running = 0.0
    for arm, weight in weighted_arms:
        running += max(0.0, weight)
        if draw <= running:
            return arm
    return weighted_arms[-1][0]


def bandit_epsilon_for_lane(state: CampaignState, lane: str) -> float:
    phase = lane_search_phase(state, lane)
    if phase == "discovery":
        epsilon = BANDIT_EPSILON
    elif phase == "consolidation":
        epsilon = 0.14
    else:
        epsilon = 0.16 if lane_stalled(state, lane) else 0.12
    return max(BANDIT_EPSILON_FLOOR, epsilon)


def choose_bandit_arm(state: CampaignState, lane: str, rng: random.Random) -> str:
    phase = lane_search_phase(state, lane)
    untried = [arm for arm in BANDIT_ARMS if bandit_arm_stats(state, lane, arm)["pulls"] <= 0]
    if untried:
        return weighted_bandit_choice(
            rng,
            [(arm, bandit_phase_base_weight(phase, arm)) for arm in untried],
        )
    weighted_arms = [(arm, bandit_arm_selection_weight(state, lane, arm)) for arm in BANDIT_ARMS]
    if rng.random() < bandit_epsilon_for_lane(state, lane):
        return weighted_bandit_choice(rng, weighted_arms)
    ranked = sorted(
        weighted_arms,
        key=lambda item: (
            item[1],
            bandit_arm_strength(state, lane, item[0]),
            -bandit_arm_stats(state, lane, item[0])["pulls"],
            item[0],
        ),
    )
    return ranked[-1][0]


def update_bandit_arm(state: CampaignState, lane: str, arm: str, reward: float) -> None:
    stats = bandit_arm_stats(state, lane, arm)
    stats["pulls"] += 1.0
    stats["total_reward"] += max(0.0, reward)
    recent_rewards = bandit_recent_rewards(state, lane, arm)
    recent_rewards.append(max(0.0, reward))
    if len(recent_rewards) > BANDIT_RECENT_WINDOW:
        del recent_rewards[:-BANDIT_RECENT_WINDOW]


def sort_candidates_by_search_strength(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.mean_score("C") if candidate.mean_score("C") is not None else float("-inf"),
            candidate.mean_score("B") if candidate.mean_score("B") is not None else float("-inf"),
            candidate.mean_score("A") if candidate.mean_score("A") is not None else float("-inf"),
            -candidate.feature_count(),
            candidate.candidate_id,
        ),
        reverse=True,
    )


def candidate_search_score(candidate: Candidate) -> float:
    return (
        candidate.mean_score("C")
        or candidate.mean_score("B")
        or candidate.mean_score("A")
        or float("-inf")
    )


def size_winner_parents(state: CampaignState, lane: str) -> list[Candidate]:
    winners: list[Candidate] = []
    seen: set[str] = set()
    for feature_count in range(1, MAX_INPUT_FEATURES + 1):
        winner = best_candidate_for_feature_count(state, lane, feature_count)
        if winner is None or winner.candidate_id in seen:
            continue
        winners.append(winner)
        seen.add(winner.candidate_id)
    return sort_candidates_by_search_strength(winners)


def root_seed_parents(state: CampaignState, lane: str) -> list[Candidate]:
    candidates = [
        candidate
        for candidate in state.candidates.values()
        if candidate.lane == lane
        and candidate.parent_id is None
        and candidate_within_feature_cap(candidate)
        and (
            candidate.mean_score("C") is not None
            or candidate.mean_score("B") is not None
            or candidate.mean_score("A") is not None
        )
    ]
    return sort_candidates_by_search_strength(candidates)


def random_restart_size(include_age: bool, rng: random.Random) -> int:
    max_biomarkers = max_biomarker_count(include_age)
    if max_biomarkers <= 1:
        return 1
    lower = min(3, max_biomarkers)
    upper = max(lower, max_biomarkers)
    if rng.random() < 0.65:
        return rng.randint(lower, upper)
    return rng.randint(1, max_biomarkers)


def create_random_restart_candidates(
    state: CampaignState,
    lane: str,
    rng: random.Random,
    round_index: int,
    count: int = BANDIT_RANDOM_RESTARTS_PER_PULL,
) -> list[Candidate]:
    include_age = LANE_CONFIGS[lane].include_age
    existing_signatures = {candidate.signature() for candidate in state.candidates.values()}
    proposals: list[Candidate] = []
    attempts = 0
    max_attempts = max(count * 8, 16)
    while len(proposals) < count and attempts < max_attempts:
        attempts += 1
        subset_size = random_restart_size(include_age, rng)
        biomarkers = tuple(sorted(rng.sample(CANDIDATE_BIOMARKER_COLUMNS, k=subset_size)))
        if not biomarkers_within_feature_cap(include_age, biomarkers):
            continue
        signature = candidate_signature(lane, biomarkers)
        if signature in existing_signatures:
            continue
        candidate = Candidate(
            candidate_id=new_candidate_id(state),
            lane=lane,
            biomarkers=biomarkers,
            parent_id=None,
            operator="bandit_seed",
            seed_family="random_restart",
            proposal_arm="random_restart",
            rationale=f"Bandit random restart for {lane}.",
            created_round=round_index,
        )
        proposals.append(candidate)
        existing_signatures.add(signature)
    return proposals


def crossover_parent_pool(state: CampaignState, lane: str) -> list[Candidate]:
    pooled: list[Candidate] = []
    seen: set[str] = set()
    for source in (current_frontier(state, lane), size_winner_parents(state, lane), root_seed_parents(state, lane)):
        for candidate in source:
            if candidate.candidate_id in seen or not candidate_within_feature_cap(candidate):
                continue
            if candidate.mean_score("B") is None and candidate.mean_score("A") is None:
                continue
            pooled.append(candidate)
            seen.add(candidate.candidate_id)
    return sort_candidates_by_search_strength(pooled)


def crossover_parent_pairs(state: CampaignState, lane: str) -> list[tuple[Candidate, Candidate]]:
    pool = crossover_parent_pool(state, lane)[:BANDIT_CROSSOVER_PARENT_POOL]
    if len(pool) < 2:
        return []
    pairs: list[tuple[Candidate, Candidate]] = []
    for index, parent_a in enumerate(pool):
        for parent_b in pool[index + 1 :]:
            overlap = jaccard_overlap(parent_a, parent_b)
            if overlap >= 0.98:
                continue
            pairs.append((parent_a, parent_b))
    pairs.sort(
        key=lambda pair: (
            candidate_search_score(pair[0]) + candidate_search_score(pair[1]),
            -abs(jaccard_overlap(pair[0], pair[1]) - 0.5),
            -abs(len(pair[0].biomarkers) - len(pair[1].biomarkers)),
        ),
        reverse=True,
    )
    return pairs


def crossover_biomarkers_for_pair(
    parent_a: Candidate,
    parent_b: Candidate,
    target_size: int,
    rng: random.Random,
) -> tuple[str, ...]:
    shared = sorted(set(parent_a.biomarkers) & set(parent_b.biomarkers))
    left_only = [marker for marker in parent_a.biomarkers if marker not in shared]
    right_only = [marker for marker in parent_b.biomarkers if marker not in shared]
    rng.shuffle(left_only)
    rng.shuffle(right_only)
    selected = list(shared)
    if len(selected) > target_size:
        selected = rng.sample(selected, k=target_size)
    mixed: list[str] = []
    while left_only or right_only:
        if left_only:
            mixed.append(left_only.pop())
        if right_only:
            mixed.append(right_only.pop())
    for marker in mixed:
        if len(selected) >= target_size:
            break
        if marker not in selected:
            selected.append(marker)
    if len(selected) < target_size:
        union = [marker for marker in sorted(set(parent_a.biomarkers) | set(parent_b.biomarkers)) if marker not in selected]
        rng.shuffle(union)
        selected.extend(union[: target_size - len(selected)])
    return tuple(sorted(selected))


def create_elite_crossover_candidates(
    state: CampaignState,
    lane: str,
    rng: random.Random,
    round_index: int,
    count: int = BANDIT_CROSSOVER_CHILDREN_PER_PULL,
) -> list[Candidate]:
    include_age = LANE_CONFIGS[lane].include_age
    max_biomarkers = max_biomarker_count(include_age)
    existing_signatures = {candidate.signature() for candidate in state.candidates.values()}
    proposals: list[Candidate] = []
    for parent_a, parent_b in crossover_parent_pairs(state, lane):
        if len(proposals) >= count:
            break
        shared_count = len(set(parent_a.biomarkers) & set(parent_b.biomarkers))
        union_count = len(set(parent_a.biomarkers) | set(parent_b.biomarkers))
        target_sizes: list[int] = []
        for size in (
            max(shared_count, min(len(parent_a.biomarkers), len(parent_b.biomarkers))),
            max(shared_count, round((len(parent_a.biomarkers) + len(parent_b.biomarkers)) / 2)),
            max(shared_count, max(len(parent_a.biomarkers), len(parent_b.biomarkers))),
        ):
            clipped = min(max(1, int(size)), max_biomarkers, union_count)
            if clipped not in target_sizes:
                target_sizes.append(clipped)
        primary_parent, secondary_parent = sorted(
            (parent_a, parent_b),
            key=candidate_search_score,
            reverse=True,
        )
        for target_size in target_sizes:
            if len(proposals) >= count:
                break
            biomarkers = crossover_biomarkers_for_pair(primary_parent, secondary_parent, target_size, rng)
            if not biomarkers or not biomarkers_within_feature_cap(include_age, biomarkers):
                continue
            signature = candidate_signature(lane, biomarkers)
            if signature in existing_signatures:
                continue
            candidate = Candidate(
                candidate_id=new_candidate_id(state),
                lane=lane,
                biomarkers=biomarkers,
                parent_id=primary_parent.candidate_id,
                operator="crossover",
                seed_family="elite_crossover",
                proposal_arm="elite_crossover",
                rationale=(
                    f"Elite crossover of {primary_parent.candidate_id} "
                    f"with {secondary_parent.candidate_id}."
                ),
                created_round=round_index,
            )
            proposals.append(candidate)
            existing_signatures.add(signature)
    for candidate in proposals:
        state.candidates[candidate.candidate_id] = candidate
    return proposals


def parents_for_bandit_arm(state: CampaignState, lane: str, arm: str) -> list[Candidate]:
    if arm == "frontier_local":
        return current_frontier(state, lane)
    if arm == "size_winner_local":
        return size_winner_parents(state, lane)
    if arm == "root_seed_local":
        return root_seed_parents(state, lane)
    return []


def bandit_reward_for_children(
    state: CampaignState,
    lane: str,
    children: list[Candidate],
    best_b_before: float | None,
) -> float:
    reward = 0.0
    for child in children:
        score_b = child.mean_score("B")
        if score_b is not None:
            reward += 0.4
            if child.status in {"confirmed", "frontier"}:
                reward += 0.8
        elif child.status in {"promote_to_B", "retest_A"} and child.mean_score("A") is not None:
            reward += 0.05
    best_b_after = lane_best_score(state, lane, "B")
    if best_b_after is not None:
        if best_b_before is None:
            reward += best_b_after
        else:
            reward += max(0.0, best_b_after - best_b_before) * 1000.0
    return reward


def execute_bandit_pull(
    dataset: DatasetBundle,
    state: CampaignState,
    lane: str,
    rng: random.Random,
    round_index: int,
    *,
    result_cache: dict[tuple[str, str, float], TrainingResult],
    evaluation_limit: int,
    write_results: bool,
    commit_hash: str,
) -> bool:
    arm = choose_bandit_arm(state, lane, rng)
    best_b_before = lane_best_score(state, lane, "B")
    children: list[Candidate] = []
    if arm == "random_restart":
        children = create_random_restart_candidates(state, lane, rng, round_index)
    elif arm == "elite_crossover":
        children = create_elite_crossover_candidates(state, lane, rng, round_index)
    else:
        parents = parents_for_bandit_arm(state, lane, arm)[:BANDIT_PARENT_LIMIT]
        for parent in parents:
            children.extend(
                propose_children_for_candidate(
                    state,
                    parent,
                    rng,
                    round_index,
                    source_arm=arm,
                )
            )
    if not children:
        update_bandit_arm(state, lane, arm, 0.0)
        return False
    proposals_made = False
    for child in children:
        if state.evaluation_count >= evaluation_limit:
            break
        proposals_made = True
        process_candidate(
            dataset,
            state,
            child,
            result_cache=result_cache,
            evaluation_limit=evaluation_limit,
            write_results=write_results,
            commit_hash=commit_hash,
        )
    update_bandit_arm(
        state,
        lane,
        arm,
        bandit_reward_for_children(state, lane, children, best_b_before),
    )
    return proposals_made


def propose_children_for_candidate(
    state: CampaignState,
    parent: Candidate,
    rng: random.Random,
    round_index: int,
    *,
    source_arm: str = "frontier_local",
) -> list[Candidate]:
    existing_signatures = {candidate.signature() for candidate in state.candidates.values()}
    included = list(parent.biomarkers)
    excluded = [biomarker for biomarker in CANDIDATE_BIOMARKER_COLUMNS if biomarker not in parent.biomarkers]
    proposals: list[Candidate] = []

    def maybe_add(
        biomarkers: list[str] | tuple[str, ...],
        operator: str,
        rationale: str,
        seed_family: str = "frontier_branch",
    ) -> None:
        biomarker_tuple = tuple(sorted(set(biomarkers)))
        if not biomarker_tuple:
            return
        if not biomarkers_within_feature_cap(LANE_CONFIGS[parent.lane].include_age, biomarker_tuple):
            return
        signature = candidate_signature(parent.lane, biomarker_tuple)
        if signature in existing_signatures:
            return
        candidate = Candidate(
            candidate_id=new_candidate_id(state),
            lane=parent.lane,
            biomarkers=biomarker_tuple,
            parent_id=parent.candidate_id,
            operator=operator,
            seed_family=seed_family,
            proposal_arm=source_arm,
            rationale=rationale,
            created_round=round_index,
        )
        proposals.append(candidate)
        existing_signatures.add(signature)

    if excluded:
        for biomarker in rng.sample(excluded, k=min(2, len(excluded))):
            maybe_add(
                included + [biomarker],
                "add",
                f"Add {biomarker} to {parent.candidate_id}.",
            )

    if len(included) > 1:
        for biomarker in rng.sample(included, k=min(2, len(included))):
            maybe_add(
                [value for value in included if value != biomarker],
                "drop",
                f"Drop {biomarker} from {parent.candidate_id}.",
            )

    if included and excluded:
        swap_count = min(2, len(included), len(excluded))
        for drop_marker, add_marker in zip(
            rng.sample(included, k=swap_count),
            rng.sample(excluded, k=swap_count),
        ):
            child = [value for value in included if value != drop_marker] + [add_marker]
            maybe_add(
                child,
                "swap",
                f"Swap {drop_marker} for {add_marker} from {parent.candidate_id}.",
            )

    if included:
        overlapping_groups = [
            (group_name, group_markers)
            for group_name, group_markers in BIOLOGICAL_GROUPS.items()
            if set(group_markers) & set(included)
        ]
        if overlapping_groups:
            group_name, group_markers = rng.choice(overlapping_groups)
            group_markers = list(group_markers)
            if rng.random() < 0.5:
                additions = [marker for marker in group_markers if marker not in included]
                if additions:
                    additions = additions[:GROUP_MOVE_MAX_CHANGE]
                    maybe_add(
                        included + additions,
                        "group",
                        f"Grouped add from {group_name} around {parent.candidate_id}.",
                    )
            else:
                removals = [marker for marker in included if marker in group_markers]
                if removals and len(included) > len(removals):
                    maybe_add(
                        [marker for marker in included if marker not in removals[:GROUP_MOVE_MAX_CHANGE]],
                        "group",
                        f"Grouped drop from {group_name} around {parent.candidate_id}.",
                    )

    perturb = set(included)
    if perturb:
        remove_count = min(len(perturb), 1 + int(rng.random() < 0.5))
        for biomarker in rng.sample(list(perturb), k=remove_count):
            perturb.discard(biomarker)
    available = [biomarker for biomarker in CANDIDATE_BIOMARKER_COLUMNS if biomarker not in perturb]
    if available:
        add_count = min(len(available), 1 + int(rng.random() < 0.5))
        perturb.update(rng.sample(available, k=add_count))
    maybe_add(
        tuple(sorted(perturb)),
        "perturb",
        f"Random perturbation from {parent.candidate_id}.",
    )

    for candidate in proposals:
        state.candidates[candidate.candidate_id] = candidate
    return proposals


def frontier_seed_order(state: CampaignState) -> list[Candidate]:
    ordered: list[Candidate] = []
    for lane in LANE_ORDER:
        frontier = current_frontier(state, lane)
        if frontier:
            ordered.extend(frontier)
        else:
            seeds = [
                candidate
                for candidate in state.candidates.values()
                if candidate.lane == lane
                and candidate_within_feature_cap(candidate)
                and candidate.parent_id is None
                and candidate.mean_score("A") is not None
            ]
            seeds.sort(
                key=lambda candidate: (-(candidate.mean_score("A") or float("-inf")), candidate.feature_count())
            )
            ordered.extend(seeds[:2])
    return ordered


def emit_status_snapshot(state: CampaignState, path: Path = CAMPAIGN_STATUS_PATH) -> None:
    snapshot: dict[str, Any] = {
        "timestamp": now_utc(),
        "evaluation_count": state.evaluation_count,
        "round_index": state.round_index,
        "frontier": {},
        "lane_best_scores": {},
        "lane_phase": {},
        "benchmark_gap": {},
        "size_winners": {},
        "bandit": {},
        "lane_winners": state.lane_winner_ids,
        "overall_winner_id": state.overall_winner_id,
    }
    for lane in LANE_ORDER:
        phase = lane_search_phase(state, lane)
        frontier = current_frontier(state, lane)
        snapshot["frontier"][lane] = [
            {
                "candidate_id": candidate.candidate_id,
                "feature_count": candidate.feature_count(),
                "tier_b_score": candidate.mean_score("B"),
                "tier_c_score": candidate.mean_score("C"),
                "dispersion_b": candidate.score_dispersion("B"),
            }
            for candidate in frontier
        ]
        snapshot["lane_best_scores"][lane] = {
            "tier_a": lane_best_score(state, lane, "A"),
            "tier_b": lane_best_score(state, lane, "B"),
            "tier_c": lane_best_score(state, lane, "C"),
        }
        snapshot["lane_phase"][lane] = phase
        snapshot["benchmark_gap"][lane] = lane_reference_gap(state, lane)
        arm_ranking = sorted(
            BANDIT_ARMS,
            key=lambda arm: (
                bandit_arm_selection_weight(state, lane, arm),
                bandit_arm_strength(state, lane, arm),
                arm,
            ),
            reverse=True,
        )
        snapshot["bandit"][lane] = {
            arm: {
                "pulls": int(bandit_arm_stats(state, lane, arm)["pulls"]),
                "total_reward": bandit_arm_stats(state, lane, arm)["total_reward"],
                "mean_reward": (
                    None
                    if bandit_arm_stats(state, lane, arm)["pulls"] <= 0
                    else bandit_arm_mean_reward(state, lane, arm)
                ),
                "recent_reward_count": len(bandit_recent_rewards(state, lane, arm)),
                "recent_mean_reward": bandit_recent_mean_reward(state, lane, arm),
                "selection_weight": bandit_arm_selection_weight(state, lane, arm),
                "selection_rank": arm_ranking.index(arm) + 1,
            }
            for arm in BANDIT_ARMS
        }
        snapshot["size_winners"][lane] = {}
        for feature_count in range(1, MAX_INPUT_FEATURES + 1):
            winner = best_candidate_for_feature_count(state, lane, feature_count)
            snapshot["size_winners"][lane][str(feature_count)] = None if winner is None else {
                "candidate_id": winner.candidate_id,
                "feature_count": winner.feature_count(),
                "selected_biomarkers": list(winner.biomarkers),
                "tier_a_score": winner.mean_score("A"),
                "tier_b_score": winner.mean_score("B"),
                "tier_c_score": winner.mean_score("C"),
            }
    state.latest_status = snapshot
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")


def evaluate_lane_finalists(
    dataset: DatasetBundle,
    state: CampaignState,
    lane: str,
    finalists_per_lane: int,
    *,
    result_cache: dict[tuple[str, tuple[str, ...], str, str], TrainingResult],
    evaluation_limit: int,
    write_results: bool,
    commit_hash: str,
) -> None:
    contenders = [
        candidate
        for candidate in state.candidates.values()
        if candidate.lane == lane
        and candidate_within_feature_cap(candidate)
        and candidate.mean_score("B") is not None
        and candidate.status != "crashed"
    ]
    contenders.sort(
        key=lambda candidate: (
            -(candidate.mean_score("B") or float("-inf")),
            candidate.feature_count(),
            candidate.score_dispersion("B"),
        )
    )
    selected: list[Candidate] = []
    for candidate in contenders:
        redundant = any(jaccard_overlap(candidate, kept) >= OVERLAP_REDUNDANCY_THRESHOLD for kept in selected)
        if redundant:
            continue
        selected.append(candidate)
        if len(selected) >= finalists_per_lane:
            break

    for candidate in selected:
        if state.evaluation_count >= evaluation_limit:
            break
        candidate.locked_finalist = True
        if candidate_needs_tier(candidate, "C"):
            result_c = evaluate_candidate_tier(
                dataset,
                state,
                candidate,
                "C",
                result_cache=result_cache,
                write_results=write_results,
                commit_hash=commit_hash,
                persist_artifact=True,
            )
            if result_c is None:
                continue
            record_evaluation(
                state,
                candidate,
                "C",
                STATUS_KEEP,
                "locked_finalist",
                result_c,
                write_results=write_results,
                commit_hash=commit_hash,
            )

    winner = choose_lane_winner(state, lane)
    if winner is not None:
        winner.locked_winner = True
        state.lane_winner_ids[lane] = winner.candidate_id


def copy_locked_winner_artifact(state: CampaignState) -> Candidate | None:
    winner = choose_overall_winner(state)
    if winner is None:
        return None
    artifact_path = winner.artifact_path("C")
    if artifact_path is None or not artifact_path.exists():
        raise FileNotFoundError(
            f"Locked winner {winner.candidate_id} is missing a Tier C artifact."
        )
    metadata_path = artifact_path.with_suffix(".metadata.json")
    shutil.copy2(artifact_path, DEFAULT_CANDIDATE_MODEL_PATH)
    shutil.copy2(metadata_path, DEFAULT_CANDIDATE_MODEL_PATH.with_suffix(".metadata.json"))
    state.overall_winner_id = winner.candidate_id
    return winner


def evaluate_locked_winner_on_test(
    dataset: DatasetBundle,
    state: CampaignState,
    winner: Candidate,
    *,
    skip_test: bool,
) -> None:
    if skip_test or state.test_evaluated:
        return
    scores = score_scripted_model(DEFAULT_CANDIDATE_MODEL_PATH, dataset.test_rows, device="cpu")
    test_c_index = harrell_c_index(dataset.test_times_np, dataset.test_events_np, scores)
    result = build_result_summary(test_c_index=test_c_index)
    result.update(
        {
            "benchmark_dataset": "nhanes3-bioage",
            "evaluation_split": "test",
            "participants": len(dataset.test_rows),
            "aging_related_deaths": int(dataset.test_events_np.sum()),
            "candidate_model_path": str(DEFAULT_CANDIDATE_MODEL_PATH),
            "selected_biomarkers": list(winner.biomarkers),
            "input_feature_count": winner.feature_count(),
            "age_included_in_inputs": LANE_CONFIGS[winner.lane].include_age,
            "lane": winner.lane,
            "validation_cindex": winner.mean_score("C") or winner.mean_score("B") or winner.mean_score("A"),
            "missing_data_strategy": "train-split mean imputation fit on development-train rows only",
            "frozen_split_seed": DEV_VAL_SEED,
            "training_time_ceiling_s": TIME_BUDGET,
        }
    )
    write_json(DEFAULT_RESULT_PATH, result)
    state.test_evaluated = True
    print(
        f"c-index: {test_c_index:.6f}, "
        f"subsets: {winner.feature_count()}, "
        f"biomarkers: {format_inputs(winner)}"
    )


def write_campaign_summary(state: CampaignState) -> None:
    summary: dict[str, Any] = {
        "timestamp": now_utc(),
        "evaluation_count": state.evaluation_count,
        "lane_winner_ids": state.lane_winner_ids,
        "overall_winner_id": state.overall_winner_id,
        "frontier_ids": state.frontier_ids,
        "finalists": {},
    }
    for lane in LANE_ORDER:
        summary["finalists"][lane] = [
            {
                "candidate_id": candidate.candidate_id,
                "feature_count": candidate.feature_count(),
                "selected_biomarkers": list(candidate.biomarkers),
                "tier_a_score": candidate.mean_score("A"),
                "tier_b_score": candidate.mean_score("B"),
                "tier_c_score": candidate.mean_score("C"),
                "dispersion_b": candidate.score_dispersion("B"),
                "artifact_path": str(candidate.artifact_path("C")) if candidate.artifact_path("C") else None,
            }
            for candidate in state.candidates.values()
            if candidate.lane == lane and candidate.locked_finalist
        ]
    CAMPAIGN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_campaign(args: argparse.Namespace) -> None:
    validate_biological_groups()
    dataset = build_dataset_bundle()
    state = load_or_create_state(args.fresh)
    result_cache = load_result_cache()
    rng = random.Random(DEV_VAL_SEED)
    commit_hash = short_git_hash()

    if args.smoke_test:
        args.max_evaluations = min(args.max_evaluations, 8)
        args.max_rounds = min(args.max_rounds, 1)
        args.finalists_per_lane = 1
        args.no_results_log = True
        args.skip_test = True

    initialize_seeds(state, rng)
    save_state(state)
    evaluation_limit = state.evaluation_count + args.max_evaluations

    pending_seeds = build_pending_seed_queue(state, alternate_lanes=True)

    while pending_seeds and state.evaluation_count < evaluation_limit:
        candidate = pending_seeds.pop(0)
        process_candidate(
            dataset,
            state,
            candidate,
            result_cache=result_cache,
            evaluation_limit=evaluation_limit,
            write_results=not args.no_results_log,
            commit_hash=commit_hash,
        )
        if state.evaluation_count % max(1, args.status_every) == 0:
            emit_status_snapshot(state)
            save_state(state)

    for lane in LANE_ORDER:
        refresh_frontier(state, lane)
    emit_status_snapshot(state)
    save_state(state)

    for round_index in range(state.round_index + 1, args.max_rounds + 1):
        if state.evaluation_count >= evaluation_limit:
            break
        state.round_index = round_index
        proposals_made = False
        for lane in LANE_ORDER:
            if state.evaluation_count >= evaluation_limit:
                break
            for _ in range(BANDIT_PULLS_PER_LANE_PER_ROUND):
                if state.evaluation_count >= evaluation_limit:
                    break
                proposals_made = execute_bandit_pull(
                    dataset,
                    state,
                    lane,
                    rng,
                    round_index,
                    result_cache=result_cache,
                    evaluation_limit=evaluation_limit,
                    write_results=not args.no_results_log,
                    commit_hash=commit_hash,
                ) or proposals_made
                if state.evaluation_count % max(1, args.status_every) == 0:
                    emit_status_snapshot(state)
                    save_state(state)
            refresh_frontier(state, lane)
        emit_status_snapshot(state)
        save_state(state)
        if not proposals_made:
            break

    if not args.smoke_test:
        for lane in LANE_ORDER:
            if state.evaluation_count >= evaluation_limit:
                break
            evaluate_lane_finalists(
                dataset,
                state,
                lane,
                args.finalists_per_lane,
                result_cache=result_cache,
                evaluation_limit=evaluation_limit,
                write_results=not args.no_results_log,
                commit_hash=commit_hash,
            )
            refresh_frontier(state, lane)

        winner = copy_locked_winner_artifact(state)
        if winner is not None:
            evaluate_locked_winner_on_test(dataset, state, winner, skip_test=args.skip_test)

    emit_status_snapshot(state)
    write_campaign_summary(state)
    save_state(state)



def main() -> None:
    run_campaign(parse_args())


if __name__ == "__main__":
    main()
