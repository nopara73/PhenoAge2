"""Best-first biomarker subset search guided by low-order priors.

This mode reuses the existing benchmark harness and `results.tsv` cache from
`train.py`, but runs a separate search policy:

- build singleton priors from all size-1 subsets
- build pair priors from all size-2 subsets
- seed a priority frontier from the strongest low-order signals
- expand the best-looking subsets with add / swap / shrink moves

The intent is to use singles and pairs as reusable guidance for arbitrary
subset sizes rather than forcing a rigid size-by-size ladder.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from train import (
    AUTORESEARCH_DIR,
    BIOLOGICAL_GROUPS,
    CANDIDATE_BIOMARKER_COLUMNS,
    INPUT_DISPLAY_NAMES,
    LANE_CONFIGS,
    LANE_ORDER,
    REFERENCE_PHENOAGE_BIOMARKERS,
    STATUS_CRASH,
    STATUS_DISCARD,
    STATUS_KEEP,
    TIER_ORDER,
    Candidate,
    TrainingResult,
    append_result_row,
    biomarkers_within_feature_cap,
    build_dataset_bundle,
    budget_failure,
    candidate_signature,
    load_result_cache,
    max_biomarker_count,
    max_result_candidate_index,
    now_utc,
    result_cache_key,
    short_git_hash,
    train_subset,
)

STATE_VERSION = 1
DEFAULT_STATE_PATH = AUTORESEARCH_DIR / "best_first_search_state.json"
DEFAULT_STATUS_PATH = AUTORESEARCH_DIR / "best_first_search_status.json"
DEFAULT_PRIOR_TIER = "A"
DEFAULT_SEARCH_TIER = "B"
DEFAULT_MAX_EVALUATIONS = 200
DEFAULT_MAX_FRONTIER_POPS = 1200
DEFAULT_STATUS_EVERY = 12
DEFAULT_SEED_SINGLETONS = 8
DEFAULT_SEED_PAIRS = 16
DEFAULT_ADD_CHILDREN = 6
DEFAULT_SWAP_CHILDREN = 4
DEFAULT_SHRINK_CHILDREN = 2
DEFAULT_UNDEREXPLORED_SIZE_WEIGHT = 0.08

PRIOR_SINGLETON_STAGE = "singleton_priors"
PRIOR_PAIR_STAGE = "pair_priors"
SEARCH_STAGE = "search"


@dataclass
class FrontierEntry:
    lane: str
    biomarkers: tuple[str, ...]
    priority: float
    source: str
    depth: int
    parent_signature: str | None = None
    parent_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "biomarkers": list(self.biomarkers),
            "priority": self.priority,
            "source": self.source,
            "depth": self.depth,
            "parent_signature": self.parent_signature,
            "parent_score": self.parent_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrontierEntry":
        return cls(
            lane=payload["lane"],
            biomarkers=tuple(payload["biomarkers"]),
            priority=float(payload["priority"]),
            source=payload["source"],
            depth=int(payload["depth"]),
            parent_signature=payload.get("parent_signature"),
            parent_score=(
                None if payload.get("parent_score") is None else float(payload["parent_score"])
            ),
        )


@dataclass
class BestFirstState:
    version: int = STATE_VERSION
    next_candidate_index: int = 1
    evaluation_count: int = 0
    cache_hit_count: int = 0
    frontier_pop_count: int = 0
    stage: str = PRIOR_SINGLETON_STAGE
    prior_lane_index: int = 0
    prior_index: int = 0
    frontier_entries: list[dict[str, Any]] = field(default_factory=list)
    frontier_seen_signatures: list[str] = field(default_factory=list)
    singleton_raw: dict[str, dict[str, float]] = field(
        default_factory=lambda: {lane: {} for lane in LANE_ORDER}
    )
    singleton_prior: dict[str, dict[str, float]] = field(
        default_factory=lambda: {lane: {} for lane in LANE_ORDER}
    )
    pair_raw: dict[str, dict[str, float]] = field(
        default_factory=lambda: {lane: {} for lane in LANE_ORDER}
    )
    pair_prior: dict[str, dict[str, float]] = field(
        default_factory=lambda: {lane: {} for lane in LANE_ORDER}
    )
    best_by_size: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: {lane: {} for lane in LANE_ORDER}
    )
    size_visit_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {lane: {} for lane in LANE_ORDER}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "next_candidate_index": self.next_candidate_index,
            "evaluation_count": self.evaluation_count,
            "cache_hit_count": self.cache_hit_count,
            "frontier_pop_count": self.frontier_pop_count,
            "stage": self.stage,
            "prior_lane_index": self.prior_lane_index,
            "prior_index": self.prior_index,
            "frontier_entries": self.frontier_entries,
            "frontier_seen_signatures": self.frontier_seen_signatures,
            "singleton_raw": self.singleton_raw,
            "singleton_prior": self.singleton_prior,
            "pair_raw": self.pair_raw,
            "pair_prior": self.pair_prior,
            "best_by_size": self.best_by_size,
            "size_visit_counts": self.size_visit_counts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BestFirstState":
        state = cls(version=int(payload.get("version", STATE_VERSION)))
        state.next_candidate_index = int(
            payload.get("next_candidate_index", max_result_candidate_index() + 1)
        )
        state.evaluation_count = int(payload.get("evaluation_count", 0))
        state.cache_hit_count = int(payload.get("cache_hit_count", 0))
        state.frontier_pop_count = int(payload.get("frontier_pop_count", 0))
        state.stage = payload.get("stage", PRIOR_SINGLETON_STAGE)
        state.prior_lane_index = int(payload.get("prior_lane_index", 0))
        state.prior_index = int(payload.get("prior_index", 0))
        state.frontier_entries = list(payload.get("frontier_entries", []))
        state.frontier_seen_signatures = list(payload.get("frontier_seen_signatures", []))
        state.singleton_raw = {
            lane: {marker: float(value) for marker, value in payload.get("singleton_raw", {}).get(lane, {}).items()}
            for lane in LANE_ORDER
        }
        state.singleton_prior = {
            lane: {marker: float(value) for marker, value in payload.get("singleton_prior", {}).get(lane, {}).items()}
            for lane in LANE_ORDER
        }
        state.pair_raw = {
            lane: {key: float(value) for key, value in payload.get("pair_raw", {}).get(lane, {}).items()}
            for lane in LANE_ORDER
        }
        state.pair_prior = {
            lane: {key: float(value) for key, value in payload.get("pair_prior", {}).get(lane, {}).items()}
            for lane in LANE_ORDER
        }
        state.best_by_size = {
            lane: {str(key): dict(value) for key, value in payload.get("best_by_size", {}).get(lane, {}).items()}
            for lane in LANE_ORDER
        }
        state.size_visit_counts = {
            lane: {str(key): int(value) for key, value in payload.get("size_visit_counts", {}).get(lane, {}).items()}
            for lane in LANE_ORDER
        }
        return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run best-first BioAge search with low-order priors.")
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing best-first checkpoint.")
    parser.add_argument(
        "--prior-tier",
        choices=TIER_ORDER,
        default=DEFAULT_PRIOR_TIER,
        help=f"Tier used to build singleton/pair priors (default: {DEFAULT_PRIOR_TIER}).",
    )
    parser.add_argument(
        "--search-tier",
        choices=TIER_ORDER,
        default=DEFAULT_SEARCH_TIER,
        help=f"Tier used for best-first frontier evaluations (default: {DEFAULT_SEARCH_TIER}).",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=DEFAULT_MAX_EVALUATIONS,
        help=f"Maximum new evaluations to execute in this invocation (default: {DEFAULT_MAX_EVALUATIONS}).",
    )
    parser.add_argument(
        "--max-frontier-pops",
        type=int,
        default=DEFAULT_MAX_FRONTIER_POPS,
        help=f"Maximum frontier items to pop in this invocation (default: {DEFAULT_MAX_FRONTIER_POPS}).",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=DEFAULT_STATUS_EVERY,
        help=f"Write checkpoint/status after this many new evaluations (default: {DEFAULT_STATUS_EVERY}).",
    )
    parser.add_argument(
        "--seed-singletons",
        type=int,
        default=DEFAULT_SEED_SINGLETONS,
        help=f"How many top singleton seeds to enqueue per lane (default: {DEFAULT_SEED_SINGLETONS}).",
    )
    parser.add_argument(
        "--seed-pairs",
        type=int,
        default=DEFAULT_SEED_PAIRS,
        help=f"How many top pair seeds to enqueue per lane (default: {DEFAULT_SEED_PAIRS}).",
    )
    parser.add_argument(
        "--add-children",
        type=int,
        default=DEFAULT_ADD_CHILDREN,
        help=f"Maximum add-one children per expansion (default: {DEFAULT_ADD_CHILDREN}).",
    )
    parser.add_argument(
        "--swap-children",
        type=int,
        default=DEFAULT_SWAP_CHILDREN,
        help=f"Maximum swap-one children per expansion (default: {DEFAULT_SWAP_CHILDREN}).",
    )
    parser.add_argument(
        "--shrink-children",
        type=int,
        default=DEFAULT_SHRINK_CHILDREN,
        help=f"Maximum shrink-one children per expansion (default: {DEFAULT_SHRINK_CHILDREN}).",
    )
    parser.add_argument(
        "--underexplored-size-weight",
        type=float,
        default=DEFAULT_UNDEREXPLORED_SIZE_WEIGHT,
        help=(
            "Exploration bonus applied to under-visited sizes "
            f"(default: {DEFAULT_UNDEREXPLORED_SIZE_WEIGHT})."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"Checkpoint path for this mode (default: {DEFAULT_STATE_PATH}).",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=DEFAULT_STATUS_PATH,
        help=f"Status snapshot path for this mode (default: {DEFAULT_STATUS_PATH}).",
    )
    parser.add_argument(
        "--no-results-log",
        action="store_true",
        help="Do not append rows to results.tsv. Useful for smoke tests.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny best-first search using the cache and without writing results.tsv.",
    )
    return parser.parse_args()


def pair_key(values: tuple[str, str] | list[str] | tuple[str, ...]) -> str:
    first, second = sorted(values)
    return f"{first};{second}"


def make_candidate_id(state: BestFirstState) -> str:
    candidate_id = f"cand_{state.next_candidate_index:05d}"
    state.next_candidate_index += 1
    return candidate_id


def load_or_create_state(path: Path, fresh: bool) -> BestFirstState:
    if fresh or not path.exists():
        return BestFirstState(next_candidate_index=max_result_candidate_index() + 1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = BestFirstState.from_dict(payload)
    state.next_candidate_index = max(state.next_candidate_index, max_result_candidate_index() + 1)
    return state


def save_state(path: Path, state: BestFirstState) -> None:
    path.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")


def lane_feature_count(lane: str, biomarkers: tuple[str, ...]) -> int:
    return len(biomarkers) + (1 if LANE_CONFIGS[lane].include_age else 0)


def rank_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        key, _ = ordered[0]
        return {key: 1.0}
    normalized: dict[str, float] = {}
    last_index = len(ordered) - 1
    for index, (key, _) in enumerate(ordered):
        normalized[key] = index / last_index
    return normalized


def normalize_score(value: float, floor: float, ceiling: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if ceiling <= floor:
        return 0.5
    clipped = max(floor, min(ceiling, value))
    return (clipped - floor) / (ceiling - floor)


def update_prior_maps(state: BestFirstState) -> None:
    for lane in LANE_ORDER:
        state.singleton_prior[lane] = rank_normalize(state.singleton_raw[lane])
        compatibility_raw: dict[str, float] = {}
        for key, pair_score in state.pair_raw[lane].items():
            left, right = key.split(";")
            left_single = state.singleton_raw[lane].get(left)
            right_single = state.singleton_raw[lane].get(right)
            if left_single is None or right_single is None:
                continue
            compatibility_raw[key] = pair_score - ((left_single + right_single) / 2.0)
        state.pair_prior[lane] = rank_normalize(compatibility_raw)


def candidate_priority_score(
    state: BestFirstState,
    lane: str,
    biomarkers: tuple[str, ...],
    *,
    parent_score: float | None = None,
    underexplored_size_weight: float,
) -> float:
    singleton_values = [state.singleton_prior[lane].get(marker, 0.0) for marker in biomarkers]
    singleton_component = sum(singleton_values) / max(1, len(singleton_values))

    pair_values = [
        state.pair_prior[lane].get(pair_key(pair), 0.0)
        for pair in combinations(biomarkers, 2)
    ]
    pair_component = sum(pair_values) / len(pair_values) if pair_values else singleton_component
    strongest_pair_component = max(pair_values) if pair_values else singleton_component

    raw_singleton_values = list(state.singleton_raw[lane].values())
    raw_pair_values = list(state.pair_raw[lane].values())
    raw_floor_candidates = raw_singleton_values + raw_pair_values
    raw_ceiling_candidates = raw_singleton_values + raw_pair_values
    raw_floor = min(raw_floor_candidates) if raw_floor_candidates else 0.0
    raw_ceiling = max(raw_ceiling_candidates) if raw_ceiling_candidates else 1.0
    observed_component = 0.0 if parent_score is None else normalize_score(parent_score, raw_floor, raw_ceiling)

    feature_count = lane_feature_count(lane, biomarkers)
    size_key = str(feature_count)
    size_visits = state.size_visit_counts[lane].get(size_key, 0)
    size_bonus = underexplored_size_weight / math.sqrt(1.0 + size_visits)

    return (
        0.45 * singleton_component
        + 0.35 * pair_component
        + 0.10 * strongest_pair_component
        + 0.10 * observed_component
        + size_bonus
    )


def score_marker_addition(
    state: BestFirstState,
    lane: str,
    biomarkers: tuple[str, ...],
    marker: str,
) -> float:
    if marker in biomarkers:
        return -1.0
    singleton_part = state.singleton_prior[lane].get(marker, 0.0)
    pair_values = [
        state.pair_prior[lane].get(pair_key((marker, existing)), 0.0)
        for existing in biomarkers
    ]
    pair_part = sum(pair_values) / len(pair_values) if pair_values else singleton_part
    strongest_pair = max(pair_values) if pair_values else singleton_part
    return 0.50 * singleton_part + 0.35 * pair_part + 0.15 * strongest_pair


def choose_additions(
    state: BestFirstState,
    lane: str,
    biomarkers: tuple[str, ...],
    limit: int,
) -> list[str]:
    candidates = [
        (score_marker_addition(state, lane, biomarkers, marker), marker)
        for marker in CANDIDATE_BIOMARKER_COLUMNS
        if marker not in biomarkers
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [marker for _, marker in candidates[:limit]]


def biomarker_contribution(
    state: BestFirstState,
    lane: str,
    biomarkers: tuple[str, ...],
    marker: str,
) -> float:
    singleton_part = state.singleton_prior[lane].get(marker, 0.0)
    pair_values = [
        state.pair_prior[lane].get(pair_key((marker, other)), 0.0)
        for other in biomarkers
        if other != marker
    ]
    pair_part = sum(pair_values) / len(pair_values) if pair_values else singleton_part
    return 0.5 * singleton_part + 0.5 * pair_part


def build_neighbors(
    state: BestFirstState,
    entry: FrontierEntry,
    args: argparse.Namespace,
    observed_score: float,
) -> list[FrontierEntry]:
    children: list[FrontierEntry] = []
    include_age = LANE_CONFIGS[entry.lane].include_age
    current = entry.biomarkers

    if len(current) < max_biomarker_count(include_age):
        for marker in choose_additions(state, entry.lane, current, args.add_children):
            proposed = tuple(sorted((*current, marker)))
            if not biomarkers_within_feature_cap(include_age, proposed):
                continue
            priority = candidate_priority_score(
                state,
                entry.lane,
                proposed,
                parent_score=observed_score,
                underexplored_size_weight=args.underexplored_size_weight,
            )
            children.append(
                FrontierEntry(
                    lane=entry.lane,
                    biomarkers=proposed,
                    priority=priority,
                    source="add",
                    depth=entry.depth + 1,
                    parent_signature=candidate_signature(entry.lane, current),
                    parent_score=observed_score,
                )
            )

    if len(current) >= 2 and args.swap_children > 0:
        victims = sorted(
            current,
            key=lambda marker: (
                biomarker_contribution(state, entry.lane, current, marker),
                marker,
            ),
        )
        additions = choose_additions(state, entry.lane, current, args.swap_children * 2)
        for victim in victims[: max(1, min(len(victims), args.swap_children))]:
            for marker in additions:
                if marker in current:
                    continue
                proposed = tuple(sorted(marker if value == victim else value for value in current))
                if proposed == current or not biomarkers_within_feature_cap(include_age, proposed):
                    continue
                priority = candidate_priority_score(
                    state,
                    entry.lane,
                    proposed,
                    parent_score=observed_score,
                    underexplored_size_weight=args.underexplored_size_weight,
                )
                children.append(
                    FrontierEntry(
                        lane=entry.lane,
                        biomarkers=proposed,
                        priority=priority,
                        source="swap",
                        depth=entry.depth + 1,
                        parent_signature=candidate_signature(entry.lane, current),
                        parent_score=observed_score,
                    )
                )
                if len([child for child in children if child.source == "swap"]) >= args.swap_children:
                    break
            if len([child for child in children if child.source == "swap"]) >= args.swap_children:
                break

    if len(current) > 1 and args.shrink_children > 0:
        victims = sorted(
            current,
            key=lambda marker: (
                biomarker_contribution(state, entry.lane, current, marker),
                marker,
            ),
        )
        for victim in victims[: args.shrink_children]:
            proposed = tuple(sorted(marker for marker in current if marker != victim))
            priority = candidate_priority_score(
                state,
                entry.lane,
                proposed,
                parent_score=observed_score,
                underexplored_size_weight=args.underexplored_size_weight,
            )
            children.append(
                FrontierEntry(
                    lane=entry.lane,
                    biomarkers=proposed,
                    priority=priority,
                    source="shrink",
                    depth=entry.depth + 1,
                    parent_signature=candidate_signature(entry.lane, current),
                    parent_score=observed_score,
                )
            )

    unique: dict[str, FrontierEntry] = {}
    for child in children:
        signature = candidate_signature(child.lane, child.biomarkers)
        existing = unique.get(signature)
        if existing is None or child.priority > existing.priority:
            unique[signature] = child
    return sorted(unique.values(), key=lambda child: (-child.priority, child.biomarkers))


def record_best_by_size(
    state: BestFirstState,
    lane: str,
    biomarkers: tuple[str, ...],
    tier: str,
    result: TrainingResult,
    *,
    candidate_id: str,
    source: str,
) -> bool:
    feature_count = lane_feature_count(lane, biomarkers)
    key = str(feature_count)
    current = state.best_by_size[lane].get(key)
    replacement = {
        "candidate_id": candidate_id,
        "biomarkers": list(biomarkers),
        "tier": tier,
        "val_cindex": float(result.val_cindex),
        "source": source,
        "timestamp": now_utc(),
    }
    if current is None:
        state.best_by_size[lane][key] = replacement
        return True

    current_tier_index = TIER_ORDER.index(current["tier"])
    candidate_tier_index = TIER_ORDER.index(tier)
    if result.val_cindex > float(current["val_cindex"]) + 1e-12:
        state.best_by_size[lane][key] = replacement
        return True
    elif abs(result.val_cindex - float(current["val_cindex"])) <= 1e-12 and candidate_tier_index > current_tier_index:
        state.best_by_size[lane][key] = replacement
        return True
    return False


def would_update_best_by_size(
    state: BestFirstState,
    lane: str,
    biomarkers: tuple[str, ...],
    tier: str,
    result: TrainingResult,
) -> bool:
    feature_count = lane_feature_count(lane, biomarkers)
    key = str(feature_count)
    current = state.best_by_size[lane].get(key)
    if current is None:
        return True

    current_tier_index = TIER_ORDER.index(current["tier"])
    candidate_tier_index = TIER_ORDER.index(tier)
    if result.val_cindex > float(current["val_cindex"]) + 1e-12:
        return True
    if abs(result.val_cindex - float(current["val_cindex"])) <= 1e-12 and candidate_tier_index > current_tier_index:
        return True
    return False


def evaluate_subset(
    dataset: Any,
    state: BestFirstState,
    result_cache: dict[tuple[str, tuple[str, ...], str, str], TrainingResult],
    commit_hash: str,
    *,
    lane: str,
    biomarkers: tuple[str, ...],
    tier: str,
    source: str,
    rationale: str,
    write_results: bool,
    parent_signature: str | None = None,
) -> tuple[TrainingResult | None, bool, str]:
    biomarkers = tuple(sorted(biomarkers))
    requested_budget_s = LANE_CONFIGS[lane].tier_budgets[tier]
    cache_key = result_cache_key(lane, biomarkers, tier, requested_budget_s)
    cached_result = result_cache.get(cache_key)
    if cached_result is not None:
        state.cache_hit_count += 1
        cached_id = f"cached::{candidate_signature(lane, biomarkers)}"
        return cached_result, True, cached_id

    candidate = Candidate(
        candidate_id=make_candidate_id(state),
        lane=lane,
        biomarkers=biomarkers,
        parent_id=None,
        operator=source,
        seed_family="best_first",
        proposal_arm="best_first",
        rationale=rationale,
        created_round=state.frontier_pop_count,
    )
    try:
        result = train_subset(dataset, candidate, tier, requested_budget_s)
        if budget_failure(result):
            raise RuntimeError(
                f"Budget-control failure: actual {result.training_seconds:.3f}s exceeded request {requested_budget_s:.3f}s."
            )
    except Exception as exc:
        state.evaluation_count += 1
        append_result_row(
            commit_hash=commit_hash,
            candidate=candidate,
            tier=tier,
            status=STATUS_CRASH,
            promotion="crash",
            requested_budget_s=requested_budget_s,
            actual_training_s=0.0,
            val_cindex=0.0,
            peak_vram_mb=0.0,
            write_results=write_results,
            extra_fields={
                "error": str(exc),
                "cache_hit": "false",
                "num_steps": 0,
                "best_step": -1,
                "proposal_arm": "best_first",
                "parent_signature": parent_signature or "root",
                "rationale": rationale.replace(" ", "_"),
            },
        )
        return None, False, candidate.candidate_id

    state.evaluation_count += 1
    retained = would_update_best_by_size(state, lane, biomarkers, tier, result)
    append_result_row(
        commit_hash=commit_hash,
        candidate=candidate,
        tier=tier,
        status=STATUS_KEEP if retained else STATUS_DISCARD,
        promotion="keep_best_by_size" if retained else f"measured_{source}",
        requested_budget_s=requested_budget_s,
        actual_training_s=result.training_seconds,
        val_cindex=result.val_cindex,
        peak_vram_mb=result.peak_vram_mb,
        write_results=write_results,
        extra_fields={
            "error": "none",
            "cache_hit": "false",
            "num_steps": result.num_steps,
            "best_step": result.best_step,
            "proposal_arm": "best_first",
            "parent_signature": parent_signature or "root",
            "rationale": rationale.replace(" ", "_"),
        },
    )
    result.cache_hit = False
    result_cache[cache_key] = result
    return result, False, candidate.candidate_id


def write_status_snapshot(path: Path, state: BestFirstState, args: argparse.Namespace) -> None:
    summary = {
        "timestamp": now_utc(),
        "stage": state.stage,
        "evaluation_count": state.evaluation_count,
        "cache_hit_count": state.cache_hit_count,
        "frontier_pop_count": state.frontier_pop_count,
        "frontier_size": len(state.frontier_entries),
        "prior_tier": args.prior_tier,
        "search_tier": args.search_tier,
        "best_by_size": state.best_by_size,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def print_scoreboard(state: BestFirstState, tier: str) -> None:
    print(f"Best-by-size scoreboard ({tier} search tier):")
    for lane in LANE_ORDER:
        print(f"  lane={lane}")
        winners = state.best_by_size[lane]
        if not winners:
            print("    no winners yet")
            continue
        for size_key in sorted(winners, key=lambda value: int(value)):
            entry = winners[size_key]
            names = ", ".join(INPUT_DISPLAY_NAMES.get(marker, marker) for marker in entry["biomarkers"])
            print(
                f"    size={size_key} c-index={entry['val_cindex']:.6f} "
                f"tier={entry['tier']} source={entry['source']} biomarkers=[{names}]"
            )


def rebuild_heap(frontier_entries: list[dict[str, Any]]) -> list[tuple[float, int, FrontierEntry]]:
    heap: list[tuple[float, int, FrontierEntry]] = []
    for index, payload in enumerate(frontier_entries):
        entry = FrontierEntry.from_dict(payload)
        heapq.heappush(heap, (-entry.priority, index, entry))
    return heap


def serialize_heap(heap: list[tuple[float, int, FrontierEntry]]) -> list[dict[str, Any]]:
    entries = [entry.to_dict() for _, _, entry in sorted(heap, key=lambda item: (item[0], item[1]))]
    entries.sort(key=lambda payload: (-float(payload["priority"]), payload["lane"], payload["biomarkers"]))
    return entries


def enqueue_frontier(
    state: BestFirstState,
    heap: list[tuple[float, int, FrontierEntry]],
    counter: list[int],
    entry: FrontierEntry,
) -> None:
    signature = candidate_signature(entry.lane, entry.biomarkers)
    seen = set(state.frontier_seen_signatures)
    if signature in seen:
        return
    state.frontier_seen_signatures.append(signature)
    heapq.heappush(heap, (-entry.priority, counter[0], entry))
    counter[0] += 1


def seed_frontier(
    state: BestFirstState,
    heap: list[tuple[float, int, FrontierEntry]],
    counter: list[int],
    args: argparse.Namespace,
) -> None:
    if heap:
        return

    for lane in LANE_ORDER:
        top_singletons = sorted(
            state.singleton_prior[lane].items(),
            key=lambda item: (-item[1], item[0]),
        )[: args.seed_singletons]
        for marker, _ in top_singletons:
            biomarkers = (marker,)
            enqueue_frontier(
                state,
                heap,
                counter,
                FrontierEntry(
                    lane=lane,
                    biomarkers=biomarkers,
                    priority=candidate_priority_score(
                        state,
                        lane,
                        biomarkers,
                        underexplored_size_weight=args.underexplored_size_weight,
                    ),
                    source="seed_singleton",
                    depth=0,
                ),
            )

        top_pairs = sorted(
            state.pair_raw[lane].items(),
            key=lambda item: (-item[1], item[0]),
        )[: args.seed_pairs]
        for key, _ in top_pairs:
            biomarkers = tuple(key.split(";"))
            enqueue_frontier(
                state,
                heap,
                counter,
                FrontierEntry(
                    lane=lane,
                    biomarkers=biomarkers,
                    priority=candidate_priority_score(
                        state,
                        lane,
                        biomarkers,
                        underexplored_size_weight=args.underexplored_size_weight,
                    ),
                    source="seed_pair",
                    depth=0,
                ),
            )

        reference_subset = tuple(sorted(REFERENCE_PHENOAGE_BIOMARKERS))
        if biomarkers_within_feature_cap(LANE_CONFIGS[lane].include_age, reference_subset):
            enqueue_frontier(
                state,
                heap,
                counter,
                FrontierEntry(
                    lane=lane,
                    biomarkers=reference_subset,
                    priority=candidate_priority_score(
                        state,
                        lane,
                        reference_subset,
                        underexplored_size_weight=args.underexplored_size_weight,
                    ),
                    source="seed_reference",
                    depth=0,
                ),
            )

        # Diverse medium-size seeds from biological groups help the queue avoid
        # only chasing the strongest pair families.
        for group_name, markers in BIOLOGICAL_GROUPS.items():
            allowed = [marker for marker in markers if marker in state.singleton_prior[lane]]
            if len(allowed) < 2:
                continue
            allowed.sort(key=lambda marker: (-state.singleton_prior[lane].get(marker, 0.0), marker))
            subset_size = min(3, max_biomarker_count(LANE_CONFIGS[lane].include_age), len(allowed))
            candidate_markers = tuple(sorted(allowed[:subset_size]))
            if not biomarkers_within_feature_cap(LANE_CONFIGS[lane].include_age, candidate_markers):
                continue
            enqueue_frontier(
                state,
                heap,
                counter,
                FrontierEntry(
                    lane=lane,
                    biomarkers=candidate_markers,
                    priority=candidate_priority_score(
                        state,
                        lane,
                        candidate_markers,
                        underexplored_size_weight=args.underexplored_size_weight,
                    ),
                    source=f"seed_group_{group_name}",
                    depth=0,
                ),
            )


def build_prior_job_list(kind: str) -> list[tuple[str, ...]]:
    if kind == PRIOR_SINGLETON_STAGE:
        return [(marker,) for marker in CANDIDATE_BIOMARKER_COLUMNS]
    if kind == PRIOR_PAIR_STAGE:
        return list(combinations(CANDIDATE_BIOMARKER_COLUMNS, 2))
    raise ValueError(f"Unexpected prior stage: {kind}")


def run_prior_stage(
    dataset: Any,
    state: BestFirstState,
    args: argparse.Namespace,
    result_cache: dict[tuple[str, tuple[str, ...], str, str], TrainingResult],
    commit_hash: str,
    *,
    evaluation_limit: int,
) -> bool:
    progress_made = False
    next_status_target = state.evaluation_count + max(1, args.status_every)
    while state.stage in {PRIOR_SINGLETON_STAGE, PRIOR_PAIR_STAGE} and state.evaluation_count < evaluation_limit:
        jobs = build_prior_job_list(state.stage)
        if state.prior_lane_index >= len(LANE_ORDER):
            update_prior_maps(state)
            if state.stage == PRIOR_SINGLETON_STAGE:
                state.stage = PRIOR_PAIR_STAGE
                state.prior_lane_index = 0
                state.prior_index = 0
                continue
            state.stage = SEARCH_STAGE
            state.prior_lane_index = 0
            state.prior_index = 0
            break

        lane = LANE_ORDER[state.prior_lane_index]
        if state.prior_index >= len(jobs):
            state.prior_lane_index += 1
            state.prior_index = 0
            continue

        biomarkers = tuple(sorted(jobs[state.prior_index]))
        source = "prior_singleton" if state.stage == PRIOR_SINGLETON_STAGE else "prior_pair"
        result, _, _ = evaluate_subset(
            dataset,
            state,
            result_cache,
            commit_hash,
            lane=lane,
            biomarkers=biomarkers,
            tier=args.prior_tier,
            source=source,
            rationale=f"Build {source} low-order prior.",
            write_results=not args.no_results_log,
        )
        if result is not None:
            if len(biomarkers) == 1:
                state.singleton_raw[lane][biomarkers[0]] = float(result.val_cindex)
            else:
                state.pair_raw[lane][pair_key(biomarkers)] = float(result.val_cindex)
            record_best_by_size(
                state,
                lane,
                biomarkers,
                args.prior_tier,
                result,
                candidate_id="prior::" + candidate_signature(lane, biomarkers),
                source=source,
            )
        state.prior_index += 1
        progress_made = True

        if state.evaluation_count >= next_status_target:
            update_prior_maps(state)
            return True

    update_prior_maps(state)
    return progress_made


def run_search_stage(
    dataset: Any,
    state: BestFirstState,
    args: argparse.Namespace,
    result_cache: dict[tuple[str, tuple[str, ...], str, str], TrainingResult],
    commit_hash: str,
    *,
    evaluation_limit: int,
    frontier_pop_limit: int,
) -> bool:
    heap = rebuild_heap(state.frontier_entries)
    counter = [len(heap)]
    seed_frontier(state, heap, counter, args)
    progress_made = False
    next_status_target = state.evaluation_count + max(1, args.status_every)

    while heap and state.frontier_pop_count < frontier_pop_limit and state.evaluation_count < evaluation_limit:
        _, _, entry = heapq.heappop(heap)
        state.frontier_pop_count += 1
        size_key = str(lane_feature_count(entry.lane, entry.biomarkers))
        state.size_visit_counts[entry.lane][size_key] = state.size_visit_counts[entry.lane].get(size_key, 0) + 1

        result, _, candidate_id = evaluate_subset(
            dataset,
            state,
            result_cache,
            commit_hash,
            lane=entry.lane,
            biomarkers=entry.biomarkers,
            tier=args.search_tier,
            source=entry.source,
            rationale="Best-first frontier evaluation.",
            write_results=not args.no_results_log,
            parent_signature=entry.parent_signature,
        )
        if result is None:
            progress_made = True
            continue

        progress_made = True
        record_best_by_size(
            state,
            entry.lane,
            entry.biomarkers,
            args.search_tier,
            result,
            candidate_id=candidate_id,
            source=entry.source,
        )
        for child in build_neighbors(state, entry, args, float(result.val_cindex)):
            enqueue_frontier(state, heap, counter, child)

        if state.evaluation_count >= next_status_target:
            break

    state.frontier_entries = serialize_heap(heap)
    return progress_made


def configure_smoke_test(args: argparse.Namespace) -> None:
    args.max_evaluations = min(args.max_evaluations, 6)
    args.max_frontier_pops = min(args.max_frontier_pops, 16)
    args.seed_singletons = min(args.seed_singletons, 3)
    args.seed_pairs = min(args.seed_pairs, 4)
    args.add_children = min(args.add_children, 3)
    args.swap_children = min(args.swap_children, 2)
    args.shrink_children = min(args.shrink_children, 1)
    args.no_results_log = True


def run_best_first_search(args: argparse.Namespace) -> None:
    if args.smoke_test:
        configure_smoke_test(args)

    dataset = build_dataset_bundle()
    state = load_or_create_state(args.checkpoint_path, args.fresh)
    result_cache = load_result_cache()
    commit_hash = short_git_hash()

    any_progress = False
    while True:
        evaluation_limit = state.evaluation_count + args.max_evaluations
        frontier_pop_limit = state.frontier_pop_count + args.max_frontier_pops
        prior_progress = False
        search_progress = False

        if state.stage in {PRIOR_SINGLETON_STAGE, PRIOR_PAIR_STAGE} and state.evaluation_count < evaluation_limit:
            prior_progress = run_prior_stage(
                dataset,
                state,
                args,
                result_cache,
                commit_hash,
                evaluation_limit=evaluation_limit,
            )

        if (
            state.stage == SEARCH_STAGE
            and state.evaluation_count < evaluation_limit
            and state.frontier_pop_count < frontier_pop_limit
        ):
            search_progress = run_search_stage(
                dataset,
                state,
                args,
                result_cache,
                commit_hash,
                evaluation_limit=evaluation_limit,
                frontier_pop_limit=frontier_pop_limit,
            )

        save_state(args.checkpoint_path, state)
        write_status_snapshot(args.status_path, state, args)

        any_progress = any_progress or prior_progress or search_progress
        if args.smoke_test:
            break
        if not prior_progress and not search_progress:
            break

    print_scoreboard(state, args.search_tier)

    if not any_progress:
        print("No new work was required; cache and frontier were already up to date for this invocation.")
    elif state.stage == SEARCH_STAGE and not state.frontier_entries:
        print("Best-first search exhausted its current frontier and exited cleanly.")


def main() -> None:
    run_best_first_search(parse_args())


if __name__ == "__main__":
    main()
