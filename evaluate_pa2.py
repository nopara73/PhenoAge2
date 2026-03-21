from __future__ import annotations

import argparse
from pathlib import Path

from autoresearch.prepare import (
    DEFAULT_CANDIDATE_MODEL_PATH,
    DEFAULT_RESULT_PATH,
    build_result_summary,
    compute_original_phenoage_scores,
    get_rows_for_split,
    harrell_c_index,
    load_joined_rows,
    score_scripted_model,
    survival_arrays,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PA2 candidate against original PhenoAge on the frozen held-out test set."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_CANDIDATE_MODEL_PATH,
        help=f"Path to the scripted PA2 model artifact (default: {DEFAULT_CANDIDATE_MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help=f"Where to write the JSON result summary (default: {DEFAULT_RESULT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Candidate model artifact not found: {args.model}")

    rows = load_joined_rows()
    test_rows = get_rows_for_split(rows, "test")
    times, events = survival_arrays(test_rows)

    phenoage_scores = compute_original_phenoage_scores(test_rows)
    pa2_scores = score_scripted_model(args.model, test_rows, device="cpu")

    phenoage_c_index = harrell_c_index(times, events, phenoage_scores)
    pa2_c_index = harrell_c_index(times, events, pa2_scores)
    result = build_result_summary(pa2_c_index=pa2_c_index, phenoage_c_index=phenoage_c_index)
    result.update(
        {
            "benchmark_dataset": "nhanes3-phenoage",
            "evaluation_split": "test",
            "participants": len(test_rows),
            "aging_related_deaths": int(events.sum()),
            "candidate_model_path": str(args.model),
            "age_excluded_from_pa2_inputs": True,
        }
    )

    write_json(args.output, result)

    print("---")
    print(f"phenoage_c_index: {result['phenoage_c_index']:.6f}")
    print(f"pa2_c_index:      {result['pa2_c_index']:.6f}")
    print(f"delta:            {result['delta']:.6f}")
    print(f"verdict:          {result['verdict']}")
    print(f"output_path:      {args.output}")


if __name__ == "__main__":
    main()
