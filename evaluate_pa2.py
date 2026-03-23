from __future__ import annotations

import argparse
from pathlib import Path

from autoresearch.prepare import (
    DEFAULT_CANDIDATE_MODEL_PATH,
    DEFAULT_RESULT_PATH,
    build_result_summary,
    candidate_feature_columns,
    get_rows_for_split,
    harrell_c_index,
    load_joined_rows,
    load_candidate_metadata,
    score_scripted_model,
    survival_arrays,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current BioAge candidate on the frozen held-out test set."
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

    test_scores = score_scripted_model(args.model, test_rows, device="cpu")

    test_c_index = harrell_c_index(times, events, test_scores)
    metadata = load_candidate_metadata(args.model)
    feature_columns = candidate_feature_columns(args.model)
    result = build_result_summary(test_c_index=test_c_index)
    result.update(
        {
            "benchmark_dataset": "nhanes3-bioage",
            "evaluation_split": "test",
            "participants": len(test_rows),
            "aging_related_deaths": int(events.sum()),
            "candidate_model_path": str(args.model),
            "input_feature_count": len(feature_columns),
            "age_included_in_inputs": bool(metadata["age_included_in_inputs"]),
            "feature_columns": list(feature_columns),
        }
    )

    write_json(args.output, result)

    print("---")
    print(f"test_c_index:     {result['test_c_index']:.6f}")
    print(f"feature_count:    {result['input_feature_count']}")
    print(f"age_included:     {str(result['age_included_in_inputs']).lower()}")
    print(f"output_path:      {args.output}")


if __name__ == "__main__":
    main()
