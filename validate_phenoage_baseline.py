from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "nhanes3-phenoage"
DEFAULT_INPUT_PATH = DEFAULT_DATA_DIR / "phenoage_baseline.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "phenoage_baseline_report.md"

REQUIRED_COLUMNS = (
    "SEQN",
    "time_months",
    "mortstat",
    "aging_related_event",
    "phenoage",
    "phenoage_advance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a simple validation report for the original PhenoAge baseline "
            "using the joined nhanes3-phenoage cohort."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to phenoage_baseline.csv (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the Markdown report (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row.")
        rows = list(reader)
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        return rows


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        average_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            original_index = indexed[k][0]
            ranks[original_index] = average_rank
        i = j + 1

    return ranks


def roc_auc(scores: list[float], labels: list[int]) -> float:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires at least one positive and one negative example")

    ranks = rankdata(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return auc


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return math.nan

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_products = 0.0
    centered_x = 0.0
    centered_y = 0.0

    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        centered_products += dx * dy
        centered_x += dx * dx
        centered_y += dy * dy

    if centered_x == 0.0 or centered_y == 0.0:
        return math.nan

    return centered_products / math.sqrt(centered_x * centered_y)


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def summarize_metric(rows: list[dict[str, float]], score_key: str, event_key: str) -> dict[str, float]:
    event_rows = [row for row in rows if int(row[event_key]) == 1]
    non_event_rows = [row for row in rows if int(row[event_key]) == 0]

    scores = [float(row[score_key]) for row in rows]
    labels = [int(row[event_key]) for row in rows]

    event_scores = [float(row[score_key]) for row in event_rows]
    non_event_scores = [float(row[score_key]) for row in non_event_rows]

    event_times = [float(row["time_months"]) for row in event_rows]
    top_decile_threshold = statistics.quantiles(scores, n=10, method="inclusive")[8]
    bottom_decile_threshold = statistics.quantiles(scores, n=10, method="inclusive")[0]

    top_decile_rows = [row for row in rows if float(row[score_key]) >= top_decile_threshold]
    bottom_decile_rows = [row for row in rows if float(row[score_key]) <= bottom_decile_threshold]

    return {
        "auc": roc_auc(scores, labels),
        "event_mean": statistics.fmean(event_scores),
        "event_median": statistics.median(event_scores),
        "non_event_mean": statistics.fmean(non_event_scores),
        "non_event_median": statistics.median(non_event_scores),
        "event_time_corr": pearson_correlation(event_scores, event_times),
        "top_decile_threshold": top_decile_threshold,
        "bottom_decile_threshold": bottom_decile_threshold,
        "top_decile_event_rate": statistics.fmean(int(row[event_key]) for row in top_decile_rows),
        "bottom_decile_event_rate": statistics.fmean(int(row[event_key]) for row in bottom_decile_rows),
    }


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_report(rows: list[dict[str, str]], input_path: Path) -> str:
    typed_rows = [
        {
            **row,
            "time_months": float(row["time_months"]),
            "mortstat": int(row["mortstat"]),
            "aging_related_event": int(row["aging_related_event"]),
            "phenoage": float(row["phenoage"]),
            "phenoage_advance": float(row["phenoage_advance"]),
        }
        for row in rows
    ]

    n = len(typed_rows)
    all_cause_deaths = sum(row["mortstat"] for row in typed_rows)
    aging_related_events = sum(row["aging_related_event"] for row in typed_rows)
    follow_up_months = [row["time_months"] for row in typed_rows]

    phenoage_metrics = summarize_metric(typed_rows, "phenoage", "aging_related_event")
    advance_metrics = summarize_metric(typed_rows, "phenoage_advance", "aging_related_event")

    return f"""# PhenoAge Baseline Validation Report

## Cohort

- Input file: `{input_path}`
- Participants: {n}
- All-cause deaths: {all_cause_deaths}
- Aging-related deaths: {aging_related_events}
- Mean follow-up (months): {format_float(statistics.fmean(follow_up_months))}
- Median follow-up (months): {format_float(statistics.median(follow_up_months))}

## Aging-Related Mortality Discrimination

| Metric | ROC AUC | Mean if event=1 | Median if event=1 | Mean if event=0 | Median if event=0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PhenoAge | {format_float(phenoage_metrics["auc"])} | {format_float(phenoage_metrics["event_mean"])} | {format_float(phenoage_metrics["event_median"])} | {format_float(phenoage_metrics["non_event_mean"])} | {format_float(phenoage_metrics["non_event_median"])} |
| PhenoAge Advance | {format_float(advance_metrics["auc"])} | {format_float(advance_metrics["event_mean"])} | {format_float(advance_metrics["event_median"])} | {format_float(advance_metrics["non_event_mean"])} | {format_float(advance_metrics["non_event_median"])} |

## Time-To-Event Sanity Check

These correlations are computed only among participants with `aging_related_event = 1`. More positive score should generally correspond to shorter observed follow-up, so a negative correlation is directionally sensible.

| Metric | Pearson corr(score, time_months) among aging-related deaths |
| --- | ---: |
| PhenoAge | {format_float(phenoage_metrics["event_time_corr"])} |
| PhenoAge Advance | {format_float(advance_metrics["event_time_corr"])} |

## Decile Contrast

This compares the aging-related event rate in the highest-score decile versus the lowest-score decile.

| Metric | Bottom decile cutoff | Bottom decile event rate | Top decile cutoff | Top decile event rate |
| --- | ---: | ---: | ---: | ---: |
| PhenoAge | {format_float(phenoage_metrics["bottom_decile_threshold"])} | {format_float(phenoage_metrics["bottom_decile_event_rate"])} | {format_float(phenoage_metrics["top_decile_threshold"])} | {format_float(phenoage_metrics["top_decile_event_rate"])} |
| PhenoAge Advance | {format_float(advance_metrics["bottom_decile_threshold"])} | {format_float(advance_metrics["bottom_decile_event_rate"])} | {format_float(advance_metrics["top_decile_threshold"])} | {format_float(advance_metrics["top_decile_event_rate"])} |

## Notes

- This report benchmarks the original PhenoAge baseline against the repo's `aging_related_event` endpoint.
- ROC AUC treats the endpoint as binary and ignores censoring.
- The time-to-event correlation is only a quick directional check, not a full survival-model evaluation.
"""


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(args.input)
    report = build_report(rows, args.input)
    write_report(args.output, report)
    print(f"Wrote validation report to: {args.output}")


if __name__ == "__main__":
    main()
