from __future__ import annotations

from pathlib import Path

from results_ledger import RESULTS_PATH, load_rows


def metric_value(row: dict[str, str]) -> float:
    raw = row.get("development_cindex")
    if raw is None:
        raw = row["val_cindex"]
    return float(raw)


def format_row(row: dict[str, str]) -> str:
    candidate_suffix = ""
    if row.get("candidate_path"):
        candidate_suffix = f" | candidate={Path(row['candidate_path']).name}"
    baseline_raw = row.get("baseline_cindex", "")
    delta_raw = row.get("delta_cindex", "")
    comparison = ""
    if baseline_raw:
        comparison = f" | baseline={float(baseline_raw):.6f}"
    if delta_raw:
        comparison += f" | delta={float(delta_raw):+.6f}"
    return (
        f"{row['status']:>7} | development={metric_value(row):.6f} | mem_gb={row['memory_gb']} "
        f"| commit={row['commit']}{comparison}{candidate_suffix} | {row['description']}"
    )


def main() -> None:
    rows = load_rows()
    if not rows:
        print("No experiment rows found in results.tsv yet.")
        return

    parsed = [
        {
            **row,
            "_metric": metric_value(row),
        }
        for row in rows
    ]
    saved_rows = [row for row in parsed if row["status"] == "saved"]
    keep_rows = [row for row in parsed if row["status"] == "keep"]
    best_overall = max(parsed, key=lambda row: row["_metric"])
    best_saved = max(saved_rows, key=lambda row: row["_metric"]) if saved_rows else None
    best_keep = max(keep_rows, key=lambda row: row["_metric"]) if keep_rows else None
    recent = parsed[-5:]

    print(f"Total runs:        {len(parsed)}")
    print(
        f"Saved/keep/discard/crash:"
        f"{sum(r['status']=='saved' for r in parsed)}/"
        f"{sum(r['status']=='keep' for r in parsed)}/"
        f"{sum(r['status']=='discard' for r in parsed)}/"
        f"{sum(r['status']=='crash' for r in parsed)}"
    )
    print("")
    print("Best overall:")
    print(format_row(best_overall))
    print("")
    if best_saved is None:
        print("Best saved:        none yet")
    else:
        print("Best saved:")
        print(format_row(best_saved))
    print("")
    if best_keep is None:
        print("Best kept:         none yet")
    else:
        print("Best kept:")
        print(format_row(best_keep))
    print("")
    print("Recent runs:")
    for row in recent:
        print(format_row(row))


if __name__ == "__main__":
    main()
