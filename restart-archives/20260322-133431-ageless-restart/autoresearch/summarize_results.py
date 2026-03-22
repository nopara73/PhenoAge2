from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.tsv"


def load_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def format_row(row: dict[str, str]) -> str:
    return (
        f"{row['status']:>7} | val={row['val_cindex']} | mem_gb={row['memory_gb']} "
        f"| commit={row['commit']} | {row['description']}"
    )


def main() -> None:
    rows = load_rows()
    if not rows:
        print("No experiment rows found in results.tsv yet.")
        return

    parsed = [
        {
            **row,
            "_val": float(row["val_cindex"]),
        }
        for row in rows
    ]
    keep_rows = [row for row in parsed if row["status"] == "keep"]
    best_overall = max(parsed, key=lambda row: row["_val"])
    best_keep = max(keep_rows, key=lambda row: row["_val"]) if keep_rows else None
    recent = parsed[-5:]

    print(f"Total runs:        {len(parsed)}")
    print(f"Keep/discard/crash:{sum(r['status']=='keep' for r in parsed)}/"
          f"{sum(r['status']=='discard' for r in parsed)}/"
          f"{sum(r['status']=='crash' for r in parsed)}")
    print("")
    print("Best overall:")
    print(format_row(best_overall))
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
