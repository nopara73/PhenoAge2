from __future__ import annotations

import argparse
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "train.py"
KEPT_PATH = HERE / "last_kept_train.py"


def save() -> None:
    shutil.copyfile(TRAIN_PATH, KEPT_PATH)
    print(f"Saved kept snapshot to {KEPT_PATH}")


def restore() -> None:
    if not KEPT_PATH.exists():
        raise FileNotFoundError(
            f"No kept snapshot exists at {KEPT_PATH}. Run `python manage_kept.py save` first."
        )
    shutil.copyfile(KEPT_PATH, TRAIN_PATH)
    print(f"Restored train.py from {KEPT_PATH}")


def status() -> None:
    print(f"train.py:          {TRAIN_PATH}")
    print(f"last_kept_train.py:{KEPT_PATH}")
    print(f"snapshot_exists:   {KEPT_PATH.exists()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save or restore the last kept train.py snapshot.")
    parser.add_argument("command", choices=("save", "restore", "status"))
    args = parser.parse_args()

    if args.command == "save":
        save()
    elif args.command == "restore":
        restore()
    else:
        status()


if __name__ == "__main__":
    main()
