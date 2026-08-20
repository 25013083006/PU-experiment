from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.data.pu_mat import inspect_mat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    for path in sorted(args.dataset_root.rglob("*.mat"))[:args.limit]:
        print(inspect_mat(path))


if __name__ == "__main__":
    main()
