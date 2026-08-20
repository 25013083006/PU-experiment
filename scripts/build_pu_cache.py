from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.cache import build_all_caches


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize normalized PU windows into torch caches.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    metadata = build_all_caches(root, args.config)
    for name, value in metadata["caches"].items():
        counts = {partition: info["count"] for partition, info in value["partitions"].items()}
        print(f"{name}: {counts}")


if __name__ == "__main__":
    main()
