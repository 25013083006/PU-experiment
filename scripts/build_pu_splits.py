from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from pathlib import Path
import yaml
from src.data.manifest import read_csv
from src.data.split import build_reproduction, build_strict
from src.utils.hashing import sha256_file
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).parents[1]
    rows = read_csv(root / "data" / "manifests" / "sample_manifest.csv")
    splits = root / "data" / "splits"
    build_reproduction(rows, splits / "split_reproduction.json", config["reproduction_seed"])
    build_strict(rows, splits)
    metadata = {"config": str(args.config.resolve()), "config_sha256": sha256_file(args.config), "sample_manifest_sha256": sha256_file(root / "data" / "manifests" / "sample_manifest.csv"), "sample_count": len(rows), "split_files": sorted(path.name for path in splits.glob("split_*.json") if path.name != "split_metadata.json")}
    (splits / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"samples={len(rows)} splits={splits}")


if __name__ == "__main__":
    main()
