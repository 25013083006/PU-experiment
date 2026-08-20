from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from pathlib import Path
import yaml
from src.data.manifest import build_manifests, manifest_hash
from src.utils.hashing import sha256_file
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records, samples = build_manifests(config["dataset_root"], Path(__file__).parents[1] / "data" / "manifests", config["window_length"], config["windows_per_file"])
    manifest_dir = Path(__file__).parents[1] / "data" / "manifests"
    metadata = {"config": str(args.config.resolve()), "config_sha256": sha256_file(args.config), "file_manifest_sha256": manifest_hash(manifest_dir / "file_manifest.csv"), "sample_manifest_sha256": manifest_hash(manifest_dir / "sample_manifest.csv"), "file_count": len(records), "accepted_file_count": sum(bool(row.get("accepted")) for row in records), "sample_count": len(samples)}
    (manifest_dir / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"files={len(records)} accepted={sum(bool(row.get('accepted')) for row in records)} samples={len(samples)}")


if __name__ == "__main__":
    main()
