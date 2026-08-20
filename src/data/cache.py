from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .dataset import _window, SignalCache
from .manifest import read_csv


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_by_ids(rows: list[dict[str, str]], sample_ids: Iterable[str]) -> list[dict[str, str]]:
    wanted = set(sample_ids)
    selected = [row for row in rows if row["sample_id"] in wanted]
    if len(selected) != len(wanted) or len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("split references missing or duplicate sample IDs")
    return selected


def _normalization(rows: list[dict[str, str]], cache_capacity: int = 1) -> dict[str, object]:
    cache = SignalCache(cache_capacity)
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = 0
    for row in rows:
        vibration, current = _window(row, cache)
        values = (vibration[0], current[0], current[1])
        for index, value in enumerate(values):
            sums[index] += float(value.sum(dtype=np.float64))
            squares[index] += float(np.square(value, dtype=np.float64).sum())
        count += vibration.shape[-1]
    mean = sums / count
    variance = np.maximum(squares / count - np.square(mean), 1e-12)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(), "count": int(count)}


def _materialize(rows: list[dict[str, str]], normalization: dict[str, object], cache_capacity: int = 1) -> dict[str, object]:
    cache = SignalCache(cache_capacity)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    vibrations, currents, labels, sample_ids, metadata = [], [], [], [], []
    for row in rows:
        vibration, current = _window(row, cache)
        vibration = ((vibration - mean[0]) / std[0]).astype(np.float32, copy=False)
        current = ((current - mean[1:, None]) / std[1:, None]).astype(np.float32, copy=False)
        vibrations.append(vibration)
        currents.append(current)
        labels.append(int(row["label"]))
        sample_ids.append(row["sample_id"])
        metadata.append(row)
    if not rows:
        raise ValueError("cannot materialize an empty split")
    return {
        "signals": {"vibration": torch.from_numpy(np.stack(vibrations)), "current": torch.from_numpy(np.stack(currents))},
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": sample_ids,
        "metadata": metadata,
        "normalization": normalization,
    }


def build_cache_for_split(project_root: str | Path, split_name: str, split_path: str | Path, output_dir: str | Path, config_path: str | Path, cache_capacity: int = 1) -> dict[str, object]:
    root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(root / "data" / "manifests" / "sample_manifest.csv")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    split_hash = sha256_file(split_path)
    manifest_hash = sha256_file(root / "data" / "manifests" / "sample_manifest.csv")
    config_hash = sha256_file(config_path)
    train_rows = _rows_by_ids(rows, split["partitions"]["train"])
    normalization = _normalization(train_rows, cache_capacity)
    norm_path = output_dir / f"{split_name}_normalization.json"
    norm_path.write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    result = {"split": split_name, "normalization_file": norm_path.name, "partitions": {}}
    for partition in ("train", "val", "test"):
        partition_rows = _rows_by_ids(rows, split["partitions"][partition])
        payload = _materialize(partition_rows, normalization, cache_capacity)
        payload.update({"source_manifest_sha256": manifest_hash, "source_split_sha256": split_hash, "config_sha256": config_hash, "window_length": int(partition_rows[0]["window_end"]) - int(partition_rows[0]["window_start"]), "split_name": split_name, "partition": partition})
        output_path = output_dir / f"{split_name}_{partition}.pt"
        torch.save(payload, output_path)
        result["partitions"][partition] = {"file": output_path.name, "sha256": sha256_file(output_path), "count": len(partition_rows), "vibration_shape": list(payload["signals"]["vibration"].shape), "current_shape": list(payload["signals"]["current"].shape)}
    return result


def build_all_caches(project_root: str | Path, config_path: str | Path) -> dict[str, object]:
    root = Path(project_root)
    splits_dir = root / "data" / "splits"
    output_dir = root / "data" / "processed"
    jobs = [("reproduction", splits_dir / "split_reproduction.json")]
    jobs.extend((f"strict_fold{index}", splits_dir / f"split_strict_fold{index}.json") for index in range(3))
    metadata = {"config": str(Path(config_path).resolve()), "config_sha256": sha256_file(config_path), "caches": {}}
    for split_name, split_path in jobs:
        metadata["caches"][split_name] = build_cache_for_split(root, split_name, split_path, output_dir, config_path)
    (output_dir / "cache_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
