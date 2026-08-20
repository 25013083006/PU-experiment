from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_partition(rows: list[dict[str, str]], partitions: dict[str, list[str]], strict: bool) -> None:
    ids = set()
    for name, sample_ids in partitions.items():
        if ids.intersection(sample_ids):
            raise ValueError(f"sample leakage in partition {name}")
        ids.update(sample_ids)
        selected = [row for row in rows if row["sample_id"] in set(sample_ids)]
        if len({row["label"] for row in selected}) != 3:
            raise ValueError(f"partition {name} does not contain all labels")
        if strict and len({row["bearing_id"] for row in selected}) == 0:
            raise ValueError(f"empty strict partition {name}")
    if ids != {row["sample_id"] for row in rows}:
        raise ValueError("split does not cover all sample IDs")
    if strict:
        bearing_sets = [{row["bearing_id"] for row in rows if row["sample_id"] in set(ids_)} for ids_ in partitions.values()]
        if any(bearing_sets[i].intersection(bearing_sets[j]) for i in range(len(bearing_sets)) for j in range(i)):
            raise ValueError("bearing leakage")
        source_sets = [{row["source_file"] for row in rows if row["sample_id"] in set(ids_)} for ids_ in partitions.values()]
        if any(source_sets[i].intersection(source_sets[j]) for i in range(len(source_sets)) for j in range(i)):
            raise ValueError("source leakage")


def build_reproduction(rows: list[dict[str, str]], output: str | Path, seed: int = 2026) -> dict[str, Any]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        grouped.setdefault((row["bearing_id"], row["label"]), []).append(row["sample_id"])
    parts = {"train": [], "val": [], "test": []}
    for sample_ids in grouped.values():
        rng.shuffle(sample_ids)
        n = len(sample_ids)
        cuts = (n * 3) // 5, (n * 4) // 5
        parts["train"].extend(sample_ids[:cuts[0]])
        parts["val"].extend(sample_ids[cuts[0]:cuts[1]])
        parts["test"].extend(sample_ids[cuts[1]:])
    for value in parts.values(): value.sort()
    _validate_partition(rows, parts, strict=False)
    payload = {"type": "reproduction/window-random", "seed": seed, "partitions": parts}
    _write(Path(output), payload)
    return payload


def build_strict(rows: list[dict[str, str]], output_dir: str | Path) -> list[dict[str, Any]]:
    groups = [("K001", "KA04", "KI04"), ("K002", "KA15", "KI14"), ("K003", "KA16", "KI16")]
    folds = []
    for index in range(3):
        train = groups[index]
        val = groups[(index + 1) % 3]
        test = groups[(index + 2) % 3]
        partitions = {name: sorted(row["sample_id"] for row in rows if row["bearing_id"] in bearings) for name, bearings in (("train", train), ("val", val), ("test", test))}
        _validate_partition(rows, partitions, strict=True)
        payload = {"type": "strict/bearing-independent", "fold": index, "partitions": partitions, "bearings": {"train": list(train), "val": list(val), "test": list(test)}}
        _write(Path(output_dir) / f"split_strict_fold{index}.json", payload)
        folds.append(payload)
    return folds
