from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import read_csv
from .pu_mat import load_target_signals

CHANNELS = ("vibration_1", "phase_current_1", "phase_current_2")


class SignalCache:
    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, source_file: str) -> dict[str, np.ndarray]:
        if source_file in self._items:
            value = self._items.pop(source_file)
            self._items[source_file] = value
            return value
        value = load_target_signals(source_file)
        self._items[source_file] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value


def _window(row: dict[str, str], cache: SignalCache) -> tuple[np.ndarray, np.ndarray]:
    signals = cache.get(row["source_file"])
    start, end = int(row["window_start"]), int(row["window_end"])
    vibration = signals["vibration_1"][start:end][None, :]
    current = np.stack([signals["phase_current_1"][start:end], signals["phase_current_2"][start:end]], axis=0)
    return vibration, current


def compute_normalization(rows: Iterable[dict[str, str]], cache_capacity: int = 16) -> dict[str, list[float]]:
    cache = SignalCache(cache_capacity)
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = 0
    for row in rows:
        vibration, current = _window(row, cache)
        values = (vibration[0], current[0], current[1])
        for index, value in enumerate(values):
            sums[index] += float(value.sum())
            squares[index] += float(np.square(value, dtype=np.float64).sum())
        count += vibration.shape[-1]
    mean = sums / count
    variance = np.maximum(squares / count - np.square(mean), 1e-12)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(), "count": [count]}


class PUSignalDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], normalization: dict[str, list[float]], mode: str, cache_capacity: int = 16) -> None:
        if mode not in {"vibration", "current", "fusion"}:
            raise ValueError(f"unknown baseline mode: {mode}")
        self.rows = rows
        self.mode = mode
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        self.cache = SignalCache(cache_capacity)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        vibration, current = _window(row, self.cache)
        vibration = (vibration - self.mean[0]) / self.std[0]
        current = (current - self.mean[1:, None]) / self.std[1:, None]
        result: dict[str, object] = {"label": torch.tensor(int(row["label"]), dtype=torch.long), "sample_id": row["sample_id"], "metadata": row}
        if self.mode in {"vibration", "fusion"}:
            result["vibration"] = torch.from_numpy(vibration.copy())
        if self.mode in {"current", "fusion"}:
            result["current"] = torch.from_numpy(current.copy())
        return result


class CachedPUSignalDataset(Dataset):
    def __init__(self, cache_path: str | Path, mode: str) -> None:
        if mode not in {"vibration", "current", "fusion"}:
            raise ValueError(f"unknown baseline mode: {mode}")
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"cache file does not exist: {self.cache_path}")
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.vibration = payload["signals"]["vibration"]
        self.current = payload["signals"]["current"]
        self.labels = payload["labels"]
        self.sample_ids = payload["sample_ids"]
        self.metadata = payload["metadata"]
        self.mode = mode
        if self.vibration.ndim != 3 or tuple(self.vibration.shape[1:]) != (1, 2048):
            raise ValueError(f"invalid cached vibration shape: {tuple(self.vibration.shape)}")
        if self.current.ndim != 3 or tuple(self.current.shape[1:]) != (2, 2048):
            raise ValueError(f"invalid cached current shape: {tuple(self.current.shape)}")
        if len(self.labels) != len(self.sample_ids) or len(self.labels) != len(self.metadata):
            raise ValueError("cached tensors and metadata have inconsistent lengths")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        result: dict[str, object] = {"label": self.labels[index], "sample_id": self.sample_ids[index], "metadata": self.metadata[index]}
        if self.mode in {"vibration", "fusion"}:
            result["vibration"] = self.vibration[index]
        if self.mode in {"current", "fusion"}:
            result["current"] = self.current[index]
        return result


def load_split_rows(project_root: str | Path, split_path: str | Path, partition: str) -> list[dict[str, str]]:
    import json
    root = Path(project_root)
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    ids = set(split["partitions"][partition])
    rows = read_csv(root / "data" / "manifests" / "sample_manifest.csv")
    selected = [row for row in rows if row["sample_id"] in ids]
    if len(selected) != len(ids):
        raise ValueError("split references unknown sample IDs")
    return selected


def pu_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"label": torch.stack([item["label"] for item in batch]), "sample_id": [item["sample_id"] for item in batch], "metadata": [item["metadata"] for item in batch]}
    for key in ("vibration", "current"):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    return result
