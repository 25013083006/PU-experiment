from pathlib import Path

import numpy as np
import torch

from src.data.cache import _materialize, _normalization
from src.data.dataset import CachedPUSignalDataset


def fake_rows():
    rows = []
    for index, label in enumerate((0, 1, 2)):
        rows.append({
            "sample_id": f"sample_{index}",
            "source_file": f"source_{index}.mat",
            "bearing_id": f"K00{index + 1}",
            "label": str(label),
            "condition": "N15_M07_F04",
            "run_id": "1",
            "window_start": "0",
            "window_end": "2048",
        })
    return rows


def test_materialized_shapes_and_finite_values(monkeypatch):
    rows = fake_rows()
    signals = {
        row["source_file"]: {
            "vibration_1": np.ones(2048, dtype=np.float32) * (index + 1),
            "phase_current_1": np.ones(2048, dtype=np.float32) * (index + 2),
            "phase_current_2": np.ones(2048, dtype=np.float32) * (index + 3),
        }
        for index, row in enumerate(rows)
    }
    monkeypatch.setattr("src.data.dataset.load_target_signals", lambda path: signals[path])
    normalization = _normalization(rows, cache_capacity=1)
    payload = _materialize(rows, normalization, cache_capacity=1)
    assert tuple(payload["signals"]["vibration"].shape) == (3, 1, 2048)
    assert tuple(payload["signals"]["current"].shape) == (3, 2, 2048)
    assert payload["labels"].dtype == torch.long
    assert torch.isfinite(payload["signals"]["vibration"]).all()
    assert torch.isfinite(payload["signals"]["current"]).all()


def test_cached_dataset_reads_saved_tensors():
    path = Path("D:/code/putest/pu_experiment/.test_outputs/cache.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signals": {"vibration": torch.zeros(2, 1, 2048), "current": torch.zeros(2, 2, 2048)},
        "labels": torch.tensor([0, 1]),
        "sample_ids": ["a", "b"],
        "metadata": [{"sample_id": "a"}, {"sample_id": "b"}],
    }
    torch.save(payload, path)
    dataset = CachedPUSignalDataset(path, "fusion")
    item = dataset[1]
    assert item["sample_id"] == "b"
    assert tuple(item["vibration"].shape) == (1, 2048)
    assert tuple(item["current"].shape) == (2, 2048)
