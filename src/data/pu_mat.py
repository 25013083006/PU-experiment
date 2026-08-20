from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

FILENAME_RE = re.compile(r"^(?P<N>N\d+)_M(?P<M>\d+)_F(?P<F>\d+)_(?P<bearing_id>[^_]+)_(?P<run_id>\d+)\.mat$")
TARGET_CHANNELS = ("vibration_1", "phase_current_1", "phase_current_2")
LABELS = {
    "K001": 0, "K002": 0, "K003": 0,
    "KA04": 1, "KA15": 1, "KA16": 1,
    "KI04": 2, "KI14": 2, "KI16": 2,
}


@dataclass(frozen=True)
class FileRecord:
    source_file: str
    bearing_id: str
    label: int
    N: str
    M: str
    F: str
    run_id: int
    channel_names: tuple[str, ...]
    channel_lengths: tuple[int, ...]
    channel_rasters: tuple[str, ...]
    channel_xindices: tuple[int, ...]
    finite: bool
    synchronized: bool
    sampling_raster: str
    start_timestamp: float | None
    stop_timestamp: float | None
    accepted: bool
    rejection_reason: str


def _root_struct(data: dict[str, Any]) -> Any:
    roots = [value for key, value in data.items() if not key.startswith("__")]
    if len(roots) != 1:
        raise ValueError(f"expected one MAT root, found {len(roots)}")
    return roots[0]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.size == 0:
        return ""
    return str(array.reshape(-1)[0])


def parse_filename(path: str | Path) -> dict[str, Any]:
    match = FILENAME_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"unsupported PU filename: {Path(path).name}")
    result = match.groupdict()
    result["run_id"] = int(result["run_id"])
    return result


def inspect_mat(path: str | Path) -> FileRecord:
    path = Path(path)
    fields = parse_filename(path)
    if fields["bearing_id"] not in LABELS:
        raise ValueError(f"unsupported bearing: {fields['bearing_id']}")
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    root = _root_struct(data)
    channels = {str(channel.Name): channel for channel in root.Y}
    names = tuple(TARGET_CHANNELS)
    missing = [name for name in names if name not in channels]
    if missing:
        return FileRecord(str(path), fields["bearing_id"], LABELS[fields["bearing_id"]], fields["N"], fields["M"], fields["F"], fields["run_id"], names, tuple(), tuple(), tuple(), False, False, "", None, None, False, "missing_channels:" + ",".join(missing))
    selected = [channels[name] for name in names]
    arrays = [np.asarray(channel.Data, dtype=np.float64).reshape(-1) for channel in selected]
    lengths = tuple(len(array) for array in arrays)
    rasters = tuple(_as_text(channel.Raster) for channel in selected)
    xindices = tuple(int(channel.XIndex) for channel in selected)
    finite = all(bool(np.isfinite(array).all()) for array in arrays)
    synchronized = len(set(lengths)) == 1 and len(set(rasters)) == 1 and len(set(xindices)) == 1
    reason = ""
    if not finite:
        reason = "nonfinite_values"
    elif not synchronized:
        reason = "unsynchronized_channels"
    measurement = getattr(getattr(root, "Description", None), "Measurement", None)
    start = getattr(measurement, "StartTimestamp", None)
    stop = getattr(measurement, "StopTimestamp", None)
    accepted = finite and synchronized
    return FileRecord(str(path), fields["bearing_id"], LABELS[fields["bearing_id"]], fields["N"], fields["M"], fields["F"], fields["run_id"], names, lengths, rasters, xindices, finite, synchronized, rasters[0] if rasters else "", None if start is None else float(start), None if stop is None else float(stop), accepted, reason)


def load_target_signals(path: str | Path) -> dict[str, np.ndarray]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    root = _root_struct(data)
    channels = {str(channel.Name): channel for channel in root.Y}
    return {name: np.asarray(channels[name].Data, dtype=np.float32).reshape(-1) for name in TARGET_CHANNELS}
