from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

from .pu_mat import FileRecord, load_target_signals, inspect_mat

FILE_FIELDS = ["source_file", "bearing_id", "label", "N", "M", "F", "run_id", "channel_names", "channel_lengths", "channel_rasters", "channel_xindices", "finite", "synchronized", "sampling_raster", "start_timestamp", "stop_timestamp", "accepted", "rejection_reason"]
SAMPLE_FIELDS = ["sample_id", "source_file", "bearing_id", "label", "condition", "run_id", "window_index", "window_start", "window_end", "signal_length", "sampling_raster", "start_timestamp", "stop_timestamp"]


def equal_spaced_starts(length: int, window_length: int, count: int) -> list[int]:
    if length < window_length or count <= 0:
        return []
    if count == 1:
        return [0]
    last = length - window_length
    return [int(round(last * index / (count - 1))) for index in range(count)]


def file_row(record: FileRecord) -> dict[str, object]:
    return {"source_file": record.source_file, "bearing_id": record.bearing_id, "label": record.label, "N": record.N, "M": record.M, "F": record.F, "run_id": record.run_id, "channel_names": "|".join(record.channel_names), "channel_lengths": "|".join(map(str, record.channel_lengths)), "channel_rasters": "|".join(record.channel_rasters), "channel_xindices": "|".join(map(str, record.channel_xindices)), "finite": record.finite, "synchronized": record.synchronized, "sampling_raster": record.sampling_raster, "start_timestamp": record.start_timestamp, "stop_timestamp": record.stop_timestamp, "accepted": record.accepted, "rejection_reason": record.rejection_reason}


def sample_rows(record: FileRecord, window_length: int, windows_per_file: int) -> list[dict[str, object]]:
    if not record.accepted:
        return []
    length = record.channel_lengths[0]
    condition = f"{record.N}_M{record.M}_F{record.F}"
    rows = []
    for index, start in enumerate(equal_spaced_starts(length, window_length, windows_per_file)):
        rows.append({"sample_id": f"{record.bearing_id}_{condition}_run{record.run_id:02d}_w{index:02d}", "source_file": record.source_file, "bearing_id": record.bearing_id, "label": record.label, "condition": condition, "run_id": record.run_id, "window_index": index, "window_start": start, "window_end": start + window_length, "signal_length": length, "sampling_raster": record.sampling_raster, "start_timestamp": record.start_timestamp, "stop_timestamp": record.stop_timestamp})
    return rows


def write_csv(path: str | Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_manifests(dataset_root: str | Path, manifests_dir: str | Path, window_length: int = 2048, windows_per_file: int = 10) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    files = sorted(Path(dataset_root).rglob("*.mat"))
    records = []
    samples = []
    for path in files:
        try:
            record = inspect_mat(path)
        except Exception as exc:
            records.append({"source_file": str(path), "accepted": False, "rejection_reason": f"parse_error:{type(exc).__name__}:{exc}"})
            continue
        records.append(file_row(record))
        samples.extend(sample_rows(record, window_length, windows_per_file))
    write_csv(Path(manifests_dir) / "file_manifest.csv", records, FILE_FIELDS)
    write_csv(Path(manifests_dir) / "sample_manifest.csv", samples, SAMPLE_FIELDS)
    return records, samples


def manifest_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
