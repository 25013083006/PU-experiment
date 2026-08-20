from pathlib import Path
import numpy as np
import pytest
from types import SimpleNamespace
from src.data.pu_mat import parse_filename, inspect_mat


def test_parse_filename():
    result = parse_filename("N15_M07_F04_KA04_12.mat")
    assert result["bearing_id"] == "KA04"
    assert result["run_id"] == 12


def test_real_file_is_synchronized():
    path = next(Path("D:/code/putest/PU-dataset-main").rglob("*.mat"))
    record = inspect_mat(path)
    assert record.accepted
    assert record.synchronized
    assert record.channel_lengths[0] >= 2048


def fake_mat(channels):
    signal = SimpleNamespace(StartTimestamp=0.0, StopTimestamp=1.0)
    root = SimpleNamespace(
        Y=channels,
        Description=SimpleNamespace(Measurement=signal),
    )
    return {"root": root}


def channel(name, data):
    return SimpleNamespace(Name=name, Data=np.asarray(data), Raster="HostService", XIndex=2)


def test_missing_channel_is_recorded(monkeypatch):
    channels = [channel("vibration_1", [1.0, 2.0])]
    monkeypatch.setattr("src.data.pu_mat.loadmat", lambda *args, **kwargs: fake_mat(channels))
    record = inspect_mat("N15_M07_F04_KA04_1.mat")
    assert not record.accepted
    assert "missing_channels" in record.rejection_reason


def test_nonfinite_detection(monkeypatch):
    channels = [channel("vibration_1", [1.0, 2.0]), channel("phase_current_1", [1.0, np.nan]), channel("phase_current_2", [1.0, 2.0])]
    monkeypatch.setattr("src.data.pu_mat.loadmat", lambda *args, **kwargs: fake_mat(channels))
    record = inspect_mat("N15_M07_F04_KA04_1.mat")
    assert not record.accepted
    assert record.rejection_reason == "nonfinite_values"
