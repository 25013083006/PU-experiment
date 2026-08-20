import torch

from src.models.baseline import BaselineCNN


def test_baseline_shapes():
    for mode, channels in (("vibration", 1), ("current", 2), ("fusion", None)):
        model = BaselineCNN(mode)
        batch = {"vibration": torch.randn(4, 1, 2048), "current": torch.randn(4, 2, 2048)}
        output = model(batch)
        assert output.shape == (4, 3)
