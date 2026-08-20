from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import classification_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, object], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str], list[dict[str, str]]]:
    tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor) and key != "label"}
    labels = batch["label"].to(device, non_blocking=True)
    return tensors, labels, list(batch["sample_id"]), list(batch["metadata"])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], list[dict[str, object]]]:
    model.eval()
    y_true, y_pred, records = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, sample_ids, metadata = _move_batch(batch, device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            for sample_id, true, pred, probability, row in zip(sample_ids, labels.cpu().tolist(), predictions.cpu().tolist(), probabilities.max(dim=1).values.cpu().tolist(), metadata):
                records.append({"sample_id": sample_id, "true_label": true, "predicted_label": pred, "confidence": probability, "bearing_id": row["bearing_id"], "source_file": row["source_file"], "condition": row["condition"], "run_id": row["run_id"], "window_start": row["window_start"]})
    return classification_metrics(y_true, y_pred), records


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, patience: int, learning_rate: float, weight_decay: float) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_state, best_metric, stale = None, -1.0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        started = time.perf_counter()
        for batch in train_loader:
            inputs, labels, _, _ = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
        scheduler.step()
        val_metrics, _ = evaluate(model, val_loader, device)
        epoch_row = {"epoch": epoch, "train_loss": running_loss / len(train_loader.dataset), "val_macro_f1": val_metrics["macro_f1"], "seconds": time.perf_counter() - started}
        history.append(epoch_row)
        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_macro_f1": best_metric, "epochs_completed": len(history), "history": history}


def save_evaluation(output_dir: str | Path, metrics: dict[str, Any], records: list[dict[str, object]], config: dict[str, Any], model: nn.Module, normalization: dict[str, Any], training: dict[str, Any]) -> None:
    import csv
    import yaml
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps({"metrics": metrics, "training": training}, indent=2) + "\n", encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True, allow_unicode=True), encoding="utf-8")
    (output / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), output / "checkpoint.pt")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "true_label", "predicted_label", "confidence", "bearing_id", "source_file", "condition", "run_id", "window_start"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["true/pred", 0, 1, 2]); writer.writerows([[index, *row] for index, row in enumerate(metrics["confusion_matrix"])])
