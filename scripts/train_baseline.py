from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CachedPUSignalDataset, PUSignalDataset, compute_normalization, load_split_rows, pu_collate
from src.models.baseline import BaselineCNN
from src.training.trainer import evaluate, save_evaluation, seed_everything, train_model


def run_one(root: Path, model_name: str, split_path: Path, seed: int, epochs: int, patience: int, batch_size: int, device: torch.device, tag: str, data_source: str, num_workers: int) -> dict[str, object]:
    seed_everything(seed)
    train_rows = load_split_rows(root, split_path, "train")
    val_rows = load_split_rows(root, split_path, "val")
    test_rows = load_split_rows(root, split_path, "test")
    if data_source == "cache":
        cache_prefix = "reproduction" if tag == "reproduction" else tag
        cache_dir = root / "data" / "processed"
        train_cache = cache_dir / f"{cache_prefix}_train.pt"
        val_cache = cache_dir / f"{cache_prefix}_val.pt"
        test_cache = cache_dir / f"{cache_prefix}_test.pt"
        train_set = CachedPUSignalDataset(train_cache, model_name)
        val_set = CachedPUSignalDataset(val_cache, model_name)
        test_set = CachedPUSignalDataset(test_cache, model_name)
        normalization = torch.load(train_cache, map_location="cpu", weights_only=False)["normalization"]
        cache_file = str(train_cache)
    else:
        normalization = compute_normalization(train_rows)
        train_set = PUSignalDataset(train_rows, normalization, model_name)
        val_set = PUSignalDataset(val_rows, normalization, model_name)
        test_set = PUSignalDataset(test_rows, normalization, model_name)
        cache_file = None
    pin_memory = device.type == "cuda"
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "collate_fn": pu_collate, "pin_memory": pin_memory}
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    model = BaselineCNN(model_name).to(device)
    model, training = train_model(model, train_loader, val_loader, device, epochs, patience, 3e-4, 1e-4)
    metrics, records = evaluate(model, test_loader, device)
    output = root / "runs" / "stage1" / tag / model_name / f"seed_{seed}"
    config = {"stage": 1, "model": model_name, "seed": seed, "split": str(split_path), "train_count": len(train_set), "val_count": len(val_set), "test_count": len(test_set), "device": str(device), "epochs": epochs, "batch_size": batch_size, "data_source": data_source, "cache_file": cache_file, "num_workers": num_workers}
    save_evaluation(output, metrics, records, config, model, normalization, training)
    print(json.dumps({"model": model_name, "tag": tag, "seed": seed, **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")}}, ensure_ascii=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vibration", "current", "fusion"], required=True)
    parser.add_argument("--split", choices=["reproduction", "strict"], default="reproduction")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-source", choices=["cache", "mat"], default="cache")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    split_path = root / "data" / "splits" / ("split_reproduction.json" if args.split == "reproduction" else f"split_strict_fold{args.fold}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.split if args.split == "reproduction" else f"strict_fold{args.fold}"
    run_one(root, args.model, split_path, args.seed, args.epochs, args.patience, args.batch_size, device, tag, args.data_source, args.num_workers)


if __name__ == "__main__":
    main()
