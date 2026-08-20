# PU Experiment

Stage 0 audits the external Paderborn University bearing subset and creates deterministic manifests and splits. Stage 0.5 materializes normalized windows into PyTorch caches so baseline training does not reread MAT files per batch.

## Run

From this directory:

```powershell
python scripts/build_pu_manifest.py --config configs/pu_stage0.yaml
python scripts/build_pu_splits.py --config configs/pu_stage0.yaml
python scripts/build_pu_cache.py --config configs/pu_stage0.yaml
python scripts/train_baseline.py --model vibration --split reproduction --data-source cache --epochs 1
```

The raw dataset remains outside this project at the configured path. Do not implement model or LLM stages until Stage 0 passes and the method is confirmed.
