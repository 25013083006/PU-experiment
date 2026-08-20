current_stage: Stage 1 baseline complete
completed:
  - Audited 720 PU MAT files across 9 bearings and 3 labels
  - Implemented deterministic MAT parsing and synchronization validation
  - Generated 7200 samples using 2048-point windows and 10 windows per file
  - Generated reproduction split: train 4320, val 1440, test 1440
  - Generated three strict bearing-independent splits: 2400/2400/2400 each
  - Implemented vibration-only, current-only, and feature-fusion 1D CNN baselines
  - Added cached tensor datasets, train/validation/test evaluation, checkpoints, predictions, confusion matrices, and config files
  - Completed reproduction evaluation for 3 models x 3 seeds
  - Completed strict evaluation for 3 models x 3 folds x 3 seeds
best_metrics:
  reproduction_macro_f1_mean:
    vibration: 0.961270
    current: 0.683764
    fusion: 0.975890
  strict_macro_f1_mean:
    vibration: 0.348130
    current: 0.177317
    fusion: 0.463683
known_failures:
  - Strict bearing-independent performance has high fold variance
  - Current-only is near or below random in all strict folds
  - Vibration-only is near random in strict folds 1 and 2
  - Stage 1 scientific acceptance criterion for reliably non-random single modalities is not met
next_action: Ask user whether to diagnose and strengthen Stage 1 preprocessing/baselines or proceed to their proposed method with this limitation documented
last_summary: Stage 0.5 is complete. On 2026-08-20 the project was prepared for Git management with generated data, caches, checkpoints, runs, and temporary files excluded from version control.

