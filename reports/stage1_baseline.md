# Stage 1 CNN Baseline Results

Training: PyTorch 2.11.0 + CUDA, 3 seeds (42, 43, 44), 30 epochs maximum, early stopping patience 7.

## Reproduction Split

| Model | Accuracy mean ? SD | Macro F1 mean ? SD |
|---|---:|---:|
| vibration | 0.9613 ? 0.0039 | 0.9613 ? 0.0040 |
| current | 0.6875 ? 0.0116 | 0.6838 ? 0.0122 |
| fusion | 0.9759 ? 0.0051 | 0.9759 ? 0.0052 |

## Strict Bearing-Independent Split

| Model | Fold 0 F1 | Fold 1 F1 | Fold 2 F1 | Overall Macro F1 mean ? SD |
|---|---:|---:|---:|---:|
| vibration | 0.6115 | 0.2397 | 0.1932 | 0.3481 ? 0.2039 |
| current | 0.1959 | 0.0754 | 0.2606 | 0.1773 ? 0.0870 |
| fusion | 0.7281 | 0.3044 | 0.3586 | 0.4637 ? 0.2338 |

## Interpretation

- Implementation and evaluation loops are complete; every formal run saves metrics, predictions, confusion matrix, config, normalization, and checkpoint.
- Reproduction results are stable for vibration and fusion; current is weaker but consistently above random.
- Strict split exposes strong bearing-domain shift. Current-only is near or below random in all folds.
- Vibration-only is non-random mainly in fold 0 and near random in folds 1?2. Fusion is strongest overall but varies substantially by held-out bearing group.
- Stage 1 code acceptance passes, but the scientific acceptance criterion requiring reliable non-random single-modality strict performance does not pass.
- No tokenizer, proposed method, Transformer, or LLM was introduced.
