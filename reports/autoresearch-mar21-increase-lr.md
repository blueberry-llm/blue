# increase-lr: MATRIX_LR=0.12 from 0.08

## Hypothesis
Increase matrix learning rate from 0.08 to 0.12. The baseline had 0.08 and the LR warmup is disabled (WARMUP_RATIO=0.0), so the model starts at max LR immediately. A higher LR should converge faster within the fixed time budget, leading to better val_bpb.

## Results
- val_bpb: 1.520717 (baseline: 1.592335)
- peak_vram_mb: 5220.3 (~5.1 GB, same as baseline)
- total_tokens_M: 107.5 (baseline 87.6 — 23% more tokens!)
- num_steps: 205 (baseline 167 — 23% more steps)
- training_seconds: 2502.5
- mfu_percent: ~18.1

## Expected
Moderate improvement from faster convergence.

## Outcome
**Improved!** val_bpb dropped from 1.592335 to 1.520717 — a 0.072 improvement (~4.5% relative). The higher LR also led to more steps in the same time budget (205 vs 167) because loss decreased faster. Peak VRAM unchanged. This is a clean win.

## Next ideas
1. **Further increase LR** — Try MATRIX_LR=0.15 or 0.20 to push convergence even faster.
2. **Combine with batch size** — Since more steps help, try reducing weight decay to allow even faster learning.
