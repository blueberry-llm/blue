# higher-lr: MATRIX_LR=0.16

## Hypothesis
Increase MATRIX_LR from 0.12 to 0.16. If 0.12 improved over 0.08, higher might push convergence further.

## Results
- val_bpb: 1.527447 (current best: 1.520717, baseline: 1.592335)
- peak_vram_mb: 5224.4 (~5.1 GB)
- total_tokens_M: 108.0
- num_steps: 206
- training_seconds: 2505.8

## Expected
Further improvement over 0.12.

## Outcome
Slightly worse. MATRIX_LR=0.12 appears to be near the sweet spot. Going to 0.16 adds noise without benefit.

## Next ideas
1. **MATRIX_LR=0.14** — between 0.12 and 0.16, might be the true optimum.
2. **Reduce weight decay** — WEIGHT_DECAY=0.1 instead of 0.2 to allow faster convergence.
