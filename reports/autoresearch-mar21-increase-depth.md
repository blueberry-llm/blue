# increase-depth: depth=12, N_EXPERTS=4

## Hypothesis
Increasing depth from 8 to 12 layers (n_embd: 512→768) with fewer experts (8→4) to compensate for VRAM. A deeper model should have more representational capacity, potentially improving val_bpb. Fewer experts reduces MoE overhead to make room for the wider layers.

## Results
- val_bpb: 1.905729 (baseline was 1.592335)
- peak_vram_mb: 8878.5 (~8.7 GB, baseline ~5.1 GB — within 20%)
- total_tokens_M: 45.1 (baseline 87.6 — much fewer)
- num_steps: 86 (baseline 167)
- num_params_M: 199.8 (baseline 96.5)
- training_seconds: 2017.4
- mfu_percent: 18.14

## Expected
Hypothesized improvement from deeper model.

## Outcome
Worse. The model is significantly undertrained — only 45.1M tokens vs baseline's 87.6M. The larger model is slower per step (~22k tok/s vs ~50k tok/s), so within the same time budget it sees half the tokens.

## Next ideas
1. **top-1 routing** — README notes "top-1 beats top-2 and top-3". Switch TOP_K=1 on the baseline config to reduce routing overhead and allow more tokens/sec.
2. **Increase learning rate** — Try MATRIX_LR=0.12 on baseline. The model might benefit from faster parameter updates within the time budget.
