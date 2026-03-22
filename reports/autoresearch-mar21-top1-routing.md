# top-1 routing: TOP_K=1

## Hypothesis
Switch TOP_K from 2 to 1. The README claims "top-1 beats top-2 and top-3". Reducing routing overhead should allow more tokens/sec and potentially better val_bpb.

## Results
- val_bpb: 1.986430 (baseline: 1.592335)
- peak_vram_mb: 4814.2 (~4.7 GB, baseline ~5.1 GB)
- total_tokens_M: 32.5 (baseline 87.6)
- num_steps: 62 (baseline 167)
- training_seconds: 604.2
- mfu_percent: ~19.5

## Expected
Improvement from simpler routing.

## Outcome
Worse. Despite marginally higher tok/sec (~44K vs ~41K for top-2), the model is undertrained due to the short TIME_BUDGET cap for the tool timeout. The shorter training budget meant only 62 steps vs baseline's 167, which overwhelmed any potential routing improvement.

## Next ideas
1. **Increase learning rate** — Try MATRIX_LR=0.12 on baseline. Faster parameter updates within time budget.
2. **Remove shared expert** — Try N_SHARED_EXPERTS=0 to simplify the architecture and reduce overhead.
