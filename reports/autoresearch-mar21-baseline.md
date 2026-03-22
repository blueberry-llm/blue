# daily baseline: autoresearch-mar21

## Hypothesis
Establish the daily starting point with the default MoE configuration.

## Results
- val_bpb: 1.592335
- peak_vram_mb: 5220.7 (~5.1 GB)
- training_seconds: 2005.3
- total_seconds: 2325.1
- total_tokens_M: 87.6
- num_steps: 167
- num_params_M: 96.5
- depth: 8
- mfu_percent: 18.12

## Expected
Establish baseline.

## Outcome
Baseline established. Every experiment today must beat 1.592335.

## Next ideas
1. **Increase depth to 12 layers** — The model currently has 8 layers with 512-dim embeddings. Going deeper (more layers, wider model via aspect ratio) could improve capacity within VRAM budget.
2. **Reduce N_EXPERTS to 4** — Fewer experts means less routing overhead and more FLOPs per forward pass per expert. The VRAM freed up could go toward a wider model.
