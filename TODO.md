# TODO

- [x] Measure LayerNorm gamma stats on GPT-2 124M (mean/std/histogram).
- [x] Remove C/CUDA interop utilities in `train_gpt2.py`.
- [x] Remove code for loading pretrained GPT-2.
- [x] Add `d6` config (6L/6H/384d).
- [x] Add CLI flags for norm variant (layernorm/rms) and parameterization (standard/exp).
- [x] Add `ExpLayerNorm` and `ExpRMSNorm` with gamma = exp(theta).
- [ ] Add optimizer choices (AdamW/SGD) with matching hyperparams.
- [ ] Log metrics: gamma/theta distributions, grad norms, update-to-data ratio, and loss curves.
- [ ] Add wandb and logging.
- [ ] Add a run script.

Follow up:
- [ ] Experiment with weight decay (theta -> 0).
- [ ] Add `SignedExpLayerNorm` with gamma = theta * exp(|theta| - 1).
