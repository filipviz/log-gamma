# TODO

- [x] Measure LayerNorm gamma stats on GPT-2 124M (mean/std/histogram).
- [x] Remove C/CUDA interop utilities in `train_gpt2.py`.
- [x] Remove code for loading pretrained GPT-2.
- [x] Add `d6` config (6L/6H/384d).
- [x] Add CLI flags for norm variant (layernorm/rms) and parameterization (standard/exp).
- [x] Add `ExpLayerNorm` and `ExpRMSNorm` with gamma = exp(theta).
- [x] Log metrics: gamma/theta distributions, grad norms, update-to-data ratio, and loss curves.
- [x] Add wandb and logging.
- [x] Add a run script.

Follow up:
- [ ] Compare across optimizers (AdamW/SGD).
- [ ] Experiment with weight decay (theta -> 0).
- [ ] Add `SignedExpLayerNorm` with something like `gamma = theta * exp(|theta|)`.
