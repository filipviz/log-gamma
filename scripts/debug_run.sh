#!/usr/bin/env bash
set -euo pipefail

# Debug matrix: d6 x {layernorm,rmsnorm} x {standard,exp} on tinyshakespeare (single GPU)

DATA_DIR="data/tinyshakespeare"
TRAIN_BIN="${DATA_DIR}/tiny_shakespeare_train.bin"
VAL_BIN="${DATA_DIR}/tiny_shakespeare_val.bin"

NORM_TYPES=(layernorm rmsnorm)
NORM_PARAMS=(standard exp)

COMMON_ARGS=(
  # --overfit_single_batch 0
  --input_bin "${TRAIN_BIN}"
  --input_val_bin "${VAL_BIN}"
  --num_iterations 1000
  --wandb
)

for norm_type in "${NORM_TYPES[@]}"; do
  for norm_param in "${NORM_PARAMS[@]}"; do
    echo "=== Running d6 | ${norm_type}/${norm_param} ==="
    python3 train_gpt2.py \
      --norm_type "${norm_type}" \
      --norm_param "${norm_param}" \
      "${COMMON_ARGS[@]}"
  done
done
