#!/usr/bin/env bash
set -euo pipefail

# Experiment matrix: d12 x {layernorm,rms} x {standard,exp} on FineWeb (classic) with 4x A100

MODEL="d12"
DATA_DIR="data/fineweb10B"
TRAIN_PATTERN="${DATA_DIR}/fineweb_train_*.bin"
VAL_BIN="${DATA_DIR}/fineweb_val_000000.bin"

NORM_TYPES=(layernorm rmsnorm)
NORM_PARAMS=(exp standard)

# Hyperparams from https://github.com/karpathy/llm.c/discussions/481
NUM_ITERS=18865
SEQ_LEN=1024
BATCH_SIZE=64
TOTAL_BATCH_SIZE=524288
WEIGHT_DECAY=0.1

COMMON_ARGS=(
  --model "${MODEL}"
  --input_bin "${TRAIN_PATTERN}"
  --input_val_bin "${VAL_BIN}"
  --num_iterations "${NUM_ITERS}"
  --sequence_length "${SEQ_LEN}"
  --batch_size "${BATCH_SIZE}"
  --total_batch_size "${TOTAL_BATCH_SIZE}"
  --weight_decay "${WEIGHT_DECAY}"
  --overfit_single_batch 0
  --val_loss_every 250
  --val_max_steps 20
  --sample_every 0
  --compile=1
  --flash=1
  --tensorcores=1
  --dtype=bfloat16
  --wandb_project "log-gamma-full"
  --wandb
)

for norm_type in "${NORM_TYPES[@]}"; do
  for norm_param in "${NORM_PARAMS[@]}"; do
    echo "=== Running ${MODEL} | ${norm_type}/${norm_param} ==="
    torchrun --standalone --nproc_per_node=4 train_gpt2.py \
      --norm_type "${norm_type}" \
      --norm_param "${norm_param}" \
      "${COMMON_ARGS[@]}"
  done
done
