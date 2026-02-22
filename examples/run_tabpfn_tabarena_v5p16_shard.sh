#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <shard_index> [num_shards]"
  echo "Example: $0 0 16"
  exit 1
fi

SHARD_INDEX="$1"
NUM_SHARDS="${2:-16}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi
EXTRA_ARGS=("$@")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

OUT_DIR="$ROOT_DIR/results/tabarena_all51/tpu_v5p16"
mkdir -p "$OUT_DIR"

python -u examples/tabpfn_tabarena_batch.py \
  --dataset_slugs all \
  --dataset_order sorted \
  --folds all \
  --num_shards "$NUM_SHARDS" \
  --shard_index "$SHARD_INDEX" \
  --device xla:0 \
  --pjrt_device "$PJRT_DEVICE" \
  --xla_use_bf16 \
  --xla_default_matmul_precision high \
  --torch_num_threads 1 \
  --torch_num_interop_threads 1 \
  --sample_size 50000 \
  --n_estimators 8 \
  --n_preprocessing_jobs 1 \
  --ignore_pretraining_limits \
  --resume \
  --output_csv "$OUT_DIR/tabarena_tabpfn_all51_tpu_v5p16_shard${SHARD_INDEX}_of_${NUM_SHARDS}.csv" \
  --summary_csv "$OUT_DIR/tabarena_tabpfn_all51_tpu_v5p16_shard${SHARD_INDEX}_of_${NUM_SHARDS}_summary.csv" \
  --plan_csv "$OUT_DIR/tabarena_tabpfn_all51_tpu_v5p16_shard${SHARD_INDEX}_of_${NUM_SHARDS}_plan.csv" \
  "${EXTRA_ARGS[@]}"
