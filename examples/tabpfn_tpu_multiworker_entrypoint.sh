#!/usr/bin/env bash
set -euo pipefail

required_vars=(
  OUTER_SHARD_INDEX
  OUTER_NUM_SHARDS
  INTRA_JOB_WORKERS
  TPU_TAG
  OUT_DIR_REMOTE
  SAMPLE_SIZE
  N_ESTIMATORS
  N_PREPROCESSING_JOBS
  TORCH_NUM_THREADS
  TORCH_NUM_INTEROP_THREADS
  INFERENCE_PRECISION
  SYNC_TO_HEAD_DIR
  GCS_ARTIFACT_DIR
  SYNC_EVERY_ROWS
  MAX_RUNS_TOTAL
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required env var: ${var_name}" >&2
    exit 1
  fi
done

if ! [[ "$OUTER_SHARD_INDEX" =~ ^[0-9]+$ ]]; then
  echo "OUTER_SHARD_INDEX must be an integer, got: $OUTER_SHARD_INDEX" >&2
  exit 1
fi
if ! [[ "$OUTER_NUM_SHARDS" =~ ^[0-9]+$ ]] || (( OUTER_NUM_SHARDS < 1 )); then
  echo "OUTER_NUM_SHARDS must be >=1, got: $OUTER_NUM_SHARDS" >&2
  exit 1
fi
if ! [[ "$INTRA_JOB_WORKERS" =~ ^[0-9]+$ ]] || (( INTRA_JOB_WORKERS < 1 )); then
  echo "INTRA_JOB_WORKERS must be >=1, got: $INTRA_JOB_WORKERS" >&2
  exit 1
fi
if ! [[ "$MAX_RUNS_TOTAL" =~ ^[0-9]+$ ]]; then
  echo "MAX_RUNS_TOTAL must be >=0, got: $MAX_RUNS_TOTAL" >&2
  exit 1
fi

TOTAL_SHARDS=$((OUTER_NUM_SHARDS * INTRA_JOB_WORKERS))
LOCAL_MAX_RUNS=0
if (( MAX_RUNS_TOTAL > 0 )); then
  LOCAL_MAX_RUNS=$(((MAX_RUNS_TOTAL + INTRA_JOB_WORKERS - 1) / INTRA_JOB_WORKERS))
fi

EXTRA_BATCH_ARGS=("$@")
PIDS=()
DELETE_SYNCED_NONCORE="${DELETE_SYNCED_NONCORE:-0}"

for (( local_worker_idx=0; local_worker_idx<INTRA_JOB_WORKERS; local_worker_idx++ )); do
  global_shard_idx=$((OUTER_SHARD_INDEX * INTRA_JOB_WORKERS + local_worker_idx))
  out_prefix="${OUT_DIR_REMOTE}/tabarena_tabpfn_all51_tpu_${TPU_TAG}_shard${global_shard_idx}_of_${TOTAL_SHARDS}"
  device="xla:${local_worker_idx}"

  cmd=(
    python examples/tabpfn_tabarena_batch.py
    --dataset_slugs all
    --dataset_order as_is
    --folds all
    --num_shards "$TOTAL_SHARDS"
    --shard_index "$global_shard_idx"
    --device "$device"
    --inference_precision "$INFERENCE_PRECISION"
    --pjrt_device TPU
    --xla_use_bf16
    --xla_default_matmul_precision high
    --torch_num_threads "$TORCH_NUM_THREADS"
    --torch_num_interop_threads "$TORCH_NUM_INTEROP_THREADS"
    --log_runtime_config
    --sample_size "$SAMPLE_SIZE"
    --n_estimators "$N_ESTIMATORS"
    --n_preprocessing_jobs "$N_PREPROCESSING_JOBS"
    --ignore_pretraining_limits
    --resume
    --sync_to_head_dir "$SYNC_TO_HEAD_DIR"
    --gcs_artifact_dir "$GCS_ARTIFACT_DIR"
    --sync_every_rows "$SYNC_EVERY_ROWS"
    --sync_glob "${out_prefix}*"
    --output_csv "${out_prefix}.csv"
    --summary_csv "${out_prefix}_summary.csv"
    --plan_csv "${out_prefix}_plan.csv"
  )

  if (( LOCAL_MAX_RUNS > 0 )); then
    cmd+=(--max_runs "$LOCAL_MAX_RUNS")
  fi
  if [[ "$DELETE_SYNCED_NONCORE" == "1" ]]; then
    cmd+=(--delete_synced_noncore)
  fi
  if (( ${#EXTRA_BATCH_ARGS[@]} > 0 )); then
    cmd+=("${EXTRA_BATCH_ARGS[@]}")
  fi

  echo "[Launch] local_worker=${local_worker_idx} global_shard=${global_shard_idx}/${TOTAL_SHARDS} device=${device}"
  "${cmd[@]}" > >(sed -u "s/^/[w${local_worker_idx}] /") 2> >(sed -u "s/^/[w${local_worker_idx}] /" >&2) &
  PIDS+=($!)
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "[Multiworker Done] status=${status} total_shards=${TOTAL_SHARDS}"
echo "[Combine Hint] python examples/combine_tabpfn_shards.py --input_glob '${OUT_DIR_REMOTE}/tabarena_tabpfn_all51_tpu_${TPU_TAG}_shard*_of_${TOTAL_SHARDS}.csv' --output_csv '${OUT_DIR_REMOTE}/tabarena_tabpfn_all51_tpu_${TPU_TAG}_jobshard${OUTER_SHARD_INDEX}_of_${OUTER_NUM_SHARDS}.csv' --summary_csv '${OUT_DIR_REMOTE}/tabarena_tabpfn_all51_tpu_${TPU_TAG}_jobshard${OUTER_SHARD_INDEX}_of_${OUTER_NUM_SHARDS}_summary.csv'"
exit "$status"
