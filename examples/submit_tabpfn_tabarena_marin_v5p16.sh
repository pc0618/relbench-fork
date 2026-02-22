#!/usr/bin/env bash
set -euo pipefail

infer_tpu_devices() {
  local tpu_type="$1"
  if [[ "$tpu_type" =~ -([0-9]+)$ ]]; then
    local cores="${BASH_REMATCH[1]}"
    if (( cores >= 2 )); then
      echo $((cores / 2))
      return
    fi
  fi
  echo 1
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <shard_index> [num_shards] [max_runs] [job_suffix] [extra_batch_args...]"
  echo "Example: $0 0 1 0 smoke --plan_only"
  exit 1
fi

SHARD_INDEX="$1"
NUM_SHARDS="${2:-1}"
MAX_RUNS="${3:-0}"
JOB_SUFFIX="${4:-}"
EXTRA_BATCH_ARGS=("${@:5}")

if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]]; then
  echo "shard_index must be an integer >= 0, got: $SHARD_INDEX" >&2
  exit 1
fi
if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || (( NUM_SHARDS < 1 )); then
  echo "num_shards must be an integer >= 1, got: $NUM_SHARDS" >&2
  exit 1
fi
if ! [[ "$MAX_RUNS" =~ ^[0-9]+$ ]]; then
  echo "max_runs must be an integer >= 0, got: $MAX_RUNS" >&2
  exit 1
fi

RELBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARIN_ROOT="${MARIN_ROOT:-/home/pc0618/marin}"
CLUSTER="${MARIN_CLUSTER:-us-central1}"
TPU_TYPE="${TPU_TYPE:-v5p-8}"
TPU_TAG="${TPU_TYPE//-/_}"
OUT_DIR_REMOTE="${OUT_DIR_REMOTE:-results/tabarena_all51/tpu_${TPU_TAG}_marin}"
HEAD_SYNC_ROOT="${HEAD_SYNC_ROOT:-}"
HEAD_SYNC_DIR=""
GCS_RESULTS_ROOT="${GCS_RESULTS_ROOT:-gs://marin-us-central1/tabpfn-tabarena-results}"
TPU_DEVICES_PER_NODE="${TPU_DEVICES_PER_NODE:-$(infer_tpu_devices "$TPU_TYPE")}"
INTRA_JOB_WORKERS="${INTRA_JOB_WORKERS:-$TPU_DEVICES_PER_NODE}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50000}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
N_PREPROCESSING_JOBS="${N_PREPROCESSING_JOBS:-1}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-bfloat16}"
SPAWN_NPROCS="${SPAWN_NPROCS:-$INTRA_JOB_WORKERS}"
TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-$INTRA_JOB_WORKERS}"
SYNC_EVERY_ROWS="${SYNC_EVERY_ROWS:-1}"
DELETE_SYNCED_NONCORE="${DELETE_SYNCED_NONCORE:-1}"
AUTO_PULL_ON_EXIT="${AUTO_PULL_ON_EXIT:-0}"

if ! [[ "$INTRA_JOB_WORKERS" =~ ^[0-9]+$ ]] || (( INTRA_JOB_WORKERS < 1 )); then
  echo "INTRA_JOB_WORKERS must be an integer >=1, got: $INTRA_JOB_WORKERS" >&2
  exit 1
fi
if (( INTRA_JOB_WORKERS > TPU_DEVICES_PER_NODE )); then
  echo "Requested INTRA_JOB_WORKERS=$INTRA_JOB_WORKERS > TPU_DEVICES_PER_NODE=$TPU_DEVICES_PER_NODE, clamping."
  INTRA_JOB_WORKERS="$TPU_DEVICES_PER_NODE"
fi
TOTAL_SHARDS=$((NUM_SHARDS * INTRA_JOB_WORKERS))

mkdir -p "$RELBENCH_ROOT/$OUT_DIR_REMOTE"

TS="$(date -u +%Y%m%d-%H%M%S)"
SUBMISSION_ID="tabpfn-all51-seq-shard${SHARD_INDEX}-of-${NUM_SHARDS}-${TS}"
if [[ -n "$JOB_SUFFIX" ]]; then
  SUBMISSION_ID="${SUBMISSION_ID}-${JOB_SUFFIX}"
fi
if [[ -n "$HEAD_SYNC_ROOT" ]]; then
  HEAD_SYNC_DIR="${HEAD_SYNC_ROOT%/}/${SUBMISSION_ID}"
fi

if [[ "$GCS_RESULTS_ROOT" != gs://* ]]; then
  echo "GCS_RESULTS_ROOT must be a gs:// URI, got: $GCS_RESULTS_ROOT" >&2
  exit 1
fi
GCS_ARTIFACT_DIR="${GCS_RESULTS_ROOT%/}/${SUBMISSION_ID}"

MANIFEST_PATH="$RELBENCH_ROOT/$OUT_DIR_REMOTE/${SUBMISSION_ID}_sync_manifest.env"
cat >"$MANIFEST_PATH" <<EOF
SUBMISSION_ID=$SUBMISSION_ID
CLUSTER=$CLUSTER
HEAD_SYNC_ROOT=$HEAD_SYNC_ROOT
HEAD_SYNC_DIR=$HEAD_SYNC_DIR
GCS_RESULTS_ROOT=$GCS_RESULTS_ROOT
GCS_ARTIFACT_DIR=$GCS_ARTIFACT_DIR
RELBENCH_ROOT=$RELBENCH_ROOT
MARIN_ROOT=$MARIN_ROOT
TPU_TYPE=$TPU_TYPE
TPU_DEVICES_PER_NODE=$TPU_DEVICES_PER_NODE
INTRA_JOB_WORKERS=$INTRA_JOB_WORKERS
TOTAL_SHARDS=$TOTAL_SHARDS
SAMPLE_SIZE=$SAMPLE_SIZE
N_ESTIMATORS=$N_ESTIMATORS
N_PREPROCESSING_JOBS=$N_PREPROCESSING_JOBS
TORCH_NUM_THREADS=$TORCH_NUM_THREADS
TORCH_NUM_INTEROP_THREADS=$TORCH_NUM_INTEROP_THREADS
INFERENCE_PRECISION=$INFERENCE_PRECISION
TPU_NUM_DEVICES=$TPU_NUM_DEVICES
EOF

echo "Submitting to cluster=${CLUSTER} tpu=${TPU_TYPE} shard=${SHARD_INDEX}/${NUM_SHARDS} workers=${INTRA_JOB_WORKERS} total_shards=${TOTAL_SHARDS} spawn_nprocs=${SPAWN_NPROCS}"
echo "submission_id=${SUBMISSION_ID}"
if [[ -n "$HEAD_SYNC_DIR" ]]; then
  echo "head_sync_dir=${HEAD_SYNC_DIR}"
else
  echo "head_sync_dir=<disabled>"
fi
echo "gcs_artifact_dir=${GCS_ARTIFACT_DIR}"
echo "sync_manifest=${MANIFEST_PATH}"
echo "throughput_knobs: sample_size=${SAMPLE_SIZE} n_estimators=${N_ESTIMATORS} n_preproc_jobs=${N_PREPROCESSING_JOBS} inference_precision=${INFERENCE_PRECISION} torch_threads=${TORCH_NUM_THREADS}/${TORCH_NUM_INTEROP_THREADS}"
echo "artifact_cleanup: delete_synced_noncore=${DELETE_SYNCED_NONCORE}"
if [[ -n "$TPU_NUM_DEVICES" ]]; then
  echo "tpu_num_devices_limit=${TPU_NUM_DEVICES}"
fi

RAY_ENV_ARGS=()
if [[ -n "$TPU_NUM_DEVICES" ]]; then
  RAY_ENV_ARGS+=(-e TPU_NUM_DEVICES "$TPU_NUM_DEVICES")
fi

(
  cd "$MARIN_ROOT"
  ./.venv/bin/python ./lib/marin/src/marin/run/ray_run.py \
    --cluster "$CLUSTER" \
    --no_wait \
    --submission-id "$SUBMISSION_ID" \
    --working-dir "$RELBENCH_ROOT" \
    --exclude "results/" \
    --exclude ".pytest_cache/" \
    --extra cpu \
    --tpu "$TPU_TYPE" \
    --pip-package "duckdb==1.4.2" \
    --pip-package "pooch==1.8.2" \
    --pip-package "libtpu==0.0.21" \
    --pip-package "torch_xla[tpu]==2.9.0" \
    --pip-package "tabpfn==6.3.2" \
    --pip-package "openml==0.15.1" \
    --pip-package "google-cloud-storage==3.8.0" \
    -e TOKENIZERS_PARALLELISM false \
    -e OMP_NUM_THREADS 1 \
    -e OPENBLAS_NUM_THREADS 1 \
    -e MKL_NUM_THREADS 1 \
    -e NUMEXPR_NUM_THREADS 1 \
    "${RAY_ENV_ARGS[@]}" \
    -- \
    python examples/tabpfn_tpu_spawn_entrypoint.py \
      --outer_shard_index "$SHARD_INDEX" \
      --outer_num_shards "$NUM_SHARDS" \
      --tpu_tag "$TPU_TAG" \
      --out_dir_remote "$OUT_DIR_REMOTE" \
      --sample_size "$SAMPLE_SIZE" \
      --n_estimators "$N_ESTIMATORS" \
      --n_preprocessing_jobs "$N_PREPROCESSING_JOBS" \
      --torch_num_threads "$TORCH_NUM_THREADS" \
      --torch_num_interop_threads "$TORCH_NUM_INTEROP_THREADS" \
      --inference_precision "$INFERENCE_PRECISION" \
      --sync_to_head_dir "$HEAD_SYNC_DIR" \
      --gcs_artifact_dir "$GCS_ARTIFACT_DIR" \
      --sync_every_rows "$SYNC_EVERY_ROWS" \
      --max_runs_total "$MAX_RUNS" \
      --spawn_nprocs "$SPAWN_NPROCS" \
      $([[ "$DELETE_SYNCED_NONCORE" == "1" ]] && echo "--delete_synced_noncore") \
      -- \
      "${EXTRA_BATCH_ARGS[@]}"
)

if [[ -n "$HEAD_SYNC_ROOT" ]]; then
  echo "To pull head artifacts now:"
  echo "  $RELBENCH_ROOT/examples/pull_marin_head_artifacts.sh $SUBMISSION_ID $CLUSTER \"$HEAD_SYNC_ROOT\" \"$RELBENCH_ROOT\" \"$MARIN_ROOT\""
fi
echo "To inspect GCS artifacts:"
echo "  gsutil ls -r \"$GCS_ARTIFACT_DIR/**\""
echo "After pull, combine worker shards:"
echo "  . \"$RELBENCH_ROOT/.venv/bin/activate\" && python \"$RELBENCH_ROOT/examples/combine_tabpfn_shards.py\" --input_glob \"$RELBENCH_ROOT/$OUT_DIR_REMOTE/tabarena_tabpfn_all51_tpu_${TPU_TAG}_shard*_of_${TOTAL_SHARDS}.csv\" --output_csv \"$RELBENCH_ROOT/$OUT_DIR_REMOTE/tabarena_tabpfn_all51_tpu_${TPU_TAG}_jobshard${SHARD_INDEX}_of_${NUM_SHARDS}.csv\" --summary_csv \"$RELBENCH_ROOT/$OUT_DIR_REMOTE/tabarena_tabpfn_all51_tpu_${TPU_TAG}_jobshard${SHARD_INDEX}_of_${NUM_SHARDS}_summary.csv\""

if [[ "$AUTO_PULL_ON_EXIT" == "1" ]]; then
  if [[ -z "$HEAD_SYNC_ROOT" ]]; then
    echo "AUTO_PULL_ON_EXIT=1 requires HEAD_SYNC_ROOT to be set." >&2
    exit 1
  fi
  (
    cd "$MARIN_ROOT"
    uv run scripts/ray/cluster.py --cluster "$CLUSTER" wait-job "$SUBMISSION_ID"
  )
  "$RELBENCH_ROOT/examples/pull_marin_head_artifacts.sh" \
    "$SUBMISSION_ID" "$CLUSTER" "$HEAD_SYNC_ROOT" "$RELBENCH_ROOT" "$MARIN_ROOT"
fi
