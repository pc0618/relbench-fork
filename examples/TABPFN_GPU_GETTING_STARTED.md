# TabPFN GPU Getting Started (From RT-Only Machine)

This guide assumes you start on a new machine that already has only the `relational-transformer` repo checked out.

## 1) Bootstrap `relbench` next to RT

```bash
# from the RT repo root
cd ~/relational-transformer

# clone relbench fork next to RT
cd ..
git clone https://github.com/pc0618/relbench-fork.git relbench
cd relbench

# optional: checkout the working branch
# git checkout relbench-tabarena-clean
```

## 2) Create environment and install dependencies

```bash
cd ~/relbench
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel

# relbench + tabpfn stack
pip install -e .
pip install \
  tabpfn==6.3.2 \
  openml==0.15.1 \
  duckdb==1.4.2 \
  pooch==1.8.2 \
  google-cloud-storage==3.8.0 \
  pyarrow

# verify
python -c "import tabpfn, relbench, torch; print('ok', tabpfn.__version__)"
```

## 3) Auth for GCS (required for upload/download)

```bash
gcloud auth login
gcloud auth application-default login
```

## 4) Run all-51 TabArena folds sequentially on GPU (with GCS sync + local cleanup)

This runs sequentially (`--dataset_order as_is`, `--num_shards 1`) and uploads artifacts every completed fold (`--sync_every_rows 1`).

```bash
cd ~/relbench
source .venv/bin/activate

RUN_ID="tabpfn-gpu-$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="results/tabarena_all51/gpu_local"
GCS_DIR="gs://marin-us-central1/tabpfn-tabarena-results/${RUN_ID}"

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHONPATH=. python examples/tabpfn_tabarena_batch.py \
  --dataset_slugs all \
  --dataset_order as_is \
  --folds all \
  --num_shards 1 \
  --shard_index 0 \
  --device cuda \
  --inference_precision autocast \
  --sample_size 50000 \
  --n_estimators 8 \
  --n_preprocessing_jobs 1 \
  --ignore_pretraining_limits \
  --resume \
  --sync_every_rows 1 \
  --gcs_artifact_dir "${GCS_DIR}" \
  --delete_synced_noncore \
  --sync_glob "${OUT_DIR}/tabarena_tabpfn_all51_cuda_shard0_of_1*" \
  --output_csv "${OUT_DIR}/tabarena_tabpfn_all51_cuda_shard0_of_1.csv" \
  --summary_csv "${OUT_DIR}/tabarena_tabpfn_all51_cuda_shard0_of_1_summary.csv" \
  --plan_csv "${OUT_DIR}/tabarena_tabpfn_all51_cuda_shard0_of_1_plan.csv"
```

## 5) Pull results from GCS and combine locally

```bash
cd ~/relbench
source .venv/bin/activate

RUN_ID="<your-run-id>"
GCS_DIR="gs://marin-us-central1/tabpfn-tabarena-results/${RUN_ID}"
PULL_DIR="results/tabarena_all51/gcs_pull_${RUN_ID}"
mkdir -p "${PULL_DIR}"

gsutil -m rsync -r "${GCS_DIR}" "${PULL_DIR}"

python examples/combine_tabpfn_shards.py \
  --input_glob "${PULL_DIR}/results/tabarena_all51/gpu_local/tabarena_tabpfn_all51_cuda_shard*_of_*.csv" \
  --output_csv "${PULL_DIR}/tabarena_tabpfn_all51_cuda_combined.csv" \
  --summary_csv "${PULL_DIR}/tabarena_tabpfn_all51_cuda_combined_summary.csv"
```

## 6) Monitor quick checks

```bash
# inspect live artifact growth in GCS
gsutil ls -r "${GCS_DIR}/**"

# row count in local shard csv (if present)
python - <<'PY'
import pandas as pd
p='results/tabarena_all51/gpu_local/tabarena_tabpfn_all51_cuda_shard0_of_1.csv'
try:
    df=pd.read_csv(p)
    print('rows', len(df), 'ok', (df['status']=='ok').sum(), 'err', (df['status']=='error').sum())
except Exception as e:
    print('not ready:', e)
PY
```

## 7) Notes

- `--delete_synced_noncore` removes synced non-core artifacts matched by `--sync_glob` after successful sync. Core CSV/summary/plan files are kept.
- For lower VRAM GPUs, start with `--sample_size 10000 --n_estimators 2` and scale up.
- To resume an interrupted run, reuse the same `--output_csv` and `--gcs_artifact_dir` with `--resume`.
