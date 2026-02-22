import argparse
import sys

import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

import tabpfn_tabarena_batch as batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer_shard_index", type=int, required=True)
    parser.add_argument("--outer_num_shards", type=int, required=True)
    parser.add_argument("--tpu_tag", type=str, required=True)
    parser.add_argument("--out_dir_remote", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=50_000)
    parser.add_argument("--n_estimators", type=int, default=4)
    parser.add_argument("--n_preprocessing_jobs", type=int, default=2)
    parser.add_argument("--torch_num_threads", type=int, default=2)
    parser.add_argument("--torch_num_interop_threads", type=int, default=1)
    parser.add_argument("--inference_precision", type=str, default="bfloat16")
    parser.add_argument("--sync_to_head_dir", type=str, default="")
    parser.add_argument("--gcs_artifact_dir", type=str, default="")
    parser.add_argument("--sync_every_rows", type=int, default=1)
    parser.add_argument("--delete_synced_noncore", action="store_true", default=False)
    parser.add_argument("--max_runs_total", type=int, default=0)
    parser.add_argument(
        "--spawn_nprocs",
        type=int,
        default=0,
        help="0 means torch-xla auto (all available devices).",
    )
    parser.add_argument("batch_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _worker(index: int, args: argparse.Namespace) -> None:
    rank = xr.global_ordinal() if hasattr(xr, "global_ordinal") else index
    world_size = (
        xr.global_runtime_device_count()
        if hasattr(xr, "global_runtime_device_count")
        else 1
    )
    total_shards = args.outer_num_shards * world_size
    global_shard_idx = args.outer_shard_index * world_size + rank
    local_max_runs = (
        (args.max_runs_total + world_size - 1) // world_size
        if args.max_runs_total > 0
        else 0
    )

    out_prefix = (
        f"{args.out_dir_remote}/tabarena_tabpfn_all51_tpu_{args.tpu_tag}"
        f"_shard{global_shard_idx}_of_{total_shards}"
    )
    batch_argv = [
        "tabpfn_tabarena_batch.py",
        "--dataset_slugs",
        "all",
        "--dataset_order",
        "as_is",
        "--folds",
        "all",
        "--num_shards",
        str(total_shards),
        "--shard_index",
        str(global_shard_idx),
        "--device",
        "xla:0",
        "--inference_precision",
        args.inference_precision,
        "--pjrt_device",
        "TPU",
        "--xla_use_bf16",
        "--xla_default_matmul_precision",
        "high",
        "--torch_num_threads",
        str(args.torch_num_threads),
        "--torch_num_interop_threads",
        str(args.torch_num_interop_threads),
        "--log_runtime_config",
        "--sample_size",
        str(args.sample_size),
        "--n_estimators",
        str(args.n_estimators),
        "--n_preprocessing_jobs",
        str(args.n_preprocessing_jobs),
        "--ignore_pretraining_limits",
        "--resume",
        "--sync_to_head_dir",
        args.sync_to_head_dir,
        "--gcs_artifact_dir",
        args.gcs_artifact_dir,
        "--sync_every_rows",
        str(args.sync_every_rows),
        "--sync_glob",
        f"{out_prefix}*",
        "--output_csv",
        f"{out_prefix}.csv",
        "--summary_csv",
        f"{out_prefix}_summary.csv",
        "--plan_csv",
        f"{out_prefix}_plan.csv",
    ]
    if local_max_runs > 0:
        batch_argv.extend(["--max_runs", str(local_max_runs)])
    if args.delete_synced_noncore:
        batch_argv.append("--delete_synced_noncore")

    passthrough = list(args.batch_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    batch_argv.extend(passthrough)

    print(
        f"[spawn-worker] rank={rank} world_size={world_size} "
        f"global_shard={global_shard_idx}/{total_shards} device=xla:0",
        flush=True,
    )
    if rank == 0:
        print(
            "[spawn-worker] combine hint: "
            f"python examples/combine_tabpfn_shards.py --input_glob "
            f"'{args.out_dir_remote}/tabarena_tabpfn_all51_tpu_{args.tpu_tag}_shard*_of_{total_shards}.csv' "
            f"--output_csv '{args.out_dir_remote}/tabarena_tabpfn_all51_tpu_{args.tpu_tag}_jobshard{args.outer_shard_index}_of_{args.outer_num_shards}.csv' "
            f"--summary_csv '{args.out_dir_remote}/tabarena_tabpfn_all51_tpu_{args.tpu_tag}_jobshard{args.outer_shard_index}_of_{args.outer_num_shards}_summary.csv'",
            flush=True,
        )

    sys.argv = batch_argv
    batch.main()


def main() -> None:
    args = _parse_args()
    nprocs = None if args.spawn_nprocs <= 0 else args.spawn_nprocs
    print(f"[spawn-entrypoint] xmp.spawn nprocs={nprocs}", flush=True)
    xmp.spawn(_worker, args=(args,), nprocs=nprocs, start_method="spawn")
    print("[spawn-entrypoint] completed", flush=True)


if __name__ == "__main__":
    main()
