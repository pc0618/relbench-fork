import argparse
import os
import traceback
from pathlib import Path
from typing import Iterable

import pandas as pd

from relbench.datasets.tabarena import TABARENA_DATASETS, get_tabarena_dataset_slugs
from tabpfn_tabarena import (
    HF_RESULTS_URL,
    INFERENCE_PRECISION_CHOICES,
    load_hf_results,
    run_tabarena_fold,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_slugs",
        type=str,
        default="all",
        help="Comma-separated slugs or 'all'. Example: credit-g,airfoil-self-noise",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Either 'all' or comma-separated indices/ranges like 0,1,2-4",
    )
    parser.add_argument("--sample_size", type=int, default=50_000)
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--n_preprocessing_jobs", type=int, default=1)
    parser.add_argument("--ignore_pretraining_limits", action="store_true", default=False)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="TabPFN device spec (e.g. cpu, cuda, auto, xla, xla:0).",
    )
    parser.add_argument(
        "--inference_precision",
        type=str,
        default="auto",
        choices=INFERENCE_PRECISION_CHOICES,
        help="TabPFN inference precision override.",
    )
    parser.add_argument(
        "--pjrt_device",
        type=str,
        default="",
        help="If set, exports PJRT_DEVICE before TabPFN model construction.",
    )
    parser.add_argument("--xla_use_bf16", action="store_true", default=False)
    parser.add_argument("--xla_downcast_bf16", action="store_true", default=False)
    parser.add_argument(
        "--xla_default_matmul_precision",
        type=str,
        default="",
        help="Optional XLA_DEFAULT_MATMUL_PRECISION value.",
    )
    parser.add_argument("--xla_flags", type=str, default="", help="Optional XLA_FLAGS value.")
    parser.add_argument("--torch_num_threads", type=int, default=0)
    parser.add_argument("--torch_num_interop_threads", type=int, default=0)
    parser.add_argument("--log_runtime_config", action="store_true", default=False)
    parser.add_argument(
        "--dataset_order",
        type=str,
        default="as_is",
        choices=["as_is", "sorted", "reverse"],
        help="Ordering of datasets after parsing --dataset_slugs.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of deterministic shards over the expanded (dataset, fold) plan.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="0-based shard index for this worker.",
    )
    parser.add_argument(
        "--plan_only",
        action="store_true",
        default=False,
        help="Print/write execution plan and exit without running folds.",
    )
    parser.add_argument(
        "--plan_csv",
        type=str,
        default="",
        help="Optional path to write the shard execution plan CSV.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/tabarena_tabpfn_results.csv",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="",
        help="Optional summary output path. Defaults to <output_csv stem>_summary.csv",
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--max_runs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--stop_on_error", action="store_true", default=False)
    parser.add_argument(
        "--hf_method",
        type=str,
        default="TabPFN (default)",
    )
    parser.add_argument(
        "--hf_results_url",
        type=str,
        default=HF_RESULTS_URL,
    )
    parser.add_argument(
        "--sync_to_head_dir",
        type=str,
        default="",
        help=(
            "If set, periodically copies artifact files to this directory on the Ray head "
            "node (for later rsync-down to this VM)."
        ),
    )
    parser.add_argument(
        "--gcs_artifact_dir",
        type=str,
        default="",
        help=(
            "If set (gs://bucket/prefix), periodically uploads artifact files to GCS. "
            "With --resume, attempts to restore artifacts from this location first."
        ),
    )
    parser.add_argument(
        "--sync_every_rows",
        type=int,
        default=1,
        help="Sync cadence in processed rows when head/GCS sync is enabled.",
    )
    parser.add_argument(
        "--sync_glob",
        action="append",
        default=[],
        help="Additional glob pattern(s) to sync relative to current working directory.",
    )
    parser.add_argument(
        "--delete_synced_noncore",
        action="store_true",
        default=False,
        help=(
            "After successful sync, delete files matched via --sync_glob except "
            "for core CSV artifacts (output/summary/plan)."
        ),
    )
    return parser.parse_args()


def _parse_dataset_slugs(arg: str, *, dataset_order: str) -> list[str]:
    if arg.strip().lower() == "all":
        slugs = get_tabarena_dataset_slugs()
    else:
        slugs = [s.strip() for s in arg.split(",") if s.strip()]
    valid = set(get_tabarena_dataset_slugs())
    invalid = [s for s in slugs if s not in valid]
    if invalid:
        raise ValueError(f"Unknown dataset slugs: {invalid}")

    if dataset_order == "sorted":
        return sorted(slugs)
    if dataset_order == "reverse":
        return list(reversed(slugs))
    return slugs


def _parse_fold_spec_for_dataset(fold_arg: str, *, fold_count: int) -> list[int]:
    if fold_arg.strip().lower() == "all":
        return list(range(fold_count))

    folds: set[int] = set()
    for token in [t.strip() for t in fold_arg.split(",") if t.strip()]:
        if "-" in token:
            lo_str, hi_str = token.split("-", 1)
            lo = int(lo_str)
            hi = int(hi_str)
            if lo > hi:
                lo, hi = hi, lo
            folds.update(range(lo, hi + 1))
        else:
            folds.add(int(token))
    valid_folds = sorted([f for f in folds if 0 <= f < fold_count])
    return valid_folds


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


_HEAD_SYNC_REMOTE = None
_GCS_STORAGE_CLIENT = None


def _resolve_relative_sync_path(path: Path, *, cwd: Path) -> Path:
    rel_path = path
    if path.is_absolute():
        try:
            rel_path = path.relative_to(cwd)
        except ValueError:
            rel_path = Path(path.name)
    return rel_path


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")

    remainder = gcs_uri[len("gs://") :]
    if not remainder:
        raise ValueError(f"Missing bucket in GCS URI: {gcs_uri}")

    bucket, _, key_prefix = remainder.partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket in GCS URI: {gcs_uri}")
    return bucket, key_prefix.strip("/")


def _sync_artifacts_to_head(
    *,
    sync_dir: str,
    paths: Iterable[Path],
) -> None:
    global _HEAD_SYNC_REMOTE

    if not sync_dir:
        return

    try:
        import ray

        if not ray.is_initialized():
            ray.init(address="auto", namespace="default", ignore_reinit_error=True)

        if _HEAD_SYNC_REMOTE is None:

            @ray.remote(num_cpus=0, resources={"head_node": 0.001})
            def _write_file(sync_root: str, rel_path: str, payload: bytes) -> int:
                out_path = Path(sync_root) / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(payload)
                return len(payload)

            _HEAD_SYNC_REMOTE = _write_file

        refs = []
        cwd = Path.cwd()
        seen: set[str] = set()
        for path in paths:
            if not path.exists() or not path.is_file():
                continue

            rel_path = _resolve_relative_sync_path(path, cwd=cwd)

            rel_key = str(rel_path)
            if rel_key in seen:
                continue
            seen.add(rel_key)

            refs.append(_HEAD_SYNC_REMOTE.remote(sync_dir, rel_key, path.read_bytes()))

        if refs:
            ray.get(refs)
    except Exception as exc:
        print(f"[Sync WARN] failed to sync artifacts to head: {type(exc).__name__}: {exc}")


def _sync_artifacts_to_gcs(
    *,
    gcs_dir: str,
    paths: Iterable[Path],
) -> None:
    global _GCS_STORAGE_CLIENT

    if not gcs_dir:
        return

    try:
        from google.cloud import storage

        if _GCS_STORAGE_CLIENT is None:
            _GCS_STORAGE_CLIENT = storage.Client()

        bucket_name, prefix = _parse_gcs_uri(gcs_dir)
        bucket = _GCS_STORAGE_CLIENT.bucket(bucket_name)
        cwd = Path.cwd()
        seen: set[str] = set()
        uploaded = 0
        for path in paths:
            if not path.exists() or not path.is_file():
                continue

            rel_path = _resolve_relative_sync_path(path, cwd=cwd)
            rel_key = str(rel_path).replace(os.sep, "/")
            if rel_key in seen:
                continue
            seen.add(rel_key)

            blob_name = f"{prefix}/{rel_key}" if prefix else rel_key
            bucket.blob(blob_name).upload_from_filename(str(path))
            uploaded += 1

        if uploaded > 0:
            print(f"[GCS Sync] uploaded_files={uploaded} destination={gcs_dir}")
    except Exception as exc:
        print(f"[GCS Sync WARN] failed to sync artifacts: {type(exc).__name__}: {exc}")


def _restore_artifacts_from_gcs(
    *,
    gcs_dir: str,
    paths: Iterable[Path],
) -> None:
    global _GCS_STORAGE_CLIENT

    if not gcs_dir:
        return

    try:
        from google.cloud import storage

        if _GCS_STORAGE_CLIENT is None:
            _GCS_STORAGE_CLIENT = storage.Client()

        bucket_name, prefix = _parse_gcs_uri(gcs_dir)
        bucket = _GCS_STORAGE_CLIENT.bucket(bucket_name)
        cwd = Path.cwd()
        seen: set[str] = set()
        restored = 0
        for path in paths:
            rel_path = _resolve_relative_sync_path(path, cwd=cwd)
            rel_key = str(rel_path).replace(os.sep, "/")
            if rel_key in seen:
                continue
            seen.add(rel_key)

            blob_name = f"{prefix}/{rel_key}" if prefix else rel_key
            blob = bucket.blob(blob_name)
            if not blob.exists(_GCS_STORAGE_CLIENT):
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(path))
            restored += 1

        if restored > 0:
            print(f"[GCS Restore] restored_files={restored} source={gcs_dir}")
    except Exception as exc:
        print(f"[GCS Restore WARN] failed to restore artifacts: {type(exc).__name__}: {exc}")


def _delete_synced_noncore_files(
    *,
    paths: Iterable[Path],
    keep_paths: Iterable[Path],
) -> None:
    keep = {str(p.resolve()) for p in keep_paths if p is not None}
    deleted = 0
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            if str(path.resolve()) in keep:
                continue
            path.unlink()
            deleted += 1
        except Exception as exc:
            print(
                f"[Cleanup WARN] failed deleting {path}: {type(exc).__name__}: {exc}"
            )
    if deleted > 0:
        print(f"[Cleanup] deleted_noncore_files={deleted}")


def _build_run_plan(
    *,
    dataset_slugs: list[str],
    fold_spec: str,
    num_shards: int,
    shard_index: int,
) -> list[tuple[int, str, int]]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >=1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards - 1}], got {shard_index}"
        )

    global_plan: list[tuple[int, str, int]] = []
    for dataset_slug in dataset_slugs:
        spec = TABARENA_DATASETS[dataset_slug]
        folds = _parse_fold_spec_for_dataset(fold_spec, fold_count=spec.fold_count)
        if not folds:
            print(
                f"[Skip] dataset={dataset_slug}: no valid folds selected for fold_count={spec.fold_count}"
            )
            continue
        for fold in folds:
            global_plan.append((len(global_plan), dataset_slug, int(fold)))

    if num_shards == 1:
        return global_plan

    return [row for row in global_plan if row[0] % num_shards == shard_index]


def _write_summary(df: pd.DataFrame, summary_path: Path) -> None:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(summary_path, index=False)
        return

    numeric_cols = [
        "test_metric_error",
        "val_metric_error",
        "hf_metric_error",
        "hf_delta_local_minus_hf",
        "hf_local_rank",
        "hf_total_methods",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    grouped = ok.groupby(
        ["dataset_slug", "dataset_name", "problem_type", "hf_method"], as_index=False
    ).agg(
        runs=("fold", "count"),
        mean_test_metric_error=("test_metric_error", "mean"),
        std_test_metric_error=("test_metric_error", "std"),
        mean_val_metric_error=("val_metric_error", "mean"),
        mean_hf_metric_error=("hf_metric_error", "mean"),
        mean_delta_local_minus_hf=("hf_delta_local_minus_hf", "mean"),
        mean_hf_local_rank=("hf_local_rank", "mean"),
        mean_hf_total_methods=("hf_total_methods", "mean"),
    )
    grouped = grouped.sort_values(
        by=["mean_delta_local_minus_hf", "mean_test_metric_error"],
        ascending=[True, True],
    ).reset_index(drop=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(summary_path, index=False)


def main() -> None:
    args = _parse_args()
    dataset_slugs = _parse_dataset_slugs(
        args.dataset_slugs, dataset_order=args.dataset_order
    )
    run_plan = _build_run_plan(
        dataset_slugs=dataset_slugs,
        fold_spec=args.folds,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    print(
        f"[Plan] datasets={len(dataset_slugs)} "
        f"pairs_in_shard={len(run_plan)} "
        f"shard={args.shard_index}/{args.num_shards}"
    )

    output_path = Path(args.output_csv)
    summary_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else output_path.with_name(f"{output_path.stem}_summary.csv")
    )
    plan_path = Path(args.plan_csv) if args.plan_csv else None

    if args.resume and args.gcs_artifact_dir:
        restore_paths = [output_path, summary_path]
        if plan_path is not None:
            restore_paths.append(plan_path)
        _restore_artifacts_from_gcs(
            gcs_dir=args.gcs_artifact_dir,
            paths=restore_paths,
        )

    if args.plan_csv:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "global_plan_index": global_idx,
                    "dataset_slug": dataset_slug,
                    "fold": fold,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                }
                for global_idx, dataset_slug, fold in run_plan
            ]
        ).to_csv(plan_path, index=False)
        print(f"[Plan] wrote {plan_path}")

    if args.sync_every_rows < 1:
        raise ValueError(f"sync_every_rows must be >=1, got {args.sync_every_rows}")

    def _collect_sync_paths() -> list[Path]:
        paths = [output_path, summary_path]
        if plan_path is not None:
            paths.append(plan_path)
        for pattern in args.sync_glob:
            paths.extend([p for p in Path.cwd().glob(pattern) if p.is_file()])
        return paths

    core_keep_paths = [output_path, summary_path]
    if plan_path is not None:
        core_keep_paths.append(plan_path)

    if args.plan_only:
        sync_paths = _collect_sync_paths()
        if args.sync_to_head_dir:
            _sync_artifacts_to_head(
                sync_dir=args.sync_to_head_dir,
                paths=sync_paths,
            )
        if args.gcs_artifact_dir:
            _sync_artifacts_to_gcs(
                gcs_dir=args.gcs_artifact_dir,
                paths=sync_paths,
            )
            if args.delete_synced_noncore:
                _delete_synced_noncore_files(
                    paths=sync_paths, keep_paths=core_keep_paths
                )
        print("[Plan] plan_only=true, exiting without execution.")
        return

    if output_path.exists() and not args.resume:
        output_path.unlink()

    done_pairs: set[tuple[str, int]] = set()
    if output_path.exists() and args.resume:
        df_existing = pd.read_csv(output_path)
        if {"dataset_slug", "fold", "status"}.issubset(df_existing.columns):
            df_ok = df_existing[df_existing["status"] == "ok"]
            done_pairs = {
                (str(ds), int(fold))
                for ds, fold in zip(df_ok["dataset_slug"], df_ok["fold"])
            }

    hf_df = load_hf_results(args.hf_results_url) if args.hf_results_url else None

    total_attempted = 0
    total_ok = 0
    total_err = 0
    runtime_logged = False
    rows_since_sync = 0
    for global_idx, dataset_slug, fold in run_plan:
        if args.max_runs > 0 and total_attempted >= args.max_runs:
            print(f"[Stop] reached max_runs={args.max_runs}")
            break
        if (dataset_slug, fold) in done_pairs:
            print(f"[Resume] skipping already done: {dataset_slug} fold={fold}")
            continue

        total_attempted += 1
        print(
            f"[Run {total_attempted}] dataset={dataset_slug} fold={fold} "
            f"sample_size={args.sample_size} n_estimators={args.n_estimators} "
            f"device={args.device} plan_idx={global_idx} "
            f"shard={args.shard_index}/{args.num_shards}"
        )

        try:
            row = run_tabarena_fold(
                dataset_slug=dataset_slug,
                fold=fold,
                sample_size=args.sample_size,
                seed=args.seed,
                hf_method=args.hf_method,
                hf_results_df=hf_df,
                n_estimators=args.n_estimators,
                n_preprocessing_jobs=args.n_preprocessing_jobs,
                ignore_pretraining_limits=args.ignore_pretraining_limits,
                device=args.device,
                inference_precision=args.inference_precision,
                pjrt_device=args.pjrt_device,
                xla_use_bf16=args.xla_use_bf16,
                xla_downcast_bf16=args.xla_downcast_bf16,
                xla_default_matmul_precision=args.xla_default_matmul_precision,
                xla_flags=args.xla_flags,
                torch_num_threads=args.torch_num_threads,
                torch_num_interop_threads=args.torch_num_interop_threads,
                log_runtime_config=args.log_runtime_config and not runtime_logged,
            )
            runtime_logged = True
            total_ok += 1
            print(
                f"[OK] {dataset_slug} fold={fold} "
                f"test={row['test_metric_error']:.6f} "
                f"hf={row.get('hf_metric_error', float('nan')):.6f} "
                f"delta={row['hf_delta_local_minus_hf']:+.6f}"
            )
        except Exception as exc:
            total_err += 1
            runtime_logged = True
            row = {
                "dataset_slug": dataset_slug,
                "dataset_name": f"tabarena-{dataset_slug}",
                "task_name": f"fold-{fold}",
                "fold": int(fold),
                "problem_type": None,
                "n_estimators": int(args.n_estimators),
                "sample_size": int(args.sample_size),
                "seed": int(args.seed),
                "device": str(args.device),
                "inference_precision": str(args.inference_precision),
                "pjrt_device": args.pjrt_device or os.getenv("PJRT_DEVICE", ""),
                "xla_use_bf16": os.getenv("XLA_USE_BF16", ""),
                "xla_downcast_bf16": os.getenv("XLA_DOWNCAST_BF16", ""),
                "xla_default_matmul_precision": os.getenv(
                    "XLA_DEFAULT_MATMUL_PRECISION", ""
                ),
                "torch_num_threads": int(args.torch_num_threads),
                "torch_num_interop_threads": int(args.torch_num_interop_threads),
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=20),
                "hf_method": args.hf_method,
            }
            print(f"[ERR] {dataset_slug} fold={fold}: {type(exc).__name__}: {exc}")
            print(row["traceback"])
            if args.stop_on_error:
                _append_row(output_path, row)
                sync_paths = _collect_sync_paths()
                if args.sync_to_head_dir:
                    _sync_artifacts_to_head(
                        sync_dir=args.sync_to_head_dir,
                        paths=sync_paths,
                    )
                if args.gcs_artifact_dir:
                    _sync_artifacts_to_gcs(
                        gcs_dir=args.gcs_artifact_dir,
                        paths=sync_paths,
                    )
                    if args.delete_synced_noncore:
                        _delete_synced_noncore_files(
                            paths=sync_paths, keep_paths=core_keep_paths
                        )
                raise

        _append_row(output_path, row)
        rows_since_sync += 1

        if rows_since_sync >= args.sync_every_rows:
            sync_paths = _collect_sync_paths()
            if args.sync_to_head_dir:
                _sync_artifacts_to_head(
                    sync_dir=args.sync_to_head_dir,
                    paths=sync_paths,
                )
            if args.gcs_artifact_dir:
                _sync_artifacts_to_gcs(
                    gcs_dir=args.gcs_artifact_dir,
                    paths=sync_paths,
                )
                if args.delete_synced_noncore:
                    _delete_synced_noncore_files(
                        paths=sync_paths, keep_paths=core_keep_paths
                    )
            rows_since_sync = 0

    if output_path.exists():
        df_out = pd.read_csv(output_path)
        _write_summary(df_out, summary_path)

    sync_paths = _collect_sync_paths()
    if args.sync_to_head_dir:
        _sync_artifacts_to_head(
            sync_dir=args.sync_to_head_dir,
            paths=sync_paths,
        )
    if args.gcs_artifact_dir:
        _sync_artifacts_to_gcs(
            gcs_dir=args.gcs_artifact_dir,
            paths=sync_paths,
        )
        if args.delete_synced_noncore:
            _delete_synced_noncore_files(paths=sync_paths, keep_paths=core_keep_paths)

    print(
        f"[Done] attempted={total_attempted} ok={total_ok} error={total_err}\n"
        f"Results CSV: {output_path}\n"
        f"Summary CSV: {summary_path}"
    )


if __name__ == "__main__":
    main()
