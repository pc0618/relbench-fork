import argparse
import traceback
from pathlib import Path

import pandas as pd

from relbench.datasets.tabarena import TABARENA_DATASETS, get_tabarena_dataset_slugs
from tabpfn_tabarena import HF_RESULTS_URL, load_hf_results, run_tabarena_fold


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
    return parser.parse_args()


def _parse_dataset_slugs(arg: str) -> list[str]:
    if arg.strip().lower() == "all":
        return get_tabarena_dataset_slugs()
    slugs = [s.strip() for s in arg.split(",") if s.strip()]
    valid = set(get_tabarena_dataset_slugs())
    invalid = [s for s in slugs if s not in valid]
    if invalid:
        raise ValueError(f"Unknown dataset slugs: {invalid}")
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
    dataset_slugs = _parse_dataset_slugs(args.dataset_slugs)
    output_path = Path(args.output_csv)
    summary_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else output_path.with_name(f"{output_path.stem}_summary.csv")
    )

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
    for dataset_slug in dataset_slugs:
        spec = TABARENA_DATASETS[dataset_slug]
        folds = _parse_fold_spec_for_dataset(args.folds, fold_count=spec.fold_count)
        if not folds:
            print(
                f"[Skip] dataset={dataset_slug}: no valid folds selected for fold_count={spec.fold_count}"
            )
            continue

        for fold in folds:
            if args.max_runs > 0 and total_attempted >= args.max_runs:
                print(f"[Stop] reached max_runs={args.max_runs}")
                break
            if (dataset_slug, fold) in done_pairs:
                print(f"[Resume] skipping already done: {dataset_slug} fold={fold}")
                continue

            total_attempted += 1
            print(
                f"[Run {total_attempted}] dataset={dataset_slug} fold={fold} "
                f"sample_size={args.sample_size} n_estimators={args.n_estimators}"
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
                )
                total_ok += 1
                print(
                    f"[OK] {dataset_slug} fold={fold} "
                    f"test={row['test_metric_error']:.6f} "
                    f"hf={row.get('hf_metric_error', float('nan')):.6f} "
                    f"delta={row['hf_delta_local_minus_hf']:+.6f}"
                )
            except Exception as exc:
                total_err += 1
                row = {
                    "dataset_slug": dataset_slug,
                    "dataset_name": f"tabarena-{dataset_slug}",
                    "task_name": f"fold-{fold}",
                    "fold": int(fold),
                    "problem_type": None,
                    "n_estimators": int(args.n_estimators),
                    "sample_size": int(args.sample_size),
                    "seed": int(args.seed),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                    "hf_method": args.hf_method,
                }
                print(f"[ERR] {dataset_slug} fold={fold}: {type(exc).__name__}: {exc}")
                if args.stop_on_error:
                    _append_row(output_path, row)
                    raise

            _append_row(output_path, row)
        else:
            continue
        break

    if output_path.exists():
        df_out = pd.read_csv(output_path)
        _write_summary(df_out, summary_path)

    print(
        f"[Done] attempted={total_attempted} ok={total_ok} error={total_err}\n"
        f"Results CSV: {output_path}\n"
        f"Summary CSV: {summary_path}"
    )


if __name__ == "__main__":
    main()
