import argparse
import glob
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--summary_csv", type=str, required=True)
    return parser.parse_args()


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
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched --input_glob: {args.input_glob}")

    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue

    if not frames:
        raise RuntimeError(f"No readable CSV content for --input_glob: {args.input_glob}")

    combined = pd.concat(frames, ignore_index=True)
    combined["_row_idx"] = range(len(combined))
    if {"dataset_slug", "fold"}.issubset(combined.columns):
        combined = combined.sort_values("_row_idx").drop_duplicates(
            subset=["dataset_slug", "fold"],
            keep="last",
        )
    combined = combined.drop(columns=["_row_idx"]).reset_index(drop=True)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    _write_summary(combined, Path(args.summary_csv))

    ok = int((combined.get("status") == "ok").sum()) if "status" in combined.columns else 0
    err = int((combined.get("status") == "error").sum()) if "status" in combined.columns else 0
    print(f"[Combined] files={len(files)} rows={len(combined)} ok={ok} error={err}")
    print(f"[Combined] output={output_path}")
    print(f"[Combined] summary={args.summary_csv}")


if __name__ == "__main__":
    main()
