import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HF_COLUMNS = [
    "dataset",
    "fold",
    "method",
    "metric_error",
    "time_train_s",
    "time_infer_s",
    "metric_error_val",
    "seed",
    "method_metadata",
    "ensemble_weight",
    "problem_type",
    "metric",
    "method_type",
    "method_subtype",
    "config_type",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lightgbm_csv",
        type=str,
        default="results/tabarena_lightgbm_credit_g_trials5.csv",
        help="CSV from lightgbm_tabarena_batch.py (optional).",
    )
    parser.add_argument(
        "--rf_csv",
        type=str,
        default="results/tabarena_rf_all51_fold0.csv",
        help="CSV from compare_tabarena_baselines.py with model_key=rf (optional).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/tabarena_relbench_hf_compatible.csv",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/tabarena_relbench_pr_summary.csv",
    )
    return parser.parse_args()


def _problem_type_from_relbench(problem_type: str) -> str:
    mapping = {
        "binary_classification": "binary",
        "multiclass_classification": "multiclass",
        "regression": "regression",
    }
    return mapping.get(problem_type, problem_type)


def _method_config_type(method: str) -> str | None:
    if method.startswith("GBM"):
        return "GBM"
    if method.startswith("RF"):
        return "RF"
    if method.startswith("LR"):
        return "LR"
    return None


def _from_lightgbm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=HF_COLUMNS)
    out = pd.DataFrame(
        {
            "dataset": df["hf_dataset_name"].fillna(df["dataset_slug"]),
            "fold": df["fold"].astype(int),
            "method": "GBM (default) [RelBench-LightGBM]",
            "metric_error": df["test_metric_error"].astype(float),
            "time_train_s": df.get("model_train_seconds", np.nan).astype(float),
            "time_infer_s": np.nan,
            "metric_error_val": df.get("val_metric_error", np.nan).astype(float),
            "seed": df.get("seed", 42).astype(int),
            "ensemble_weight": np.nan,
            "problem_type": df["problem_type"].map(_problem_type_from_relbench),
            "metric": df["hf_metric_name"],
            "method_type": "baseline",
            "method_subtype": "default",
            "config_type": "GBM",
        }
    )
    metadata = []
    for _, row in df.iterrows():
        metadata.append(
            json.dumps(
                {
                    "source": "relbench.examples.lightgbm_tabarena_batch",
                    "num_trials": int(row.get("num_trials", 0)),
                    "sample_size": int(row.get("sample_size", 0)),
                    "seed": int(row.get("seed", 42)),
                },
                sort_keys=True,
            )
        )
    out["method_metadata"] = metadata
    return out[HF_COLUMNS]


def _from_rf_compare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=HF_COLUMNS)
    df = df[df["model_key"] == "rf"].copy()
    if df.empty:
        return pd.DataFrame(columns=HF_COLUMNS)
    out = pd.DataFrame(
        {
            "dataset": df["dataset_name"],
            "fold": df["fold"].astype(int),
            "method": "RF (default) [RelBench-Sklearn]",
            "metric_error": df["metric_error_local"].astype(float),
            "time_train_s": df["fit_seconds"].astype(float),
            "time_infer_s": np.nan,
            "metric_error_val": np.nan,
            "seed": df.get("seed", 42).astype(int),
            "ensemble_weight": np.nan,
            "problem_type": df["problem_type"],
            "metric": df["hf_metric_name"],
            "method_type": "baseline",
            "method_subtype": "default",
            "config_type": "RF",
        }
    )
    out["method_metadata"] = json.dumps(
        {"source": "relbench.examples.compare_tabarena_baselines", "model_key": "rf"},
        sort_keys=True,
    )
    return out[HF_COLUMNS]


def _make_pr_summary(df_hf: pd.DataFrame) -> pd.DataFrame:
    if df_hf.empty:
        return pd.DataFrame()
    df = df_hf.copy()
    summary = (
        df.groupby(["method", "config_type", "problem_type"], as_index=False)
        .agg(
            runs=("dataset", "count"),
            datasets=("dataset", "nunique"),
            mean_metric_error=("metric_error", "mean"),
            std_metric_error=("metric_error", "std"),
            mean_time_train_s=("time_train_s", "mean"),
        )
        .sort_values(by=["method", "problem_type"])
        .reset_index(drop=True)
    )
    return summary


def main() -> None:
    args = _parse_args()
    dfs = []

    lightgbm_path = Path(args.lightgbm_csv)
    if lightgbm_path.exists():
        dfs.append(_from_lightgbm(pd.read_csv(lightgbm_path)))
    else:
        print(f"[Warn] missing lightgbm_csv: {lightgbm_path}")

    rf_path = Path(args.rf_csv)
    if rf_path.exists():
        dfs.append(_from_rf_compare(pd.read_csv(rf_path)))
    else:
        print(f"[Warn] missing rf_csv: {rf_path}")

    if not dfs:
        raise FileNotFoundError("No input CSVs found.")

    df_hf = pd.concat(dfs, ignore_index=True)
    for col in HF_COLUMNS:
        if col not in df_hf.columns:
            df_hf[col] = np.nan
    df_hf = df_hf[HF_COLUMNS]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_hf.to_csv(output_path, index=False)
    print(f"[Saved] {output_path} rows={len(df_hf)}")

    summary = _make_pr_summary(df_hf=df_hf)
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path} rows={len(summary)}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
