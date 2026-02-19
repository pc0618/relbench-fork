import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch_frame
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything

from relbench.base import EntityTask, TaskType
from relbench.datasets.tabarena import TABARENA_DATASETS, get_tabarena_dataset_slugs
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from relbench.tasks import get_task

HF_RESULTS_URL = (
    "https://huggingface.co/datasets/TabArena/benchmark_results/resolve/main/"
    "df_results.parquet"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_slug",
        type=str,
        default="credit-g",
        choices=get_tabarena_dataset_slugs(),
        help="TabArena dataset slug used in relbench as tabarena-<slug>.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="OpenML fold index used in relbench task name fold-<k>.",
    )
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50_000,
        help="Subsample size for LightGBM training rows. Use <=0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/relbench_examples"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Try to download prepared relbench artifacts (TabArena tasks skip this).",
    )
    parser.add_argument(
        "--hf_method",
        type=str,
        default="GBM (default)",
        help="TabArena method name in benchmark_results/df_results.parquet for comparison.",
    )
    parser.add_argument(
        "--hf_results_url",
        type=str,
        default=HF_RESULTS_URL,
        help="URL to TabArena benchmark parquet.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional output CSV path. If set, appends one row.",
    )
    return parser.parse_args()


def _get_col_to_stype(
    dataset,
    task: EntityTask,
    cache_dir: str,
    dataset_name: str,
    task_name: str,
) -> Dict[str, stype]:
    stypes_cache_path = Path(
        f"{cache_dir}/{dataset_name}/tasks/{task_name}/stypes.json"
    )
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    entity_table = dataset.get_db().table_dict[task.entity_table]
    col_to_stype = col_to_stype_dict[task.entity_table]
    remove_pkey_fkey(col_to_stype, entity_table)

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        col_to_stype[task.target_col] = torch_frame.categorical
    elif task.task_type == TaskType.REGRESSION:
        col_to_stype[task.target_col] = torch_frame.numerical
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        col_to_stype[task.target_col] = torch_frame.categorical
    else:
        raise ValueError(f"Unsupported task type called {task.task_type}")

    return col_to_stype


def _select_tune_metric(task: EntityTask) -> Metric:
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return Metric.ROCAUC
    if task.task_type == TaskType.REGRESSION:
        return Metric.RMSE
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return Metric.ACCURACY
    raise ValueError(f"Unsupported task type called {task.task_type}")


def load_hf_results(hf_results_url: str) -> pd.DataFrame:
    return pd.read_parquet(hf_results_url)


def _compare_against_hf(
    *,
    dataset_slug: str,
    fold: int,
    hf_method: str,
    local_metric_error: float,
    hf_results_df: pd.DataFrame,
) -> dict:
    dataset_name = TABARENA_DATASETS[dataset_slug].name
    fold_rows = hf_results_df[
        (hf_results_df["dataset"] == dataset_name)
        & (hf_results_df["method"] == hf_method)
        & (hf_results_df["fold"] == fold)
    ]
    if fold_rows.empty:
        return {
            "hf_found": False,
            "hf_dataset_name": dataset_name,
            "hf_method": hf_method,
            "hf_metric_name": None,
            "hf_metric_error": np.nan,
            "hf_delta_local_minus_hf": np.nan,
            "hf_local_rank": np.nan,
            "hf_total_methods": np.nan,
            "hf_best_metric_error": np.nan,
        }

    hf_metric_error = float(fold_rows.iloc[0]["metric_error"])
    metric_name = str(fold_rows.iloc[0]["metric"])
    delta = local_metric_error - hf_metric_error

    fold_all = hf_results_df[
        (hf_results_df["dataset"] == dataset_name) & (hf_results_df["fold"] == fold)
    ]
    if fold_all.empty:
        local_rank = np.nan
        total = np.nan
        best = np.nan
    else:
        total = int(len(fold_all))
        local_rank = int((fold_all["metric_error"] < local_metric_error).sum() + 1)
        local_rank = min(local_rank, total)
        best = float(fold_all["metric_error"].min())

    return {
        "hf_found": True,
        "hf_dataset_name": dataset_name,
        "hf_method": hf_method,
        "hf_metric_name": metric_name,
        "hf_metric_error": hf_metric_error,
        "hf_delta_local_minus_hf": delta,
        "hf_local_rank": local_rank,
        "hf_total_methods": total,
        "hf_best_metric_error": best,
    }


def run_tabarena_fold(
    *,
    dataset_slug: str,
    fold: int,
    num_trials: int = 10,
    sample_size: int = 50_000,
    seed: int = 42,
    cache_dir: str = os.path.expanduser("~/.cache/relbench_examples"),
    download: bool = False,
    hf_method: str = "GBM (default)",
    hf_results_url: str = HF_RESULTS_URL,
    hf_results_df: Optional[pd.DataFrame] = None,
) -> dict:
    spec = TABARENA_DATASETS[dataset_slug]
    if fold < 0 or fold >= spec.fold_count:
        raise ValueError(
            f"Fold {fold} is invalid for {dataset_slug}. "
            f"Valid range is [0, {spec.fold_count - 1}]."
        )

    run_tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    seed_everything(seed)

    dataset_name = f"tabarena-{dataset_slug}"
    task_name = f"fold-{fold}"
    task: EntityTask = get_task(dataset_name, task_name, download=download)
    dataset = task.dataset

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df
    col_to_stype = _get_col_to_stype(
        dataset, task, cache_dir, dataset_name, task_name
    )

    if sample_size > 0 and sample_size < len(train_table):
        sampled_idx = np.random.permutation(len(train_table))[:sample_size]
        train_table.df = train_table.df.iloc[sampled_idx]

    dfs: Dict[str, pd.DataFrame] = {}
    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df = entity_df.astype({entity_table.pkey_col: table.df[left_entity].dtype})
        dfs[split] = table.df.merge(
            entity_df,
            how="left",
            left_on=left_entity,
            right_on=entity_table.pkey_col,
        )

    train_dataset = torch_frame.data.Dataset(
        df=dfs["train"],
        col_to_stype=col_to_stype,
        target_col=task.target_col,
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=256,
        ),
    )
    sample_tag = str(sample_size) if sample_size > 0 else "all"
    materialized_path = Path(
        f"{cache_dir}/{dataset_name}/tasks/{task_name}/materialized/"
        f"sample_{sample_tag}_node_train.pt"
    )
    materialized_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset = train_dataset.materialize(path=materialized_path)

    tf_train = train_dataset.tensor_frame
    tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
    tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

    model_kwargs = {
        "task_type": train_dataset.task_type,
        "metric": _select_tune_metric(task),
    }
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        model_kwargs["num_classes"] = task.num_classes

    model_tic = time.time()
    model = LightGBM(**model_kwargs)
    model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=num_trials)
    model_train_seconds = float(time.time() - model_tic)

    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        train_x, _, _ = model._to_lightgbm_input(tf_train)
        val_x, _, _ = model._to_lightgbm_input(tf_val)
        test_x, _, _ = model._to_lightgbm_input(tf_test)
        train_pred = model.model.predict(train_x)
        val_pred = model.model.predict(val_x)
        test_pred = model.model.predict(test_x)
    else:
        train_pred = model.predict(tf_test=tf_train).numpy()
        val_pred = model.predict(tf_test=tf_val).numpy()
        test_pred = model.predict(tf_test=tf_test).numpy()

    train_metrics = task.evaluate(train_pred, train_table)
    val_metrics = task.evaluate(val_pred, val_table)
    test_metrics = task.evaluate(test_pred)
    local_metric_error = float(test_metrics["metric_error"])

    if hf_results_df is None:
        hf_results_df = load_hf_results(hf_results_url)
    hf_cmp = _compare_against_hf(
        dataset_slug=dataset_slug,
        fold=fold,
        hf_method=hf_method,
        local_metric_error=local_metric_error,
        hf_results_df=hf_results_df,
    )

    row = {
        "dataset_slug": dataset_slug,
        "dataset_name": dataset_name,
        "task_name": task_name,
        "fold": int(fold),
        "problem_type": str(task.task_type.value),
        "num_trials": int(num_trials),
        "sample_size": int(sample_size),
        "seed": int(seed),
        "train_metric_error": float(train_metrics["metric_error"]),
        "val_metric_error": float(val_metrics["metric_error"]),
        "test_metric_error": local_metric_error,
        "train_rows": int(len(train_table)),
        "val_rows": int(len(val_table)),
        "test_rows": int(len(test_table)),
        "model_train_seconds": model_train_seconds,
        "run_seconds": float(time.time() - run_tic),
        "status": "ok",
    }
    row.update(hf_cmp)
    return row


def _append_csv(path: str, row: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)


def main() -> None:
    args = _parse_args()
    row = run_tabarena_fold(
        dataset_slug=args.dataset_slug,
        fold=args.fold,
        num_trials=args.num_trials,
        sample_size=args.sample_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
        download=args.download,
        hf_method=args.hf_method,
        hf_results_url=args.hf_results_url,
    )
    print(f"Dataset={row['dataset_name']} Task={row['task_name']}")
    print(f"Train: metric_error={row['train_metric_error']:.6f}")
    print(f"Val: metric_error={row['val_metric_error']:.6f}")
    print(f"Test: metric_error={row['test_metric_error']:.6f}")
    if row["hf_found"]:
        print(
            f"[HF] method={row['hf_method']} metric={row['hf_metric_name']} "
            f"metric_error={row['hf_metric_error']:.6f}"
        )
        print(
            f"[HF] delta_local_minus_hf={row['hf_delta_local_minus_hf']:+.6f} "
            f"rank~{int(row['hf_local_rank'])}/{int(row['hf_total_methods'])}"
        )
    else:
        print(
            f"[HF] No row found for method={args.hf_method} "
            f"dataset={row['hf_dataset_name']} fold={args.fold}"
        )
    if args.output_csv:
        _append_csv(args.output_csv, row)
        print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
