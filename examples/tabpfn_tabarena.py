import argparse
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from relbench.base import TaskType
from relbench.datasets.tabarena import TABARENA_DATASETS, get_tabarena_dataset_slugs
from relbench.tasks import get_task

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "tabpfn is required for this script. Install it with:\n"
        "  pip install tabpfn"
    ) from exc

from torch_geometric.seed import seed_everything


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
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50_000,
        help="Subsample size for TabPFN training rows. Use <=0 to disable.",
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
        default="TabPFN (default)",
        help="Method name for HF benchmark comparison.",
    )
    parser.add_argument(
        "--hf_results_url",
        type=str,
        default=HF_RESULTS_URL,
        help="URL to TabArena benchmark parquet.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=8,
        help="Number of TabPFN estimators in the internal ensemble.",
    )
    parser.add_argument(
        "--n_preprocessing_jobs",
        type=int,
        default=1,
        help="Preprocessing worker count used by TabPFN internals.",
    )
    parser.add_argument(
        "--ignore_pretraining_limits",
        action="store_true",
        default=False,
        help=(
            "Ignore TabPFN pre-training limits; required for some feature/sample sizes."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional output CSV path. If set, appends one row.",
    )
    return parser.parse_args()


def _select_feature_columns(df: pd.DataFrame, target_col: str, id_col: str) -> List[str]:
    # Keep all non-target fields except the synthetic split marker and entity ids.
    drop = {target_col, "timestamp", id_col}
    return [c for c in df.columns if c not in drop]


def _build_split_frame(task, entity_df: pd.DataFrame, split: str):
    table = task.get_table(split, mask_input_cols=False)
    fkeys = list(table.fkey_col_to_pkey_table.keys())
    if not fkeys:
        raise RuntimeError(f"No FKs found for split={split} dataset task.")
    left_entity = fkeys[0]
    pkey_col = entity_df.columns[0]
    merged = table.df.merge(
        entity_df,
        how="left",
        left_on=left_entity,
        right_on=pkey_col,
    )
    return table, merged


def _fit_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray]:
    cat_cols = [
        c
        for c in X_train.columns
        if (
            X_train[c].dtype == object
            or X_train[c].dtype == bool
            or str(X_train[c].dtype).startswith("category")
        )
    ]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessor = Pipeline(
        steps=[
            (
                "union",
                ColumnTransformer(
                    transformers=[
                        ("num", num_pipe, num_cols),
                        ("cat", cat_pipe, cat_cols),
                    ],
                    remainder="drop",
                ),
            )
        ]
    )

    X_train_t = preprocessor.fit_transform(X_train)
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
    return preprocessor, np.asarray(X_train_t), cat_cols, num_cols


def _transform_split(preprocessor: Pipeline, X: pd.DataFrame) -> np.ndarray:
    X_t = preprocessor.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    return np.asarray(X_t)


def _categorical_feature_indices(cat_cols: List[str], num_cols: List[str]) -> Optional[List[int]]:
    if not cat_cols:
        return None
    start = len(num_cols)
    return list(range(start, start + len(cat_cols)))


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
        hf_local_rank = np.nan
        hf_total = np.nan
        hf_best = np.nan
    else:
        hf_total = int(len(fold_all))
        hf_local_rank = int((fold_all["metric_error"] < local_metric_error).sum() + 1)
        hf_local_rank = min(hf_local_rank, hf_total)
        hf_best = float(fold_all["metric_error"].min())

    return {
        "hf_found": True,
        "hf_dataset_name": dataset_name,
        "hf_method": hf_method,
        "hf_metric_name": metric_name,
        "hf_metric_error": hf_metric_error,
        "hf_delta_local_minus_hf": delta,
        "hf_local_rank": hf_local_rank,
        "hf_total_methods": hf_total,
        "hf_best_metric_error": hf_best,
    }


def run_tabarena_fold(
    *,
    dataset_slug: str,
    fold: int,
    sample_size: int = 50_000,
    seed: int = 42,
    cache_dir: str = os.path.expanduser("~/.cache/relbench_examples"),
    download: bool = False,
    hf_method: str = "TabPFN (default)",
    hf_results_url: str = HF_RESULTS_URL,
    hf_results_df: Optional[pd.DataFrame] = None,
    n_estimators: int = 8,
    n_preprocessing_jobs: int = 1,
    ignore_pretraining_limits: bool = False,
) -> dict:
    if fold < 0 or fold >= TABARENA_DATASETS[dataset_slug].fold_count:
        raise ValueError(
            f"Fold {fold} is invalid for {dataset_slug}. "
            f"Valid range is [0, {TABARENA_DATASETS[dataset_slug].fold_count - 1}]."
        )

    run_tic = time.time()
    dataset_name = f"tabarena-{dataset_slug}"
    task_name = f"fold-{fold}"
    task = get_task(dataset_name, task_name, download=download)

    train_table = task.get_table("train", mask_input_cols=False)
    val_table = task.get_table("val", mask_input_cols=False)
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_df = dataset.get_db().table_dict[task.entity_table].df

    train_table, train_df = _build_split_frame(task, entity_df, "train")
    val_table, val_df = _build_split_frame(task, entity_df, "val")
    test_table, test_df = _build_split_frame(task, entity_df, "test")

    feature_cols = _select_feature_columns(
        train_df,
        target_col=task.target_col,
        id_col=task.entity_col,
    )

    X_train = train_df[feature_cols]
    y_train = train_table.df[task.target_col].to_numpy(copy=True)
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    seed_everything(seed)
    np.random.seed(seed)
    if sample_size > 0 and sample_size < len(X_train):
        sampled_idx = np.random.permutation(len(X_train))[:sample_size]
        X_train = X_train.iloc[sampled_idx]
        y_train = y_train[sampled_idx]

    preprocessor, X_train_t, cat_cols, num_cols = _fit_preprocessor(X_train)
    X_val_t = _transform_split(preprocessor, X_val)
    X_test_t = _transform_split(preprocessor, X_test)

    categorical_feature_indices = _categorical_feature_indices(cat_cols, num_cols)

    model_tic = time.time()
    if task.task_type == TaskType.REGRESSION:
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            n_estimators=n_estimators,
            device="cpu",
            ignore_pretraining_limits=ignore_pretraining_limits,
            n_preprocessing_jobs=n_preprocessing_jobs,
            random_state=seed,
            fit_mode="fit_preprocessors",
        )
    else:
        model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            n_estimators=n_estimators,
            categorical_features_indices=categorical_feature_indices,
            device="cpu",
            ignore_pretraining_limits=ignore_pretraining_limits,
            n_preprocessing_jobs=n_preprocessing_jobs,
            random_state=seed,
            fit_mode="fit_preprocessors",
        )

    # The first fit also warms up and downloads weights if needed.
    model.fit(X_train_t, y_train)
    model_train_seconds = float(time.time() - model_tic)

    if task.task_type == TaskType.REGRESSION:
        train_pred = model.predict(X_train_t)
        val_pred = model.predict(X_val_t)
        test_pred = model.predict(X_test_t)
    else:
        train_pred = model.predict_proba(X_train_t)
        val_pred = model.predict_proba(X_val_t)
        test_pred = model.predict_proba(X_test_t)

    train_metrics = task.evaluate(train_pred, target_table=train_table)
    val_metrics = task.evaluate(val_pred, target_table=val_table)
    test_metrics = task.evaluate(test_pred, target_table=test_table)
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

    has_test_target = "target" in test_table.df.columns

    return {
        "dataset_slug": dataset_slug,
        "dataset_name": dataset_name,
        "task_name": task_name,
        "fold": int(fold),
        "problem_type": str(task.task_type.value),
        "num_estimators": int(n_estimators),
        "sample_size": int(sample_size),
        "seed": int(seed),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "n_features": int(len(feature_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "n_numerical_features": int(len(num_cols)),
        "model_train_seconds": model_train_seconds,
        "run_seconds": float(time.time() - run_tic),
        "train_metric_error": float(train_metrics["metric_error"]),
        "val_metric_error": float(val_metrics["metric_error"]),
        "test_metric_error": local_metric_error,
        "status": "ok",
        "test_has_target": has_test_target,
    } | hf_cmp


def _append_csv(path: str, row: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_path, mode="a", header=not output_path.exists(), index=False)


def main() -> None:
    args = _parse_args()
    if args.hf_results_url:
        hf_df = load_hf_results(args.hf_results_url)
    else:
        hf_df = None

    row = run_tabarena_fold(
        dataset_slug=args.dataset_slug,
        fold=args.fold,
        sample_size=args.sample_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
        download=args.download,
        hf_method=args.hf_method,
        hf_results_url=args.hf_results_url,
        hf_results_df=hf_df,
        n_estimators=args.n_estimators,
        n_preprocessing_jobs=args.n_preprocessing_jobs,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
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
