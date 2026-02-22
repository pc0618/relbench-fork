import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch

from relbench.base import TaskType
from relbench.datasets.tabarena import TABARENA_DATASETS, get_tabarena_dataset_slugs
from relbench.tasks import get_task

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

HF_RESULTS_URL = (
    "https://huggingface.co/datasets/TabArena/benchmark_results/resolve/main/"
    "df_results.parquet"
)

INFERENCE_PRECISION_CHOICES = (
    "auto",
    "autocast",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)
XLA_MATMUL_PRECISION_CHOICES = {"highest", "high", "medium"}
_THREAD_CONFIG_CACHE: Optional[tuple[int, int]] = None


def _import_tabpfn():
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tabpfn is required for this script. Install it with:\n"
            "  pip install tabpfn"
        ) from exc
    return TabPFNClassifier, TabPFNRegressor, ModelVersion


def _resolve_inference_precision(value: str) -> Any:
    key = value.strip().lower()
    if key in {"auto", "autocast"}:
        return key

    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if key not in mapping:
        raise ValueError(
            f"Unknown inference precision {value!r}. Choices: {INFERENCE_PRECISION_CHOICES}"
        )
    return mapping[key]


def _normalize_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "xla":
        return "xla:0"
    return normalized


def _validate_device_runtime(device: str) -> None:
    if not device.startswith("xla"):
        return
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "device=xla requested but torch_xla is not installed. "
            "Install torch-xla in the runtime environment."
        ) from exc

    if not hasattr(torch, "xla"):
        torch.xla = torch_xla

    xla_device = xm.xla_device()
    if str(xla_device).lower() != str(device).lower():
        print(f"[Runtime] requested device={device}, resolved_xla_device={xla_device}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_runtime_env(
    *,
    pjrt_device: str,
    xla_use_bf16: bool,
    xla_downcast_bf16: bool,
    xla_default_matmul_precision: str,
    xla_flags: str,
    torch_num_threads: int,
    torch_num_interop_threads: int,
) -> None:
    global _THREAD_CONFIG_CACHE

    if pjrt_device:
        os.environ["PJRT_DEVICE"] = pjrt_device
    if xla_use_bf16:
        os.environ["XLA_USE_BF16"] = "1"
    if xla_downcast_bf16:
        os.environ["XLA_DOWNCAST_BF16"] = "1"
    if xla_default_matmul_precision:
        precision = xla_default_matmul_precision.strip().lower()
        if precision not in XLA_MATMUL_PRECISION_CHOICES:
            raise ValueError(
                "xla_default_matmul_precision must be one of "
                f"{sorted(XLA_MATMUL_PRECISION_CHOICES)}"
            )
        os.environ["XLA_DEFAULT_MATMUL_PRECISION"] = precision
    if xla_flags:
        os.environ["XLA_FLAGS"] = xla_flags

    thread_cfg = (int(torch_num_threads), int(torch_num_interop_threads))
    if thread_cfg == (0, 0):
        return

    if _THREAD_CONFIG_CACHE is None:
        import torch

        if torch_num_threads > 0:
            torch.set_num_threads(torch_num_threads)
        if torch_num_interop_threads > 0:
            torch.set_num_interop_threads(torch_num_interop_threads)
        _THREAD_CONFIG_CACHE = thread_cfg
        return

    if _THREAD_CONFIG_CACHE != thread_cfg:
        raise ValueError(
            "torch thread settings were already configured earlier in this process as "
            f"{_THREAD_CONFIG_CACHE}, cannot switch to {thread_cfg}."
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
        "--device",
        type=str,
        default="cpu",
        help=(
            "TabPFN device spec (e.g. cpu, cuda, auto, xla, xla:0). "
            "Use xla/xla:0 with torch-xla on TPU VMs."
        ),
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
        help="If set, exports PJRT_DEVICE before TabPFN model construction (e.g. TPU).",
    )
    parser.add_argument(
        "--xla_use_bf16",
        action="store_true",
        default=False,
        help="Set XLA_USE_BF16=1 for TPU execution.",
    )
    parser.add_argument(
        "--xla_downcast_bf16",
        action="store_true",
        default=False,
        help="Set XLA_DOWNCAST_BF16=1 for TPU execution.",
    )
    parser.add_argument(
        "--xla_default_matmul_precision",
        type=str,
        default="",
        help="Set XLA_DEFAULT_MATMUL_PRECISION to one of: highest, high, medium.",
    )
    parser.add_argument(
        "--xla_flags",
        type=str,
        default="",
        help="Optional XLA_FLAGS string.",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=0,
        help="If >0, call torch.set_num_threads(n).",
    )
    parser.add_argument(
        "--torch_num_interop_threads",
        type=int,
        default=0,
        help="If >0, call torch.set_num_interop_threads(n).",
    )
    parser.add_argument(
        "--log_runtime_config",
        action="store_true",
        default=False,
        help="Print runtime config/env values before execution.",
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
    device: str = "cpu",
    inference_precision: str = "auto",
    pjrt_device: str = "",
    xla_use_bf16: bool = False,
    xla_downcast_bf16: bool = False,
    xla_default_matmul_precision: str = "",
    xla_flags: str = "",
    torch_num_threads: int = 0,
    torch_num_interop_threads: int = 0,
    log_runtime_config: bool = False,
) -> dict:
    if fold < 0 or fold >= TABARENA_DATASETS[dataset_slug].fold_count:
        raise ValueError(
            f"Fold {fold} is invalid for {dataset_slug}. "
            f"Valid range is [0, {TABARENA_DATASETS[dataset_slug].fold_count - 1}]."
        )

    normalized_device = _normalize_device(device)

    _configure_runtime_env(
        pjrt_device=pjrt_device,
        xla_use_bf16=xla_use_bf16,
        xla_downcast_bf16=xla_downcast_bf16,
        xla_default_matmul_precision=xla_default_matmul_precision,
        xla_flags=xla_flags,
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
    )

    if log_runtime_config:
        print(
            "[Runtime] "
            f"device={normalized_device} inference_precision={inference_precision} "
            f"PJRT_DEVICE={os.getenv('PJRT_DEVICE', '')} "
            f"XLA_USE_BF16={os.getenv('XLA_USE_BF16', '')} "
            f"XLA_DOWNCAST_BF16={os.getenv('XLA_DOWNCAST_BF16', '')} "
            f"XLA_DEFAULT_MATMUL_PRECISION={os.getenv('XLA_DEFAULT_MATMUL_PRECISION', '')}"
        )

    _validate_device_runtime(normalized_device)

    TabPFNClassifier, TabPFNRegressor, ModelVersion = _import_tabpfn()
    resolved_inference_precision = _resolve_inference_precision(inference_precision)

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
    _seed_everything(seed)
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
            device=normalized_device,
            inference_precision=resolved_inference_precision,
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
            device=normalized_device,
            inference_precision=resolved_inference_precision,
            ignore_pretraining_limits=ignore_pretraining_limits,
            n_preprocessing_jobs=n_preprocessing_jobs,
            random_state=seed,
            fit_mode="fit_preprocessors",
        )

    # The first fit also warms up and downloads weights if needed.
    model.fit(X_train_t, y_train)

    if normalized_device.startswith("xla"):
        executor = getattr(model, "executor_", None)
        toggle = getattr(executor, "use_torch_inference_mode", None)
        if callable(toggle):
            # TabPFN re-enables torch.inference_mode() during predict(); on torch-xla
            # this can fail with "Cannot set version_counter for inference tensor".
            toggle(use_inference=False)

            def _keep_inference_mode_off(*, use_inference: bool) -> None:
                return None

            executor.use_torch_inference_mode = _keep_inference_mode_off

    model_train_seconds = float(time.time() - model_tic)

    restore_is_tracing = None
    if normalized_device.startswith("xla"):
        # TabPFN uses torch.Generator(device=xla:0) for positional embeddings.
        # On current torch-xla/libtpu this raises "XLA device type not an accelerator".
        # Forcing the tracing path skips the device-bound generator creation.
        restore_is_tracing = torch.jit.is_tracing
        torch.jit.is_tracing = lambda: True

    try:
        if task.task_type == TaskType.REGRESSION:
            train_pred = model.predict(X_train_t)
            val_pred = model.predict(X_val_t)
            test_pred = model.predict(X_test_t)
        else:
            train_pred = model.predict_proba(X_train_t)
            val_pred = model.predict_proba(X_val_t)
            test_pred = model.predict_proba(X_test_t)
    finally:
        if restore_is_tracing is not None:
            torch.jit.is_tracing = restore_is_tracing

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
        "device": str(normalized_device),
        "inference_precision": str(inference_precision),
        "pjrt_device": os.getenv("PJRT_DEVICE", ""),
        "xla_use_bf16": os.getenv("XLA_USE_BF16", ""),
        "xla_downcast_bf16": os.getenv("XLA_DOWNCAST_BF16", ""),
        "xla_default_matmul_precision": os.getenv("XLA_DEFAULT_MATMUL_PRECISION", ""),
        "torch_num_threads": int(torch_num_threads),
        "torch_num_interop_threads": int(torch_num_interop_threads),
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
        device=args.device,
        inference_precision=args.inference_precision,
        pjrt_device=args.pjrt_device,
        xla_use_bf16=args.xla_use_bf16,
        xla_downcast_bf16=args.xla_downcast_bf16,
        xla_default_matmul_precision=args.xla_default_matmul_precision,
        xla_flags=args.xla_flags,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
        log_runtime_config=args.log_runtime_config,
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
