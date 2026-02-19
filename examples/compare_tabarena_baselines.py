import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from relbench.datasets.tabarena import TABARENA_DATASETS, TabArenaDataset

HF_RESULTS_URL = (
    "https://huggingface.co/datasets/TabArena/benchmark_results/resolve/main/"
    "df_results.parquet"
)

MODEL_TO_HF_METHOD = {
    "lr": "LR (default)",
    "rf": "RF (default)",
    "lgbm": "GBM (default)",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_slug",
        type=str,
        default="credit-g",
        choices=sorted(TABARENA_DATASETS.keys()),
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0-4",
        help="Comma-separated list/ranges, e.g. 0,1,2-4",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lr,rf,lgbm",
        help="Comma-separated from: lr,rf,lgbm",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--hf_results_url", type=str, default=HF_RESULTS_URL)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/tabarena_model_compare.csv",
    )
    parser.add_argument("--append", action="store_true", default=False)
    return parser.parse_args()


def _parse_folds(fold_arg: str, fold_count: int) -> list[int]:
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
    folds = {f for f in folds if 0 <= f < fold_count}
    return sorted(folds)


def _make_preprocessor(X: pd.DataFrame, *, onehot: bool) -> ColumnTransformer:
    cat_cols = [
        c
        for c in X.columns
        if pd.api.types.is_object_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    if onehot:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
    else:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def _make_model_pipeline(
    model_key: str,
    problem_type: str,
    X: pd.DataFrame,
    seed: int,
    rf_n_estimators: int,
):
    if model_key == "lr":
        if problem_type == "regression":
            raise ValueError("lr is classification-only")
        pre = _make_preprocessor(X, onehot=True)
        model = LogisticRegression(
            solver="saga",
            max_iter=2000,
            n_jobs=-1,
            random_state=seed,
        )
    elif model_key == "rf":
        pre = _make_preprocessor(X, onehot=True)
        if problem_type == "regression":
            model = RandomForestRegressor(
                n_estimators=rf_n_estimators,
                random_state=seed,
                n_jobs=-1,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                random_state=seed,
                n_jobs=-1,
            )
    elif model_key == "lgbm":
        pre = _make_preprocessor(X, onehot=False)
        if problem_type == "regression":
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(random_state=seed, verbose=-1)
        else:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(random_state=seed, verbose=-1)
    else:
        raise ValueError(f"Unknown model_key={model_key}")

    return Pipeline(
        steps=[
            ("pre", pre),
            ("model", model),
        ]
    )


def _metric_error(problem_type: str, y_true: np.ndarray, pred) -> float:
    if problem_type == "binary":
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        return float(1.0 - roc_auc_score(y_true.astype(np.int64), pred))
    if problem_type == "multiclass":
        pred = np.asarray(pred, dtype=np.float64)
        labels = np.arange(pred.shape[1], dtype=np.int64)
        return float(log_loss(y_true.astype(np.int64), pred, labels=labels))
    if problem_type == "regression":
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        return float(np.sqrt(np.mean((y_true.astype(np.float64) - pred) ** 2)))
    raise ValueError(f"Unknown problem_type={problem_type}")


def _predict_for_metric(model, X: pd.DataFrame, problem_type: str):
    if problem_type == "regression":
        return model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if problem_type == "binary":
            if prob.ndim == 2 and prob.shape[1] > 1:
                return prob[:, 1]
            return prob.reshape(-1)
        return prob
    if problem_type == "binary":
        raw = model.decision_function(X)
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        raw = np.clip(raw, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-raw))
    labels = np.asarray(model.predict(X), dtype=np.int64).reshape(-1)
    num_classes = int(labels.max(initial=0) + 1)
    eps = 1e-7
    probs = np.full(
        (len(labels), num_classes),
        fill_value=eps / max(num_classes - 1, 1),
        dtype=np.float64,
    )
    probs[np.arange(len(labels), dtype=np.int64), labels] = 1.0 - eps
    return probs


def main() -> None:
    args = _parse_args()
    spec = TABARENA_DATASETS[args.dataset_slug]
    folds = _parse_folds(args.folds, fold_count=spec.fold_count)
    if not folds:
        raise ValueError(f"No valid folds selected from {args.folds}")

    requested_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    allowed = {"lr", "rf", "lgbm"}
    bad = [m for m in requested_models if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported models: {bad}. Allowed={sorted(allowed)}")

    dataset = TabArenaDataset(dataset_slug=args.dataset_slug)
    problem_type = dataset.problem_type
    X = dataset.get_db().table_dict["records"].df.drop(columns=["record_id"])
    y = dataset.get_target_array()
    hf_df = pd.read_parquet(args.hf_results_url)

    rows: list[dict] = []
    for fold in folds:
        train_idx, test_idx = dataset.get_openml_fold_indices(fold)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_key in requested_models:
            if problem_type == "regression" and model_key == "lr":
                continue
            tic = time.time()
            model = _make_model_pipeline(
                model_key=model_key,
                problem_type=problem_type,
                X=X_train,
                seed=args.seed,
                rf_n_estimators=args.rf_n_estimators,
            )
            model.fit(X_train, y_train)
            pred = _predict_for_metric(model, X_test, problem_type=problem_type)
            metric_error = _metric_error(problem_type, y_test, pred)
            fit_seconds = float(time.time() - tic)

            hf_method = MODEL_TO_HF_METHOD[model_key]
            hf_row = hf_df[
                (hf_df["dataset"] == spec.name)
                & (hf_df["fold"] == fold)
                & (hf_df["method"] == hf_method)
            ]
            if hf_row.empty:
                hf_metric_error = np.nan
                hf_metric_name = None
                delta = np.nan
            else:
                hf_metric_error = float(hf_row.iloc[0]["metric_error"])
                hf_metric_name = str(hf_row.iloc[0]["metric"])
                delta = float(metric_error - hf_metric_error)

            rows.append(
                {
                    "dataset_slug": args.dataset_slug,
                    "dataset_name": spec.name,
                    "problem_type": problem_type,
                    "fold": int(fold),
                    "model_key": model_key,
                    "hf_method": hf_method,
                    "hf_metric_name": hf_metric_name,
                    "metric_error_local": float(metric_error),
                    "metric_error_hf": hf_metric_error,
                    "delta_local_minus_hf": delta,
                    "fit_seconds": fit_seconds,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "seed": int(args.seed),
                }
            )
            print(
                f"[{args.dataset_slug} fold={fold}] {model_key}: "
                f"local={metric_error:.6f} hf={hf_metric_error:.6f} "
                f"delta={delta:+.6f}"
            )

    df = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.append and output_path.exists():
        df.to_csv(output_path, mode="a", header=False, index=False)
        df_full = pd.read_csv(output_path)
    else:
        df.to_csv(output_path, index=False)
        df_full = df

    summary = (
        df_full.groupby(["dataset_slug", "problem_type", "model_key", "hf_method"], as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_local=("metric_error_local", "mean"),
            std_local=("metric_error_local", "std"),
            mean_hf=("metric_error_hf", "mean"),
            mean_delta=("delta_local_minus_hf", "mean"),
        )
        .sort_values(by=["mean_delta", "mean_local"], ascending=[True, True])
        .reset_index(drop=True)
    )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[Saved] {output_path}")
    print(f"[Saved] {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
