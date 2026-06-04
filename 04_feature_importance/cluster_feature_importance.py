"""Select final energy models and compute cluster-level feature importance.

This script is intentionally independent of the model-development notebooks.
It reads the universal comparison CSV, selects one model per ticker, refits
only those selected models on the pre-2022 training data, and writes feature
importance outputs.
"""

from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DATE_COL = "date"
INSTRUMENT_COL = "instrument"
TARGET_COL = "metalabel"
TEST_START_DATE = "2022-01-01"
TRAIN_ON_NONZERO_SIGNALS_ONLY = True
ENERGY_TICKERS = ["cl1s", "ho1s", "rb1s", "ng1s"]
PROTECTED_FEATURES = ["primary_signal"]
RANDOM_STATE = 42
FEATURE_CLIP_LOWER = 0.01
FEATURE_CLIP_UPPER = 0.99
FEATURE_ABS_CAP = 1_000_000.0

NON_FEATURE_COLS = {
    DATE_COL,
    INSTRUMENT_COL,
    TARGET_COL,
    "target",
    "label",
    "y",
    "meta_label",
    "tb_label",
    "barrier_label",
    "signal_column",
    "training_end",
    "num_days",
    "t1",
    "timeout_date",
    "timeout_close",
    "touch_date",
    "touch_price",
    "touched_barrier",
    "vertical_barrier_date",
    "barrier_touch_date",
    "event_end_date",
    "exit_date",
    "exit_price",
    "triple_barrier_label",
    "holding_period_days",
    "raw_touch_return",
    "signed_touch_return",
    "exit_return",
    "triple_barrier_return",
    "tb_return",
    "realised_return",
    "realized_return",
    "pnl",
    "profit",
    "barrier_touched",
    "first_barrier_touched",
    "hit_barrier",
    "pt",
    "sl",
    "tp",
    "pt_mult",
    "sl_mult",
    "take_profit_mult",
    "stop_loss_mult",
    "vol",
    "volatility",
    "target_vol",
    "volatility_method",
    "ewma_span",
    "volatility_window",
    "close_tb",
}

RF_CONFIG_DEFAULTS = {
    "rf_baseline": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "rf_shallow": {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_leaf": 10,
        "min_samples_split": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "rf_medium_depth": {
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_leaf": 8,
        "min_samples_split": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "rf_conservative": {
        "n_estimators": 500,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "min_samples_split": 40,
        "max_features": 0.5,
        "class_weight": "balanced",
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "features").exists() and (candidate / "03_model_development").exists():
            return candidate
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
FEATURE_SELECTION_DIR = PROJECT_ROOT / "03_model_development" / "Feature Selection"
if str(FEATURE_SELECTION_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_SELECTION_DIR))

from feature_selection_methods import CorrelationClusterSelector, PCAFeatureReducer  # noqa: E402


def parse_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, str):
        value = value.strip()
        if value in {"", "nan", "None"}:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return {}


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def read_model_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the logistic/RF model notebooks first."
        )

    df = pd.read_csv(path)
    required = {
        "model_type",
        "ticker",
        "tb_config_name",
        "model_name",
        "feature_method",
        "mean_auc",
        "std_auc",
        "median_path_trade_sharpe",
        "path_sharpe_iqr",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"model_comparison.csv is missing columns: {sorted(missing)}")
    return df[df["ticker"].isin(ENERGY_TICKERS)].copy()


def select_top3_auc_then_sharpe(comparison_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    top3_rows = []
    selected_rows = []

    for ticker, group in comparison_df.groupby("ticker", sort=True):
        top3 = (
            group.sort_values(
                ["mean_auc", "std_auc", "median_auc"],
                ascending=[False, True, False],
            )
            .head(3)
            .copy()
            .reset_index(drop=True)
        )
        top3["auc_rank_within_ticker"] = np.arange(1, len(top3) + 1)
        top3_rows.append(top3)

        selected = (
            top3.sort_values(
                ["median_path_trade_sharpe", "path_sharpe_iqr", "mean_auc", "std_auc"],
                ascending=[False, True, False, True],
            )
            .head(1)
            .copy()
        )
        selected["selection_rule"] = "top3_mean_auc_then_highest_median_path_trade_sharpe"
        selected_rows.append(selected)

    return (
        pd.concat(top3_rows, ignore_index=True),
        pd.concat(selected_rows, ignore_index=True),
    )


def load_clean_features(ticker: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "features" / "clean_feature_set" / f"{ticker}_clean_feature_set.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    if INSTRUMENT_COL in df.columns:
        df[INSTRUMENT_COL] = df[INSTRUMENT_COL].str.lower()
        df = df[df[INSTRUMENT_COL] == ticker].copy()
    return df.sort_values(DATE_COL).reset_index(drop=True)


def load_labels(ticker: str, tb_config_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "features" / "triple_barrier" / f"{ticker}_{tb_config_name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns
    date_cols = [col for col in [DATE_COL, "training_end", "timeout_date", "touch_date"] if col in header]
    df = pd.read_csv(path, parse_dates=date_cols)
    if INSTRUMENT_COL in df.columns:
        df[INSTRUMENT_COL] = df[INSTRUMENT_COL].str.lower()
        df = df[df[INSTRUMENT_COL] == ticker].copy()
    return df.sort_values(DATE_COL).reset_index(drop=True)


def build_training_data(ticker: str, tb_config_name: str) -> tuple[pd.DataFrame, list[str]]:
    features = load_clean_features(ticker)
    labels = load_labels(ticker, tb_config_name)
    labels = labels.drop(columns=["primary_signal"], errors="ignore")

    merge_keys = [DATE_COL]
    if INSTRUMENT_COL in features.columns and INSTRUMENT_COL in labels.columns:
        merge_keys.append(INSTRUMENT_COL)

    data = features.merge(labels, on=merge_keys, how="inner", suffixes=("", "_tb"))
    data = data.sort_values(DATE_COL).reset_index(drop=True)

    feature_cols = [
        col for col in data.columns
        if col not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(data[col])
    ]
    if "primary_signal" not in feature_cols:
        raise ValueError(f"primary_signal missing from feature columns for {ticker}/{tb_config_name}.")

    required_cols = feature_cols + [TARGET_COL, DATE_COL]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()
    if TRAIN_ON_NONZERO_SIGNALS_ONLY and "primary_signal" in data.columns:
        data = data[data["primary_signal"] != 0].copy()

    data[TARGET_COL] = data[TARGET_COL].astype(int)
    train_df = data[data[DATE_COL] < pd.Timestamp(TEST_START_DATE)].copy()
    if train_df[TARGET_COL].nunique() < 2:
        raise ValueError(f"Training rows need both classes for {ticker}/{tb_config_name}.")

    return train_df.reset_index(drop=True), feature_cols


def split_protected_and_reducible(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    protected = [col for col in PROTECTED_FEATURES if col in feature_cols]
    reducible = [col for col in feature_cols if col not in protected]
    return protected, reducible


def apply_feature_processing(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    y_train: pd.Series,
    feature_method: str,
    feature_params: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    protected_cols, reducible_cols = split_protected_and_reducible(feature_cols)
    X_protected = train_df[protected_cols].reset_index(drop=True) if protected_cols else pd.DataFrame(index=range(len(train_df)))

    if feature_method == "none":
        return train_df[feature_cols].reset_index(drop=True).copy(), feature_cols.copy(), {
            "feature_method": feature_method,
            "processor": None,
        }

    if feature_method == "corr_cluster":
        selector = CorrelationClusterSelector(**feature_params)
        selector.fit(train_df[reducible_cols], y=y_train, feature_columns=reducible_cols)
        X_selected = selector.transform(train_df[reducible_cols]).reset_index(drop=True)
        processed_cols = protected_cols + selector.selected_features_
        return pd.concat([X_protected, X_selected], axis=1), processed_cols, {
            "feature_method": feature_method,
            "processor": selector,
        }

    if feature_method == "pca":
        # PCA is numerically sensitive on heavy-tailed futures features. Fit all
        # preprocessing on the selected model's training rows only, then use the
        # fitted PCA components as model inputs.
        X_reducible = train_df[reducible_cols].replace([np.inf, -np.inf], np.nan)
        lower = X_reducible.quantile(FEATURE_CLIP_LOWER)
        upper = X_reducible.quantile(FEATURE_CLIP_UPPER)
        X_reducible = X_reducible.clip(lower=lower, upper=upper, axis=1).clip(-FEATURE_ABS_CAP, FEATURE_ABS_CAP)

        train_std = X_reducible.std(skipna=True)
        pca_cols = train_std[train_std > 1e-8].index.tolist()
        if not pca_cols:
            raise ValueError("No usable non-constant features left for PCA.")

        X_reducible = X_reducible[pca_cols]
        pca_imputer = SimpleImputer(strategy=feature_params.get("impute_strategy", "median"))
        X_imp = pca_imputer.fit_transform(X_reducible)

        pca_scaler = None
        if feature_params.get("standardize", True):
            pca_scaler = StandardScaler()
            X_for_pca = pca_scaler.fit_transform(X_imp)
        else:
            X_for_pca = X_imp

        pca = PCA(
            n_components=feature_params.get("n_components", 0.95),
            svd_solver=feature_params.get("svd_solver", "full"),
            random_state=feature_params.get("random_state", RANDOM_STATE),
        )
        X_components = pca.fit_transform(X_for_pca)
        if not np.isfinite(X_components).all():
            raise ValueError("PCA produced non-finite component values.")

        component_prefix = feature_params.get("component_prefix", "pca")
        component_cols = [f"{component_prefix}_{i:03d}" for i in range(1, X_components.shape[1] + 1)]
        X_pca = pd.DataFrame(X_components, columns=component_cols)
        processed_cols = protected_cols + component_cols
        return pd.concat([X_protected, X_pca], axis=1), processed_cols, {
            "feature_method": feature_method,
            "processor": {
                "pca": pca,
                "imputer": pca_imputer,
                "scaler": pca_scaler,
                "feature_columns": pca_cols,
                "component_columns": component_cols,
            },
        }

    raise ValueError(f"Unknown feature method: {feature_method}")


def make_logistic_model(row: pd.Series) -> LogisticRegression:
    penalty = row["penalty"]
    params = {
        "penalty": penalty,
        "C": float(row["C"]),
        "class_weight": clean_scalar(row.get("class_weight", "balanced")),
        "max_iter": 5000,
    }
    if penalty == "l2":
        params["solver"] = "lbfgs"
    elif penalty == "l1":
        params["solver"] = "saga"
        params["random_state"] = RANDOM_STATE
    elif penalty == "elasticnet":
        params["solver"] = "saga"
        params["l1_ratio"] = float(row["l1_ratio"])
        params["random_state"] = RANDOM_STATE
    else:
        raise ValueError(f"Unsupported logistic penalty: {penalty}")
    return LogisticRegression(**params)


def make_rf_model(row: pd.Series) -> RandomForestClassifier:
    params = dict(RF_CONFIG_DEFAULTS.get(row["model_name"], {}))
    params.update(
        {
            "n_estimators": int(row["n_estimators"]),
            "max_depth": None if pd.isna(row["max_depth"]) else int(row["max_depth"]),
            "min_samples_leaf": int(row["min_samples_leaf"]),
            "max_features": row["max_features"],
            "class_weight": clean_scalar(row.get("class_weight", "balanced")),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    )
    return RandomForestClassifier(**params)


def fit_selected_model(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ticker = row["ticker"]
    tb_config_name = row["tb_config_name"]
    model_type = row["model_type"]
    feature_method = row["feature_method"]
    feature_params = parse_params(row.get("feature_params", {}))

    train_df, feature_cols = build_training_data(ticker, tb_config_name)
    y_train = train_df[TARGET_COL].astype(int)
    X_processed, processed_cols, processing_info = apply_feature_processing(
        train_df=train_df,
        feature_cols=feature_cols,
        y_train=y_train,
        feature_method=feature_method,
        feature_params=feature_params,
    )

    if model_type == "logistic_regression":
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_imp = imputer.fit_transform(X_processed)
        X_fit = scaler.fit_transform(X_imp)
        model = make_logistic_model(row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(X_fit, y_train)
        importance_df = pd.DataFrame(
            {
                "feature": processed_cols,
                "coefficient": model.coef_.ravel(),
            }
        )
        importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
        importance_df["importance"] = importance_df["abs_coefficient"]
        importance_df["importance_type"] = "abs_standardized_logistic_coefficient"
        fit_metadata = {"model": model, "imputer": imputer, "scaler": scaler}
    elif model_type == "random_forest":
        model = make_rf_model(row)
        model.fit(X_processed, y_train)
        importance_df = pd.DataFrame(
            {
                "feature": processed_cols,
                "importance": model.feature_importances_,
            }
        )
        importance_df["importance_type"] = "random_forest_mdi"
        fit_metadata = {"model": model}
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    for col in ["model_type", "ticker", "tb_config_name", "model_name", "feature_method"]:
        importance_df.insert(0, col, row[col])

    fit_metadata.update(
        {
            "train_rows": len(train_df),
            "original_feature_count": len(feature_cols),
            "processed_feature_count": len(processed_cols),
            "processed_cols": processed_cols,
            "processing_info": processing_info,
        }
    )
    return importance_df, build_cluster_importance(importance_df, processing_info), fit_metadata


def build_cluster_importance(feature_importance_df: pd.DataFrame, processing_info: dict[str, Any]) -> pd.DataFrame:
    method = processing_info.get("feature_method", "none")
    importance_df = feature_importance_df.copy()
    importance_df["importance_group"] = importance_df["feature"]
    importance_df["group_type"] = "single_feature"
    importance_df["cluster_id"] = np.nan
    importance_df["cluster_n_features"] = 1
    importance_df["cluster_features"] = importance_df["feature"].apply(lambda x: [x])
    importance_df["dropped_cluster_features"] = [[] for _ in range(len(importance_df))]

    if method == "corr_cluster":
        selector = processing_info.get("processor")
        if selector is not None and hasattr(selector, "cluster_summary_"):
            cluster_summary = selector.cluster_summary_.copy()
            if not cluster_summary.empty:
                rep_to_group = {}
                rep_to_cluster_id = {}
                rep_to_n_features = {}
                rep_to_cluster_features = {}
                rep_to_dropped_features = {}
                for row in cluster_summary.itertuples(index=False):
                    cluster_id = int(row.cluster_id)
                    representative = row.representative_feature
                    cluster_features = list(row.cluster_features)
                    dropped_features = list(row.dropped_features)
                    rep_to_group[representative] = f"cluster_{cluster_id:03d}: {representative} (rep, n={len(cluster_features)})"
                    rep_to_cluster_id[representative] = cluster_id
                    rep_to_n_features[representative] = len(cluster_features)
                    rep_to_cluster_features[representative] = cluster_features
                    rep_to_dropped_features[representative] = dropped_features
                is_cluster_rep = importance_df["feature"].isin(rep_to_group.keys())
                importance_df.loc[is_cluster_rep, "importance_group"] = importance_df.loc[is_cluster_rep, "feature"].map(rep_to_group)
                importance_df.loc[is_cluster_rep, "group_type"] = "correlation_cluster"
                importance_df.loc[is_cluster_rep, "cluster_id"] = importance_df.loc[is_cluster_rep, "feature"].map(rep_to_cluster_id)
                importance_df.loc[is_cluster_rep, "cluster_n_features"] = importance_df.loc[is_cluster_rep, "feature"].map(rep_to_n_features)
                importance_df.loc[is_cluster_rep, "cluster_features"] = importance_df.loc[is_cluster_rep, "feature"].map(rep_to_cluster_features)
                importance_df.loc[is_cluster_rep, "dropped_cluster_features"] = importance_df.loc[is_cluster_rep, "feature"].map(rep_to_dropped_features)
    elif method == "pca":
        importance_df["group_type"] = np.where(
            importance_df["feature"].str.startswith("pca_"),
            "pca_component",
            "single_feature",
        )

    cluster_df = (
        importance_df
        .groupby(
            [
                "model_type",
                "ticker",
                "tb_config_name",
                "model_name",
                "feature_method",
                "importance_group",
            ],
            as_index=False,
        )
        .agg(
            cluster_importance=("importance", "sum"),
            group_type=("group_type", "first"),
            representative_model_features=("feature", lambda x: list(x)),
            n_model_features=("feature", "count"),
            cluster_id=("cluster_id", "first"),
            cluster_n_features=("cluster_n_features", "first"),
            cluster_features=("cluster_features", "first"),
            dropped_cluster_features=("dropped_cluster_features", "first"),
            importance_type=("importance_type", "first"),
        )
        .sort_values(["ticker", "cluster_importance"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return cluster_df


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    comparison_path = results_dir / "model_comparison.csv"
    top3_path = results_dir / "feature_importance_top3_by_auc.csv"
    selected_path = results_dir / "feature_importance_selected_models.csv"
    feature_path = results_dir / "feature_importance_feature_level.csv"
    cluster_path = results_dir / "feature_importance_cluster_level.csv"

    comparison_df = read_model_comparison(comparison_path)
    top3_df, selected_df = select_top3_auc_then_sharpe(comparison_df)

    feature_tables = []
    cluster_tables = []
    fit_rows = []

    for row in selected_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        feature_df, cluster_df, metadata = fit_selected_model(row_series)
        feature_tables.append(feature_df)
        cluster_tables.append(cluster_df)
        fit_rows.append(
            {
                "ticker": row_series["ticker"],
                "model_type": row_series["model_type"],
                "tb_config_name": row_series["tb_config_name"],
                "model_name": row_series["model_name"],
                "feature_method": row_series["feature_method"],
                "train_rows": metadata["train_rows"],
                "original_feature_count": metadata["original_feature_count"],
                "processed_feature_count": metadata["processed_feature_count"],
            }
        )

    top3_df.to_csv(top3_path, index=False)
    selected_df.merge(pd.DataFrame(fit_rows), on=["ticker", "model_type", "tb_config_name", "model_name", "feature_method"], how="left").to_csv(selected_path, index=False)
    pd.concat(feature_tables, ignore_index=True).to_csv(feature_path, index=False)
    pd.concat(cluster_tables, ignore_index=True).to_csv(cluster_path, index=False)

    print("Saved:")
    print(top3_path)
    print(selected_path)
    print(feature_path)
    print(cluster_path)


if __name__ == "__main__":
    main()
