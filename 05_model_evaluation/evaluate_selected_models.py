"""Evaluate selected meta-models on the 2022+ test window.

The script reads the selected model run configurations produced by
04_feature_importance/cluster_feature_importance.py, retrains each selected
model on pre-test data, scores the 2022+ labelled events, and writes the final
deliverable CSV:

    date,instrument,prediction
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="Pandas requires version '2.10.2' or newer of 'numexpr'.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "results" / "selected_model_run_configs.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / "04_feature_importance"
if str(FEATURE_IMPORTANCE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_IMPORTANCE_DIR))

import cluster_feature_importance as cfi  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "results" / "selected_model_run_configs.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "deliverables" / "final_predictions.csv"


def parse_json(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        return json.loads(value)
    return value


def build_test_data(
    ticker: str,
    tb_config_name: str,
    feature_cols: list[str],
    test_start_date: str,
    train_on_nonzero_signals_only: bool,
) -> pd.DataFrame:
    features = cfi.load_clean_features(ticker)
    labels = cfi.load_labels(ticker, tb_config_name)
    labels = labels.drop(columns=["primary_signal"], errors="ignore")

    merge_keys = [cfi.DATE_COL]
    if cfi.INSTRUMENT_COL in features.columns and cfi.INSTRUMENT_COL in labels.columns:
        merge_keys.append(cfi.INSTRUMENT_COL)

    data = features.merge(labels, on=merge_keys, how="inner", suffixes=("", "_tb"))
    data = data.sort_values(cfi.DATE_COL).reset_index(drop=True)
    data = data[data[cfi.DATE_COL] >= pd.Timestamp(test_start_date)].copy()

    if train_on_nonzero_signals_only and "primary_signal" in data.columns:
        data = data[data["primary_signal"] != 0].copy()

    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"{ticker}/{tb_config_name} is missing feature columns: {missing_cols}")

    required_cols = feature_cols + [cfi.DATE_COL, cfi.INSTRUMENT_COL]
    before_drop = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()
    dropped = before_drop - len(data)
    if dropped:
        print(f"Warning: dropped {dropped} incomplete test rows for {ticker}/{tb_config_name}.")

    if data.empty:
        raise ValueError(f"No test rows available for {ticker}/{tb_config_name}.")

    return data.reset_index(drop=True)


def transform_features(
    data: pd.DataFrame,
    feature_cols: list[str],
    processing_info: dict[str, Any],
) -> pd.DataFrame:
    method = processing_info.get("feature_method", "none")
    protected_cols, reducible_cols = cfi.split_protected_and_reducible(feature_cols)
    X_protected = (
        data[protected_cols].reset_index(drop=True)
        if protected_cols
        else pd.DataFrame(index=range(len(data)))
    )

    if method == "none":
        return data[feature_cols].reset_index(drop=True).copy()

    if method == "corr_cluster":
        selector = processing_info["processor"]
        X_selected = selector.transform(data[reducible_cols]).reset_index(drop=True)
        return pd.concat([X_protected, X_selected], axis=1)

    if method == "pca":
        processor = processing_info["processor"]
        pca_cols = processor["feature_columns"]
        X_reducible = data[pca_cols].replace([np.inf, -np.inf], np.nan)
        X_reducible = (
            X_reducible
            .clip(lower=processor["clip_lower"], upper=processor["clip_upper"], axis=1)
            .clip(-cfi.FEATURE_ABS_CAP, cfi.FEATURE_ABS_CAP)
        )
        X_imp = processor["imputer"].transform(X_reducible)
        scaler = processor["scaler"]
        X_for_pca = scaler.transform(X_imp) if scaler is not None else X_imp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X_components = processor["pca"].transform(X_for_pca)
        X_pca = pd.DataFrame(X_components, columns=processor["component_columns"])
        return pd.concat([X_protected, X_pca], axis=1)

    raise ValueError(f"Unknown feature method: {method}")


def make_model(model_type: str, model_params: dict[str, Any]) -> LogisticRegression | RandomForestClassifier:
    if model_type == "logistic_regression":
        return LogisticRegression(**model_params)
    if model_type == "random_forest":
        return RandomForestClassifier(**model_params)
    raise ValueError(f"Unsupported model_type for evaluation: {model_type}")


def fit_predict_detailed_config(config: pd.Series) -> pd.DataFrame:
    ticker = config["ticker"]
    tb_config_name = config["tb_config_name"]
    feature_cols = parse_json(config["original_feature_columns"], default=[])
    feature_params = parse_json(config["feature_params"], default={})
    model_params = parse_json(config["model_params"], default={})
    test_start_date = config.get("test_start_date", cfi.TEST_START_DATE)
    train_on_nonzero = bool(config.get("train_on_nonzero_signals_only", True))

    train_df, discovered_feature_cols = cfi.build_training_data(ticker, tb_config_name)
    if not feature_cols:
        feature_cols = discovered_feature_cols

    missing_train_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_train_cols:
        raise ValueError(f"{ticker}/{tb_config_name} training data is missing: {missing_train_cols}")

    y_train = train_df[cfi.TARGET_COL].astype(int)
    X_train, _, processing_info = cfi.apply_feature_processing(
        train_df=train_df,
        feature_cols=feature_cols,
        y_train=y_train,
        feature_method=config["feature_method"],
        feature_params=feature_params,
    )

    test_df = build_test_data(
        ticker=ticker,
        tb_config_name=tb_config_name,
        feature_cols=feature_cols,
        test_start_date=test_start_date,
        train_on_nonzero_signals_only=train_on_nonzero,
    )
    X_test = transform_features(test_df, feature_cols, processing_info)

    model = make_model(config["model_type"], model_params)
    if config["model_type"] == "logistic_regression":
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_fit = scaler.transform(imputer.transform(X_test))
    else:
        X_train_fit = X_train
        X_test_fit = X_test

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_train_fit, y_train)
        y_proba = model.predict_proba(X_test_fit)[:, 1]

    detailed = pd.DataFrame(
        {
            "date": test_df[cfi.DATE_COL].dt.strftime("%Y-%m-%d"),
            "instrument": test_df[cfi.INSTRUMENT_COL].str.lower(),
            "prediction": y_proba,
            "predicted_label": (y_proba >= 0.5).astype(int),
            "model_type": config["model_type"],
            "model_name": config["model_name"],
            "tb_config_name": tb_config_name,
            "feature_method": config["feature_method"],
        }
    )
    if cfi.TARGET_COL in test_df.columns:
        detailed["y_true"] = test_df[cfi.TARGET_COL].astype(int).to_numpy()
    if "primary_signal" in test_df.columns:
        detailed["primary_signal"] = test_df["primary_signal"].to_numpy()
    if "signed_touch_return" in test_df.columns:
        detailed["signed_touch_return"] = test_df["signed_touch_return"].to_numpy()
    return detailed


def fit_and_predict_config(config: pd.Series) -> pd.DataFrame:
    detailed = fit_predict_detailed_config(config)
    return detailed[["date", "instrument", "prediction"]].copy()


def run_evaluation(config_path: Path, output_path: Path) -> pd.DataFrame:
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    configs = pd.read_csv(config_path)
    if configs.empty:
        raise ValueError(f"No selected model configs found in {config_path}.")

    prediction_tables = []
    for config in configs.itertuples(index=False):
        config_series = pd.Series(config._asdict())
        predictions = fit_and_predict_config(config_series)
        prediction_tables.append(predictions)
        print(
            f"{config_series['ticker']}: wrote {len(predictions)} predictions "
            f"from {predictions['date'].min()} to {predictions['date'].max()}"
        )

    final_predictions = (
        pd.concat(prediction_tables, ignore_index=True)
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    final_predictions = final_predictions[["date", "instrument", "prediction"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_predictions.to_csv(output_path, index=False)
    return final_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to selected_model_run_configs.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the final date,instrument,prediction deliverable CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_predictions = run_evaluation(args.config_path, args.output_path)
    print(f"Saved {len(final_predictions)} final predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
