"""CPCV model comparison using precomputed triple-barrier labels.

This script is an isolated version of `03_model_development/cpcv_energy_modelling.ipynb`.
It uses the cached label files listed in:

    kaushaltest/data/labels/triple_barrier/manifest_reduced.csv

Instead of recalculating triple-barrier labels inside every candidate run, it
loads each cached ticker/barrier CSV once and reuses it across feature-selection
and model combinations.

Date policy:
- training labels and features are restricted to dates before 2022-01-01
- Jan-Jun 2022 is not used here; that period is for final predictions/evaluation
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from itertools import combinations, product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
TRAINING_END = pd.Timestamp("2022-01-01")
RANDOM_STATE = 42
PROBA_THRESHOLD = 0.50
TOP_K = 5
MIN_VALID_AUC_PATHS = 3

MANIFEST_PATH = EXPERIMENT_ROOT / "data" / "labels" / "triple_barrier" / "manifest_reduced.csv"
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "features" / "merged_feature_matrix.csv"
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs" / "enhancedcombocpcv"

ENERGY_TICKERS = ["cl1s", "ho1s", "rb1s", "ng1s"]

CPCV_SETTINGS = {
    "cl1s": {"n_groups": 6, "k_test_groups": 2, "embargo": 5},
    "rb1s": {"n_groups": 6, "k_test_groups": 2, "embargo": 5},
    "ho1s": {"n_groups": 4, "k_test_groups": 1, "embargo": 5},
    "ng1s": {"n_groups": 4, "k_test_groups": 1, "embargo": 5},
}

FEATURE_SELECTION_CONFIGS = [
    {"name": "none", "method": "none"},
    {"name": "mdi_top20", "method": "mdi", "k": 20},
    {"name": "cluster_corr90", "method": "cluster", "corr_threshold": 0.90, "max_features": 50},
    {"name": "pca90", "method": "pca", "variance": 0.90},
]

MODEL_CONFIGS = [
    {
        "name": "logistic",
        "model_type": "logistic",
        "needs_scaling": True,
        "params": {"max_iter": 1000, "class_weight": "balanced"},
    },
    {
        "name": "random_forest",
        "model_type": "random_forest",
        "needs_scaling": False,
        "params": {
            "n_estimators": 100,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
    },
    {
        "name": "extra_trees",
        "model_type": "extra_trees",
        "needs_scaling": False,
        "params": {
            "n_estimators": 150,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    },
    {
        "name": "mlp",
        "model_type": "mlp",
        "needs_scaling": True,
        "params": {"hidden_layer_sizes": (32,), "max_iter": 500, "early_stopping": True},
    },
]

LEAKAGE_COLUMNS = {
    "training_end",
    "vol",
    "tp",
    "sl",
    "timeout_date",
    "timeout_close",
    "touch_date",
    "touch_price",
    "touched_barrier",
    "triple_barrier_label",
    "metalabel",
    "holding_period_days",
    "raw_touch_return",
    "signed_touch_return",
    "volatility_method",
    "ewma_span",
    "volatility_window",
    "num_days",
    "take_profit_mult",
    "stop_loss_mult",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPCV using cached triple-barrier labels.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tickers", nargs="+", default=ENERGY_TICKERS)
    parser.add_argument("--models", nargs="+", default=[c["name"] for c in MODEL_CONFIGS])
    parser.add_argument("--feature-selectors", nargs="+", default=[c["name"] for c in FEATURE_SELECTION_CONFIGS])
    parser.add_argument("--max-candidates", type=int, default=None, help="Debug limit after filtering.")
    parser.add_argument("--dry-run", action="store_true", help="Print candidate count without fitting models.")
    parser.add_argument("--no-save-models", action="store_true", help="Skip final joblib model files.")
    return parser.parse_args()


def resolve_manifest_label_path(path_value: str) -> Path:
    path = Path(path_value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(PROJECT_ROOT / path)
        parts = path.parts
        if parts and parts[0] in {"kaushal_label_generator", "kaushaltest"}:
            candidates.append(EXPERIMENT_ROOT / Path(*parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve label path {path_value!r}. Checked: {checked}")


def load_manifest(manifest_path: Path, tickers: list[str]) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest["ticker"] = manifest["ticker"].str.lower()
    manifest = manifest[manifest["ticker"].isin(tickers)].copy()
    manifest["training_end"] = pd.to_datetime(manifest["training_end"])
    manifest = manifest[manifest["training_end"].eq(TRAINING_END)].copy()
    manifest["label_path"] = manifest["path"].map(resolve_manifest_label_path)
    if manifest.empty:
        raise ValueError("No manifest rows remain after ticker/date filtering.")
    return manifest.sort_values(["ticker", "barrier"]).reset_index(drop=True)


def load_features(tickers: list[str]) -> pd.DataFrame:
    features = pd.read_csv(FEATURE_MATRIX_PATH, parse_dates=["date"])
    features["instrument"] = features["instrument"].str.lower()
    features = features[features["instrument"].isin(tickers)].copy()
    features = features[features["date"] < TRAINING_END].copy()
    return features.sort_values(["instrument", "date"]).reset_index(drop=True)


def make_model(model_config: dict):
    params = dict(model_config.get("params", {}))
    params.setdefault("random_state", RANDOM_STATE)
    model_type = model_config["model_type"]

    if model_type == "logistic":
        return LogisticRegression(**params)
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    if model_type == "extra_trees":
        return ExtraTreesClassifier(**params)
    if model_type == "mlp":
        return MLPClassifier(**params)
    raise ValueError(f"Unknown model_type: {model_type}")


def make_groups(n_rows: int, n_groups: int) -> list[np.ndarray]:
    return [g.astype(int) for g in np.array_split(np.arange(n_rows), n_groups) if len(g) > 0]


def embargo_date(dates: pd.Series, end_date: pd.Timestamp, embargo: int) -> pd.Timestamp:
    unique_dates = pd.Series(pd.to_datetime(dates).sort_values().unique())
    pos = unique_dates.searchsorted(end_date, side="right")
    if pos >= len(unique_dates):
        return end_date
    return unique_dates.iloc[min(len(unique_dates) - 1, pos + embargo - 1)]


def make_cpcv_splits(data: pd.DataFrame, ticker: str) -> list[dict]:
    settings = CPCV_SETTINGS[ticker]
    groups = make_groups(len(data), settings["n_groups"])
    splits = []

    for split_id, test_group_ids in enumerate(combinations(range(len(groups)), settings["k_test_groups"])):
        test_idx = np.concatenate([groups[i] for i in test_group_ids])
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[test_idx] = False

        for group_id in test_group_ids:
            group_idx = groups[group_id]
            test_start = data.loc[group_idx, "date"].min()
            test_end = data.loc[group_idx, "date"].max()
            test_horizon_end = data.loc[group_idx, "timeout_date"].max()
            emb_end = embargo_date(data["date"], test_end, settings["embargo"])

            overlapping = (data["date"] <= test_horizon_end) & (data["timeout_date"] >= test_start)
            embargoed = (data["date"] > test_end) & (data["date"] <= emb_end)
            train_mask[overlapping | embargoed] = False

        train_idx = np.where(train_mask)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append({"split_id": split_id, "train_idx": train_idx, "test_idx": test_idx})
    return splits


def select_none(X_train, X_test, y_train, fs_config):
    return X_train.values, X_test.values, list(X_train.columns), None


def select_mdi(X_train, X_test, y_train, fs_config):
    k = min(fs_config.get("k", 20), X_train.shape[1])
    tree = ExtraTreesClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    tree.fit(X_train, y_train)
    importance = pd.Series(tree.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    selected = importance.head(k).index.tolist()
    report = importance.head(k).reset_index().rename(columns={"index": "feature", 0: "importance"})
    return X_train[selected].values, X_test[selected].values, selected, report


def select_cluster(X_train, X_test, y_train, fs_config):
    corr_threshold = fs_config.get("corr_threshold", 0.90)
    max_features = fs_config.get("max_features", 50)
    corr = X_train.corr().abs().fillna(0).clip(0, 1)
    distance = (1 - corr).clip(0, 1)
    np.fill_diagonal(distance.values, 0)

    linkage = hierarchy.linkage(squareform(distance.values, checks=False), method="average")
    clusters = hierarchy.fcluster(linkage, t=1 - corr_threshold, criterion="distance")
    mi = pd.Series(mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE), index=X_train.columns)

    rows = []
    for cluster_id in sorted(np.unique(clusters)):
        members = [X_train.columns[i] for i, c in enumerate(clusters) if c == cluster_id]
        best = mi.loc[members].sort_values(ascending=False).index[0]
        rows.append({"feature": best, "cluster": cluster_id, "score": mi.loc[best], "cluster_size": len(members)})

    report = pd.DataFrame(rows).sort_values("score", ascending=False).head(max_features)
    selected = report["feature"].tolist()
    return X_train[selected].values, X_test[selected].values, selected, report


def select_pca(X_train, X_test, y_train, fs_config):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=fs_config.get("variance", 0.90), random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    names = [f"PC{i + 1}" for i in range(X_train_pca.shape[1])]
    report = pd.DataFrame(
        {
            "component": names,
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return X_train_pca, X_test_pca, names, report


def apply_feature_selection(X_train, X_test, y_train, fs_config):
    method = fs_config["method"]
    if method == "none":
        return select_none(X_train, X_test, y_train, fs_config)
    if method == "mdi":
        return select_mdi(X_train, X_test, y_train, fs_config)
    if method == "cluster":
        return select_cluster(X_train, X_test, y_train, fs_config)
    if method == "pca":
        return select_pca(X_train, X_test, y_train, fs_config)
    raise ValueError(f"Unknown feature-selection method: {method}")


def prediction_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def sharpe_ratio(returns) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return np.nan
    return returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns))


class CachedCPCVRunner:
    def __init__(self, manifest: pd.DataFrame, features: pd.DataFrame):
        self.manifest = manifest
        self.features = features
        self.label_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def load_labels(self, ticker: str, barrier: str) -> pd.DataFrame:
        key = (ticker, barrier)
        if key not in self.label_cache:
            row = self.manifest[(self.manifest["ticker"] == ticker) & (self.manifest["barrier"] == barrier)]
            if row.empty:
                raise KeyError(f"No cached labels for {ticker} {barrier}")
            labels = pd.read_csv(
                row.iloc[0]["label_path"],
                parse_dates=["training_end", "date", "timeout_date", "touch_date"],
            )
            labels["instrument"] = labels["instrument"].str.lower()
            labels = labels[labels["date"] < TRAINING_END].copy()
            if not labels.empty and labels["date"].max() >= TRAINING_END:
                raise ValueError(f"Date leakage in cached labels for {ticker} {barrier}")
            self.label_cache[key] = labels
        return self.label_cache[key].copy()

    def make_model_data(self, ticker: str, barrier: str) -> tuple[pd.DataFrame, list[str]]:
        labels = self.load_labels(ticker, barrier)
        ticker_features = self.features[self.features["instrument"] == ticker].copy()

        labels_for_join = labels.drop(columns=["close"], errors="ignore")
        ticker_features = ticker_features.drop(columns=["primary_signal"], errors="ignore")
        data = labels_for_join.merge(ticker_features, on=["date", "instrument"], how="inner")
        data = data.sort_values("date").reset_index(drop=True)

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in LEAKAGE_COLUMNS]
        data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)
        return data, feature_cols

    def evaluate_candidate(self, ticker: str, barrier: str, fs_config: dict, model_config: dict):
        data, feature_cols = self.make_model_data(ticker, barrier)
        splits = make_cpcv_splits(data, ticker)
        rows = []
        first_report = None

        for split in splits:
            train = data.iloc[split["train_idx"]]
            test = data.iloc[split["test_idx"]]
            y_train = train["metalabel"].astype(int)
            y_test = test["metalabel"].astype(int)
            if y_train.nunique() < 2:
                continue

            X_train_raw = train[feature_cols].replace([np.inf, -np.inf], np.nan)
            X_test_raw = test[feature_cols].replace([np.inf, -np.inf], np.nan)

            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_cols, index=train.index)
            X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=feature_cols, index=test.index)

            X_train_selected, X_test_selected, selected_features, fs_report = apply_feature_selection(
                X_train, X_test, y_train, fs_config
            )

            if model_config["needs_scaling"]:
                scaler = StandardScaler()
                X_train_model = scaler.fit_transform(X_train_selected)
                X_test_model = scaler.transform(X_test_selected)
            else:
                X_train_model = X_train_selected
                X_test_model = X_test_selected

            model = make_model(model_config)
            model.fit(X_train_model, y_train)
            y_score = prediction_scores(model, X_test_model)
            y_pred = (y_score >= PROBA_THRESHOLD).astype(int)
            auc = roc_auc_score(y_test, y_score) if y_test.nunique() == 2 else np.nan
            trade_returns = test.loc[y_pred == 1, "signed_touch_return"]

            rows.append(
                {
                    "ticker": ticker,
                    "barrier": barrier,
                    "feature_selection": fs_config["name"],
                    "model": model_config["name"],
                    "split_id": split["split_id"],
                    "auc": auc,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "sharpe": sharpe_ratio(trade_returns),
                    "trade_count": int((y_pred == 1).sum()),
                    "selected_feature_count": len(selected_features),
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )

            if first_report is None:
                first_report = {
                    "ticker": ticker,
                    "barrier": barrier,
                    "feature_selection": fs_config["name"],
                    "model": model_config["name"],
                    "selected_features": selected_features,
                    "report": fs_report,
                }

        return rows, first_report

    def fit_final_model(self, selected: dict, feature_selection_configs: list[dict], model_configs: list[dict]) -> dict:
        ticker = selected["ticker"]
        barrier = selected["barrier"]
        fs_config = get_config_by_name(feature_selection_configs, selected["feature_selection"])
        model_config = get_config_by_name(model_configs, selected["model"])

        data, feature_cols = self.make_model_data(ticker, barrier)
        y = data["metalabel"].astype(int)
        X_raw = data[feature_cols].replace([np.inf, -np.inf], np.nan)

        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_cols, index=data.index)
        X_selected, _, selected_features, fs_report = apply_feature_selection(X, X, y, fs_config)

        scaler = None
        if model_config["needs_scaling"]:
            scaler = StandardScaler()
            X_model = scaler.fit_transform(X_selected)
        else:
            X_model = X_selected

        model = make_model(model_config)
        model.fit(X_model, y)
        return {
            "ticker": ticker,
            "barrier": barrier,
            "feature_selection_config": fs_config,
            "model_config": model_config,
            "imputer": imputer,
            "scaler": scaler,
            "selected_features": selected_features,
            "feature_selection_report": fs_report,
            "model": model,
            "training_rows": len(data),
            "feature_cols_before_selection": feature_cols,
        }


def get_config_by_name(configs: list[dict], name: str) -> dict:
    for config in configs:
        if config["name"] == name:
            return config
    raise KeyError(name)


def summarize_candidates(all_cpcv_results: pd.DataFrame) -> pd.DataFrame:
    candidate_summary = (
        all_cpcv_results.groupby(["ticker", "barrier", "feature_selection", "model"], as_index=False)
        .agg(
            mean_auc=("auc", "mean"),
            median_auc=("auc", "median"),
            valid_auc_paths=("auc", lambda x: x.notna().sum()),
            median_sharpe=("sharpe", "median"),
            mean_sharpe=("sharpe", "mean"),
            sharpe_iqr=("sharpe", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            sharpe_std=("sharpe", "std"),
            total_trades=("trade_count", "sum"),
            avg_selected_features=("selected_feature_count", "mean"),
            avg_train_rows=("train_rows", "mean"),
            avg_test_rows=("test_rows", "mean"),
        )
        .reset_index(drop=True)
    )
    candidate_summary = candidate_summary[candidate_summary["valid_auc_paths"] >= MIN_VALID_AUC_PATHS].copy()
    return candidate_summary.sort_values(["ticker", "mean_auc"], ascending=[True, False]).reset_index(drop=True)


def choose_selected_configs(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for ticker, group in candidate_summary.groupby("ticker"):
        top = group.head(TOP_K).copy()
        ranked = top.sort_values(
            ["median_sharpe", "sharpe_iqr", "sharpe_std", "mean_auc"],
            ascending=[False, True, True, False],
        )
        selected.append(ranked.iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def flatten_feature_reports(feature_selection_reports: list[dict]) -> pd.DataFrame:
    rows = []
    for item in feature_selection_reports:
        report = item["report"]
        if isinstance(report, pd.DataFrame):
            for rank, row in report.head(50).reset_index(drop=True).iterrows():
                rows.append(
                    {
                        "ticker": item["ticker"],
                        "barrier": item["barrier"],
                        "feature_selection": item["feature_selection"],
                        "model": item["model"],
                        "rank": rank + 1,
                        **row.to_dict(),
                    }
                )
        else:
            for rank, feature in enumerate(item["selected_features"][:50], start=1):
                rows.append(
                    {
                        "ticker": item["ticker"],
                        "barrier": item["barrier"],
                        "feature_selection": item["feature_selection"],
                        "model": item["model"],
                        "rank": rank,
                        "feature": feature,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    tickers = [ticker.lower() for ticker in args.tickers]
    model_configs = [c for c in MODEL_CONFIGS if c["name"] in set(args.models)]
    fs_configs = [c for c in FEATURE_SELECTION_CONFIGS if c["name"] in set(args.feature_selectors)]

    manifest = load_manifest(args.manifest, tickers)
    features = load_features(tickers)
    barriers_by_ticker = manifest.groupby("ticker")["barrier"].apply(list).to_dict()

    candidates = [
        (ticker, barrier, fs_config, model_config)
        for ticker in tickers
        for barrier in barriers_by_ticker.get(ticker, [])
        for fs_config in fs_configs
        for model_config in model_configs
    ]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    print("Enhanced cached-label CPCV")
    print("Manifest:", args.manifest)
    print("Output dir:", args.output_dir)
    print("Tickers:", ", ".join(tickers))
    print("Cached ticker/barrier labels:", len(manifest))
    print("Feature selectors:", ", ".join(c["name"] for c in fs_configs))
    print("Models:", ", ".join(c["name"] for c in model_configs))
    print("Total candidates:", len(candidates))
    print("Feature matrix:", features.shape)

    if args.dry_run:
        return

    runner = CachedCPCVRunner(manifest, features)
    all_rows = []
    feature_selection_reports = []

    for count, (ticker, barrier, fs_config, model_config) in enumerate(candidates, start=1):
        print(f"{count}/{len(candidates)}: {ticker} | {barrier} | {fs_config['name']} | {model_config['name']}")
        rows, report = runner.evaluate_candidate(ticker, barrier, fs_config, model_config)
        all_rows.extend(rows)
        if report is not None:
            feature_selection_reports.append(report)

    all_cpcv_results = pd.DataFrame(all_rows)
    if all_cpcv_results.empty:
        raise RuntimeError("No CPCV result rows were produced.")

    candidate_summary = summarize_candidates(all_cpcv_results)
    selected_configs_df = choose_selected_configs(candidate_summary)
    feature_reports_df = flatten_feature_reports(feature_selection_reports)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_cpcv_results.to_csv(args.output_dir / "path_level_results.csv", index=False)
    candidate_summary.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    selected_configs_df.to_csv(args.output_dir / "selected_configs.csv", index=False)
    feature_reports_df.to_csv(args.output_dir / "feature_selection_reports.csv", index=False)

    if not args.no_save_models:
        final_models = {
            row["ticker"]: runner.fit_final_model(row.to_dict(), fs_configs, model_configs)
            for _, row in selected_configs_df.iterrows()
        }
        for ticker, obj in final_models.items():
            joblib.dump(obj, args.output_dir / f"{ticker}_final_model.joblib")

    print("Path-level results:", all_cpcv_results.shape)
    print("Candidate summary:", candidate_summary.shape)
    print("Selected configs:")
    print(selected_configs_df[["ticker", "barrier", "feature_selection", "model", "mean_auc", "median_sharpe"]])
    print("Saved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
