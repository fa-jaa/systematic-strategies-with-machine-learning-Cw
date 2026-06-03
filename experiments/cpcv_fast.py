"""
experiments/cpcv_fast.py
========================
Optimised version of cpcv_energy_modelling.ipynb.

Two optimisations over the original notebook:
  1. Disk label cache  — triple-barrier labels computed once per ticker+barrier,
                         saved to data/labels/triple_barrier/{ticker}/{name}.csv,
                         reused across all feature-selection/model candidates.
                         Reduces label computation from 15,840 to 240 runs.

  2. Parallelisation   — model candidates evaluated in parallel using joblib.
                         Uses threading backend (safe on Windows + Jupyter).

This file lives in experiments/ so it can be deleted without touching the main
notebook. The output format is identical to the original notebook.

Usage:
    cd systematic-strategies-with-machine-learning-Cw
    python experiments/cpcv_fast.py              # smoke mode
    python experiments/cpcv_fast.py --full       # full grid
    python experiments/cpcv_fast.py --save       # save outputs
"""

import sys
import warnings
import argparse
from itertools import combinations, product
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# ── Project root ──────────────────────────────────────────────────────────────
def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "02_triple_barrier" / "triple_barrier.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root")

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT / "02_triple_barrier"))
sys.path.insert(0, str(PROJECT_ROOT / "03_model_development"))

from triple_barrier import run_triple_barrier_pipeline
import model_configs as cfg

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Run full grid")
parser.add_argument("--save", action="store_true", help="Save outputs")
parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs (-1 = all cores)")
args = parser.parse_args()

RUN_FULL_GRID = args.full
SAVE_OUTPUTS  = args.save
N_JOBS        = args.jobs

TRAINING_END = pd.Timestamp(cfg.TRAINING_END)
OUTPUT_DIR   = PROJECT_ROOT / "data" / "models" / "cpcv_energy_fast"
LABEL_DIR    = PROJECT_ROOT / "data" / "labels" / "triple_barrier"

if RUN_FULL_GRID:
    tickers_to_run         = cfg.ENERGY_TICKERS
    barrier_configs        = cfg.FULL_BARRIER_CONFIGS
    feature_selection_configs = cfg.FULL_FEATURE_SELECTION_CONFIGS
    model_configs_list     = cfg.FULL_MODEL_CONFIGS
else:
    tickers_to_run         = cfg.SMOKE_TICKERS
    barrier_configs        = cfg.SMOKE_BARRIER_CONFIGS
    feature_selection_configs = cfg.SMOKE_FEATURE_SELECTION_CONFIGS
    model_configs_list     = cfg.SMOKE_MODEL_CONFIGS

if not SHAP_AVAILABLE:
    feature_selection_configs = [c for c in feature_selection_configs if c["method"] != "shap"]
if not XGBOOST_AVAILABLE:
    model_configs_list = [c for c in model_configs_list if c["model_type"] != "xgboost"]

total = len(tickers_to_run) * len(barrier_configs) * len(feature_selection_configs) * len(model_configs_list)
print(f"Mode: {'FULL' if RUN_FULL_GRID else 'SMOKE'}")
print(f"Tickers: {tickers_to_run}")
print(f"Barrier configs: {len(barrier_configs)}")
print(f"Feature selection: {len(feature_selection_configs)}")
print(f"Models: {len(model_configs_list)}")
print(f"Total candidates: {total}")
print(f"Label cache: {LABEL_DIR}")
print()

# ── Load features ─────────────────────────────────────────────────────────────
def load_features():
    merged_path = PROJECT_ROOT / "data" / "features" / "merged_feature_matrix.csv"
    if merged_path.exists():
        print(f"Loading merged feature matrix...")
        df = pd.read_csv(merged_path, parse_dates=["date"])
    else:
        raise FileNotFoundError(f"merged_feature_matrix.csv not found at {merged_path}")
    df = df[df["instrument"].isin(cfg.ENERGY_TICKERS)].copy()
    df = df[df["date"] < TRAINING_END].copy()
    return df.sort_values(["instrument", "date"]).reset_index(drop=True)

features = load_features()
print(f"Feature matrix: {features.shape}")

# ── Disk label cache ──────────────────────────────────────────────────────────
_label_mem_cache = {}
DATE_COLS = ["date", "timeout_date", "touch_date", "training_end"]


def get_labels(ticker, barrier_config):
    """Load from disk if cached, otherwise compute and save — then memoise."""
    key = (ticker, barrier_config["name"])
    if key in _label_mem_cache:
        return _label_mem_cache[key].copy()

    label_path = LABEL_DIR / ticker / f"{barrier_config['name']}.csv"

    if label_path.exists():
        labels = pd.read_csv(label_path, parse_dates=DATE_COLS)
    else:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {k: v for k, v in barrier_config.items() if k != "name"}
        labels = run_triple_barrier_pipeline(
            instrument=ticker,
            training_end=TRAINING_END,
            output_path=label_path,
            **kwargs,
        )
        print(f"  Cached: {label_path.relative_to(PROJECT_ROOT)}")

    _label_mem_cache[key] = labels
    return labels.copy()


# ── Step 1: precompute all labels up front ────────────────────────────────────
print("=== Step 1: Precompute labels ===")
unique_combinations = list(product(tickers_to_run, barrier_configs))
for ticker, bc in unique_combinations:
    get_labels(ticker, bc)
print(f"Labels ready: {len(unique_combinations)} ticker+barrier combinations\n")

# ── Leakage columns ───────────────────────────────────────────────────────────
leakage_columns = {
    "training_end", "vol", "tp", "sl", "timeout_date", "timeout_close",
    "touch_date", "touch_price", "touched_barrier", "triple_barrier_label",
    "metalabel", "holding_period_days", "raw_touch_return", "signed_touch_return",
    "volatility_method", "ewma_span", "volatility_window", "num_days",
    "take_profit_mult", "stop_loss_mult",
}


def make_model_data(ticker, barrier_config):
    labels = get_labels(ticker, barrier_config)
    ticker_features = features[features["instrument"] == ticker].copy()
    labels_for_join = labels.drop(columns=["close"], errors="ignore")
    ticker_features = ticker_features.drop(columns=["primary_signal"], errors="ignore")
    data = labels_for_join.merge(ticker_features, on=["date", "instrument"], how="inner")
    data = data.sort_values("date").reset_index(drop=True)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in leakage_columns]
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    return data, feature_cols


# ── CPCV splits ───────────────────────────────────────────────────────────────
def make_groups(n_rows, n_groups):
    return [g.astype(int) for g in np.array_split(np.arange(n_rows), n_groups) if len(g) > 0]


def embargo_date(dates, end_date, embargo):
    unique_dates = pd.Series(pd.to_datetime(dates).sort_values().unique())
    pos = unique_dates.searchsorted(end_date, side="right")
    if pos >= len(unique_dates):
        return end_date
    return unique_dates.iloc[min(len(unique_dates) - 1, pos + embargo - 1)]


def make_cpcv_splits(data, ticker):
    settings = cfg.CPCV_SETTINGS[ticker] if RUN_FULL_GRID else cfg.SMOKE_CPCV_SETTINGS
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


# ── Feature selection ─────────────────────────────────────────────────────────
def select_none(X_tr, X_te, y_tr, cfg_fs):
    return X_tr.values, X_te.values, list(X_tr.columns), None

def select_mdi(X_tr, X_te, y_tr, cfg_fs):
    k = min(cfg_fs.get("k", 20), X_tr.shape[1])
    tree = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced", random_state=cfg.RANDOM_STATE, n_jobs=1)
    tree.fit(X_tr, y_tr)
    imp = pd.Series(tree.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    sel = imp.head(k).index.tolist()
    return X_tr[sel].values, X_te[sel].values, sel, imp.head(k)

def select_cluster(X_tr, X_te, y_tr, cfg_fs):
    corr_thr = cfg_fs.get("corr_threshold", 0.90)
    max_f = cfg_fs.get("max_features", 50)
    corr = X_tr.corr().abs().fillna(0).clip(0, 1)
    dist = (1 - corr).clip(0, 1)
    np.fill_diagonal(dist.values, 0)
    linkage = hierarchy.linkage(squareform(dist.values, checks=False), method="average")
    clusters = hierarchy.fcluster(linkage, t=1 - corr_thr, criterion="distance")
    mi = pd.Series(mutual_info_classif(X_tr, y_tr, random_state=cfg.RANDOM_STATE), index=X_tr.columns)
    rows = []
    for cid in sorted(np.unique(clusters)):
        members = [X_tr.columns[i] for i, c in enumerate(clusters) if c == cid]
        best = mi.loc[members].sort_values(ascending=False).index[0]
        rows.append({"feature": best, "score": mi.loc[best]})
    report = pd.DataFrame(rows).sort_values("score", ascending=False).head(max_f)
    sel = report["feature"].tolist()
    return X_tr[sel].values, X_te[sel].values, sel, report

def select_pca(X_tr, X_te, y_tr, cfg_fs):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    pca = PCA(n_components=cfg_fs.get("variance", 0.90), random_state=cfg.RANDOM_STATE)
    Xtr_p = pca.fit_transform(Xtr)
    Xte_p = pca.transform(Xte)
    names = [f"PC{i+1}" for i in range(Xtr_p.shape[1])]
    return Xtr_p, Xte_p, names, None

def apply_fs(X_tr, X_te, y_tr, cfg_fs):
    m = cfg_fs["method"]
    if m == "none":    return select_none(X_tr, X_te, y_tr, cfg_fs)
    if m == "mdi":     return select_mdi(X_tr, X_te, y_tr, cfg_fs)
    if m == "cluster": return select_cluster(X_tr, X_te, y_tr, cfg_fs)
    if m == "pca":     return select_pca(X_tr, X_te, y_tr, cfg_fs)
    raise ValueError(f"Unknown method: {m}")


# ── Model factory ─────────────────────────────────────────────────────────────
def make_model(mc):
    p = dict(mc.get("params", {}))
    p.setdefault("random_state", cfg.RANDOM_STATE)
    t = mc["model_type"]
    if t == "logistic":            return LogisticRegression(**p)
    if t == "random_forest":       return RandomForestClassifier(**p)
    if t == "extra_trees":         return ExtraTreesClassifier(**p)
    if t == "hist_gradient_boosting": return HistGradientBoostingClassifier(**p)
    if t == "mlp":                 return MLPClassifier(**p)
    if t == "xgboost":             return XGBClassifier(**p)
    raise ValueError(f"Unknown model: {t}")


# ── Evaluate one candidate ────────────────────────────────────────────────────
def sharpe_ratio(returns):
    s = pd.Series(returns).dropna()
    if len(s) < 2 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / s.std(ddof=1) * np.sqrt(len(s))


def evaluate_candidate(ticker, barrier_config, fs_config, model_config):
    data, feature_cols = make_model_data(ticker, barrier_config)
    splits = make_cpcv_splits(data, ticker)
    rows = []

    for split in splits:
        train = data.iloc[split["train_idx"]]
        test  = data.iloc[split["test_idx"]]
        y_tr  = train["metalabel"].astype(int)
        y_te  = test["metalabel"].astype(int)
        if y_tr.nunique() < 2:
            continue

        imp = SimpleImputer(strategy="median")
        X_tr = pd.DataFrame(imp.fit_transform(train[feature_cols].replace([np.inf,-np.inf], np.nan)), columns=feature_cols, index=train.index)
        X_te = pd.DataFrame(imp.transform(test[feature_cols].replace([np.inf,-np.inf], np.nan)), columns=feature_cols, index=test.index)

        X_tr_s, X_te_s, sel_feats, _ = apply_fs(X_tr, X_te, y_tr, fs_config)

        if model_config["needs_scaling"]:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_s)
            X_te_s = sc.transform(X_te_s)

        model = make_model(model_config)
        model.fit(X_tr_s, y_tr)
        y_score = model.predict_proba(X_te_s)[:, 1] if hasattr(model, "predict_proba") else 1 / (1 + np.exp(-model.decision_function(X_te_s)))
        y_pred  = (y_score >= cfg.PROBA_THRESHOLD).astype(int)

        auc = roc_auc_score(y_te, y_score) if y_te.nunique() == 2 else np.nan
        trade_returns = test.loc[y_pred == 1, "signed_touch_return"]

        rows.append({
            "ticker": ticker, "barrier": barrier_config["name"],
            "feature_selection": fs_config["name"], "model": model_config["name"],
            "split_id": split["split_id"], "auc": auc,
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "sharpe": sharpe_ratio(trade_returns),
            "trade_count": int((y_pred == 1).sum()),
            "selected_feature_count": len(sel_feats),
        })

    return rows


# ── Step 2: run grid in parallel ──────────────────────────────────────────────
print("=== Step 2: Run model grid (parallel) ===")
all_combos = list(product(tickers_to_run, barrier_configs, feature_selection_configs, model_configs_list))
print(f"Running {len(all_combos)} candidates with {N_JOBS} jobs...\n")

results_list = Parallel(n_jobs=N_JOBS, backend="threading", verbose=5)(
    delayed(evaluate_candidate)(ticker, bc, fs, mc)
    for ticker, bc, fs, mc in all_combos
)

all_rows = [row for rows in results_list for row in rows]
all_cpcv_results = pd.DataFrame(all_rows)
print(f"\nPath-level results: {all_cpcv_results.shape}")

# ── Summarise and rank ────────────────────────────────────────────────────────
candidate_summary = (
    all_cpcv_results
    .groupby(["ticker", "barrier", "feature_selection", "model"], as_index=False)
    .agg(
        mean_auc=("auc", "mean"),
        median_auc=("auc", "median"),
        valid_auc_paths=("auc", lambda x: x.notna().sum()),
        median_sharpe=("sharpe", "median"),
        mean_sharpe=("sharpe", "mean"),
        sharpe_iqr=("sharpe", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        total_trades=("trade_count", "sum"),
    )
)
candidate_summary = candidate_summary[candidate_summary["valid_auc_paths"] >= cfg.MIN_VALID_AUC_PATHS]
candidate_summary = candidate_summary.sort_values(["ticker", "mean_auc"], ascending=[True, False]).reset_index(drop=True)

print("\n=== Top 5 per ticker ===")
for ticker, group in candidate_summary.groupby("ticker"):
    print(f"\n{ticker}:")
    print(group.head(cfg.TOP_K).to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
if SAVE_OUTPUTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_cpcv_results.to_csv(OUTPUT_DIR / "path_level_results.csv", index=False)
    candidate_summary.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR}")
else:
    print("\nSAVE_OUTPUTS=False — nothing written (pass --save to save)")
