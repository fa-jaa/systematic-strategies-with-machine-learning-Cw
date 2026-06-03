"""Configuration lists for CPCV energy model development.

This file is deliberately simple: it only contains plain dictionaries/lists.
The notebook imports these configs and turns them into labels, selectors, and
models.

Edit this file when you want to add/remove:
- tickers
- triple-barrier settings
- feature-selection methods
- model hyperparameters
"""

ENERGY_TICKERS = ["cl1s", "ho1s", "rb1s", "ng1s"]
SMOKE_TICKERS = ["cl1s"]

TRAINING_END = "2022-01-01"
RANDOM_STATE = 42

PROBA_THRESHOLD = 0.50
TOP_K = 5
MIN_VALID_AUC_PATHS = 3

CPCV_SETTINGS = {
    "cl1s": {"n_groups": 6, "k_test_groups": 2, "embargo": 5},
    "rb1s": {"n_groups": 6, "k_test_groups": 2, "embargo": 5},
    "ho1s": {"n_groups": 4, "k_test_groups": 1, "embargo": 5},
    "ng1s": {"n_groups": 4, "k_test_groups": 1, "embargo": 5},
}

SMOKE_CPCV_SETTINGS = {"n_groups": 4, "k_test_groups": 1, "embargo": 5}


SMOKE_BARRIER_CONFIGS = [
    {
        "name": "ewma_d10_tp2_sl2",
        "volatility_method": "ewma",
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": 10,
        "take_profit_mult": 2.0,
        "stop_loss_mult": 2.0,
    }
]

FULL_BARRIER_CONFIGS = [
    {
        "name": f"{vol_method}_d{num_days}_tp{tp}_sl{sl}",
        "volatility_method": vol_method,
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": num_days,
        "take_profit_mult": tp,
        "stop_loss_mult": sl,
    }
    for vol_method in ["ewma", "rolling", "parkinson", "garman_klass"]
    for num_days in [5, 10, 20]
    for tp, sl in [(1.5, 1.5), (2.0, 2.0), (3.0, 3.0), (2.0, 1.5), (1.5, 2.0)]
]


SMOKE_FEATURE_SELECTION_CONFIGS = [
    {"name": "none", "method": "none"},
    {"name": "mdi_top20", "method": "mdi", "k": 20},
    {"name": "cluster_corr90", "method": "cluster", "corr_threshold": 0.90, "max_features": 30},
    {"name": "pca90", "method": "pca", "variance": 0.90},
    {"name": "shap_top20", "method": "shap", "k": 20},
]

FULL_FEATURE_SELECTION_CONFIGS = [
    {"name": "none", "method": "none"},
    {"name": "mdi_top20", "method": "mdi", "k": 20},
    {"name": "mdi_top50", "method": "mdi", "k": 50},
    {"name": "cluster_corr90", "method": "cluster", "corr_threshold": 0.90, "max_features": 50},
    {"name": "cluster_corr95", "method": "cluster", "corr_threshold": 0.95, "max_features": 75},
    {"name": "pca80", "method": "pca", "variance": 0.80},
    {"name": "pca90", "method": "pca", "variance": 0.90},
    {"name": "pca95", "method": "pca", "variance": 0.95},
    {"name": "cluster90_pca90", "method": "cluster_pca", "corr_threshold": 0.90, "max_features": 75, "variance": 0.90},
    {"name": "shap_top20", "method": "shap", "k": 20},
    {"name": "shap_top50", "method": "shap", "k": 50},
]


SMOKE_MODEL_CONFIGS = [
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
]

FULL_MODEL_CONFIGS = SMOKE_MODEL_CONFIGS + [
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
        "name": "hist_gradient_boosting",
        "model_type": "hist_gradient_boosting",
        "needs_scaling": False,
        "params": {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15},
    },
    {
        "name": "mlp",
        "model_type": "mlp",
        "needs_scaling": True,
        "params": {"hidden_layer_sizes": (32,), "max_iter": 500, "early_stopping": True},
    },
    {
        "name": "xgboost",
        "model_type": "xgboost",
        "needs_scaling": False,
        "params": {
            "n_estimators": 120,
            "max_depth": 2,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "n_jobs": -1,
        },
    },
]
