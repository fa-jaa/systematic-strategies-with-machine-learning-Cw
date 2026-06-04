"""Reusable feature-selection and feature-reduction utilities for linear/meta models.

Purpose
-------
This module contains CPCV-safe feature-selection helpers that can be imported
from notebooks, training scripts, and hyperparameter-search code.

Available methods
-----------------
1. Correlation clustering
   `CorrelationClusterSelector` removes redundant highly correlated features by:
   - computing an absolute correlation matrix on the training features only,
   - building clusters of features where abs(correlation) >= threshold,
   - keeping one representative feature per cluster,
   - dropping the remaining features in that cluster.

2. PCA feature reduction
   `PCAFeatureReducer` converts numeric features into principal components by:
   - fitting imputation values on the training features only,
   - optionally standardising features using training-fold statistics only,
   - fitting PCA on the training fold only,
   - transforming validation/test folds with the fitted preprocessing + PCA.

Important CPCV note
-------------------
Fit selectors/reducers on the TRAIN fold only, then transform validation/test
folds using the fitted object. Do not fit on the full dataset before CPCV.

Examples
--------
Correlation clustering:

    from feature_selection_methods import CorrelationClusterSelector

    selector = CorrelationClusterSelector(
        corr_threshold=0.95,
        corr_method="spearman",
        selection_method="target_corr",
    )

    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

PCA:

    from feature_selection_methods import PCAFeatureReducer

    pca_reducer = PCAFeatureReducer(
        n_components=0.95,
        standardize=True,
        component_prefix="pca",
    )

    pca_reducer.fit(X_train)
    X_train_pca = pca_reducer.transform(X_train)
    X_test_pca = pca_reducer.transform(X_test)

    print(pca_reducer.pca_summary_)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_EXCLUDE_COLUMNS = (
    "date",
    "instrument",
    "primary_signal",
    "triple_barrier_label",
    "metalabel",
    "target",
    "y",
)

CorrelationMethod = Literal["pearson", "spearman", "kendall"]
SelectionMethod = Literal["target_corr", "missing_then_variance", "variance", "first"]
ImputeStrategy = Literal["median", "mean", "zero"]


@dataclass
class CorrelationClusterResult:
    """Lightweight result object returned by `run_correlation_cluster_selection`."""

    selected_df: pd.DataFrame
    selected_features: list[str]
    dropped_features: list[str]
    cluster_summary: pd.DataFrame
    pair_summary: pd.DataFrame
    selector: "CorrelationClusterSelector"


@dataclass
class PCAFeatureReductionResult:
    """Lightweight result object returned by `run_pca_feature_reduction`."""

    transformed_df: pd.DataFrame
    component_columns: list[str]
    explained_variance_summary: pd.DataFrame
    reducer: "PCAFeatureReducer"


def _normalise_columns(columns: Iterable[str] | None) -> list[str]:
    """Convert column input to a clean list while preserving order."""

    if columns is None:
        return []
    return list(dict.fromkeys(columns))


def infer_numeric_feature_columns(
    df: pd.DataFrame,
    exclude_columns: Iterable[str] = DEFAULT_EXCLUDE_COLUMNS,
) -> list[str]:
    """Infer numeric feature columns, excluding ID/target/signal columns."""

    excluded = set(exclude_columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded]


def _coerce_y_to_series(y: pd.Series | np.ndarray | list | None, index: pd.Index) -> pd.Series | None:
    """Return y as a numeric Series aligned to X's index where possible."""

    if y is None:
        return None

    if isinstance(y, pd.Series):
        if y.index.equals(index):
            y_series = y.copy()
        elif len(y) == len(index):
            y_series = pd.Series(y.to_numpy(), index=index, name=y.name)
        else:
            y_series = y.reindex(index)
    else:
        if len(y) != len(index):
            raise ValueError("y must have the same length as X if it is not a pandas Series.")
        y_series = pd.Series(y, index=index, name="target")

    return pd.to_numeric(y_series, errors="coerce")


def _target_correlation_scores(
    X: pd.DataFrame,
    y: pd.Series,
    method: CorrelationMethod,
) -> pd.Series:
    """Compute abs(feature-target correlation), handling constants safely."""

    scores: dict[str, float] = {}
    y_name = y.name if y.name is not None else "target"

    for col in X.columns:
        pair = pd.concat([X[col], y.rename(y_name)], axis=1).dropna()
        if pair.shape[0] < 3:
            scores[col] = np.nan
            continue
        if pair.iloc[:, 0].nunique(dropna=True) <= 1:
            scores[col] = np.nan
            continue
        if pair.iloc[:, 1].nunique(dropna=True) <= 1:
            scores[col] = np.nan
            continue

        corr_value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
        scores[col] = abs(corr_value) if pd.notna(corr_value) else np.nan

    return pd.Series(scores, name="target_abs_corr")


def _high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Return all upper-triangle feature pairs with abs corr >= threshold."""

    if corr_matrix.empty:
        return pd.DataFrame(columns=["feature_1", "feature_2", "abs_corr"])

    corr_values = corr_matrix.to_numpy(dtype=float)
    upper_mask = np.triu(np.ones(corr_values.shape, dtype=bool), k=1)
    high_mask = upper_mask & np.isfinite(corr_values) & (corr_values >= threshold)
    row_idx, col_idx = np.where(high_mask)

    pairs = pd.DataFrame(
        {
            "feature_1": corr_matrix.index[row_idx],
            "feature_2": corr_matrix.columns[col_idx],
            "abs_corr": corr_values[row_idx, col_idx],
        }
    )

    if pairs.empty:
        return pairs

    return pairs.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def _connected_components(
    features: list[str],
    pair_summary: pd.DataFrame,
) -> list[list[str]]:
    """Build correlation clusters from high-correlation edges."""

    order = {feature: position for position, feature in enumerate(features)}
    adjacency = {feature: set() for feature in features}

    for row in pair_summary.itertuples(index=False):
        f1 = row.feature_1
        f2 = row.feature_2
        adjacency[f1].add(f2)
        adjacency[f2].add(f1)

    clusters: list[list[str]] = []
    visited: set[str] = set()

    for feature in features:
        if feature in visited:
            continue

        stack = [feature]
        component: list[str] = []
        visited.add(feature)

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

        if len(component) > 1:
            clusters.append(sorted(component, key=lambda col: order[col]))

    return clusters


def _choose_representative(
    cluster: list[str],
    X: pd.DataFrame,
    selection_method: SelectionMethod,
    original_order: dict[str, int],
    target_scores: pd.Series | None = None,
) -> tuple[str, pd.DataFrame]:
    """Choose one representative feature from a correlation cluster."""

    stats = pd.DataFrame(index=cluster)
    stats["feature"] = cluster
    stats["original_order"] = [original_order[col] for col in cluster]
    stats["missing_rate"] = X[cluster].isna().mean()
    stats["n_unique"] = X[cluster].nunique(dropna=True)
    stats["variance"] = X[cluster].var(skipna=True)

    if target_scores is not None:
        stats["target_abs_corr"] = target_scores.reindex(cluster)
    else:
        stats["target_abs_corr"] = np.nan

    # If target correlation is requested but unavailable, fall back safely.
    effective_method = selection_method
    if selection_method == "target_corr" and stats["target_abs_corr"].notna().sum() == 0:
        effective_method = "missing_then_variance"

    if effective_method == "target_corr":
        ranked = stats.assign(
            target_abs_corr_for_sort=stats["target_abs_corr"].fillna(-np.inf),
            variance_for_sort=stats["variance"].fillna(-np.inf),
        ).sort_values(
            by=[
                "target_abs_corr_for_sort",
                "missing_rate",
                "n_unique",
                "variance_for_sort",
                "original_order",
            ],
            ascending=[False, True, False, False, True],
        )
    elif effective_method == "missing_then_variance":
        ranked = stats.assign(
            variance_for_sort=stats["variance"].fillna(-np.inf),
        ).sort_values(
            by=["missing_rate", "n_unique", "variance_for_sort", "original_order"],
            ascending=[True, False, False, True],
        )
    elif effective_method == "variance":
        ranked = stats.assign(
            variance_for_sort=stats["variance"].fillna(-np.inf),
        ).sort_values(
            by=["variance_for_sort", "missing_rate", "n_unique", "original_order"],
            ascending=[False, True, False, True],
        )
    elif effective_method == "first":
        ranked = stats.sort_values("original_order", ascending=True)
    else:
        raise ValueError(
            "selection_method must be one of "
            "'target_corr', 'missing_then_variance', 'variance', or 'first'."
        )

    representative = str(ranked.iloc[0]["feature"])
    return representative, stats.reset_index(drop=True)


class CorrelationClusterSelector:
    """Remove redundant features by keeping one feature per correlation cluster.

    Parameters
    ----------
    corr_threshold:
        Absolute correlation threshold used to connect two features into the
        same cluster. A value of 0.95 is a common starting point.
    corr_method:
        Correlation method passed to pandas. Spearman is often sensible for
        financial features because it is rank-based and less sensitive to
        extreme values than Pearson.
    selection_method:
        How to choose the representative feature from each cluster.
        - "target_corr": keep the feature with strongest abs correlation to y.
        - "missing_then_variance": prefer fewer missing values, then more unique
          values, then larger variance.
        - "variance": keep the highest-variance feature.
        - "first": keep the first feature in the original feature order.
    exclude_columns:
        Columns not to treat as candidate features when feature_columns is None.
    min_periods:
        Minimum number of overlapping observations required for a correlation.
    """

    def __init__(
        self,
        corr_threshold: float = 0.95,
        corr_method: CorrelationMethod = "spearman",
        selection_method: SelectionMethod = "target_corr",
        exclude_columns: Iterable[str] = DEFAULT_EXCLUDE_COLUMNS,
        min_periods: int = 20,
    ) -> None:
        if not 0 < corr_threshold < 1:
            raise ValueError("corr_threshold must be between 0 and 1.")
        if corr_method not in {"pearson", "spearman", "kendall"}:
            raise ValueError("corr_method must be 'pearson', 'spearman', or 'kendall'.")

        self.corr_threshold = corr_threshold
        self.corr_method = corr_method
        self.selection_method = selection_method
        self.exclude_columns = tuple(exclude_columns)
        self.min_periods = min_periods

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | list | None = None,
        feature_columns: Iterable[str] | None = None,
    ) -> "CorrelationClusterSelector":
        """Fit the correlation cluster selector on training data only."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if feature_columns is None:
            feature_cols = infer_numeric_feature_columns(X, self.exclude_columns)
        else:
            feature_cols = _normalise_columns(feature_columns)

        missing_cols = sorted(set(feature_cols) - set(X.columns))
        if missing_cols:
            raise ValueError(f"feature_columns not found in X: {missing_cols}")

        X_numeric = X[feature_cols].apply(pd.to_numeric, errors="coerce")
        feature_cols = X_numeric.columns.tolist()
        original_order = {feature: position for position, feature in enumerate(feature_cols)}

        y_series = _coerce_y_to_series(y, X.index)
        if self.selection_method == "target_corr" and y_series is not None:
            target_scores = _target_correlation_scores(X_numeric, y_series, self.corr_method)
        else:
            target_scores = None

        corr_matrix = X_numeric.corr(method=self.corr_method, min_periods=self.min_periods).abs()
        pair_summary = _high_correlation_pairs(corr_matrix, self.corr_threshold)
        clusters = _connected_components(feature_cols, pair_summary)

        selected_features = set(feature_cols)
        dropped_features: list[str] = []
        cluster_rows: list[dict[str, object]] = []

        for cluster_id, cluster in enumerate(clusters, start=1):
            representative, stats = _choose_representative(
                cluster=cluster,
                X=X_numeric,
                selection_method=self.selection_method,
                original_order=original_order,
                target_scores=target_scores,
            )

            cluster_drops = [feature for feature in cluster if feature != representative]
            selected_features.difference_update(cluster_drops)
            dropped_features.extend(cluster_drops)

            cluster_corr = corr_matrix.loc[cluster, cluster].copy()
            cluster_corr_values = cluster_corr.where(
                np.triu(np.ones(cluster_corr.shape, dtype=bool), k=1)
            ).stack()

            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "n_features": len(cluster),
                    "representative_feature": representative,
                    "dropped_features": cluster_drops,
                    "cluster_features": cluster,
                    "max_abs_corr_in_cluster": (
                        float(cluster_corr_values.max()) if not cluster_corr_values.empty else np.nan
                    ),
                    "mean_abs_corr_in_cluster": (
                        float(cluster_corr_values.mean()) if not cluster_corr_values.empty else np.nan
                    ),
                    "selection_method": self.selection_method,
                }
            )

        self.feature_columns_ = feature_cols
        self.selected_features_ = [feature for feature in feature_cols if feature in selected_features]
        self.dropped_features_ = [feature for feature in feature_cols if feature in set(dropped_features)]
        self.corr_matrix_ = corr_matrix
        self.pair_summary_ = pair_summary
        self.cluster_summary_ = pd.DataFrame(cluster_rows)
        self.target_scores_ = target_scores
        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: pd.DataFrame,
        keep_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return X with only selected features, plus optional ID/target columns."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit before transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        keep_cols = [col for col in _normalise_columns(keep_columns) if col in X.columns]
        missing_selected = [col for col in self.selected_features_ if col not in X.columns]
        if missing_selected:
            raise ValueError(f"Selected features missing from X: {missing_selected}")

        output_cols = keep_cols + [col for col in self.selected_features_ if col not in keep_cols]
        return X.loc[:, output_cols].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | list | None = None,
        feature_columns: Iterable[str] | None = None,
        keep_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Fit on X/y and return the selected version of X."""

        return self.fit(X, y=y, feature_columns=feature_columns).transform(
            X,
            keep_columns=keep_columns,
        )

    def get_support(self) -> pd.Series:
        """Boolean support mask indexed by the fitted feature columns."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit before get_support.")

        selected = set(self.selected_features_)
        return pd.Series(
            [feature in selected for feature in self.feature_columns_],
            index=self.feature_columns_,
            name="selected",
        )


class PCAFeatureReducer:
    """Reduce numeric features into principal components in a CPCV-safe way.

    Parameters
    ----------
    n_components:
        Number of components to keep. This follows sklearn PCA behaviour:
        - int: exact number of components, e.g. 10
        - float between 0 and 1: keep enough components to explain this fraction
          of variance, e.g. 0.95
        - None: keep all possible components
    standardize:
        If True, fit a StandardScaler on the training fold before PCA. This is
        usually recommended when features have different units/scales.
    impute_strategy:
        How to fill missing values before PCA, using training-fold statistics.
        Use "median" by default for financial features.
    component_prefix:
        Prefix for generated component column names.
    exclude_columns:
        Columns not to treat as candidate features when feature_columns is None.
    random_state:
        Random state passed to PCA. Mostly relevant for randomized solvers.
    svd_solver:
        Solver passed to sklearn PCA. "full" is deterministic and supports
        variance-ratio n_components such as 0.95.
    """

    def __init__(
        self,
        n_components: int | float | None = 0.95,
        standardize: bool = True,
        impute_strategy: ImputeStrategy = "median",
        component_prefix: str = "pca",
        exclude_columns: Iterable[str] = DEFAULT_EXCLUDE_COLUMNS,
        random_state: int | None = 42,
        svd_solver: str = "full",
    ) -> None:
        if isinstance(n_components, float) and not 0 < n_components <= 1:
            raise ValueError("If n_components is a float, it must be in the interval (0, 1].")
        if isinstance(n_components, int) and n_components <= 0:
            raise ValueError("If n_components is an int, it must be positive.")
        if impute_strategy not in {"median", "mean", "zero"}:
            raise ValueError("impute_strategy must be 'median', 'mean', or 'zero'.")
        if not component_prefix:
            raise ValueError("component_prefix must be a non-empty string.")

        self.n_components = n_components
        self.standardize = standardize
        self.impute_strategy = impute_strategy
        self.component_prefix = component_prefix
        self.exclude_columns = tuple(exclude_columns)
        self.random_state = random_state
        self.svd_solver = svd_solver

    def _fit_imputer(self, X_numeric: pd.DataFrame) -> pd.Series:
        """Fit per-feature imputation values using training data only."""

        if self.impute_strategy == "median":
            values = X_numeric.median(skipna=True)
        elif self.impute_strategy == "mean":
            values = X_numeric.mean(skipna=True)
        elif self.impute_strategy == "zero":
            values = pd.Series(0.0, index=X_numeric.columns)
        else:
            raise ValueError("Unsupported impute_strategy.")

        # A column that is entirely NaN will have NaN median/mean. Use 0 safely.
        return values.fillna(0.0)

    def _prepare_matrix_for_fit(self, X: pd.DataFrame, feature_columns: Iterable[str] | None) -> pd.DataFrame:
        """Select and numeric-coerce feature columns for fitting."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if feature_columns is None:
            feature_cols = infer_numeric_feature_columns(X, self.exclude_columns)
        else:
            feature_cols = _normalise_columns(feature_columns)

        if not feature_cols:
            raise ValueError("No feature columns were provided/inferred for PCA.")

        missing_cols = sorted(set(feature_cols) - set(X.columns))
        if missing_cols:
            raise ValueError(f"feature_columns not found in X: {missing_cols}")

        return X[feature_cols].apply(pd.to_numeric, errors="coerce")

    def _prepare_matrix_for_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted column order and imputation to new data."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit before transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        missing_features = [col for col in self.feature_columns_ if col not in X.columns]
        if missing_features:
            raise ValueError(f"PCA input features missing from X: {missing_features}")

        X_numeric = X[self.feature_columns_].apply(pd.to_numeric, errors="coerce")
        X_imputed = X_numeric.fillna(self.impute_values_)
        return X_imputed

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | list | None = None,
        feature_columns: Iterable[str] | None = None,
    ) -> "PCAFeatureReducer":
        """Fit imputation, optional scaling, and PCA on training data only.

        The y argument is accepted for API symmetry with selectors but is not
        used, because PCA is unsupervised.
        """

        del y  # PCA is unsupervised; this avoids accidental use of labels.

        X_numeric = self._prepare_matrix_for_fit(X, feature_columns=feature_columns)
        self.feature_columns_ = X_numeric.columns.tolist()
        self.impute_values_ = self._fit_imputer(X_numeric)
        X_imputed = X_numeric.fillna(self.impute_values_)

        if X_imputed.shape[0] < 2:
            raise ValueError("PCA requires at least two rows in the training data.")

        max_components = min(X_imputed.shape[0], X_imputed.shape[1])
        if isinstance(self.n_components, int) and self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is too large. Maximum allowed is "
                f"min(n_rows, n_features)={max_components}."
            )

        if self.standardize:
            self.scaler_ = StandardScaler()
            X_for_pca = self.scaler_.fit_transform(X_imputed)
        else:
            self.scaler_ = None
            X_for_pca = X_imputed.to_numpy(dtype=float)

        self.pca_ = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )
        self.pca_.fit(X_for_pca)

        n_components_fitted = int(self.pca_.n_components_)
        self.component_columns_ = [
            f"{self.component_prefix}_{component_number:03d}"
            for component_number in range(1, n_components_fitted + 1)
        ]

        explained = self.pca_.explained_variance_ratio_
        self.explained_variance_ratio_ = pd.Series(
            explained,
            index=self.component_columns_,
            name="explained_variance_ratio",
        )
        self.cumulative_explained_variance_ = self.explained_variance_ratio_.cumsum()
        self.pca_summary_ = pd.DataFrame(
            {
                "component": self.component_columns_,
                "explained_variance_ratio": self.explained_variance_ratio_.values,
                "cumulative_explained_variance": self.cumulative_explained_variance_.values,
                "singular_value": self.pca_.singular_values_,
            }
        )

        self.n_original_features_ = len(self.feature_columns_)
        self.n_components_fitted_ = n_components_fitted
        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        keep_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return PCA component dataframe, plus optional ID/target columns."""

        X_imputed = self._prepare_matrix_for_transform(X)

        if self.standardize:
            X_for_pca = self.scaler_.transform(X_imputed)
        else:
            X_for_pca = X_imputed.to_numpy(dtype=float)

        component_values = self.pca_.transform(X_for_pca)
        component_df = pd.DataFrame(
            component_values,
            columns=self.component_columns_,
            index=X.index,
        )

        keep_cols = [col for col in _normalise_columns(keep_columns) if col in X.columns]
        if keep_cols:
            return pd.concat([X.loc[:, keep_cols].copy(), component_df], axis=1)
        return component_df

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | list | None = None,
        feature_columns: Iterable[str] | None = None,
        keep_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Fit on X and return PCA-transformed X."""

        return self.fit(X, y=y, feature_columns=feature_columns).transform(
            X,
            keep_columns=keep_columns,
        )

    def get_component_loadings(self) -> pd.DataFrame:
        """Return PCA loadings with original features as rows and PCs as columns."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit before get_component_loadings.")

        return pd.DataFrame(
            self.pca_.components_.T,
            index=self.feature_columns_,
            columns=self.component_columns_,
        )

    def get_top_loadings(self, n: int = 10) -> pd.DataFrame:
        """Return the largest absolute loadings for each component.

        This is useful for interpreting which original features drive each PC.
        """

        if n <= 0:
            raise ValueError("n must be positive.")

        loadings = self.get_component_loadings()
        rows: list[dict[str, object]] = []
        for component in loadings.columns:
            top = loadings[component].abs().sort_values(ascending=False).head(n)
            for feature in top.index:
                rows.append(
                    {
                        "component": component,
                        "feature": feature,
                        "loading": loadings.loc[feature, component],
                        "abs_loading": abs(loadings.loc[feature, component]),
                    }
                )
        return pd.DataFrame(rows)


def run_correlation_cluster_selection(
    train_df: pd.DataFrame,
    target_col: str | None = None,
    feature_columns: Iterable[str] | None = None,
    keep_columns: Iterable[str] | None = None,
    corr_threshold: float = 0.95,
    corr_method: CorrelationMethod = "spearman",
    selection_method: SelectionMethod = "target_corr",
    exclude_columns: Iterable[str] = DEFAULT_EXCLUDE_COLUMNS,
    min_periods: int = 20,
) -> CorrelationClusterResult:
    """Convenience wrapper for fitting the selector and returning useful outputs.

    This should be called on the training fold only. Use the returned selector to
    transform validation/test folds.
    """

    if target_col is not None:
        if target_col not in train_df.columns:
            raise ValueError(f"target_col {target_col!r} not found in train_df.")
        y = train_df[target_col]
    else:
        y = None

    selector = CorrelationClusterSelector(
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        selection_method=selection_method,
        exclude_columns=exclude_columns,
        min_periods=min_periods,
    )
    selected_df = selector.fit_transform(
        train_df,
        y=y,
        feature_columns=feature_columns,
        keep_columns=keep_columns,
    )

    return CorrelationClusterResult(
        selected_df=selected_df,
        selected_features=selector.selected_features_,
        dropped_features=selector.dropped_features_,
        cluster_summary=selector.cluster_summary_,
        pair_summary=selector.pair_summary_,
        selector=selector,
    )


def run_pca_feature_reduction(
    train_df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
    keep_columns: Iterable[str] | None = None,
    n_components: int | float | None = 0.95,
    standardize: bool = True,
    impute_strategy: ImputeStrategy = "median",
    component_prefix: str = "pca",
    exclude_columns: Iterable[str] = DEFAULT_EXCLUDE_COLUMNS,
    random_state: int | None = 42,
    svd_solver: str = "full",
) -> PCAFeatureReductionResult:
    """Convenience wrapper for fitting PCA and returning useful outputs.

    This should be called on the training fold only. Use the returned reducer to
    transform validation/test folds.
    """

    reducer = PCAFeatureReducer(
        n_components=n_components,
        standardize=standardize,
        impute_strategy=impute_strategy,
        component_prefix=component_prefix,
        exclude_columns=exclude_columns,
        random_state=random_state,
        svd_solver=svd_solver,
    )
    transformed_df = reducer.fit_transform(
        train_df,
        feature_columns=feature_columns,
        keep_columns=keep_columns,
    )

    return PCAFeatureReductionResult(
        transformed_df=transformed_df,
        component_columns=reducer.component_columns_,
        explained_variance_summary=reducer.pca_summary_,
        reducer=reducer,
    )


__all__ = [
    "CorrelationClusterResult",
    "CorrelationClusterSelector",
    "DEFAULT_EXCLUDE_COLUMNS",
    "PCAFeatureReducer",
    "PCAFeatureReductionResult",
    "infer_numeric_feature_columns",
    "run_correlation_cluster_selection",
    "run_pca_feature_reduction",
]
