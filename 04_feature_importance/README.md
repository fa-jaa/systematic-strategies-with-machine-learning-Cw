# Feature Importance

This folder selects the final model for each energy ticker and computes
cluster-level feature importance.

Selection rule:

1. Read `results/model_comparison.csv`.
2. For each ticker, take the top 3 rows by `mean_auc`.
3. From those top 3, select the row with the highest
   `median_path_trade_sharpe`.
4. Refit only those 4 selected final models on the pre-2022 training rows.
5. Compute feature-level and cluster/component-level importance.

The notebook also plots the top-3 AUC shortlist for each ticker and highlights
the model selected by highest median path Sharpe.

Outputs:

- `results/feature_importance_top3_by_auc.csv`
- `results/feature_importance_selected_models.csv`
- `results/feature_importance_feature_level.csv`
- `results/feature_importance_cluster_level.csv`

Run the notebook:

- `04_feature_importance/feature_importance.ipynb`

Or run the script version:

```bash
python "04_feature_importance/cluster_feature_importance.py"
```

Notes:

- If `results/model_comparison.csv` only contains logistic rows because Random
  Forest is still running, the selection will be made from the available rows.
  Re-run this script after RF finishes to select across both model families.
- For logistic regression, feature importance is absolute standardized
  coefficient magnitude.
- For Random Forest, feature importance is sklearn MDI importance.
- For correlation-clustered models, the representative feature's importance is
  treated as the importance of the whole cluster.
- For PCA models, each principal component is treated as a component-level
  importance group.
