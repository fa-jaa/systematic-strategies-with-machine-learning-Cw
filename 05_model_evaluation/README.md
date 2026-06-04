# Model Evaluation

This folder runs the selected meta-models on the 2022+ test window.

Notebook:

- `05_model_evaluation/model_evaluation.ipynb`

Input:

- `results/selected_model_run_configs.csv`

Default output:

- `deliverables/final_predictions.csv`
- `deliverables/insample_predicitons/insample_predictions.csv`

Both deliverable CSVs contain only:

- `date`
- `instrument`
- `prediction`

Run:

```bash
python 05_model_evaluation/evaluate_selected_models.py
```

The notebook also writes evaluation diagnostics:

- `05_model_evaluation/outputs/test_set_detailed_predictions.csv`
- `05_model_evaluation/outputs/test_set_metrics.csv`
- `05_model_evaluation/outputs/test_set_return_diagnostics.csv`
