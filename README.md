# Systematic Trading Coursework

This project builds a metamodel on top of the supplied primary trading signals.
The current implementation focuses on the Energy asset class:

- `cl1s` - WTI crude oil
- `ho1s` - heating oil
- `rb1s` - RBOB gasoline
- `ng1s` - natural gas

The coursework deliverable is a probability CSV for January-June 2022 with:

```csv
date,instrument,prediction
2022-01-03,cl1s,0.74
```

## Current Stage

The project is at late model-development stage, but is not submission-ready yet.

Completed or mostly completed:

- raw data is present in `data/raw/`
- HMM regime features have been generated in `data/hmm/`
- technical and merged feature matrices have been generated in `data/features/`
- triple-barrier labelling code exists in `02_triple_barrier/`
- cached reduced-grid triple-barrier labels exist in `kaushaltest/data/labels/triple_barrier/`
- an enhanced cached-label CPCV script exists in `kaushaltest/enhancedcombocpcv.py`

Still required before final submission:

- run the full cached-label CPCV comparison, not only the smoke test
- choose final model/config per Energy instrument
- generate the required Jan-Jun 2022 prediction CSV
- add final out-of-sample metrics, threshold analysis, per-instrument breakdown, and baseline comparison
- write up feature importance at cluster/feature-group level

## Setup

Create an environment and install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Order

Run commands from the repository root.

### 1. Raw Data

Ensure these files exist:

```text
data/raw/ohlcv_data.csv
data/raw/primary_signals.csv
```

### 2. HMM Regime Features

Run the HMM notebooks/scripts first because the feature matrix can merge their outputs:

```powershell
python 01_feature_engineering\hmm\regime_detection.py
python 01_feature_engineering\hmm\regime_categories.py
```

Expected outputs include:

```text
data/hmm/predictions/latent_regime_predictions.csv
data/hmm/probabilities/*.csv
data/hmm/categories/*.csv
```

### 3. Feature Matrix

Run:

```text
01_feature_engineering/feature_engineering_matrix.ipynb
```

Expected outputs:

```text
data/features/technical_analysis_features.csv
data/features/merged_feature_matrix.csv
data/features/feature_catalog.csv
```

### 4. Triple-Barrier Labels

For the isolated reduced-grid label cache:

```powershell
python kaushaltest\generate_triple_barrier_labels.py
```

This writes cached labels to:

```text
kaushaltest/data/labels/triple_barrier/
```

These labels use dates strictly before `2022-01-01`, so Jan-Jun 2022 remains untouched for final prediction/evaluation.

### 5. Cached-Label CPCV Model Comparison

Quick dry run:

```powershell
python kaushaltest\enhancedcombocpcv.py --dry-run
```

Small smoke run:

```powershell
python kaushaltest\enhancedcombocpcv.py --max-candidates 2 --no-save-models --output-dir kaushaltest\outputs\enhancedcombocpcv_smoke
```

Full reduced-grid comparison:

```powershell
python kaushaltest\enhancedcombocpcv.py
```

Expected outputs:

```text
kaushaltest/outputs/enhancedcombocpcv/path_level_results.csv
kaushaltest/outputs/enhancedcombocpcv/candidate_summary.csv
kaushaltest/outputs/enhancedcombocpcv/selected_configs.csv
kaushaltest/outputs/enhancedcombocpcv/feature_selection_reports.csv
kaushaltest/outputs/enhancedcombocpcv/*_final_model.joblib
```

### 6. Original CPCV Notebook

The original notebook remains:

```text
03_model_development/cpcv_energy_modelling.ipynb
```

Use this for explanation, plots, and comparison with the scripted run. The enhanced script is the faster isolated path because it reuses cached triple-barrier labels.

### 7. Final Prediction CSV

The final metamodel probability file is:

```text
deliverables/final_predictions.csv
```

Format:

```csv
date,instrument,prediction
2022-01-03,cl1s,0.9576887296513032
```

This file is used as the inference input for portfolio-weight generation.

### 8. Optional Portfolio Weights

The portfolio-weight allocation strategy lives in:

```text
portfolio_weight_allocation_strategy/
```

Generate fixed-strategy weights from `deliverables/final_predictions.csv`:

```powershell
python portfolio_weight_allocation_strategy\allocate_portfolio_weights.py
```

Expected outputs:

```text
portfolio_weight_allocation_strategy/outputs/strategy_weights.csv
portfolio_weight_allocation_strategy/outputs/allocation_diagnostics.csv
```

Generate the neural strategy from real pre-2022 in-sample probabilities and
`deliverables/final_predictions.csv`:

```powershell
python portfolio_weight_allocation_strategy\neural_portfolio_allocator.py
```

Compare fixed vs neural and rebuild the plots/notebook inputs:

```powershell
python portfolio_weight_allocation_strategy\visualise_strategy_weights.py --fixed-weights deliverables\portfolio_weight_outputs\fixed_strategy\strategy_weights.csv --neural-weights deliverables\portfolio_weight_outputs\neural_strategy\neural_strategy_weights.csv --output-dir deliverables\portfolio_weight_outputs\strategy_comparison
```

Grouped portfolio outputs live under:

```text
deliverables/portfolio_weight_outputs/
```

The comparison notebook is:

```text
deliverables/portfolio_weight_outputs/strategy_comparison/strategy_comparison_explained.ipynb
```

## Notes

- `archive/` contains older experiments.
- `experiments/` contains alternative optimized experiments and is not the main pipeline.
- `kaushaltest/` is isolated experimental work and currently contains the reduced-grid label cache and cached-label CPCV script.
- `portfolio_weight_allocation_strategy/` contains the optional strategy-weight generator.
