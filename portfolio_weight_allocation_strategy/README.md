# Portfolio Weight Allocation Strategy

This folder is for the optional strategy-construction track. It is separate from
the model-development notebooks/scripts and only needs:

- primary signals from `data/raw/primary_signals.csv`
- OHLCV prices from `data/raw/ohlcv_data.csv`
- a metamodel probability CSV with `date,instrument,prediction`

Until the cleaned model probabilities are ready, use the placeholder files in
`probabilities/`.

## PDF Guidance Used

From `Systematic_CW.pdf`:

- optional strategy output format is `date,instrument,weight`
- weight is signed: positive means long, negative means short
- report CAGR, annualised volatility, Sharpe, Sortino, max drawdown, holding period, and turnover

From `BUSI70575_Optional_Session_3.pdf`:

- use the primary signal as the direction
- use the metamodel probability as confidence/sizing
- probabilities should eventually be calibrated before serious sizing
- fixed sizing choices include all-or-nothing, model-confidence, and NCDF sizing
- volatility targeting scales conviction by `target_vol / ex_ante_vol`
- weights at date `t` must use only information available up to `t`

Lecture 4 and Lecture 7 support the neural/VSN/TFT modelling side, but this
folder implements the simpler fixed sizing route so it can run immediately.

## Run Order

From the repository root:

```powershell
python portfolio_weight_allocation_strategy\generate_placeholder_probabilities.py
python portfolio_weight_allocation_strategy\allocate_portfolio_weights.py
```

The first command creates placeholder probability CSVs:

```text
portfolio_weight_allocation_strategy/probabilities/placeholder_energy_active_055.csv
portfolio_weight_allocation_strategy/probabilities/placeholder_energy_neutral_050.csv
portfolio_weight_allocation_strategy/probabilities/placeholder_all_assets_active_055.csv
```

The second command creates weights from the default Energy placeholder:

```text
portfolio_weight_allocation_strategy/outputs/strategy_weights.csv
portfolio_weight_allocation_strategy/outputs/allocation_diagnostics.csv
```

## Replacing Placeholders

When cleaned metamodel probabilities arrive, save them as a CSV with:

```csv
date,instrument,prediction
2022-01-03,cl1s,0.74
```

Then run:

```powershell
python portfolio_weight_allocation_strategy\allocate_portfolio_weights.py --probability-csv path\to\clean_probabilities.csv
```

## Default Allocation Rule

The default method is `model_confidence`:

```text
size = prediction if prediction > 0.50 else 0
raw_weight = primary_signal * size * target_vol / annualized_ewma_vol
```

The risk constraints are hard-coded in `allocate_portfolio_weights.py`:

```text
target_vol = 0.10
probability_threshold = 0.50
max_abs_weight_per_instrument = 0.25
max_gross_exposure_per_day = 1.00
ewma_vol_span = 60
```

After volatility targeting, the script caps individual weights and scales down
each date if gross exposure exceeds the maximum.

Other sizing methods:

```powershell
python portfolio_weight_allocation_strategy\allocate_portfolio_weights.py --sizing-method all_or_nothing
python portfolio_weight_allocation_strategy\allocate_portfolio_weights.py --sizing-method ncdf
```
