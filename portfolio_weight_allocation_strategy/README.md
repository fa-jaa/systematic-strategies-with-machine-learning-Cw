# Portfolio Weight Allocation Strategy

This folder contains the optional portfolio-weight layer for the coursework.
It converts metamodel probabilities into signed strategy weights with target
volatility of `10%`.

The probability inputs use the coursework format:

```csv
date,instrument,prediction
2022-01-03,cl1s,0.9576887296513032
```

The weight outputs use:

```csv
date,instrument,weight
2022-01-03,cl1s,0.1234
```

## Inputs

Real inference probabilities:

```text
deliverables/final_predictions.csv
```

Real in-sample probabilities for the neural allocator:

```text
deliverables/insample_predicitons/insample_predictions.csv
```

The neural input is pre-2022 training data. The final prediction CSV is Jan-Jun
2022 inference data.

## Fixed Strategy

Run:

```powershell
python portfolio_weight_allocation_strategy\allocate_portfolio_weights.py
```

Strategy:

- primary signal supplies direction in `{-1, 0, +1}`
- metamodel probability supplies confidence
- probabilities at or below `0.50` are not traded
- position size uses model confidence
- weights are volatility targeted at `10%`
- each instrument is capped at `25%`
- total daily gross exposure is capped at `100%`

Formula:

```text
weight = primary_signal * probability_size * target_vol / ex_ante_vol
```

Outputs:

```text
portfolio_weight_allocation_strategy/outputs/strategy_weights.csv
portfolio_weight_allocation_strategy/outputs/allocation_diagnostics.csv
```

Clean deliverable copies:

```text
deliverables/portfolio_weight_outputs/fixed_strategy/strategy_weights.csv
deliverables/portfolio_weight_outputs/fixed_strategy/allocation_diagnostics.csv
```

## Neural Strategy

Run:

```powershell
python portfolio_weight_allocation_strategy\neural_portfolio_allocator.py
```

Strategy:

- input combines technical features, primary signal, metamodel probability, and
  instrument bias
- a lightweight neural head outputs conviction strength in `[0, 1]`
- probabilities at or below `0.50` are not traded
- conviction strength is multiplied by the primary signal and volatility targeting
- the model is trained on penalized negative Sharpe, equivalent to maximizing
  Sharpe after exposure and turnover costs
- validation early stopping is used to reduce overfitting
- the neural overlay is not allowed to flip the primary signal
- the same `10%` target volatility, `25%` instrument cap, and `100%` gross cap
  are applied

This is not a full TFT/VSN implementation. It is a dependency-light NumPy
baseline using:

```text
conviction_strength = tanh(x @ beta + ticker_bias) ** 2
```

Outputs:

```text
portfolio_weight_allocation_strategy/outputs/neural_portfolio/neural_strategy_weights.csv
portfolio_weight_allocation_strategy/outputs/neural_portfolio/neural_allocation_diagnostics.csv
portfolio_weight_allocation_strategy/outputs/neural_portfolio/neural_feature_weights.csv
portfolio_weight_allocation_strategy/outputs/neural_portfolio/neural_ticker_bias.csv
```

Clean deliverable copies:

```text
deliverables/portfolio_weight_outputs/neural_strategy/
```

## Comparison

Run:

```powershell
python portfolio_weight_allocation_strategy\visualise_strategy_weights.py --fixed-weights deliverables\portfolio_weight_outputs\fixed_strategy\strategy_weights.csv --neural-weights deliverables\portfolio_weight_outputs\neural_strategy\neural_strategy_weights.csv --output-dir deliverables\portfolio_weight_outputs\strategy_comparison
```

Main outputs:

```text
deliverables/portfolio_weight_outputs/strategy_comparison/strategy_summary.csv
deliverables/portfolio_weight_outputs/strategy_comparison/instrument_summary.csv
deliverables/portfolio_weight_outputs/strategy_comparison/daily_exposure_and_pnl.csv
deliverables/portfolio_weight_outputs/strategy_comparison/*.png
deliverables/portfolio_weight_outputs/strategy_comparison/strategy_comparison_explained.ipynb
```

Current result:

- fixed strategy annualized Sharpe: about `5.05`
- regularised neural strategy annualized Sharpe: about `5.13`

The regularised neural version fixed the previous overfitting symptoms: it now
uses the probability gate, avoids primary-signal flips, uses fewer features, and
stops on validation Sharpe. It slightly improves Sharpe versus fixed, while
fixed still has higher cumulative PnL because it takes larger positions.
