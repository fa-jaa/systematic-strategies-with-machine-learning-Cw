# Portfolio Weight Outputs

This folder holds the clean portfolio-weight deliverables generated from the
real probability files.

## Source Inputs

```text
deliverables/final_predictions.csv
deliverables/insample_predicitons/insample_predictions.csv
```

`final_predictions.csv` is used for Jan-Jun 2022 inference. The in-sample file
is used only to train the neural allocation head.

## Folder Contents

```text
fixed_strategy/
neural_strategy/
strategy_comparison/
```

## Fixed Strategy

```text
fixed_strategy/strategy_weights.csv
fixed_strategy/allocation_diagnostics.csv
```

This is the cleaner deliverable candidate. It uses primary-signal direction,
metamodel probability confidence, `10%` target volatility, `25%` per-instrument
cap, and `100%` max daily gross exposure.

## Neural Strategy

```text
neural_strategy/neural_strategy_weights.csv
neural_strategy/neural_allocation_diagnostics.csv
neural_strategy/neural_feature_weights.csv
neural_strategy/neural_ticker_bias.csv
```

This is the experimental optional-session style allocator. It trains a
regularised lightweight neural conviction head with a penalized negative-Sharpe
objective, then turns convictions into volatility-targeted weights. The current
version uses validation early stopping, a probability gate, no primary-signal
flips, fewer features, stronger L2 regularisation, and exposure/turnover
penalties.

## Comparison

```text
strategy_comparison/strategy_summary.csv
strategy_comparison/instrument_summary.csv
strategy_comparison/daily_exposure_and_pnl.csv
strategy_comparison/strategy_comparison_explained.ipynb
strategy_comparison/*.png
```

Current headline:

- fixed strategy annualized Sharpe: about `5.05`
- regularised neural strategy annualized Sharpe: about `5.13`

The notebook explains the comparison step by step. In short, the regularised
neural allocator fixed the old overfitting problem and now slightly beats fixed
on Sharpe, while fixed still has higher cumulative PnL because it sizes accepted
trades more aggressively.
