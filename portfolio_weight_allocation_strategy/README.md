# Portfolio Weight Allocation Strategy

This folder contains the optional strategy-construction layer for the coursework.
It does not train the metamodel. It assumes a metamodel has already produced a
CSV of probabilities and converts those probabilities into signed portfolio
weights.

The design is intentionally simple and defensible: primary signals provide trade
direction, metamodel probabilities provide conviction, and EWMA volatility
targeting turns conviction into risk-scaled weights.

## Inputs

The allocator needs three inputs:

```text
data/raw/primary_signals.csv
data/raw/ohlcv_data.csv
portfolio_weight_allocation_strategy/probabilities/*.csv
```

The probability CSV must follow the coursework format:

```csv
date,instrument,prediction
2022-01-03,cl1s,0.9576887296513032
2022-01-03,rb1s,0.630837109352178
```

The probability file can be sparse: it may contain only rows where the primary
model has an active signal. This matches the current placeholder files and the
format expected from cleaned model outputs.

## PDF Guidance Used

From `Systematic_CW.pdf`:

- the optional strategy deliverable is a CSV with `date,instrument,weight`
- `weight` is signed: positive means long, negative means short
- report CAGR, annualised volatility, Sharpe, Sortino, maximum drawdown,
  average holding period, and turnover

From `BUSI70575_Optional_Session_3.pdf`:

- the primary signal supplies direction
- the metamodel probability supplies confidence
- calibrated probabilities are preferable before probability-based sizing
- fixed sizing choices include all-or-nothing, model-confidence, and NCDF sizing
- volatility targeting scales a prediction by `target_vol / ex_ante_vol`
- the weight at date `t` must use only information available up to `t`

Lecture 4 and Lecture 7 support the neural-network modelling side, especially
VSN/TFT ideas. This folder implements the simpler fixed allocation route so the
strategy can run immediately once probabilities are available.

## Strategy Overview

For each active metamodel probability row, the script builds:

```text
date, instrument, prediction, primary_signal, annualized_ewma_vol
```

Then it computes the final signed weight in four stages.

### 1. Direction

The primary signal determines direction:

```text
primary_signal = +1  -> long
primary_signal = -1  -> short
primary_signal =  0  -> no trade
```

The metamodel probability is direction-agnostic. It answers:

```text
How likely is the primary signal to be worth taking?
```

So direction and confidence are deliberately kept separate.

### 2. Probability Sizing

The default sizing method is `model_confidence`:

```text
size = prediction if prediction > 0.50 else 0
```

Example:

```text
prediction = 0.74 -> size = 0.74
prediction = 0.49 -> size = 0.00
```

Other implemented sizing methods:

```text
all_or_nothing: size = 1 if prediction > 0.50 else 0
ncdf:           smooth nonlinear sizing using a normal CDF transform
```

### 3. Volatility Targeting

The script estimates causal annualized EWMA volatility from daily close returns.
The default span is 60 trading days.

The raw weight is:

```text
raw_weight =
    primary_signal
    * size
    * target_vol
    / annualized_ewma_vol
```

With the hard-coded target volatility:

```text
target_vol = 0.10
```

This means high-volatility instruments receive smaller weights, while
low-volatility instruments can receive larger weights for the same model
confidence.

### 4. Exposure Controls

The strategy applies two caps:

```text
max_abs_weight_per_instrument = 0.25
max_gross_exposure_per_day = 1.00
```

First, each individual position is clipped to:

```text
-0.25 <= weight <= 0.25
```

Then, if the sum of absolute weights on a date exceeds 1.00, all weights on
that date are scaled down proportionally. This preserves relative conviction
while respecting the gross-exposure limit.

## Hard-Coded Constraints

These constants are fixed in `allocate_portfolio_weights.py`:

```python
PROBABILITY_THRESHOLD = 0.50
TARGET_VOL = 0.10
EWMA_SPAN = 60
MAX_ABS_WEIGHT = 0.25
MAX_GROSS_EXPOSURE = 1.00
```

They are intentionally hard-coded so the reported strategy is reproducible.

## Run Order

From the repository root:

```powershell
python portfolio_weight_allocation_strategy\generate_placeholder_probabilities.py
python portfolio_weight_allocation_strategy\allocate_portfolio_weights.py
```

The first command creates temporary probability CSVs:

```text
portfolio_weight_allocation_strategy/probabilities/placeholder_energy_active_055.csv
portfolio_weight_allocation_strategy/probabilities/placeholder_energy_neutral_050.csv
portfolio_weight_allocation_strategy/probabilities/placeholder_all_assets_active_055.csv
```

The second command creates:

```text
portfolio_weight_allocation_strategy/outputs/strategy_weights.csv
portfolio_weight_allocation_strategy/outputs/allocation_diagnostics.csv
```

## Replacing Placeholder Probabilities

When cleaned metamodel probabilities arrive, run:

```powershell
python portfolio_weight_allocation_strategy\allocate_portfolio_weights.py --probability-csv path\to\clean_probabilities.csv
```

The cleaned probability file should have:

```csv
date,instrument,prediction
2022-01-03,cl1s,0.74
```

## Outputs

The deliverable-style output is:

```text
portfolio_weight_allocation_strategy/outputs/strategy_weights.csv
```

with:

```csv
date,instrument,weight
2022-01-03,cl1s,0.1351027262564132
```

The debugging/reporting output is:

```text
portfolio_weight_allocation_strategy/outputs/allocation_diagnostics.csv
```

It contains:

```text
date
instrument
prediction
primary_signal
size
annualized_ewma_vol
raw_weight
capped_weight
weight
```

Use this file to explain how probabilities became weights.

## Suggested Report Framing

In the report, describe the strategy as:

```text
We use the metamodel as a probability-based sizing layer on top of the primary
signal. The primary signal determines direction, while the metamodel probability
determines conviction. Conviction is converted into a signed position using
causal EWMA volatility targeting with a 10% annualized target volatility. We cap
single-instrument weights at 25% and gross exposure at 100%.
```

Metrics to report:

- CAGR
- annualised volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- average holding period
- turnover

## Important Caveat

The placeholder probabilities are not model outputs. They exist only so the
allocation pipeline can be tested before cleaned probabilities arrive.

