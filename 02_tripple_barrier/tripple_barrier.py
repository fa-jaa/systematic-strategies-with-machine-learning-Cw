"""Importable triple-barrier labelling utilities.

Purpose
-------
This file contains the tested triple-barrier notebook logic in normal Python
form. Use it when CPCV, model-training, or hyperparameter-search code needs to
generate labels repeatedly with different volatility methods, barrier widths,
or timeout lengths.

What the main pipeline does
---------------------------
`run_triple_barrier_pipeline(...)`:
- loads raw OHLCV data and primary signals
- keeps training data only by default: dates before 2022-01-01
- ignores `primary_signal == 0` rows before labelling
- builds take-profit, stop-loss, and timeout barriers
- applies triple-barrier labels and metalabels
- returns a modelling/debug dataframe with 25 columns
- optionally saves the dataframe to CSV if `output_path` is passed

How to import it
----------------
Because the folder name starts with `02_`, this is not valid Python:

    from 02_tripple_barrier.tripple_barrier import run_triple_barrier_pipeline

Instead, add the folder to `sys.path` first:

    import sys
    sys.path.append("02_tripple_barrier")

    from tripple_barrier import run_triple_barrier_pipeline

Basic use
---------
    labels = run_triple_barrier_pipeline()

This returns CL1S labels using:
- `training_end="2022-01-01"`
- `volatility_method="ewma"`
- `ewma_span=100`
- `num_days=10`
- `take_profit_mult=2.0`
- `stop_loss_mult=2.0`

Testing different configurations
--------------------------------
    labels = run_triple_barrier_pipeline(
        instrument="cl1s",
        volatility_method="garman_klass",
        volatility_window=20,
        num_days=5,
        take_profit_mult=2.5,
        stop_loss_mult=2.0,
    )

Supported volatility methods are:
- `"ewma"`
- `"rolling"`
- `"parkinson"`
- `"garman_klass"`
- `"garman-klass"`
- `"gk"`

Saving labels
-------------
By default the function does not write files, which is better for CPCV loops.
Pass `output_path` only when you want a CSV:

    labels = run_triple_barrier_pipeline(
        output_path="data/features/tripple_barrier/cl1s_labels.csv"
    )

Important output columns
------------------------
- `date`, `instrument`, `primary_signal`
- `vol`, `tp`, `sl`, `timeout_date`, `timeout_close`
- `touch_date`, `touch_price`, `touched_barrier`
- `triple_barrier_label`, `metalabel`
- `volatility_method`, `num_days`, `take_profit_mult`, `stop_loss_mult`
- `holding_period_days`, `raw_touch_return`, `signed_touch_return`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OHLCV_PATH = PROJECT_ROOT / "data" / "raw" / "ohlcv_data.csv"
DEFAULT_SIGNALS_PATH = PROJECT_ROOT / "data" / "raw" / "primary_signals.csv"


def ewma_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Exponentially weighted daily log-return volatility."""

    log_returns = np.log(close / close.shift(1)).dropna()
    vol = log_returns.ewm(span=span, min_periods=span).std()
    return vol.reindex(close.index)


def rolling_daily_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of daily log returns."""

    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=window, min_periods=window).std()


def parkinson_daily_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson high/low range volatility estimator."""

    log_hl = np.log(high / low)
    parkinson_sq = (log_hl**2) / (4 * np.log(2))
    return np.sqrt(parkinson_sq.rolling(window=window, min_periods=window).mean())


def garman_klass_daily_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Garman-Klass open/high/low/close volatility estimator."""

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(gk.rolling(window=window, min_periods=window).mean())


def create_barriers(
    close: pd.Series,
    signal: pd.Series,
    vol: pd.Series,
    num_days: int = 10,
    take_profit_mult: float = 2.0,
    stop_loss_mult: float = 2.0,
) -> pd.DataFrame:
    """Create take-profit, stop-loss, and timeout barriers for active signals."""

    close = close.sort_index()
    signal = signal.sort_index()
    vol = vol.sort_index()

    df = pd.DataFrame(
        {
            "close": close,
            "primary_signal": signal,
            "vol": vol,
        }
    )
    df = df.dropna(subset=["close", "primary_signal", "vol"]).copy()
    df["primary_signal"] = df["primary_signal"].astype(int)
    df = df[df["primary_signal"].isin([-1, 1])].copy()

    price_series = close.dropna().sort_index()
    price_index = price_series.index
    event_positions = price_index.get_indexer(df.index)
    timeout_positions = event_positions + num_days
    valid_timeout = (event_positions >= 0) & (timeout_positions < len(price_index))

    df = df.loc[valid_timeout].copy()
    timeout_positions = timeout_positions[valid_timeout]
    df["timeout_date"] = price_index[timeout_positions]
    df["timeout_close"] = price_series.iloc[timeout_positions].values

    barrier_side = df["primary_signal"]
    df["tp"] = df["close"] * (1 + barrier_side * take_profit_mult * df["vol"])
    df["sl"] = df["close"] * (1 - barrier_side * stop_loss_mult * df["vol"])

    df = df.reset_index().rename(columns={"index": "date"})
    return df[
        [
            "date",
            "close",
            "primary_signal",
            "vol",
            "tp",
            "sl",
            "timeout_date",
            "timeout_close",
        ]
    ]


def apply_triple_barrier_labels(
    barriers: pd.DataFrame,
    close: pd.Series,
) -> pd.DataFrame:
    """Apply triple-barrier labels and metalabels to a barrier table."""

    close = close.dropna().sort_index()
    labelled = barriers.copy()

    triple_barrier_labels = []
    touch_dates = []
    touch_prices = []
    touched_barriers = []

    for _, row in labelled.iterrows():
        start_date = row["date"]
        timeout_date = row["timeout_date"]
        upper_barrier = max(row["tp"], row["sl"])
        lower_barrier = min(row["tp"], row["sl"])

        price_path = close.loc[(close.index > start_date) & (close.index <= timeout_date)]

        label = 0
        touch_date = timeout_date
        touch_price = row.get("timeout_close", np.nan)
        touched_barrier = "timeout"

        for current_date, current_close in price_path.items():
            if current_close >= upper_barrier:
                label = 1
                touch_date = current_date
                touch_price = current_close
                touched_barrier = "upper"
                break

            if current_close <= lower_barrier:
                label = -1
                touch_date = current_date
                touch_price = current_close
                touched_barrier = "lower"
                break

        triple_barrier_labels.append(label)
        touch_dates.append(touch_date)
        touch_prices.append(touch_price)
        touched_barriers.append(touched_barrier)

    labelled["triple_barrier_label"] = triple_barrier_labels
    labelled["touch_date"] = touch_dates
    labelled["touch_price"] = touch_prices
    labelled["touched_barrier"] = touched_barriers

    labelled["metalabel"] = 0
    correct = labelled["primary_signal"] == labelled["triple_barrier_label"]
    labelled.loc[correct, "metalabel"] = 1

    return labelled


def run_triple_barrier_pipeline(
    instrument: str = "cl1s",
    ohlcv_path: str | Path | None = None,
    signals_path: str | Path | None = None,
    signal_column: str | None = None,
    training_end: str | pd.Timestamp | None = "2022-01-01",
    volatility_method: str = "ewma",
    ewma_span: int = 100,
    volatility_window: int = 20,
    num_days: int = 10,
    take_profit_mult: float = 2.0,
    stop_loss_mult: float = 2.0,
    output_path: str | Path | None = None,
    save_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full triple-barrier labelling workflow for one instrument."""

    signal_column = instrument if signal_column is None else signal_column
    ohlcv_path = DEFAULT_OHLCV_PATH if ohlcv_path is None else Path(ohlcv_path)
    signals_path = DEFAULT_SIGNALS_PATH if signals_path is None else Path(signals_path)

    ohlcv_data = pd.read_csv(ohlcv_path, parse_dates=["date"]).set_index("date")
    signals_data = pd.read_csv(signals_path, parse_dates=["date"]).set_index("date")

    training_end_timestamp = None if training_end is None else pd.Timestamp(training_end)
    if training_end_timestamp is not None:
        ohlcv_data = ohlcv_data.loc[ohlcv_data.index < training_end_timestamp].copy()
        signals_data = signals_data.loc[signals_data.index < training_end_timestamp].copy()

    instrument_data = ohlcv_data[ohlcv_data["instrument"] == instrument].sort_index().copy()
    if instrument_data.empty:
        raise ValueError(
            f"No OHLCV rows found for instrument={instrument!r} "
            f"before training_end={training_end!r}"
        )
    if signal_column not in signals_data.columns:
        raise ValueError(f"Signal column {signal_column!r} not found in {signals_path}")

    close_series = instrument_data["close"]
    high_series = instrument_data["high"]
    low_series = instrument_data["low"]
    open_series = instrument_data["open"]
    signal_series = signals_data[signal_column]
    active_signal_series = signal_series[signal_series.isin([-1, 1])]

    volatility_method_key = volatility_method.lower().replace("-", "_")
    if volatility_method_key == "ewma":
        vol_series = ewma_daily_vol(close_series, span=ewma_span)
        volatility_method_label = "ewma"
    elif volatility_method_key == "rolling":
        vol_series = rolling_daily_vol(close_series, window=volatility_window)
        volatility_method_label = "rolling"
    elif volatility_method_key == "parkinson":
        vol_series = parkinson_daily_vol(high_series, low_series, window=volatility_window)
        volatility_method_label = "parkinson"
    elif volatility_method_key in {"garman_klass", "gk"}:
        vol_series = garman_klass_daily_vol(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            window=volatility_window,
        )
        volatility_method_label = "garman_klass"
    else:
        raise ValueError(
            "volatility_method must be one of 'ewma', 'rolling', 'parkinson', "
            "'garman_klass', 'garman-klass', or 'gk'"
        )

    barrier_table = create_barriers(
        close=close_series,
        signal=active_signal_series,
        vol=vol_series,
        num_days=num_days,
        take_profit_mult=take_profit_mult,
        stop_loss_mult=stop_loss_mult,
    )
    labelled = apply_triple_barrier_labels(
        barriers=barrier_table,
        close=close_series,
    )

    labelled.insert(0, "instrument", instrument)
    labelled.insert(1, "signal_column", signal_column)
    labelled["training_end"] = training_end_timestamp
    labelled["volatility_method"] = volatility_method_label
    labelled["ewma_span"] = ewma_span
    labelled["volatility_window"] = volatility_window
    labelled["num_days"] = num_days
    labelled["take_profit_mult"] = take_profit_mult
    labelled["stop_loss_mult"] = stop_loss_mult

    labelled["holding_period_days"] = (
        pd.to_datetime(labelled["touch_date"]) - pd.to_datetime(labelled["date"])
    ).dt.days
    labelled["raw_touch_return"] = (labelled["touch_price"] / labelled["close"]) - 1
    labelled["signed_touch_return"] = labelled["raw_touch_return"] * labelled["primary_signal"]

    output_columns = [
        "instrument",
        "signal_column",
        "training_end",
        "date",
        "close",
        "primary_signal",
        "vol",
        "tp",
        "sl",
        "timeout_date",
        "timeout_close",
        "touch_date",
        "touch_price",
        "touched_barrier",
        "triple_barrier_label",
        "metalabel",
        "volatility_method",
        "ewma_span",
        "volatility_window",
        "num_days",
        "take_profit_mult",
        "stop_loss_mult",
        "holding_period_days",
        "raw_touch_return",
        "signed_touch_return",
    ]
    labelled = labelled[output_columns]

    if output_path is not None:
        columns_to_save = output_columns if save_columns is None else save_columns
        missing_columns = sorted(set(columns_to_save) - set(labelled.columns))
        if missing_columns:
            raise ValueError(f"save_columns contains columns not in output: {missing_columns}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labelled[columns_to_save].to_csv(output_path, index=False)

    return labelled


__all__ = [
    "apply_triple_barrier_labels",
    "create_barriers",
    "ewma_daily_vol",
    "garman_klass_daily_vol",
    "parkinson_daily_vol",
    "rolling_daily_vol",
    "run_triple_barrier_pipeline",
]
