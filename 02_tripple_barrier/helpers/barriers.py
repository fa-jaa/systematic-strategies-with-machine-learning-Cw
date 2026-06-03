# Computes the three barrier levels (take-profit, stop-loss, vertical time limit) for each event.
import numpy as np
import pandas as pd


def vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_days: int = 10, # Days going forward
) -> pd.Series:
    """
    Compute the vertical (time) barrier for each event.

    Looks up the trading date that is exactly `num_days` business days after
    each event, capped at the last available price date.

    Parameters
    ----------
    t_events : pd.DatetimeIndex  — event entry dates.
    close    : pd.Series         — closing prices (used only for its index).
    num_days : int               — maximum holding period in trading days.

    Returns
    -------
    pd.Series  — index: t_events,  values: vertical barrier dates.
    """
    price_dates = close.index
    # For each event, find the index position in price_dates then step forward
    t1 = []
    for t in t_events:
        pos = price_dates.searchsorted(t)
        target_pos = pos + num_days
        if target_pos >= len(price_dates):
            target_pos = len(price_dates) - 1
        t1.append(price_dates[target_pos])
    return pd.Series(t1, index=t_events, name="t1")


def barrier_levels(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    daily_vol: pd.Series,
    pt_multiplier: float = 1.0,
    sl_multiplier: float = 1.0,
    num_days: int = 10,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute take-profit (pt), stop-loss (sl), and vertical (t1) barriers.

    Barriers are centred on the entry close price and scaled by the daily
    volatility at the event date.

    Parameters
    ----------
    close          : pd.Series          — daily closing prices.
    t_events       : pd.DatetimeIndex   — entry event dates (subset of close.index).
    daily_vol      : pd.Series          — daily volatility aligned to close.index.
    pt_multiplier  : float              — number of daily-σ for take-profit barrier.
                                          Set to 0 to disable upper barrier.
    sl_multiplier  : float              — number of daily-σ for stop-loss barrier.
                                          Set to 0 to disable lower barrier.
    num_days       : int                — vertical barrier in trading days.
    side           : pd.Series | None   — optional signal direction {-1, +1}.
                                          When provided, barriers are directional:
                                          a long (+1) entry has pt above and sl below;
                                          a short (-1) entry has sl above and pt below.
                                          When None, symmetric barriers are assumed.

    Returns
    -------
    pd.DataFrame with columns:
        t1   : vertical barrier date
        pt   : take-profit price level  (NaN if pt_multiplier == 0)
        sl   : stop-loss price level    (NaN if sl_multiplier == 0)
        vol  : daily volatility at entry
        side : signal direction (+1 / -1 / NaN)
    """
    events = pd.DataFrame(index=t_events)
    events["t1"] = vertical_barrier(t_events, close, num_days)
    events["vol"] = daily_vol.reindex(t_events)
    events["entry_price"] = close.reindex(t_events)

    if side is not None:
        events["side"] = side.reindex(t_events)
    else:
        events["side"] = 1.0  # default: long

    # Directional barriers
    # For a long (+1):  pt = entry * (1 + pt_mult * vol),  sl = entry * (1 - sl_mult * vol)
    # For a short (-1): pt = entry * (1 - pt_mult * vol),  sl = entry * (1 + sl_mult * vol)
    if pt_multiplier > 0:
        events["pt"] = events["entry_price"] * (
            1 + events["side"] * pt_multiplier * events["vol"]
        )
    else:
        events["pt"] = np.nan

    if sl_multiplier > 0:
        events["sl"] = events["entry_price"] * (
            1 - events["side"] * sl_multiplier * events["vol"]
        )
    else:
        events["sl"] = np.nan

    return events[["t1", "pt", "sl", "vol", "side", "entry_price"]]


def filter_events_by_cusum(
    close: pd.Series,
    daily_vol: pd.Series,
    threshold_multiplier: float = 1.0,
) -> pd.DatetimeIndex:
    """
    CUSUM filter: generate event dates when cumulative absolute return exceeds
    a volatility-scaled threshold.

    Useful for generating events endogenously from price data rather than
    relying solely on external signals.  From Lopez de Prado (2018), ch.2.

    Parameters
    ----------
    close                : pd.Series — closing prices.
    daily_vol            : pd.Series — daily volatility aligned to close.index.
    threshold_multiplier : float     — scale factor applied to daily_vol as the
                                       trigger threshold.  Higher = fewer events.

    Returns
    -------
    pd.DatetimeIndex  — dates that passed the CUSUM filter.
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    threshold = (daily_vol * threshold_multiplier).reindex(log_returns.index)

    t_events = []
    s_pos = 0.0
    s_neg = 0.0

    for date, ret in log_returns.items():
        h = threshold.loc[date]
        if np.isnan(h):
            continue
        s_pos = max(0, s_pos + ret)
        s_neg = min(0, s_neg + ret)
        if s_pos >= h:
            s_pos = 0
            t_events.append(date)
        elif s_neg <= -h:
            s_neg = 0
            t_events.append(date)

    return pd.DatetimeIndex(t_events)
