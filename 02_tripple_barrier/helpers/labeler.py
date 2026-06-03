"""
labeler.py — Triple Barrier labeling engine.

Core logic: for each entry event, scan the forward price path and record
which of the three barriers (take-profit, stop-loss, vertical time limit)
is touched first.  The first-touch determines the label.

Label convention
----------------
 +1  : take-profit barrier hit first      (profitable trade)
 -1  : stop-loss barrier hit first        (loss trade)
  0  : vertical barrier hit first without either horizontal barrier touched
       (label = sign of return over the holding period, or 0 if flat)
"""

import numpy as np
import pandas as pd


def _barrier_touch_date(
    path: pd.Series,
    pt: float,
    sl: float,
    side: float,
) -> tuple[pd.Timestamp | None, int]:
    """
    Scan a single price path for the first barrier touch.

    Parameters
    ----------
    path : pd.Series  — prices from entry+1 up to and including t1.
    pt   : float      — take-profit price level (NaN = disabled).
    sl   : float      — stop-loss price level   (NaN = disabled).
    side : float      — +1 for long, -1 for short.

    Returns
    -------
    (touch_date, label)
        touch_date : first date a barrier was breached, or None if time-out.
        label      : +1, -1, or 0.
    """
    for date, price in path.items():
        # Take-profit: for long, price >= pt; for short, price <= pt
        if not np.isnan(pt):
            if side >= 0 and price >= pt:
                return date, 1
            if side < 0 and price <= pt:
                return date, 1
        # Stop-loss: for long, price <= sl; for short, price >= sl
        if not np.isnan(sl):
            if side >= 0 and price <= sl:
                return date, -1
            if side < 0 and price >= sl:
                return date, -1
    return None, 0


def label_events(
    close: pd.Series,
    events: pd.DataFrame,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    Apply the triple barrier method to every row in `events`.

    Parameters
    ----------
    close    : pd.Series   — full daily closing price series.
    events   : pd.DataFrame — output of `barriers.barrier_levels()`, with
                              columns [t1, pt, sl, vol, side, entry_price].
    min_ret  : float        — minimum absolute return required to keep an event
                              as a labelled observation.  Events whose maximum
                              potential return (based on vol) is below this
                              threshold are dropped.  Default 0 = keep all.

    Returns
    -------
    pd.DataFrame with columns:
        t1         : vertical barrier date
        ret        : log return from entry to first barrier touch (or t1)
        label      : +1, -1, 0
        touch_date : date of first barrier touch (NaT if time-out)
        pt, sl     : barrier levels
        side       : signal direction
        entry_price: closing price at entry
    """
    out = events.copy()
    out["label"] = 0
    out["touch_date"] = pd.NaT
    out["ret"] = np.nan

    for t0, row in events.iterrows():
        t1 = row["t1"]
        pt = row["pt"]
        sl = row["sl"]
        side = row["side"]
        entry_price = row["entry_price"]

        # Minimum-return filter: skip event if vol is too low
        if min_ret > 0:
            max_possible_ret = row["vol"] * max(
                abs(row.get("pt_multiplier", 1.0)),
                abs(row.get("sl_multiplier", 1.0)),
            )
            if max_possible_ret < min_ret:
                out.drop(index=t0, inplace=True)
                continue

        # Slice price path: from day after entry up to (and including) t1
        path = close.loc[t0:t1].iloc[1:]  # exclude entry bar itself

        if path.empty:
            continue

        touch_date, label = _barrier_touch_date(path, pt, sl, side)

        if touch_date is not None:
            exit_price = close.loc[touch_date]
        else:
            # Time-out: use price at vertical barrier
            touch_date = t1
            exit_price = close.loc[t1]
            # Label by sign of return if no horizontal barrier was hit
            raw_ret = np.log(exit_price / entry_price) * side
            label = int(np.sign(raw_ret))

        ret = np.log(exit_price / entry_price) * side
        out.loc[t0, "ret"] = ret
        out.loc[t0, "label"] = label
        out.loc[t0, "touch_date"] = touch_date

    return out


def get_labels(
    close: pd.Series,
    signals: pd.Series,
    daily_vol: pd.Series,
    pt_multiplier: float = 1.0,
    sl_multiplier: float = 1.0,
    num_days: int = 10,
    min_ret: float = 0.0,
    vol_estimator: str = "ewma",
    vol_span: int = 100,
) -> pd.DataFrame:
    """
    End-to-end convenience function: compute labels from a signal series.

    This is the main entry point for the pipeline.  It calls
    `volatility`, `barriers`, and `label_events` internally.

    Parameters
    ----------
    close          : pd.Series  — daily closing prices.
    signals        : pd.Series  — primary model signal {-1, 0, +1}, indexed by date.
                                   Only non-zero dates become events.
    daily_vol      : pd.Series  — pre-computed daily volatility series.
    pt_multiplier  : float      — take-profit barrier width in units of daily vol.
    sl_multiplier  : float      — stop-loss barrier width in units of daily vol.
    num_days       : int        — vertical barrier (max holding period in trading days).
    min_ret        : float      — minimum vol-scaled return to include an event.

    Returns
    -------
    pd.DataFrame — labelled events.  See `label_events` for column details.
    """
    from barriers import barrier_levels

    # Extract non-zero signal dates that fall within the price history
    t_events = signals[signals != 0].index
    t_events = t_events[t_events.isin(close.index)]

    side = signals[signals != 0].reindex(t_events).astype(float)

    events = barrier_levels(
        close=close,
        t_events=t_events,
        daily_vol=daily_vol,
        pt_multiplier=pt_multiplier,
        sl_multiplier=sl_multiplier,
        num_days=num_days,
        side=side,
    )

    labels = label_events(close=close, events=events, min_ret=min_ret)
    return labels


def label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Print a summary of the label distribution and average returns.

    Parameters
    ----------
    labels : pd.DataFrame — output of `get_labels` or `label_events`.

    Returns
    -------
    pd.DataFrame  — summary table with counts, proportions, and mean return.
    """
    summary = (
        labels.groupby("label")
        .agg(
            count=("ret", "count"),
            mean_ret=("ret", "mean"),
            std_ret=("ret", "std"),
        )
        .rename(index={-1: "Stop-Loss (-1)", 0: "Time-Out (0)", 1: "Take-Profit (+1)"})
    )
    summary["proportion"] = summary["count"] / summary["count"].sum()
    return summary
