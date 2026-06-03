"""
features.py  —  Systematic Trading Feature Library
====================================================

Standard OHLCV input:  date | instrument | open | high | low | close | volume
Every function returns: date | instrument | <feature columns>

Merge individual outputs on [date, instrument] to build a wide feature matrix,
or call build_all_features() to do it in one step.

TABLE OF CONTENTS
-----------------
  HELPERS
    _base                    select & sort OHLCV columns
    _log_ret                 add a log_ret column

  1  RETURNS & MOMENTUM
    returns                  pct and log returns across multiple horizons
    momentum_signs           direction (+1 / −1) of each return horizon
    rate_of_change           (close_t − close_{t−n}) / close_{t−n}
    return_spreads           cross-horizon return differences

  2  MOVING AVERAGES & MACD
    sma_vs_price             normalised distance of close from SMAs
    ema_vs_price             normalised distance of close from EMAs
    sma_crosses              relative spread between pairs of SMAs
    ema_crosses              relative spread between pairs of EMAs
    ma_slopes                1-day percentage change of a moving average
    macd                     MACD line, signal, histogram, z-score

  3  VOLATILITY
    rolling_volatility       annualised rolling close-to-close vol
    ewma_volatility          annualised EWMA vol (span = window)
    volatility_ratio         short-window vol / long-window vol
    parkinson_volatility     Parkinson (1980) high/low estimator
    garman_klass_volatility  Garman-Klass OHLC estimator

  4  PRICE STRUCTURE & ATR
    price_ranges             HL/CO ranges, true range, overnight/intraday returns
    atr                      Wilder ATR and ATR as % of close

  5  OSCILLATORS
    rsi                      Relative Strength Index (Wilder smoothing)
    stochastic               Stochastic %K and %D
    williams_r               Williams %R
    cci                      Commodity Channel Index
    mfi                      Money Flow Index (volume-weighted RSI on typical price)
    ultimate_oscillator      Williams Ultimate Oscillator (three-timeframe)

  6  BANDS & CHANNELS
    price_zscore             rolling z-score of close
    bollinger_bands          Bollinger position, width, and width z-score
    donchian_position        close position within rolling high/low channel

  BUILDER
    build_all_features       compute and merge every feature group into one DataFrame
"""

import numpy as np
import pandas as pd


# =============================================================================
# HELPERS
# =============================================================================

def _base(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Select OHLC columns and sort by [instrument, date]."""
    return (
        ohlcv[["date", "instrument", "open", "high", "low", "close"]]
        .sort_values(["instrument", "date"])
        .copy()
    )


def _log_ret(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_ret column to a df already sorted by [instrument, date]."""
    df = df.copy()
    df["log_ret"] = np.log(
        df["close"] / df.groupby("instrument")["close"].shift(1)
    )
    return df


# =============================================================================
# 1. RETURNS & MOMENTUM
# =============================================================================

def returns(
    ohlcv: pd.DataFrame,
    periods: list[int] = [1, 2, 3, 5, 10, 20, 63, 126, 252],
) -> pd.DataFrame:
    """
    Simple percentage and log returns across multiple horizons.

    Columns produced:
      log_ret    — 1-day log return: ln(close_t / close_{t-1})
      ret_{n}d   — n-day simple pct return for each n in periods

    Returns: date | instrument | log_ret | ret_{n}d ...
    """
    df = _base(ohlcv)
    g  = df.groupby("instrument")["close"]

    cols = {
        "log_ret": np.log(df["close"] / df.groupby("instrument")["close"].shift(1))
    }
    for n in periods:
        cols[f"ret_{n}d"] = g.transform(lambda s, n=n: s.pct_change(n))

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def momentum_signs(
    ohlcv: pd.DataFrame,
    periods: list[int] = [5, 20, 63, 126, 252],
) -> pd.DataFrame:
    """
    Binary return direction: +1 when positive, −1 otherwise, NaN when return is NaN.

    Returns: date | instrument | mom_sign_{n}d ...
    """
    df = _base(ohlcv)
    g  = df.groupby("instrument")["close"]

    cols = {}
    for n in periods:
        ret = g.transform(lambda s, n=n: s.pct_change(n))
        cols[f"mom_sign_{n}d"] = np.where(ret.isna(), np.nan, np.where(ret > 0, 1.0, -1.0))

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def rate_of_change(
    ohlcv: pd.DataFrame,
    periods: list[int] = [5, 20, 63],
) -> pd.DataFrame:
    """
    Rate of change: (close_t − close_{t−n}) / close_{t−n}.
    Equivalent to pct_change but computed explicitly.

    Returns: date | instrument | roc_{n}d ...
    """
    df = _base(ohlcv)
    g  = df.groupby("instrument")["close"]

    cols = {}
    for n in periods:
        lag = g.transform(lambda s, n=n: s.shift(n))
        cols[f"roc_{n}d"] = (df["close"] - lag) / lag.replace(0, np.nan)

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def return_spreads(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-horizon return differences — a simple trend-quality / term-structure proxy.

    Columns produced:
      ret_spread_5_20    — 5d return minus 20d return
      ret_spread_20_63   — 20d return minus 63d return

    Returns: date | instrument | ret_spread_5_20 | ret_spread_20_63
    """
    df  = _base(ohlcv)
    g   = df.groupby("instrument")["close"]
    r5  = g.transform(lambda s: s.pct_change(5))
    r20 = g.transform(lambda s: s.pct_change(20))
    r63 = g.transform(lambda s: s.pct_change(63))

    return (
        df[["date", "instrument"]]
        .assign(
            ret_spread_5_20  = r5  - r20,
            ret_spread_20_63 = r20 - r63,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 2. MOVING AVERAGES & MACD
# =============================================================================

def sma_vs_price(
    ohlcv: pd.DataFrame,
    periods: list[int] = [5, 10, 20, 50, 100, 200],
) -> pd.DataFrame:
    """
    Normalised distance of close from each SMA: (close − SMA_n) / SMA_n.
    Positive means price is trading above the average.

    Returns: date | instrument | price_vs_sma{n} ...
    """
    df = _base(ohlcv)
    g  = df.groupby("instrument")["close"]

    cols = {}
    for n in periods:
        sma = g.transform(lambda s, n=n: s.rolling(n).mean())
        cols[f"price_vs_sma{n}"] = (df["close"] - sma) / sma.replace(0, np.nan)

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def ema_vs_price(
    ohlcv: pd.DataFrame,
    periods: list[int] = [5, 10, 20, 50],
) -> pd.DataFrame:
    """
    Normalised distance of close from each EMA: (close − EMA_n) / EMA_n.

    Returns: date | instrument | price_vs_ema{n} ...
    """
    df = _base(ohlcv)
    g  = df.groupby("instrument")["close"]

    cols = {}
    for n in periods:
        ema = g.transform(lambda s, n=n: s.ewm(span=n, adjust=False).mean())
        cols[f"price_vs_ema{n}"] = (df["close"] - ema) / ema.replace(0, np.nan)

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def sma_crosses(
    ohlcv: pd.DataFrame,
    pairs: list[tuple] = [(5, 20), (10, 50), (20, 50), (50, 100), (50, 200), (100, 200)],
) -> pd.DataFrame:
    """
    Relative spread between two SMAs: (SMA_a − SMA_b) / SMA_b.
    Positive means the shorter MA is above the longer MA (bullish alignment).

    Returns: date | instrument | sma{a}_vs_sma{b} ...
    """
    df      = _base(ohlcv)
    g       = df.groupby("instrument")["close"]
    all_n   = sorted({n for pair in pairs for n in pair})
    smas    = {n: g.transform(lambda s, n=n: s.rolling(n).mean()) for n in all_n}

    cols = {
        f"sma{a}_vs_sma{b}": (smas[a] - smas[b]) / smas[b].replace(0, np.nan)
        for a, b in pairs
    }

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def ema_crosses(
    ohlcv: pd.DataFrame,
    pairs: list[tuple] = [(5, 20), (12, 26), (20, 50), (20, 100)],
) -> pd.DataFrame:
    """
    Relative spread between two EMAs: (EMA_a − EMA_b) / EMA_b.

    Returns: date | instrument | ema{a}_vs_ema{b} ...
    """
    df    = _base(ohlcv)
    g     = df.groupby("instrument")["close"]
    all_n = sorted({n for pair in pairs for n in pair})
    emas  = {n: g.transform(lambda s, n=n: s.ewm(span=n, adjust=False).mean()) for n in all_n}

    cols = {
        f"ema{a}_vs_ema{b}": (emas[a] - emas[b]) / emas[b].replace(0, np.nan)
        for a, b in pairs
    }

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def ma_slopes(
    ohlcv: pd.DataFrame,
    sma_periods: list[int] = [20, 50, 100],
    ema_periods: list[int] = [20, 50],
) -> pd.DataFrame:
    """
    1-day percentage change of a moving average: (MA_t − MA_{t−1}) / MA_{t−1}.
    Captures whether the trend is accelerating or decelerating.

    Returns: date | instrument | sma{n}_slope | ema{n}_slope ...
    """
    df = _base(ohlcv)
    g  = df.groupby("instrument")["close"]

    cols = {}
    for n in sma_periods:
        sma = g.transform(lambda s, n=n: s.rolling(n).mean())
        cols[f"sma{n}_slope"] = sma.groupby(df["instrument"]).transform(
            lambda s: s.diff(1) / s.shift(1)
        )

    for n in ema_periods:
        ema = g.transform(lambda s, n=n: s.ewm(span=n, adjust=False).mean())
        cols[f"ema{n}_slope"] = ema.groupby(df["instrument"]).transform(
            lambda s: s.diff(1) / s.shift(1)
        )

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def macd(
    ohlcv: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
    zscore_window: int = 63,
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    Columns produced:
      macd_line      — EMA(fast) − EMA(slow)
      macd_signal    — EMA(signal_span) of macd_line (Wilder's signal)
      macd_hist      — macd_line − macd_signal
      macd_hist_chg  — 1-day change in histogram (momentum of momentum)
      macd_zscore    — rolling z-score of macd_line over zscore_window bars

    Returns: date | instrument | macd_line | macd_signal | macd_hist | macd_hist_chg | macd_zscore
    """
    df = _base(ohlcv)
    g  = df.groupby("instrument")["close"]

    ema_fast  = g.transform(lambda s, f=fast: s.ewm(span=f, adjust=False).mean())
    ema_slow  = g.transform(lambda s, sl=slow: s.ewm(span=sl, adjust=False).mean())
    macd_line = ema_fast - ema_slow

    signal = macd_line.groupby(df["instrument"]).transform(
        lambda s, sp=signal_span: s.ewm(span=sp, adjust=False).mean()
    )
    hist = macd_line - signal

    rm = macd_line.groupby(df["instrument"]).transform(
        lambda s, w=zscore_window: s.rolling(w).mean()
    )
    rs = macd_line.groupby(df["instrument"]).transform(
        lambda s, w=zscore_window: s.rolling(w).std()
    )

    return (
        df[["date", "instrument"]]
        .assign(
            macd_line     = macd_line,
            macd_signal   = signal,
            macd_hist     = hist,
            macd_hist_chg = hist.groupby(df["instrument"]).transform(lambda s: s.diff(1)),
            macd_zscore   = (macd_line - rm) / rs.replace(0, np.nan),
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 3. VOLATILITY
# =============================================================================

def rolling_volatility(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Annualised rolling close-to-close volatility (std of log returns × √252).

    Returns: date | instrument | vol_{n}d  (one col per window)
    """
    df = _log_ret(_base(ohlcv))

    cols = {
        f"vol_{w}d": df.groupby("instrument")["log_ret"]
        .transform(lambda s, w=w: s.rolling(w).std() * np.sqrt(252))
        for w in windows
    }

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def ewma_volatility(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Annualised EWMA volatility (span = window, min_periods = window).

    Returns: date | instrument | ewma_vol_{n}d
    """
    df = _log_ret(_base(ohlcv))

    cols = {
        f"ewma_vol_{w}d": df.groupby("instrument")["log_ret"]
        .transform(lambda s, w=w: s.ewm(span=w, min_periods=w).std() * np.sqrt(252))
        for w in windows
    }

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def volatility_ratio(ohlcv: pd.DataFrame, pairs: list[tuple]) -> pd.DataFrame:
    """
    Ratio of short-window vol to long-window vol.
    Values > 1 mean recent volatility is elevated relative to the longer window.

    pairs : list of (short, long) tuples, e.g. [(5, 20), (20, 63), (63, 126)]

    Returns: date | instrument | vol_ratio_{a}_{b}d
    """
    all_windows = sorted({w for pair in pairs for w in pair})
    vols        = rolling_volatility(ohlcv, all_windows).set_index(["date", "instrument"])

    cols = {
        f"vol_ratio_{a}_{b}d": vols[f"vol_{a}d"] / vols[f"vol_{b}d"]
        for a, b in pairs
    }

    return vols.assign(**cols)[[*cols]].reset_index()


def parkinson_volatility(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Rolling Parkinson (1980) volatility — uses high/low only.
    σ = sqrt( 1 / (4·ln2) · mean( ln(H/L)² ) ) · sqrt(252)

    Returns: date | instrument | park_vol_{n}d
    """
    df = _base(ohlcv)
    df["hl2"] = np.log(df["high"] / df["low"]) ** 2
    factor    = 1.0 / (4.0 * np.log(2))

    cols = {
        f"park_vol_{w}d": df.groupby("instrument")["hl2"]
        .transform(lambda s, w=w: np.sqrt(factor * s.rolling(w).mean()) * np.sqrt(252))
        for w in windows
    }

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def garman_klass_volatility(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Rolling Garman-Klass volatility using O/H/L/C.

    Returns: date | instrument | gk_vol_{n}d
    """
    df    = _base(ohlcv)
    valid = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)

    df["gk_daily"] = np.nan
    df.loc[valid, "gk_daily"] = (
        0.5 * np.log(df.loc[valid, "high"] / df.loc[valid, "low"]) ** 2
        - (2 * np.log(2) - 1) * np.log(df.loc[valid, "close"] / df.loc[valid, "open"]) ** 2
    )

    cols = {}
    for w in windows:
        gk_var = df.groupby("instrument")["gk_daily"].transform(
            lambda s, w=w: s.rolling(w).mean()
        )
        # clip guards against rare negatives when a large gap open-to-close dominates the intraday range
        cols[f"gk_vol_{w}d"] = np.sqrt(gk_var.clip(lower=0) * 252)

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


# =============================================================================
# 4. PRICE STRUCTURE & ATR
# =============================================================================

def price_ranges(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Single-bar price structure features.

    Columns produced:
      hl_range      — high minus low
      co_range      — close minus open
      intraday_ret  — log(close / open)
      overnight_ret — log(open / prev_close)
      true_range    — max(H−L, |H−prev_C|, |L−prev_C|)

    Returns: date | instrument | hl_range | co_range | intraday_ret | overnight_ret | true_range
    """
    df         = _base(ohlcv)
    prev_close = df.groupby("instrument")["close"].shift(1)

    df["hl_range"] = df["high"] - df["low"]
    df["co_range"] = df["close"] - df["open"]

    df["intraday_ret"] = np.where(
        (df["close"] > 0) & (df["open"] > 0),
        np.log(df["close"] / df["open"]),
        np.nan,
    )

    df["overnight_ret"] = np.where(
        (df["open"] > 0) & (prev_close > 0),
        np.log(df["open"] / prev_close),
        np.nan,
    )

    df["true_range"] = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    cols = ["date", "instrument", "hl_range", "co_range",
            "intraday_ret", "overnight_ret", "true_range"]

    return df[cols].reset_index(drop=True)


def atr(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Wilder ATR and ATR normalised by close (Wilder smoothing: alpha = 1/n).

    Columns produced:
      atr_{n}d     — Wilder smoothed true range
      atr_{n}d_pct — ATR as a fraction of close

    Returns: date | instrument | atr_{n}d | atr_{n}d_pct ...
    """
    df = price_ranges(ohlcv)[["date", "instrument", "true_range"]].copy()

    close = (
        ohlcv[["date", "instrument", "close"]]
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )
    df = df.merge(close, on=["date", "instrument"], how="left")

    cols = {}
    for w in windows:
        atr_col       = f"atr_{w}d"
        cols[atr_col] = (
            df.groupby("instrument")["true_range"]
            .transform(lambda s, w=w: s.ewm(alpha=1 / w, min_periods=w, adjust=False).mean())
        )
        cols[f"{atr_col}_pct"] = cols[atr_col] / df["close"].replace(0, np.nan)

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


# =============================================================================
# 5. OSCILLATORS
# =============================================================================

def rsi(ohlcv: pd.DataFrame, windows: list[int], include_change: bool = True) -> pd.DataFrame:
    """
    Relative Strength Index using Wilder's smoothing (alpha = 1/window).

    Edge cases:
      RSI = 100 when avg_loss == 0 and avg_gain > 0 (no down days)
      RSI = NaN  when both avg_gain and avg_loss == 0 (completely flat price)

    Optionally appends rsi_{n}d_change: 1-day difference of the RSI (momentum of RSI).

    Returns: date | instrument | rsi_{n}d | [rsi_{n}d_change] ...
    """
    df   = _base(ohlcv)
    diff = df.groupby("instrument")["close"].diff()
    df["gain"] = diff.clip(lower=0)
    df["loss"] = (-diff).clip(lower=0)

    cols = {}
    for w in windows:
        alpha    = 1.0 / w
        avg_gain = df.groupby("instrument")["gain"].transform(
            lambda s, a=alpha, mp=w: s.ewm(alpha=a, min_periods=mp, adjust=False).mean()
        )
        avg_loss = df.groupby("instrument")["loss"].transform(
            lambda s, a=alpha, mp=w: s.ewm(alpha=a, min_periods=mp, adjust=False).mean()
        )

        rsi_col = pd.Series(
            np.where(
                avg_loss == 0,
                np.where(avg_gain > 0, 100.0, np.nan),
                100.0 - 100.0 / (1.0 + avg_gain / avg_loss),
            ),
            index=df.index,
        )
        cols[f"rsi_{w}d"] = rsi_col

        if include_change:
            cols[f"rsi_{w}d_change"] = rsi_col.groupby(df["instrument"]).transform(
                lambda s: s.diff()
            )

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def stochastic(ohlcv: pd.DataFrame, windows: list[int], d_smooth: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator %K and %D.
    %K = (close − lowest_low) / (highest_high − lowest_low) × 100
    %D = d_smooth-day SMA of %K

    windows    : lookback periods for %K, e.g. [14]
    d_smooth   : smoothing period for %D (default 3)

    Returns: date | instrument | stoch_k_{n}d | stoch_d_{n}d ...
    """
    df = _base(ohlcv)

    cols = {}
    for w in windows:
        low_min  = df.groupby("instrument")["low"].transform(lambda s, w=w: s.rolling(w).min())
        high_max = df.groupby("instrument")["high"].transform(lambda s, w=w: s.rolling(w).max())
        k = (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        cols[f"stoch_k_{w}d"] = k
        cols[f"stoch_d_{w}d"] = k.groupby(df["instrument"]).transform(
            lambda s: s.rolling(d_smooth).mean()
        )

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def williams_r(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Williams %R = (highest_high − close) / (highest_high − lowest_low) × −100
    Range: −100 (most oversold) to 0 (most overbought).

    Returns: date | instrument | williams_r_{n}d ...
    """
    df = _base(ohlcv)

    cols = {}
    for w in windows:
        high_max = df.groupby("instrument")["high"].transform(lambda s, w=w: s.rolling(w).max())
        low_min  = df.groupby("instrument")["low"].transform(lambda s, w=w: s.rolling(w).min())
        cols[f"williams_r_{w}d"] = (
            (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan) * -100
        )

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def cci(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Commodity Channel Index.
    typical_price = (H + L + C) / 3
    CCI = (TP − SMA(TP, n)) / (0.015 × MAD(TP, n))

    Returns: date | instrument | cci_{n}d ...
    """
    df       = _base(ohlcv)
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3

    cols = {}
    for w in windows:
        sma = df.groupby("instrument")["tp"].transform(lambda s, w=w: s.rolling(w).mean())
        mad = df.groupby("instrument")["tp"].transform(
            lambda s, w=w: s.rolling(w).apply(
                lambda x: np.abs(x - x.mean()).mean(), raw=True
            )
        )
        cols[f"cci_{w}d"] = (df["tp"] - sma) / (0.015 * mad.replace(0, np.nan))

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def mfi(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Money Flow Index — volume-weighted RSI on typical price.

    Edge cases match RSI:
      MFI = 100 when negative money flow == 0 and positive > 0
      MFI = NaN  when both flows are zero

    Returns: date | instrument | mfi_{n}d ...
    """
    df = _base(ohlcv).merge(
        ohlcv[["date", "instrument", "volume"]],
        on=["date", "instrument"],
        how="left",
    )

    df["tp"]     = (df["high"] + df["low"] + df["close"]) / 3
    df["mf"]     = df["tp"] * df["volume"]
    prev_tp      = df.groupby("instrument")["tp"].shift(1)
    df["pos_mf"] = np.where(df["tp"] > prev_tp, df["mf"], 0.0)
    df["neg_mf"] = np.where(df["tp"] < prev_tp, df["mf"], 0.0)

    cols = {}
    for w in windows:
        pos = df.groupby("instrument")["pos_mf"].transform(lambda s, w=w: s.rolling(w).sum())
        neg = df.groupby("instrument")["neg_mf"].transform(lambda s, w=w: s.rolling(w).sum())
        cols[f"mfi_{w}d"] = pd.Series(
            np.where(
                neg == 0,
                np.where(pos > 0, 100.0, np.nan),
                100.0 - 100.0 / (1.0 + pos / neg),
            ),
            index=df.index,
        )

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def ultimate_oscillator(
    ohlcv: pd.DataFrame,
    short: int = 7,
    mid: int = 14,
    long: int = 28,
) -> pd.DataFrame:
    """
    Ultimate Oscillator (Williams, 1985) — blends three timeframes.
    BP = close − min(low, prev_close)
    TR = max(high, prev_close) − min(low, prev_close)
    UO = 100 × (4·RA(short) + 2·RA(mid) + RA(long)) / 7
    where RA(n) = sum(BP, n) / sum(TR, n)

    Returns: date | instrument | uo
    """
    df         = _base(ohlcv)
    prev_close = df.groupby("instrument")["close"].shift(1)

    df["bp"] = df["close"] - pd.concat([df["low"], prev_close], axis=1).min(axis=1)
    df["tr"] = (
        pd.concat([df["high"], prev_close], axis=1).max(axis=1)
        - pd.concat([df["low"],  prev_close], axis=1).min(axis=1)
    )

    def _raw_avg(w):
        bp_sum = df.groupby("instrument")["bp"].transform(lambda s, w=w: s.rolling(w).sum())
        tr_sum = df.groupby("instrument")["tr"].transform(lambda s, w=w: s.rolling(w).sum())
        return bp_sum / tr_sum.replace(0, np.nan)

    uo = 100 * (4 * _raw_avg(short) + 2 * _raw_avg(mid) + _raw_avg(long)) / 7

    return df[["date", "instrument"]].assign(uo=uo).reset_index(drop=True)


# =============================================================================
# 6. BANDS & CHANNELS
# =============================================================================

def price_zscore(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Rolling z-score of close: (close − rolling_mean) / rolling_std.
    Measures how many standard deviations price sits above/below its recent mean.

    Returns: date | instrument | zscore_{n}d ...
    """
    df = _base(ohlcv)

    cols = {}
    for w in windows:
        mean = df.groupby("instrument")["close"].transform(lambda s, w=w: s.rolling(w).mean())
        std  = df.groupby("instrument")["close"].transform(lambda s, w=w: s.rolling(w).std())
        cols[f"zscore_{w}d"] = (df["close"] - mean) / std.replace(0, np.nan)

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


def bollinger_bands(ohlcv: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands (SMA ± n_std × rolling std).

    Columns produced:
      bb_upper_dist    — (upper − close) / close
      bb_lower_dist    — (close − lower) / close
      bb_position      — 0 = at lower band, 1 = at upper band
      bb_width         — (upper − lower) / mid
      bb_width_zscore  — rolling z-score of bb_width over the same window
                         (requires 2× window rows before first non-NaN)

    Returns: date | instrument | bb_upper_dist | bb_lower_dist | bb_position | bb_width | bb_width_zscore
    """
    df    = _base(ohlcv)
    mid   = df.groupby("instrument")["close"].transform(lambda s: s.rolling(window).mean())
    sigma = df.groupby("instrument")["close"].transform(lambda s: s.rolling(window).std())
    upper = mid + n_std * sigma
    lower = mid - n_std * sigma
    width = (upper - lower) / mid.replace(0, np.nan)

    # temporary frame avoids recomputing width inside a lambda
    _w     = pd.DataFrame({"instrument": df["instrument"], "width": width})
    w_mean = _w.groupby("instrument")["width"].transform(lambda s: s.rolling(window).mean())
    w_std  = _w.groupby("instrument")["width"].transform(lambda s: s.rolling(window).std())

    safe_close = df["close"].replace(0, np.nan)

    return (
        df[["date", "instrument"]]
        .assign(
            bb_upper_dist   = (upper - df["close"]) / safe_close,
            bb_lower_dist   = (df["close"] - lower) / safe_close,
            bb_position     = (df["close"] - lower) / (upper - lower).replace(0, np.nan),
            bb_width        = width,
            bb_width_zscore = (width - w_mean) / w_std.replace(0, np.nan),
        )
        .reset_index(drop=True)
    )


def donchian_position(ohlcv: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Position of close within the Donchian channel (rolling high/low range).
    0 = at the n-day low, 1 = at the n-day high.

    Returns: date | instrument | donchian_pos_{n}d ...
    """
    df = _base(ohlcv)

    cols = {}
    for w in windows:
        high_max = df.groupby("instrument")["high"].transform(lambda s, w=w: s.rolling(w).max())
        low_min  = df.groupby("instrument")["low"].transform(lambda s, w=w: s.rolling(w).min())
        cols[f"donchian_pos_{w}d"] = (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)

    return df[["date", "instrument"]].assign(**cols).reset_index(drop=True)


# =============================================================================
# BUILDER
# =============================================================================

def build_all_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every feature group and left-merge into a single wide DataFrame.

    Input must have columns: date | instrument | open | high | low | close | volume
    All functions use their default window/parameter settings.

    Returns a DataFrame sorted by [instrument, date] with inf values replaced by NaN.

    Approximate output: ~110 feature columns across ~26 feature groups.
    """
    from functools import reduce

    frames = [
        # ── Returns & Momentum ──────────────────────────────────────────────
        returns(ohlcv),
        momentum_signs(ohlcv),
        rate_of_change(ohlcv),
        return_spreads(ohlcv),
        # ── Moving Averages & MACD ───────────────────────────────────────────
        sma_vs_price(ohlcv),
        ema_vs_price(ohlcv),
        sma_crosses(ohlcv),
        ema_crosses(ohlcv),
        ma_slopes(ohlcv),
        macd(ohlcv),
        # ── Volatility ───────────────────────────────────────────────────────
        rolling_volatility(ohlcv,       [5, 10, 20, 63, 126, 252]),
        ewma_volatility(ohlcv,          [5, 10, 20, 63]),
        volatility_ratio(ohlcv,         [(5, 20), (20, 63), (63, 126)]),
        parkinson_volatility(ohlcv,     [5, 10, 20, 63]),
        garman_klass_volatility(ohlcv,  [5, 10, 20, 63]),
        # ── Price Structure & ATR ────────────────────────────────────────────
        price_ranges(ohlcv),
        atr(ohlcv,                      [5, 10, 14, 20]),
        # ── Oscillators ──────────────────────────────────────────────────────
        rsi(ohlcv,                      [7, 14, 21]),
        stochastic(ohlcv,               [14]),
        williams_r(ohlcv,               [14]),
        cci(ohlcv,                      [14, 20]),
        mfi(ohlcv,                      [14]),
        ultimate_oscillator(ohlcv),
        # ── Bands & Channels ─────────────────────────────────────────────────
        price_zscore(ohlcv,             [20, 63, 126]),
        bollinger_bands(ohlcv),
        donchian_position(ohlcv,        [20, 52, 252]),
    ]

    merged = reduce(
        lambda left, right: left.merge(right, on=["date", "instrument"], how="left"),
        frames,
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.sort_values(["instrument", "date"]).reset_index(drop=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ohlcv_data.csv"

    print(f"Loading {data_path} ...")
    ohlcv = pd.read_csv(data_path, parse_dates=["date"])
    print(f"  {len(ohlcv):,} rows, {ohlcv['instrument'].nunique()} instruments")

    print("\nComputing features ...")
    features = build_all_features(ohlcv)

    out_path = "data/features/technical_analysis_features.csv"
    features.to_csv(out_path, index=False)

    print(f"\n[done]  {len(features):,} rows × {len(features.columns)} columns")
    print(f"  Instruments : {sorted(features['instrument'].unique().tolist())}")
    print(f"  Date range  : {features['date'].min().date()} → {features['date'].max().date()}")
    print(f"  Saved       : {out_path}")
