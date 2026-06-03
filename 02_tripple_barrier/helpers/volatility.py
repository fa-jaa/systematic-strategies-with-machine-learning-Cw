import numpy as np
import pandas as pd

def ewma_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """
    - Exponentially weighted moving average of daily log-return volatility.
    - This is the estimator recommended in Lopez de Prado (2018) for calibrating
    triple-barrier widths
    - Weights recent observations more heavily, making barriers adaptive to current market regimes
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    vol = log_returns.ewm(span=span, min_periods=span).std()
    return vol.reindex(close.index)


def rolling_daily_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """
    - Simple rolling standard deviation of daily log returns
    - Equal-weighted (more baseline)
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=window, min_periods=window).std()


def parkinson_daily_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    - Parkinson (1980) range-based volatility estimator.
    - Uses intraday high/low to estimate volatility (5x more efficinet)
    """
    log_hl = np.log(high / low)
    parkinson_sq = (log_hl ** 2) / (4 * np.log(2))
    return np.sqrt(parkinson_sq.rolling(window=window, min_periods=window).mean())


def garman_klass_daily_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    - Garman-Klass (1980) volatility estimator
    - Incorporates open, high, low, close for a more efficient estimate than either close-to-close or Parkinson alone.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(gk.rolling(window=window, min_periods=window).mean())
