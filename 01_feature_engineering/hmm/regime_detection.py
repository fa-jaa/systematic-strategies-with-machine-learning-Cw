"""HMM regime detection for the coursework.

Output:
    data/hmm/predictions/latent_regime_predictions.csv

The file is deliberately small: one row per primary-signal date/instrument,
with the HMM's predicted latent regime and regime probabilities. These are the
features to merge into the metamodel.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

INSTRUMENTS = [
    "es1s",
    "nq1s",
    "fesx1s",
    "cl1s",
    "ho1s",
    "rb1s",
    "ng1s",
    "gc1s",
    "si1s",
    "hg1s",
    "pl1s",
]


@dataclass(frozen=True)
class RegimeConfig:
    n_states: int = 4
    covariance_type: str = "diag"
    training_end: str = "2019-12-31"
    n_iter: int = 250
    tol: float = 1e-4
    random_state: int = 42


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ohlcv = pd.read_csv("data/raw/ohlcv_data.csv", parse_dates=["date"])
    signals = pd.read_csv("data/raw/primary_signals.csv", parse_dates=["date"])
    ohlcv["instrument"] = ohlcv["instrument"].str.lower()
    signals.columns = [c.lower() for c in signals.columns]
    return ohlcv.sort_values(["instrument", "date"]), signals.sort_values("date")


def build_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build the HMM observation vector for one instrument."""

    out = df.sort_values("date").copy()
    safe_volume = out["volume"].where(out["volume"] > 0)
    safe_open_interest = out["open_interest"].where(out["open_interest"] > 0)

    out["ret_1d"] = np.log(out["close"]).diff()
    out["intraday_ret"] = np.log(out["close"] / out["open"]).replace([np.inf, -np.inf], np.nan)
    out["range"] = np.log(out["high"] / out["low"]).replace([np.inf, -np.inf], np.nan)
    out["volume_chg"] = np.log(safe_volume).diff()
    out["open_interest_chg"] = np.log(safe_open_interest).diff()
    out["vol_20d"] = out["ret_1d"].rolling(20, min_periods=10).std()

    obs_cols = ["ret_1d", "intraday_ret", "range", "volume_chg", "open_interest_chg", "vol_20d"]
    out[obs_cols] = out[obs_cols].replace([np.inf, -np.inf], np.nan)
    return out, obs_cols


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    max_a = np.max(a, axis=axis, keepdims=True)
    out = max_a + np.log(np.sum(np.exp(a - max_a), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def filtered_probabilities(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """Compute xi(t, h) = P(H_t = h | x_1:t), without future leakage."""

    log_emit = model._compute_log_likelihood(x)
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)
    log_alpha = np.empty_like(log_emit)

    log_alpha[0] = log_start + log_emit[0]
    for t in range(1, len(x)):
        log_alpha[t] = log_emit[t] + _logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)

    normalizer = _logsumexp(log_alpha, axis=1)
    return np.exp(log_alpha - normalizer[:, None])


def order_states(model: GaussianHMM, obs_cols: list[str]) -> np.ndarray:
    """Make HMM state labels stable by sorting by return, then volatility."""

    return_idx = obs_cols.index("ret_1d")
    vol_idx = obs_cols.index("vol_20d")
    return np.lexsort((model.means_[:, vol_idx], model.means_[:, return_idx]))


def detect_regimes_for_instrument(
    ohlcv: pd.DataFrame,
    signals_long: pd.DataFrame,
    instrument: str,
    config: RegimeConfig,
) -> pd.DataFrame:
    instrument_ohlcv = ohlcv[ohlcv["instrument"] == instrument]
    observations, obs_cols = build_observations(instrument_ohlcv)

    train = observations[observations["date"] <= pd.Timestamp(config.training_end)].dropna(subset=obs_cols)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[obs_cols])

    model = GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        tol=config.tol,
        random_state=config.random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train)

    clean = observations.dropna(subset=obs_cols).copy()
    x_all = scaler.transform(clean[obs_cols])
    probs_raw = filtered_probabilities(model, x_all)
    order = order_states(model, obs_cols)
    probs = probs_raw[:, order]

    regimes = clean[["date", "instrument"]].copy()
    regimes["hmm_predicted_regime"] = probs.argmax(axis=1)
    regimes["hmm_regime_confidence"] = probs.max(axis=1)
    for state in range(config.n_states):
        regimes[f"hmm_regime_prob_{state}"] = probs[:, state]

    signal_slice = signals_long[signals_long["instrument"] == instrument]
    return signal_slice.merge(regimes, on=["date", "instrument"], how="left")


def run_regime_detection(config: RegimeConfig = RegimeConfig()) -> pd.DataFrame:
    ohlcv, signals = load_data()
    signals_long = signals.melt(id_vars="date", var_name="instrument", value_name="primary_signal")

    frames = [
        detect_regimes_for_instrument(ohlcv, signals_long, instrument, config)
        for instrument in INSTRUMENTS
    ]
    regimes = pd.concat(frames, ignore_index=True).sort_values(["date", "instrument"])
    regimes["hmm_predicted_regime"] = regimes["hmm_predicted_regime"].astype("Int64")

    output_dir = Path("data/hmm/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    regimes.to_csv(output_dir / "latent_regime_predictions.csv", index=False)
    return regimes


if __name__ == "__main__":
    result = run_regime_detection()
    print(
        "Wrote data/hmm/predictions/latent_regime_predictions.csv: "
        f"{result.shape[0]} rows x {result.shape[1]} columns"
    )
