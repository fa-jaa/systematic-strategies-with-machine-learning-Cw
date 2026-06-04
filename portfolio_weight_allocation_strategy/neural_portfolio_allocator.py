"""Neural portfolio construction trained by maximizing Sharpe.

This is the advanced strategy route described in the optional session:

    features + primary_signal + metamodel_probability
        -> neural conviction in [-1, 1]
        -> volatility-targeted weight
        -> portfolio return path
        -> train on negative Sharpe

To keep the script dependency-light, the model is a small NumPy neural head:

    conviction = tanh(x @ beta + ticker_bias)

It is not a full TFT/VSN implementation, but it implements the same portfolio
training objective and inference mechanics from the slides. If the team later
wants a deeper sequence model, this file is the clean baseline to replace.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ROOT = Path(__file__).resolve().parent
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "features" / "merged_feature_matrix.csv"
OHLCV_PATH = PROJECT_ROOT / "data" / "raw" / "ohlcv_data.csv"
PRIMARY_SIGNALS_PATH = PROJECT_ROOT / "data" / "raw" / "primary_signals.csv"
DEFAULT_TRAIN_PROBABILITY_CSV = STRATEGY_ROOT / "probabilities" / "placeholder_energy_train_active_055.csv"
DEFAULT_INFERENCE_PROBABILITY_CSV = STRATEGY_ROOT / "probabilities" / "placeholder_energy_active_055.csv"
DEFAULT_OUTPUT_DIR = STRATEGY_ROOT / "outputs" / "neural_portfolio"

ENERGY_TICKERS = ["cl1s", "ho1s", "rb1s", "ng1s"]
TRAIN_END = pd.Timestamp("2022-01-01")
INFERENCE_START = pd.Timestamp("2022-01-01")
INFERENCE_END = pd.Timestamp("2022-06-30")

# Hard-coded portfolio policy, matching the fixed allocator.
TARGET_VOL = 0.10
EWMA_SPAN = 60
MAX_ABS_WEIGHT = 0.25
MAX_GROSS_EXPOSURE = 1.00
ANNUALIZATION = np.sqrt(252.0)


@dataclass
class NeuralPortfolioState:
    feature_cols: list[str]
    instruments: list[str]
    feature_mean: pd.Series
    feature_std: pd.Series
    beta: np.ndarray
    ticker_bias: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Sharpe-optimized neural portfolio head.")
    parser.add_argument("--train-probability-csv", type=Path, default=DEFAULT_TRAIN_PROBABILITY_CSV)
    parser.add_argument("--inference-probability-csv", type=Path, default=DEFAULT_INFERENCE_PROBABILITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_primary_long() -> pd.DataFrame:
    """Read primary signals in long form for direction."""

    signals = pd.read_csv(PRIMARY_SIGNALS_PATH, parse_dates=["date"])
    signals.columns = [col.lower() for col in signals.columns]
    long = signals.melt(id_vars="date", var_name="instrument", value_name="primary_signal")
    long["instrument"] = long["instrument"].str.lower()
    return long


def load_probabilities(path: Path) -> pd.DataFrame:
    """Load sparse active-row probabilities in the coursework format."""

    probs = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "instrument", "prediction"}
    missing = required - set(probs.columns)
    if missing:
        raise ValueError(f"Probability CSV missing columns: {sorted(missing)}")
    probs["instrument"] = probs["instrument"].str.lower()
    probs["prediction"] = probs["prediction"].clip(0.0, 1.0)
    return probs


def load_features(tickers: list[str]) -> pd.DataFrame:
    """Load numeric features and keep Energy instruments only."""

    features = pd.read_csv(FEATURE_MATRIX_PATH, parse_dates=["date"])
    features["instrument"] = features["instrument"].str.lower()
    return features[features["instrument"].isin(tickers)].copy()


def load_returns_and_vol(tickers: list[str]) -> pd.DataFrame:
    """Compute next-day returns and causal EWMA vol for training/inference."""

    ohlcv = pd.read_csv(OHLCV_PATH, parse_dates=["date"])
    ohlcv["instrument"] = ohlcv["instrument"].str.lower()
    ohlcv = ohlcv[ohlcv["instrument"].isin(tickers)].sort_values(["instrument", "date"]).copy()
    ohlcv["daily_return"] = ohlcv.groupby("instrument")["close"].pct_change()
    ohlcv["next_return"] = ohlcv.groupby("instrument")["daily_return"].shift(-1)
    ohlcv["annualized_ewma_vol"] = (
        ohlcv.groupby("instrument")["daily_return"]
        .transform(lambda s: s.ewm(span=EWMA_SPAN, min_periods=20, adjust=False).std() * np.sqrt(252))
    )
    return ohlcv[["date", "instrument", "next_return", "annualized_ewma_vol"]]


def choose_feature_cols(data: pd.DataFrame, max_features: int) -> list[str]:
    """Choose stable numeric feature columns without leakage or identity fields."""

    excluded = {
        "primary_signal",
        "prediction",
        "next_return",
        "annualized_ewma_vol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    }
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [col for col in numeric_cols if col not in excluded]
    missing = data[candidates].isna().mean().sort_values()
    return missing.head(max_features).index.tolist()


def build_dataset(probability_csv: Path, tickers: list[str]) -> pd.DataFrame:
    """Join probabilities, primary side, features, next returns, and vol."""

    probs = load_probabilities(probability_csv)
    primary = load_primary_long()
    features = load_features(tickers)
    returns = load_returns_and_vol(tickers)

    data = (
        probs.merge(primary, on=["date", "instrument"], how="left", validate="one_to_one")
        .merge(features.drop(columns=["primary_signal"], errors="ignore"), on=["date", "instrument"], how="left")
        .merge(returns, on=["date", "instrument"], how="left", validate="many_to_one")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    data["primary_signal"] = data["primary_signal"].fillna(0).astype(int)
    data = data[data["instrument"].isin(tickers)].copy()
    data = data[data["annualized_ewma_vol"].notna() & (data["annualized_ewma_vol"] > 0)].copy()
    return data.reset_index(drop=True)


def standardize_features(data: pd.DataFrame, feature_cols: list[str], mean: pd.Series | None = None, std: pd.Series | None = None):
    """Median-impute and standardize features using training statistics."""

    x = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    if mean is None:
        median = x.median()
        x = x.fillna(median)
        mean = x.mean()
        std = x.std(ddof=0).replace(0, 1.0)
        return ((x - mean) / std).values.astype(float), mean, std

    x = x.fillna(mean)
    return ((x - mean) / std).values.astype(float), mean, std


def apply_weight_caps(weights: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Apply individual and daily gross-exposure constraints."""

    capped = np.clip(weights, -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
    out = capped.copy()
    for date in pd.unique(dates):
        mask = dates == date
        gross = np.abs(out[mask]).sum()
        if gross > MAX_GROSS_EXPOSURE:
            out[mask] *= MAX_GROSS_EXPOSURE / gross
    return out


def train_sharpe_head(train: pd.DataFrame, feature_cols: list[str], args: argparse.Namespace) -> NeuralPortfolioState:
    """Train tanh projection parameters by gradient ascent on annualized Sharpe."""

    rng = np.random.default_rng(args.random_state)
    instruments = sorted(train["instrument"].unique().tolist())
    ticker_to_idx = {ticker: i for i, ticker in enumerate(instruments)}

    x, feature_mean, feature_std = standardize_features(train, feature_cols)
    ticker_idx = train["instrument"].map(ticker_to_idx).values
    side = train["primary_signal"].astype(float).values
    vol = train["annualized_ewma_vol"].astype(float).values
    next_return = train["next_return"].astype(float).fillna(0.0).values

    beta = rng.normal(0.0, 0.02, size=x.shape[1])
    ticker_bias = np.zeros(len(instruments))

    for epoch in range(1, args.epochs + 1):
        z = x @ beta + ticker_bias[ticker_idx]
        conviction = np.tanh(z)
        raw_weight = side * conviction * TARGET_VOL / vol
        weight = apply_weight_caps(raw_weight, train["date"].values)
        pnl = weight * next_return

        mean = pnl.mean()
        centered = pnl - mean
        std = np.sqrt(np.mean(centered**2) + 1e-8)
        sharpe = mean / std * ANNUALIZATION

        # Gradient of Sharpe with respect to each return contribution. This is
        # full-batch training over the return path, as in the optional session.
        n = len(pnl)
        dsharpe_dpnl = ANNUALIZATION * ((1.0 / (n * std)) - (mean * centered / (n * std**3)))

        # We do not backpropagate through the clipping/gross cap. This keeps the
        # optimizer simple and treats caps as the final execution layer.
        draw_weight_draw = (np.abs(raw_weight) <= MAX_ABS_WEIGHT).astype(float)
        dweight_dz = side * TARGET_VOL / vol * (1.0 - conviction**2) * draw_weight_draw
        dz_grad = dsharpe_dpnl * next_return * dweight_dz

        grad_beta = x.T @ dz_grad - args.l2 * beta
        grad_bias = np.bincount(ticker_idx, weights=dz_grad, minlength=len(instruments)) - args.l2 * ticker_bias

        beta += args.learning_rate * grad_beta
        ticker_bias += args.learning_rate * grad_bias

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d} | train sharpe {sharpe: .4f} | mean daily pnl {mean: .6f}")

    return NeuralPortfolioState(feature_cols, instruments, feature_mean, feature_std, beta, ticker_bias)


def infer_weights(data: pd.DataFrame, state: NeuralPortfolioState) -> pd.DataFrame:
    """Apply the trained neural head and volatility targeting to inference rows."""

    ticker_to_idx = {ticker: i for i, ticker in enumerate(state.instruments)}
    data = data[data["instrument"].isin(ticker_to_idx)].copy()
    x, _, _ = standardize_features(data, state.feature_cols, state.feature_mean, state.feature_std)
    ticker_idx = data["instrument"].map(ticker_to_idx).values
    side = data["primary_signal"].astype(float).values
    vol = data["annualized_ewma_vol"].astype(float).values

    z = x @ state.beta + state.ticker_bias[ticker_idx]
    conviction = np.tanh(z)
    raw_weight = side * conviction * TARGET_VOL / vol
    weight = apply_weight_caps(raw_weight, data["date"].values)

    out = data[["date", "instrument", "prediction", "primary_signal", "annualized_ewma_vol"]].copy()
    out["neural_conviction"] = conviction
    out["raw_weight"] = raw_weight
    out["weight"] = weight
    return out


def save_state(state: NeuralPortfolioState, output_dir: Path) -> None:
    """Save model parameters in CSV form so the run is auditable."""

    pd.DataFrame({"feature": state.feature_cols, "beta": state.beta}).to_csv(output_dir / "neural_feature_weights.csv", index=False)
    pd.DataFrame({"instrument": state.instruments, "ticker_bias": state.ticker_bias}).to_csv(
        output_dir / "neural_ticker_bias.csv", index=False
    )


def main() -> None:
    args = parse_args()
    tickers = ENERGY_TICKERS

    train = build_dataset(args.train_probability_csv, tickers)
    train = train[(train["date"] < TRAIN_END) & train["next_return"].notna()].copy()
    inference = build_dataset(args.inference_probability_csv, tickers)
    inference = inference[(inference["date"] >= INFERENCE_START) & (inference["date"] <= INFERENCE_END)].copy()

    feature_cols = choose_feature_cols(train, args.max_features)
    print("Neural portfolio construction")
    print("Train probability CSV:", args.train_probability_csv)
    print("Inference probability CSV:", args.inference_probability_csv)
    print("Train rows:", len(train))
    print("Inference rows:", len(inference))
    print("Features:", len(feature_cols))
    print("Target vol:", TARGET_VOL)

    state = train_sharpe_head(train, feature_cols, args)
    diagnostics = infer_weights(inference, state)
    weights = diagnostics[["date", "instrument", "weight"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(args.output_dir / "neural_strategy_weights.csv", index=False)
    diagnostics.to_csv(args.output_dir / "neural_allocation_diagnostics.csv", index=False)
    save_state(state, args.output_dir)

    gross = weights.groupby("date")["weight"].apply(lambda x: x.abs().sum())
    print("Wrote:", args.output_dir / "neural_strategy_weights.csv")
    print("Non-zero weights:", int((weights["weight"] != 0).sum()))
    print("Max gross exposure:", float(gross.max()))


if __name__ == "__main__":
    main()
