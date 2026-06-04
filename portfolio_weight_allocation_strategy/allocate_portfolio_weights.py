"""Allocate portfolio weights from metamodel probabilities.

This implements a simple PDF-aligned strategy:

1. primary_signal supplies direction in {-1, 0, +1}
2. metamodel probability supplies confidence/sizing
3. EWMA volatility targeting converts confidence into risk-scaled weights
4. per-instrument and gross-exposure caps keep the placeholder strategy bounded

The default input is a placeholder probability CSV. Replace it with cleaned
model probabilities once they are available.
"""

from __future__ import annotations

import argparse
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ROOT = Path(__file__).resolve().parent
DEFAULT_PROBABILITY_CSV = STRATEGY_ROOT / "probabilities" / "placeholder_energy_active_055.csv"
PRIMARY_SIGNALS_PATH = PROJECT_ROOT / "data" / "raw" / "primary_signals.csv"
OHLCV_PATH = PROJECT_ROOT / "data" / "raw" / "ohlcv_data.csv"
DEFAULT_OUTPUT_DIR = STRATEGY_ROOT / "outputs"
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2022-06-30")

# Fixed strategy constraints. Keep these hard-coded so every run uses the same
# coursework portfolio policy unless the code is deliberately edited.
PROBABILITY_THRESHOLD = 0.50
TARGET_VOL = 0.10
EWMA_SPAN = 60
MAX_ABS_WEIGHT = 0.25
MAX_GROSS_EXPOSURE = 1.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate signed strategy weights from probability CSV.")
    parser.add_argument("--probability-csv", type=Path, default=DEFAULT_PROBABILITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sizing-method", choices=["model_confidence", "all_or_nothing", "ncdf"], default="model_confidence")
    parser.add_argument("--start-date", type=pd.Timestamp, default=START_DATE)
    parser.add_argument("--end-date", type=pd.Timestamp, default=END_DATE)
    return parser.parse_args()


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def probability_size(probability: float, method: str, threshold: float) -> float:
    p = float(np.clip(probability, 0.0, 1.0))
    if p <= threshold:
        return 0.0
    if method == "all_or_nothing":
        return 1.0
    if method == "model_confidence":
        return p
    if method == "ncdf":
        denom = sqrt(max(p * (1.0 - p), 1e-12))
        return normal_cdf((p - 0.5) / denom)
    raise ValueError(f"Unknown sizing method: {method}")


def load_primary_signals(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    signals = pd.read_csv(PRIMARY_SIGNALS_PATH, parse_dates=["date"])
    signals.columns = [col.lower() for col in signals.columns]
    signals = signals[(signals["date"] >= start_date) & (signals["date"] <= end_date)].copy()
    long = signals.melt(id_vars="date", var_name="instrument", value_name="primary_signal")
    long["instrument"] = long["instrument"].str.lower()
    return long


def load_probabilities(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    probabilities = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "instrument", "prediction"}
    missing = required - set(probabilities.columns)
    if missing:
        raise ValueError(f"Probability CSV missing columns: {sorted(missing)}")
    probabilities["instrument"] = probabilities["instrument"].str.lower()
    probabilities["prediction"] = probabilities["prediction"].clip(0.0, 1.0)
    return probabilities[(probabilities["date"] >= start_date) & (probabilities["date"] <= end_date)].copy()


def load_annualized_ewma_vol(span: int) -> pd.DataFrame:
    ohlcv = pd.read_csv(OHLCV_PATH, parse_dates=["date"])
    ohlcv["instrument"] = ohlcv["instrument"].str.lower()
    ohlcv = ohlcv.sort_values(["instrument", "date"]).copy()
    ohlcv["daily_return"] = ohlcv.groupby("instrument")["close"].pct_change()
    ohlcv["annualized_ewma_vol"] = (
        ohlcv.groupby("instrument")["daily_return"]
        .transform(lambda s: s.ewm(span=span, min_periods=20, adjust=False).std() * np.sqrt(252))
    )
    return ohlcv[["date", "instrument", "annualized_ewma_vol"]]


def allocate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    probabilities = load_probabilities(args.probability_csv, args.start_date, args.end_date)
    primary = load_primary_signals(args.start_date, args.end_date)
    vol = load_annualized_ewma_vol(EWMA_SPAN)

    data = (
        probabilities.merge(primary, on=["date", "instrument"], how="left", validate="one_to_one")
        .merge(vol, on=["date", "instrument"], how="left", validate="many_to_one")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    data["primary_signal"] = data["primary_signal"].fillna(0).astype(int)
    data["size"] = data["prediction"].map(lambda p: probability_size(p, args.sizing_method, PROBABILITY_THRESHOLD))
    active = data["primary_signal"].isin([-1, 1]) & data["annualized_ewma_vol"].notna() & (data["annualized_ewma_vol"] > 0)

    data["raw_weight"] = 0.0
    data.loc[active, "raw_weight"] = (
        data.loc[active, "primary_signal"]
        * data.loc[active, "size"]
        * TARGET_VOL
        / data.loc[active, "annualized_ewma_vol"]
    )
    data["capped_weight"] = data["raw_weight"].clip(-MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)

    gross = data.groupby("date")["capped_weight"].transform(lambda x: x.abs().sum())
    scale = np.where(gross > MAX_GROSS_EXPOSURE, MAX_GROSS_EXPOSURE / gross.replace(0, np.nan), 1.0)
    data["weight"] = data["capped_weight"] * pd.Series(scale, index=data.index).fillna(1.0)

    weights = data[["date", "instrument", "weight"]].copy()
    diagnostics = data[
        [
            "date",
            "instrument",
            "prediction",
            "primary_signal",
            "size",
            "annualized_ewma_vol",
            "raw_weight",
            "capped_weight",
            "weight",
        ]
    ].copy()
    return weights, diagnostics


def main() -> None:
    args = parse_args()
    weights, diagnostics = allocate(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "strategy_weights.csv"
    diagnostics_path = args.output_dir / "allocation_diagnostics.csv"
    weights.to_csv(weights_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    print(f"Wrote {len(weights):,} weights -> {weights_path}")
    print(f"Wrote diagnostics -> {diagnostics_path}")
    print("Non-zero weights:", int((weights["weight"] != 0).sum()))
    print("Max gross exposure:", float(weights.groupby("date")["weight"].apply(lambda x: x.abs().sum()).max()))


if __name__ == "__main__":
    main()
