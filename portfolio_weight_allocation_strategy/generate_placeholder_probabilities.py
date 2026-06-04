"""Generate placeholder metamodel probability CSVs.

These files are temporary stand-ins until cleaned model probabilities are
available. They follow the coursework format:

    date,instrument,prediction

The placeholders keep only active primary-signal rows, matching the expected
model-output shape where a probability is produced for a trade candidate:

    date,instrument,prediction
    2022-01-03,cl1s,0.55
    2022-01-03,rb1s,0.55

Inactive primary signals are omitted because there is no primary trade to size.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "probabilities"
PRIMARY_SIGNALS_PATH = PROJECT_ROOT / "data" / "raw" / "primary_signals.csv"
TRAIN_START_DATE = pd.Timestamp("2020-01-01")
TRAIN_END_DATE = pd.Timestamp("2021-12-31")
INFERENCE_START_DATE = pd.Timestamp("2022-01-01")
INFERENCE_END_DATE = pd.Timestamp("2022-06-30")
ENERGY_TICKERS = ["cl1s", "ho1s", "rb1s", "ng1s"]


def make_long_signals(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    signals = pd.read_csv(PRIMARY_SIGNALS_PATH, parse_dates=["date"])
    signals.columns = [col.lower() for col in signals.columns]
    signals = signals[(signals["date"] >= start_date) & (signals["date"] <= end_date)].copy()
    long = signals.melt(id_vars="date", var_name="instrument", value_name="primary_signal")
    long["instrument"] = long["instrument"].str.lower()
    return long.sort_values(["date", "instrument"]).reset_index(drop=True)


def write_placeholder(df: pd.DataFrame, output_name: str, active_probability: float) -> Path:
    out = df.loc[df["primary_signal"].isin([-1, 1]), ["date", "instrument"]].copy()
    out["prediction"] = active_probability
    out = out[["date", "instrument", "prediction"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name
    out.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    train_long = make_long_signals(TRAIN_START_DATE, TRAIN_END_DATE)
    inference_long = make_long_signals(INFERENCE_START_DATE, INFERENCE_END_DATE)
    train_energy = train_long[train_long["instrument"].isin(ENERGY_TICKERS)].copy()
    inference_energy = inference_long[inference_long["instrument"].isin(ENERGY_TICKERS)].copy()

    outputs = [
        write_placeholder(inference_energy, "placeholder_energy_active_055.csv", active_probability=0.55),
        write_placeholder(inference_energy, "placeholder_energy_neutral_050.csv", active_probability=0.50),
        write_placeholder(inference_long, "placeholder_all_assets_active_055.csv", active_probability=0.55),
        write_placeholder(train_energy, "placeholder_energy_train_active_055.csv", active_probability=0.55),
    ]

    for path in outputs:
        rows = pd.read_csv(path).shape[0]
        print(f"Wrote {rows:,} rows -> {path}")


if __name__ == "__main__":
    main()
