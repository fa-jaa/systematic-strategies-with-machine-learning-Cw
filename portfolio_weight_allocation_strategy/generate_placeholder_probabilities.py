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
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2022-06-30")
ENERGY_TICKERS = ["cl1s", "ho1s", "rb1s", "ng1s"]


def make_long_signals() -> pd.DataFrame:
    signals = pd.read_csv(PRIMARY_SIGNALS_PATH, parse_dates=["date"])
    signals.columns = [col.lower() for col in signals.columns]
    signals = signals[(signals["date"] >= START_DATE) & (signals["date"] <= END_DATE)].copy()
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
    long = make_long_signals()
    energy = long[long["instrument"].isin(ENERGY_TICKERS)].copy()

    outputs = [
        write_placeholder(energy, "placeholder_energy_active_055.csv", active_probability=0.55),
        write_placeholder(energy, "placeholder_energy_neutral_050.csv", active_probability=0.50),
        write_placeholder(long, "placeholder_all_assets_active_055.csv", active_probability=0.55),
    ]

    for path in outputs:
        rows = pd.read_csv(path).shape[0]
        print(f"Wrote {rows:,} rows -> {path}")


if __name__ == "__main__":
    main()
