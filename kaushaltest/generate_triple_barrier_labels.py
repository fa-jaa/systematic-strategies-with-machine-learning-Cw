"""Precompute triple-barrier label CSVs for CPCV experiments.

This is intentionally separate from the shared notebooks. It caches labels once
per ticker/barrier configuration so the CPCV grid can reuse them instead of
rerunning triple-barrier labelling for every model and feature-selection combo.

Date policy:
- labels are generated only from dates strictly before 2022-01-01
- Jan-Jun 2022 is left untouched for the coursework prediction/evaluation step
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
TRIPLE_BARRIER_DIR = PROJECT_ROOT / "02_triple_barrier"
MODEL_CONFIG_DIR = PROJECT_ROOT / "03_model_development"
OUTPUT_ROOT = EXPERIMENT_ROOT / "data" / "labels" / "triple_barrier"
TRAINING_END = "2022-01-01"

sys.path.insert(0, str(TRIPLE_BARRIER_DIR))

from triple_barrier import run_triple_barrier_pipeline  # noqa: E402


ENERGY_TICKERS = ["cl1s", "ho1s", "rb1s", "ng1s"]

SMOKE_BARRIER_CONFIGS = [
    {
        "name": "ewma_d10_tp2.0_sl2.0",
        "volatility_method": "ewma",
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": 10,
        "take_profit_mult": 2.0,
        "stop_loss_mult": 2.0,
    }
]


REDUCED_BARRIER_CONFIGS = [
    {
        "name": f"{vol_method}_d{num_days}_tp2.0_sl2.0",
        "volatility_method": vol_method,
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": num_days,
        "take_profit_mult": 2.0,
        "stop_loss_mult": 2.0,
    }
    for vol_method in ["ewma", "parkinson", "garman_klass"]
    for num_days in [5, 10, 20]
]


FULL_BARRIER_CONFIGS = [
    {
        "name": f"{vol_method}_d{num_days}_tp{tp}_sl{sl}",
        "volatility_method": vol_method,
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": num_days,
        "take_profit_mult": tp,
        "stop_loss_mult": sl,
    }
    for vol_method in ["ewma", "rolling", "parkinson", "garman_klass"]
    for num_days in [5, 10, 20]
    for tp, sl in [(1.5, 1.5), (2.0, 2.0), (3.0, 3.0), (2.0, 1.5), (1.5, 2.0)]
]


PIPELINE_KEYS = {
    "volatility_method",
    "ewma_span",
    "volatility_window",
    "num_days",
    "take_profit_mult",
    "stop_loss_mult",
}


def label_path(ticker: str, barrier_config: dict) -> Path:
    """Return the cache path for one ticker/barrier combination."""

    safe_name = barrier_config["name"].replace(".", "p")
    return OUTPUT_ROOT / ticker / f"{safe_name}.csv"


def pipeline_kwargs(barrier_config: dict) -> dict:
    """Keep only arguments accepted by run_triple_barrier_pipeline."""

    return {key: barrier_config[key] for key in PIPELINE_KEYS if key in barrier_config}


def configs_for_mode(mode: str) -> list[dict]:
    if mode == "smoke":
        return SMOKE_BARRIER_CONFIGS
    if mode == "reduced":
        return REDUCED_BARRIER_CONFIGS
    if mode == "full":
        return FULL_BARRIER_CONFIGS
    raise ValueError(f"Unknown mode: {mode}")


def precompute_labels(
    tickers: Iterable[str],
    barrier_configs: Iterable[dict],
    overwrite: bool,
    dry_run: bool,
) -> pd.DataFrame:
    """Generate label CSVs and return a run manifest."""

    manifest_rows = []
    tickers = list(tickers)
    barrier_configs = list(barrier_configs)
    total = len(tickers) * len(barrier_configs)
    count = 0

    for ticker in tickers:
        for barrier_config in barrier_configs:
            count += 1
            output_path = label_path(ticker, barrier_config)
            exists = output_path.exists()

            row = {
                "ticker": ticker,
                "barrier": barrier_config["name"],
                "training_end": TRAINING_END,
                "path": str(output_path.relative_to(PROJECT_ROOT)),
                "status": "pending",
                "rows": None,
            }

            print(f"{count}/{total}: {ticker} | {barrier_config['name']}")

            if dry_run:
                row["status"] = "dry_run"
                manifest_rows.append(row)
                continue

            if exists and not overwrite:
                cached = pd.read_csv(output_path, usecols=["date"])
                row["status"] = "cached"
                row["rows"] = len(cached)
                manifest_rows.append(row)
                print(f"  cached: {output_path}")
                continue

            labels = run_triple_barrier_pipeline(
                instrument=ticker,
                training_end=TRAINING_END,
                output_path=output_path,
                **pipeline_kwargs(barrier_config),
            )

            max_date = pd.to_datetime(labels["date"]).max()
            if max_date >= pd.Timestamp(TRAINING_END):
                raise ValueError(
                    f"{ticker} {barrier_config['name']} leaked past {TRAINING_END}: "
                    f"max label date is {max_date}"
                )

            row["status"] = "written"
            row["rows"] = len(labels)
            manifest_rows.append(row)
            print(f"  wrote {len(labels):,} rows -> {output_path}")

    return pd.DataFrame(manifest_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute triple-barrier labels for Energy CPCV runs."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "reduced", "full"],
        default="reduced",
        help="Barrier grid to cache. Default: reduced 9-config grid.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=ENERGY_TICKERS,
        help="Tickers to process. Default: all Energy tickers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even if cached CSVs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned files without generating labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.lower() for ticker in args.tickers]
    barrier_configs = configs_for_mode(args.mode)

    print("Triple-barrier label cache")
    print("Mode:", args.mode)
    print("Tickers:", ", ".join(tickers))
    print("Barrier configs:", len(barrier_configs))
    print("Training cutoff: dates strictly before", TRAINING_END)
    print("Output root:", OUTPUT_ROOT)

    manifest = precompute_labels(
        tickers=tickers,
        barrier_configs=barrier_configs,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / f"manifest_{args.mode}.csv"
    manifest.to_csv(manifest_path, index=False)
    print("Manifest:", manifest_path)
    print(manifest["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
