from pathlib import Path
import sys

import pandas as pd

# ------------------------------------------------------------
# Make triple_barrier.py importable
# ------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0]

sys.path.append(str(CURRENT_DIR))

from triple_barrier import run_triple_barrier_pipeline


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "triple_barrier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "triple_barrier_config_summary.csv"


# ------------------------------------------------------------
# Instruments to label
# ------------------------------------------------------------
INSTRUMENTS = [
    "cl1s",
    "ho1s",
    "rb1s",
    "ng1s",
    "gc1s",
    "si1s",
    "hg1s",
    "pl1s",
    "es1s",
    "nq1s",
    "fesx1s",
]


# ------------------------------------------------------------
# Triple-barrier configurations
# ------------------------------------------------------------
CONFIGS = [
    {
        "config_name": "ewma_10d_tp2_sl2",
        "volatility_method": "ewma",
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": 10,
        "take_profit_mult": 2.0,
        "stop_loss_mult": 2.0,
    },
    {
        "config_name": "ewma_5d_tp2_sl2",
        "volatility_method": "ewma",
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": 5,
        "take_profit_mult": 2.0,
        "stop_loss_mult": 2.0,
    },
    {
        "config_name": "ewma_10d_tp1_5_sl1_5",
        "volatility_method": "ewma",
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": 10,
        "take_profit_mult": 1.5,
        "stop_loss_mult": 1.5,
    },
    {
        "config_name": "rolling_10d_tp2_sl2",
        "volatility_method": "rolling",
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": 10,
        "take_profit_mult": 2.0,
        "stop_loss_mult": 2.0,
    },
    {
        "config_name": "parkinson_10d_tp2_sl2",
        "volatility_method": "parkinson",
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": 10,
        "take_profit_mult": 2.0,
        "stop_loss_mult": 2.0,
    },
    {
        "config_name": "garman_klass_10d_tp2_sl2",
        "volatility_method": "garman_klass",
        "ewma_span": 100,
        "volatility_window": 20,
        "num_days": 10,
        "take_profit_mult": 2.0,
        "stop_loss_mult": 2.0,
    },
]


# ------------------------------------------------------------
# Run all configs
# ------------------------------------------------------------
summary_rows = []

for instrument in INSTRUMENTS:
    for config in CONFIGS:
        config_name = config["config_name"]

        output_path = OUTPUT_DIR / f"{instrument}_{config_name}.csv"

        print(f"Running {instrument} | {config_name}")

        labels = run_triple_barrier_pipeline(
            instrument=instrument,
            signal_column=instrument,
            training_end="2022-01-01",
            volatility_method=config["volatility_method"],
            ewma_span=config["ewma_span"],
            volatility_window=config["volatility_window"],
            num_days=config["num_days"],
            take_profit_mult=config["take_profit_mult"],
            stop_loss_mult=config["stop_loss_mult"],
            output_path=output_path,
        )

        summary_rows.append(
            {
                "instrument": instrument,
                "config_name": config_name,
                "output_path": str(output_path),
                "rows": len(labels),
                "metalabel_0_count": int((labels["metalabel"] == 0).sum()),
                "metalabel_1_count": int((labels["metalabel"] == 1).sum()),
                "metalabel_1_rate": labels["metalabel"].mean(),
                "tb_label_-1_count": int((labels["triple_barrier_label"] == -1).sum()),
                "tb_label_0_count": int((labels["triple_barrier_label"] == 0).sum()),
                "tb_label_1_count": int((labels["triple_barrier_label"] == 1).sum()),
                "avg_holding_period_days": labels["holding_period_days"].mean(),
                "volatility_method": config["volatility_method"],
                "ewma_span": config["ewma_span"],
                "volatility_window": config["volatility_window"],
                "num_days": config["num_days"],
                "take_profit_mult": config["take_profit_mult"],
                "stop_loss_mult": config["stop_loss_mult"],
            }
        )

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

print()
print(f"Saved {len(summary)} triple-barrier CSV files to:")
print(OUTPUT_DIR)
print()
print(f"Saved summary file to:")
print(SUMMARY_PATH)