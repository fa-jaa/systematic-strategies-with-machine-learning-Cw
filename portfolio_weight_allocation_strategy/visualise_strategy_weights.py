"""Compare fixed-rule and neural portfolio weights.

Outputs:
- aligned row-level comparison CSV
- per-strategy summary CSV
- per-instrument summary CSV
- PNG plots comparing exposure, weights, cumulative returns, and scatter fit
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ROOT = Path(__file__).resolve().parent
DEFAULT_FIXED_WEIGHTS = STRATEGY_ROOT / "outputs" / "strategy_weights.csv"
DEFAULT_NEURAL_WEIGHTS = STRATEGY_ROOT / "outputs" / "neural_portfolio" / "neural_strategy_weights.csv"
DEFAULT_OUTPUT_DIR = STRATEGY_ROOT / "outputs" / "strategy_comparison"
OHLCV_PATH = PROJECT_ROOT / "data" / "raw" / "ohlcv_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise fixed vs neural strategy weights.")
    parser.add_argument("--fixed-weights", type=Path, default=DEFAULT_FIXED_WEIGHTS)
    parser.add_argument("--neural-weights", type=Path, default=DEFAULT_NEURAL_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_weights(path: Path, column_name: str) -> pd.DataFrame:
    weights = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "instrument", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    weights["instrument"] = weights["instrument"].str.lower()
    return weights.rename(columns={"weight": column_name})


def load_next_returns() -> pd.DataFrame:
    ohlcv = pd.read_csv(OHLCV_PATH, parse_dates=["date"])
    ohlcv["instrument"] = ohlcv["instrument"].str.lower()
    ohlcv = ohlcv.sort_values(["instrument", "date"]).copy()
    ohlcv["daily_return"] = ohlcv.groupby("instrument")["close"].pct_change()
    ohlcv["next_return"] = ohlcv.groupby("instrument")["daily_return"].shift(-1)
    return ohlcv[["date", "instrument", "next_return"]]


def align_weights(fixed_path: Path, neural_path: Path) -> pd.DataFrame:
    fixed = load_weights(fixed_path, "fixed_weight")
    neural = load_weights(neural_path, "neural_weight")
    returns = load_next_returns()

    comparison = (
        fixed.merge(neural, on=["date", "instrument"], how="outer", validate="one_to_one")
        .merge(returns, on=["date", "instrument"], how="left", validate="many_to_one")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    comparison[["fixed_weight", "neural_weight"]] = comparison[["fixed_weight", "neural_weight"]].fillna(0.0)
    comparison["weight_diff"] = comparison["neural_weight"] - comparison["fixed_weight"]
    comparison["same_sign"] = np.sign(comparison["fixed_weight"]).eq(np.sign(comparison["neural_weight"]))
    comparison["fixed_pnl"] = comparison["fixed_weight"] * comparison["next_return"]
    comparison["neural_pnl"] = comparison["neural_weight"] * comparison["next_return"]
    return comparison


def annualized_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return np.nan
    return returns.mean() / returns.std(ddof=1) * np.sqrt(252)


def strategy_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, weight_col, pnl_col in [
        ("fixed", "fixed_weight", "fixed_pnl"),
        ("neural", "neural_weight", "neural_pnl"),
    ]:
        daily = comparison.groupby("date").agg(
            gross=(weight_col, lambda x: x.abs().sum()),
            net=(weight_col, "sum"),
            pnl=(pnl_col, "mean"),
            active_count=(weight_col, lambda x: (x != 0).sum()),
        )
        rows.append(
            {
                "strategy": name,
                "rows": len(comparison),
                "nonzero_weights": int((comparison[weight_col] != 0).sum()),
                "mean_abs_weight": comparison[weight_col].abs().mean(),
                "max_abs_weight": comparison[weight_col].abs().max(),
                "mean_daily_gross": daily["gross"].mean(),
                "max_daily_gross": daily["gross"].max(),
                "mean_daily_net": daily["net"].mean(),
                "mean_active_count": daily["active_count"].mean(),
                "mean_daily_pnl": daily["pnl"].mean(),
                "annualized_vol": daily["pnl"].std(ddof=1) * np.sqrt(252),
                "annualized_sharpe": annualized_sharpe(daily["pnl"]),
                "cumulative_pnl": daily["pnl"].sum(),
            }
        )
    return pd.DataFrame(rows)


def instrument_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instrument, group in comparison.groupby("instrument"):
        rows.append(
            {
                "instrument": instrument,
                "rows": len(group),
                "fixed_mean_abs_weight": group["fixed_weight"].abs().mean(),
                "neural_mean_abs_weight": group["neural_weight"].abs().mean(),
                "fixed_max_abs_weight": group["fixed_weight"].abs().max(),
                "neural_max_abs_weight": group["neural_weight"].abs().max(),
                "mean_weight_diff": group["weight_diff"].mean(),
                "mean_abs_weight_diff": group["weight_diff"].abs().mean(),
                "same_sign_rate": group["same_sign"].mean(),
                "weight_correlation": group["fixed_weight"].corr(group["neural_weight"]),
            }
        )
    return pd.DataFrame(rows)


def daily_panel(comparison: pd.DataFrame) -> pd.DataFrame:
    return comparison.groupby("date").agg(
        fixed_gross=("fixed_weight", lambda x: x.abs().sum()),
        neural_gross=("neural_weight", lambda x: x.abs().sum()),
        fixed_net=("fixed_weight", "sum"),
        neural_net=("neural_weight", "sum"),
        fixed_pnl=("fixed_pnl", "mean"),
        neural_pnl=("neural_pnl", "mean"),
    )


def plot_daily_exposure(daily: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    daily[["fixed_gross", "neural_gross"]].plot(ax=ax)
    ax.set_title("Daily Gross Exposure")
    ax.set_ylabel("Gross exposure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "daily_gross_exposure.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    daily[["fixed_net", "neural_net"]].plot(ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Daily Net Exposure")
    ax.set_ylabel("Net exposure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "daily_net_exposure.png", dpi=150)
    plt.close(fig)


def plot_cumulative_pnl(daily: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    cumulative = daily[["fixed_pnl", "neural_pnl"]].cumsum()
    cumulative.plot(ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Cumulative Next-Day PnL Proxy")
    ax.set_ylabel("Cumulative mean daily PnL")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_pnl_proxy.png", dpi=150)
    plt.close(fig)


def plot_weight_scatter(comparison: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for instrument, group in comparison.groupby("instrument"):
        ax.scatter(group["fixed_weight"], group["neural_weight"], s=18, alpha=0.65, label=instrument)
    lim = max(comparison["fixed_weight"].abs().max(), comparison["neural_weight"].abs().max(), 0.25)
    ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=1, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlim(-lim * 1.1, lim * 1.1)
    ax.set_ylim(-lim * 1.1, lim * 1.1)
    ax.set_xlabel("Fixed-rule weight")
    ax.set_ylabel("Neural weight")
    ax.set_title("Fixed vs Neural Weights")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fixed_vs_neural_weight_scatter.png", dpi=150)
    plt.close(fig)


def plot_instrument_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width / 2, summary["fixed_mean_abs_weight"], width, label="fixed")
    ax.bar(x + width / 2, summary["neural_mean_abs_weight"], width, label="neural")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["instrument"])
    ax.set_title("Mean Absolute Weight by Instrument")
    ax.set_ylabel("Mean abs weight")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_abs_weight_by_instrument.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = align_weights(args.fixed_weights, args.neural_weights)
    daily = daily_panel(comparison)
    summary = strategy_summary(comparison)
    by_instrument = instrument_summary(comparison)

    comparison.to_csv(args.output_dir / "fixed_vs_neural_row_comparison.csv", index=False)
    daily.to_csv(args.output_dir / "daily_exposure_and_pnl.csv")
    summary.to_csv(args.output_dir / "strategy_summary.csv", index=False)
    by_instrument.to_csv(args.output_dir / "instrument_summary.csv", index=False)

    plot_daily_exposure(daily, args.output_dir)
    plot_cumulative_pnl(daily, args.output_dir)
    plot_weight_scatter(comparison, args.output_dir)
    plot_instrument_bars(by_instrument, args.output_dir)

    print("Comparison rows:", len(comparison))
    print("Saved comparison outputs to:", args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
