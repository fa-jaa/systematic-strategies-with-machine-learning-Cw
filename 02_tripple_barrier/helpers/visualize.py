"""
visualize.py — Plotting utilities for the Triple Barrier Method.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


LABEL_COLORS = {1: "#2ecc71", -1: "#e74c3c", 0: "#95a5a6"}
LABEL_NAMES = {1: "Take-Profit (+1)", -1: "Stop-Loss (−1)", 0: "Time-Out (0)"}


def plot_label_distribution(labels: pd.DataFrame, title: str = "Label Distribution") -> None:
    """Bar chart of label counts and proportions."""
    counts = labels["label"].value_counts().sort_index()
    colors = [LABEL_COLORS.get(l, "steelblue") for l in counts.index]
    names = [LABEL_NAMES.get(l, str(l)) for l in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, counts.values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, counts.values):
        pct = 100 * val / counts.sum()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{val}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_barrier_examples(
    close: pd.Series,
    labels: pd.DataFrame,
    n_examples: int = 6,
    title_prefix: str = "Triple Barrier",
) -> None:
    """
    Plot n individual trade examples showing barriers and price path.

    One subplot per example; shaded region from entry to exit; horizontal
    lines for take-profit and stop-loss; dotted line for the vertical barrier.
    """
    sample = labels.dropna(subset=["touch_date"]).sample(
        min(n_examples, len(labels)), random_state=42
    )

    ncols = 3
    nrows = int(np.ceil(len(sample) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, (t0, row) in enumerate(sample.iterrows()):
        ax = axes[i]
        t1 = row["t1"]
        touch = row["touch_date"]
        label = int(row["label"])
        color = LABEL_COLORS[label]

        # Price path from entry to vertical barrier
        path = close.loc[t0:t1]
        ax.plot(path.index, path.values, color="steelblue", lw=1.5, zorder=2)

        # Barrier lines
        if not np.isnan(row["pt"]):
            ax.axhline(row["pt"], color="#2ecc71", lw=1.2, linestyle="--", label="TP")
        if not np.isnan(row["sl"]):
            ax.axhline(row["sl"], color="#e74c3c", lw=1.2, linestyle="--", label="SL")
        ax.axvline(t1, color="grey", lw=1.0, linestyle=":", label="Vertical")

        # Entry marker
        ax.scatter([t0], [close.loc[t0]], color="black", s=40, zorder=5)
        # Touch marker
        ax.scatter([touch], [close.loc[touch]], color=color, s=60, zorder=5)

        ax.set_title(
            f"{t0.date()} → {LABEL_NAMES[label]}",
            fontsize=9,
            color=color,
        )
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    tp_patch = mpatches.Patch(color="#2ecc71", label="Take-Profit")
    sl_patch = mpatches.Patch(color="#e74c3c", label="Stop-Loss")
    to_patch = mpatches.Patch(color="#95a5a6", label="Time-Out")
    fig.legend(
        handles=[tp_patch, sl_patch, to_patch],
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(f"{title_prefix} — Example Trades", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_labels_on_price(
    close: pd.Series,
    labels: pd.DataFrame,
    title: str = "Labels on Price Series",
    start: str | None = None,
    end: str | None = None,
) -> None:
    """
    Overlay entry-point labels (coloured dots) on the price series.

    Parameters
    ----------
    close  : pd.Series  — full price series.
    labels : pd.DataFrame — labelled events with 'label' column.
    start, end : str     — optional date slice for readability.
    """
    subset_close = close.loc[start:end] if (start or end) else close
    subset_labels = labels.loc[
        labels.index >= (pd.Timestamp(start) if start else labels.index.min())
    ]
    subset_labels = subset_labels.loc[
        subset_labels.index <= (pd.Timestamp(end) if end else labels.index.max())
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(subset_close.index, subset_close.values, color="steelblue", lw=1, alpha=0.8)

    for lbl, grp in subset_labels.groupby("label"):
        ax.scatter(
            grp.index,
            close.reindex(grp.index),
            color=LABEL_COLORS[lbl],
            label=LABEL_NAMES[lbl],
            s=20,
            zorder=4,
            alpha=0.8,
        )

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_return_distribution(labels: pd.DataFrame, title: str = "Return Distribution by Label") -> None:
    """Histogram of per-trade log returns, broken down by label."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, (lbl, name) in zip(axes, LABEL_NAMES.items()):
        grp = labels[labels["label"] == lbl]["ret"].dropna()
        if grp.empty:
            ax.set_title(name)
            continue
        ax.hist(grp, bins=40, color=LABEL_COLORS[lbl], edgecolor="white", alpha=0.85)
        ax.axvline(grp.mean(), color="black", lw=1.5, linestyle="--", label=f"mean={grp.mean():.4f}")
        ax.set_title(f"{name}\nn={len(grp)}")
        ax.set_xlabel("Log Return (direction-adjusted)")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Count")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_volatility(
    close: pd.Series,
    daily_vol: pd.Series,
    title: str = "Price and Daily Volatility",
) -> None:
    """Two-panel plot: closing price above, daily vol below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(close.index, close.values, color="steelblue", lw=1)
    ax1.set_ylabel("Close Price")
    ax1.set_title(title)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(daily_vol.index, daily_vol.values, color="darkorange", lw=1)
    ax2.set_ylabel("Daily Vol (σ)")
    ax2.set_xlabel("Date")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()
