"""Create interpretable HMM regime categories and category probabilities.

The base HMM gives numbered latent-state probabilities:

    hmm_regime_prob_0, ..., hmm_regime_prob_3

This script maps those instrument-specific latent states into interpreted
categories, then sums latent probabilities into category probabilities:

    P(extreme_vol), P(stress), P(strong_upside), ...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HMM_ROOT = Path("data/features/hmm")
PREDICTION_DIR = HMM_ROOT / "predictions"
CATEGORY_DIR = HMM_ROOT / "categories"
PROBABILITY_DIR = HMM_ROOT / "probabilities"
OHLCV_PATH = Path("data/raw/ohlcv_data.csv")

LATENT_PATH = PREDICTION_DIR / "latent_regime_predictions.csv"

VOLATILITY_LABELS = ["low_vol", "normal_vol", "high_vol", "extreme_vol"]
RETURN_LABELS = ["downside", "weak", "positive", "strong_upside"]
MARKET_STATE_LABELS = ["stress", "upside_breakout", "calm_positive", "calm_negative", "chop"]


def load_latent_regimes() -> pd.DataFrame:
    if LATENT_PATH.exists():
        return pd.read_csv(LATENT_PATH, parse_dates=["date"])
    misplaced_path = HMM_ROOT / "latent_regime_predictions.csv"
    if misplaced_path.exists():
        return pd.read_csv(misplaced_path, parse_dates=["date"])
    legacy_path = HMM_ROOT / "hmm_regime_predictions.csv"
    if legacy_path.exists():
        return pd.read_csv(legacy_path, parse_dates=["date"])
    raise FileNotFoundError(
        f"Could not find {LATENT_PATH}, {misplaced_path}, or {legacy_path}. "
        "Run regime_detection.py first."
    )


def load_panel() -> pd.DataFrame:
    regimes = load_latent_regimes()
    ohlcv = pd.read_csv(OHLCV_PATH, parse_dates=["date"])
    ohlcv["instrument"] = ohlcv["instrument"].str.lower()
    ohlcv = ohlcv.sort_values(["instrument", "date"])
    ohlcv["daily_return"] = ohlcv.groupby("instrument")["close"].pct_change()
    ohlcv["realized_vol_20d"] = (
        ohlcv.groupby("instrument")["daily_return"]
        .rolling(20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )
    return regimes.merge(
        ohlcv[["date", "instrument", "close", "daily_return", "realized_vol_20d"]],
        on=["date", "instrument"],
        how="left",
    )


def ranked_labels(values: pd.Series, labels: list[str]) -> dict[float, str]:
    ordered_regimes = values.sort_values().index.tolist()
    return {regime: labels[i] for i, regime in enumerate(ordered_regimes)}


def market_state(return_label: str, vol_label: str) -> str:
    if vol_label in {"high_vol", "extreme_vol"} and return_label in {"downside", "weak"}:
        return "stress"
    if vol_label in {"high_vol", "extreme_vol"} and return_label in {"positive", "strong_upside"}:
        return "upside_breakout"
    if vol_label in {"low_vol", "normal_vol"} and return_label in {"positive", "strong_upside"}:
        return "calm_positive"
    if vol_label in {"low_vol", "normal_vol"} and return_label in {"downside", "weak"}:
        return "calm_negative"
    return "chop"


def build_regime_map(panel: pd.DataFrame) -> pd.DataFrame:
    """Map numbered latent regimes to interpreted categories per instrument."""

    rows = []
    clean = panel.dropna(subset=["hmm_predicted_regime", "daily_return", "realized_vol_20d"]).copy()
    for instrument, inst_frame in clean.groupby("instrument"):
        stats = (
            inst_frame.groupby("hmm_predicted_regime")
            .agg(
                days=("date", "count"),
                avg_daily_return=("daily_return", "mean"),
                avg_abs_return=("daily_return", lambda x: x.abs().mean()),
                avg_realized_vol_20d=("realized_vol_20d", "mean"),
                avg_confidence=("hmm_regime_confidence", "mean"),
            )
        )
        vol_map = ranked_labels(stats["avg_realized_vol_20d"], VOLATILITY_LABELS[: len(stats)])
        ret_map = ranked_labels(stats["avg_daily_return"], RETURN_LABELS[: len(stats)])

        for regime, stat_row in stats.iterrows():
            vol_label = vol_map[regime]
            ret_label = ret_map[regime]
            rows.append(
                {
                    "instrument": instrument,
                    "hmm_predicted_regime": int(regime),
                    "hmm_volatility_regime": vol_label,
                    "hmm_return_regime": ret_label,
                    "hmm_market_state": market_state(ret_label, vol_label),
                    "days": int(stat_row["days"]),
                    "avg_daily_return": stat_row["avg_daily_return"],
                    "avg_abs_return": stat_row["avg_abs_return"],
                    "avg_realized_vol_20d": stat_row["avg_realized_vol_20d"],
                    "avg_confidence": stat_row["avg_confidence"],
                }
            )
    return pd.DataFrame(rows)


def add_category_probabilities(enriched: pd.DataFrame, regime_map: pd.DataFrame) -> pd.DataFrame:
    """Sum latent-state probabilities into interpreted category probabilities."""

    out = enriched.copy()
    for label in VOLATILITY_LABELS:
        out[f"hmm_prob_{label}"] = 0.0
    for label in RETURN_LABELS:
        out[f"hmm_prob_{label}"] = 0.0
    for label in MARKET_STATE_LABELS:
        out[f"hmm_prob_{label}"] = 0.0

    for _, row in regime_map.iterrows():
        instrument = row["instrument"]
        regime = int(row["hmm_predicted_regime"])
        prob_col = f"hmm_regime_prob_{regime}"
        mask = out["instrument"].eq(instrument) & out[prob_col].notna()
        out.loc[mask, f"hmm_prob_{row['hmm_volatility_regime']}"] += out.loc[mask, prob_col]
        out.loc[mask, f"hmm_prob_{row['hmm_return_regime']}"] += out.loc[mask, prob_col]
        out.loc[mask, f"hmm_prob_{row['hmm_market_state']}"] += out.loc[mask, prob_col]

    out["hmm_prob_high_or_extreme_vol"] = out["hmm_prob_high_vol"] + out["hmm_prob_extreme_vol"]
    out["hmm_prob_low_or_normal_vol"] = out["hmm_prob_low_vol"] + out["hmm_prob_normal_vol"]
    out["hmm_prob_downside_or_weak"] = out["hmm_prob_downside"] + out["hmm_prob_weak"]
    out["hmm_prob_positive_or_strong_upside"] = out["hmm_prob_positive"] + out["hmm_prob_strong_upside"]
    out["hmm_prob_not_stress"] = 1.0 - out["hmm_prob_stress"]
    latent_prob_cols = [c for c in out.columns if c.startswith("hmm_regime_prob_")]
    category_prob_cols = [c for c in out.columns if c.startswith("hmm_prob_")]
    missing_hmm = out[latent_prob_cols].isna().all(axis=1)
    out.loc[missing_hmm, category_prob_cols] = pd.NA
    return out


def write_outputs(enriched: pd.DataFrame, regime_map: pd.DataFrame) -> None:
    CATEGORY_DIR.mkdir(parents=True, exist_ok=True)
    PROBABILITY_DIR.mkdir(parents=True, exist_ok=True)

    latent_cols = [
        "date",
        "instrument",
        "primary_signal",
        "hmm_predicted_regime",
        "hmm_regime_confidence",
        "hmm_regime_prob_0",
        "hmm_regime_prob_1",
        "hmm_regime_prob_2",
        "hmm_regime_prob_3",
    ]
    label_cols = [
        "date",
        "instrument",
        "primary_signal",
        "hmm_predicted_regime",
        "hmm_volatility_regime",
        "hmm_return_regime",
        "hmm_market_state",
        "hmm_regime_confidence",
        "daily_return",
        "realized_vol_20d",
    ]
    volatility_prob_cols = [
        "date",
        "instrument",
        "primary_signal",
        "hmm_volatility_regime",
        "hmm_prob_low_vol",
        "hmm_prob_normal_vol",
        "hmm_prob_high_vol",
        "hmm_prob_extreme_vol",
        "hmm_prob_low_or_normal_vol",
        "hmm_prob_high_or_extreme_vol",
    ]
    return_prob_cols = [
        "date",
        "instrument",
        "primary_signal",
        "hmm_return_regime",
        "hmm_prob_downside",
        "hmm_prob_weak",
        "hmm_prob_positive",
        "hmm_prob_strong_upside",
        "hmm_prob_downside_or_weak",
        "hmm_prob_positive_or_strong_upside",
    ]
    market_prob_cols = [
        "date",
        "instrument",
        "primary_signal",
        "hmm_market_state",
        "hmm_prob_stress",
        "hmm_prob_upside_breakout",
        "hmm_prob_calm_positive",
        "hmm_prob_calm_negative",
        "hmm_prob_chop",
        "hmm_prob_not_stress",
    ]
    all_prob_cols = [
        "date",
        "instrument",
        "primary_signal",
        "hmm_predicted_regime",
        "hmm_volatility_regime",
        "hmm_return_regime",
        "hmm_market_state",
        "hmm_regime_confidence",
        *[c for c in enriched.columns if c.startswith("hmm_prob_")],
    ]

    enriched[latent_cols].to_csv(PROBABILITY_DIR / "latent_regime_probabilities.csv", index=False)
    enriched[label_cols].to_csv(CATEGORY_DIR / "regime_category_labels.csv", index=False)
    regime_map.to_csv(CATEGORY_DIR / "latent_regime_to_category_map.csv", index=False)
    enriched[volatility_prob_cols].to_csv(PROBABILITY_DIR / "volatility_regime_probabilities.csv", index=False)
    enriched[return_prob_cols].to_csv(PROBABILITY_DIR / "return_regime_probabilities.csv", index=False)
    enriched[market_prob_cols].to_csv(PROBABILITY_DIR / "market_state_probabilities.csv", index=False)
    enriched[all_prob_cols].to_csv(PROBABILITY_DIR / "all_category_probabilities.csv", index=False)


def main() -> None:
    panel = load_panel()
    regime_map = build_regime_map(panel)
    enriched = panel.merge(regime_map, on=["instrument", "hmm_predicted_regime"], how="left")
    enriched = add_category_probabilities(enriched, regime_map)
    write_outputs(enriched, regime_map)

    print("Wrote organized HMM category outputs under data/hmm/categories and probabilities")


if __name__ == "__main__":
    main()
