"""
Phase 1 – Cointegration Analysis
Runs the Engle-Granger two-step method on every unique pair of tickers and
ranks them by p-value.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from config import COINT_P_VALUE_THRESHOLD


# ──────────────────────────────────────────────
# Pair-wise cointegration scan
# ──────────────────────────────────────────────

def run_cointegration_scan(
    prices: pd.DataFrame,
    p_threshold: float = COINT_P_VALUE_THRESHOLD,
) -> pd.DataFrame:
    """
    Test every unique pair for cointegration (Engle-Granger).

    Returns a DataFrame with columns:
        ticker_a, ticker_b, t_stat, p_value, crit_1pct, crit_5pct, crit_10pct, cointegrated
    sorted by ``p_value`` ascending.
    """
    tickers = list(prices.columns)
    records: List[Dict] = []

    for a, b in combinations(tickers, 2):
        t_stat, p_val, crit_vals = coint(prices[a], prices[b])
        records.append(
            {
                "ticker_a": a,
                "ticker_b": b,
                "t_stat": round(t_stat, 4),
                "p_value": round(p_val, 6),
                "crit_1pct": round(crit_vals[0], 4),
                "crit_5pct": round(crit_vals[1], 4),
                "crit_10pct": round(crit_vals[2], 4),
                "cointegrated": p_val < p_threshold,
            }
        )

    df = pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)
    return df


def best_pair(
    prices: pd.DataFrame,
    p_threshold: float = COINT_P_VALUE_THRESHOLD,
    strict: bool = False,
) -> Tuple[str, str, float]:
    """
    Return (ticker_a, ticker_b, p_value) for the most cointegrated pair.

    Parameters
    ----------
    strict : bool
        If True, raise ValueError when no pair passes *p_threshold*.
        If False (default), fall back to the lowest-p pair regardless.
    """
    results = run_cointegration_scan(prices, p_threshold)
    valid = results[results["cointegrated"]]

    if valid.empty:
        if strict:
            raise ValueError(
                f"No cointegrated pair found at p < {p_threshold}.  "
                "Try a different sector or longer history."
            )
        # Fallback: use the pair with the lowest p-value even if > threshold
        top = results.iloc[0]
    else:
        top = valid.iloc[0]

    return top["ticker_a"], top["ticker_b"], top["p_value"]


# ──────────────────────────────────────────────
# ADF test helper (used for diagnostics)
# ──────────────────────────────────────────────

def adf_test(series: pd.Series) -> Dict[str, float]:
    """Run Augmented Dickey-Fuller and return a tidy dict."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 6),
        "used_lag": result[2],
        "n_obs": result[3],
        "crit_1pct": round(result[4]["1%"], 4),
        "crit_5pct": round(result[4]["5%"], 4),
        "crit_10pct": round(result[4]["10%"], 4),
    }


if __name__ == "__main__":
    from data_ingestion import download_prices

    prices = download_prices()
    scan = run_cointegration_scan(prices)
    print(scan.to_string(index=False))
    print()

    a, b, p = best_pair(prices)
    print(f"Best pair: {a} & {b}  (p = {p:.6f})")
