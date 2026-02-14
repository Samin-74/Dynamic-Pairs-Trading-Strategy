"""
Phase 3 – Signal Generation
Converts the Kalman Filter spread into Z-Scores and emits trading signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SPREAD_LOOKBACK, ZSCORE_ENTRY_THRESHOLD, ZSCORE_EXIT_THRESHOLD


# ──────────────────────────────────────────────
# Z-Score calculation
# ──────────────────────────────────────────────

def compute_zscore(spread: pd.Series, lookback: int = SPREAD_LOOKBACK) -> pd.Series:
    """
    Rolling Z-Score:  (spread - rolling_mean) / rolling_std
    """
    mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    std = spread.rolling(window=lookback, min_periods=lookback).std()
    # Avoid division by zero
    std = std.replace(0.0, np.nan)
    zscore = (spread - mean) / std
    return zscore


# ──────────────────────────────────────────────
# Signal generation
# ──────────────────────────────────────────────

def generate_signals(
    zscore: pd.Series,
    entry: float = ZSCORE_ENTRY_THRESHOLD,
    exit_: float = ZSCORE_EXIT_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate a trading-signal DataFrame from Z-Scores.

    Signal conventions (for asset A in the pair):
        +1  → Long A, Short B   (z < -entry)
        -1  → Short A, Long B   (z >  entry)
         0  → Flat / exit       (|z| crosses exit threshold)

    Returns
    -------
    pd.DataFrame  with columns: zscore, raw_signal, position
    """
    raw = pd.Series(np.nan, index=zscore.index, name="raw_signal")

    raw[zscore > entry] = -1.0      # spread too wide  → short A
    raw[zscore < -entry] = 1.0      # spread too narrow → long A

    # Exit condition: |z| falls within the exit band around zero,
    # OR a zero-crossing occurs (z changes sign).
    exit_band = max(abs(exit_), 1e-6)  # guard against == 0 edge case
    zero_cross = (zscore * zscore.shift(1) < 0)  # sign change
    exit_mask = (zscore.abs() <= exit_band) | zero_cross
    raw[exit_mask] = 0.0

    # Forward-fill to stay in position until an exit signal appears
    position = raw.ffill().fillna(0.0).astype(int)

    signals = pd.DataFrame(
        {
            "zscore": zscore,
            "raw_signal": raw,
            "position": position,
        }
    )
    return signals


# ──────────────────────────────────────────────
# Convenience: full pipeline from Kalman output
# ──────────────────────────────────────────────

def build_signals(
    kalman_df: pd.DataFrame,
    lookback: int = SPREAD_LOOKBACK,
    entry: float = ZSCORE_ENTRY_THRESHOLD,
    exit_: float = ZSCORE_EXIT_THRESHOLD,
) -> pd.DataFrame:
    """
    End-to-end: takes the DataFrame produced by ``KalmanPairFilter.fit()``
    and appends zscore / signal columns.
    """
    zscore = compute_zscore(kalman_df["spread"], lookback)
    sigs = generate_signals(zscore, entry, exit_)
    return pd.concat([kalman_df, sigs], axis=1)


if __name__ == "__main__":
    from data_ingestion import download_prices
    from cointegration import best_pair
    from kalman_filter import KalmanPairFilter

    prices = download_prices()
    a, b, _ = best_pair(prices)

    kf = KalmanPairFilter()
    kdf = kf.fit(prices[a], prices[b])
    result = build_signals(kdf)
    print(result.dropna().tail(20))
