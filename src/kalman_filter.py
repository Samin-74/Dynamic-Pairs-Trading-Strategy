"""
Phase 2 – Kalman Filter for Dynamic Hedge Ratio
Implements an online Kalman Filter (state-space model) where:
  * Hidden state = [intercept, hedge_ratio]
  * Observation  = Price_A  ≈  intercept + hedge_ratio * Price_B
The filter updates day-by-day and outputs a time-varying hedge ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    KALMAN_DELTA,
    KALMAN_INITIAL_STATE_COV,
    KALMAN_INITIAL_STATE_MEAN,
    KALMAN_OBS_COV,
)


class KalmanPairFilter:
    """
    Online Kalman Filter for estimating a dynamic hedge ratio between two
    cointegrated price series.

    State vector:  x_t = [intercept_t, hedge_ratio_t]ᵀ
    Observation:   y_t = Price_A_t
    Design matrix: H_t = [1, Price_B_t]

    Transition:    x_t = x_{t-1}  (random walk)
    """

    def __init__(
        self,
        delta: float = KALMAN_DELTA,
        obs_cov: float = KALMAN_OBS_COV,
        initial_state_mean: list | None = None,
        initial_state_cov: float = KALMAN_INITIAL_STATE_COV,
    ):
        self.delta = delta
        self.obs_cov = obs_cov

        # ── State initialisation ──
        self.x = np.array(
            initial_state_mean or KALMAN_INITIAL_STATE_MEAN, dtype=float
        )
        self.P = np.eye(2) * initial_state_cov

        # Transition covariance (random-walk noise)
        self.Q = self.delta / (1.0 - self.delta) * np.eye(2)

        # Observation noise covariance (scalar, wrapped for matrix ops)
        self.R = np.array([[obs_cov]])

        # Storage
        self.intercepts: list[float] = []
        self.hedge_ratios: list[float] = []
        self.spread: list[float] = []

    # ─────────────────────────────────────────
    # Core update step
    # ─────────────────────────────────────────
    def update(self, price_a: float, price_b: float) -> None:
        """
        Run one Kalman predict → update cycle.

        Parameters
        ----------
        price_a : float   Dependent asset price (y).
        price_b : float   Independent asset price (used in observation matrix).
        """
        # Design / observation vector
        H = np.array([[1.0, price_b]])  # (1, 2)

        # ── Predict ──
        x_pred = self.x                     # random walk → same mean
        P_pred = self.P + self.Q            # add process noise

        # ── Update ──
        y = np.array([price_a])             # observation
        y_hat = H @ x_pred                  # predicted observation
        e = y - y_hat                        # innovation (spread)

        S = H @ P_pred @ H.T + self.R       # innovation covariance (1×1)
        K = P_pred @ H.T @ np.linalg.inv(S) # Kalman gain (2×1)

        self.x = x_pred + (K @ e).flatten()
        self.P = P_pred - K @ H @ P_pred

        # ── Store results ──
        self.intercepts.append(self.x[0])
        self.hedge_ratios.append(self.x[1])
        self.spread.append(float(e[0]))

    # ─────────────────────────────────────────
    # Convenience: run over full series
    # ─────────────────────────────────────────
    def fit(self, price_a: pd.Series, price_b: pd.Series) -> pd.DataFrame:
        """
        Run the filter across the full history and return a tidy DataFrame.

        Returns columns: date, intercept, hedge_ratio, spread
        """
        for pa, pb in zip(price_a.values, price_b.values):
            self.update(pa, pb)

        df = pd.DataFrame(
            {
                "date": price_a.index,
                "intercept": self.intercepts,
                "hedge_ratio": self.hedge_ratios,
                "spread": self.spread,
            }
        )
        df.set_index("date", inplace=True)
        return df


if __name__ == "__main__":
    from data_ingestion import download_prices
    from cointegration import best_pair

    prices = download_prices()
    a, b, p = best_pair(prices)
    print(f"Running Kalman filter on {a} / {b}  (coint p = {p:.6f})")

    kf = KalmanPairFilter()
    result = kf.fit(prices[a], prices[b])
    print(result.tail(10))
