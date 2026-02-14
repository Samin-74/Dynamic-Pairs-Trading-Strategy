"""
Phase 3 / 4 – Backtesting Engine
Vectorised simulation of the pairs-trading strategy over historic data.
Computes PnL, cumulative return, Sharpe ratio, and max drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, TRANSACTION_COST_BPS


# ──────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Holds all outputs of a backtest run."""

    daily: pd.DataFrame        # day-level detail (positions, returns, equity)
    cumulative_return: float   # total return %
    annualised_return: float
    sharpe_ratio: float
    max_drawdown: float        # as a negative percentage
    n_trades: int              # round-trip count
    # Extended KPIs
    win_rate: float            # % of winning round-trips
    profit_factor: float       # gross profit / gross loss
    avg_holding_days: float    # mean duration of a position
    total_pnl: float           # dollar PnL
    winning_trades: int
    losing_trades: int


# ──────────────────────────────────────────────
# Core back-tester
# ──────────────────────────────────────────────

def run_backtest(
    prices_a: pd.Series,
    prices_b: pd.Series,
    signals_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    risk_free: float = RISK_FREE_RATE,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> BacktestResult:
    """
    Simulate the strategy and return a ``BacktestResult``.

    The strategy is "dollar-neutral":
      * When position = +1 → long 1 unit of A, short hedge_ratio units of B.
      * When position = -1 → short 1 unit of A, long hedge_ratio units of B.

    Parameters
    ----------
    prices_a, prices_b : pd.Series
        Aligned price series for the two assets.
    signals_df : pd.DataFrame
        Must contain columns: ``spread``, ``hedge_ratio``, ``position``.
    initial_capital : float
        Starting equity in dollars.
    risk_free : float
        Annualised risk-free rate for Sharpe calculation.
    transaction_cost_bps : float
        One-way transaction cost in basis points (e.g. 5 = 0.05 %).
    """
    df = signals_df[["spread", "hedge_ratio", "position"]].copy()
    df = df.dropna()

    # Daily return on A and B
    ret_a = prices_a.pct_change().reindex(df.index)
    ret_b = prices_b.pct_change().reindex(df.index)

    # Portfolio return when position is active.
    hr = df["hedge_ratio"].shift(1).abs()
    raw_spread_ret = ret_a - df["hedge_ratio"].shift(1) * ret_b
    normalised = raw_spread_ret / (1.0 + hr)

    df["port_return"] = df["position"].shift(1) * normalised
    df["port_return"] = df["port_return"].fillna(0.0)

    # ── Transaction costs ──
    tc_rate = transaction_cost_bps / 10_000.0  # bps → decimal
    position_changes = df["position"].diff().fillna(0)
    # Cost proportional to the magnitude of position change (0→±1 = 1, ±1→∓1 = 2)
    df["trade_cost"] = position_changes.abs() * tc_rate
    df["port_return"] = df["port_return"] - df["trade_cost"]

    # Equity curve
    df["equity"] = initial_capital * (1.0 + df["port_return"]).cumprod()

    # Drawdown
    running_max = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - running_max) / running_max

    # ── Core KPIs ──
    total_return = df["equity"].iloc[-1] / initial_capital - 1.0
    n_days = len(df)
    ann_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / max(n_days, 1)) - 1.0

    daily_ret = df["port_return"]
    excess = daily_ret - risk_free / TRADING_DAYS_PER_YEAR
    sharpe = (
        np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std()
        if excess.std() != 0
        else 0.0
    )
    max_dd = df["drawdown"].min()

    # Trade count (a "trade" = transition from 0 → ±1)
    n_trades = int((position_changes != 0).sum()) // 2  # entry + exit = 1 round trip

    # ── Extended KPIs (per-trade analysis) ──
    # Identify individual round-trip trades and their PnL
    trade_returns = []
    trade_durations = []
    in_trade = False
    trade_start = None
    trade_ret_accum = 0.0

    for i in range(len(df)):
        pos = df["position"].iloc[i]
        prev_pos = df["position"].iloc[i - 1] if i > 0 else 0

        if not in_trade and pos != 0:
            # Entry
            in_trade = True
            trade_start = i
            trade_ret_accum = df["port_return"].iloc[i]
        elif in_trade and pos != 0:
            # Holding
            trade_ret_accum += df["port_return"].iloc[i]
        elif in_trade and pos == 0:
            # Exit
            trade_ret_accum += df["port_return"].iloc[i]
            trade_returns.append(trade_ret_accum)
            trade_durations.append(i - trade_start)
            in_trade = False
            trade_ret_accum = 0.0

    # If still in a trade at the end, close it
    if in_trade:
        trade_returns.append(trade_ret_accum)
        trade_durations.append(len(df) - 1 - (trade_start or 0))

    winners = [r for r in trade_returns if r > 0]
    losers = [r for r in trade_returns if r <= 0]
    n_winners = len(winners)
    n_losers = len(losers)
    win_rate = n_winners / max(len(trade_returns), 1) * 100.0
    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    avg_holding = np.mean(trade_durations) if trade_durations else 0.0
    total_pnl = df["equity"].iloc[-1] - initial_capital

    return BacktestResult(
        daily=df,
        cumulative_return=round(total_return * 100, 2),
        annualised_return=round(ann_return * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown=round(max_dd * 100, 2),
        n_trades=n_trades,
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        avg_holding_days=round(avg_holding, 1),
        total_pnl=round(total_pnl, 2),
        winning_trades=n_winners,
        losing_trades=n_losers,
    )


if __name__ == "__main__":
    from data_ingestion import download_prices
    from cointegration import best_pair
    from kalman_filter import KalmanPairFilter
    from signals import build_signals

    prices = download_prices()
    a, b, _ = best_pair(prices)

    kf = KalmanPairFilter()
    kdf = kf.fit(prices[a], prices[b])
    sig = build_signals(kdf)

    result = run_backtest(prices[a], prices[b], sig)

    print(f"Pair         : {a} / {b}")
    print(f"Cum. Return  : {result.cumulative_return}%")
    print(f"Ann. Return  : {result.annualised_return}%")
    print(f"Sharpe Ratio : {result.sharpe_ratio}")
    print(f"Max Drawdown : {result.max_drawdown}%")
    print(f"# Trades     : {result.n_trades}")
    print(f"Win Rate     : {result.win_rate}%")
    print(f"Profit Factor: {result.profit_factor}")
    print(f"Avg Holding  : {result.avg_holding_days} days")
    print(f"Total P&L    : ${result.total_pnl:,.2f}")
