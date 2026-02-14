"""
Phase 1 – Data Ingestion
Downloads historical daily closes for a list of tickers via yfinance.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pandas as pd
import yfinance as yf

from config import DATA_PERIOD_YEARS, DATA_INTERVAL, SECTOR_TICKERS


def download_prices(
    tickers: Optional[List[str]] = None,
    years: int = DATA_PERIOD_YEARS,
    interval: str = DATA_INTERVAL,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily *Adj Close* prices, one column per ticker.

    Parameters
    ----------
    tickers : list[str] | None
        Symbols to download.  Defaults to ``SECTOR_TICKERS`` from config.
    years : int
        How many years of history to fetch.
    interval : str
        Bar size (``"1d"``, ``"1h"``, …).

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex, columns = ticker symbols.
    """
    tickers = tickers or SECTOR_TICKERS
    end = dt.datetime.today()
    start = end - dt.timedelta(days=years * 365)

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    # yf.download returns multi-level columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        # Try "Close" first, fall back to first level key
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            first_key = raw.columns.get_level_values(0).unique()[0]
            prices = raw[first_key].copy()
    else:
        # Single ticker: ensure we end up with a 1-column DataFrame
        if isinstance(raw, pd.Series):
            prices = raw.to_frame(name=tickers[0])
        else:
            prices = raw[["Close"]].copy()
            prices.columns = tickers if len(tickers) == 1 else [tickers[0]]

    prices.dropna(how="all", inplace=True)
    prices.ffill(inplace=True)
    prices.dropna(inplace=True)

    return prices


if __name__ == "__main__":
    df = download_prices()
    print(f"Downloaded {len(df)} rows for {list(df.columns)}")
    print(df.tail())
