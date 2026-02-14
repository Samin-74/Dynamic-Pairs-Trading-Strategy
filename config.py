"""
Configuration for the Dynamic Pairs Trading Strategy.
"""

# ──────────────────────────────────────────────
# Data Settings
# ──────────────────────────────────────────────
SECTOR_TICKERS = ["JPM", "BAC", "WFC", "C", "GS", "MS"]
DATA_PERIOD_YEARS = 2  # years of historical data to download
DATA_INTERVAL = "1d"   # daily bars

# ──────────────────────────────────────────────
# Cointegration Settings
# ──────────────────────────────────────────────
COINT_P_VALUE_THRESHOLD = 0.05  # reject H0 when p < threshold

# ──────────────────────────────────────────────
# Kalman Filter Settings
# ──────────────────────────────────────────────
KALMAN_DELTA = 1e-5          # transition covariance scaling
KALMAN_OBS_COV = 1.0         # observation noise covariance
KALMAN_INITIAL_STATE_MEAN = [0.0, 0.0]  # [intercept, hedge_ratio]
KALMAN_INITIAL_STATE_COV = 1.0          # initial state covariance scalar

# ──────────────────────────────────────────────
# Signal / Trading Settings
# ──────────────────────────────────────────────
ZSCORE_ENTRY_THRESHOLD = 2.0   # open position when |z| > threshold
ZSCORE_EXIT_THRESHOLD = 0.5    # close position when |z| < threshold (band around 0)
SPREAD_LOOKBACK = 30           # rolling window for mean / std of spread

# ──────────────────────────────────────────────
# Backtesting Settings
# ──────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_BPS = 0       # one-way cost in basis points (0 = frictionless)
RISK_FREE_RATE = 0.04          # annualised, for Sharpe calc
TRADING_DAYS_PER_YEAR = 252

# ──────────────────────────────────────────────
# Dashboard / Streamlit Settings
# ──────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "Pairs Trading – Kalman Filter Strategy"
STREAMLIT_LAYOUT = "wide"
