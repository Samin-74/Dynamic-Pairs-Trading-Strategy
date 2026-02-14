# Dynamic Pairs Trading Strategy using Kalman Filters

An algorithmic trading engine that identifies mispriced assets via statistical cointegration and uses a **Kalman Filter** to dynamically adjust the hedge ratio in real-time — visualised on an interactive Streamlit dashboard.

---

## Key Concepts

| Concept | Description |
|---|---|
| **Cointegration** | Two price series are "tethered" long-term (Engle-Granger test, p < 0.05) |
| **Kalman Filter** | Online state-space model that estimates a time-varying hedge ratio from noisy prices |
| **Z-Score Signals** | Normalised spread converted into probabilistic entry / exit signals |

## Technology Stack

| Component | Technology |
|---|---|
| Data Source | `yfinance` / `pandas` |
| Statistics | `statsmodels` (ADF, Cointegration) |
| Dynamic Model | Custom NumPy Kalman Filter |
| Visualisation | `Streamlit` + `Plotly` |
| Backtesting | Vectorised `pandas` simulation |

---

## Features

- **Fully configurable** — tune every parameter from the sidebar without touching code
- **Initial equity control** — set your starting capital ($1 K – $10 M)
- **Transaction costs** — realistic slippage / commission modelling (0–50 bps)
- **Kalman delta tuning** — control how fast the hedge ratio adapts to regime shifts
- **Extended KPIs** — Win Rate, Profit Factor, Avg Holding Period, P&L breakdown
- **Monthly returns heatmap** — colour-coded calendar view of strategy performance
- **Rolling Sharpe ratio** — 60-day rolling risk-adjusted return chart
- **Correlation matrix** — visualise inter-ticker dependencies
- **Manual pair override** — pick any pair from the scanned universe
- **CSV export** — download signals and equity curve data
- **Landing page** — quick-start guide shown before first run

---

## Project Structure

```
Pairs Trading Algo/
├── app.py                   # Streamlit dashboard (entry point)
├── config.py                # All tuneable parameters
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── data_ingestion.py    # Phase 1 – download prices via yfinance
    ├── cointegration.py     # Phase 1 – Engle-Granger scan
    ├── kalman_filter.py     # Phase 2 – online Kalman Filter
    ├── signals.py           # Phase 3 – Z-Score & signal generation
    └── backtester.py        # Phase 3/4 – vectorised PnL simulation
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

The sidebar lets you:

- Pick tickers (default: JPM, BAC, WFC, C, GS, MS)
- Set years of history, Z-Score lookback, entry/exit thresholds
- Configure starting equity, transaction costs, risk-free rate
- Adjust Kalman Filter sensitivity (delta)
- Press **Run Strategy** to execute the full pipeline

---

## Implementation Phases

### Phase 1 – Pair Discovery
- Downloads daily closes for a configurable sector.
- Runs Engle-Granger cointegration on every unique pair.
- Selects the pair with the lowest p-value (< 0.05).

### Phase 2 – Dynamic Hedge Ratio (Kalman Filter)
- State: `[intercept, hedge_ratio]` — modelled as a random walk.
- Observation: `Price_A ≈ intercept + hedge_ratio × Price_B`.
- Updates day-by-day; adapts to regime changes automatically.

### Phase 3 – Signal Generation & Backtesting
- **Spread**: $Spread_t = Price_{A,t} - (HedgeRatio_t \times Price_{B,t})$
- **Z-Score**: $(Spread_t - \mu_t) / \sigma_t$ (rolling window)
- **Rules**:
  - Short signal: Z > +2.0
  - Long signal: Z < −2.0
  - Exit signal: |Z| ≤ exit threshold or zero-crossing
- Dollar-neutral backtest with equity curve, drawdown, and trade counting.
- Transaction costs applied at each position change.

### Phase 4 – Analyst Dashboard
- Normalised price chart of the selected pair.
- Live hedge-ratio plot.
- Spread & Z-Score chart with green/red shading for long/short positions.
- Equity curve & drawdown chart.
- KPI panels: Cumulative Return, Annualised Return, Sharpe Ratio, Max Drawdown, Trade Count.
- Extended metrics: Win Rate, Profit Factor, Avg Holding Days, Total P&L.
- Monthly returns heatmap with annual summary.
- Rolling 60-day Sharpe ratio chart.
- Full cointegration scan table.
- Correlation matrix heatmap.
- Trade log with entry/exit timestamps and CSV export.

---

## Configuration

All parameters live in [`config.py`](config.py) and can also be adjusted from the Streamlit sidebar:

| Parameter | Default | Description |
|---|---|---|
| `SECTOR_TICKERS` | JPM, BAC, WFC, C, GS, MS | Tickers to scan |
| `DATA_PERIOD_YEARS` | 2 | Years of history |
| `COINT_P_VALUE_THRESHOLD` | 0.05 | Cointegration significance |
| `KALMAN_DELTA` | 1e-5 | Process noise scaling |
| `ZSCORE_ENTRY_THRESHOLD` | 2.0 | Open position threshold |
| `ZSCORE_EXIT_THRESHOLD` | 0.5 | Close position threshold |
| `SPREAD_LOOKBACK` | 30 | Rolling window (days) |
| `INITIAL_CAPITAL` | $100,000 | Backtest starting equity |
| `TRANSACTION_COST_BPS` | 0 | One-way cost in basis points |
| `RISK_FREE_RATE` | 0.04 | Annual risk-free rate |

---

## License

MIT
