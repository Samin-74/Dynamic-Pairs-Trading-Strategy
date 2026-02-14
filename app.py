"""
Phase 4 – Streamlit Analyst Dashboard
Interactive visualisation of the pairs trading strategy.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os

# Ensure src/ is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from config import (
    SECTOR_TICKERS,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_TITLE,
    ZSCORE_ENTRY_THRESHOLD,
    ZSCORE_EXIT_THRESHOLD,
    SPREAD_LOOKBACK,
    INITIAL_CAPITAL,
    KALMAN_DELTA,
    RISK_FREE_RATE,
    TRANSACTION_COST_BPS,
    TRADING_DAYS_PER_YEAR,
)
from src.data_ingestion import download_prices
from src.cointegration import run_cointegration_scan, best_pair
from src.kalman_filter import KalmanPairFilter
from src.signals import build_signals
from src.backtester import run_backtest

# ──────────────────────────────────────────────
# Page config & custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    layout=STREAMLIT_LAYOUT,
    page_icon="📈",
)

st.markdown(
    """
    <style>
    /* ── KPI cards: equal height, centred, consistent spacing ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%);
        border: 1px solid rgba(100, 120, 200, 0.12);
        border-radius: 10px;
        padding: 18px 16px 14px 16px;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }
    /* Label row */
    div[data-testid="stMetric"] label {
        font-size: 0.74rem !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        opacity: 0.65;
    }
    /* Value row – prevent wrapping */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700;
        white-space: nowrap;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Dynamic Pairs Trading Strategy")
st.caption("Kalman Filter · Cointegration · Z-Score Signals")

# ──────────────────────────────────────────────
# Sidebar – user controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    st.subheader("Data")
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value=", ".join(SECTOR_TICKERS),
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    years = st.slider("Years of data", 1, 10, 2)

    st.markdown("---")
    st.subheader("Strategy")
    lookback = st.slider("Z-Score lookback (days)", 10, 120, SPREAD_LOOKBACK)
    entry_z = st.slider("Entry |Z| threshold", 1.0, 4.0, ZSCORE_ENTRY_THRESHOLD, 0.1)
    exit_z = st.slider("Exit |Z| threshold", 0.0, 2.0, ZSCORE_EXIT_THRESHOLD, 0.1)

    st.markdown("---")
    st.subheader("Capital & Costs")
    initial_capital = st.number_input(
        "Initial equity ($)",
        min_value=1_000,
        max_value=10_000_000,
        value=int(INITIAL_CAPITAL),
        step=10_000,
        help="Starting portfolio value in dollars.",
    )
    transaction_cost_bps = st.slider(
        "Transaction cost (bps)",
        min_value=0,
        max_value=50,
        value=int(TRANSACTION_COST_BPS),
        step=1,
        help="One-way cost per trade in basis points (1 bp = 0.01 %).",
    )
    risk_free_rate = st.slider(
        "Risk-free rate (annual %)",
        min_value=0.0,
        max_value=10.0,
        value=RISK_FREE_RATE * 100,
        step=0.25,
        help="Used for Sharpe ratio calculation.",
    ) / 100.0  # convert back to decimal

    st.markdown("---")
    st.subheader("Advanced")
    kalman_delta = st.select_slider(
        "Kalman delta (process noise)",
        options=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=KALMAN_DELTA,
        format_func=lambda x: f"{x:.0e}",
        help="Controls how fast the hedge ratio adapts. Larger = more responsive.",
    )
    allow_pair_override = st.checkbox("Manually select pair", value=False)

    st.markdown("---")
    run_btn = st.button("🚀  Run Strategy", type="primary", use_container_width=True)

    st.markdown("---")
    with st.expander("ℹ️  How it works"):
        st.markdown(
            """
            1. **Download** daily closes for the chosen tickers.
            2. **Cointegration scan** — Engle-Granger test on every pair.
            3. **Kalman Filter** estimates a time-varying hedge ratio.
            4. **Z-Score** of the spread triggers entry / exit signals.
            5. **Backtest** simulates a dollar-neutral strategy.
            """
        )


# ──────────────────────────────────────────────
# Cached helpers
# ──────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading price data …")
def _download(tickers: tuple, years: int):
    """Cache-friendly wrapper (tuple is hashable)."""
    return download_prices(list(tickers), years)



# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

if run_btn or "result" in st.session_state:
    if run_btn:
        # ── Step 1: Data ──
        with st.spinner("Downloading price data …"):
            prices = _download(tuple(tickers), years)
        st.session_state["prices"] = prices

        # ── Step 2: Cointegration ──
        with st.spinner("Running cointegration scan …"):
            scan = run_cointegration_scan(prices)
        st.session_state["scan"] = scan

        # Pair selection (auto or manual)
        if allow_pair_override:
            st.session_state["pair_override"] = True
        else:
            st.session_state["pair_override"] = False
            try:
                a, b, p_val = best_pair(prices)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            st.session_state["pair"] = (a, b, p_val)
            if p_val >= 0.05:
                st.session_state["pair_warn"] = True
            else:
                st.session_state["pair_warn"] = False

            # ── Step 3: Kalman filter ──
            kf = KalmanPairFilter(delta=kalman_delta)
            kdf = kf.fit(prices[a], prices[b])
            sig = build_signals(kdf, lookback=lookback, entry=entry_z, exit_=exit_z)
            st.session_state["signals"] = sig

            # ── Step 4: Backtest ──
            bt = run_backtest(
                prices[a], prices[b], sig,
                initial_capital=float(initial_capital),
                risk_free=risk_free_rate,
                transaction_cost_bps=float(transaction_cost_bps),
            )
            st.session_state["result"] = bt

    # ─────────────────────────────────────
    # Manual pair override (shown after scan)
    # ─────────────────────────────────────
    prices = st.session_state["prices"]
    scan = st.session_state["scan"]

    if st.session_state.get("pair_override"):
        st.markdown("---")
        st.subheader("Select Pair Manually")
        avail_tickers = sorted(prices.columns.tolist())
        c1, c2 = st.columns(2)
        sel_a = c1.selectbox("Asset A", avail_tickers, index=0)
        remaining = [t for t in avail_tickers if t != sel_a]
        sel_b = c2.selectbox("Asset B", remaining, index=0)

        if st.button("▶ Run with selected pair", use_container_width=True):
            # Get p-value from scan table
            row = scan[
                ((scan["ticker_a"] == sel_a) & (scan["ticker_b"] == sel_b))
                | ((scan["ticker_a"] == sel_b) & (scan["ticker_b"] == sel_a))
            ]
            p_val = float(row["p_value"].iloc[0]) if not row.empty else 1.0
            st.session_state["pair"] = (sel_a, sel_b, p_val)

            kf = KalmanPairFilter(delta=kalman_delta)
            kdf = kf.fit(prices[sel_a], prices[sel_b])
            sig = build_signals(kdf, lookback=lookback, entry=entry_z, exit_=exit_z)
            st.session_state["signals"] = sig
            bt = run_backtest(
                prices[sel_a], prices[sel_b], sig,
                initial_capital=float(initial_capital),
                risk_free=risk_free_rate,
                transaction_cost_bps=float(transaction_cost_bps),
            )
            st.session_state["result"] = bt
            st.session_state["pair_override"] = False
            st.rerun()
        else:
            # Show scan table while waiting for selection
            st.dataframe(scan, width="stretch")
            st.stop()

    # ─────────────────────────────────────
    # Results display
    # ─────────────────────────────────────
    a, b, p_val = st.session_state["pair"]
    sig = st.session_state["signals"]
    bt = st.session_state["result"]

    # ── KPI row ──
    st.markdown("---")
    st.subheader(f"📊  {a}  &  {b}  ·  Cointegration p = {p_val:.6f}")

    if st.session_state.get("pair_warn", False):
        st.warning(
            f"No pair passed the p < 0.05 threshold. "
            f"Using the best available pair ({a} & {b}, p = {p_val:.4f}). "
            f"Consider adding more tickers or extending the history window."
        )
    elif p_val < 0.01:
        st.success(f"Strong cointegration detected (p = {p_val:.6f})")

    k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
    k1.metric("Cumulative Return", f"{bt.cumulative_return:+.2f} %")
    k2.metric("Annualised Return", f"{bt.annualised_return:+.2f} %")
    k3.metric("Sharpe Ratio", f"{bt.sharpe_ratio:.3f}")
    k4.metric("Max Drawdown", f"{bt.max_drawdown:.2f} %")
    k5.metric("Round-trip Trades", f"{bt.n_trades}")

    # ── Extended KPIs ──
    with st.expander("📋  Extended Performance Metrics", expanded=False):
        e1, e2, e3, e4, e5, e6 = st.columns(6, gap="medium")
        e1.metric("Total P&L", f"${bt.total_pnl:+,.2f}")
        e2.metric("Win Rate", f"{bt.win_rate:.1f} %")
        e3.metric("Profit Factor", f"{bt.profit_factor:.2f}" if bt.profit_factor != float("inf") else "∞")
        e4.metric("Avg Holding", f"{bt.avg_holding_days:.0f} days")
        e5.metric("Winning Trades", f"{bt.winning_trades}")
        e6.metric("Losing Trades", f"{bt.losing_trades}")

    # ── Tabbed charts ──
    st.markdown("---")
    tab_charts, tab_signals, tab_monthly, tab_scan, tab_data = st.tabs(
        ["📈 Charts", "⚡ Signals Detail", "📅 Monthly Returns", "🔬 Cointegration Scan", "📋 Raw Data"]
    )

    # ═══════════════════════════════════════
    # TAB 1 – Charts
    # ═══════════════════════════════════════
    with tab_charts:

        # -- Chart 1: Normalised prices --
        st.markdown("#### Normalised Price Series")
        norm = prices[[a, b]].div(prices[[a, b]].iloc[0]) * 100
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(x=norm.index, y=norm[a], name=a,
                       line=dict(width=2, color="#636EFA"))
        )
        fig_price.add_trace(
            go.Scatter(x=norm.index, y=norm[b], name=b,
                       line=dict(width=2, color="#EF553B"))
        )
        fig_price.update_layout(
            yaxis_title="Normalised Price (base = 100)",
            template="plotly_white",
            height=370,
            margin=dict(t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # -- Chart 2: Hedge ratio --
        st.markdown("#### Dynamic Hedge Ratio (Kalman Filter)")
        fig_hr = go.Figure()
        fig_hr.add_trace(
            go.Scatter(
                x=sig.index, y=sig["hedge_ratio"],
                name="Hedge Ratio",
                line=dict(color="#AB63FA", width=2),
                fill="tozeroy",
                fillcolor="rgba(171,99,250,0.07)",
            )
        )
        fig_hr.update_layout(
            yaxis_title="Hedge Ratio",
            template="plotly_white",
            height=300,
            margin=dict(t=30, b=30),
            hovermode="x unified",
        )
        st.plotly_chart(fig_hr, use_container_width=True)

        # -- Chart 3: Equity curve & Drawdown --
        st.markdown("#### Equity Curve & Drawdown")
        daily = bt.daily

        fig_eq = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.06,
            subplot_titles=("Portfolio Equity", "Drawdown"),
        )
        fig_eq.add_trace(
            go.Scatter(
                x=daily.index, y=daily["equity"], name="Equity",
                line=dict(color="royalblue", width=2),
                fill="tozeroy", fillcolor="rgba(65,105,225,0.08)",
            ),
            row=1, col=1,
        )
        fig_eq.add_hline(
            y=float(initial_capital), line_dash="dot", line_color="grey",
            annotation_text="Starting Capital", row=1, col=1,
        )
        fig_eq.add_trace(
            go.Scatter(
                x=daily.index, y=daily["drawdown"] * 100, name="Drawdown %",
                line=dict(color="crimson", width=1.5),
                fill="tozeroy", fillcolor="rgba(220,20,60,0.10)",
            ),
            row=2, col=1,
        )
        fig_eq.update_yaxes(title_text="$", tickprefix="$", tickformat=",.0f", row=1, col=1)
        fig_eq.update_yaxes(title_text="%", ticksuffix="%", row=2, col=1)
        fig_eq.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(t=40, b=30),
            hovermode="x unified",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ═══════════════════════════════════════
    # TAB 2 – Signals Detail
    # ═══════════════════════════════════════
    with tab_signals:
        st.markdown("#### Spread & Z-Score with Trading Signals")
        valid = sig.dropna(subset=["zscore"])

        fig_sig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.55],
            vertical_spacing=0.07,
            subplot_titles=("Kalman Spread", "Z-Score & Signals"),
        )

        # -- Spread --
        fig_sig.add_trace(
            go.Scatter(x=valid.index, y=valid["spread"], name="Spread",
                       line=dict(color="steelblue", width=1.5)),
            row=1, col=1,
        )
        fig_sig.add_hline(y=0, line_dash="dot", line_color="grey", row=1, col=1)

        # -- Z-Score --
        fig_sig.add_trace(
            go.Scatter(x=valid.index, y=valid["zscore"], name="Z-Score",
                       line=dict(color="darkorange", width=1.5)),
            row=2, col=1,
        )

        # Thresholds
        fig_sig.add_hline(y=entry_z, line_dash="dash", line_color="red",
                          annotation_text=f"Short entry (+{entry_z})",
                          annotation_font_size=11, row=2, col=1)
        fig_sig.add_hline(y=-entry_z, line_dash="dash", line_color="green",
                          annotation_text=f"Long entry (−{entry_z})",
                          annotation_font_size=11, row=2, col=1)
        fig_sig.add_hline(y=exit_z, line_dash="dot", line_color="grey",
                          annotation_text=f"Exit (+{exit_z})",
                          annotation_font_size=10, row=2, col=1)
        fig_sig.add_hline(y=-exit_z, line_dash="dot", line_color="grey",
                          annotation_text=f"Exit (−{exit_z})",
                          annotation_font_size=10, row=2, col=1)

        # Background shading for positions
        pos = valid["position"]
        in_long = pos == 1
        in_short = pos == -1

        def _shade_regions(mask, color, label, fig, row):
            """Add vertical rectangles for consecutive True runs."""
            diff = mask.astype(int).diff().fillna(0)
            starts = valid.index[diff == 1]
            ends = valid.index[diff == -1]
            # If a region is still open at the end, close it at the last date
            if mask.iloc[-1]:
                ends = ends.union(pd.DatetimeIndex([valid.index[-1]]))
            for s, e in zip(starts, ends):
                fig.add_vrect(x0=s, x1=e, fillcolor=color, line_width=0, row=row, col=1)

        _shade_regions(in_long, "rgba(0,180,0,0.10)", "Long", fig_sig, 2)
        _shade_regions(in_short, "rgba(220,0,0,0.10)", "Short", fig_sig, 2)

        fig_sig.update_layout(
            template="plotly_white",
            height=600,
            margin=dict(t=40, b=30),
            showlegend=True,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        # Trade log
        st.markdown("#### Trade Log")
        pos_diff = valid["position"].diff().fillna(0)
        trades = valid[pos_diff != 0][["zscore", "position"]].copy()
        trades["action"] = trades["position"].map({1: "🟢 Long A / Short B", -1: "🔴 Short A / Long B", 0: "⚪ Exit"})
        st.dataframe(trades[["action", "zscore"]], width="stretch", height=300)

    # ═══════════════════════════════════════
    # TAB 3 – Monthly Returns Heatmap
    # ═══════════════════════════════════════
    with tab_monthly:
        st.markdown("#### Monthly Strategy Returns (%)")
        daily = bt.daily
        monthly_ret = daily["port_return"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100.0

        if len(monthly_ret) > 0:
            month_df = pd.DataFrame({
                "Year": monthly_ret.index.year,
                "Month": monthly_ret.index.month,
                "Return": monthly_ret.values,
            })
            pivot = month_df.pivot_table(index="Year", columns="Month", values="Return", aggfunc="sum")
            pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

            fig_heat = px.imshow(
                pivot.values,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index.tolist()],
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                text_auto=".2f",
                aspect="auto",
                labels=dict(x="Month", y="Year", color="Return %"),
            )
            fig_heat.update_layout(
                template="plotly_white",
                height=max(200, 80 * len(pivot)),
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Yearly summary
            pivot["Annual"] = pivot.sum(axis=1)
            st.markdown("#### Annual Summary")
            st.dataframe(
                pivot.style.format("{:.2f}"),
                width="stretch",
            )
        else:
            st.info("Not enough data for monthly breakdown.")

        # Rolling Sharpe ratio
        st.markdown("#### Rolling Sharpe Ratio (60-day)")
        rolling_window = min(60, len(daily) // 2) if len(daily) > 10 else len(daily)
        if rolling_window >= 5:
            daily_excess = daily["port_return"] - risk_free_rate / TRADING_DAYS_PER_YEAR
            rolling_mean = daily_excess.rolling(rolling_window).mean()
            rolling_std = daily_excess.rolling(rolling_window).std()
            rolling_sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * rolling_mean / rolling_std.replace(0, np.nan)

            fig_rs = go.Figure()
            fig_rs.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index, y=rolling_sharpe.values,
                    name="Rolling Sharpe", line=dict(color="#19D3F3", width=2),
                    fill="tozeroy", fillcolor="rgba(25,211,243,0.06)",
                )
            )
            fig_rs.add_hline(y=0, line_dash="dot", line_color="grey")
            fig_rs.add_hline(y=bt.sharpe_ratio, line_dash="dash", line_color="orange",
                             annotation_text=f"Overall: {bt.sharpe_ratio:.3f}")
            fig_rs.update_layout(
                yaxis_title="Sharpe Ratio",
                template="plotly_white",
                height=320,
                margin=dict(t=30, b=30),
                hovermode="x unified",
            )
            st.plotly_chart(fig_rs, use_container_width=True)

    # ═══════════════════════════════════════
    # TAB 4 – Cointegration Scan (renumbered)
    # ═══════════════════════════════════════
    with tab_scan:
        st.markdown("#### Engle-Granger Cointegration Results")
        st.caption(f"Testing {len(scan)} unique pairs  ·  Significance threshold p < 0.05")

        # Colour the boolean column safely (map replaces deprecated applymap)
        def _highlight_coint(val):
            if val is True:
                return "background-color: #d4edda; color: #155724; font-weight: 600"
            return ""

        styled = scan.style.map(_highlight_coint, subset=["cointegrated"])
        styled = styled.format({"p_value": "{:.6f}", "t_stat": "{:.4f}",
                                "crit_1pct": "{:.4f}", "crit_5pct": "{:.4f}", "crit_10pct": "{:.4f}"})
        st.dataframe(styled, width="stretch", height=400)

        # Summary
        n_coint = scan["cointegrated"].sum()
        st.success(f"✅  {n_coint} / {len(scan)} pairs are cointegrated at p < 0.05")

    # ═══════════════════════════════════════
    # TAB 5 – Raw Data
    # ═══════════════════════════════════════
    with tab_data:
        st.markdown("#### Price Data")
        st.dataframe(prices.style.format("{:.2f}"), width="stretch", height=300)

        st.markdown("#### Strategy Signals & Backtest")
        display_cols = ["intercept", "hedge_ratio", "spread", "zscore", "position"]
        st.dataframe(
            sig[display_cols].dropna().style.format(
                {"intercept": "{:.4f}", "hedge_ratio": "{:.4f}",
                 "spread": "{:.4f}", "zscore": "{:.4f}"}
            ),
            width="stretch",
            height=300,
        )

        # Download button
        csv = sig.dropna().to_csv()
        st.download_button(
            "\u2b07\ufe0f  Download signals as CSV",
            data=csv,
            file_name=f"pairs_signals_{a}_{b}.csv",
            mime="text/csv",
        )

        # ── Equity curve download ──
        equity_csv = bt.daily[["equity", "port_return", "drawdown"]].to_csv()
        st.download_button(
            "📊  Download equity curve as CSV",
            data=equity_csv,
            file_name=f"equity_curve_{a}_{b}.csv",
            mime="text/csv",
        )

        # Correlation matrix of all tickers
        st.markdown("#### Correlation Matrix (all tickers)")
        corr = prices.pct_change().dropna().corr()
        fig_corr = px.imshow(
            corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            color_continuous_scale="Blues",
            text_auto=".2f",
            aspect="auto",
            zmin=0, zmax=1,
        )
        fig_corr.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    # ── Landing page when nothing has run yet ──
    st.markdown("---")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown(
            """
            ### How This Works

            | Phase | Description |
            |:------|:------------|
            | **1. Discovery** | Download daily prices → Engle-Granger cointegration scan on every pair |
            | **2. Kalman Filter** | Estimate a *dynamic* hedge ratio that adapts to market regime shifts |
            | **3. Signals** | Normalise the spread as a Z-Score → entry when \\|Z\\| > threshold, exit at mean reversion |
            | **4. Backtest** | Simulate a dollar-neutral strategy and compute Sharpe, drawdown, return |

            👈 **Configure parameters** in the sidebar and press **Run Strategy** to start.
            """
        )

    with col_r:
        st.markdown(
            """
            ### Quick Start
            1. Enter tickers for a sector
            2. Set lookback & thresholds
            3. Press **🚀 Run Strategy**
            4. Explore the tabs

            *Default sector: US Large-Cap Banks*
            """
        )
