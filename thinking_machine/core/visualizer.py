"""Plotly trade chart: candlesticks + entry/exit markers + equity curve.

Writes a self-contained HTML file you can open in a browser.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_trades(df: pd.DataFrame, trades: pd.DataFrame, instrument: str,
                out_path: str, max_bars: int = 4000) -> str:
    """Render price with trade markers and a cumulative-R subplot."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Keep the chart light: show the window that actually contains trades.
    plot_df = df
    if len(df) > max_bars and not trades.empty:
        lo = df["date"].searchsorted(trades["entry_time"].min()) - 50
        hi = df["date"].searchsorted(trades["exit_time"].max()) + 50
        plot_df = df.iloc[max(0, lo):min(len(df), hi)]
    elif len(df) > max_bars:
        plot_df = df.tail(max_bars)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=(f"{instrument} - price & trades", "Equity (cumulative R)"),
    )

    fig.add_trace(go.Candlestick(
        x=plot_df["date"], open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"], name="price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    if not trades.empty:
        longs = trades[trades["side"] == "long"]
        shorts = trades[trades["side"] == "short"]
        wins = trades[trades["r_multiple"] > 0]
        losses = trades[trades["r_multiple"] <= 0]

        fig.add_trace(go.Scatter(
            x=longs["entry_time"], y=longs["entry"], mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#2962ff",
                        line=dict(width=1, color="white")),
            name="long entry", hovertext=longs["r_multiple"],
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=shorts["entry_time"], y=shorts["entry"], mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#d500f9",
                        line=dict(width=1, color="white")),
            name="short entry", hovertext=shorts["r_multiple"],
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=wins["exit_time"], y=wins["exit"], mode="markers",
            marker=dict(symbol="circle", size=8, color="#00e676"),
            name="exit win",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=losses["exit_time"], y=losses["exit"], mode="markers",
            marker=dict(symbol="x", size=9, color="#ff1744"),
            name="exit loss",
        ), row=1, col=1)

        eq = trades.sort_values("exit_time")
        fig.add_trace(go.Scatter(
            x=eq["exit_time"], y=eq["r_multiple"].cumsum(),
            mode="lines", line=dict(color="#ffc107", width=2),
            name="equity (R)", fill="tozeroy",
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=820,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.06),
        margin=dict(l=40, r=20, t=60, b=30),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path
