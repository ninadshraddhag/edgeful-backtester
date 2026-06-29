"""Monthly P/L heatmap (Edgeful-style calendar view).

Rows = years, columns = months (Jan..Dec). Each cell is net R for that month,
coloured red->green with the value annotated, plus a yearly total column.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_pnl_heatmap(trades: pd.DataFrame, instrument: str, out_path: str,
                        value: str = "r_multiple") -> str:
    """Build a year x month heatmap of summed P/L and write it to HTML."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if trades is None or trades.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trades to display", showarrow=False,
                           font=dict(size=18, color="#bbb"))
        fig.update_layout(template="plotly_dark", height=300)
        fig.write_html(out_path, include_plotlyjs="cdn")
        return out_path

    t = trades.copy()
    t["dt"] = pd.to_datetime(t["exit_time"])
    t["year"] = t["dt"].dt.year
    t["mon"] = t["dt"].dt.month

    pivot = (t.groupby(["year", "mon"])[value].sum()
             .unstack("mon").reindex(columns=range(1, 13)))
    years = sorted(pivot.index.tolist())
    z = pivot.reindex(index=years).to_numpy(float)

    # Annotated text incl. a yearly total column appended on the right.
    yearly = np.nansum(z, axis=1, keepdims=True)
    z_full = np.hstack([z, yearly])
    cols = _MONTHS + ["YTD"]

    text = np.where(np.isnan(z_full), "", np.vectorize(lambda v: f"{v:+.1f}")(z_full))
    # Don't let the YTD column dominate the colour scale.
    cmax = np.nanmax(np.abs(z)) if np.isfinite(np.nanmax(np.abs(z))) else 1.0
    cmax = max(cmax, 1.0)

    fig = go.Figure(go.Heatmap(
        z=z_full, x=cols, y=[str(y) for y in years], text=text,
        texttemplate="%{text}", textfont=dict(size=12),
        zmid=0, zmin=-cmax, zmax=cmax,
        colorscale=[[0, "#b71c1c"], [0.5, "#1b1b1b"], [1, "#1b5e20"]],
        colorbar=dict(title="R"),
        hovertemplate="%{y} %{x}: %{z:+.2f} R<extra></extra>",
        xgap=2, ygap=2,
    ))
    total = np.nansum(z)
    fig.update_layout(
        template="plotly_dark",
        title=f"{instrument} - Monthly P/L (R)   |   Total: {total:+.2f} R",
        height=120 + 60 * len(years), margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path
