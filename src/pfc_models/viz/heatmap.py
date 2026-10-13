"""Density heatmaps.

Provenance: log/cpu-spectral-base-implementation/_densityHeatmap.py,
byte-identical in all three source forks. Carried verbatim, including the
module-level `display(HTML(...))` MathJax injection - see the migration log
for why that is flagged rather than removed.
"""
# plotting
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display
from plotly.subplots import make_subplots

display(
    HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    )
)


def GetDensityHeatmap(
    psi,
    X,
    Y,
    title=None,
    colorscale="gray",
    ppx=1,
    zmin=None,
    zmax=None,
    showScale=False,
    rollShiftX=None,
    rollShiftY=None,
):
    print(
        f"min: {psi.min()}, mean: {psi.mean()}, max: {psi.max()}, std: {psi.std()}, relative error: {psi.std() / psi.mean()}"
    )

    if rollShiftX is not None:
        psi = np.roll(psi, shift=rollShiftX, axis=1)
        X = np.roll(X, shift=rollShiftX, axis=1)

    if rollShiftY is not None:
        psi = np.roll(psi, shift=rollShiftY, axis=0)
        Y = np.roll(Y, shift=rollShiftY, axis=0)

    if zmin is None:
        zmin = psi.min()

    if zmax is None:
        zmax = psi.max()

    _x = X[0, :]
    _y = Y[:, 0]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=psi,
            x=X[0, :],
            y=Y[:, 0],
            colorscale=colorscale,
            showscale=showScale,
            colorbar=dict(
                thickness=5 * ppx,  # pixels
                len=0.6,  # fraction
                lenmode="fraction",
                x=0.95,  # fraction
                xanchor="right",
                y=0.5,  # fraction
                yanchor="middle",
                # --- BACKGROUND CONFIGURATION ---
                bgcolor="rgba(255, 255, 255, 0.8)",  # White with 80% opacity
                outlinewidth=1,  # Optional: adds a thin border
                outlinecolor="rgba(200, 200, 200, 0.5)",  # Optional: soft border color
            ),
            zmin=zmin,
            zmax=zmax,
        )
    )

    # Add title as an annotation with formatting similar to legend (outline, bg color, etc.)
    if title is not None:
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=16),
            align="center",
            # --- TITLE CONFIGURATION ---
            bgcolor="rgba(255, 255, 255, 0.8)",  # White with 80% opacity
            borderwidth=1,  # Optional: adds a thin border
            bordercolor="rgba(200, 200, 200, 0.5)",  # Optional: soft border color
            borderpad=4,  # Optional: padding around the text
        )

    # make axis use same scale, keep it square
    fig.update_layout(
        # yaxis=dict(scaleanchor="x"),
        width=200 * ppx,
        height=200 * ppx * (Y.max() - Y.min()) / (X.max() - X.min()),
        margin=dict(l=0, r=0, t=0, b=0),
        # xaxis_showgrid=False,
        # yaxis_showgrid=False,
        # xaxis_zeroline=False,
        # yaxis_zeroline=False,
        # xaxis_visible=False,
        # yaxis_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # hide axis labels
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig

