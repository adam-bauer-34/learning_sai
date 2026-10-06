"""Grid plot of SEs vs window end year (drawing only, no computation here).

Defaults reproduce the Gamma-ramps figure layout: columns = SAI ramp (z), rows = raw and
relative-to-2030 SEs of Gamma_CO2 and Gamma_SAI, color = scenario, line style = Delta_T.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gamma_model import SHORT_NAMES

DEFAULT_COLUMNS = [
    (1 / 3, r"Fast ramp-up of SAI ($f^{\mathrm{SAI}}_t \propto t^{1/3}$)"),
    (1.0, r"Linear ramp-up of SAI ($f^{\mathrm{SAI}}_t \propto t$)"),
    (3.0, r"Slow ramp-up of SAI ($f^{\mathrm{SAI}}_t \propto t^{3}$)"),
]
DEFAULT_ROWS = [
    ("se_co2", r"Standard error in $\Gamma^r_{\mathrm{CO_2}}$"),
    ("se_sai", r"Standard error in $\Gamma^r_{\mathrm{SAI}}$"),
    ("se_co2_rel", r"$\Gamma^r_{\mathrm{CO_2}}$ (relative to 2030)"),
    ("se_sai_rel", r"$\Gamma^r_{\mathrm{SAI}}$ (relative to 2030)"),
]
STYLE_PREFIX = {"delta_T": "ΔT", "rho": "ρ", "sig_IV": "σ_IV", "sigma": "σ"}


def _values(df, col, order=None):
    if order is not None:
        return list(order)
    if pd.api.types.is_numeric_dtype(df[col]):
        return sorted(df[col].unique())
    return list(dict.fromkeys(df[col]))


def _select(d, col, v):
    if pd.api.types.is_numeric_dtype(d[col]):
        return d[np.isclose(d[col], v)]
    return d[d[col] == v]


def _label(col, v, short_names, prefixes):
    if col == "scenario":
        return short_names.get(v, v)
    if isinstance(v, str):
        return v
    return f"{prefixes.get(col, col)} = {v:g}"


def plot_se_grid(
    df,
    hue="scenario",
    style="delta_T",
    columns=DEFAULT_COLUMNS,
    rows=DEFAULT_ROWS,
    hue_order=None,
    style_order=None,
    short_names=SHORT_NAMES,
    prefixes=STYLE_PREFIX,
    cmap=None,
    colors=None,
    styles=None,
    sharey="row",
    logy=True,
    cap_year=2075,
    panel_labels=True,
    legend_ax=(0, 2),
    legend_loc="upper left",
    legend_ncol=2,
    figsize=(15, 15),
    xlim=(2030, 2100),
    xticks=None,
    xlabel="Observational window upper bound",
    lw=2.0,
    lw_alt=1.4,
):
    """Draw a rows x columns grid of SE curves vs window end year.

    Parameters
    ----------
    df : output of `gamma_model.se_table`, `gamma_gls.se_table` or `gamma_gls.ratio_table`
    hue : column mapped to color: "scenario" (default; red = highest 2100 CO2), any other
          column, or None for a single color
    style : column mapped to line style: "delta_T" (default), "variant" (white vs IV), any
            other column, or None for one style. The first style value is drawn solid and
            thicker and carries the hue legend labels.
    columns : list of (z, title); one panel column per z
    rows : list of (df column, y label); one panel row each
    hue_order, style_order : which values to draw and in what order (default: all, in order
                             of appearance for strings, sorted for numbers)
    short_names : dict scenario -> legend label
    prefixes : dict column -> legend prefix for numeric style/hue values
    cmap, colors : colormap name or explicit list of colors
    styles : dict style value -> line style (default: '-', '-.', '--', ':' in order); if given
             without style_order, also selects which style values to draw
    sharey, logy : y-axis sharing ('row', 'all', False) and log scale
    cap_year : vertical dotted line (None to omit)
    legend_ax, legend_loc, legend_ncol : where the legend goes ((row, col) index)
    Returns (fig, axes).
    """
    hue_vals = [None] if hue is None else _values(df, hue, hue_order)
    if colors is None:
        if hue is None:
            colors = ["k"]
        elif hue == "scenario" or not pd.api.types.is_numeric_dtype(df[hue]):
            colors = plt.get_cmap(cmap or "coolwarm")(np.linspace(1, 0, len(hue_vals)))
        else:
            colors = plt.get_cmap(cmap or "viridis")(
                np.linspace(0, 0.85, len(hue_vals))
            )
    if style is None:
        style_vals = [None]
    elif styles is not None and style_order is None:
        style_vals = list(styles)
    else:
        style_vals = _values(df, style, style_order)
    if styles is None:
        styles = dict(zip(style_vals, ["-", "-.", "--", ":"]))

    fig, ax = plt.subplots(
        len(rows),
        len(columns),
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )
    for j, (z, title) in enumerate(columns):
        dz = df[np.isclose(df["z"], z)]
        for hv, col in zip(hue_vals, colors):
            dh = dz if hv is None else _select(dz, hue, hv)
            for sv in style_vals:
                d = (dh if sv is None else _select(dh, style, sv)).sort_values("year")
                if d.empty:
                    continue
                first = sv == style_vals[0]
                label = (
                    _label(hue, hv, short_names, prefixes)
                    if (first and hv is not None)
                    else None
                )
                for i, (var, _) in enumerate(rows):
                    plot = ax[i, j].semilogy if logy else ax[i, j].plot
                    plot(
                        d["year"],
                        d[var],
                        color=col,
                        ls=styles[sv],
                        lw=lw if first else lw_alt,
                        label=label,
                    )
        ax[0, j].set_title(title, fontsize=16)
    for i, (_, ylab) in enumerate(rows):
        ax[i, 0].set_ylabel(ylab, fontsize=14)
    for k, a in enumerate(ax.flat):
        if panel_labels:
            a.text(
                0.97,
                0.95,
                "abcdefghijklmnopqrstuvwxyz"[k],
                transform=a.transAxes,
                va="top",
                ha="right",
                fontsize=14,
                fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.8),
            )
        if cap_year is not None:
            a.axvline(cap_year, c="k", lw=0.8, ls=":")
        # a.grid(True, which="both", lw=0.3, alpha=0.5)
        a.tick_params(which="both", labelsize=12)
    for a in ax[-1]:
        a.set_xlabel(xlabel, fontsize=14)
        a.set_xticks(
            np.arange(xlim[0], xlim[1] + 1, 10) if xticks is None else xticks,
        )
        a.set_xlim(*xlim)
    la = ax[legend_ax]
    if style is not None:
        for sv in style_vals:
            la.plot(
                [],
                [],
                "gray",
                ls=styles[sv],
                lw=lw if sv == style_vals[0] else lw_alt,
                label=_label(style, sv, short_names, prefixes),
            )
    la.legend(loc=legend_loc, fontsize=14, ncol=legend_ncol)
    fig.tight_layout()
    return fig, ax
