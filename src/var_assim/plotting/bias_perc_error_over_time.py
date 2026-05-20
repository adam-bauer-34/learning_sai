"""Function base for making bias and percent error over time plots
for the SAI inequality variable, theta.

Adam Bauer
UChicago
"""

import gc
import string

import numpy as np
import matplotlib.pyplot as plt

from datatree import open_datatree
from var_assim.plotting.filtering import filter_datatree_by_cost_ratio_memeff
from var_assim.plotting.utils import make_figure_filename
from var_assim.config import FIGS_DIR

color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
              '#0072B2', '#CC79A7', '#D55E00']

def make_grid_plot(angles, angles_tr, angles_prior, angles_names,
                   winds, PLO, PHI, row_titles,
                   save_figs=False, fname='bias_error_grid'):
    
    # extract plot dimensions
    NROWS, NANGS, _, _ = np.shape(angles) 

    # set up
    fig, ax = plt.subplots(NROWS, 3, figsize=(30, 30), sharex=True)

    # data plotting
    for rowx in range(ax.shape[0]):
        for angx in range(NANGS):
            ax[rowx, 0].plot(winds,
                            np.nanmedian(angles[rowx, angx], axis=1),
                            linestyle='solid', color=color_list[angx+1],
                            label=r'$\vartheta^\dagger=${}$^\circ$'.format(angles_names[angx]),
                            zorder=100)
            
            ax[rowx, 0].axhline(angles_tr[angx],
                               linestyle='dashed', linewidth=1.25,
                               color=color_list[angx+1])
            
            ax[rowx, 1].plot(winds, np.nanmedian(angles[rowx, angx], axis=1) - angles_tr[angx],
                            color=color_list[angx+1], zorder=100, linestyle='solid')
            
            ax[rowx, 2].plot(
                winds, np.nanpercentile(angles[rowx, angx], PHI, axis=1) - np.nanpercentile(angles[rowx, angx], PLO, axis=1),
                zorder=100, color=color_list[angx+1], linestyle='solid'
            )

    # Set y labels, horizontal line, and y limit
    for rowx in range(NROWS):
        ax[rowx, 0].set_ylabel(r"med$(\vartheta)$ (Degrees)")
        ax[rowx, 1].set_ylabel(r"med$(\vartheta) - \vartheta^\dagger$ (Degrees)")
        ax[rowx, 1].axhline(0, color='k', linestyle='solid')
        ax[rowx, 2].set_ylabel(f"{PLO}–{PHI} Percentile Range (Degrees)")
        ax[rowx, 2].set_ylim((0,
                              5 + np.nanpercentile(angles_prior, PHI) - np.nanpercentile(angles_prior, PLO)))

    # set x labels for only bottom row
    for a in ax[NROWS-1]:
        a.set_xlabel("Year")

    # set vertical lines for assimilations windows
    for a in ax.flatten():
        for w in winds:
            a.axvline(w, linestyle='dotted', color='grey', linewidth=1.25)

    # set twin y axis in leftmost panels
    for rowx in range(ax.shape[0]):
        ax2_rel = ax[rowx, 2].twinx()
        ax2_rel.set_ylabel(f"Relative {PLO}–{PHI} Range (vs prior)")
        ax2_rel.spines['right'].set_visible(True)
        ax2_rel.set_ylim((0, (2 + np.nanpercentile(angles_prior, PHI) - np.nanpercentile(angles_prior, PLO)) / (np.nanpercentile(angles_prior, PHI) - np.nanpercentile(angles_prior, PLO))))

    # set legend
    ax[0, 0].legend(bbox_to_anchor=(3.87, -0.82), ncols=1, frameon=True)

    # set panel labels
    panel_labels = list(string.ascii_uppercase)[:len(ax.flatten())]
    for a, label in zip(ax.flatten(), panel_labels):
        a.text(0.925, 0.98, label, transform=a.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='left')

    # set row titles
    for rowx, title in enumerate(row_titles):

        # get top y-position of middle axis in row
        y = ax[rowx, 1].get_position().y1 + 0.01

        fig.text(
            0.5, y,
            title,
            ha='center',
            va='bottom',
            fontsize=18,
            fontweight='bold'
        )

    if save_figs:
        filename = make_figure_filename(
            fname, outdir=FIGS_DIR / 'results'
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filename}")

    return fig, ax


def make_cleaned_datatree_list(fnames: list,
                               angles: list,
                               dropped_vars: list,
                               THRESHOLD: float = 1e-2,):
    """Open datatrees, drop unnecessary variables, and filter by initial vs
    final cost function values.

    Parameters
    ----------
    fnames: list[str]
        list of datatree file paths

    dropped_vars: list[str]
        list of variables in the datatree to drop before filtering (saves memory)

    THRESHOLD: float (default = 1e-2)
        cost function ratio threshold

    Returns
    -------
    dts: list[DataTree]
        list of cleaned datatrees
    """

    dts = []

    for idx, f in enumerate(fnames):
        print(f"Processing angle = {angles[idx]}...")
        dt = open_datatree(f, engine='netcdf4', drop_variables=dropped_vars)
    
        dt_clean = filter_datatree_by_cost_ratio_memeff(dt, 
                                                        threshold=THRESHOLD,
                                                        clear_after_each_node=True)
        dts.append(dt_clean)

        # cleanup
        for node in dt.subtree:
            if node.ds is not None:
                node.ds.close()

        del dt
        gc.collect()

    return dts


def test_make_grid_plot():
    """
    Generate synthetic data and test make_grid_plot().
    """

    # ---------------------------------
    # synthetic dimensions
    # ---------------------------------
    NROWS = 4
    NANGS = 6
    NWINDS = 15
    NSAMPLES = 300

    rng = np.random.default_rng(42)

    # ---------------------------------
    # x-axis / assimilation windows
    # ---------------------------------
    winds = np.arange(2000, 2000 + NWINDS)

    # ---------------------------------
    # true theta values
    # ---------------------------------
    angles_tr = np.array([0, 15, 30, 45, 60, 75])

    # ---------------------------------
    # prior ensemble
    # ---------------------------------
    angles_prior = rng.normal(
        loc=35,
        scale=18,
        size=5000
    )

    # ---------------------------------
    # synthetic posterior ensembles
    #
    # shape:
    # (NROWS, NANGS, NWINDS, NSAMPLES)
    # ---------------------------------
    angles = np.zeros((NROWS, NANGS, NWINDS, NSAMPLES))

    for rowx in range(NROWS):

        # rows become progressively tighter
        noise_scale = [16, 10, 6, 3][rowx]

        for angx in range(NANGS):

            truth = angles_tr[angx]

            for windx in range(NWINDS):

                # decreasing bias over time
                bias = np.linspace(12, 0, NWINDS)[windx]

                # ensemble
                angles[rowx, angx, windx] = (
                    truth
                    + bias
                    + rng.normal(
                        loc=0,
                        scale=noise_scale,
                        size=NSAMPLES
                    )
                )

    # ---------------------------------
    # percentile bounds
    # ---------------------------------
    PLO = 5
    PHI = 95

    # ---------------------------------
    # row titles
    # ---------------------------------
    global row_titles
    row_titles = [
        "Baseline",
        "Weak Constraint",
        "Strong Constraint",
        "No Noise Model"
    ]

    # ---------------------------------
    # call plotting function
    # ---------------------------------
    fig, ax = make_grid_plot(
        angles=angles,
        angles_tr=angles_tr,
        angles_prior=angles_prior,
        winds=winds,
        PLO=PLO,
        PHI=PHI,
        row_titles=['row'] * NROWS,
        save_figs=False,
        fname='test_bias_error_grid'
    )

    plt.show()