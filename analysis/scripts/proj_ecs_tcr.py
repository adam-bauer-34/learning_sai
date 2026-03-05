"""Compute projections and ECS and TCR distributions after learning.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.14.2024

To run: proj_ecs_tcr.py assim scenario Nwinds Nens obs save_figs
"""

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from datatree import open_datatree, DataTree
from src.presets import get_presets

sys.path.append('../model')
from src.emis import EmissionsBaseline
from src.pco2sul.dynamics import get_nonlin_path

# parse command line
assim = sys.argv[1]
scenario = sys.argv[2]
Nwinds = int(sys.argv[3])
Nens = int(sys.argv[4])
obs = sys.argv[5]
save_figs = int(sys.argv[6])

# get presets and basefile
presets, basefile = get_presets()
plt.rcParams.update(presets)

# steal and set color list
wind_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
               '#0072B2', '#D55E00', '#CC79A7']

# import file
cwd = os.getcwd()
filepath = cwd + "/../model/data/output/margobs_" + scenario + "_" + assim\
    + "_Nwinds" + str(Nwinds) + "_N" + str(Nens) + "_" + obs + ".nc"

# open datatree
dt = open_datatree(filepath, engine='netcdf4')

tmin = int(min(dt['2000'].ds.time.values))

# compute paths
e = EmissionsBaseline(scenario, tmin, 2100)
DT = 1.0
times = np.arange(tmin, 2101, int(DT))

# make empty list of paths
T_ens_paths = np.zeros((Nwinds, Nens, len(times)))

wind_c = 0
for wind in dt.children:
    for mem in range(Nens):
        T_ens_paths[wind_c, mem] = get_nonlin_path(e,
                                                   dt[str(wind)].ds.controls.sel(ens_mem=mem).values,
                                                   tmin, 2100, DT)[0][0]

    wind_c += 1

# take mean
T_ens_avg_path = np.nanmean(T_ens_paths, axis=1)

# compute ECS dists and TCR dists
ECS_dists = np.zeros((Nwinds, Nens))
TCR_dists = np.zeros_like(ECS_dists)

ECS_true = (dt['2000'].ds.controls_truth.sel(vari='F1_CO2').values
            * np.log(2)
            + dt['2000'].ds.controls_truth.sel(vari='F3_CO2').values
            * (np.sqrt(2 * 278.3) - np.sqrt(278.3)))\
            / dt['2000'].ds.controls_truth.sel(vari='L').values

TCR_true = (dt['2000'].ds.controls_truth.sel(vari='F1_CO2').values
            * np.log(2)
            + dt['2000'].ds.controls_truth.sel(vari='F3_CO2').values
            * (np.sqrt(2 * 278.3) - np.sqrt(278.3)))\
            / (dt['2000'].ds.controls_truth.sel(vari='L').values
               + dt['2000'].ds.controls_truth.sel(vari='G').values)

ECS_prior = (dt['2000'].ds.controls_hist.sel(vari='F1_CO2', iter=1).values
             * np.log(2)
             + dt['2000'].ds.controls_hist.sel(vari='F3_CO2', iter=1).values
             * (np.sqrt(2 * 278.3) - np.sqrt(278.3)))\
        / dt['2000'].ds.controls_hist.sel(vari='L', iter=1).values

TCR_prior = (dt['2000'].ds.controls_hist.sel(vari='F1_CO2', iter=1).values
             * np.log(2)
             + dt['2000'].ds.controls_hist.sel(vari='F3_CO2', iter=1).values
             * (np.sqrt(2 * 278.3) - np.sqrt(278.3)))\
        / (dt['2000'].ds.controls_hist.sel(vari='L', iter=1).values
           + dt['2000'].ds.controls_hist.sel(vari='G', iter=1).values)

assim_c = 0
for node in dt.children:
    ECS_dists[assim_c] = (dt[str(node)].ds.controls.sel(vari='F1_CO2').values
                          * np.log(2)
                          + dt[str(node)].ds.controls.sel(vari='F3_CO2').values
                          * (np.sqrt(2 * 278.3) - np.sqrt(278.3)))\
                          / dt[str(node)].ds.controls.sel(vari='L').values

    TCR_dists[assim_c] = (dt[str(node)].ds.controls.sel(vari='F1_CO2').values
                          * np.log(2)
                          + dt[str(node)].ds.controls.sel(vari='F3_CO2').values
                          * (np.sqrt(2 * 278.3) - np.sqrt(278.3)))\
        / (dt[str(node)].ds.controls.sel(vari='L').values
           + dt[str(node)].ds.controls.sel(vari='G').values)

    assim_c += 1

# make plot
fig, ax = plt.subplot_mosaic([['a', 'a'],
                              ['b', 'c']],
                             figsize=(15, 10),
                             layout='constrained')

window_labels = ['1850-' + str(ds) for ds in dt.children]

# plot observations
ax['a'].scatter(times, dt['2100'].ds.data_truth.sel(vari='T1').values,
                marker='o', color='k', label='Observations')

# set fill between percentiles
fill_btw_upper_perc = 95
fill_btw_lower_perc = 5

# plot ensemble average and percentile ragnes for different assimilation
# windows
for wind in range(Nwinds-1):
    ax['a'].plot(times, T_ens_avg_path[wind], color=wind_colors[wind],
                 label=window_labels[wind] + ": Ensemble Average",
                 linestyle='solid', linewidth=2)

    ax['a'].fill_between(times, np.nanpercentile(T_ens_paths, fill_btw_lower_perc,
                                              axis=1)[wind],
                         np.nanpercentile(T_ens_paths, fill_btw_upper_perc,
                                       axis=1)[wind],
                         color=wind_colors[wind], alpha=0.4,
                         label=window_labels[wind] + ": " +
                         str(fill_btw_lower_perc) + "-"
                         + str(fill_btw_upper_perc) + " Percentile Range")

ax['a'].set_xlim((2000, 2100))
ax['a'].set_xlabel("Year")
ax['a'].set_ylabel("Temperature above preindustrial (K)")
ax['a'].legend()

# plot ECS and TCR histograms
Nbins = 30

# plot priors
sb.kdeplot(data=ECS_prior, ax=ax['b'], color=wind_colors[-1], label='Prior')
ax['b'].axvline(np.percentile(ECS_prior, fill_btw_lower_perc),
                linestyle='dashed', color=wind_colors[-1])
ax['b'].axvline(np.percentile(ECS_prior, fill_btw_upper_perc),
                linestyle='dashed', color=wind_colors[-1])

sb.kdeplot(data=TCR_prior, ax=ax['c'], color=wind_colors[-1], label='Prior')
ax['c'].axvline(np.percentile(TCR_prior, fill_btw_lower_perc),
                linestyle='dashed', color=wind_colors[-1])
ax['c'].axvline(np.percentile(TCR_prior, fill_btw_upper_perc),
                linestyle='dashed', color=wind_colors[-1])

# plot ECS and TCR histograms for each assimilation window
for wind in range(Nwinds-1):
    sb.kdeplot(ECS_dists[wind], color=wind_colors[wind],
               label=window_labels[wind], ax=ax['b'], linestyle='solid')
    ax['b'].axvline(np.percentile(ECS_dists[wind], fill_btw_lower_perc),
                    linestyle='dashed', color=wind_colors[wind])
    ax['b'].axvline(np.percentile(ECS_dists[wind], fill_btw_upper_perc),
                    linestyle='dashed', color=wind_colors[wind])

    sb.kdeplot(TCR_dists[wind], color=wind_colors[wind],
               label=window_labels[wind], ax=ax['c'], linestyle='solid')
    ax['c'].axvline(np.percentile(TCR_dists[wind], fill_btw_lower_perc),
                    linestyle='dashed', color=wind_colors[wind])
    ax['c'].axvline(np.percentile(TCR_dists[wind], fill_btw_upper_perc),
                    linestyle='dashed', color=wind_colors[wind])

# plot vertical lines for the true values
ax['b'].axvline(ECS_true, color='k', label='Truth', linestyle='solid')
ax['c'].axvline(TCR_true, color='k', label='Truth', linestyle='solid')

# set y and x axis labels and the legend
ax['b'].set_ylabel("Density")
ax['b'].set_xlabel("Equilibrium Climate Sensitivity (K)")
ax['c'].set_xlabel("Transient Climate Response (K)")

ax['b'].legend()

# set panel labels
for label, ax in ax.items():
    ax.annotate(
        label,
        xy=(0.02, 1.05), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=16, verticalalignment='top', fontweight='bold',
        bbox=dict(facecolor='none', edgecolor='none'))

if save_figs:
    fig_filename = basefile + "proj-ecs-tcr-" + assim + "-" + scenario\
        + '-Nwinds' + str(Nwinds) + '-N' + str(Nens) + '-' + obs + '.png'
    fig.savefig(fig_filename, dpi=400)
    print("Figure successfully saved to:\n{}".format(fig_filename))

else:
    plt.show()
