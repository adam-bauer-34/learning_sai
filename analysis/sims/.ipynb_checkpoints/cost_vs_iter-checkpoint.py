"""Compare observations to retrieved values.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.7.2024

To run: python cost_vs_iter.py assim TMIN TMAX N obs save_figs
"""

import sys
import os

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datatree import open_datatree

from src.presets import get_presets

# pull command line input
assim = sys.argv[1]
scenario = sys.argv[2]
TMIN = int(sys.argv[3])
AR = int(sys.argv[4])
L_or_F = sys.argv[5]
sig = float(sys.argv[6])
Nwinds = int(sys.argv[7])
Nens = int(sys.argv[8])
save_figs = int(sys.argv[9])

presets, basefile = get_presets()

# update presets
plt.rcParams.update(presets)

# pull date file
#file = "/../../../a/two-layer-4dvar/data/output/pco2sulpatwc/margobs_ws_"\
#        + scenario + "_"\
#        + assim + "_TMIN" + str(TMIN) + "_AR" + str(AR) + "_d" + L_or_F\
#        + "_sig1.0" + "_Nwinds" + str(Nwinds) + "_Nens" + str(Nens)\
#        + ".nc"

cwd = os.getcwd()
file = '/data/keeling/a/adammb4/a/two-layer-4dvar/data/output/pco2sulpatwc/margobs_ws_ssp245_pco2sulpatwc_TMIN2020_AR1_dL_sig1.0_Nwinds9_Nens500.nc'

assim_wind = '2050'
ds = open_datatree(file)[assim_wind].ds

ds_ensavg = ds.median('ens_mem')

J_ens = ds_ensavg.cost_hist.values

fig, ax = plt.subplot_mosaic([['a', 'b']])

obs_c = '#CC79A7'
obs_s = 25
obs_z = 100
ensavg_w = 3
mem_w = 0.5
mem_ls = 'solid'

ax['a'].plot(ds_ensavg.iter.values, J_ens, label='Ensemble median',
             linewidth=ensavg_w, zorder=1e3)

ax['b'].plot(ds_ensavg.iter.values, np.log10(J_ens), label='Ensemble median',
             linewidth=ensavg_w, zorder=1e3)

for mem in range(len(ds.ens_mem.values)):
    if mem == 0:
        label = 'Ensemble median'
    else:
        label = None

    ax['a'].plot(ds_ensavg.iter.values,
                 ds.cost_hist.loc[mem].values,
                 linewidth=mem_w, color='grey', label=label, linestyle=mem_ls)

    ax['b'].plot(ds_ensavg.iter.values,
                 np.log10(ds.cost_hist.loc[mem].values),
                 linewidth=mem_w, color='grey', label=label, linestyle=mem_ls)

ax['a'].set_xlim((0, 20))
ax['b'].set_xlim((0, 20))
ax['a'].set_ylim((0, 1e6))
ax['b'].set_ylim((-2, 6))

ax['a'].set_ylabel("Cost")
ax['b'].set_ylabel("log$_{10}$(Cost)")

ax['a'].set_xlabel("Iteration")
ax['b'].set_xlabel("Iteration")

ax['b'].legend()

fig.tight_layout()

if save_figs:
    filename = basefile\
            + "cost-vs-iter-" + scenario + "-" + assim_wind\
            + "-AR" + str(AR)\
            + "-d" + L_or_F\
            + "-Nwinds" + str(Nwinds)\
            + "-Nens" + str(Nens)\
            + ".png"

    fig.savefig(filename, dpi=400)
    print("Figure saved to:\n{}".format(filename))

else:
    plt.show()
