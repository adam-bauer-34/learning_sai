"""Corner plots for parameter fitting.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.7.2024

To run: param_corner.py assim N save_figs
"""

import sys
import os

import corner as cr
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from src.presets import get_presets

# pull command line input
assim = sys.argv[1]
TMIN = int(sys.argv[2])
TMAX = int(sys.argv[3])
N = int(sys.argv[4])
obs = sys.argv[5]
save_figs = int(sys.argv[6])

presets, basefile = get_presets()

# update presets
plt.rcParams.update(presets)

# pull date file
file = "../model/data/output/" + assim + "_wind" + str(TMIN) + "_" + str(TMAX)\
      + "_N" + str(N) + "_" + obs + ".nc"

# file = "../model/data/output/heatfull_N1000_iid.nc"

ds = xr.open_dataset(file)

ds_ensavg = ds.mean('ens_mem')

# fig = cr.corner(ds.controls_hist.values[:, :, 0],
#                color='cyan')
                #range=[(min(ds.controls_hist.values[:, i, 0]),
                #        max(ds.controls_hist.values[:, i, 0]))
                #        for i in range(np.shape(ds.controls_hist.values)[1])])

fig = cr.corner(ds.controls.values,
          labels=['T1', 'T2', 'Q', 'L', 'G', 'C1', 'C2', 'F1', 'F3',
                  'A_SO2', 'B_SO2', 'C_SO2'],
          truths=ds.controls_truth.values,
          truth_color='#CC79A7',
          labelpad=0.25)

cr.overplot_lines(fig, ds_ensavg.controls.values, color='g')
cr.overplot_points(fig, ds_ensavg.controls.values[None], marker='s', color='g')

fig.tight_layout()

# plt.show()

if save_figs:
    filename = basefile + assim + "_wind"\
        + str(TMIN) + "_" + str(TMAX) + "_N" + str(N) + "_" + obs\
        + "_param_corner.png"
    fig.savefig(filename, dpi=400)
    print("Figure saved to:\n{}".format(filename))
