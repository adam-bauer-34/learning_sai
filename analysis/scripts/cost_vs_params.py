"""Compare observations to retrieved values.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.7.2024

To run: python cost_vs_iter.py assim TMIN TMAX N obs save_figs
"""

import sys

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from src.presets import get_presets

# pull command line input
assim = sys.argv[1]
obs = sys.argv[2]
save_figs = int(sys.argv[3])

presets, basefile = get_presets()

# update presets
plt.rcParams.update(presets)

# pull date file
file = "../model/data/output/" + "cost_struc_" + assim + "_" + obs + ".nc" 

ds = xr.open_dataset(file)

fig, ax = plt.subplot_mosaic([['a', 'b', 'c'],
                              ['d', 'e', 'f'],
                              ['g', 'h', 'i']],
                             figsize=(26, 20), layout='constrained')

t0_grid, t1_grid = np.meshgrid(ds.t1.values, ds.t2.values)
l_grid, f1_grid = np.meshgrid(ds.l.values, ds.f1.values)
a_grid, b_grid = np.meshgrid(ds.a.values, ds.b.values)

N_levels = 50

# temperature initial conditions
ax['a'].contourf(t0_grid, t1_grid,
                 np.log10(ds.cost_ts.values[0]),
                 levels=N_levels)

ax['b'].contourf(t0_grid, t1_grid,
                 np.log10(ds.cost_ts.values[1]),
                 levels=N_levels)

a = ax['c'].contourf(t0_grid, t1_grid,
                     np.log10(ds.cost_ts.values[2]),
                     levels=N_levels)

# Lambda vs F1
ax['d'].contourf(l_grid, f1_grid,
                 np.log10(ds.cost_lf.values[0]),
                 levels=N_levels)

ax['e'].contourf(l_grid, f1_grid,
                 np.log10(ds.cost_lf.values[1]),
                 levels=N_levels)

ax['f'].contourf(l_grid, f1_grid,
                 np.log10(ds.cost_lf.values[2]),
                 levels=N_levels)

# A_SO2 vs B_SO2
ax['g'].contourf(a_grid, b_grid,
                 np.log10(ds.cost_aero.values[0]),
                 levels=N_levels)

ax['h'].contourf(a_grid, b_grid,
                 np.log10(ds.cost_aero.values[1]),
                 levels=N_levels)

ax['i'].contourf(a_grid, b_grid,
                 np.log10(ds.cost_aero.values[2]),
                 levels=N_levels)

t_panels = ['a', 'b', 'c']
lf_panels = ['d', 'e', 'f']
aero_panels = ['g', 'h', 'i']

c = 0
for p in t_panels:
    ax[p].axvline(ds.controls_truth.values[0],
                  linestyle='dashed', color='k', zorder=1000)
    ax[p].axhline(ds.controls_truth.values[1],
                  linestyle='dashed', color='k',
                  label="Truth", zorder=1000)
    ax[p].set_xlabel(r"$T_0^{(1)}$", fontsize=20)
    ax[p].set_ylabel(r"$T_0^{(2)}$", fontsize=20)
    ax[p].set_title('Assimilation window: {}'.format(ds.t_range.values[c]),
                    fontsize=25)
    c += 1

for p in lf_panels:
    ax[p].axvline(ds.controls_truth.values[3],
                  linestyle='dashed', color='k', zorder=1000)
    ax[p].axhline(ds.controls_truth.values[7],
                  linestyle='dashed', color='k',
                  label="Truth", zorder=1000)
    ax[p].set_xlabel(r"$\lambda$", fontsize=20)
    ax[p].set_ylabel(r"$F_1^{CO_2}$", fontsize=20)

for p in aero_panels:
    ax[p].axvline(ds.controls_truth.values[-3],
                  linestyle='dashed', color='k', zorder=1000)
    ax[p].axhline(ds.controls_truth.values[-2],
                  linestyle='dashed', color='k',
                  label="Truth", zorder=1000)
    ax[p].set_xlabel(r"$A_{SO_2}$", fontsize=20)
    ax[p].set_ylabel(r"$B_{SO_2}$", fontsize=20)

cb = fig.colorbar(a, ax=[ax['c'], ax['f'], ax['i']], pad=0.1, shrink=0.9)
cb.set_label(label=r'$\log_{10}\left( J \right)$', fontsize=25)

if save_figs:
    filename = basefile + "cost_struc_" + assim + "_" + obs + ".png"
    fig.savefig(filename, dpi=400)
    print("Figure saved to:\n{}".format(filename))
