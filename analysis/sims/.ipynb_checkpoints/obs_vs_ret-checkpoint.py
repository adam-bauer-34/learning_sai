"""Compare observations to retrieved values.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.7.2024

To run: python obs_vs_ret.py assim TMIN TMAX N obs save_figs
"""

import sys

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

ds = xr.open_dataset(file)

ds_ensavg = ds.mean('ens_mem')

T1_ens = ds_ensavg.data_final.loc['T1'].values
Q_ens = ds_ensavg.data_final.loc['Q'].values

T1_obs = ds.obs.loc['T1'].values
Q_obs = ds.obs.loc['Q'].values

fig, ax = plt.subplot_mosaic([['a', 'b']])

obs_c = '#CC79A7'
obs_s = 25
obs_z = 100
ensavg_w = 3
mem_w = 0.5
mem_ls = 'solid'

ax['a'].plot(ds.time.values, T1_ens, label='Ensemble average',
             linewidth=ensavg_w, zorder=50)
ax['a'].scatter(ds.time.values, T1_obs, marker='.', color=obs_c, s=obs_s,
                label='Observations', zorder=obs_z)

ax['b'].plot(ds.time.values, Q_ens, label='Ensemble average',
             linewidth=ensavg_w, zorder=50)
ax['b'].scatter(ds.time.values, Q_obs, marker='.', color=obs_c, s=obs_s,
                label='Observations', zorder=obs_z)

for mem in range(len(ds.ens_mem.values)):
    if mem == 0:
        label = 'Ensemble members'
    else:
        label = None

    ax['a'].plot(ds.time.values,
                 ds.data_final.loc[mem, 'T1'].values,
                 linewidth=mem_w, color='grey', label=label, linestyle=mem_ls)

    ax['b'].plot(ds.time.values,
                 ds.data_final.loc[mem, 'Q'].values,
                 linewidth=mem_w, color='grey', label=label, linestyle=mem_ls)

ax['a'].set_ylabel("Surface temperature (K)")
ax['b'].set_ylabel("Ocean heat content (J)")

ax['a'].set_xlabel("Year")
ax['b'].set_xlabel("Year")

ax['b'].legend()

fig.tight_layout()

plt.show()
if save_figs:
    filename = basefile + assim + "_wind"\
        + str(TMIN) + "_" + str(TMAX) + "_N" + str(N) + "_" + obs\
        + "_obs_vs_ret.png"
    fig.savefig(filename, dpi=400)
    print("Figure saved to:\n{}".format(filename))
