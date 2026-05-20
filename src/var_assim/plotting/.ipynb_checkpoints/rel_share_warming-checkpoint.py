"""Compute relative share of warming between fast and slow modes over time for
an ensemble of model runs.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.14.2024

To run: rel_share_warming.py assim scenario Nwinds Nens obs save_figs
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from datatree import open_datatree
from src.presets import get_presets
from src.analytics import get_path_pieces_analytic

sys.path.append('../model')
from src.emis import EmissionsBaseline
from src.pco2sul.dynamics import get_nonlin_path

# get presets and basefile
presets, basefile = get_presets()
plt.rcParams.update(presets)
save_figs = True

# steal and set color list
wind_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
               '#0072B2', '#D55E00', '#CC79A7']

tmin, tmax, dt = 0, 2000, 1
times = np.arange(tmin, tmax, dt)
forcing = np.ones(len(times))

params = np.array([0.1, 0.05, 0, 2.0, 0.7, 9, 95])
t_ic_lo, t_fast_lo, t_slow_lo, t1_lo, _ = get_path_pieces_analytic(tmin, tmax,
                                                                   params,
                                                                   forcing, dt)

params_av = np.array([0.1, 0.05, 0, 1.3, 0.7, 9, 95])
t_ic_av, t_fast_av, t_slow_av, t1_av, _ = get_path_pieces_analytic(tmin, tmax,
                                                                   params_av,
                                                                   forcing, dt)

params_hi = np.array([0.1, 0.05, 0, 0.6, 0.7, 9, 95])
t_ic_hi, t_fast_hi, t_slow_hi, t1_hi, _ = get_path_pieces_analytic(tmin, tmax,
                                                                   params_hi,
                                                                   forcing, dt)

total_warming_lo = t_ic_lo + t_fast_lo + t_slow_lo
total_warming_av = t_ic_av + t_fast_av + t_slow_av
total_warming_hi = t_ic_hi + t_fast_hi + t_slow_hi

# make figure
fig, ax = plt.subplot_mosaic([['a', 'b', 'c', 'd', 'e', 'f']], figsize=(20, 5),
                             width_ratios=(1, 0.1, 1, 0.1, 1, 0.1),
                             constrained_layout=True)


def make_panel(fig, ax, big, sma, t_ic_lo, t_fast_lo, t_slow_lo):
    # note the total warming
    total_warming_lo = t_ic_lo + t_fast_lo + t_slow_lo

    # do this when i'm back
    ax[big].plot(times, t_fast_lo / total_warming_lo, linestyle='solid',
                 color=wind_colors[1])
    ax[big].fill_between(times, np.zeros_like(times), t_fast_lo / total_warming_lo,
                         color=wind_colors[1], alpha=0.5, label='Fast mode')

    ax[big].plot(times, (t_fast_lo + t_slow_lo) / total_warming_lo,
                 linestyle='solid',
                 color=wind_colors[2])
    ax[big].fill_between(times, t_fast_lo / total_warming_lo,
                         (t_fast_lo + t_slow_lo) / total_warming_lo,
                         color=wind_colors[2], alpha=0.5,
                         label='Slow mode')

    ax[big].plot(times, (t_fast_lo + t_slow_lo + t_ic_lo) / total_warming_lo,
                 linestyle='solid', color=wind_colors[3])
    ax[big].fill_between(times, (t_fast_lo + t_slow_lo) / total_warming_lo,
                         (t_fast_lo + t_slow_lo + t_ic_lo) / total_warming_lo,
                         color=wind_colors[3], alpha=0.5,
                         label='Initial conditions')

    ax[big].spines['right'].set_visible(False)
    ax[big].set_xlim((0, 55))

    # break x axis to show what happens in equilibrium
    ax[sma].plot(times, t_fast_lo / total_warming_lo, linestyle='solid',
                 color=wind_colors[1])
    ax[sma].fill_between(times, np.zeros_like(times),
                         t_fast_lo / total_warming_lo,
                         color=wind_colors[1], alpha=0.5, label='Fast mode')

    ax[sma].plot(times, (t_fast_lo + t_slow_lo) / total_warming_lo,
                 linestyle='solid',
                 color=wind_colors[2])
    ax[sma].fill_between(times, t_fast_lo / total_warming_lo,
                         (t_fast_lo + t_slow_lo) / total_warming_lo,
                         color=wind_colors[2], alpha=0.5,
                         label='Slow mode')

    ax[sma].plot(times, (t_fast_lo + t_slow_lo + t_ic_lo) / total_warming_lo,
                 linestyle='solid', color=wind_colors[3])
    ax[sma].fill_between(times, (t_fast_lo + t_slow_lo) / total_warming_lo,
                         (t_fast_lo + t_slow_lo + t_ic_lo) / total_warming_lo,
                         color=wind_colors[3], alpha=0.5,
                         label='Initial conditions')

    ax[sma].spines['left'].set_visible(False)
    ax[sma].set_xlim((890, 1000))
    ax[sma].set_xticks([900, 1000])
    ax[sma].tick_params(labelleft='off')
    ax[sma].set_yticks([])
    ax[sma].set_yticklabels([])
    ax[sma].set_xticklabels([None, r'$\infty$'])

    ax[big].set_ylim((0, 1.01))
    ax[sma].set_ylim((0, 1.01))


make_panel(fig, ax, 'a', 'b', t_ic_lo, t_fast_lo, t_slow_lo)
make_panel(fig, ax, 'c', 'd', t_ic_av, t_fast_av, t_slow_av)
make_panel(fig, ax, 'e', 'f', t_ic_hi, t_fast_hi, t_slow_hi)

ax['a'].set_title("Low ECS")
ax['c'].set_title("Average ECS")
ax['e'].set_title("High ECS")
ax['a'].set_ylabel("Relative share of global warming")
ax['a'].set_xlabel("Time (years)")
ax['c'].set_xlabel("Time (years)")
ax['e'].set_xlabel("Time (years)")

ax['e'].legend(facecolor='white', frameon=True)

# set panel labels
for label, ax in ax.items():
    ax.annotate(
        label,
        xy=(0.02, 0.98), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=16, verticalalignment='top', fontweight='bold',
        bbox=dict(facecolor='none', edgecolor='none'))

if save_figs:
    fig_name = basefile + "rel-frac-warming.png"
    fig.savefig(fig_name, dpi=400)

else:
    plt.show()
