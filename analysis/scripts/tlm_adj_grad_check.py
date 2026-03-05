"""Plot check plots for correct TLM and ADJ models.

Adam Michael Bauer
University of Illinois Urbana-Champaign
5.13.2024

To run: python check_plots_param.py [save_figs]
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.presets import get_presets
import matplotlib.transforms as mtransforms

# get presets
presets, basefile = get_presets()
plt.rcParams.update(presets)

# unpack command line inputs
assim = sys.argv[1]
save_figs = int(sys.argv[2])

# get paths and import data
cwd = os.getcwd()

tlm_path = cwd + '/../../model/data/checks/' + assim + '/tlm_check.csv'
adj_id_path = cwd + '/../../model/data/checks/' + assim + '/adj_id_check.csv'
cost_grad_path = cwd + '/../../model/data/checks/' + assim + '/cost_grad_check.csv'

# read csvs
tlm_df = pd.read_csv(tlm_path, delimiter=',')
adj_df = pd.read_csv(adj_id_path, delimiter=',')
cost_df = pd.read_csv(cost_grad_path, delimiter=',')

fig, ax = plt.subplot_mosaic([['a', 'b', 'c'], ['d', 'e', 'f']],
                             figsize=(14, 8))

# plot tlm check
ax['a'].plot(tlm_df['perturbation_size'], tlm_df['norms'], label="Data")
ax['a'].set_xscale("log")
ax['a'].set_ylabel(r"$R_{\alpha}$")
ax['a'].set_xlabel(r"Perturbation size, $\alpha$")
ax['a'].set_title(r"$R_{\alpha} = \frac{|| M(\vec{X}_0 + \alpha \delta \vec{X}_0) - M(\vec{X}_{0}) ||}{|| \alpha L \delta \vec{X}_0||}$")
ax['a'].hlines(1, min(tlm_df['perturbation_size']), max(tlm_df['perturbation_size']), linestyle='dashed', 
               color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], zorder=0, label="Limiting value")
ax['a'].legend()
ax['a'].set_ylim((0.8, 1.2))

# plot log(identity - 1)
ax['d'].plot(tlm_df['perturbation_size'], tlm_df['log(|norms - 1|)'], label="Data")
ax['d'].set_xscale("log")
ax['d'].set_ylabel(r"$\log_{10}(R_{\alpha} - 1)$")
ax['d'].set_xlabel(r"Perturbation size, $\alpha$")

# plot adjoint identity
ax['b'].plot(adj_df['timesteps taken'], adj_df['identity'])
ax['b'].set_ylabel(r"$\Lambda_{T}$")
ax['b'].set_xlabel(r"Timesteps taken, $T$")
ax['b'].set_title(r"$\Lambda_{T} = \frac{\langle L\delta \vec{X}_0, L \delta \vec{X}_0 \rangle}{\langle\delta \vec{X}_0, L^{*} \left( L \delta \vec{X}_0 \right) \rangle}$")
ax['b'].hlines(1, min(adj_df['timesteps taken']), max(adj_df['timesteps taken']), linestyle='dashed', 
               color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], zorder=0)
ax['b'].set_ylim((0.95, 1.05))

if adj_df['timesteps taken'].values[-1] > 20:
    ax['b'].set_xlim((0, 20))

# plot adjoint identity
ax['e'].plot(adj_df['timesteps taken'], adj_df['log(|identity - 1|)'])
ax['e'].set_ylabel(r"$\log_{10}(|\Lambda_{T} - 1|)$")
ax['e'].set_xlabel(r"Timesteps taken, $T$")

if adj_df['timesteps taken'].values[-1] > 20:
    ax['e'].set_xlim((0, 20))

# plot cost function derivative
ax['c'].plot(cost_df['perturbation_size'], cost_df['phi'])
ax['c'].set_xscale("log")
ax['c'].set_ylabel(r"$\Phi_{\alpha}$")
ax['c'].set_xlabel(r"Perturbation size, $\alpha$")
ax['c'].set_title(r"$\Phi_{\alpha} = \frac{J( \vec{X} + \alpha \vec{h}) - J(\vec{X})}{\alpha \vec{h}^T \vec{\nabla} J(\vec{X})}$")
ax['c'].set_ylim((0, 3))
ax['c'].hlines(1, min(cost_df['perturbation_size']), max(cost_df['perturbation_size']), linestyle='dashed', 
               color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], zorder=0)
ax['c'].set_ylim((-1, 3))

# plot log(cost function derivative - 1)
ax['f'].plot(cost_df['perturbation_size'], cost_df['log(phi - 1)'])
ax['f'].set_xscale("log")
ax['f'].set_ylabel(r"$\log_{10}(\Phi_{\alpha} - 1$)")
ax['f'].set_xlabel(r"Perturbation size, $\alpha$")

labels = ['a', 'b', 'c', 'd', 'e', 'f']
ls = ['a', 'b', 'c', 'd', 'e', 'f']

for i in range(6):
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[labels[i]].text(0.85, 1.0, ls[i], transform=ax[labels[i]].transAxes + trans, 
                       fontweight='bold', fontsize=20, 
                       verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))

fig.tight_layout()

if save_figs:
    filename = basefile + assim + "-check-plots.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print("\nPlot successfully saved to: {}\n".format(filename))

plt.show()
