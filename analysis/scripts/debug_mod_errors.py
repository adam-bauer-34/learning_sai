import gc
import argparse
import yaml

import matplotlib.pyplot as plt
import numpy as np

from datatree import open_datatree
from scipy.stats import describe

from var_assim.plotting.presets import get_presets
from var_assim.plotting.filtering import (
    filter_by_max_conceivable,
    filter_datatree_by_cost_ratio_memeff,
)
from var_assim.plotting.pproc import get_angle_r2, get_angle_r3
from var_assim.plotting.utils import make_figure_filename
from var_assim.config import (
    PRIOR_PATH,
    NOISE_PATH,
    TRUTH_PATH,
    WINDOW_PATH,
    DATA_DIR_ABS,
    FIGS_DIR,
)
from var_assim.calibration.priors import ClimateModelPriors
from var_assim.calibration.noise import ClimateModelNoise
from var_assim.calibration.truth import ClimateModelTruth
from var_assim.calibration.windowing import AssimilationWindowing

presets, _ = get_presets()
plt.rcParams.update(presets)
SAVE_FIGS = False

dt_15 = open_datatree(
    DATA_DIR_ABS
    / "output"
    / "pco2geowc3_reg"
    / "var-assim-output_ssp245_pco2geowc3_reg_ws_gradual_AR1+reg_TMIN2023_THETA15_ECS3.0_DEGpDEC0.1_NYRSRAMP50_Nens500.nc"
)

ds = dt_15["2100"].ds

qR1s_final = ds.controls.sel(vari="qR1_75").values
qR2s_final = ds.controls.sel(vari="qR2_75").values
qR3s_final = ds.controls.sel(vari="qR3_75").values

# print(f"Final r1s: {qR1s_final}")
# print(f"Final r2s: {qR2s_final}")
# print(f"Final r3s: {qR3s_final}")

x = ["qR1_" + str(i) for i in range(75)]
y = ["qR2_" + str(i) for i in range(75)]
z = ["qR3_" + str(i) for i in range(75)]

print(f"Median qr1s: {ds.controls.sel(vari=x).median('ens_mem').values}")
print(f"Mean qr1s: {ds.controls.sel(vari=x).mean('ens_mem').values}")
print(f"Std qr1s: {ds.controls.sel(vari=x).std('ens_mem').values}")
print(f"Max qr1s: {ds.controls.sel(vari=x).max('ens_mem').values}")
print(f"Min qr1s: {ds.controls.sel(vari=x).min('ens_mem').values}")

print(f"Median qr2s: {ds.controls.sel(vari=y).median('ens_mem').values}")
print(f"Mean qr2s: {ds.controls.sel(vari=y).mean('ens_mem').values}")
print(f"Std qr2s: {ds.controls.sel(vari=y).std('ens_mem').values}")
print(f"Max qr2s: {ds.controls.sel(vari=y).max('ens_mem').values}")
print(f"Min qr2s: {ds.controls.sel(vari=y).min('ens_mem').values}")

print(f"Median qr3s: {ds.controls.sel(vari=z).median('ens_mem').values}")
print(f"Mean qr3s: {ds.controls.sel(vari=z).mean('ens_mem').values}")
print(f"Std qr3s: {ds.controls.sel(vari=z).std('ens_mem').values}")
print(f"Max qr3s: {ds.controls.sel(vari=z).max('ens_mem').values}")
print(f"Min qr3s: {ds.controls.sel(vari=z).min('ens_mem').values}")
