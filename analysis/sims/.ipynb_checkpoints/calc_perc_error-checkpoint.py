"""Calculate percent error of ensemble fits to data in 4DVAR.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.6.2024

To run: python calc_perc_error.py assim TMIN TMAX N obs save_tab
"""

import sys
import os

import numpy as np
import pandas as pd
import xarray as xr

# save table?
assim = sys.argv[1]
TMIN = int(sys.argv[2])
TMAX = int(sys.argv[3])
N = int(sys.argv[4])
obs = sys.argv[5]
save_tab = int(sys.argv[6])

# pull the file for data
file = "../model/data/output/" + assim + "_wind" + str(TMIN) + "_" + str(TMAX)\
    + "_N" + str(N) + "_" + obs + ".nc"

# make dataset
ds = xr.open_dataset(file)

# extract true control variables and the ensemble control variables
controls_truth = ds.controls_truth.values
controls = ds.controls.values

# compute percent error for each fit
perc_errors = 100 * abs(((controls - controls_truth[:, None])
                         / controls_truth[:, None]))

# stack mean and std of percent errors
perc_errors = np.hstack([perc_errors, np.mean(perc_errors, axis=1)[:, None],
                         np.std(perc_errors, axis=1)[:, None]])

# make big ol' dictionary for saving
data_dict = {'member': np.hstack([np.arange(0, N, 1), ['means'], ['var']]),
             'perc_error_T1': perc_errors[0],
             'perc_error_T2': perc_errors[1],
             'perc_error_Q': perc_errors[2],
             'perc_error_L': perc_errors[3],
             'perc_error_G': perc_errors[4],
             'perc_error_C1': perc_errors[5],
             'perc_error_C2': perc_errors[6],
             'perc_error_F1': perc_errors[7],
             'perc_error_F3': perc_errors[8],
             'perc_error_A_SO2': perc_errors[9],
             'perc_error_B_SO2': perc_errors[10],
             'perc_error_C_SO2': perc_errors[11]}

# make dataframe
df = pd.DataFrame.from_dict(data_dict)

# if you want, save, and if not, print
if save_tab:
    cwd = os.getcwd()
    filename = cwd + "/data/perc_error/perc_error_" + assim + "_wind"\
        + str(TMIN) + "_" + str(TMAX) + "_N" + str(N) + "_" + obs\
        + "_perc_error.csv"

    df.to_csv(filename)
    print("Data successfully saved to:\n{}".format(filename))

else:
    print(df)
