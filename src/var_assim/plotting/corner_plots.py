import numpy as np
import seaborn as sb
import pandas as pd

def gen_dataframe(ds, angles, angles_prior,
                  true_val_indx, year_indx, Nens):
    
    data = np.zeros((13, Nens))

    # define mask for angles
    mask = angles > np.percentile(angles_prior, 95)

    data[0] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='L').values)
    data[1] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='G').values)
    data[2] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='EPS').values)
    data[3] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='C1').values)
    data[4] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='C2').values)
    data[5] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='F1_CO2').values)
    data[6] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='ALPHA_R1').values)
    data[7] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='ALPHA_R2').values)
    data[8] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='BETA_R1').values/0.09)
    data[9] = np.where(mask, 
                       np.nan, ds.controls.sel(vari='BETA_R2').values/0.09)
    data[10] = np.where(mask, 
                        np.nan, angles)
    data[11] = np.where(mask, 
                        np.nan, data[5] / data[0])  # ECS
    data[12] = np.where(mask, 
                        np.nan, data[5] / (data[0] + data[1] * data[2]))  # TCR