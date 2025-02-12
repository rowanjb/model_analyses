# For short functions relating to model analyses
# A constant work-in-progress 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm
from xmitgcm import open_mdsdataset 
from MITgcmutils import density
import xgcm
import os

def avg_temp(data_dir):
    """For comparing average temperatures across runs."""
    ds = open_mdsdataset(data_dir,geometry='cartesian',prefix=['S','T','U','V'])
    areas = np.tile(ds['rA'].values[:, :, np.newaxis],(1,1,50))
    thicknesses = np.array([ 0.4, 1.2, 2.0, 2.8, 3.6, 4.4, 5.2, 6.0, 6.8, 7.6,
                             8.4, 9.2,10.0,10.8,11.6,12.4,13.2,14.0,14.8,15.6,
                            16.4,17.2,18.0,18.8,19.6,20.4,21.2,22.0,22.8,23.6,
                            24.4,25.2,26.0,26.8,27.6,28.4,29.2,30.0,30.8,31.6,
                            32.4,33.2,34.0,34.8,35.6,36.4,37.2,38.0,38.8,39.6])
    vols = areas*thicknesses
    ds['vols'] = (('YC', 'XC', 'Z'), vols)
    weights = ds['vols']
    weights.name = "weights"
    T_weighted = ds['T'].weighted(weights)
    T_weighted_mean = T_weighted.mean()
    print(T_weighted_mean.values)
    #avg_temp('../MITgcm/so_plumes/mrb_002') # 0.2599955576563594 production run
    #avg_temp('../MITgcm/so_plumes/mrb_003') # 0.2601810829108668 side tau =86400
    #avg_temp('../MITgcm/so_plumes/mrb_004') # 0.2599837272808078 side tau =8640
    #avg_temp('../MITgcm/so_plumes/mrb_005') # 0.2614877021773202 side tau =864 *crashes
    #avg_temp('../MITgcm/so_plumes/mrb_006') # 0.2600030499680336 side tau =864000
    #avg_temp('../MITgcm/so_plumes/mrb_007') # 0.26368029388014746 full tau =86400
    #avg_temp('../MITgcm/so_plumes/mrb_008') # 0.2772914436059543 full tau =8640
    #avg_temp('../MITgcm/so_plumes/mrb_011') # 0.26413005811064794    

def open_mitgcm_output_all_vars(data_dir):
    """Returns a dataset associated with the specified directory (which contains the model output) that has all variables, 
    included ['S','T','U','V','Eta'] and ['PH','PHL']. Note that the pressures have -1 time indices compared to the other vars."""
    ds = xr.merge([ # The pressures sometimes are missing the first timestep
        open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['S','T','U','V','Eta']),
        open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['PH']),
        open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['PHL'])])    
    return ds

def cell_diffs(ds):
    """Returns list with cell dz based on ds['Z']==cell depths."""
    cell_depths = ds['Z'].values
    cell_diffs = np.full(len(cell_depths),999.) # init at 999 for unambiguity
    for n,i in enumerate(cell_depths[:-1]): 
        cell_diffs[n+1] = cell_depths[n+1] - i
    return cell_diffs # should it be negative?

def calculate_pressure(ds,rhoConst=1000,gravity=10,):
    """Calculates sea pressure, since the model doesn't output it directly.
    Adds pressure as a variable?"""
    #today, test how accurate the approximate version is compared to the gsw/init bins
    #http://mailman.mitgcm.org/pipermail/mitgcm-support/2013-August/008449.html 
    #http://mailman.mitgcm.org/pipermail/mitgcm-support/2016-August/010541.html
    #http://mailman.mitgcm.org/pipermail/mitgcm-support/2013-November/008636.html 
    #CAN LIKELY STICK TO THE LINEAR EOS IF THE NEW N2 CALCS YIELD STABILITY 
    # Is rho ref given or base donf tref and sref?? 

def calculate_zeta(ds):
    """Calculates vorticity in and adds it as a variable."""
    # Regarding interpolating; don't forget to manually test this! (Both the interpolating and the vorticity.)
    # Uses the xgcm package for interpolating and various related calculations
    # Also test if you got the signs right
    # Example from: https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
    grid = xgcm.Grid(ds,periodic=False) 
    zeta = (-grid.diff(ds.U * ds.dxC, 'Y') + grid.diff(ds.V * ds.dyC, 'X'))/ds.rAz
    zeta = grid.interp(zeta,['X','Y'])
    ds['zeta'] = zeta
    #zeta = zeta.isel(XC=slice(0,76)).transpose('YC','XC','Z','time') # Slicing and interpolating for plotting reasons
    return ds 

'''
def SA 

def rho 

def zeta
pressure
pot temp
def circulation
'''

if __name__ == "__main__":
    
    run = 'mrb_011'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    ds = open_mitgcm_output_all_vars(data_dir).isel(time=0)
    fig, ax = plt.subplots()
    T = ds['T'].isel(XC=50,YC=50).values
    Z = ds['Z'].values
    ax.plot(T,Z)
    ax.set_title('')
    plt.savefig('test.png')
    #calculate_pressure(ds)
    calculate_zeta(ds)