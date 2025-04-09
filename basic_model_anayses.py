# For short functions relating to model analyses
# A constant work-in-progress 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm
from xmitgcm import open_mdsdataset 
from MITgcmutils import density
import gsw
import xgcm
import os
import re

def open_mitgcm_output_all_vars(data_dir,var='all',iter='all',fix_timestep=True):
    """Returns a dataset associated with the specified directory (which contains the model output) that has all variables by default
    (including ['S','T','U','V','Eta'] and ['PH','PHL']), OR you can pass in a "var", which in this case refers to a variable that you
    might want to plot like 'T', 'S', 'rho', 'rho_theta', 'N2', and 'quiver'. Note that the pressures have -1 time indices 
    compared to the other vars.
    fix_timestep can be True (default; automatically search for a deltaT to apply to the output), an integer (apply this deltaT to the 
    output) or False (not recommended; let the output time be the timestep expressed as nanoseconds)."""
    if var=='all':
        ds = xr.merge([ # The pressures sometimes are missing the first timestep
            open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['S','T','U','V','W','Eta'], iters=iter),
            open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['PH'], iters=iter),
            open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['PHL'], iters=iter),
            #open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['PNH']), # often missing?
            ])
    elif var=='T': ds = open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['T'], iters=iter)
    elif var=='S': ds = open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['S'], iters=iter)
    elif var=='quiver': ds = open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['U','V','W'], iters=iter)
    elif var=='rho' or var=='rho_theta': 
        ds = open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['S','T'], iters=iter)
    elif var=='N2':
        #ds = open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['S','T'])
        ds = xr.merge([ # The pressures sometimes are missing the first timestep
            open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['S','T','Eta'], iters=iter), # Eta improves the Z accuracy slightly
            open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['PH'], iters=iter),
        #    open_mdsdataset(data_dir,   geometry='cartesian',   prefix=['PHL']),
            ])
    # For some reason the iters parameter isn't working as I expect, so I'm using this "if" instead. 
    # Likely slow but maybe not due to lazy loading. Square brackets are to preserve the len-1 time dim.
    else:
        print("Illegal variable name")
        return
    #if iter!='all': ds = ds.isel(time=[iter]) 
    if fix_timestep!=False: # If we want to fix the timestep...
        if fix_timestep==True: # In this case, seach for a deltaT and correct the (incorrect) np.timedelta64 
            try: 
                with open(data_dir+'/data','r') as f: data = f.readlines()
                for line in data: 
                    if 'deltaT' in line:
                        deltaT = int(re.sub(r'\D', '', line)) # Replaces any character that is not a digit with a string and outputs int
                try: 
                    ds['time'] = ds['time']*deltaT
                except:
                    print("Couldn't find a deltaT in the data file")
                    quit()
            except:
                print("Couldn't open the data file")
                quit()
        elif type(fix_timestep)==int:
            ds['time'] = ds['time']*fix_timestep
        else:
            print("Invalid fix_timestep")
            quit()
    return ds

def colocate_velocities(ds):
    """Co-locate velocity vectors at T points so that quiver plots (etc.) can be made.
    Makes use of the xgcm grid object and the interp method.
        Original dims:          Updated dims:
            W (Zl, YC, XC) -------> W (Z, YC, XC)
            U (Z, YC, XG) --------> U (Z, YC, XC)
            V (Z, YG, XC) --------> V (Z, YC, XC)"""
    # 19.02.2025; based off: https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
    # For the horizontal grid, see: https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html 
    # For the vertical grid, see: https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html
    grid = xgcm.Grid(ds,periodic=False) # Create the xgcm grid
    ds['U'] = grid.interp(ds['U'], 'X')
    ds['V'] = grid.interp(ds['V'], 'Y')
    ds['W'] = grid.interp(ds['W'], 'Z')
    ds['speed'] = (ds['U']**2 + ds['V']**2 + ds['W']**2)**0.5
    return ds 

def calculate_zeta(ds):
    """Calculates vorticity and adds it as a variable."""
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

def cell_diffs(ds):
    """Returns list with cell dz based on ds['Z']==cell depths.
    Mostly not necessary, since the model outputs cell thicknesses in drC and drF."""
    cell_depths = ds['Z'].values
    cell_diffs = np.full(len(cell_depths),999.) # init at 999 for unambiguity
    for n,i in enumerate(cell_depths[:-1]): 
        cell_diffs[n+1] = cell_depths[n+1] - i
    return cell_diffs # should it be negative?

def calculate_pressure(ds,rhoConst=1000,g=10):
    """Calculates hydrostatic sea pressure, since the model doesn't output it directly. Adds pressure as a variable. 
    In MOST cases, rho = rho_const + rho_prime (always confirm this with your specific simulation).
    And the output PHIHYD (i.e., PH from the diagnostic package) is calculated with 
    the contribution of rho_const omitted, i.e., using rho_prime.
    Hence, P = (PHIHYD + gravity*abs(RC))*rho_const where RC is cell depth. 
    See "stratification.ipynb" or http://mailman.mitgcm.org/pipermail/mitgcm-support/2013-November/008636.html for more details."""
    print("UPDATE THIS TO USE RHOREF AND MAYBE PREF WITH TEOS")
    ds['p'] = (-1)*g*rhoConst*(ds['Z']) + ds['PH']*rhoConst # +ds['Eta'] MIGHT improve the Z accuracy slightly, but I /expect/ not
    ds['p'] = ds['p'].transpose('time', 'Z', 'YC', 'XC')
    return ds 

def calculate_sigma0_TEOS10(ds,lon=-27.0048,lat=-69.0005):
    """Calculates potential density using the Gibbs-SeaWater package.
    Gibbs-SeaWater is based on TEOS-10, so this only works if eosType='TEOS10'.
    Adds sigma0 as a variable named 'rho_theta'.
    sigma0 is potential density in sigma notation with p_ref=0 dbar, i.e., if all water is brought to p_ref adiabatically.
    (I am assuming that thetas are referenced to 0 dbar; should probably check this.)"""
    # Note with TEOS10, the salinity is absolute, not practical (see the MITgcm manual; 'If 
    # TEOS-10 is selected, the model variable salt can be interpreted as “Absolute Salinity”.')
    # See https://www.teos-10.org/pubs/gsw/html/gsw_sigma0.html for more details
    print("You need to test if you can use gsw in this way, feeding it ds and da etc")
    #CT = gsw.CT_from_pt(ds['S'],ds['T'])
    ds['rho_theta'] = gsw.sigma0(ds['S'],gsw.CT_from_pt(ds['S'],ds['T']))
    return ds

def calculate_N2_TEOS10(ds,lat=-69.0005,rhoConst=1000,g=10):
    """Calculates buoyancy frequency using the Gibbs-SeaWater package.
    Gibbs-SeaWater is based on TEOS-10, so this only works if eosType='TEOS10'.
    Adds N2 as a variable.
    Directly uses the gsw.Nsquared function"""
    # See https://www.teos-10.org/pubs/gsw/html/gsw_Nsquared.html for more details about gsw.Nsquared
    # Note with TEOS10, the salinity is absolute, not practical (see the MITgcm manual; 'If 
    # TEOS-10 is selected, the model variable salt can be interpreted as “Absolute Salinity”.')
    # See https://www.teos-10.org/pubs/gsw/html/gsw_sigma0.html for more details
    print("You need to test if you can use gsw in this way, feeding it ds and da etc")
    print("The use of gsw tools causes memory usage to explode and the job to be killed")
    print("UPDATE THIS TO USE RHOREF AND MAYBE PREF WITH TEOS")
    ds['T'] = gsw.CT_from_pt(ds['S'],ds['T']) # The output from CT_from_pt is large, so we're writing over T
    ds = calculate_pressure(ds,rhoConst=rhoConst,g=g) # Adds pressure as a variable
    N2,p_mid = gsw.Nsquared(ds['S'],ds['T'],ds['p'],lat=lat)
    return

def calculate_N2_linear_EOS(ds,g=10,a=0.0002,b=0):
    """Calculates buoyancy frequency and adds it as a variable.
    We can do this with the linear EOS since it allows us to express N2 in terms of theta_z, s_z, alpha, beta, and gravity.
    Important: Assumes constant theta and s reference profiles!
    Also note that I haven't tested this yet with b != 0."""
    # 18.02.2025; much of this code is based off: https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
    grid = xgcm.Grid(ds,periodic=False) # Create the xgcm grid
    # Take the difference between temperature points; 'extend' is kinda like a ghost point; right puts "0" at bottom not surface
    dT = grid.diff(ds['T'],axis='Z',boundary='extend',to='right') # Uses "Zu" as the vertical coord
    # Repeating above with salinity; I have not tested this, so it makes sense in theory but maybe should be reviewed
    dS = grid.diff(ds['S'],axis='Z',boundary='extend',to='right') # Uses "Zu" as the vertical coord
    # Add cell thickness as a coordinate (drF is z thickness, drC is the cell thickness at w locations)
    dT = dT.assign_coords(drC=('Zu',ds['drC'].values[1:])) # Need to slice drC since it has len(Z)+1 values
    # Calculate N2 and add it as a var to ds 
    ds['N2'] = -g*a*dT/dT['drC'] + g*b*dS/dT['drC']  
    return ds 

if __name__ == "__main__":
    
    run = 'mrb_038'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    ds = open_mitgcm_output_all_vars(data_dir)
    print(ds)
    print(ds.time)
    #calculate_N2_TEOS10(ds)