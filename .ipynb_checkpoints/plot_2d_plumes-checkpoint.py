# For making movies of convective plumes in 2D
# Simpler than in 3D (i.e., plot_3d_plumes.py)
# Some help from GitHub copilot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm, Normalize
from xmitgcm import open_mdsdataset 
from MITgcmutils import density
import basic_model_anayses as bma
import datetime
import xgcm
import os
import gsw

def plot_vertical_plane(da, var, figs_dir, vmin=None, vmax=None):
    """Make a 2D plot of some variable at the center of the domain.
    da must have only 1 time."""

    # Shrink the dataset by revealing the face that you want to plot
    # The dimension will be either XC or XG; we'll try both
    try: 
        X = da['XC'].size//2
        da = da.isel(XC=X)
        x_dim, y_dim = 'XC', 'YC'
    except: 
        X = da['XG'].size//2
        da = da.isel(XC=X)
        x_dim, y_dim = 'XG', 'YG'
    
    # Extract the time
    td = pd.to_timedelta(da['time'].data, unit='ns') # Time in nanoseconds SEEMS WRONG
    timedelta_str = '+'+str(td.components.hours) + ' hrs '+str(td.components.minutes)+' min' # Title of the figure
    timestep_str = str(int(td.total_seconds())).zfill(10) # Time for using in the figure file name

    # Colourbar label dictionary 
    cbar_label = {
        'T':          "$θ$ ($℃$)",  
        'S':          "Salinity ($g$ $kg^{-1}$)", # Might depend on EOS?
        'rho':        r"$\rho_{t}$ ($kg$ $m^{-3}$)",
        'rho_theta':  r"$\rho_{θ}$ ($kg$ $m^{-3}$)",
        'N2':         "Buoyancy frequency ($s^{-2}$)",
        'quiver':     "Speed ($m$ $s^{-1}$)"
        }
    
    # Colourmap dictionary
    cmap = {
        'T':          "seismic",
        'S':          "viridis", # Might depend on EOS?
        'rho':        "copper_r",
        'rho_theta':  "copper_r",
        'N2':         "RdGy_r",
        'quiver':     "viridis_r",
        }

    # Create a directory for the figures if it doesn't exits
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    # Plot
    plt.rcParams["font.family"] = "serif" # change the base font
    fig, ax = plt.subplots(figsize=(3.54, 2.5)) # 3.54 is a typical half-page figure width
    if var=='quiver': # i.e., if you have multiple variables (i.e., da is really a dataset)
        p = xr.plot.pcolormesh(da['speed'], vmin=vmin, vmax=vmax, cmap=cmap[var], cbar_kwargs={'label': cbar_label[var]})
        n = 3
        da.isel(YC=slice(None,None,n),Z=slice(None,None,n)).plot.quiver(x='YC',y='Z',u='V',v='W',scale=15, add_guide=False)
    else:
        p = xr.plot.pcolormesh(da[var], vmin=vmin, vmax=vmax, cmap=cmap[var], cbar_kwargs={'label': cbar_label[var]})
    cbar = p.colorbar
    cbar.ax.tick_params(labelsize=9)  # Font size for colorbar ticks
    cbar.set_label(cbar_label[var], size=9) 
    ax.set_title(timedelta_str,fontsize=11)
    ax.set_ylabel('Depth ($m$)',fontsize=9)
    ax.set_xlabel('Y ($m$)',fontsize=9)
    ax.tick_params(size=9)
    plt.tight_layout()
    plt.savefig(figs_dir+'/plume2D_'+timestep_str+'.png',dpi=450,bbox_inches="tight")
    print(figs_dir+'/plume2D_'+timestep_str+'.png created')
    plt.close(fig)

def run_plot_vertical_plane(run, var, vmin=None, vmax=None, eos=None):
    """Run plot_vertical_plane in a loop over time.
    Do it this way (separate functions) so that plots for 1 time can be created with ease."""
    
    # Some necessary filepaths
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/model_analyses/figures/figs2D_'+run+'_'+var
    
    # Opening the data
    ds = bma.open_mitgcm_output_all_vars(data_dir,var=var)
    
    # Adding any necessary variables (I'll keep expanding this with new variables later)
    if var=='zeta':
        ds = bma.calculate_zeta(ds) 
        print('Zeta added to the dataset')
    if var=='quiver':
        ds = bma.colocate_velocities(ds)
        print('Velocities co-located at C points')
    if var=='N2':
        if eos=='LINEAR':
            ds = bma.calculate_N2_linear_EOS(ds)
        elif eos=='TEOS10':
            ds = bma.calculate_N2_TEOS10(ds)
        else:
            print("You need to specify eos='LINEAR' or eos='TEOS10'")
            return
        print('N2 added to the dataset')

    # Extracting the variable that we're interested in 
    var_dir = {'T':['T'],'S':['S'],'quiver':['V','W','speed'],'N2':['N2']} # Needed 
    da = ds[var_dir[var]]

    # Plotting    
    i_times = len(da.time.to_numpy())
    for i_time in range(i_times):
        plot_vertical_plane(da.isel(time=i_time), var, figs_dir, vmin=vmin, vmax=vmax)

if __name__ == "__main__":
    run_plot_vertical_plane(run='mrb_028', var='N2',eos='TEOS10')
    quit()
    run_plot_vertical_plane(run='mrb_028', var='quiver', vmin=0, vmax=2)
    run_plot_vertical_plane(run='mrb_028', var='T', vmin=-2, vmax=2)
    run_plot_vertical_plane(run='mrb_028', var='S', vmin=34.4, vmax=34.9)
    