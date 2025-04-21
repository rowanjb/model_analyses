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
import matplotlib.font_manager as fm # For dealing with fonts
import matplotlib.textpath as textpath # For dealing with fonts

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
    d, h, m = str(td.components.days).zfill(2), str(td.components.hours).zfill(2), str(td.components.minutes).zfill(2)
    timedelta_str_nonmono = d+':'+h+':'+m # Title of the figure
    timestep_str = str(int(td.total_seconds())).zfill(10) # Time for using in the figure file name

    # Colourbar label dictionary 
    cbar_label = {
        'T':          "$θ$ ($℃$)",  
        'S':          "Salinity ($g$ $kg^{-1}$)", # Might depend on EOS?
        'rho':        r"$\rho_{t}$ ($kg$ $m^{-3}$)",
        'rho_theta':  r"$\sigma_{θ}$ ($kg$ $m^{-3}$)",
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
    plt.rcParams['font.family'] = "Serif"
    fig, ax = plt.subplots(figsize=(3.54, 2.5)) # 3.54 is a typical half-page figure width
    if var=='quiver': # i.e., if you have multiple variables (i.e., da is really a dataset)
        p = xr.plot.pcolormesh(da['speed'], vmin=vmin, vmax=vmax, cmap=cmap[var], cbar_kwargs={'label': cbar_label[var], 'extend': 'neither'})
        n = 3
        da.isel(YC=slice(None,None,n),Z=slice(None,None,n)).plot.quiver(x='YC',y='Z',u='V',v='W',scale=0.33*n, add_guide=False) # old scale was 15
    else:
        p = xr.plot.pcolormesh(da[var], vmin=vmin, vmax=vmax, cmap=cmap[var], cbar_kwargs={'label': cbar_label[var], 'extend': 'neither'})
    cbar = p.colorbar
    cbar.ax.tick_params(labelsize=9)  # Font size for colorbar ticks
    cbar.set_label(cbar_label[var], size=9) 

    # Temporary
    p.cmap.set_over('white')
    p.cmap.set_under('white')

    ax.set_title(timedelta_str_nonmono, fontsize=11)
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
    
    # Creating filepaths and opening the data
    # Reason for the try-except is that there are only two locations where the data might be
    figs_dir = './figures/figs2D_'+run+'_'+var
    try:
        data_dir = '../MITgcm/so_plumes/'+run
        ds = bma.open_mitgcm_output_all_vars(data_dir,var=var)
    except: 
        data_dir = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/'+run
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
    if var=='rho_theta':
        if eos=='LINEAR':
            print("Can't yet calculate potential density with eos='LINEAR'")
            return
        elif eos=='TEOS10':
            ds = bma.calculate_sigma0_TEOS10(ds)
        else:
            print("You need to specify eos='LINEAR' or eos='TEOS10'")
            return
        print('rho_theta added to the dataset')

    # Extracting the variable that we're interested in 
    var_dir = {'T':['T'],'S':['S'],'quiver':['V','W','speed'],'N2':['N2'],'rho_theta':['rho_theta']} # Needed 
    da = ds[var_dir[var]]

    # Plotting    
    i_times = len(da.time.to_numpy())
    for i_time in range(i_times):
        plot_vertical_plane(da.isel(time=i_time), var, figs_dir, vmin=vmin, vmax=vmax)

if __name__ == "__main__":
    #run_plot_vertical_plane(run='mrb_034', var='quiver', vmin=0, vmax=0.2)
    run_plot_vertical_plane(run='mrb_050', var='T', vmin=-2.05, vmax=2.05)
    run_plot_vertical_plane(run='mrb_050', var='quiver', vmin=0, vmax=0.2)
    run_plot_vertical_plane(run='mrb_050', var='S', vmin=34.4, vmax=34.9)
    run_plot_vertical_plane(run='mrb_050', var='rho_theta', vmin=27.7, vmax=27.85,eos='TEOS10')
