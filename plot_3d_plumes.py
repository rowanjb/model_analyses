# For making movies of convective plumes in 3D
# Currently broken and I'm at my wit's end
# Axes3D doesn't really work with vmin and vmax in the colourbar

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm, Normalize
from xmitgcm import open_mdsdataset 
from MITgcmutils import density
import basic_model_anayses as bma
import xgcm
import os
import gsw

def chop_ds(ds):
    """Shrink the dataset by dividing into sections and revealing the faces that you want to plot."""
    X = ds['XC'].size//2 # Y = ds['YC'].size//2  
    ds = ds.isel(XC=slice(0,X),XG=slice(0,X))#YC=slice(0,Y),YG=slice(0,Y))
    return ds

def get_mins_and_maxs(ds,variable,plane):
    """Return the min and max values pertinent to the plots (i.e., on the exposed planes)."""
    da = ds[variable]
    if plane=='horizontal':
        min = da.isel(Z=0).min().to_numpy()
        max = da.isel(Z=0).max().to_numpy()
    elif plane=='vertical':
        try: 
            min = da.isel(XC=-1).min().to_numpy()
            max = da.isel(XC=-1).max().to_numpy()
        except:
            min = da.isel(XG=-1).min().to_numpy()
            max = da.isel(XG=-1).max().to_numpy()
    minmax = min, max
    return minmax

def plume_plot_engine(ds, run, figs_dir, vertical_plane_variable, horizontal_plane_variable, horizontal_plane_minmax, vertical_plane_minmax, i_time=10, dt=10, cb=False):
    """Plotting mechanism for creating ONE plot. 
    i_time is the index of the time you want to plot. 
    minmax parameters are tuples. 
    cb controls the colourbar; can either plot the cb or the fig, not both."""
    # https://matplotlib.org/stable/gallery/mplot3d/box3d.html#sphx-glr-gallery-mplot3d-box3d-py 

    Z, Y, X = np.meshgrid(ds['Z'].values, ds['YC'].values, ds['XC'].values, indexing='ij')

    timestep_str = str(ds['time'].dt.seconds.to_numpy()) #mightn't work > 1 day!!!!!!!
    time_hours_str = str(float(timestep_str)*dt/60/60) # Have to do this manually 

    horizontal_np = ds[horizontal_plane_variable].isel(Z=0).to_numpy()
    try: vertical_np = ds[vertical_plane_variable].isel(XC=-1).to_numpy()
    except: vertical_np = ds[vertical_plane_variable].isel(XG=-1).to_numpy()

    horizontal_min, horizontal_max = horizontal_plane_minmax 
    linscale, linthresh = 0.001, 0.0001
    positive_levels = np.logspace(np.log10(linthresh), np.log10(horizontal_max), 51)
    negative_levels = -np.logspace(np.log10(linthresh), np.log10(-horizontal_min), 51)
    levels = np.concatenate([negative_levels[::-1], [0], positive_levels]) #[0]
    kw_horizontal = { 'vmin': horizontal_min, 'vmax': horizontal_max, 'levels': levels,
        'norm': SymLogNorm(linscale=linscale, linthresh=linthresh, vmin=horizontal_min, vmax=horizontal_max)}

    vertical_min, vertical_max = vertical_plane_minmax
    
    """
    if vertical_plane_variable=='T': 
        kw_vertical = {'levels': 102, 'vmin': vertical_min, 'vmax': vertical_max, 'norm': 'linear'} 
        cm = 'seismic' 
    elif vertical_plane_variable=='s':
        kw_vertical = {'levels': 102, 'vmin': vertical_min, 'vmax': vertical_max, 'norm': 'linear'}
        cm = 'viridis' 
    """

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    C_horizontal = ax.contourf(
        X[0, :, :], Y[0, :, :], horizontal_np[:, :],
        zdir='z', offset=0, cmap='seismic', **kw_horizontal
    )

    norm = Normalize(vmin=vertical_min, vmax=vertical_max)
    print(vertical_plane_minmax)
    C_vertical = ax.contourf(
        vertical_np[:, :], Y[:, :, -1], Z[:, :, -1], #the :-1's are to avoid plotting the bottom row of non-zero values
        zdir='x', offset=X.max(), cmap='seismic', levels=102, vmin=vertical_min, vmax=vertical_max, norm=norm
    )
    cbar = fig.colorbar(C_vertical, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')
    cbar.set_ticks([vertical_min, vertical_max])

    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Set zoom and angle view
    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=0.9)

    # Scale X and Y axes properly
    ax.set_aspect('equalxy')
    
    #https://stackoverflow.com/questions/44001613/matplotlib-3d-surface-plot-turn-off-background-but-keep-axes
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    # Set labels and zticks
    ax.set(
        xlabel='X ($m$)',
        ylabel='Y ($m$)',
        zlabel='Depth ($m$)',
    )

    ax.xaxis._axinfo['juggled'] = (2,0,0)

    ax.set_title(run+', +'+time_hours_str + ' hrs')
    
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    
    # Save figure
    timestep_str = timestep_str.zfill(10)

    #c_bar = fig.colorbar(C_vertical)
    #c_bar.set_clim(vmin=vertical_min, vmax=vertical_max)

    '''
    bbox_ax = ax.get_position()
    #C_horizontal.set_clim(horizontal_min, horizontal_max)
    #C_vertical.set_clim(vertical_min, vertical_max)
    cbar_horizontal_ax = fig.add_axes([0.98,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0]) #([bbox_ax.x0, 0.09, bbox_ax.x1-bbox_ax.x0, 0.02]) # for horiz bars
    cbar_horizontal = plt.colorbar(C_horizontal, cax=cbar_horizontal_ax,shrink=0.5, pad=0.1, label='Vorticity ($s^{-1}$)')
    cbar_horizontal.formatter.set_powerlimits((0, 0))
    cbar_horizontal.formatter.set_useMathText(True)
    cbar_horizontal.update_ticks()
    vertical_labels = {'N2': '$N^2$ ($s^{-2}$)', 'rho': r"$\rho$ ($kg$ $m^{-3}$)", 'T': 'Temperature ($℃$)', 'S': 'Salinity ($PSU$)'}
    cbar_vertical_ax = fig.add_axes([1.22,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0])
    cbar_vertical = plt.colorbar(C_vertical, cax=cbar_vertical_ax, extend='both', shrink=0.5, pad=0.1, label=vertical_labels[vertical_plane_variable])
    
    cbar_vertical.formatter.set_powerlimits((0, 0))
    cbar_vertical.formatter.set_useMathText(True)
    cbar_vertical.update_ticks()
    '''
    plt.savefig(figs_dir+'/plume_'+timestep_str+'.png',dpi=450,bbox_inches="tight")
    print(figs_dir+'/plume_'+timestep_str+'.png')
    plt.close()

def run_plume_plot(run,vertical_plane_variable='T',horizontal_plane_variable='zeta'):
    
    # Some necessary filepaths
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/model_analyses/figures/figs_'+run+'_'+horizontal_plane_variable+'_'+vertical_plane_variable
    
    # Opening the data
    ds = bma.open_binaries_all_vars(data_dir) 
    
    # Adding any necessary variables (I'll keep expanding this with new variables later)
    if horizontal_plane_variable=='zeta':
        ds = bma.calculate_zeta(ds) 
    
    # Shrink the dataset by dividing into sections and revealing the faces that you want to plot
    ds = chop_ds(ds)
    
    # Necessary for the colourbars; can also manually specify; will make more professional later
    vertical_plane_minmax = get_mins_and_maxs(ds,vertical_plane_variable,'vertical')
    horizontal_plane_minmax = get_mins_and_maxs(ds,horizontal_plane_variable,'horizontal')
    vertical_plane_minmax = (-1.8,1.8)
    horizontal_plane_minmax = (-0.04,0.04)#(-0.0075,0.0125)
    
    # Plotting    
    # Based on Vreugdenhil and Gayen 2021
    i_times = len(ds.time.to_numpy())
    for i_time in range(i_times):
        ds_i_time = ds.isel(time=i_time)
        plume_plot_engine(ds_i_time, run, figs_dir, vertical_plane_variable, horizontal_plane_variable, horizontal_plane_minmax, vertical_plane_minmax, i_time, dt=3) 

if __name__ == "__main__":
    run, vertical_plane_variable, horizontal_plane_variable = 'mrb_017', 'T', 'zeta'
    run_plume_plot(run, vertical_plane_variable, horizontal_plane_variable)
