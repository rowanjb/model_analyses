# For making movies of convective plumes in 3D

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm
import numpy as np
from xmitgcm import open_mdsdataset 
from MITgcmutils import density
import xgcm
import os

def cell_diffs(ds):
    """Returns list with cell dz based on ds['Z']==cell depths."""
    cell_depths = ds['Z'].values
    cell_diffs = np.full(len(cell_depths),999.) # init at 999 for unambiguity
    for n,i in enumerate(cell_depths[:-1]): 
        cell_diffs[n+1] = cell_depths[n+1] - i
    return cell_diffs

def calculate_output(data_dir,vertical_variable): 
    """Returns a dataarray of vorticity."""
    # Regarding interpolating; don't forget to manually test this! (Both the interpolating and the vorticity.)
    # Uses the xgcm package for interpolating and various related calculations
    # https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
    ds = open_mdsdataset(data_dir,geometry='cartesian',prefix=['S','T','U','V'])
    grid = xgcm.Grid(ds,periodic=False) 
    zeta = (-grid.diff(ds.U * ds.dxC, 'Y') + grid.diff(ds.V * ds.dyC, 'X'))/ds.rAz
    zeta = grid.interp(zeta,['X','Y'])
    zeta = zeta.isel(XC=slice(0,76)).transpose('YC','XC','Z','time') # Slicing and interpolating for plotting reasons
    ds['rho'] = ( ('time', 'Z', 'YC', 'XC'), density.linear(ds.S, ds.T, sref=35, tref=20, sbeta=0, talpha=0.0002, rhonil=1000)) # SHOULD THIS BE T? PT? IS IT INSITU RHO???

    ds = ds.assign_coords(dz=('Zl',cell_diffs(ds)))
    if vertical_variable=='N2': 
        N2 = (-1) * (9.81) * (1/1000) * grid.diff(ds.rho, 'Z', boundary='extend')/ds['dz']  # replace this with the gsw function and CHECK IF IT SHOULD BE PD or INSITU D???
        N2 = grid.interp(N2,'Z')
        N2 = N2.isel(XC=slice(0,76)).transpose('YC','XC','Z','time')
        out = N2
    elif vertical_variable=='t': out = ds['T'].transpose('YC','XC','Z','time')
    elif vertical_variable=='rho': out = ds['rho'].transpose('YC','XC','Z','time')
    elif vertical_variable=='s': out = ds['S'].transpose('YC','XC','Z','time')
    return zeta, out

def get_mins_and_maxs(da,plane):
    """Return the min and max values pertinent to the plots (i.e., zeta and N2)."""
    if plane=='horizontal':
        min = da.isel(Z=0).min().to_numpy()
        max = da.isel(Z=0).max().to_numpy()
    elif plane=='vertical':
        min = da.isel(XC=-1,Z=slice(0,-1)).min().to_numpy()
        max = da.isel(XC=-1,Z=slice(0,-1)).max().to_numpy()
    minmax = min, max
    return minmax

def plume_plot_engine(run, figs_dir, zeta, vertical_data, vertical_variable, i_time, zeta_minmax=None, vertical_minmax=None, cb=False, dt=10):
    """Plotting mechanism for creating ONE plot. 
    i_time is the index of the time you want to plot.
    minmax parameters are tuples.
    cb controls the colourbar; can either plot the cb or the fig, not both."""
    # https://matplotlib.org/stable/gallery/mplot3d/box3d.html#sphx-glr-gallery-mplot3d-box3d-py 

    X, Y, Z = np.meshgrid(zeta.XC.to_numpy(), zeta.YC.to_numpy(), zeta.Z.to_numpy())

    timestep_str = str(zeta.isel(time=i_time)['time'].dt.seconds.to_numpy()) #mightn't work > 1 day!!!!!!!
    time_hours_str = str(float(timestep_str)*dt/60/60) # Have to do this manually 

    zeta_np = zeta.isel(time=i_time).to_numpy()
    vertical_np = vertical_data.isel(time=i_time).to_numpy()

    if zeta_minmax==None:
        zeta_min, zeta_max = zeta_np.min(), zeta_np.max()
    else:
        zeta_min, zeta_max = zeta_minmax
    if vertical_minmax==None:
        vertical_min, vertical_max = vertical_np.min(), vertical_np.max()
    else:
        vertical_min, vertical_max = vertical_minmax

    linscale, linthresh = 0.001, 0.0001
    positive_levels = np.logspace(np.log10(linthresh), np.log10(zeta_max), 51)
    negative_levels = -np.logspace(np.log10(linthresh), np.log10(-zeta_min), 51)
    levels = np.concatenate([negative_levels[::-1], [0], positive_levels]) #[0]
    kw_zeta = {
        'vmin': zeta_min,
        'vmax': zeta_max,
        'levels': levels,
        'norm': SymLogNorm(linscale=linscale, linthresh=linthresh, vmin=zeta_min, vmax=zeta_max),
    }

    if vertical_variable=='N2':
        linscale, linthresh = 0.01, 0.0002 #0.0001, 0.000002
        positive_levels = np.logspace(np.log10(linthresh), np.log10(vertical_max), 51)
        negative_levels = -np.logspace(np.log10(linthresh), np.log10(-vertical_min), 51)
        levels = np.concatenate([negative_levels[::-1], [0], positive_levels])
        kw_vertical = {
            'vmin': vertical_min,
            'vmax': vertical_max,
            'levels': levels,
            'norm': SymLogNorm(linscale=linscale, linthresh=linthresh, vmin=vertical_min, vmax=vertical_max),
        }
        cm = 'RdGy'
    elif vertical_variable=='rho':
        kw_vertical = {
            'levels': 102,
            'vmin': vertical_min,
            'vmax': vertical_max,
            'norm': 'linear'}
        cm = 'hot_r' #plt.colormaps['hot_r']
    elif vertical_variable=='t':
        kw_vertical = {
            'levels': 102,
            'vmin': vertical_min,
            'vmax': vertical_max,
            'norm': 'linear'}
        cm = 'seismic' 
    elif vertical_variable=='s':
        kw_vertical = {
            'levels': 102,
            'vmin': vertical_min,
            'vmax': vertical_max,
            'norm': 'linear'}
        cm = 'viridis' 

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    C_zeta = ax.contourf(
        X[:, :, 0], Y[:, :, 0], zeta_np[:, :, 0],
        zdir='z', offset=0, cmap='seismic', **kw_zeta
    )

    C_vertical = ax.contourf(
        vertical_np[:, -1, :-1], Y[:, -1, :-1], Z[:, -1, :-1], #the :-1's are to avoid plotting the bottom row of non-zero values
        zdir='x', offset=X.max(), cmap=cm, **kw_vertical
    )

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

    ax.xaxis._axinfo['juggled'] = (2,0,0)

    ax.set_title(run+', +'+time_hours_str + 'hrs')
    
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    
    # Save figure
    timestep_str = timestep_str.zfill(10)
    if cb == True: # Colorbar (can specify to use global mins and maxes for the whole run by providing minmaxes to this function)
        bbox_ax = ax.get_position()
        cbar_zeta_ax = fig.add_axes([1.09,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0]) #([bbox_ax.x0, 0.09, bbox_ax.x1-bbox_ax.x0, 0.02]) # for horiz bars
        cbar_zeta = plt.colorbar(C_zeta, cax=cbar_zeta_ax,shrink=0.5, pad=0.1, label='Vorticity ($s^{-1}$)')
        cbar_zeta.formatter.set_powerlimits((0, 0))
        cbar_zeta.formatter.set_useMathText(True)

        vertical_labels = {'N2': '$N^2$ ($s^{-2}$)', 'rho': r"$\rho$ ($kg$ $m^{-3}$)", 't': 'Temperature ($^\circ C$)', 's': 'Salinity ($PSU$)'}
        cbar_vertical_ax = fig.add_axes([1.35,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0])
        cbar_vertical = plt.colorbar(C_vertical, cax=cbar_vertical_ax, extend='both', shrink=0.5, pad=0.1, label=vertical_labels[vertical_variable])
        cbar_vertical.formatter.set_powerlimits((0, 0))
        cbar_vertical.formatter.set_useMathText(True)

        ax.remove()
        plt.savefig(figs_dir+'/plume_cbar_'+timestep_str+'.png',dpi=450,bbox_inches="tight")
        print(figs_dir+'/plume_cbar_'+timestep_str+'.png')
        plt.close()
    else:    
        plt.savefig(figs_dir+'/plume_'+timestep_str+'.png',dpi=450, bbox_inches="tight")    
        print(figs_dir+'/plume_'+timestep_str+'.png')
        plt.close()    

def plot_plumes(run, figs_dir, zeta, vertical_data, vertical_variable, zeta_minmax=None, vertical_minmax=None, i_time=None, dt=10):
    """Plot vorticity on the sea surface and buoyancy on the vertical face.
    Based on Vreugdenhil and Gayen 2021.
    Accepts datarrays of vorticity and buoyancy, as well as tuples of the min and max values to plot."""

    # Produces a colourbar
    i_time=10
    plume_plot_engine(run, figs_dir, zeta, vertical_data, vertical_variable, i_time, zeta_minmax, vertical_minmax, cb=True, dt=dt) #figs_dir, zeta, N2, i_time, zeta_minmax=None, N2_minmax=None, cb=False

    # For making a movie
    i_times = len(zeta.time.to_numpy())
    for i_time in range(i_times):
        plume_plot_engine(run, figs_dir, zeta, vertical_data, vertical_variable, i_time, zeta_minmax, vertical_minmax, cb=False, dt=dt) #figs_dir, zeta, N2, i_time, zeta_minmax=None, N2_minmax=None, cb=False

def avg_temp(data_dir):
    """For comparing average temps across runs."""
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

if __name__ == "__main__":

    #avg_temp('../MITgcm/so_plumes/mrb_002') # 0.2599955576563594 production run
    #avg_temp('../MITgcm/so_plumes/mrb_003') # 0.2601810829108668 side tau =86400
    #avg_temp('../MITgcm/so_plumes/mrb_004') # 0.2599837272808078 side tau =8640
    #avg_temp('../MITgcm/so_plumes/mrb_005') # 0.2614877021773202 side tau =864 *crashes
    #avg_temp('../MITgcm/so_plumes/mrb_006') # 0.2600030499680336 side tau =864000
    #avg_temp('../MITgcm/so_plumes/mrb_007') # 0.26368029388014746 full tau =86400
    #avg_temp('../MITgcm/so_plumes/mrb_008') # 0.2772914436059543 full tau =8640
    #avg_temp('../MITgcm/so_plumes/mrb_011') # 0.26413005811064794
    #quit()

    run = 'mrb_016'
    vertical_variable = 'rho'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/model_analyses/figures/figs_'+run+'_'+vertical_variable
    zeta, vertical_data = calculate_output(data_dir,vertical_variable)
    #zeta_minmax = get_mins_and_maxs(zeta,'horizontal')
    vertical_minmax = get_mins_and_maxs(vertical_data,'vertical')
    print(vertical_minmax)
    zeta_minmax = (-0.04,0.04)#(-0.0075,0.0125)
    vertical_minmax = (1003.8,1004.4)#(-0.0002,0.0002), (-2,1)
    plot_plumes(run, figs_dir, zeta, vertical_data, vertical_variable, zeta_minmax=zeta_minmax, vertical_minmax=vertical_minmax, dt=3) #vertical_minmax 
