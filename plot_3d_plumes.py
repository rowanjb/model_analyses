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

def zeta_and_out2(data_dir):
    """Returns a dataarray of vorticity and a dataarray of buoyancy."""
    # Regarding interpolating; don't forget to manually test this! (Both the interpolating and the vorticity.)
    # Uses the xgcm package for interpolating and various related calculations
    # https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
    ds = open_mdsdataset(data_dir,geometry='cartesian',prefix=['S','T','U','V'])
    grid = xgcm.Grid(ds,periodic=False) 
    zeta = (-grid.diff(ds.U * ds.dxC, 'Y') + grid.diff(ds.V * ds.dyC, 'X'))/ds.rAz
    zeta = grid.interp(zeta,['X','Y'])
    zeta = zeta.isel(XC=slice(0,51)).transpose('YC','XC','Z','time') # Slicing and interpolating for plotting reasons
    ds['rho'] = ( ('time', 'Z', 'YC', 'XC'), density.linear(ds.S, ds.T, sref=35, tref=20, sbeta=0, talpha=0.0002, rhonil=1000)) #Constants from ./input/data          
    ds = ds.assign_coords(dz=('Zl',cell_diffs(ds)))
    out2 = (-1) * (9.81) * (1/1000) * grid.diff(ds.rho, 'Z', boundary='extend')/ds['dz'] 
    out2 = grid.interp(out2,'Z')
    out2 = out2.isel(XC=slice(0,51)).transpose('YC','XC','Z','time')
    return zeta, out2 

def get_mins_and_maxs(zeta,N2):
    """Return the min and max values pertinent to the plots (i.e., zeta and N2)."""
    zeta_min = zeta.isel(Z=0).min().to_numpy()
    zeta_max = zeta.isel(Z=0).max().to_numpy()
    zeta_minmax = zeta_min, zeta_max
    N2_min = N2.isel(XC=-1,Z=slice(0,-1)).min().to_numpy()
    N2_max = N2.isel(XC=-1,Z=slice(0,-1)).max().to_numpy()
    N2_minmax = N2_min, N2_max
    return zeta_minmax, N2_minmax

def plume_plot_engine(figs_dir, zeta, N2, i_time, zeta_minmax=None, N2_minmax=None, cb=False, dt=10):
    """Plotting mechanism for creating ONE plot. 
    i_time is the index of the time you want to plot.
    minmax parameters are tuples.
    cb controls the colourbar; can either plot the cb or the fig, not both."""
    # https://matplotlib.org/stable/gallery/mplot3d/box3d.html#sphx-glr-gallery-mplot3d-box3d-py 

    X, Y, Z = np.meshgrid(N2.XC.to_numpy(), N2.YC.to_numpy(), N2.Z.to_numpy())

    timestep_str = str(zeta.isel(time=i_time)['time'].dt.seconds.to_numpy()) #mightn't work > 1 day!!!!!!!
    time_hours_str = str(float(timestep_str)*dt/60/60) # Have to do this manually 

    zeta_np = zeta.isel(time=i_time).to_numpy()
    N2_np = N2.isel(time=i_time).to_numpy()

    if zeta_minmax==None:
        zeta_min, zeta_max = zeta_np.min(), zeta_np.max()
    else:
        zeta_min, zeta_max = zeta_minmax
    if N2_minmax==None:
        N2_min, N2_max = N2_np.min(), N2_np.max()
    else:
        N2_min, N2_max = N2_minmax

    linscale, linthresh = 1, 0.0004
    positive_levels = np.logspace(np.log10(linthresh), np.log10(zeta_max), 5)
    negative_levels = -np.logspace(np.log10(linthresh), np.log10(-zeta_min), 5)
    levels = np.concatenate([negative_levels[::-1], [0], positive_levels])
    kw_zeta = {
        'vmin': zeta_min,
        'vmax': zeta_max,
        'levels': levels,
        'norm': SymLogNorm(linscale=linscale, linthresh=linthresh, vmin=zeta_min, vmax=zeta_max),
    }

    linscale, linthresh = 1, 0.00002
    positive_levels = np.logspace(np.log10(linthresh), np.log10(N2_max), 5)
    negative_levels = -np.logspace(np.log10(linthresh), np.log10(-N2_min), 5)
    levels = np.concatenate([negative_levels[::-1], [0], positive_levels])
    kw_N2 = {
        'vmin': N2_min,
        'vmax': N2_max,
        'levels': levels,
        'norm': SymLogNorm(linscale=linscale, linthresh=linthresh, vmin=N2_min, vmax=N2_max),
    }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    C_zeta = ax.contourf(
        X[:, :, 0], Y[:, :, 0], zeta_np[:, :, 0],
        zdir='z', offset=0, cmap='seismic', **kw_zeta
    )

    C_N2 = ax.contourf(
        N2_np[:, -1, :-1], Y[:, -1, :-1], Z[:, -1, :-1], #the :-1's are to avoid plotting the bottom row of non-zero values
        zdir='x', offset=X.max(), cmap='RdGy', **kw_N2
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

    ax.set_title('+'+time_hours_str + 'hrs')
    
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

        cbar_N2_ax = fig.add_axes([1.35,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0])
        cbar_N2 = plt.colorbar(C_N2, cax=cbar_N2_ax, extend='both', shrink=0.5, pad=0.1, label='$N^2$ ($s^{-2}$)')
        cbar_N2.formatter.set_powerlimits((0, 0))
        cbar_N2.formatter.set_useMathText(True)

        ax.remove()
        plt.savefig(figs_dir+'/plume_cbar_'+timestep_str+'.png',dpi=450,bbox_inches="tight")
        print(figs_dir+'/plume_cbar_'+timestep_str+'.png')
        plt.close()
    else:    
        plt.savefig(figs_dir+'/plume_'+timestep_str+'.png',dpi=450, bbox_inches="tight")    
        print(figs_dir+'/plume_'+timestep_str+'.png')
        plt.close()    

def plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None, dt=10):
    """Plot vorticity on the sea surface and buoyancy on the vertical face.
    Based on Vreugdenhil and Gayen 2021.
    Accepts datarrays of vorticity and buoyancy, as well as tuples of the min and max values to plot."""

    # Produces a colourbar
    i_time=10
    plume_plot_engine(figs_dir, zeta, N2, i_time, zeta_minmax, N2_minmax, cb=True, dt=dt) #figs_dir, zeta, N2, i_time, zeta_minmax=None, N2_minmax=None, cb=False

    # For making a movie
    i_times = len(zeta.time.to_numpy())
    for i_time in range(i_times):
        plume_plot_engine(figs_dir, zeta, N2, i_time, zeta_minmax, N2_minmax, cb=False, dt=dt) #figs_dir, zeta, N2, i_time, zeta_minmax=None, N2_minmax=None, cb=False

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

    #avg_temp('../MITgcm/so_plumes/mrb_002')
    #avg_temp('../MITgcm/so_plumes/mrb_003')
    #avg_temp('../MITgcm/so_plumes/mrb_004')
    #avg_temp('../MITgcm/so_plumes/mrb_005')
    #avg_temp('../MITgcm/so_plumes/mrb_006')
    #avg_temp('../MITgcm/so_plumes/mrb_007')
    #quit()

    run = 'mrb_002'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/model_analyses/figs_'+run
    zeta, out2 = zeta_and_out2(data_dir)
    zeta_minmax = (-0.04,0.04)#(-0.0075,0.0125)
    out2_minmax = (-0.0002,0.0002)#(3,5)#(3.045,4.8)#(-0.00015,0.00015)#(-0.0000069,0.00000585)
    plot_plumes(figs_dir, zeta, out2, zeta_minmax, out2_minmax, dt=3) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None
    
    run = 'mrb_003'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/model_analyses/figs_'+run
    zeta, out2 = zeta_and_out2(data_dir)
    zeta_minmax = (-0.04,0.04)#(-0.0075,0.0125)
    out2_minmax = (-0.0002,0.0002)#(3,5)#(3.045,4.8)#(-0.00015,0.00015)#(-0.0000069,0.00000585)
    plot_plumes(figs_dir, zeta, out2, zeta_minmax, out2_minmax, dt=3) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None
    
    run = 'mrb_005'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/model_analyses/figs_'+run
    zeta, out2 = zeta_and_out2(data_dir)
    zeta_minmax = (-0.04,0.04)#(-0.0075,0.0125)
    out2_minmax = (-0.0002,0.0002)#(3,5)#(3.045,4.8)#(-0.00015,0.00015)#(-0.0000069,0.00000585)
    plot_plumes(figs_dir, zeta, out2, zeta_minmax, out2_minmax, dt=3) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    run = 'mrb_006'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/model_analyses/figs_'+run
    zeta, out2 = zeta_and_out2(data_dir)
    zeta_minmax = (-0.04,0.04)#(-0.0075,0.0125)
    out2_minmax = (-0.0002,0.0002)#(3,5)#(3.045,4.8)#(-0.00015,0.00015)#(-0.0000069,0.00000585)
    plot_plumes(figs_dir, zeta, out2, zeta_minmax, out2_minmax, dt=3) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None
    
    run = 'mrb_007'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/model_analyses/figs_'+run
    zeta, out2 = zeta_and_out2(data_dir)
    zeta_minmax = (-0.04,0.04)#(-0.0075,0.0125)
    out2_minmax = (-0.0002,0.0002)#(3,5)#(3.045,4.8)#(-0.00015,0.00015)#(-0.0000069,0.00000585)
    plot_plumes(figs_dir, zeta, out2, zeta_minmax, out2_minmax, dt=3) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    quit()

    # Refers to my first "production" run
    run = 'mrb_001'
    data_dir = '/dss/dsshome1/0B/ra85duq/MITgcm/so_plumes/'+run
    figs_dir = '/dss/dsshome1/0B/ra85duq/model_analyses/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    print(zeta_minmax)
    print(N2_minmax)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    # For creating multiple moves of comparable runs 

    #For consistency across multiple runs
    zeta_minmax = (-0.002,0.18)#(-0.0075,0.0125)
    N2_minmax = (-0.000005,0.00000225)#(-0.0000069,0.00000585)

    run = 'run_parallel'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/MITgcm/so_plumes/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    #zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    run = 'run_40m'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/MITgcm/so_plumes/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    #zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    run = 'run_80m'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/MITgcm/so_plumes/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    #zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    run = 'run_160m'
    data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    figs_dir = '/albedo/home/robrow001/MITgcm/so_plumes/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    #zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None
    
