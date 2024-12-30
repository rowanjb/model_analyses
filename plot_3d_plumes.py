# For making movies of convective plumes in 3D

import matplotlib.pyplot as plt
import numpy as np
from xmitgcm import open_mdsdataset 
from MITgcmutils import density
import xgcm
import os

def zeta_and_N2(data_dir):
    """Returns a dataarray of vorticity and a dataarray of buoyancy."""
    # Regarding interpolating; don't forget to manually test this! (Both the interpolating and the vorticity.)
    # Uses the xgcm package for interpolating and various related calculations
    # https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
    ds = open_mdsdataset(data_dir,geometry='cartesian',prefix=['S','T','U','V'])
    grid = xgcm.Grid(ds,periodic=False) 
    zeta = (-grid.diff(ds.U * ds.dxC, 'Y') + grid.diff(ds.V * ds.dyC, 'X'))/ds.rAz
    zeta = grid.interp(zeta,['X','Y'])
    ds['rho'] = ( ('time', 'Z', 'YC', 'XC'), density.linear(ds.S, ds.T, sref=35, tref=20, sbeta=0, talpha=0.0002, rhonil=1000)) #Constants from ./input/data       
    N2 = (-1) * (9.81) * (1/1000) * grid.diff(ds.rho, 'Z')/20 #I'd rather use something like drC or drF than Z but c'est la vie
    N2 = grid.interp(N2,'Z')
    zeta = zeta.isel(XC=slice(0,51)).transpose('YC','XC','Z','time') # Slicing and interpolating for plotting reasons
    N2 = N2.isel(XC=slice(0,51)).transpose('YC','XC','Z','time')
    return zeta, N2 

def get_mins_and_maxs(zeta,N2):
    """Return the min and max values pertinent to the plots (i.e., zeta and N2)."""
    zeta_min = zeta.isel(Z=0).min().to_numpy()
    zeta_max = zeta.isel(Z=0).max().to_numpy()
    zeta_minmax = zeta_min, zeta_max
    N2_min = N2.isel(XC=-1,Z=slice(0,-1)).min().to_numpy()
    N2_max = N2.isel(XC=-1,Z=slice(0,-1)).max().to_numpy()
    N2_minmax = N2_min, N2_max
    return zeta_minmax, N2_minmax

def plume_plot_engine(figs_dir, zeta, N2, i_time, zeta_minmax=None, N2_minmax=None, cb=False):
    """Plotting mechanism for creating ONE plot. 
    i_time is the index of the time you want to plot.
    minmax parameters are tuples.
    cb controls the colourbar; can either plot the cb or the fig, not both."""
    # https://matplotlib.org/stable/gallery/mplot3d/box3d.html#sphx-glr-gallery-mplot3d-box3d-py 

    X, Y, Z = np.meshgrid(N2.XC.to_numpy(), N2.YC.to_numpy(), N2.Z.to_numpy())

    time_dseconds_str = str(zeta.isel(time=i_time)['time'].dt.seconds.to_numpy()) #mightn't work > 1 day!!!!!!!
    time_hours_str = str(float(time_dseconds_str)/60/6)

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

    kw_zeta = {
        'vmin': zeta_min,
        'vmax': zeta_max,
        'levels': np.linspace(zeta_min, zeta_max, 500),
    }

    kw_N2 = {
        'vmin': N2_min,
        'vmax': N2_max,
        'levels': np.linspace(N2_min, N2_max, 500),
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
    # --

    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    ## Set labels and zticks
    #ax.set(
    #    xlabel='X ($m$)',
    #    ylabel='Y ($m$)',
    #    zlabel='Z ($m$)'#,
    #    #zticks=[0, -150, -300, -450],
    #)

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
    time_dseconds_str = time_dseconds_str.zfill(10)
    if cb == True: # Colorbar (can specify to use global mins and maxes for the whole run by providing minmaxes to this function)
        bbox_ax = ax.get_position()
        cbar_zeta_ax = fig.add_axes([1.09,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0]) #([bbox_ax.x0, 0.09, bbox_ax.x1-bbox_ax.x0, 0.02]) # for horiz bars
        cbar_zeta = plt.colorbar(C_zeta, cax=cbar_zeta_ax,shrink=0.5, pad=0.1, label='Vorticity ($s^{-1}$)')
        cbar_zeta.formatter.set_powerlimits((0, 0))
        cbar_zeta.formatter.set_useMathText(True)

        cbar_N2_ax = fig.add_axes([1.35,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0])
        cbar_N2 = plt.colorbar(C_N2, cax=cbar_N2_ax,shrink=0.5, pad=0.1, label='$N^2$ ($s^{-2}$)')
        cbar_N2.formatter.set_powerlimits((0, 0))
        cbar_N2.formatter.set_useMathText(True)

        ax.remove()
        plt.savefig(figs_dir+'/plume_cbar_'+time_dseconds_str+'.png',dpi=900,bbox_inches="tight")
        plt.close()
    else:    
        plt.savefig(figs_dir+'/plume_'+time_dseconds_str+'.png',dpi=900, bbox_inches="tight")    
        plt.close()    

def plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None):
    """Plot vorticity on the sea surface and buoyancy on the vertical face.
    Based on Vreugdenhil and Gayen 2021.
    Accepts datarrays of vorticity and buoyancy, as well as tuples of the min and max values to plot."""
    
    # For making a movie
    i_times = len(zeta.time.to_numpy())
    for i_time in range(i_times):
        plume_plot_engine(figs_dir, zeta, N2, i_time, zeta_minmax, N2_minmax, cb=False) #figs_dir, zeta, N2, i_time, zeta_minmax=None, N2_minmax=None, cb=False

    # Produces a colourbar
    i_time=10
    plume_plot_engine(figs_dir, zeta, N2, i_time, zeta_minmax, N2_minmax, cb=True) #figs_dir, zeta, N2, i_time, zeta_minmax=None, N2_minmax=None, cb=False

if __name__ == "__main__":
    
    #For consistency across multiple runs
    zeta_minmax = (-0.002,0.18)#(-0.0075,0.0125)
    N2_minmax = (-0.000005,0.00000225)#(-0.0000069,0.00000585)

    run = 'mrb_001'
    data_dir = '/dss/dsshome1/0B/ra85duq/MITgcm/so_plumes/'+run
    figs_dir = '/dss/dsshome1/0B/ra85duq/model_analyses/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    print(zeta_minmax)
    print(N2_minmax)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    quit()

    run = 'run_parallel'
    data_dir = '/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/'+run
    figs_dir = '/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    #zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    run = 'run_40m'
    data_dir = '/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/'+run
    figs_dir = '/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    #zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    run = 'run_80m'
    data_dir = '/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/'+run
    figs_dir = '/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    #zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None

    run = 'run_160m'
    data_dir = '/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/'+run
    figs_dir = '/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/figs_'+run
    zeta, N2 = zeta_and_N2(data_dir)
    #zeta_minmax, N2_minmax = get_mins_and_maxs(zeta, N2)
    plot_plumes(figs_dir, zeta, N2, zeta_minmax, N2_minmax) #figs_dir, zeta, N2, zeta_minmax, N2_minmax, i_time=None
