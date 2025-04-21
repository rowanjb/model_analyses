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

def plume_plot_engine(ds, run, vertical_plane_variable, horizontal_plane_variable, dt=10):
    """Plotting mechanism for creating ONE 3D plot.
    Reduced complexity compared to V1."""
    # https://matplotlib.org/stable/gallery/mplot3d/box3d.html#sphx-glr-gallery-mplot3d-box3d-py 

    # Essentially creating a grid (would be cool to try open in paraview one day)
    Z, Y, X = np.meshgrid(ds['Z'].values, ds['YC'].values, ds['XC'].values, indexing='ij')

    # Getting the time information
    timestep_str = str(ds['time'].dt.seconds.to_numpy()[0]) #mightn't work > 1 day!!!!!!!
    time_hours_str = str(float(timestep_str)*dt/60/60) # Have to do this manually 

    # Identifying the important grid faces
    horizontal_np = ds[horizontal_plane_variable].isel(Z=0).to_numpy()
    try: vertical_np = ds[vertical_plane_variable].isel(XC=-1).to_numpy() 
    except: vertical_np = ds[vertical_plane_variable].isel(XG=-1).to_numpy()

    # Clipping the data... not great scientifically but I THINK it relates to numerical issues not results
    if vertical_plane_variable=='T':
        vertical_np = np.where(vertical_np>-2,vertical_np,-2)
        vertical_np = np.where(vertical_np<2,vertical_np,2)
    if horizontal_plane_variable=='T':
        horizontal_np = np.where(horizontal_np>-2,horizontal_np,-2)
        horizontal_np = np.where(horizontal_np<2,horizontal_np,2)

    # Specifying the scaling of the coutourf parameters in the horz plane
    if horizontal_plane_variable=='zeta':
        linscale, linthresh = 0.001, 0.0001
        vmin, vmax = -0.04, 0.04
        positive_levels = np.logspace(np.log10(linthresh), np.log10(vmax), 51)
        negative_levels = -np.logspace(np.log10(linthresh), np.log10(-vmin), 51) #np.log10 can't be neg
        levels = np.concatenate([negative_levels[::-1], [0], positive_levels]) #[0]
        kw_horizontal = {'levels': levels, 'cmap':'PiYG', 'norm': SymLogNorm(linscale=linscale, linthresh=linthresh, vmin=vmin, vmax=vmax)}
    elif horizontal_plane_variable=='T':
        kw_horizontal = {'levels': 102, 'norm': 'linear', 'vmin':-2, 'vmax':2, 'cmap':'seismic'} 
    elif horizontal_plane_variable=='S':
        kw_horizontal = {'levels': 102, 'norm': 'linear', 'vmin':33.8, 'vmax':35.1, 'cmap':'viridis'} 

    # Specifying the scaling of the coutourf parameters in the vert plane
    if vertical_plane_variable=='T': 
        kw_vertical = {'levels': 102, 'norm': 'linear', 'vmin':-2, 'vmax':2} 
        cm = 'seismic' 
    elif vertical_plane_variable=='S':
        kw_vertical = {'levels': 102, 'norm': 'linear', 'vmin':33.97, 'vmax':35}
        cm = 'viridis' 
    elif vertical_plane_variable=='N2':
        linscale, linthresh = 0.0001, 0.00001
        vmin, vmax = -0.0025, 0.0025
        positive_levels = np.logspace(np.log10(linthresh), np.log10(vmax), 51)
        negative_levels = -np.logspace(np.log10(linthresh), np.log10(-vmin), 51) #np.log10 can't be neg
        levels = np.concatenate([negative_levels[::-1], [0], positive_levels]) #[0]
        kw_vertical = {'levels': levels, 'norm': SymLogNorm(linscale=linscale, linthresh=linthresh, vmin=vmin, vmax=vmax)}
        cm='RdGy_r'

    # Create a figure with 3D ax
    plt.rcParams["font.family"] = "serif" # Change the base font
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='3d')

    # Plot horizontal surface
    C_horizontal = ax.contourf(
        X[0, :, :], Y[0, :, :], horizontal_np[0, :, :], # The "0" in the horz_np array is the time dim
        zdir='z', offset=0, **kw_horizontal
    )

    # Plot vertical surface
    C_vertical = ax.contourf(
        # The "0" in the vert_np array is the time dim
        vertical_np[0, :, :], Y[:, :, -1], Z[:, :, -1], #the :-1's are to avoid plotting the bottom row of non-zero values
        zdir='x', offset=X.max(), cmap=cm, **kw_vertical
    )

    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Set zoom and angle view
    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=9)

    # Scale X and Y axes 
    ax.set_aspect('equalxy')

    # Reference: https://stackoverflow.com/questions/44001613/matplotlib-3d-surface-plot-turn-off-background-but-keep-axes
    # Make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    # Can't remember why we do this, but CAN remember that it was tedious to figure out
    ax.xaxis._axinfo['juggled'] = (2,0,0)

    # Set the title according to the timestamp
    ax.set_title('+'+time_hours_str + ' hrs',fontsize=11) #run+', 

    # Set labels etc
    ax.set_xlabel('X ($m$)',fontsize=9,labelpad=-2)
    ax.tick_params(axis='x', which='major', pad=1.5, labelsize=9)
    plt.locator_params(axis = 'x', nbins = 5)
    ax.set_ylabel('Y ($m$)',fontsize=9)
    ax.tick_params(axis='y', which='major', pad=-1, labelsize=9)
    ax.set_zlabel('Depth ($m$)',fontsize=9)
    ax.tick_params(axis='z', which='major', labelsize=9)
    
    # Colourbar stuff
    bbox_ax = ax.get_position()

    horizontal_labels = {'zeta': 'Vorticity ($s^{-1}$)', 'T': 'Temperature ($℃$)', 'S': 'Salinity ($PSU$)'}
    cbar_horizontal_ax = fig.add_axes([1.15,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0]) # (left, bottom, width, height)
    cbar_horizontal = plt.colorbar(C_horizontal, cax=cbar_horizontal_ax,shrink=0.5, pad=0.1)
    #cbar_horizontal.formatter.set_powerlimits((0, 0))
    #cbar_horizontal.formatter.set_useMathText(True)
    cbar_horizontal.ax.tick_params(labelsize=9)
    cbar_horizontal.set_label(label=horizontal_labels[horizontal_plane_variable], size=9) 
    cbar_horizontal.update_ticks()

    vertical_labels = {'N2': '$N^2$ ($s^{-2}$)', 'pot_rho': r"$\rho$ ($kg$ $m^{-3}$)", 'T': 'Temperature ($℃$)', 'S': 'Salinity ($PSU$)', 'quiver': 'Velocity ($m$ $s^{-1}$)'}
    cbar_vertical_ax = fig.add_axes([1.4,bbox_ax.y0, 0.025, bbox_ax.y1-bbox_ax.y0])
    cbar_vertical = plt.colorbar(C_vertical, cax=cbar_vertical_ax, extend='both', shrink=0.5, pad=0.1)
    cbar_vertical.ax.tick_params(labelsize=9)
    cbar_vertical.set_label(label=vertical_labels[vertical_plane_variable], size=9) 
    #cbar_vertical.formatter.set_powerlimits((0, 0))
    #cbar_vertical.formatter.set_useMathText(True)
    cbar_vertical.update_ticks()

    # Save figure
    timestep_str = timestep_str.zfill(10)

    filepath = './figures/figs_3D/'+run+'_'+timestep_str+'_'+vertical_plane_variable+'_'+horizontal_plane_variable+'_4x4.png'
    #plt.savefig(filepath,dpi=450,bbox_inches="tight")
    plt.savefig(filepath,dpi=1200,bbox_inches="tight")
    print(filepath+' saved')
    plt.close()

def run_plume_plot(run,vertical_plane_variable='T',horizontal_plane_variable='zeta',i_time=10):
    
    # Some necessary filepaths
    #data_dir = '/albedo/home/robrow001/MITgcm/so_plumes/'+run
    data_dir = '/albedo/work/projects/p_so-clim/GCM_data/RowanMITgcm/'+run
    
    # Opening the data
    ds = bma.open_mitgcm_output_all_vars(data_dir).isel(time=[i_time]) # Square brackets preserve dim

    # Adding any necessary variables (I'll keep expanding this with new variables later)
    if horizontal_plane_variable=='zeta':
        ds = bma.calculate_zeta(ds) 
        print("Constructing plot with zeta in the horizontal plane")
    elif horizontal_plane_variable=='T' or horizontal_plane_variable=='S':
        print("Constructing plot with "+horizontal_plane_variable+" in the horizontal plane")
    else: 
        print("Functionality for horiz-plane variable not added... yet"); quit()

    if vertical_plane_variable=='T' or vertical_plane_variable=='S': 
        print("Constructing plot with "+vertical_plane_variable+" in the vertical plane")
    elif vertical_plane_variable=='quiver':
        print("Quiver functionality not yet added"); quit()
    elif vertical_plane_variable=='N2':
        ds = bma.calculate_N2_linear_EOS(ds)
        print("Constructing plot with N2 in the vertical plane")
    elif vertical_plane_variable=='pot_rho':
        print("Potential density functionality not yet added"); quit()
    else: 
        print("Functionality for vert-plane variable not added... yet"); quit()

    # Shrink the dataset by dividing into sections and revealing the faces that you want to plot
    ds = chop_ds(ds)
    
    # Plotting    
    # Based on Vreugdenhil and Gayen 2021
    plume_plot_engine(ds, run, vertical_plane_variable, horizontal_plane_variable, dt=3) 

if __name__ == "__main__":
   
    run, vertical_plane_variable, horizontal_plane_variable, i_time = 'mrb_038', 'T', 'T', 480 
    run_plume_plot(run, vertical_plane_variable, horizontal_plane_variable, i_time)