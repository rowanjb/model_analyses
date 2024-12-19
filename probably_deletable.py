#https://matplotlib.org/stable/gallery/mplot3d/box3d.html#sphx-glr-gallery-mplot3d-box3d-py
#https://matplotlib.org/stable/gallery/mplot3d/intersecting_planes.html 

from xmitgcm import open_mdsdataset 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import xarray as xr
import numpy as np

def plot_quadrants(ax, array, fixed_coord, cmap):
    """For a given 3d *array* plot a plane with *fixed_coord*, using four quadrants."""
    
    nx, ny, nz = array.shape
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane_data = array[index]

    n0, n1 = plane_data.shape
    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]
    
    min_val = array.min()
    max_val = array.max()

    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))
        if fixed_coord == 'x':
            Y, Z = np.mgrid[0:ny // 2, 0:nz // 2]
            X = nx // 2 * np.ones_like(Y)
            Y_offset = (i // 2) * ny // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'y':
            X, Z = np.mgrid[0:nx // 2, 0:nz // 2]
            Y = ny // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'z':
            X, Y = np.mgrid[0:nx // 2, 0:ny // 2]
            Z = nz // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Y_offset = (i % 2) * ny // 2
            ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)

def figure_3D_array_slices(array, cmap=None):
    """Plot a 3d array using three intersecting centered planes."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(array.shape)
    plot_quadrants(ax, array, 'x', cmap=cmap)
    plot_quadrants(ax, array, 'y', cmap=cmap)
    plot_quadrants(ax, array, 'z', cmap=cmap)
    return fig, ax

data_dir = './run_serial'#global_oce_latlon'
ds = open_mdsdataset(data_dir,geometry='cartesian',prefix=['T'])
ds['T'] = ds.T.transpose('time','XC','YC','Z')
T = ds.T.isel(time=5,XC=slice(0,49)).to_numpy()
T = np.flip(T,axis=2)

figure_3D_array_slices(T, cmap='viridis')
plt.savefig('test_fig.png')
quit()









ds['T'] = xr.where(ds.T<0.98,ds.Z,0)
ds['Depth ($m$)'] = ds.T.min(dim='Z')
for t in range(0,len(ds.time.to_numpy())):
    dst = ds.isel(time=t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    dst['Depth ($m$)'].plot.surface(cmap='viridis',add_colorbar=False)
    hours = dst.time.dt.seconds.to_numpy()/360
    plt.title('T < 0.999$\degree$C - ' + str(hours) + ' hrs')
    plt.xlabel('$m$')
    plt.ylabel('$m$')
    ax.set_zlim(-1000, 0)
    plt.savefig('plume_figs_1c/plume_'+str(t)+'.png', dpi=900, bbox_inches="tight", pad_inches = 0.5)
    plt.clf()
quit()

#t=20 

#fig = plt.figure(figsize=(5, 4))
#ax = fig.add_subplot(111, projection='3d')

#X, Y, Z = ds.XC.to_numpy(), ds.YC.to_numpy(), ds.Z.to_numpy()
#ds['T'] = ds.T.transpose('time','XC','YC','Z')

# Plot contour surfaces
#_ = ax.contourf(                                           
#    X, Y, ds.T.isel(time=t,Z=0).to_numpy(),
#    zdir='z', offset=0 #offset should really be -10
#)

## Set zoom and angle view
#ax.view_init(40, -30, 0)
#ax.set_box_aspect(None, zoom=0.9)

# Show Figure
#fig.savefig('testFig.png', dpi=900, bbox_inches="tight", pad_inches = 0.5)
#fig.clf()

#quit()

# Define dimensions
Nx, Ny, Nz = 100, 300, 500
X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))

# Create fake data
data = (((X+100)**2 + (Y-20)**2 + 2*Z)/1000+1)

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.linspace(data.min(), data.max(), 10),
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot contour surfaces
_ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], data[:, :, 0],
    zdir='z', offset=0, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=0, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), **kw
)
# --


# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Set labels and zticks
ax.set(
    xlabel='X [km]',
    ylabel='Y [km]',
    zlabel='Z [m]',
    zticks=[0, -150, -300, -450],
)

# Show Figure
plt.savefig('testFig.png', dpi=900, bbox_inches="tight", pad_inches = 0.5)
plt.clf()

quit()




















t = 20

## Define dimensions
#Nx, Ny, Nz = 100, 300, 500
#X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))
#
## Create fake data
#data = (((X+100)**2 + (Y-20)**2 + 2*Z)/1000+1)

# Define dimensions
X, Y, Z = ds.XC.to_numpy(), ds.YC.to_numpy(), ds.Z.to_numpy()

# Keywords
data = ds.T.isel(time=t).to_numpy()
kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.linspace(data.min(), data.max(), 10),
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

ds['T'] = ds.T.transpose('time','XC','YC','Z')

# Plot contour surfaces
_ = ax.contourf(                                           
    X, Y, ds.T.isel(time=t,Z=0).to_numpy(),
    zdir='z', offset=0, **kw #offset should really be -10
)
_ = ax.contourf(
    X, ds.T.isel(time=t,YC=0).to_numpy(), Z,
    zdir='y', offset=0, **kw #offset should really be 10
)
C = ax.contourf(
    ds.T.isel(time=t,XC=0).to_numpy(), Y, Z,
    zdir='x', offset=X.max(), **kw
)
# --

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Set labels and zticks
ax.set(
    xlabel='X [km]',
    ylabel='Y [km]',
    zlabel='Z [m]'#,
    #zticks=[0, -150, -300, -450],
)

# Set zoom and angle view
ax.view_init(40, -30, 0)
ax.set_box_aspect(None, zoom=0.9)

# Colorbar
fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')

# Show Figure
plt.savefig('parallel_t3602', dpi=900, bbox_inches="tight")
plt.clf()


#for i in list(ds.coords):
#    print(i)
#    print(ds[i].attrs)
#    print(ds[i])
#quit()

##times = ds.time.to_numpy()
#T = ds.T.isel(time=2,YC=50)
#T.plot.contourf(
#    x='XC',
#    y='Z',
#    cmap='viridis')
#plt.title('Cross section 1')
#plt.xlabel('Horizontal (cells)')
#plt.ylabel('Depth (cells)')
#plt.savefig('parallel_t3602', dpi=900, bbox_inches="tight")
#plt.clf()
#quit()

