# Copied from: https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html 

import numpy as np
import pandas as pd
import xmitgcm
import matplotlib.pylab as plt
import xmitgcm.file_utils
import xmitgcm.utils
import xarray as xr

def relax_mask(example_bin, shape, side_cells, side_max_M, bottom_cells, bottom_max_M, bin_name, scaling='exp'):
    """Creates a relax mask (M_rbc) based on an example input binary and its shape.
    The input binary can be anything 3D such as U, V, T, etc.
    The shape should be its dimension, e.g., (50,100,100).
    side_cells and bottom_cells refer to the number of non-zero cells along the boundary. 
    side_max_M and bottom_max_M refer to the values of the outermost cells (between 0 and 1). 
    With scaling=exp, the magnitude of the cells is halved every dx, e.g., 
    side_cells=2, side_max_M=1  =>  1,  0.5,     0,      0, 0, 0
    side_cells=4, side_max_M=1  =>  1, 0.25, 0.125, 0.0625, 0, 0 etc.
    With scaling=linear, the magnitude of the cells grows linearl, e.g.,
    side_cells=2, side_max_M=1  =>  1,  0.5,     0,      0, 0, 0
    side_cells=4, side_max_M=1  =>  1, 0.75,   0.5,   0.25, 0, 0 etc."""

    # Simple check
    if side_max_M>1 or bottom_max_M>1 or side_max_M<0 or bottom_max_M<0:
        print('The side or bottom max value is out of bounds.')
        return
    
    # Creating lists of the cell values
    side_values = np.full(side_cells,side_max_M) # np.full creates array of shape side_cells filled with value side_max_M
    bottom_values = np.full(bottom_cells,bottom_max_M)
    if scaling=='exp':
        side_values = [i/2**n for n,i in enumerate(side_values)]
        bottom_values = [i/2**n for n,i in enumerate(bottom_values)]
    if scaling=='linear':
        d_side, d_bottom = side_max_M/side_cells, bottom_max_M/bottom_cells
        side_values = [i-n*d_side for n,i in enumerate(side_values)]
        bottom_values = [i-n*d_bottom for n,i in enumerate(bottom_values)]

    # Opening an example field
    M = xmitgcm.utils.read_raw_data(example_bin, shape=shape, dtype=np.dtype('>f4') )
    M[:] = 0 # Interior M_rbc cells should be 0

    # Starting with the side cells; traverse in reverse to easily handle overlaps in the corners
    # **Maybe add some kind of gradient at the surface so that you don't get 1s at the outer edge of the surface (for example)
    for n,i in reversed(list(enumerate(side_values))):
        M[:,n,:], M[:,(-1)*n-1,:], M[:,:,n], M[:,:,(-1)*n-1] = i, i, i, i

    # Now do the bottom; in overlaps, defer to the higher value
    for n,i in reversed(list(enumerate(bottom_values))):
        M[(-1)*n-1,:,:] = np.where(M[(-1)*n-1,:,:]<i,i,M[(-1)*n-1,:,:])

    # Save the binary
    # For flattening: either np.moveaxis(M, [0, 1, 2], [-1, -2, -3]).flatten(order='F') or M.flatten(order='C')
    xmitgcm.utils.write_to_binary(np.moveaxis(M, [0, 1, 2], [-1, -2, -3]).flatten(order='F'), '../MITgcm/so_plumes/binaries/'+bin_name)

    # (Testing!) -- needs to be set manually
    M = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/'+bin_name, shape=shape, dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, M[:,:,50])
    cbar = fig.colorbar(cs)
    plt.savefig('relax_mask.png')

if __name__ == "__main__":
    example_bin = '../MITgcm/so_plumes/binaries/V.motionless.50x150x150.bin'
    shape = (50,150,150)
    side_cells = 10
    side_max_M = 1
    bottom_cells = 5 
    bottom_max_M = 1
    bin_name = 'relax_mask_linear_10x10x10x10x5.50x150x150.bin'
    scaling='linear'
    relax_mask(example_bin, shape, side_cells, side_max_M, bottom_cells, bottom_max_M, bin_name, scaling)
