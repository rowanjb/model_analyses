# Copied from: https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html 

import numpy as np
import pandas as pd
import xmitgcm
import matplotlib.pylab as plt
import xmitgcm.file_utils
import xmitgcm.utils
import xarray as xr

import sys
sys.path.insert(1, '../obs_analyses/')
import mooring_analyses
import woa_analyses

def temp_from_woa():
    dst,dss = woa_analyses.get_woa('autumn',2015)
    dy = np.array([
         0.4, 1.2, 2.0, 2.8, 3.6, 4.4, 5.2, 6.0, 6.8, 7.6,
         8.4, 9.2,10.0,10.8,11.6,12.4,13.2,14.0,14.8,15.6,
        16.4,17.2,18.0,18.8,19.6,20.4,21.2,22.0,22.8,23.6,
        24.4,25.2,26.0,26.8,27.6,28.4,29.2,30.0,30.8,31.6,
        32.4,33.2,34.0,34.8,35.6,36.4,37.2,38.0,38.8,39.6])
    y = np.zeros(len(dy))
    for i,n in enumerate(dy):
        if i==0: y[i] = n/2
        else: y[i] = np.sum(dy[:i]) + n/2
    T = dst['t_an'].isel(time=0).sel({'lat': -69, 'lon': -27},method='nearest').interp(depth=y)
    S = dss['s_an'].isel(time=0).sel({'lat': -69, 'lon': -27},method='nearest').interp(depth=y)
    pseudo = np.tile(T,(100,100,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), 'T.WOA.bin')
    pseudo = np.tile(S,(100,100,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), 'S.WOA.bin')
    
    #testing
    T2 = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, T2[:,49,:])
    cbar = fig.colorbar(cs)
    plt.savefig('T.WOA.png')
    print(T2[0,:,:])
    print(T2[:,20,20])
    print(T2[:,80,80])

def new_Q_surf():
    Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_p32.bin', shape=(100,100), dtype=np.dtype('>f4') ) 
    newQ = Q/6
    xmitgcm.utils.write_to_binary(newQ.flatten(order='F'), '../MITgcm/so_plumes/binaries/Qnet_WOA.bin')

    Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_WOA.bin', shape=(100,100), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 99, 100)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Q)
    cbar = fig.colorbar(cs)
    plt.savefig('Qnet_WOA.png')

def new_Eta():
    Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Eta.120mn.bin', shape=(100,100), dtype=np.dtype('>f4') )
    Eta[Eta != np.isnan] = 0
    xmitgcm.utils.write_to_binary(Eta.flatten(order='F'), '../MITgcm/so_plumes/binaries/Eta.flat.bin')

    Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Eta.flat.bin', shape=(100,100), dtype=np.dtype('>f4') )
    #testing
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 99, 100)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Eta)
    cbar = fig.colorbar(cs)
    plt.savefig('Eta.flat.png')

def new_U():
    U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/U.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    U[U != np.isnan] = 0 
    xmitgcm.utils.write_to_binary(U.flatten(order='F'), '../MITgcm/so_plumes/binaries/U.motionless.bin') #might need to be "C" if not all 0s

    U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/U.motionless.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    #testing
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, U[:,49,:])
    cbar = fig.colorbar(cs)
    plt.savefig('U.motionless.png')
    print(U[0,:,:])
    print(U[:,20,20])
    print(U[:,80,80])

def new_V():
    V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/V.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    V[V != np.isnan] = 0
    xmitgcm.utils.write_to_binary(V.flatten(order='F'), '../MITgcm/so_plumes/binaries/V.motionless.bin') #might need to be "C" if not all 0s

    V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/V.motionless.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    #testing
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, V[:,49,:])
    cbar = fig.colorbar(cs)
    plt.savefig('V.motionless.png')
    print(V[0,:,:])
    print(V[:,20,20])
    print(V[:,80,80])

def relax_mask():
    # opening any old file simply to get the right array shape
    M = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/V.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    M[:] = 0 # interior cells should be 0
    M[:,1,:] = 0.25 # second to outer cells are 0.25---I copied this from the reentrant channel ex., but I assume it's to avoid numerical problems
    M[:,98,:] = 0.25
    M[:,:,1] = 0.25
    M[:,:,98] = 0.25
    M[48,:,:] = 0.25
    M[:,0,:] = 1 # outer cells are 1---note they need to be spec'd after the 0.25 cells to avoid make some outter cells 0.25 accidentally
    M[:,99,:] = 1
    M[:,:,0] = 1
    M[:,:,99] = 1
    M[49,:,:] = 1
    # either np.moveaxis(M, [0, 1, 2], [-1, -2, -3]).flatten(order='F') or M.flatten(order='C')
    xmitgcm.utils.write_to_binary(np.moveaxis(M, [0, 1, 2], [-1, -2, -3]).flatten(order='F'), '../MITgcm/so_plumes/binaries/relax_mask_incl_bottom.bin')

    #testing
    M = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/relax_mask_incl_bottom.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, M[:,:,25])
    cbar = fig.colorbar(cs)
    plt.savefig('relax_mask.png')

def basic_example():
    lon = np.arange(0,100,1)
    lat = np.arange(0,100,1)
    quit()
    depth = np.arange(0,50,1)
    shape,_,_ = np.meshgrid(depth,lon,lat)
    pseudo = shape*0+1
    xmitgcm.utils.write_to_binary(pseudo.flatten(), 'T.1c_new.bin', dtype=np.dtype('>f4') )

if __name__ == "__main__":
    #temp_from_woa()
    #new_Q_surf()
    #new_Eta()
    #new_U()
    #new_V()
    relax_mask()
