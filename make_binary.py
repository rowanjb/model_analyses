# Based on: https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html 
# These are some basic functions that I use to make binaries

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

def from_woa():
    dst = woa_analyses.get_woa_Weddell_mooring('autumn',2015,'t')
    dss = woa_analyses.get_woa_Weddell_mooring('autumn',2015,'s')
    print(dst)
    print(dss)
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
    T = dst.isel(time=0).interp(depth=y)
    S = dss.isel(time=0).interp(depth=y)
    pseudo = np.tile(T,(150,150,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/T.WOA.150x150.bin')
    pseudo = np.tile(S,(150,150,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/S.WOA.150x150.bin')
    
    #testing
    T2 = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, T2[:,49,:])
    cbar = fig.colorbar(cs)
    plt.savefig('T.WOA.150x150.png')

    S2 = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, S2[:,49,:])
    cbar = fig.colorbar(cs)
    plt.savefig('S.WOA.150x150.png')

def new_Q_surf():
    Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_p32.bin', shape=(100,100), dtype=np.dtype('>f4') ) 
    newQ = np.zeros((150,150))
    newQ[25:125,25:125] = Q/12
    xmitgcm.utils.write_to_binary(newQ.flatten(order='F'), '../MITgcm/so_plumes/binaries/Qnet_WOA2.150x150.bin')

    Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_WOA2.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)
    Y = np.linspace(0, 149, 150)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, Q) #contourf
    cbar = fig.colorbar(cs)
    plt.savefig('Qnet_WOA2.150x150.bin.png')

def new_Eta():
    #Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Eta.120mn.bin', shape=(100,100), dtype=np.dtype('>f4') )
    Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_WOA2.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') )
    Eta[Eta != np.isnan] = 0
    xmitgcm.utils.write_to_binary(Eta.flatten(order='F'), '../MITgcm/so_plumes/binaries/Eta.flat.150x150.bin')

    Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Eta.flat.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') )
    #testing
    X = np.linspace(0, 149, 150)
    Y = np.linspace(0, 149, 150)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, Eta)
    cbar = fig.colorbar(cs)
    plt.savefig('Eta.flat.150x150.png')

def new_U():
    #U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/U.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    U[U != np.isnan] = 0 
    xmitgcm.utils.write_to_binary(U.flatten(order='F'), '../MITgcm/so_plumes/binaries/U.motionless.150x150.bin') #might need to be "C" if not all 0s

    # testing
    U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/U.motionless.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, U[:,49,:])
    cbar = fig.colorbar(cs)
    plt.savefig('U.motionless.150x150.png')
    print(U[0,:,:])
    print(U[:,20,20])
    print(U[:,80,80])

def new_V():
    #V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/V.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    V[V != np.isnan] = 0
    xmitgcm.utils.write_to_binary(V.flatten(order='F'), '../MITgcm/so_plumes/binaries/V.motionless.150x150.bin') #might need to be "C" if not all 0s

    # testing
    V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/V.motionless.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, V[:,49,:])
    cbar = fig.colorbar(cs)
    plt.savefig('V.motionless.150x150.png')
    print(V[0,:,:])
    print(V[:,20,20])
    print(V[:,80,80])

def basic_example():
    lon = np.arange(0,100,1)
    lat = np.arange(0,100,1)
    quit()
    depth = np.arange(0,50,1)
    shape,_,_ = np.meshgrid(depth,lon,lat)
    pseudo = shape*0+1
    xmitgcm.utils.write_to_binary(pseudo.flatten(), 'T.1c_new.bin', dtype=np.dtype('>f4') )

if __name__ == "__main__":
    #from_woa()
    #new_Q_surf()
    #new_Eta()
    #new_U()
    #new_V()
