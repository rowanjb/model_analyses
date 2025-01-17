# Script for looking at the binaries that come packaged with the deep convection test case.  

import matplotlib.pyplot as plt
import numpy as np
import xgcm 
import os
import xmitgcm
from xmitgcm import open_mdsdataset 
from MITgcmutils import density
import xmitgcm.file_utils
import xmitgcm.utils

def read_binaries():
    """Reads binaries so I can see what the initial conditions are in the binaries provided 
    in the deep convection test case."""

    #Surface forcing
    #Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/Qnet_p32.bin', shape=(100,100), dtype=np.dtype('>f4') )
    #X = np.linspace(0, 99, 100)
    #Y = np.linspace(0, 99, 100)
    #fig, ax = plt.subplots()
    #cs = ax.contourf(X, Y, Q)
    #cbar = fig.colorbar(cs)
    #plt.savefig('Qnet_p32.png')

    #Sea surface
    #Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/Eta.120mn.bin', shape=(100,100), dtype=np.dtype('>f4') )
    #X = np.linspace(0, 99, 100)
    #Y = np.linspace(0, 99, 100)
    #fig, ax = plt.subplots()
    #cs = ax.contourf(X, Y, Eta)
    #cbar = fig.colorbar(cs)
    #plt.savefig('Eta.120mn.png')

    #Temperature; note the shape of the input data is likely (depth, lon, lat)
    #T = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/T.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    #X = np.linspace(0, 99, 100)
    #Y = np.linspace(0, 99, 100)
    #fig, ax = plt.subplots()
    #cs = ax.contourf(X, Y, T[0,:,:])
    #cbar = fig.colorbar(cs)
    #plt.savefig('T.120mn_1c.png')
    #Below is for testing if my reading+writing techniques produce usable binaries that don't crash the model
    #xmitgcm.utils.write_to_binary(T.flatten(), 'T.120mn_new.bin')#, dtype=np.dtype('>f4') )
    #T = xmitgcm.utils.read_raw_data('T.120mn_new.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    #X = np.linspace(0, 99, 100)
    #Y = np.linspace(0, 99, 100)
    #fig, ax = plt.subplots()
    #cs = ax.contourf(X, Y, T[0,:,:])
    #cbar = fig.colorbar(cs)
    #plt.savefig('T.120mn_new.png')

    #U velocity
    #U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/U.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    #X = np.linspace(0, 99, 100)
    #Y = np.linspace(0, 99, 100)
    #fig, ax = plt.subplots()
    #cs = ax.contourf(X, Y, U[0,:,:])
    #cbar = fig.colorbar(cs)
    #plt.savefig('U.120mn.png')

    #V velocity 
    #V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/V.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    #X = np.linspace(0, 99, 100)
    #Y = np.linspace(0, 99, 100)
    #fig, ax = plt.subplots()
    #cs = ax.contourf(X, Y, V[0,:,:])
    #cbar = fig.colorbar(cs)
    #plt.savefig('V.120mn.png')

    #Boundary Ts from the reentrant channel tutorial
    T = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T_relax_mask.50km.bin', shape=(49,40,20), dtype=np.dtype('>f4') )
    Y = np.linspace(0, 39, 40)
    Z = np.linspace(0, 48, 49)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(Y, Z, T[:,:,10])
    cbar = fig.colorbar(cs)
    plt.savefig('T_relax_mask.50km.png')

    #Boundary Ts from the reentrant channel tutorial
    #T = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/temperature.50km.bin', shape=(49,40,20), dtype=np.dtype('>f4') )
    #Y = np.linspace(0, 39, 40)
    #Z = np.linspace(0, 48, 49)
    #fig, ax = plt.subplots()
    #cs = ax.pcolormesh(Y, Z, T[:,:,19])
    #cbar = fig.colorbar(cs)
    #plt.savefig('temperature.50km.png')

    #T_relax_mask.50km.bin ../binaries/
    #(.venv) robrow001@albedo0:~/MITgcm/so_plumes/input$ mv temperature.50km.bin ../binaries/

if __name__ == "__main__":
    read_binaries()
