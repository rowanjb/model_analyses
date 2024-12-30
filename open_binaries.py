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
    in the deep convection test case"""

    Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/Qnet_p32.bin', shape=(100,100), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 99, 100)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Q)
    cbar = fig.colorbar(cs)
    plt.savefig('Qnet_p32.png')

    Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/Eta.120mn.bin', shape=(100,100), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 99, 100)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Eta)
    cbar = fig.colorbar(cs)
    plt.savefig('Eta.120mn.png')

    T = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/T.120mn.bin', shape=(100,100,50), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 99, 100)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, T[:,:,0])
    cbar = fig.colorbar(cs)
    plt.savefig('T.120mn.png')

    U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/U.120mn.bin', shape=(100,100,50), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 99, 100)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, U[:,:,0])
    cbar = fig.colorbar(cs)
    plt.savefig('U.120mn.png')

    V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/mrb_001/V.120mn.bin', shape=(100,100,50), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 99, 100)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, V[:,:,0])
    cbar = fig.colorbar(cs)
    plt.savefig('V.120mn.png')

if __name__ == "__main__":
    
    read_binaries()
