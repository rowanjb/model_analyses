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

def read_binaries_150x150(binary):
    """Reads binaries."""
    P = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/'+binary, shape=(150,150), dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)
    Y = np.linspace(0, 149, 150)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, P)
    cbar = fig.colorbar(cs)
    plt.savefig('binary_plots/'+binary[:-4]+'.png')

def read_binaries_100x100(binary):
    """Reads binaries."""
    P = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/'+binary, shape=(100,100), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 99, 100)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, Y, P)
    cbar = fig.colorbar(cs)
    plt.savefig('binary_plots/'+binary[:-4]+'.png')

def read_binaries_50x150x150(binary):
    """Reads binaries."""
    P = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/'+binary, shape=(50,150,150), dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)
    Z = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, (-1)*Z, P[:,75,:])
    cbar = fig.colorbar(cs)
    plt.savefig('binary_plots/'+binary[:-4]+'.png')

def read_binaries_50x100x100(binary):
    """Reads binaries."""
    P = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/'+binary, shape=(50,100,100), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Z = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(X, (-1)*Z, P[:,50,:])
    cbar = fig.colorbar(cs)
    plt.savefig('binary_plots/'+binary[:-4]+'.png')

if __name__ == "__main__":
    #read_binaries_150x150('Qnet_75W.40mCirc.150x150.bin')
    #read_binaries_100x100('Qnet_0W.100x100.bin')
    read_binaries_50x100x100('S.30psu.50x100x100.bin')
    #read_binaries_50x150x150('T.WOA2015.150x150.autumn.bin')
