# Based partly on: https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html 
# These are some basic functions that I use to make and investigate MITgcm binaries

import numpy as np
import pandas as pd
import xmitgcm
import matplotlib.pylab as plt
import xmitgcm.file_utils
import xmitgcm.utils
from MITgcmutils import density
import xarray as xr
import gsw

import sys
sys.path.insert(1, '../obs_analyses/')
import mooring_analyses
import woa_analyses

def from_woa():
    """Script for making binaries out of WOA climatologies."""

    # Parameters
    season = 'summer'
    season_dict = {'winter':0,'spring':1,'summer':2,'autumn':3}
    num_levels, numAx1, numAx2 = 50, 150, 150
    size = str(num_levels) +'x' + str(numAx1) + 'x' + str(numAx2)

    # Depths used in the model
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

    # open the WOA data; seasons are ['winter', 'spring', 'summer', 'autumn'] (NORTHERN HEMISPHERE!)
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    dat = xr.open_dataset(dirpath + '/WOA_seasonally_'+'t'+'_'+str(2015)+'.nc',decode_times=False)['t_an']
    das = xr.open_dataset(dirpath + '/WOA_seasonally_'+'s'+'_'+str(2015)+'.nc',decode_times=False)['s_an']
    t = dat.isel(time=season_dict[season]).interp(depth=y)
    s = das.isel(time=season_dict[season]).interp(depth=y)

    #comment/uncomment for theta/pt instead of t (you should want theta/pt---this is what the model demands)
    p = gsw.p_from_z((-1)*y,lat=-69.0005)
    SA = gsw.SA_from_SP(s,p,lat=-69.0005,lon=-27.0048)
    pt = gsw.pt0_from_t(SA,t,p)

    #pseudo = np.tile(pt,(100,100,1))
        #pseudo[:,:,:20] = 10 #for a 2layer run
        ##pseudo[:,:,20:] = 9.99 #for a 2layer run
    #xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/theta.WOA2015.100x100.'+season+'.bin')
    pseudo = np.tile(SA,(numAx2,numAx1,1))
        #pseudo[:,:,:] = 30 #for a 2layer run
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/SA.WOA2015.'+size+'.'+season+'.bin')
    
def new_Q_surf():
    Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_p32.bin', shape=(100,100), dtype=np.dtype('>f4') ) 
    newQ = np.zeros((150,150))
    newQ[25:125,25:125] = Q/12
    xmitgcm.utils.write_to_binary(newQ.flatten(order='F'), '../MITgcm/so_plumes/binaries/Qnet_WOA2.150x150.bin')

def new_Eta():
    #Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Eta.120mn.bin', shape=(100,100), dtype=np.dtype('>f4') )
    Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_WOA2.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') )
    Eta[Eta != np.isnan] = 0
    xmitgcm.utils.write_to_binary(Eta.flatten(order='F'), '../MITgcm/so_plumes/binaries/Eta.flat.150x150.bin')

def new_U():
    #U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/U.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    U[U != np.isnan] = 0 
    xmitgcm.utils.write_to_binary(U.flatten(order='F'), '../MITgcm/so_plumes/binaries/U.motionless.150x150.bin') #might need to be "C" if not all 0s

def new_V():
    #V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/V.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    V[V != np.isnan] = 0
    xmitgcm.utils.write_to_binary(V.flatten(order='F'), '../MITgcm/so_plumes/binaries/V.motionless.150x150.bin') #might need to be "C" if not all 0s

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
    #from_woa()
    #from_mooring()
    #new_Q_surf()
    #new_Eta()
    #new_U()
    #new_V()
    #read_binaries_150x150('Qnet_75W.40mCirc.150x150.bin')
    #read_binaries_100x100('Qnet_0W.100x100.bin')
    read_binaries_50x100x100('T.WOA2015.50x100x100.autumn.bin')
    #read_binaries_50x150x150('SA.WOA2015.50x150x150.autumn.bin')
