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
    season = 'autumn'
    season_dict = {'winter':0,'spring':1,'summer':2,'autumn':3}
    depth = '1000m'
    num_levels, numAx1, numAx2 = 50, 100, 100
    size = str(num_levels) +'x' + str(numAx1) + 'x' + str(numAx2)
    pot_temp = True # Whether you want pot or in-situ temp
    abs_salt = True # Whether you want abs or practical salt
    salt_dev = True # Whether you want the salinity deviation from the mean to be reduced (reduces stratifiation)
    salt_dev_perc = '90' # Percentage of salt deviation from mean (0=full deviation, 100=no deviation)

    #== Note: Depths used in the model / cell thicknesses are parabolas scaled to the domain depth. ==#
    # 50 cells in 1000 m: 
    dy1000m = np.array([0.4,1.2,2.0,2.8,3.6,4.4,5.2,6.0,6.8,7.6,8.4,9.2,10.0,10.8,11.6,12.4,13.2,14.0,14.8,15.6,16.4,17.2,18.0,
                        18.8,19.6,20.4,21.2,22.0,22.8,23.6, 24.4,25.2,26.0,26.8,27.6,28.4,29.2,30.0,30.8,31.6,32.4,33.2,34.0,
                        34.8,35.6,36.4,37.2,38.0,38.8,39.6])
    dy_dict = {'1000m':dy1000m}

    # Depths used in the model
    y = np.zeros(len(dy_dict[depth]))
    for i,n in enumerate(dy_dict[depth]): # Getting sell depths
        if i==0: y[i] = n/2
        else: y[i] = np.sum(dy_dict[depth][:i]) + n/2

    # Opening the WOA data; seasons are ['winter', 'spring', 'summer', 'autumn'] (i.e., NORTHERN HEMISPHERE SEASONS!)
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    das = xr.open_dataset(dirpath + '/WOA_seasonally_'+'s'+'_'+str(2015)+'.nc',decode_times=False)['s_an']
    s = das.isel(time=season_dict[season]).interp(depth=y)
    dat = xr.open_dataset(dirpath + '/WOA_seasonally_'+'t'+'_'+str(2015)+'.nc',decode_times=False)['t_an']
    t = dat.isel(time=season_dict[season]).interp(depth=y)

    # Calculating pressure from depth, absolute salinity, and potential temperature 
    # (you should want theta/pt---this is what the model demands!)
    p = gsw.p_from_z((-1)*y,lat=-69.0005)
    SA = gsw.SA_from_SP(s,p,lat=-69.0005,lon=-27.0048)
    pt = gsw.pt0_from_t(SA,t,p)

    # Determining which salt and temp to use
    if pot_temp: # If potential temp, then...
        t = pt # Let t now be potential temperature
        t_name = 'theta' # Let the var name in the file be theta
    else: # etc
        t_name = 'T'
    if abs_salt: 
        s = SA
        s_name = 'SA'
    else:
        s_name = 'S'

    # Reducing the salinity's deviation
    if salt_dev:
        weighted_mean_s = np.average(s,weights=dy_dict[depth]) # Mean of the salinity profile
        delta_s = s - weighted_mean_s # Calculate the deviations
        s = s - delta_s*(1-float('0.'+salt_dev_perc)) # Remove some fraction of the deviation
        s_name = s_name + 'x0' + salt_dev_perc

    # Name dicts 
    pseudo = np.tile(t,(100,100,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/'+t_name+'.WOA2015.'+size+'.'+season+'.bin')
    pseudo = np.tile(s,(numAx2,numAx1,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/'+s_name+'.WOA2015.'+size+'.'+season+'.bin')

def from_mooring():
    """Script for making binaries out of mooring data.
    Note the WOA summer (so S.H. winter) climatologies show that surface salinity equals 50 m salinity.
    Similar for temp, but with warming in the uppwer 5ish meters.
    I will set the upper 50 m temp and salinity to the value at the 50 m sensor.
    I will for now do the same for under 220 m.
    BUT in the future consider splicing in the WOA clims above and/or below the sensor data."""

    # Parameters
    depth = '500m'
    num_levels, numAx1, numAx2 = 50, 150, 150
    size = str(num_levels) +'x' + str(numAx1) + 'x' + str(numAx2)
    pot_temp = True # Whether you want pot or in-situ temp
    abs_salt = True # Whether you want abs or practical salt

    #== Note: Depths used in the model / cell thicknesses are parabolas scaled to the domain depth. ==#
    # 50 cells in 1000 m: 
    dy1000m = np.array([0.4,1.2,2.0,2.8,3.6,4.4,5.2,6.0,6.8,7.6,8.4,9.2,10.0,10.8,11.6,12.4,13.2,14.0,14.8,15.6,16.4,17.2,18.0,
                        18.8,19.6,20.4,21.2,22.0,22.8,23.6, 24.4,25.2,26.0,26.8,27.6,28.4,29.2,30.0,30.8,31.6,32.4,33.2,34.0,
                        34.8,35.6,36.4,37.2,38.0,38.8,39.6])
    dy500m = dy1000m/2 # 50 cells in 500 m
    dy_dict = {'1000m':dy1000m, '500m':dy500m}

    # Depths used in the model
    y = np.zeros(len(dy_dict[depth]))
    for i,n in enumerate(dy_dict[depth]): # Getting sell depths
        if i==0: y[i] = n/2
        else: y[i] = np.sum(dy_dict[depth][:i]) + n/2
    
    print(dy_dict[depth])

    # Opening the mooring data
    ds = mooring_analyses.open_mooring_ml_data()
    ds = mooring_analyses.correct_mooring_salinities(ds)
    ds = ds.drop_vars('P').dropna(dim='depth') # Drop P and then you can get rid of depths with no salt observations
    ds = ds.sel(day='2021-09-01T00:00:00.000000000') # Select a day at the start of September, right before the plume

    # Opening the WOA data; seasons are ['winter', 'spring', 'summer', 'autumn'] (i.e., NORTHERN HEMISPHERE SEASONS!)
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    das = xr.open_dataset(dirpath + '/WOA_seasonally_'+'s'+'_'+str(2015)+'.nc',decode_times=False)['s_an']
    s_woa = das.isel(time=2).interp(depth=y) # time=2 refers to "summer"
    dat = xr.open_dataset(dirpath + '/WOA_seasonally_'+'t'+'_'+str(2015)+'.nc',decode_times=False)['t_an']
    t_woa = dat.isel(time=2).interp(depth=y) # time=2 refers to "summer"
    p = gsw.p_from_z((-1)*y,lat=-69.0005) # Calculating pressure from depth, absolute salinity, and potential temperature 
    SA = gsw.SA_from_SP(s_woa,p,lat=-69.0005,lon=-27.0048)           # ...(you should want theta/pt---this is what the model demands!)
    pt = gsw.pt0_from_t(SA,t_woa,p)

    # Determining which salt and temp to use
    if pot_temp: # If potential temp, then...
        dst = gsw.pt0_from_t(ds['SA'],ds['T'],ds['p_from_z']).values # Let t now be potential temperature
        t_woa = pt # Let t (WOA) now be potential temperature 
        t_name = 'theta' # Let the var name in the file be theta
    else: # etc
        dst = ds['T'].values
        t_name = 'T'
    if abs_salt: 
        dss = ds['SA'].values
        s_woa = SA # And let s (WOA) be absolute salinity
        s_name = 'SA'
    else:
        dss = ds['S'].values
        s_name = 'S'
    
    # Finding depth threshold indices 
    id50  = np.where(y == np.min(y[y>50]) )
    id135 = np.where(y == np.min(y[y>135]) )
    id220 = np.where(y == np.min(y[y>220]) )

    # Interpolating/filling values
    s, t = np.empty(len(y)), np.empty(len(y))
    for n,d in enumerate(y):
        if d<50:
            mean_diff_s = dss[0] - s_woa[id50]
            mean_diff_t = dst[0] - t_woa[id50]
            s[n] = s_woa[n] + mean_diff_s
            t[n] = t_woa[n] + mean_diff_t
        elif d<135:
            del_s = dss[1] - dss[0]
            del_t = dst[1] - dst[0]
            weight = (d-50)/(135-50)
            s[n] = dss[0] + del_s*weight
            t[n] = dst[0] + del_t*weight
        elif d<220:
            del_s = dss[2] - dss[1]
            del_t = dst[2] - dst[1]
            weight = (d-135)/(220-135)
            s[n] = dss[1] + del_s*weight
            t[n] = dst[1] + del_t*weight
        else:
            mean_diff_s = dss[2] - s_woa[id220]
            mean_diff_t = dst[2] - t_woa[id220]
            s[n] = s_woa[n] + mean_diff_s
            t[n] = t_woa[n] + mean_diff_t
    
    pseudo = np.tile(t,(numAx2,numAx1,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/'+t_name+'.mooring.'+size+'.'+depth+'.bin')
    pseudo = np.tile(s,(numAx2,numAx1,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/'+s_name+'.mooring.'+size+'.'+depth+'.bin')

def Q_surf():
    Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_p32.bin', shape=(100,100), dtype=np.dtype('>f4') ) 
    Q = Q*3
    newQ = np.zeros((150,150))
    newQ[25:125,25:125] = Q
    xmitgcm.utils.write_to_binary(newQ.flatten(order='F'), '../MITgcm/so_plumes/binaries/Qnet_2500W.40mCirc.150x150.bin')

def Q_surf_3D():
    Q2 = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_2500W.40mCirc.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') ) 
    Q1 = np.zeros(np.shape(Q2)) # Starting conditions (hend "1")
    Q = np.stack([Q1, Q2, Q2, Q2, Q2, 
                  Q2, Q1, Q1, Q1, Q1, 
                  Q1, Q1, Q1, Q1, Q1, 
                  Q1, Q1, Q1, Q1, Q1, 
                  Q1, Q1, Q1, Q1, Q1,
                  Q1, Q1, Q1, Q1, Q1,
                  Q1, Q1, Q1, Q1, Q1,
                  Q1, Q1, Q1, Q1, Q1,
                  Q1, Q1, Q1, Q1, Q1,
                  Q1, Q1, Q1,
                  ])
    print(np.shape(Q))
    xmitgcm.utils.write_to_binary(Q.flatten(order='F'), '../MITgcm/so_plumes/binaries/Qnet_2500W.40mCirc.150x150x48.bin' )

def salt_flux():
    S = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_150W.40mCirc.100x100.bin', shape=(100,100), dtype=np.dtype('>f4') ) 
    S[S != 0] = 0.00001 # Try to find a reasonable value for this 
    xmitgcm.utils.write_to_binary(S.flatten(order='F'), '../MITgcm/so_plumes/binaries/Snet_00001s-1.40mCirc.100x100.bin')

def salt_flux_3D():
    S2 = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Snet_00001s-1.40mCirc.100x100.bin', shape=(100,100), dtype=np.dtype('>f4') ) 
    S1 = np.zeros(np.shape(S2)) # Starting conditions (hend "1")
    S = np.stack([S1, S2, S1],axis=2)
    xmitgcm.utils.write_to_binary(S.flatten(order='F'), '../MITgcm/so_plumes/binaries/Snet_00001s-1.40mCirc.100x100x3.bin')

def Eta():
    #Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Eta.120mn.bin', shape=(100,100), dtype=np.dtype('>f4') )
    Eta = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_WOA2.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') )
    Eta[Eta != np.isnan] = 0
    xmitgcm.utils.write_to_binary(Eta.flatten(order='F'), '../MITgcm/so_plumes/binaries/Eta.flat.150x150.bin')

def constant_S_or_T():
    S = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/U.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    #S = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    S[S != np.isnan] = 34.8
    xmitgcm.utils.write_to_binary(S.flatten(order='F'), '../MITgcm/so_plumes/binaries/S.const34.8.50x100x100.bin') #might need to be "C" if not all 0s

def U():
    #U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/U.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA.150x150.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    U[U != np.isnan] = 0 
    xmitgcm.utils.write_to_binary(U.flatten(order='F'), '../MITgcm/so_plumes/binaries/U.motionless.150x150.bin') #might need to be "C" if not all 0s

def V():
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

def read_binaries_100x100xt(binary):
    """Reads binaries with a time dimension."""
    P = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/'+binary, shape=(100,100,24), dtype=np.dtype('>f4') ) #shape=(14*3,10*2,12), dtype=np.dtype('>f4'),order='F' ) #shape=(100,100,3), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)#(0, 41, 42)
    Y = np.linspace(0, 99, 100)#(0, 19, 20)
    _,_,z = np.shape(P)
    fig, axs = plt.subplots(nrows=z,ncols=1,squeeze=True,figsize=(1,12))
    for i in range(z):
        cs = axs[i].pcolormesh(Y, X, P[:,:,i])
        cbar = fig.colorbar(cs)
    plt.savefig('binary_plots/'+binary[:-4]+'.png')

if __name__ == "__main__":
    #from_woa()
    from_mooring()
    #Q_surf()
    #Eta()
    #U()
    #V()
    #constant_S_or_T()
    Q_surf_3D()
    #read_binaries_150x150('Qnet_2500W.40mCirc.150x150.bin')
    #read_binaries_100x100('Qnet_2500W.40mCirc.100x100.bin')
    #read_binaries_50x100x100('theta.mooring.50x100x100.bin')
    read_binaries_50x150x150('theta.mooring.50x150x150.500m.bin')
    read_binaries_50x150x150('SA.mooring.50x150x150.500m.bin')
    #read_binaries_100x100xt('Qnet_150W.40mCirc.100x100x24.bin')