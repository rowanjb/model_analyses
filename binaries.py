# Based partly on:
# https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html
# These are some basic functions that I use to make and investigate
# MITgcm binaries

import numpy as np
import xmitgcm
import matplotlib.pylab as plt
import xmitgcm.utils
import xarray as xr
import gsw
import cell_thickness_calculator as ctc

import sys
sys.path.insert(1, '../obs_analyses/')


def from_woa():
    """Script for making binaries out of WOA climatologies."""

    # Parameters
    season = 'autumn'
    season_dict = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    depth = '1000m'
    num_levels, numAx1, numAx2 = 50, 100, 100
    size = str(num_levels) +'x' + str(numAx1) + 'x' + str(numAx2)
    pot_temp = True # Whether you want pot or in-situ temp
    abs_salt = True # Whether you want abs or practical salt
    salt_dev = True # Whether you want the salinity deviation from the mean to be reduced (reduces stratifiation)
    salt_dev_perc = '90' # Percentage of salt deviation from mean (0=full deviation, 100=no deviation)

    #== Note: Depths used in the model / cell thicknesses are parabolas scaled to the domain depth. ==#
    # Note this should be changed in the future to the method used in from_mooring
    print("Check how you get the dy; it's possibly wrong!")
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
    Mooring data is interpolated onto the model grid between (e.g.,) 50 and 220 m.
    Outside these bounds, WOA climatology is used (it's "spliced in", if ya know what I mean). 
    """    
    # "Deprecated" docstring
    #Note the WOA summer (so S.H. winter) climatologies show that surface salinity equals 50 m salinity.
    #Similar for temp, but with warming in the uppwer 5ish meters.
    #I will set the upper 50 m temp and salinity to the value at the 50 m sensor.
    #I will for now do the same for under 220 m.
    #BUT in the future consider splicing in the WOA clims above and/or below the sensor data."""

    # Parameters
    depth = '500m'
    num_levels, numAx1, numAx2 = 50, 150, 150
    size = str(num_levels) +'x' + str(numAx1) + 'x' + str(numAx2)
    pot_temp = True # Whether you want pot or in-situ temp
    abs_salt = True # Whether you want abs or practical salt

    #== Note: Depths used in the model / cell thicknesses can vary. Check your specific criteria and modify "scaling". ==#
    # Deprecating this section and replacing with ctc
    #if scaling==0: # If scaling is 0, then using these parabolic thicknesses
    #    dy1000m = np.array([0.4,1.2,2.0,2.8,3.6,4.4,5.2,6.0,6.8,7.6,8.4,9.2,10.0,10.8,11.6,12.4,13.2,14.0,14.8,15.6,16.4,17.2,18.0,
    #                        18.8,19.6,20.4,21.2,22.0,22.8,23.6, 24.4,25.2,26.0,26.8,27.6,28.4,29.2,30.0,30.8,31.6,32.4,33.2,34.0,
    #                        34.8,35.6,36.4,37.2,38.0,38.8,39.6])
    #    dy500m = dy1000m/2 # 50 cells in 500 m
    #elif scaling==1: # If scaling is 1, then use these thicknesses (which I /think/ come from "cell_thickness_calculator...")
    #...    
    # Implementing usage of ctc after mrb_036; before this, you'll need to look at each run's data file for dz lists
    x1, x2 = 1, 50 # Indices of top and bottom cells
    fx1 = 1 # Depth of bottom of top cell
    min_slope = 1 # Minimum slope (should probably > x1)
    A, B, C, _, _ = ctc.find_parameters(x1, x2, fx1, 500, min_slope) # In the future, consider rewriting this so that it doesn't
    dy500m = ctc.return_cell_thicknesses(x1, x2, 500, A, B, C)             # repeat for 500 and 1000 m
    A, B, C, _, _ = ctc.find_parameters(x1, x2, fx1, 1000, min_slope)
    dy1000m = ctc.return_cell_thicknesses(x1, x2, 1000, A, B, C)
    dy_dict = {'1000m':dy1000m, '500m':dy500m}

    # Depths used in the model (calculated to the centre of the cells)
    y = np.zeros(len(dy_dict[depth]))
    for i,n in enumerate(dy_dict[depth]): # Getting sell depths
        if i==0: y[i] = n/2
        else: y[i] = np.sum(dy_dict[depth][:i]) + n/2

    # Opening the mooring data
    ds = mooring_analyses.open_mooring_ml_data(time_delta='hour')
    ds = mooring_analyses.correct_mooring_salinities(ds)
    ds = ds.isel(depth=slice(0,5,2)) # Take only -50, -135, and -220 depths (rest have nans)
    time1, time2 = '2021-09-12T21:00:00.000000000', '2021-09-13T03:00:00.000000000'
    ds = ds.sel(day=slice(time1,time2)).mean(dim='day',skipna=True) # Select a day at the start of September, right before the plume

    # Opening the WOA data; 
    # Previously used seasons, e.g., ['winter', 'spring', 'summer', 'autumn'] (i.e., NORTHERN HEMISPHERE SEASONS!)
    # Switching to monthly data, which embarrassingly I forgot existed
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    das = xr.open_dataset(dirpath + '/WOA_monthly_'+'s'+'_'+str(2015)+'.nc',decode_times=False)['s_an']
    s_woa = das.isel(time=8).interp(depth=y) # time=2 refers to "summer"
    dat = xr.open_dataset(dirpath + '/WOA_monthly_'+'t'+'_'+str(2015)+'.nc',decode_times=False)['t_an']
    t_woa = dat.isel(time=8).interp(depth=y) # time=2 refers to "summer"
    p = gsw.p_from_z((-1)*y,lat=-69.0005) # Calculating pressure from depth, then getting absolute salinity, and potential temperature 
    SA = gsw.SA_from_SP(s_woa,p,lat=-69.0005,lon=-27.0048)           # ...(you should want theta/pt---this is what the model demands!)
    pt = gsw.pt0_from_t(SA,t_woa,p)

    # Determining which salt and temp to use
    if pot_temp: # If potential temp is what we're looking for, then...
        dst = gsw.pt0_from_t(ds['SA'],ds['T'],ds['p_from_z']).values # Let t (mooring) be potential temperature
        t_woa = pt # Let t (WOA) now be potential temperature 
        t_name = 'theta' # Let the var name in the file be theta
    else: # i.e., if we /don't/ want potential temp, we will use in-situ
        dst = ds['T'].values 
        t_name = 'T'
    if abs_salt: # Similarly, if it is absolute salinity that we're looking for, then...
        dss = ds['SA'].values
        s_woa = SA # And let s (WOA) be absolute salinity
        s_name = 'SA'
    else: # i.e., if we /don't/ want absolute salinity, then we likely want PSU
        dss = ds['S'].values
        s_name = 'S'
    
    # Finding depth threshold indices, i.e., where in the model depths do mooring data apply
    id50  = np.where(y == np.min(y[y>50]) )
    id135 = np.where(y == np.min(y[y>125]) )
    id220 = np.where(y == np.min(y[y>220]) )

    # Interpolating/filling values
    s, t = np.empty(len(y)), np.empty(len(y)) # These are our final s and t vectors
    for n,d in enumerate(y):
        if d<50:
            mean_diff_s = dss[0] - s_woa[id50]
            mean_diff_t = dst[0] - t_woa[id50]
            s[n] = s_woa[n] + mean_diff_s
            t[n] = t_woa[n] + mean_diff_t
        elif d<125:
            del_s = dss[1] - dss[0]
            del_t = dst[1] - dst[0]
            weight = (d-50)/(125-50)
            s[n] = dss[0] + del_s*weight
            t[n] = dst[0] + del_t*weight
        elif d<220:
            del_s = dss[2] - dss[1]
            del_t = dst[2] - dst[1]
            weight = (d-125)/(220-125)
            s[n] = dss[1] + del_s*weight
            t[n] = dst[1] + del_t*weight
        else:
            mean_diff_s = dss[2] - s_woa[id220]
            mean_diff_t = dst[2] - t_woa[id220]
            s[n] = s_woa[n] + mean_diff_s
            t[n] = t_woa[n] + mean_diff_t
    
    pseudo = np.tile(t,(numAx2,numAx1,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/'+t_name+'.mooringSept13.'+size+'.'+depth+'.bin')
    pseudo = np.tile(s,(numAx2,numAx1,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), '../MITgcm/so_plumes/binaries/'+s_name+'.mooringSept13.'+size+'.'+depth+'.bin')

def Q_surf():
    Q = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_2500W.40mCirc.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') ) 
    Q = Q*0.4
    #newQ = np.zeros((150,150))
    #newQ[25:125,25:125] = Q
    xmitgcm.utils.write_to_binary(Q.flatten(order='F'), '../MITgcm/so_plumes/binaries/Qnet_1000W.40mCirc.150x150.bin')

def Q_surf_3D():
    Q2 = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_2500W.40mCirc.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') ) 
    Q1 = np.zeros(np.shape(Q2)) # Starting conditions (hend "1")
    Q = np.concatenate([np.tile(Q1, (1,1,1)), 
                        np.tile(Q2*2, (70,1,1)), 
                        np.tile(Q1, (25,1,1)),
                        ])
    print(np.shape(Q))
    xmitgcm.utils.write_to_binary(Q.flatten(order='C'), '../MITgcm/so_plumes/binaries/Qnet_5000W.40mCirc.96x150x150.bin' ) # Note the order!!!

def salt_flux():
    S = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_150W.40mCirc.100x100.bin', shape=(100,100), dtype=np.dtype('>f4') ) 
    S[S != 0] = 0.00001 # Try to find a reasonable value for this 
    xmitgcm.utils.write_to_binary(S.flatten(order='F'), '../MITgcm/so_plumes/binaries/Snet_00001s-1.40mCirc.100x100.bin')

def salt_flux_3D(): 
    S2 = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_2500W.40mCirc.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') ) 
    S1 = np.zeros(np.shape(S2)) # Starting conditions (hend "1")
    S2[S2.nonzero()] = -0.03#000010441682478739057 #g/kg
    S = np.concatenate([np.tile(S1, (1,1,1)), 
                        np.tile(S2, (30,1,1)), 
                        np.tile(S1, (65,1,1)),
                        ])
    xmitgcm.utils.write_to_binary(S.flatten(order='C'), '../MITgcm/so_plumes/binaries/Snet_030.40mCirc.96x150x150.bin') # Note the order!!!

def wind_stress():
    # Creates a 2D array of wind stress values for model forcing. See 3.8.6.1 "Momentum Forcing" in the manual.
    # Based on ~15 m/s wind speed (I think at 10 m...) and the equation tau = rho_air C_D U**2 = 1.394 0.0015 15**2
    # See: https://en.wikipedia.org/wiki/Wind_stress ; https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html 
    tau = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/Qnet_2500W.40mCirc.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') ) 
    stress = 1.394*0.0015*(10**2)
    tau = np.where(tau>0, stress, 0)
    xmitgcm.utils.write_to_binary(tau.flatten(order='F'), '../MITgcm/so_plumes/binaries/tau_021.40mCirc.150x150.bin')

def wind_stress_3D():
    # Creates a 3D array of wind stress values for model forcing based on the 2D arrays created using wind_stress()
    tau2 = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/tau_021.40mCirc.150x150.bin', shape=(150,150), dtype=np.dtype('>f4') ) 
    tau1 = np.zeros(np.shape(tau2))
    tau = np.concatenate([np.tile(tau2, (1,1,1)), # Copied from salt_flux_3D
                          np.tile(tau2, (30,1,1)), # Going to start with invariant wind, hence no tau1
                          np.tile(tau2, (65,1,1)),
                          ])
    xmitgcm.utils.write_to_binary(tau.flatten(order='C'), '../MITgcm/so_plumes/binaries/tau_021.40mCirc.96x150x150.bin') # Note the order!!!

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
    U = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x150x150.spring.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    U[U != np.isnan] = 0 
    U = np.random.uniform(low = -0.001, high = 0.001, size = np.shape(U)) # For init with random velocities; can comment out
    xmitgcm.utils.write_to_binary(U.flatten(order='F'), '../MITgcm/so_plumes/binaries/U.rand001init.50x150x150.bin') #might need to be "C" if not all 0s

def V():
    #V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/V.120mn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    V = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x150x150.spring.bin', shape=(50,150,150), dtype=np.dtype('>f4') )
    V[V != np.isnan] = 0
    V = np.random.uniform(low = -0.001, high = 0.001, size = np.shape(V)) # For init with random velocities; can comment out
    xmitgcm.utils.write_to_binary(V.flatten(order='F'), '../MITgcm/so_plumes/binaries/V.rand001init.50x150x150.bin') #might need to be "C" if not all 0s

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

def read_binaries_100x100xt(binary, length):
    """Reads binaries with a time dimension."""
    P = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/'+binary, shape=(length,100,100), dtype=np.dtype('>f4') ) #shape=(14*3,10*2,12), dtype=np.dtype('>f4'),order='F' ) #shape=(100,100,3), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)#(0, 41, 42)
    Y = np.linspace(0, 99, 100)#(0, 19, 20)
    fig, axs = plt.subplots(nrows=length,ncols=1,squeeze=True,figsize=(1,length))
    for i in range(length):
        cs = axs[i].pcolormesh(Y, X, P[i,:,:])
        cbar = fig.colorbar(cs)
    plt.savefig('binary_plots/'+binary[:-4]+'.png')

def read_binaries_tx150x150(binary, length):
    """Reads binaries with a time dimension."""
    P = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/'+binary, shape=(length,150,150), dtype=np.dtype('>f4') ) #shape=(14*3,10*2,12), dtype=np.dtype('>f4'),order='F' ) #shape=(100,100,3), dtype=np.dtype('>f4') )
    X = np.linspace(0, 149, 150)#(0, 41, 42)
    Y = np.linspace(0, 149, 150)#(0, 19, 20)
    fig, axs = plt.subplots(nrows=length,ncols=1,squeeze=True,figsize=(2,length))
    for i in range(length):
        cs = axs[i].pcolormesh(Y, X, P[i,:,:])
        cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig('binary_plots/'+binary[:-4]+'.png',bbox_inches="tight")

if __name__ == "__main__":
    #from_woa()
    #from_mooring()
    #Q_surf()
    wind_stress()
    #Eta()
    #U()
    #V()
    #constant_S_or_T()
    #Q_surf_3D()
    #salt_flux_3D()
    wind_stress_3D()
    #read_binaries_150x150('tau_047.40mCirc.150x150.bin')
    #read_binaries_100x100('Qnet_2500W.40mCirc.100x100.bin')
    #read_binaries_50x100x100('theta.mooring.50x100x100.bin')
    #read_binaries_50x150x150('V.rand001init.50x150x150.bin')
    #read_binaries_50x150x150('U.rand001init.50x150x150.bin')
    #read_binaries_50x150x150('SA.mooringSept13.50x150x150.500m.bin')
    #read_binaries_50x150x150('theta.mooringSept13.50x150x150.500m.bin')
    #read_binaries_100x100xt('Qnet_150W.40mCirc.100x100x24.bin',24)
    #read_binaries_150x150xt('Qnet_2500W.40mCirc_v2.150x150x24.bin',24)
    #read_binaries_tx150x150('Qnet_5000W.40mCirc.96x150x150.bin',96)
    read_binaries_tx150x150('tau_047.40mCirc.96x150x150.bin',96)
    #temporary_read_ver_bins()