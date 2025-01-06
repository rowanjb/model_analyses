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
    print(S)
    quit()
    pseudo = np.tile(T,(100,100,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), 'T.WOA.bin')
    pseudo = np.tile(S,(100,100,1))
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), 'S.WOA.bin')
    
    #testing
    T2 = xmitgcm.utils.read_raw_data('T.WOA.bin', shape=(50,100,100), dtype=np.dtype('>f4') )
    X = np.linspace(0, 99, 100)
    Y = np.linspace(0, 49, 50)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, T2[:,49,:])
    cbar = fig.colorbar(cs)
    plt.savefig('T.WOA.png')
    print(T2[0,:,:])
    print(T2[:,20,20])
    print(T2[:,80,80])
    quit()


def temp_from_mooring():
    ds = mooring_analyses.open_mooring_ml_data()
    dy = (-1)*np.array([
         0.4, 1.2, 2.0, 2.8, 3.6, 4.4, 5.2, 6.0, 6.8, 7.6,
         8.4, 9.2,10.0,10.8,11.6,12.4,13.2,14.0,14.8,15.6,
        16.4,17.2,18.0,18.8,19.6,20.4,21.2,22.0,22.8,23.6,
        24.4,25.2,26.0,26.8,27.6,28.4,29.2,30.0,30.8,31.6,
        32.4,33.2,34.0,34.8,35.6,36.4,37.2,38.0,38.8,39.6])
    y = np.zeros(len(dy))
    for i,n in enumerate(dy):
        if i==0: y[i] = n/2
        else: y[i] = np.sum(dy[:i]) + n/2
    ds = ds.assign_coords({'model_depths': y})
    print(ds['T'].isel(day=250).interp(depth=y).to_numpy())
    quit()

    lon = np.arange(0,100,1)
    lat = np.arange(0,100,1)
    depth = np.arange(0,50,1)
    shape,_,_ = np.meshgrid(depth,lon,lat,indexing='ij')
    T = ds['T'].isel(day=250).to_numpy()
    S = ds['S'].isel(day=250).to_numpy()
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
    df = pd.DataFrame(data={'y': y, 'dy': dy})
    print(df)
    quit()


def basic_example():
    lon = np.arange(0,100,1)
    lat = np.arange(0,100,1)
    quit()
    depth = np.arange(0,50,1)
    shape,_,_ = np.meshgrid(depth,lon,lat)
    pseudo = shape*0+1
    xmitgcm.utils.write_to_binary(pseudo.flatten(), 'T.1c_new.bin', dtype=np.dtype('>f4') )

def temp_profile_plot():
    ds = xr.open_dataset('20040212_prof.nc').isel(N_PROF=26)
    print('Latitude: ' + str(ds.LATITUDE.values))
    print('Latitude: ' + str(ds.LONGITUDE.values))
    c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])
    plt.plot(ds.TEMP_ADJUSTED.to_numpy(),(-1)*ds.PRES_ADJUSTED.to_numpy(),label='Argo float\n12.02,2004\n'+str((-1)*ds.LONGITUDE.values)+'$\degree$W, '+str((-1)*ds.LATITUDE.values)+'$\degree$S',color=c1)
    ax = plt.gca()
    ax.vlines(1,ymin=-2000,ymax=0,label='1$\degree$C',color=c2)
    ax.vlines(20,ymin=-2000,ymax=0,label='20$\degree$C',color=c3)
    ax.set_xlim([-2, 22])
    ax.set_ylim([-2000, 0])
    ax.set_xlabel('Temperature ($\degree C$)')
    ax.set_ylabel('Depth ($m$)')
    ax.set_title('Initial temperature profiles')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('temp_prof_deletable.png',dpi=900, bbox_inches="tight", pad_inches = 0.5)

def argo_temp_profile():
    ds = xr.open_dataset('20040212_prof.nc').isel(N_PROF=26)
    argo_temps = ds.TEMP_ADJUSTED.where(~np.isnan(ds.TEMP_ADJUSTED),drop=True).to_numpy()
    argo_depths = ds.PRES_ADJUSTED.where(~np.isnan(ds.TEMP_ADJUSTED),drop=True).to_numpy()
    model_depths = np.arange(10,1000,20)
    model_temps = np.interp(model_depths,argo_depths,argo_temps)
    #model_temps = [float(i) for i in sorted(np.interp(model_depths,argo_depths,argo_temps) + 2)]
    print(model_temps)
    pseudo = np.tile(model_temps,(100,100,1)) #Turn our column of temps into a 3d prism of temps
    xmitgcm.utils.write_to_binary(pseudo.flatten(order='F'), 'T.120mn.bin_argoProfile') #If flatten is causing troubles, we'll catch it in post...
    ##Plotting to check if measured and interpolated data are similar
    #plt.plot(argo_temps,(-1)*argo_depths,label='Argo')
    #plt.plot(model_temps,(-1)*model_depths,label='Interp')
    #ax = plt.gca()
    #ax.set_xlim([-2, 2])
    #ax.set_ylim([-1000, 0])
    #ax.set_xlabel('Temperature ($\degree C$)')
    #ax.set_ylabel('Depth ($m$)')
    #ax.set_title('Initial temperature profiles')
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #plt.savefig('temp_prof_deletable.png',dpi=900, bbox_inches="tight", pad_inches = 0.5)

#Argo relevant variables
#Reference: https://argo.ucsd.edu/data/how-to-use-argo-files/
#Generally use the adjusted values, I think 
#REFERENCE_DATE_TIME: Date of reference for Julian days (YYYYMMDDHHMISS)
#PROJECT_NAME: Name of the project
#DIRECTION: Direction of the station profiles (A: ascending profiles, D: descending profiles)
#JULD: Julian day (UTC) of the station relative to REFERENCE_DAT... (Relative julian days with decimal part (as parts of day))
#JULD_QC: Quality on date and time
#JULD_LOCATION: Julian day (UTC) of the location relative to REFERENCE_DATE... (Relative julian days with decimal part (as parts of day))
#LATITUDE: Latitude of the station, best estimate (degree_north; valid_min: -90.0; valid_max: 90.0)
#LONGITUDE: Longitude of the station, best estimate (degree_east; valid_min: -180.0; valid_max: 180.0)
#POSITION_QC: Quality on position (latitude and longitude)
#PROFILE_TEMP_QC:Global quality flag of TEMP profile (per profile)
#TEMP (N_PROF: 174, N_LEVELS: 1314): Sea temperature in-situ ITS-90 scale; sea_water_temperature; (degree_Celsius); C_format: %9.3f; resolution: 0.001
#TEMP_QC (N_PROF: 174, N_LEVELS: 1314): quality flag
#TEMP_ADJUSTED (N_PROF: 174, N_LEVELS: 1314): Sea temperature in-situ ITS-90 scale; sea_water_temperature; (degree_Celsius); C_format: %9.3f; resolution: 0.001
#TEMP_ADJUSTED_QC (N_PROF: 174, N_LEVELS: 1314): quality flag
#TEMP_ADJUSTED_ERROR (N_PROF: 174, N_LEVELS: 1314): Contains the error on the adjusted values as determined ...; (degree_Celsius) C_format: %9.3f; resolution: 0.001
#print(ds.POSITION_QC.values)
#print(ds.PROFILE_TEMP_QC.values)
#print(ds.TEMP_QC.values)
#print(ds.PROFILE_PRES_QC.values)
#print(ds.PRES_QC.values)

if __name__ == "__main__":
    temp_from_woa()
