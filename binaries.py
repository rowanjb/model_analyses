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
import mooring_analyses as ma


def from_mooring(dx: int, dy: int, Nx: int, Ny: int, Nz: int, fp: str, dz,
                 bottom: int = 500):
    """
    Script for making and saving binaries out of mooring data.
    Mooring data is interpolated onto the model grid between (e.g.,)
    50 and 220 m. Outside these bounds, WOA climatology is used (it's
    "spliced in", if ya know what I mean).

    Parameters:
        dx, dy (int): grid spacing (m) in x and y
        Nx, Ny, Nz (int): number of total grid points in x, y, and z
        fp (str): filepath of the binary
        dz: if int then dz is the grid spacing (m) in z, if 'variable'
            then the vertical dimension is varied exponentially from
            1 m at the surface to a depth set by `bottom'
        bottom (optional): if dz=='variable' then bottom is the bottom of
            the lowest cell (in m), default is 500 m
    """

    # Parameters
    if dz == "variable":
        num_levels, numAx1, numAx2 = Nz, Nx, Ny
        size = str(num_levels)+'z_x_'+str(numAx1)+'x_x_'+str(numAx2)+'y'
        x1, x2 = 1, Nz  # Indices of top and bottom cells
        fx1 = 1  # Depth of bottom of top cell (i.e., its thickness)
        min_slope = 1  # i.e., with >1, then cell #2 is a bit thicker than #1
        A, B, C, _, _ = ctc.find_parameters(x1, x2, fx1, bottom, min_slope)
        dys = ctc.return_cell_thicknesses(x1, x2, bottom, A, B, C)
        y = np.zeros(len(dys))
        for i, n in enumerate(dys):  # Getting cell centres
            if i == 0:  # If top cell, simply divide its thickness by two
                y[i] = n/2
            else:  # Otherwise, sum all cells to current cell and add a half
                y[i] = np.sum(dys[:i]) + n/2
    elif type(dz) is int:
        dys = [dz for n in range(Nz)]
        y = [i*(n)+i/2 for n, i in enumerate(dys)]
    else:
        print("Illegal dz")
        quit()

    # Opening the mooring data
    ds = ma.open_mooring_data()
    ds.correct_mooring_salinities()
    ds.append_gsw_vars()
    ds = ds.sel(depth=[-50, -125, -220])
    time1 = '2021-09-12T21:00:00.000000000'
    time2 = '2021-09-13T03:00:00.000000000'
    ds = ds.sel(time=slice(time1, time2)).mean(dim='time', skipna=True)
    # Select a day at the start of September, right before the plume

    # Opening the WOA data
    # Previously used seasons, e.g., ['winter', 'spring',
    # 'summer', 'autumn'] (i.e., NORTHERN HEMISPHERE SEASONS!)
    # Now switching to monthly data, which embarrassingly I forgot existed
    with open('../filepaths/woa_filepath') as f:
        dirpath = f.readlines()[0][:-1]
    das = xr.open_dataset(
        dirpath + '/WOA_monthly_' + 's' + '_' + str(2015) + '.nc',
        decode_times=False)['s_an']
    s_woa = das.isel(time=8).interp(depth=y)  # time is the month from jan=0
    dat = xr.open_dataset(
        dirpath + '/WOA_monthly_' + 't' + '_' + str(2015) + '.nc',
        decode_times=False)['t_an']
    t_woa = dat.isel(time=8).interp(depth=y)  # time is the month from jan=0
    # Calculating pressure from depth, then getting SA, and pt
    p = gsw.p_from_z([(-1)*i for i in y], lat=-69.0005)
    s_woa = gsw.SA_from_SP(s_woa, p, lat=-69.0005, lon=-27.0048)
    t_woa = gsw.pt0_from_t(s_woa, t_woa, p)
    dst = gsw.pt0_from_t(ds['SA'], ds['T'], ds['p_from_z']).values
    dss = ds['SA'].values

    # Finding depth threshold indices, i.e., where in the model depths
    # do mooring data apply
    y = np.array(y)
    id50 = np.where(y == np.min(y[y > 50]))
    id125 = np.where(y == np.min(y[y > 125]))
    id220 = np.where(y == np.min(y[y > 220]))

    # Interpolating/filling values
    # These are our final s and t vectors
    s, t = np.empty(len(y)), np.empty(len(y))
    for n, d in enumerate(y):
        if d < 50:
            mean_diff_s = dss[0] - s_woa[id50]
            mean_diff_t = dst[0] - t_woa[id50]
            s[n] = s_woa[n] + mean_diff_s
            t[n] = t_woa[n] + mean_diff_t
        elif d < 125:
            del_s = dss[1] - dss[0]
            del_t = dst[1] - dst[0]
            weight = (d-50)/(125-50)
            s[n] = dss[0] + del_s*weight
            t[n] = dst[0] + del_t*weight
        elif d < 220:
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

    fig, ax = plt.subplots()
    ax.plot(gsw.sigma0(s, gsw.CT_from_pt(s, t)), y)
    ax.invert_yaxis()
    plt.savefig('test.png')

    quit()

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
    from_mooring(dx=3, dy=3, Nx=75, Ny=1, Nz=75, fp='test', dz=3)
