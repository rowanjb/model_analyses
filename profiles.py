# Based on: https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html 
# These are some basic functions that I use to make binaries

import numpy as np
import pandas as pd
import xmitgcm
import matplotlib.pylab as plt
import matplotlib as mpl
import xmitgcm.file_utils
import xmitgcm.utils
from MITgcmutils import density
import xarray as xr
import gsw
import basic_model_anayses as bma

import sys
sys.path.insert(1, '../obs_analyses/')
import mooring_analyses
import woa_analyses

from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot

def test_eos():
    """Messy function for creating a plot of profiles from various sources.
    Ultimate goal was to compare different equations of state."""
    
    dy = np.array([ # Define the model cell thicknesses 
         0.4, 1.2, 2.0, 2.8, 3.6, 4.4, 5.2, 6.0, 6.8, 7.6,
         8.4, 9.2,10.0,10.8,11.6,12.4,13.2,14.0,14.8,15.6,
        16.4,17.2,18.0,18.8,19.6,20.4,21.2,22.0,22.8,23.6,
        24.4,25.2,26.0,26.8,27.6,28.4,29.2,30.0,30.8,31.6,
        32.4,33.2,34.0,34.8,35.6,36.4,37.2,38.0,38.8,39.6])
    y = np.zeros(len(dy)) # Create a list of cell depths
    for i,n in enumerate(dy):
        if i==0: y[i] = n/2
        else: y[i] = np.sum(dy[:i]) + n/2
    
    # open the WOA data
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    dst = xr.open_dataset(dirpath + '/WOA_seasonally_'+'t'+'_'+str(2015)+'.nc',decode_times=False).isel(depth=slice(0,48))
    dss = xr.open_dataset(dirpath + '/WOA_seasonally_'+'s'+'_'+str(2015)+'.nc',decode_times=False).isel(depth=slice(0,48))
    insit_temp = dst['t_an'].isel(time=3).values # seasons are ['winter', 'spring', 'summer', 'autumn'] (NORTHERN HEMISPHERE!)
    prac_sal = dss['s_an'].isel(time=3).values

    lat, lon =-69.0005, -27.0048
    depths = (-1)*dst.depth.values
    pressure = gsw.p_from_z(depths,lat=lat)
    abs_sal = gsw.SA_from_SP(prac_sal,pressure,lon=lon,lat=lat)
    pot_temp = gsw.pt0_from_t(abs_sal,insit_temp,pressure)
    linear_rho_from_pot_temp = density.linear(prac_sal, pot_temp, sref=35, tref=20, sbeta=0, talpha=0.0002, rhonil=1000) # density.linear(S2, T2, sref=35, tref=20, sbeta=0.00000001, talpha=0.0002, rhonil=1000) #sref=35, tref=20
    linear_rho_from_insit_temp = density.linear(prac_sal, insit_temp, sref=35, tref=20, sbeta=0, talpha=0.0002, rhonil=1000)
    cons_temp = gsw.CT_from_t(abs_sal,insit_temp,pressure)
    pot_dens_sigma0 = gsw.sigma0(abs_sal,cons_temp)
    N2,pressure2 = gsw.Nsquared(abs_sal,cons_temp,pressure,lat=lat)
    depths2 = gsw.z_from_p(pressure2, lat=lat)

    prac_sal_100x100_bin_winter = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA2015.50x100x100.winter.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    prac_sal_100x100_bin_spring = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA2015.50x100x100.spring.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    prac_sal_100x100_bin_summer = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA2015.50x100x100.summer.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    prac_sal_100x100_bin_autumn = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA2015.50x100x100.autumn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    insitu_temp_100x100_bin_winter = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x100x100.winter.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    insitu_temp_100x100_bin_spring = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x100x100.spring.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    insitu_temp_100x100_bin_summer = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x100x100.summer.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    insitu_temp_100x100_bin_autumn = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x100x100.autumn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    pot_temp_100x100_bin_autumn = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/theta.WOA2015.50x100x100.autumn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    model_depths = (-1)*y 

    host = host_subplot(111, axes_class=axisartist.Axes)
    plt.subplots_adjust(right=0.75)
    par1 = host.twiny()
    par2 = host.twiny()
    par3 = host.twiny()
    par4 = host.twiny()
    par1.axis["bottom"] = par1.new_fixed_axis(loc="bottom", offset=(0, -40))
    par2.axis["bottom"] = par2.new_fixed_axis(loc="bottom", offset=(0, -80))
    par3.axis["bottom"] = par3.new_fixed_axis(loc="bottom", offset=(0, -120))
    par4.axis["bottom"] = par4.new_fixed_axis(loc="bottom", offset=(0, -160))
    par1.axis["bottom"].toggle(all=True)
    par2.axis["bottom"].toggle(all=True)
    par3.axis["bottom"].toggle(all=True)
    par4.axis["bottom"].toggle(all=True)
    
    host.set_xlabel('Salinity')
    host.axis["bottom"].label.set_color('y')
    s1 = host.plot(prac_sal,depths,c='y',ls='-',label='Practical salinity (WOA)')
    s2 = host.plot(abs_sal, depths,c='y',ls='--',label='Absolute salinity (WOA)')
    s3 = host.plot(prac_sal_100x100_bin_autumn, model_depths,c='b',lw=2.5,ls=':',label='Practical salinity (100x100 model bin)')
    
    par1.set_xlabel('Temperature')
    par1.axis["bottom"].label.set_color('r')
    t1 = par1.plot(insit_temp,depths,c='b',ls='-',label='In-situ temperature (WOA)')
    t2 = par1.plot(pot_temp,depths,  c='b',ls='-',label='Potential temperature (WOA)')
    t3 = par1.plot(pot_temp_100x100_bin_autumn,model_depths, c='r',lw=2.5,ls=':',label='In-situ temperature (100x100 model bin)') 
    t4 = par1.plot(insitu_temp_100x100_bin_autumn,model_depths, c='r',lw=2.5,ls=':',label='Potential temperature (100x100 model bin)') 
    
    par2.set_xlabel('In-situ density (linear EOS from model)')
    par3.axis["bottom"].label.set_color('k')
    dens1 = par2.plot(linear_rho_from_pot_temp,depths,c='k',ls='-',label='In-situ density from linear EOS\nusing potential temperature (correct)')
    dens2 = par2.plot(linear_rho_from_insit_temp,depths,c='k',ls=':',label='In-situ density from linear EOS\nusing in-situ temperature (incorrect)')

    par3.set_xlabel('Potential density (non-linear EOS)')
    par3.axis["bottom"].label.set_color('b')
    j = par3.plot(pot_dens_sigma0,depths,c='b',ls='-',label='Potential density\n(sigma0 from gsw)')

    par4.set_xlabel('Buoyancy frequency')
    par4.axis["bottom"].label.set_color('g')
    b = par4.plot(N2,depths2,c='g',ls='-',label='Buoyancy frequency N**2\n(Nsquared from gsw)')

    N2 = np.array(N2)
    N2_min = np.nanmin(N2)
    id_min = depths2[np.nanargmin(N2)]
    par4.annotate('Min N**2 = '+str(N2_min)[:5]+str(N2_min)[-5:],
            xy=(N2_min, id_min), xycoords='data',
            xytext=(120, 23), textcoords='offset points',
            arrowprops=dict(facecolor='g',arrowstyle="->"),
            horizontalalignment='center', verticalalignment='center')

    lns = s1+s2+s3+t1+t2+t3+t4+dens1+dens2+j+b 
    labs = [l.get_label() for l in lns]
    host.legend(lns, labs,loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Comparing EOS methods at the Weddell mooring')# at 12.5N, 127.5W')
    plt.savefig('figures/profiles/profile_of_different_EOS.png',bbox_inches='tight',dpi=300)

def compare_WOA_and_mooring(): 
    """For comparing the profiles of different seasons (WOA binaries) and the mooring."""

    dy = np.array([ # Define the model cell thicknesses 
         0.4, 1.2, 2.0, 2.8, 3.6, 4.4, 5.2, 6.0, 6.8, 7.6,
         8.4, 9.2,10.0,10.8,11.6,12.4,13.2,14.0,14.8,15.6,
        16.4,17.2,18.0,18.8,19.6,20.4,21.2,22.0,22.8,23.6,
        24.4,25.2,26.0,26.8,27.6,28.4,29.2,30.0,30.8,31.6,
        32.4,33.2,34.0,34.8,35.6,36.4,37.2,38.0,38.8,39.6])
    y = np.zeros(len(dy)) # Create a list of cell depths
    for i,n in enumerate(dy):
        if i==0: y[i] = n/2
        else: y[i] = np.sum(dy[:i]) + n/2
    
    depths = (-1)*y 
    model_pressures = gsw.p_from_z(depths, lat=-69.0005)
    prac_sal_100x100_bin_winter =    xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA2015.50x100x100.winter.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    prac_sal_100x100_bin_spring =    xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA2015.50x100x100.spring.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    prac_sal_100x100_bin_summer =    xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA2015.50x100x100.summer.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    prac_sal_100x100_bin_autumn =    xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/S.WOA2015.50x100x100.autumn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    abs_sal_100x100_bin_winter =     xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/SA.WOA2015.50x100x100.winter.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    abs_sal_100x100_bin_spring =     xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/SA.WOA2015.50x100x100.spring.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    abs_sal_100x100_bin_summer =     xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/SA.WOA2015.50x100x100.summer.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    abs_sal_100x100_bin_autumn =     xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/SA.WOA2015.50x100x100.autumn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    insitu_temp_100x100_bin_winter = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x100x100.winter.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    insitu_temp_100x100_bin_spring = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x100x100.spring.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    insitu_temp_100x100_bin_summer = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x100x100.summer.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    insitu_temp_100x100_bin_autumn = xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/T.WOA2015.50x100x100.autumn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    pot_temp_100x100_bin_winter =    xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/theta.WOA2015.50x100x100.winter.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    pot_temp_100x100_bin_spring =    xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/theta.WOA2015.50x100x100.spring.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    pot_temp_100x100_bin_summer =    xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/theta.WOA2015.50x100x100.summer.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    pot_temp_100x100_bin_autumn =    xmitgcm.utils.read_raw_data('../MITgcm/so_plumes/binaries/theta.WOA2015.50x100x100.autumn.bin', shape=(50,100,100), dtype=np.dtype('>f4') )[:,20,20]
    cons_temp_100x100_winter = gsw.CT_from_t(abs_sal_100x100_bin_winter,insitu_temp_100x100_bin_winter,model_pressures)
    cons_temp_100x100_spring = gsw.CT_from_t(abs_sal_100x100_bin_spring,insitu_temp_100x100_bin_spring,model_pressures)
    cons_temp_100x100_summer = gsw.CT_from_t(abs_sal_100x100_bin_summer,insitu_temp_100x100_bin_summer,model_pressures)
    cons_temp_100x100_autumn = gsw.CT_from_t(abs_sal_100x100_bin_autumn,insitu_temp_100x100_bin_autumn,model_pressures)
    pot_dens_100x100_winter = gsw.sigma0(abs_sal_100x100_bin_winter,cons_temp_100x100_winter)
    pot_dens_100x100_spring = gsw.sigma0(abs_sal_100x100_bin_spring,cons_temp_100x100_spring)
    pot_dens_100x100_summer = gsw.sigma0(abs_sal_100x100_bin_summer,cons_temp_100x100_summer)
    pot_dens_100x100_autumn = gsw.sigma0(abs_sal_100x100_bin_autumn,cons_temp_100x100_autumn)

    ds = mooring_analyses.open_mooring_ml_data().sel(depth=[-50,-135,-220])
    mooring_depths = ds['depth'].values
    mooring_pressures = gsw.p_from_z(mooring_depths,lat=-69.0005)
    prac_sal_mooring_amj = ds['S'].sel(day=slice('2021-04-01','2021-06-30')).mean(dim='day')
    prac_sal_mooring_jas = ds['S'].sel(day=slice('2021-07-01','2021-09-30')).mean(dim='day')
    prac_sal_mooring_ond = ds['S'].sel(day=slice('2021-10-01','2021-12-31')).mean(dim='day')
    prac_sal_mooring_jfm = ds['S'].sel(day=slice('2022-01-01','2022-03-31')).mean(dim='day')
    abs_sal_mooring_amj = gsw.SA_from_SP(prac_sal_mooring_amj,mooring_pressures,lon=-27.0048,lat=-69.0005)
    abs_sal_mooring_jas = gsw.SA_from_SP(prac_sal_mooring_jas,mooring_pressures,lon=-27.0048,lat=-69.0005)
    abs_sal_mooring_ond = gsw.SA_from_SP(prac_sal_mooring_ond,mooring_pressures,lon=-27.0048,lat=-69.0005)
    abs_sal_mooring_jfm = gsw.SA_from_SP(prac_sal_mooring_jfm,mooring_pressures,lon=-27.0048,lat=-69.0005)
    insitu_temp_mooring_amj = ds['T'].sel(day=slice('2021-04-01','2021-06-30')).mean(dim='day')
    insitu_temp_mooring_jas = ds['T'].sel(day=slice('2021-07-01','2021-09-30')).mean(dim='day')
    insitu_temp_mooring_ond = ds['T'].sel(day=slice('2021-10-01','2021-12-31')).mean(dim='day')
    insitu_temp_mooring_jfm = ds['T'].sel(day=slice('2022-01-01','2022-03-31')).mean(dim='day')
    pot_temp_mooring_amj = gsw.pt0_from_t(abs_sal_mooring_amj,insitu_temp_mooring_amj,mooring_pressures)
    pot_temp_mooring_jas = gsw.pt0_from_t(abs_sal_mooring_jas,insitu_temp_mooring_jas,mooring_pressures)
    pot_temp_mooring_ond = gsw.pt0_from_t(abs_sal_mooring_ond,insitu_temp_mooring_ond,mooring_pressures)
    pot_temp_mooring_jfm = gsw.pt0_from_t(abs_sal_mooring_jfm,insitu_temp_mooring_jfm,mooring_pressures)
    cons_temp_mooring_amj = gsw.CT_from_t(abs_sal_mooring_amj,insitu_temp_mooring_amj,mooring_pressures)
    cons_temp_mooring_jas = gsw.CT_from_t(abs_sal_mooring_jas,insitu_temp_mooring_jas,mooring_pressures)
    cons_temp_mooring_ond = gsw.CT_from_t(abs_sal_mooring_ond,insitu_temp_mooring_ond,mooring_pressures)
    cons_temp_mooring_jfm = gsw.CT_from_t(abs_sal_mooring_jfm,insitu_temp_mooring_jfm,mooring_pressures)
    pot_dens_mooring_amj = gsw.sigma0(abs_sal_mooring_amj,cons_temp_mooring_amj)
    pot_dens_mooring_jas = gsw.sigma0(abs_sal_mooring_jas,cons_temp_mooring_jas)
    pot_dens_mooring_ond = gsw.sigma0(abs_sal_mooring_ond,cons_temp_mooring_ond)
    pot_dens_mooring_jfm = gsw.sigma0(abs_sal_mooring_jfm,cons_temp_mooring_jfm)

    fig, ax = plt.subplots()
    ax.set_title('Salinity WOA vs mooring')
    ax.set_xlabel('Salinity ($g$ $kg^{-1}$)')
    ax.set_ylim(-250,0)
    s1 = ax.plot(prac_sal_100x100_bin_winter,depths,c='y',ls='-' ,label='WOA practical salinity (winter)')
    s2 = ax.plot(prac_sal_100x100_bin_spring,depths,c='y',ls='--',label='WOA practical salinity (spring)')
    s3 = ax.plot(prac_sal_100x100_bin_summer,depths,c='y',ls=':' ,label='WOA practical salinity (summer)')
    s4 = ax.plot(prac_sal_100x100_bin_autumn,depths,c='y',ls='-.',label='WOA practical salinity (autumn)')
    s5 = ax.plot(abs_sal_100x100_bin_winter,depths,c='b',ls='-' ,label='WOA absolute salinity (winter)')
    s6 = ax.plot(abs_sal_100x100_bin_spring,depths,c='b',ls='--',label='WOA absolute salinity (spring)')
    s7 = ax.plot(abs_sal_100x100_bin_summer,depths,c='b',ls=':' ,label='WOA absolute salinity (summer)')
    s8 = ax.plot(abs_sal_100x100_bin_autumn,depths,c='b',ls='-.',label='WOA absolute salinity (autumn)')
    s9 =  ax.plot(prac_sal_mooring_jfm,mooring_depths,c='k',ls='-' ,label='Mooring practical salinity (winter)')
    s10 = ax.plot(prac_sal_mooring_amj,mooring_depths,c='k',ls='--',label='Mooring practical salinity (winter)')
    s11 = ax.plot(prac_sal_mooring_jas,mooring_depths,c='k',ls=':' ,label='Mooring practical salinity (spring)')
    s12 = ax.plot(prac_sal_mooring_ond,mooring_depths,c='k',ls='-.',label='Mooring practical salinity (summer)')
    s13 = ax.plot(abs_sal_mooring_jfm,mooring_depths,c='g',ls='-' ,label='Mooring absolute salinity (winter)')
    s14 = ax.plot(abs_sal_mooring_amj,mooring_depths,c='g',ls='--',label='Mooring absolute salinity (winter)')
    s15 = ax.plot(abs_sal_mooring_jas,mooring_depths,c='g',ls=':' ,label='Mooring absolute salinity (spring)')
    s16 = ax.plot(abs_sal_mooring_ond,mooring_depths,c='g',ls='-.',label='Mooring absolute salinity (summer)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('figures/profiles/profiles_WOA_binaries_and_mooring_salinity.png',bbox_inches='tight',dpi=300)
    plt.clf()

    fig, ax = plt.subplots()
    ax.set_title('Temperature WOA vs mooring')
    ax.set_xlabel('Temperature ($℃$)')
    ax.set_ylim(-250,0)
    t1 = ax.plot(insitu_temp_100x100_bin_winter,depths,c='r',ls='-' ,label='WOA in-situ temperature (winter)')
    t2 = ax.plot(insitu_temp_100x100_bin_spring,depths,c='r',ls='--',label='WOA in-situ temperature (spring)')
    t3 = ax.plot(insitu_temp_100x100_bin_summer,depths,c='r',ls=':' ,label='WOA in-situ temperature (summer)') 
    t4 = ax.plot(insitu_temp_100x100_bin_autumn,depths,c='r',ls='-.',label='WOA in-situ temperature (autumn)') 
    t5 = ax.plot(pot_temp_100x100_bin_winter,depths,c='b',ls='-' ,label='WOA potential ℃temperature (winter)')
    t6 = ax.plot(pot_temp_100x100_bin_spring,depths,c='b',ls='--',label='WOA potential temperature (spring)')
    t7 = ax.plot(pot_temp_100x100_bin_summer,depths,c='b',ls=':' ,label='WOA potential temperature (summer)') 
    t8 = ax.plot(pot_temp_100x100_bin_autumn,depths,c='b',ls='-.',label='WOA potential temperature (autumn)') 
    t9 =  ax.plot(insitu_temp_mooring_jfm,mooring_depths,c='k',ls='-' ,label='Mooring in-situ temperature (winter)')
    t10 = ax.plot(insitu_temp_mooring_amj,mooring_depths,c='k',ls='--',label='Mooring in-situ temperature (winter)')
    t11 = ax.plot(insitu_temp_mooring_jas,mooring_depths,c='k',ls=':' ,label='Mooring in-situ temperature (spring)')
    t12 = ax.plot(insitu_temp_mooring_ond,mooring_depths,c='k',ls='-.',label='Mooring in-situ temperature (summer)')
    t13 = ax.plot(pot_temp_mooring_jfm,mooring_depths,c='g',ls='-' ,label='Mooring potential temperature (winter)')
    t14 = ax.plot(pot_temp_mooring_amj,mooring_depths,c='g',ls='--',label='Mooring potential temperature (winter)')
    t15 = ax.plot(pot_temp_mooring_jas,mooring_depths,c='g',ls=':' ,label='Mooring potential temperature (spring)')
    t16 = ax.plot(pot_temp_mooring_ond,mooring_depths,c='g',ls='-.',label='Mooring potential temperature (summer)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('figures/profiles/profiles_WOA_binaries_and_mooring_temperature.png',bbox_inches='tight',dpi=300)
    plt.clf()

    fig, ax = plt.subplots()
    ax.set_title('Potential density WOA vs mooring')
    ax.set_xlabel('Potential density ($kg$ $m^{-3}$)')
    ax.set_ylim(-250,0)
    pd1 = ax.plot(pot_dens_100x100_winter,depths,c='r',ls='-' ,label='WOA potential density (winter)')
    pd2 = ax.plot(pot_dens_100x100_spring,depths,c='r',ls='--',label='WOA potential density (spring)')
    pd3 = ax.plot(pot_dens_100x100_summer,depths,c='r',ls=':' ,label='WOA potential density (summer)') 
    pd4 = ax.plot(pot_dens_100x100_autumn,depths,c='r',ls='-.',label='WOA potential density (autumn)') 
    pd5 = ax.plot(pot_dens_mooring_amj,mooring_depths,c='b',ls='-' ,label='Mooring potential density (winter)')
    pd6 = ax.plot(pot_dens_mooring_jas,mooring_depths,c='b',ls='--',label='Mooring potential density (spring)')
    pd7 = ax.plot(pot_dens_mooring_ond,mooring_depths,c='b',ls=':' ,label='Mooring potential density (summer)') 
    pd8 = ax.plot(pot_dens_mooring_jfm,mooring_depths,c='b',ls='-.',label='Mooring potential density (autumn)') 
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('figures/profiles/profiles_WOA_binaries_and_mooring_potential_density.png',bbox_inches='tight',dpi=300)
    plt.clf()

def compare_mooring_and_model(run):
    """Goal is to extract mooring profiles and model profiles as a visual checker-for similariry."""

    # Creating filepaths and opening the data
    # Reason for the try-except is that there are only two locations where the data might be
    try:
        data_dir = '../MITgcm/so_plumes/'+run
        ds = bma.open_mitgcm_output_all_vars(data_dir)
    except: 
        data_dir = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/'+run
        ds = bma.open_mitgcm_output_all_vars(data_dir)    
    ds = ds.resample(time='1h').mean()
    ds = bma.calculate_sigma0_TEOS10(ds)

    fig, axs = plt.subplots(ncols=3,nrows=1) 
    print("For now, ignoring the disagreement between ns output and endTime")
    cmap = mpl.colormaps['viridis']
    colours = cmap(np.linspace(0, 1, len(ds.time.values)))
    for n,time in enumerate(ds.time.values):
        ds['rho_theta'].isel(time=n, YC=138, XC=138).plot(y='Z',ax=axs[0],c=colours[n])
        ds['T'].isel(time=n, YC=138, XC=138).plot(y='Z',ax=axs[1],c=colours[n])
        ds['S'].isel(time=n, YC=138, XC=138).plot(y='Z',ax=axs[2],c=colours[n])
    axs[0].set_title('Pot density')
    axs[1].set_title('Pot temp')
    axs[2].set_title('Abs salinity')
    plt.tight_layout()
    plt.savefig('test_outside.png',dpi=450,bbox_inches="tight")
    
if __name__ == "__main__":
    #test_eos()
    #compare_WOA_and_mooring()
    compare_mooring_and_model('mrb_034')