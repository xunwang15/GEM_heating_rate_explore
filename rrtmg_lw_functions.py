import xarray as xr
import numpy as np
from glob import glob
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import pickle
from scipy.interpolate import griddata
from multiprocessing import Pool
from netCDF4 import Dataset
from scipy import interpolate
from datetime import datetime


def column_number_density(pressure_bound, wkl):
    g=9.8                                             # [m/s2]
    wvvmr = wkl[0,:]*1.
    M_mu = ((1 - wvvmr)*28.966 + wvvmr*18.016)/1000.  # molecular weight [kg/mol]
    N_A = 6.02214e23                                  # Avogadro constant [molecules mol-1]
    constant = N_A/M_mu/g/100
                                                      # pressure [hPa]
    cnd = abs(pressure_bound[:-1]-pressure_bound[1:])*(1-np.nansum(wkl[1:,:],axis=0))*constant
    return cnd #     Returns:Column_Number density [mol cm-2].

def get_wbroadl(wkl,p_lev):
    ## This function is the same as column_number_density.
    ## This function is the same as the matlab version "broad.m"
    ## I only use column_number_density for now
    Na = 6.02e23 # Avogadro constant mol−1
    g = 9.8 # m/s2
    wvvmr = wkl[0,:] # water vapor volume mixing ratio mol/mol correct
    amm = ((1 - wvvmr)*28.966 + wvvmr*18.016)/1000.; # The molecular weight of moist air kg/mol
    dp = (p_lev[:-1] - p_lev[1:]); # hPa
    dry_air = dp*Na/(100*g*amm*(1+wvvmr))
    summol = np.nansum(wkl[1:,],axis=0)
    wbroadl = dry_air*(1-summol)
    return wbroadl


def gas_mr_single_column(nmol, k_bnd, gas_mr_wv,gas_mr_o3, gas_mr_mixed, pressure_bound ):
    gas_mr = np.empty((nmol+1,k_bnd-1),dtype=np.float32)
    gas_mr[0,:] = gas_mr_wv[:]*1.
    gas_mr[1,:] = gas_mr_mixed[0]
    gas_mr[2,:] = gas_mr_o3[:]*1.
    gas_mr[3,:] = gas_mr_mixed[1]
    gas_mr[4,:] = gas_mr_mixed[2]
    gas_mr[5,:] = gas_mr_mixed[3]
    gas_mr[6,:] = gas_mr_mixed[4]
    #gas_mr[7,:] = column_number_density(pressure_bound, volume_mixing_ratio=1-gas_mr[0,:])
    #gas_mr[7,:] = column_number_density(pressure_bound, volume_mixing_ratio=1-np.nansum(gas_mr[1:,:],axis=0))
    gas_mr[7,:] = column_number_density(pressure_bound, gas_mr[:-1,:])
    return gas_mr


def write_input_rrtm_lw(filename, input_data):
    """ Create input for rrtm_lw v3.3
        Please see detailed instruction in rrtm_lw/rrtm_instructions
        Output file name: INPUT_RRTM
    """
    n_level, plev,ta, plev_bound, ta_bound, its, nmol, gas_mr,cflag,emi = input_data
    with open(filename,'w', encoding="utf-8") as f:
        ###############  # record 1.1
        f.write(f'$ ATMOSPHERE PROFILE\n');
        ###############  # record 1.2
        iaer = 0
        iatm    = 0      # flag for RRTATM (0, 1->yes)
        ixsect  = 0      # flag for cross-sections | 0: no cross-sections included in calculation
        numangs = 0      # (0,1,2,3,4) flag for number of quadrature angles to use in RT
        iout    = 0      # -1 no output% 0 for 10-3250% n from band n% 98 17 spectral bands
        idrv    = 0
        imca    = cflag
        icld    = cflag  #(0,1,2) flag for clouds

        f.write(f'{iaer:20d}{iatm:30d}{ixsect:20d}{numangs:15d}{iout:5d}{idrv:2d}{imca:2d}{icld:1d}\n')

        ###############  # record 1.4

        tbound = its     # temperature at the surface.  If input value < 0, then surface temperature
                         # is set to the temperature at the bottom of the first layer (see Record 2.1.1
        iemis = 1        # surface emissivity (0-1.0,1-same emissivity = semis(1), 2-different)
        ireflect = 0     # 0 for Lambertian reflection at surface, i.e. reflected radiance
                         # is equal at all angles
        semis = emi     # the surface emissivity for each band (see Table I).

        f.write(f'{tbound:10.3f}{iemis:2d}{ireflect:3d}{semis:5.3f}\n')

        ###############  # record 2.1
        iform = 0        # column amount format flag> 0: read PAVE, WKL(M,L), WBROADL(L) in F10.4, E10.3, E10.3 formats (default)
        f.write(f'{iform:2d}{n_level:3d}{nmol:5d}\n')                                                   # record 2.1
        ## surface setting
        f.write(f'{plev[0]:10.4f}{ta[0]:10.4f}{plev_bound[0]:31.3f}' +
              f'{ta_bound[0]:7.2f}{plev_bound[1]:15.3f}{ta_bound[1]:7.2f}\n')                           # record 2.1.1
        f.write(''.join([f'{gas_mr[i,0]:10.3e}' for i in range(0, nmol+1)])+'\n')                       # record 2.1.2
        ## atmospheric   profile
        for k in range(1, n_level):                                                                     # record 2.1.1&2
            f.write(f'{plev[k]:10.4f}{ta[k]:10.4f}{plev_bound[k+1]:53.3f}{ta_bound[k+1]:7.2f}\n')
            f.write(''.join([f'{gas_mr[i,k]:10.3e}' for i in range(nmol+1)])+'\n')
        f.write(f'$ %%%%%');
        f.close()


def write_input_rrtm_sw(filename, input_data):
    """ Create input for rrtm_sw v3.3
        Output file name: INPUT_RRTM
    """
    n_level, plev, ta,plev_bound, ta_bound,nmol,gas_mr,cflag,albedo,juldat,sza= input_data
    # albedo is included in the GEM validation data
    # juldat? is the day of year
    # sza is solar zenith angle in degrees

    with open(filename,'w', encoding="utf-8") as f:
        ###############  # record 1.1
        f.write(f'$ ATMOSPHERE PROFILE\n');
        ###############  # record 1.2
        iaer = 0
        iatm    = 0      # flag for RRTATM (0, 1->yes)
        iscat = 1;       # switch for DISORT or simple two-stream scattering (0-> DISORT (unavailable), 1->  two-stream (default))
        istrm = 0;       # flag for number of streams used in DISORT  (ISCAT must be equal to 0)
        iout    = 0      # -1 no output% 0 for 10-3250% n from band n% 98 17 spectral bands
        imca    = cflag
        icld    = cflag  #(0,1,2) flag for clouds
        idelm = 0;       # flag for outputting downwelling fluxes computed using the delta-M scaling approximation
        icos = 0;        # no account for instrumental cosine response (default)

        f.write(f'{iaer:20d}{iatm:30d}{iscat:33d}{istrm:2d}{iout:5d}{imca:4d}{icld:1d}{idelm:4d}{icos:1d}\n')

        ###############  # record 1.2.1

        f.write(f'{juldat:15d}{sza:10.4f}\n')
        ###############  # record 1.4

        iemis = 1        # surface emissivity (0-1.0,1-same emissivity = semis(1), 2-different)
        ireflect = 0     # 0 for Lambertian reflection at surface, i.e. reflected radiance
                         # is equal at all angles
        semis = 1. - albedo     # the surface emissivity for each band (see Table I).

        f.write(f'{iemis:12d}{ireflect:3d}{semis:5.3f}\n')

        ###############  # record 2.1
        iform = 0        # column amount format flag> 0: read PAVE, WKL(M,L), WBROADL(L) in F10.4, E10.3, E10.3 formats (default)
        f.write(f'{iform:2d}{n_level:3d}{nmol:5d}\n')                                                   # record 2.1
        ## surface setting
        f.write(f'{plev[0]:10.4f}{ta[0]:10.4f}{plev_bound[0]:31.3f}' +
              f'{ta_bound[0]:7.2f}{plev_bound[1]:15.3f}{ta_bound[1]:7.2f}\n')                           # record 2.1.1
        f.write(''.join([f'{gas_mr[i,0]:10.3e}' for i in range(0, nmol+1)])+'\n')                       # record 2.1.2
        ## atmospheric   profile
        for k in range(1, n_level):                                                                     # record 2.1.1&2
            f.write(f'{plev[k]:10.4f}{ta[k]:10.4f}{plev_bound[k+1]:53.3f}{ta_bound[k+1]:7.2f}\n')
            f.write(''.join([f'{gas_mr[i,k]:10.3e}' for i in range(nmol+1)])+'\n')
        f.write(f'$ %%%%%');
        f.close()

def write_cloud_input(filename, input_data):
    ### Treat cloud as in the mean bounded by momemtum levels
    ### input pressure: momentum levels
    cc, clwc, ciwc,  rle, rie, p_lev = input_data
    # cc - cloud cover, fraction 0.0-1.0, direct output from model
    # clwc - cloud liquid water content, kg/m3
    # ciwc - cloud ice water content, kg/m3
    # effradliq, micron
    # effradice, micron
    # cwp, g/m2
    with open(filename,'w', encoding="utf-8") as f:
        ###############  # record C1.1
        inflag = 2; # seperate ice and liquid cloud optical depth
        iceflag = 2; # optical depth (non-grey)
        liqflag = 1; # optical depth (non-grey)
        f.write(f'{inflag:5d}{iceflag:5d}{liqflag:5d}\n')
        ###############  # record C1.2
        idxc = np.where(np.logical_and(cc>0,np.logical_or(clwc>0,ciwc>0)))[0]
        lay = idxc;
        n = len(idxc);
        cldfrac = cc[idxc];
        fracice = ciwc[idxc]/(ciwc[idxc]+clwc[idxc]);
        effradliq = rle[idxc];
        effradice = rie[idxc];
        ## (INFLAG = 2 and ICEFLAG = 2)
        ## ice: Valid sizes are 5.0 - 131.0 microns
        ## (INFLAG = 2 and LIQFLAG = 1)
        ## liquid: Valid sizes are 2.5 - 60.0 microns.
        #effradice[np.logical_and(effradice < 5,effradice>0)] = 5;
        effradice[effradice < 5] = 5;
        effradice[effradice > 131]= 131;
        effradliq[effradliq < 2.5] = 2.5;
        #effradliq[np.logical_and(effradliq < 2.5,effradliq>0)] = 2.5;
        effradliq[effradliq > 60] = 60;
        dp = abs(p_lev[:-1]- p_lev[1:]);
        g = 9.8;
        cwp = 1e2*1e3*(ciwc[idxc]+clwc[idxc])*dp[idxc]/g/(1+ciwc[idxc]+clwc[idxc]);
        # -dp/g = rhou*dz, dp has unit hPa -> *1e2 convert to Pa
        # (1+ciwc[idxc]+clwc[idxc]), 1 is the density of moist air, has unit kg/m3
        # Therefore ciwc and clwc also have units kg/m3
        for i in range(n):
            #print(cwp[i])
            f.write(f'{lay[i]:5d}{cldfrac[i]:10.5f}{cwp[i]:10.5f}{fracice[i]:10.5f}{effradice[i]:10.5f}{effradliq[i]:10.5f}\n')

        f.write(f'%\n');
        f.close()



def get_standard_atmo(plevmin=10, pavemin=13.4):
    #### Read standard atm data from RRTMG MLS for levels above GEM mode top
    indir = '/storage/xwang/transfer_data/data/result_data/rrtm_lw_output_dir/run_examples_std_atm/'
    #indir = '/lustre03/project/6003571/xunwang/RRTMG_LW-master/run_examples_std_atm/'
    f = open(indir+'input_rrtm_MLS-clr')
    output = f.readlines()
    top_line = 7
    nrow=102
    n_lev = int(nrow/2)
    output_split = ([_.split() for _ in output[top_line:top_line+nrow]])
    std_pr_ave = np.zeros(n_lev)
    std_tt_ave = np.zeros(n_lev)
    std_pr_ul = np.zeros(n_lev)
    std_tt_ul = np.zeros(n_lev)
    std_hua = np.zeros(n_lev)
    std_o3a = np.zeros(n_lev)
    k1 = 0; k2 =0

    for i in range(0,n_lev,1):
        std_pr_ave[i]=float(output_split[int(i*2)][0])
        std_tt_ave[i]=float(output_split[int(i*2)][1])
        if len(output_split[int(i*2)][-2])>8:
            std_pr_ul[i]=float(output_split[int(i*2)][-2][-8:])
        else:
            std_pr_ul[i]=float(output_split[int(i*2)][-2])
        std_tt_ul[i]=float(output_split[int(i*2)][-1])

        std_hua[i] = float(output_split[int(i*2+1)][0])
        std_o3a[i] = float(output_split[int(i*2+1)][2])
    f.close()
    iabove_gem = np.where(std_pr_ul<plevmin)[0][0]
    iabove_gema = np.where(std_pr_ave<pavemin)[0][0]
    naddlayer = np.size(std_pr_ul[iabove_gem:])
    # the standard atmosphere o3 is already mol/mol, don't multiply by 1e-6
    #### read insitu o3 data (for test purpose)
    return std_pr_ave, std_tt_ave, std_pr_ul, std_tt_ul, std_hua, std_o3a,iabove_gem,iabove_gema,naddlayer

def form_clr_data(model_atm_data,attach_std_atmo=0):
    P_all,T_all,HU_all,PXM_all,PS,TS = model_atm_data
    #std_tt_ul,std_pr_ul,std_tt_ave,std_pr_ave,std_hua,iabove_gem,iabove_gema = std_atm_data
    if (P_all[1:]-P_all[:-1])[-1]>0:
        TT_tm = T_all[::-1]*1.
        PP_tm = P_all[::-1]*1.
        HU_ave = HU_all[::-1]*1.
    elif (P_all[1:]-P_all[:-1])[-1]<0:
        TT_tm = T_all*1.
        PP_tm = P_all*1.
        HU_ave = HU_all*1.
    if (PXM_all[1:]-PXM_all[:-1])[-1]>0:
        PP_mm = PXM_all[::-1]*1.
    elif (PXM_all[1:]-PXM_all[:-1])[-1]<0:
        PP_mm = PXM_all*1.
    fft = interpolate.interp1d(np.log(PP_tm),TT_tm,fill_value='extrapolate')
    TT_mm = fft(np.log(PP_mm))
    P_lev = np.append(PS,PP_mm)
    T_lev = np.append(TS,TT_mm)
    T_ave = TT_tm*1.
    P_ave = PP_tm*1.
    std_pr_ave, std_tt_ave, std_pr_ul, std_tt_ul, std_hua, std_o3a,iabove_gem,iabove_gema,naddlayer=get_standard_atmo(P_ave.min(),P_ave.min())
    # attach std atm data to GEM data
    if attach_std_atmo == 1:
        T_lev = np.append(T_lev,std_tt_ul[iabove_gem:])
        P_lev = np.append(P_lev,std_pr_ul[iabove_gem:])
        T_ave = np.append(T_ave,std_tt_ave[iabove_gema:])
        P_ave = np.append(P_ave,std_pr_ave[iabove_gema:])
        HU_ave = np.append(HU_ave,std_hua[iabove_gema:])
    else:
        naddlayer = 0
    ffo = interpolate.interp1d(np.log(std_pr_ave),std_o3a,fill_value='extrapolate')
    gas_mr_o3 = ffo(np.log(P_ave))

    return T_lev,P_lev,T_ave,P_ave,HU_ave,gas_mr_o3,naddlayer

def form_cld_data(model_cld_data,attach_std_atmo=0):
    iwc,lwc,effradice,effradliq,cloudfrac,P_all,naddlayer = model_cld_data
    if (P_all[1:]-P_all[:-1])[-1]>0:
        iwc = iwc[::-1]*1.
        lwc = lwc[::-1]*1.
        effradice = effradice[::-1]*1.
        effradliq = effradliq[::-1]*1.
        cloudfrac = cloudfrac[::-1]*1.
    else:
        iwc = iwc*1.
        lwc = lwc*1.
        effradice = effradice*1.
        effradliq = effradliq*1.
        cloudfrac = cloudfrac*1.
    if attach_std_atmo==1:
        iwc_tm = np.append(iwc,np.zeros(naddlayer))
        lwc_tm = np.append(lwc,np.zeros(naddlayer))
        eri_tm = np.append(effradice,np.zeros(naddlayer))
        erl_tm = np.append(effradliq,np.zeros(naddlayer))
        cloudfrac = np.append(cloudfrac,np.zeros(naddlayer))
    else:
        iwc_tm = iwc*1.
        lwc_tm = lwc*1.
        eri_tm = effradice*1.
        erl_tm = effradliq*1.
        cloudfrac = cloudfrac*1.
    return iwc_tm, lwc_tm, eri_tm, erl_tm, cloudfrac

def convert_wv_unit(t,p,var,case):
    # p: pressure in unit hPa
    # t: temperature in unit K
    # var is input water vapor
    if case == 'g/m3->mol/mol':
        Rs = 287.
        de = p*100./(Rs*t)
        var = var/de/1000.*28.960/18.016
        return var
    if case == 'kg/kg->mol/mol':
        var = var*28.960/18.016
        return var

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0

def get_declination(ymdhm):
    """
    This script calculates declination angle delta in radians
    juldat is a day number (1-Jan. 1, 365-Dec. 31 (366 for leap year))
    """
    yr,mon,day,hr,mn = ymdhm
    dt = datetime(yr,mon,day,hr,mn)
    juldat = dt.timetuple().tm_yday
    if is_leap_year(yr):
        theta_d = 2.*np.pi/366.*(juldat-1 + (hr-12)/24.)
    else:
        theta_d = 2.*np.pi/365.*(juldat-1 + (hr-12)/24.)
    a = [0.006918,-0.399912,-0.006758,-0.002697]
    b = [0, 0.070257, 0.000907, 0.001480];
    delta = 0
    for i in range(4):
        delta += a[i]*np.cos(i*theta_d)+b[i]*np.sin(i*theta_d)
    return delta

def get_hour_angle(lon,ymdhm):
    """
     This script calculates the solar hour angle in radians
     lon is longitude (-180,180)
     juldat is a day number (1-Jan. 1, 365-Dec. 31 (366 for leap year))
    """
    yr,mon,day,hr,mn = ymdhm
    if lon>180:
        lon-=360.
    dt = datetime(yr,mon,day,hr,mn)
    juldat = dt.timetuple().tm_yday
    if is_leap_year(yr):
        theta_d = 2.*np.pi/366.*(juldat-1 + (hr-12)/24.)
    else:
        theta_d = 2.*np.pi/365.*(juldat-1 + (hr-12)/24.)
    a = [0.000075, 0.001868,-0.014615]
    b = [0,-0.032077,-0.040849]
    eqtime = 0
    for i in range(3):
        eqtime+= 229.18*(a[i]*np.cos(i*theta_d)+b[i]*np.sin(i*theta_d))
    time_offset = eqtime+4*lon;
    #tst = hr*60.+mn+sc/60+time_offset
    tst = hr*60.+mn+time_offset # do not have second information
    hour_angle = np.radians(tst/4.-180.)
    return hour_angle

def get_sza(lat,lon,ymdhm):
    """
    ymdhm is date [yr,mon,day,hr,mn]
    This script calculates the cosine solar zenith angle sza in degrees
    where hr is the hour (0-23), mn is the minute (0-59), sc is the second (0-59)
    timezone is in hours from UTC
    """
    if lon>180:
        lon-=360.
    lat = np.radians(lat)
    ha = get_hour_angle(lon,ymdhm)
    delta = get_declination(ymdhm);
    sza = np.arccos(np.sin(lat)*np.sin(delta) + np.cos(lat)*np.cos(delta)*np.cos(ha))
    sza = sza/np.pi*180
    if sza>90:
        sza = 90
    return sza

def mean_sun_earth_dd(ymdhm):
    yr,mon,day,hr,mn = ymdhm
    dt = datetime(yr,mon,day,hr,mn)
    juldat = dt.timetuple().tm_yday
    if is_leap_year(yr):
        theta_d = 2.*np.pi/366.*(juldat-1 + (hr-12)/24.)
    else:
        theta_d = 2.*np.pi/365.*(juldat-1 + (hr-12)/24.)
    a = [1.000110,0.034221,0.000719]
    b = [0,0.001280,0.000077]
    dd = 0
    for i in range(3):
        dd += a[i]*np.cos(i*theta_d)+b[i]*np.sin(i*theta_d)
    return dd


# S0 = 1360.
# dd = mean_sun_earth_dd([2021,7,5,12,12])
# TISR = S0*dd*np.cos(np.radians(sza2))

def read_rrtm_lw_data(file_dir,nlev):
    with  open(file_dir+'/OUTPUT_RRTM','r') as f_output:
        nvar = 6
        top_line = 3
        for i in range(top_line):
            line=f_output.readline()
        data = f_output.readlines()
        data = data[:nlev+1]
        data = ''.join(data).replace('\n',' ')
        data = " ".join(data.split())
        data = data.split(' ')
        # data = np.array([float(i) for i in data])
        # data = np.reshape(data,(nlev+1,nvar)).T
        ndat = np.shape(data)[0]
        data2 = np.zeros(ndat)
        for i in range(ndat):
            if '*' not in data[i]:
                data2[i] = float(data[i])
            else:
                data2[i] = 9999
        data = np.reshape(data2,(nlev+1,nvar)).T
        return data

def read_rrtm_sw_data(file_dir,nlev):
    with  open(file_dir+'/OUTPUT_RRTM','r') as f_output:
        nvar = 8
        top_line = 5
        for i in range(top_line):
            line=f_output.readline()
        data = f_output.readlines()
        data = data[:nlev+1]
        data = ''.join(data).replace('\n',' ')
        data = " ".join(data.split())
        data = data.split(' ')
        # data = np.array([float(i) for i in data])
        # data = np.reshape(data,(nlev+1,nvar)).T
        ndat = np.shape(data)[0]
        data2 = np.zeros(ndat)
        for i in range(ndat):
            if '*' not in data[i]:
                data2[i] = float(data[i])
            else:
                data2[i] = 9999
        data = np.reshape(data2,(nlev+1,nvar)).T
        return data

def read_rrtm_sw_band_data(file_dir,nlev):
    with  open(file_dir+'/OUTPUT_RRTM','r') as f_output:
        nvar = 8
        top_line = 5
        nband=15
        data = np.zeros((nband,nvar,nlev+1))
        for i in range(top_line):
            line=f_output.readline()
        data_all = f_output.readlines()
        for i in range(15):
            dt = data_all[i*(nlev+1+6):(i+1)*(nlev+1+6)-6]
            dt = ''.join(dt).replace('\n',' ')
            dt = " ".join(dt.split())
            dt = dt.split(' ')
            dt = np.array([float(i) for i in dt])
            dt = np.reshape(dt,(nlev+1,nvar)).T
            data[i,:]=dt*1.
    return data

def save_rrtmg_retrieved_data(outdir,filename,k_bnd,lat,lon,prdata,ufluxdata,dfluxdata,nfluxdata,hrdata):
    nlat = len(lat)
    nlon = len(lon)
    fout = Dataset(outdir+filename,'w',format = 'NETCDF4')
    Level=fout.createDimension('level',k_bnd)
    Lat=fout.createDimension('lat',nlat)
    Lon=fout.createDimension('lon',nlon)

    outdata = fout.createVariable("pressure",'f4',("level","lat","lon",),zlib=True)
    outdata.units='hPa'
    outdata2 = fout.createVariable("uflux",'f4',("level","lat","lon",),zlib=True)
    outdata2.units='W m-2'
    outdata3 = fout.createVariable("dflux",'f4',("level","lat","lon",),zlib=True)
    outdata3.units='W m-2'
    outdata4 = fout.createVariable("net_flux",'f4',("level","lat","lon",),zlib=True)
    outdata4.units='W m-2'
    outdata5 = fout.createVariable("heating_rate",'f4',("level","lat","lon",),zlib=True)
    outdata5.units='K/day'
    Lon=fout.createVariable('lon','f4',('lon',),zlib=True)
    Lon.units="degree"
    Lon.long_name="Longitude"
    Lat=fout.createVariable('lat','f4',('lat',),zlib=True)
    Lat.units="degree"
    Lat.long_name="Latitude"
    Level=fout.createVariable('level','f4',('level',),zlib=True)
    outdata[:]=prdata
    outdata2[:]=ufluxdata
    outdata3[:]=dfluxdata
    outdata4[:]=nfluxdata
    outdata5[:]=hrdata
    Lon[:]=lon
    Lat[:]=lat
    Level[:]=np.arange(0,k_bnd,1)
    fout.close()

def save_rrtmg_retrieved_signle_col_data(outdir,filename,k_bnd,prdata,ufluxdata,dfluxdata,nfluxdata,hrdata):
    fout = Dataset(outdir+filename,'w',format = 'NETCDF4')
    Level=fout.createDimension('level',k_bnd)


    outdata = fout.createVariable("pressure",'f4',("level",),zlib=True)
    outdata.units='hPa'
    outdata2 = fout.createVariable("uflux",'f4',("level",),zlib=True)
    outdata2.units='W m-2'
    outdata3 = fout.createVariable("dflux",'f4',("level",),zlib=True)
    outdata3.units='W m-2'
    outdata4 = fout.createVariable("net_flux",'f4',("level",),zlib=True)
    outdata4.units='W m-2'
    outdata5 = fout.createVariable("heating_rate",'f4',("level",),zlib=True)
    outdata5.units='K/day'
    Level=fout.createVariable('level','f4',('level',),zlib=True)
    outdata[:]=prdata
    outdata2[:]=ufluxdata
    outdata3[:]=dfluxdata
    outdata4[:]=nfluxdata
    outdata5[:]=hrdata
    Level[:]=np.arange(0,k_bnd,1)
    fout.close()

def save_rrtmg_retrieved_indi_prof_data(outdir,filename,k_bnd,prdata,ufluxdata,dfluxdata,nfluxdata,hrdata,nprof):
    fout = Dataset(outdir+filename,'w',format = 'NETCDF4')
    Level=fout.createDimension('level',k_bnd)
    nn = fout.createDimension('nprof',nprof)

    outdata = fout.createVariable("pressure",'f4',("nprof","level",),zlib=True)
    outdata.units='hPa'
    outdata2 = fout.createVariable("uflux",'f4',("nprof","level",),zlib=True)
    outdata2.units='W m-2'
    outdata3 = fout.createVariable("dflux",'f4',("nprof","level",),zlib=True)
    outdata3.units='W m-2'
    outdata4 = fout.createVariable("net_flux",'f4',("nprof","level",),zlib=True)
    outdata4.units='W m-2'
    outdata5 = fout.createVariable("heating_rate",'f4',("nprof","level",),zlib=True)
    outdata5.units='K/day'
    Level=fout.createVariable('level','f4',('level',),zlib=True)
    nn=fout.createVariable('nprof','f4',('nprof',),zlib=True)
    outdata[:]=prdata
    outdata2[:]=ufluxdata
    outdata3[:]=dfluxdata
    outdata4[:]=nfluxdata
    outdata5[:]=hrdata
    Level[:]=np.arange(0,k_bnd,1)
    nn[:]=np.arange(0,nprof,1)
    fout.close()


# def column_number_density(pressure_bound, volume_mixing_ratio=1):
#     """Calculates the number density using the ideal gas law.
#         p V = N kb T
#     Args:
#         temperature: Temperature [K].
#         pressure: Pressure [hPa].
#         volume_mixing_ratio: Volume-mixing ratio [mol mol-1].
#
#     Returns:
#         Column_Number density [mol cm-2].
#     """
# #     N_A = 6.02214e23   # Avogadro constant [mol-1]
# # #     k_B = 1.380649e-23 # Boltzmann constant [J/K]
# #     M_mu = 0.0289652   # dry air density
# #     g = 9.8            # gravity
# #     constant = N_A/M_mu/g/100
#     constant = 2.1215254223435468e+22
#     cnd = abs(pressure_bound[:-1]-pressure_bound[1:])*volume_mixing_ratio*constant
#     return cnd


# def writh_input_rrtm_lw_clr(filename, input_data):
#     """ Create input for rrtm_lw v3.3
#         Please see detailed instruction in rrtm_lw/rrtm_instructions
#         Output file name: INPUT_RRTM
#     """
#     n_level, plev_bound, ta_bound, its, nmol, gas_mr = input_data
#
#     plev = (plev_bound[1:]+plev_bound[:-1])/2 ## this is layer average pressure
#
#     if np.all(np.diff(plev) > 0):
#         # change z to bot_up
#         ta = np.interp(np.log(plev),np.log(plev_bound),ta_bound)
#         # reverse order
#         ta = ta[::-1]
#         plev = plev[::-1]
#         plev_bound = plev_bound[::-1]
#         ta_bound = ta_bound[::-1]
#         gas_mr= gas_mr[:,::-1]
#     elif np.all(np.diff(plev[::-1]) > 0):
#         # good to go, but interp ta on top_down and reverse it
#         ta = np.interp(np.log(plev[::-1]),np.log(plev_bound[::-1]),ta_bound[::-1]) # get the layer average ta
#         ta = ta[::-1]
#     else:
#         raise Exception('plev not monotonic')
#
#     with open(filename,'w', encoding="utf-8") as f:
#
#         ###############  # record 1.1
#         f.write(f'$ ATMOSPHERE PROFILE\n');
#         ###############  # record 1.2
#         cflag=0;
#         iaer = 0
#         iatm    = 0      # flag for RRTATM (0, 1->yes)
#         ixsect  = 0      # flag for cross-sections | 0: no cross-sections included in calculation
#         numangs = 0      # (0,1,2,3,4) flag for number of quadrature angles to use in RT
#         iout    = 0      # -1 no output% 0 for 10-3250% n from band n% 99 17 spectral bands
#         idrv    = 0
#         imca    = cflag
#         icld    = cflag  #(0,1,2) flag for clouds
#
#         f.write(f'{iaer:20d}{iatm:30d}{ixsect:20d}{numangs:15d}{iout:5d}{idrv:2d}{imca:2d}{icld:1d}\n')
#
#         ###############  # record 1.4
#
#         tbound = its     # temperature at the surface.  If input value < 0, then surface temperature
#                          # is set to the temperature at the bottom of the first layer (see Record 2.1.1
#         iemis = 1        # surface emissivity (0-1.0,1-same emissivity = semis(1), 2-different)
#         ireflect = 0     # 0 for Lambertian reflection at surface, i.e. reflected radiance
#                          # is equal at all angles
#         semis = 0.96     # the surface emissivity for each band (see Table I).
#
#         f.write(f'{tbound:10.3f}{iemis:2d}{ireflect:3d}{semis:5.3f}\n')
#
#
#
#         ###############  # record 2.1
#         iform = 0        # column amount format flag> 0: read PAVE, WKL(M,L), WBROADL(L) in F10.4, E10.3, E10.3 formats (default)
#         f.write(f'{iform:2d}{n_level:3d}{nmol:5d}\n')                                                   # record 2.1
#         ## surface setting
#         f.write(f'{plev[0]:10.4f}{ta[0]:10.4f}{plev_bound[0]:31.3f}' +
#               f'{ta_bound[0]:7.2f}{plev_bound[1]:15.3f}{ta_bound[1]:7.2f}\n')                           # record 2.1.1
#         f.write(''.join([f'{gas_mr[i,0]:10.3e}' for i in range(0, nmol+1)])+'\n')                       # record 2.1.2
#         ## atmospheric   profile
#         for k in range(1, n_level):                                                                     # record 2.1.1&2
#             f.write(f'{plev[k]:10.4f}{ta[k]:10.4f}{plev_bound[k+1]:53.3f}{ta_bound[k+1]:7.2f}\n')
#             f.write(''.join([f'{gas_mr[i,k]:10.3e}' for i in range(nmol+1)])+'\n')
#         f.write(f'$ %%%%%');
#         f.close()
