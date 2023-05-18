import xarray as xr
from glob import glob
import pandas as pd
from scipy import stats
import subprocess
from multiprocessing import Pool
from rrtmg_lw_functions import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from netCDF4 import Dataset
from scipy import interpolate
import time
from datetime import datetime


starttime = '2013-08-25 12:01:00'
ntime_all = int(12*60)
timeseries = np.arange(1,721,1)
datatime = pd.date_range(starttime, periods=ntime_all, freq="1min")
itime=599
juldat =datetime(2013,8,datatime[itime].day,datatime[itime].hour,0).timetuple().tm_yday
nmol = 7
nco2 = 284e-6                 # CO2  [mol/mol]   284 ppmv
nn2o = 0.273e-6               # n2o  [mol/mol] 0.273 ppmv
nco =  0.14829e-6             # CO  [mol/mol]0.14829 ppmv
nch4 = 0.808e-6               # CH4  [mol/mol] 0.808 ppm
no2 = 0.209                  # O2   [mol/mol] 20.9%
gas_mr_mixed = np.empty((nmol-2),dtype=np.float32)              # input for record 2.1.2
# well mix gas
gas_mr_mixed[0] = nco2
gas_mr_mixed[1] = nn2o
gas_mr_mixed[2] = nco
gas_mr_mixed[3] = nch4
gas_mr_mixed[4] = no2

indir = '/storage/xwang/transfer_data/data/result_data/'
out_dir = '/storage/xwang/transfer_data/data/result_data/rrtm_lw_output_dir/gem_ideal2/'
out_dir_sw = '/storage/xwang/transfer_data/data/result_data/rrtm_sw_output_dir/gem_ideal2/'
rrtmg_lw_dir = '/aos/home/xwang/rrtmg_lw_v4.85/column_model/build/rrtmg_lw_v4.85_linux_pgi'
rrtmg_sw_dir = '/aos/home/xwang/rrtmg_sw_v4.02/column_model/build/rrtmg_sw_v4.02_linux_pgi'
outdir = '/storage/xwang/transfer_data/data/result_data/rrtmg_retrieved_data/'

# indir = '/lustre03/project/6003571/xunwang/data/result_data/'
# out_dir = '/lustre03/project/6003571/xunwang/data/result_data/rrtm_lw_output_dir/gem_ideal1/'
# out_dir_sw = '/lustre03/project/6003571/xunwang/data/result_data/rrtm_sw_output_dir/gem_ideal1/'
# rrtmg_lw_dir = '/lustre03/project/6003571/xunwang/RRTMG_LW-master/build/rrtmg_lw_v5.00_linux_pgi'
# rrtmg_sw_dir = '/lustre03/project/6003571/xunwang/RRTMG_SW-5.0/build/rrtmg_sw_v5.00_linux_pgi'
# outdir = '/lustre03/project/6003571/xunwang/data/result_data/rrtmg_retrieved_data/'

cld_run='cld' # choices of cld or clr
perturb = 'both' # choices of wv vs t both,
tud = 'up' # choices of up and dw (for layer by layer)
#ntype,npert = (3,18) # for layer by layer: perturb and both
ntype,npert = (3,5) # for gem: perturb and both

f2 = xr.open_dataset(indir+'idealized_gem_profiles_cloud_data_radius_center_egde.nc')
iwc = f2.iwc.data
lwc = f2.lwc.data
efi = f2.efi.data
efr = f2.efr.data
clf =f2.clf.data
ems = f2.ems.data[0]
alb=f2.alb.data[0]
fsza = f2.sza.data

if npert==5:
    foutname = 'idealized_gem_profiles_radius_center_egde_perturb_'+perturb+'_'+tud+'_'+cld_run+'_rrtmg_out.nc'
    f1 = xr.open_dataset(indir+'idealized_gem_profiles_p_t_wv_perturb_base_radius_center_egde.nc')
    fgas_mr_wv = f1.gas_mr_wv.data
    fgas_mr_o3 = f1.gas_mr_o3.data
    fT_ave = f1.T_ave.data
    fP_ave = f1.P_ave.data
    fT_lev = f1.T_lev.data
    fP_lev = f1.P_lev.data
    nlayer = np.shape(fT_ave)[2]

if npert == 18:
    foutname = 'idealized_gem_profiles_radius_center_egde_levbylev_perturb_'+perturb+'_'+tud+'_'+cld_run+'_rrtmg_out.nc'
    f1 = xr.open_dataset(indir+'idealized_gem_profiles_p_t_wv_perturb_base_levbyevl_radius_center_egde.nc')
    fgas_mr_wv = f1.gas_mr_wv.data
    fgas_mr_o3 = f1.gas_mr_o3.data[:,np.newaxis,:]*np.ones_like(fgas_mr_wv)
    fT_ave = f1.T_ave.data
    fP_ave = f1.P_ave.data[:,np.newaxis,:]*np.ones_like(fgas_mr_wv)
    fT_lev_up = f1.T_lev_up.data
    fT_lev_dw = f1.T_lev_dw.data
    fP_lev = f1.P_lev.data[:,np.newaxis,:]*np.ones_like(fT_lev_up)
    nlayer = np.shape(fT_ave)[2]
    if tud == 'up':
        fT_lev = fT_lev_up*1.
    if tud == 'dw':
        fT_lev = fT_lev_dw*1.

# sza = np.zeros((ntype,npert))
# sza[:,0]=fsza[1]*1.
# sza[:,1]=fsza[0]*1.
# sza[:,2]=fsza[1]*1.
# sza[:,3]=fsza[0]*1.
sza = np.nanmean(fsza)

n1 = int(ntype)
n2 = int(npert)
out_pr = np.zeros((n1,n2,nlayer+1))
out_uflux = np.zeros((n1,n2,nlayer+1))
out_dflux = np.zeros((n1,n2,nlayer+1))
out_nflux = np.zeros((n1,n2,nlayer+1))
out_hr = np.zeros((n1,n2,nlayer+1))
out_pr_sw = np.zeros((n1,n2,nlayer+1))
out_uflux_sw = np.zeros((n1,n2,nlayer+1))
out_dflux_sw = np.zeros((n1,n2,nlayer+1))
out_nflux_sw = np.zeros((n1,n2,nlayer+1))
out_hr_sw = np.zeros((n1,n2,nlayer+1))
for i in range(n1):
    for j in range(n2):
        if perturb == 'wv':
            gas_mr_wv = fgas_mr_wv[i,j,:]*1.
            T_lev = fT_lev[i,0,:]*1.
            T_ave = fT_ave[i,0,:]*1.
        if perturb == 't':
            gas_mr_wv = fgas_mr_wv[i,0,:]*1.
            T_lev = fT_lev[i,j,:]*1.
            #T_lev = fT_lev[i,0,:]*1.
            T_ave = fT_ave[i,j,:]*1.
        if perturb == 'both': # works for layer by layer and gem_ideal
            gas_mr_wv = fgas_mr_wv[i,j,:]*1.
            T_lev = fT_lev[i,j,:]*1.
            T_ave = fT_ave[i,j,:]*1.

        gas_mr_o3 = fgas_mr_o3[i,j,:]*1.
        P_lev = fP_lev[i,j,:]*1.
        P_ave = fP_ave[i,j,:]*1.
        gas_mr = gas_mr_single_column(nmol, nlayer+1,  gas_mr_wv, gas_mr_o3, gas_mr_mixed, P_lev)

        if cld_run=='cld':
            cflag=1
            input_data = [clf, lwc, iwc, efr, efi, P_lev]
            write_cloud_input(out_dir+'IN_CLD_RRTM', input_data)
            write_cloud_input(out_dir_sw+'IN_CLD_RRTM', input_data)
        else:
            cflag=0;

        input_rrtm_lw_single = [nlayer, P_ave, T_ave,P_lev,T_lev, T_lev[0], nmol, gas_mr,cflag,ems]
        filename = out_dir+"INPUT_RRTM"
        write_input_rrtm_lw(filename, input_rrtm_lw_single)
        input_rrtm_sw_single = [nlayer, P_ave, T_ave,P_lev,T_lev, nmol,  gas_mr,cflag,alb,juldat,sza]
        filename = out_dir_sw+"INPUT_RRTM"
        write_input_rrtm_sw(filename, input_rrtm_sw_single)

        #### run rrtmg
        bash_cmd2 = f"cd {out_dir}; {rrtmg_lw_dir}"
        tmp=subprocess.run([bash_cmd2], shell=True, capture_output=True)
        #### run rrtmg
        bash_cmd2 = f"cd {out_dir_sw}; {rrtmg_sw_dir}"
        tmp=subprocess.run([bash_cmd2], shell=True, capture_output=True)

        #### read rrtmg out data
        out_pr[i,j,:] = read_rrtm_lw_data(out_dir,nlayer)[1,:]
        out_uflux[i,j,:] = read_rrtm_lw_data(out_dir,nlayer)[2,:]
        out_dflux[i,j,:] = read_rrtm_lw_data(out_dir,nlayer)[3,:]
        out_nflux[i,j,:] = read_rrtm_lw_data(out_dir,nlayer)[4,:]
        out_hr[i,j,:]= read_rrtm_lw_data(out_dir,nlayer)[5,:]

        #### read rrtmg sw out data
        out_pr_sw[i,j,:] = read_rrtm_sw_data(out_dir_sw,nlayer)[1,:]
        out_uflux_sw[i,j,:] = read_rrtm_sw_data(out_dir_sw,nlayer)[2,:]
        out_dflux_sw[i,j,:] = read_rrtm_sw_data(out_dir_sw,nlayer)[5,:]
        out_nflux_sw[i,j,:] = read_rrtm_sw_data(out_dir_sw,nlayer)[6,:]
        out_hr_sw[i,j,:]= read_rrtm_sw_data(out_dir_sw,nlayer)[7,:]



fout = Dataset(outdir+foutname,'w',format = 'NETCDF4')
Level=fout.createDimension('level',78)
nn1 = fout.createDimension('n1',n1)
nn2 = fout.createDimension('n2',n2)
outdata = fout.createVariable("lw_pr",'f4',('n1','n2','level'),zlib=True)
outdata.units='hPa'
outdata1 = fout.createVariable("lw_uflux",'f4',('n1','n2','level'),zlib=True)
outdata1.units='W/m2'
outdata2 = fout.createVariable("lw_dflux",'f4',('n1','n2','level'),zlib=True)
outdata2.units='W/m2'
outdata3 = fout.createVariable("lw_nflux",'f4',('n1','n2','level'),zlib=True)
outdata3.units='W/m2'
outdata4 = fout.createVariable("lw_hr",'f4',('n1','n2','level'),zlib=True)
outdata4.units='K/s'

outdata5 = fout.createVariable("sw_pr",'f4',('n1','n2','level'),zlib=True)
outdata5.units='hPa'
outdata6 = fout.createVariable("sw_uflux",'f4',('n1','n2','level'),zlib=True)
outdata6.units='W/m2'
outdata7 = fout.createVariable("sw_dflux",'f4',('n1','n2','level'),zlib=True)
outdata7.units='W/m2'
outdata8 = fout.createVariable("sw_nflux",'f4',('n1','n2','level'),zlib=True)
outdata8.units='W/m2'
outdata9 = fout.createVariable("sw_hr",'f4',('n1','n2','level'),zlib=True)
outdata9.units='K/s'

Level=fout.createVariable("level",'f4',('level'),zlib=True)
nn1=fout.createVariable("n1",'f4',('n1'),zlib=True)
nn2=fout.createVariable("n2",'f4',('n2'),zlib=True)

outdata[:]=out_pr
outdata1[:]=out_uflux
outdata2[:]=out_dflux
outdata3[:]=out_nflux
outdata4[:]=out_hr

outdata5[:]=out_pr_sw
outdata6[:]=out_uflux_sw
outdata7[:]=out_dflux_sw
outdata8[:]=out_nflux_sw
outdata9[:]=out_hr_sw

Level[:]=np.arange(78)
nn1[:]=np.arange(n1)
nn2[:]=np.arange(n2)
fout.close()
