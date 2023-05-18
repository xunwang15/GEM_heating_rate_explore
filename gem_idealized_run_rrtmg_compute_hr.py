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

def compute_HR(swd,swu,lwd,lwu,plev,pave):
    g = 9.8
    Cp = 1003.5
    swn = (swd-swu)
    lwn = (lwu-lwd)
    # Rs = 287.
    # rhou = plev/(Rs*tlev)
    # rhou*dz = -dp/g
    # HR = -d(swn+lwn)/(Cp*rhou*dz)
    dswn = -(swn[1:]-swn[:-1])
    dlwn = lwn[1:]-lwn[:-1]
    dp = (plev[1:]-plev[:-1])*100.
    sHR = np.append(dswn/(Cp*dp/g),0)*60*60*24
    lHR = np.append(dlwn/(Cp*dp/g),0)*60*60*24
    return sHR,lHR

def compute_HR_cmatrix(swd,swu,lwd,lwu,lwc,swc):
    swn = (swd-swu)
    lwn = (lwu-lwd)
    dswn = -(swn[1:]-swn[:-1])
    dlwn = lwn[1:]-lwn[:-1]
    sHR = np.append(0,dswn/swc)
    lHR = np.append(0,dlwn/lwc)
    return sHR,lHR

def compute_cmatrix(swd,swu,lwd,lwu,lwhr,swhr):
    swn = (swd-swu)
    lwn = (lwu-lwd)
    dswn = -(swn[1:]-swn[:-1])
    dlwn = lwn[1:]-lwn[:-1]
    clw = dlwn/lwhr[1:]
    csw = dswn/swhr[1:]
    return csw,clw

starttime = '2013-08-25 12:01:00'
ntime_all = int(12*60)
timeseries = np.arange(1,721,1)
datatime = pd.date_range(starttime, periods=ntime_all, freq="1min")
itime=599
juldat =datetime(2013,8,datatime[itime].day,datatime[itime].hour,0).timetuple().tm_yday
nmol = 7
nco2 = 355e-6#284e-6                 # CO2  [mol/mol]   284 ppmv
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
out_dir = '/storage/xwang/transfer_data/data/result_data/rrtm_lw_output_dir/gem_validate/'
out_dir_sw = '/storage/xwang/transfer_data/data/result_data/rrtm_sw_output_dir/gem_validate/'
rrtmg_lw_dir = '/aos/home/xwang/rrtmg_lw_v4.85/column_model/build/rrtmg_lw_v4.85_linux_pgi'
rrtmg_sw_dir = '/aos/home/xwang/rrtmg_sw_v4.02/column_model/build/rrtmg_sw_v4.02_linux_pgi'
outdir = '/storage/xwang/transfer_data/data/result_data/rrtmg_retrieved_data/'

# indir = '/lustre03/project/6003571/xunwang/data/result_data/'
# out_dir = '/lustre03/project/6003571/xunwang/data/result_data/rrtm_lw_output_dir/gem_ideal1/'
# out_dir_sw = '/lustre03/project/6003571/xunwang/data/result_data/rrtm_sw_output_dir/gem_ideal1/'
# rrtmg_lw_dir = '/lustre03/project/6003571/xunwang/RRTMG_LW-master/build/rrtmg_lw_v5.00_linux_pgi'
# rrtmg_sw_dir = '/lustre03/project/6003571/xunwang/RRTMG_SW-5.0/build/rrtmg_sw_v5.00_linux_pgi'
# outdir = '/lustre03/project/6003571/xunwang/data/result_data/rrtmg_retrieved_data/'

cld_run='clr' # choices of cld or clr, cld_a,cld_b
perturb = 'both' # choices of wv vs t both
tud = 'up_dw_comp'
#ntype,npert = (3,18) # for layer by layer: perturb and both
#ntype,npert = (3,5) # for gem: perturb and both
#ntype,npert = (2,10) # for slope at 100 and 92 hPa: perturb and both
#ntype,npert = (3,10) # for slope at multi levels: perturb and both
ntype,npert = (3,242)

#f2 = xr.open_dataset(indir+'idealized_gem_profiles_cloud_data_radius_center_egde.nc')
f2 = xr.open_dataset(indir+'idealized_gem_profiles_cloud_data_above_150hPa.nc')
if cld_run == 'cld_a':
    f2 = xr.open_dataset(indir+'idealized_gem_profiles_cloud_data_above_150hPa.nc')
if cld_run == 'cld_b':
    f2 = xr.open_dataset(indir+'idealized_gem_profiles_cloud_data_below_150hPa.nc')
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
    fT_lev_up = f1.T_lev.data
    fT_lev_dw = f1.T_lev.data
    fP_lev = f1.P_lev.data
    nlayer = np.shape(fT_ave)[2]

if npert == 18:
    foutname = 'idealized_gem_profiles_radius_center_egde_levbylev_perturb_'+perturb+'_'+tud+'_'+cld_run+'_rrtmg_out.nc'
    #foutname = 'idealized_gem_profiles_radius_center_egde_levbylev_perturb_'+perturb+'2_'+tud+'_'+cld_run+'_rrtmg_out.nc'
    f1 = xr.open_dataset(indir+'idealized_gem_profiles_p_t_wv_perturb_base_levbyevl_radius_center_egde.nc')
    fgas_mr_wv = f1.gas_mr_wv.data
    fgas_mr_o3 = f1.gas_mr_o3.data[:,np.newaxis,:]*np.ones_like(fgas_mr_wv)
    fT_ave = f1.T_ave.data
    fP_ave = f1.P_ave.data[:,np.newaxis,:]*np.ones_like(fgas_mr_wv)
    fT_lev_up = f1.T_lev_up.data
    fT_lev_dw = f1.T_lev_dw.data
    fP_lev = f1.P_lev.data[:,np.newaxis,:]*np.ones_like(fT_lev_up)
    nlayer = np.shape(fT_ave)[2]

if ntype==2 and npert == 10:
    foutname = 'idealized_gem_profiles_p_t_wv_slope_perturb_base_100hPa_92hPa_'+perturb+'_'+tud+'_'+cld_run+'_rrtmg_out.nc'
    f1 = xr.open_dataset(indir+'idealized_gem_profiles_p_t_wv_slope_perturb_base_100hPa_92hPa.nc')
    fgas_mr_wv = f1.gas_mr_wv.data
    fgas_mr_o3 = f1.gas_mr_o3.data
    fT_ave = f1.T_ave.data
    fP_ave = f1.P_ave.data
    fT_lev_up = f1.T_lev_up.data
    fT_lev_dw = f1.T_lev_dw.data
    fP_lev = f1.P_lev.data
    nlayer = np.shape(fT_ave)[2]


if ntype==3 and npert == 10:
    foutname = 'idealized_gem_profiles_p_t_wv_slope_perturb_base_'+perturb+'_'+tud+'_'+cld_run+'_rrtmg_out.nc'
    f1 = xr.open_dataset(indir+'idealized_gem_profiles_p_t_wv_slope_perturb_base.nc')
    fgas_mr_wv = f1.gas_mr_wv.data
    fgas_mr_o3 = f1.gas_mr_o3.data
    fT_ave = f1.T_ave.data
    fP_ave = f1.P_ave.data
    fT_lev_up = f1.T_lev_up.data
    fT_lev_dw = f1.T_lev_dw.data
    fP_lev = f1.P_lev.data
    nlayer = np.shape(fT_ave)[2]

if ntype==3 and npert == 242:
    foutname = 'idealized_gem_profiles_p_t_wv_scatter_perturb_base_'+perturb+'_'+tud+'_'+cld_run+'_rrtmg_out.nc'
    f1 = xr.open_dataset(indir+'idealized_gem_profiles_p_t_wv_scatter_perturb_base.nc')
    fgas_mr_wv = f1.gas_mr_wv.data
    fgas_mr_o3 = f1.gas_mr_o3.data
    fT_ave = f1.T_ave.data
    fP_ave = f1.P_ave.data
    fT_lev_up = f1.T_lev_up.data
    fT_lev_dw = f1.T_lev_dw.data
    fP_lev = f1.P_lev.data
    nlayer = np.shape(fT_ave)[2]

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

chr_lw = np.zeros((n1,nlayer+1))
chr_sw = np.zeros((n1,nlayer+1))
csw = np.zeros((n1,nlayer))
clw = np.zeros((n1,nlayer))
for i in range(n1):
    for j in range(n2):
        if perturb == 'wv':
            gas_mr_wv = fgas_mr_wv[i,j,:]*1.
            T_lev_up = fT_lev_up[i,0,:]*1.
            T_lev_dw = fT_lev_dw[i,0,:]*1.
            T_ave = fT_ave[i,0,:]*1.
            # if j<n2-1 and j>0:# the perturb two layers test
            #     k=np.where(fgas_mr_wv[i,j+1,:]-fgas_mr_wv[i,0,:]!=0)[0][0]
            #     gas_mr_wv[k] = fgas_mr_wv[i,j+1,k]*1.

        if perturb == 't':
            gas_mr_wv = fgas_mr_wv[i,0,:]*1.
            T_lev_up = fT_lev_up[i,j,:]*1.
            T_lev_dw = fT_lev_dw[i,j,:]*1.
            T_ave = fT_ave[i,j,:]*1.
            # if j<n2-1 and j>0:# the perturb two layers test
            #     k=np.where(fT_lev_up[i,j+1,:]-fT_lev_up[i,0,:]!=0)[0][0]
            #     T_lev_up[k] = fT_lev_up[i,j+1,k]*1.
            #     k=np.where(fT_lev_dw[i,j+1,:]-fT_lev_dw[i,0,:]!=0)[0][0]
            #     T_lev_dw[k] = fT_lev_dw[i,j+1,k]*1.
            #     k=np.where(fT_ave[i,j+1,:]-fT_ave[i,0,:]!=0)[0][0]
            #     T_ave[k] = fT_ave[i,j+1,k]*1.

        if perturb == 'both': # works for both levbylev and gem
            gas_mr_wv = fgas_mr_wv[i,j,:]*1.
            T_lev_up = fT_lev_up[i,j,:]*1.
            T_lev_dw = fT_lev_dw[i,j,:]*1.
            T_ave = fT_ave[i,j,:]*1.


        #gas_mr_o3 = fgas_mr_o3[i,j,:]*1.
        P_lev = fP_lev[i,j,:]*1.
        P_ave = fP_ave[i,j,:]*1.

        std_pr_ave, std_tt_ave, std_pr_ul, std_tt_ul, std_hua, std_o3a,iabove_gem,iabove_gema,naddlayer=get_standard_atmo(P_ave.min(),P_ave.min())
        ffo = interpolate.interp1d(np.log(std_pr_ave),std_o3a,fill_value='extrapolate')
        gas_mr_o3 = ffo(np.log(P_ave))

        gas_mr = gas_mr_single_column(nmol, nlayer+1,  gas_mr_wv, gas_mr_o3, gas_mr_mixed, P_lev)

        if cld_run in ['cld','cld_a','cld_b']:
            cflag=1
            input_data = [clf, lwc, iwc, efr, efi, P_lev]
            write_cloud_input(out_dir+'IN_CLD_RRTM', input_data)
            write_cloud_input(out_dir_sw+'IN_CLD_RRTM', input_data)
        else:
            cflag=0;

        #############  first compute upward flux #############
        input_rrtm_lw_single = [nlayer, P_ave, T_ave,P_lev,T_lev_up, T_lev_up[0], nmol, gas_mr,cflag,ems]
        filename = out_dir+"INPUT_RRTM"
        write_input_rrtm_lw(filename, input_rrtm_lw_single)
        input_rrtm_sw_single = [nlayer, P_ave, T_ave,P_lev,T_lev_up, nmol,  gas_mr,cflag,alb,juldat,sza]
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
        #### read rrtmg sw out data
        out_pr_sw[i,j,:] = read_rrtm_sw_data(out_dir_sw,nlayer)[1,:]
        out_uflux_sw[i,j,:] = read_rrtm_sw_data(out_dir_sw,nlayer)[2,:]

        # read data for cmatrix computation
        out_uflux_sw1 = read_rrtm_sw_data(out_dir_sw,nlayer)[2,:]
        out_uflux1 = read_rrtm_lw_data(out_dir,nlayer)[2,:]
        out_dflux1 = read_rrtm_lw_data(out_dir,nlayer)[3,:]
        out_dflux_sw1 = read_rrtm_sw_data(out_dir_sw,nlayer)[5,:]
        hr_lw1= read_rrtm_lw_data(out_dir,nlayer)[5,:]
        hr_sw1= read_rrtm_sw_data(out_dir_sw,nlayer)[7,:]

        ############# second compute downward flux #############
        input_rrtm_lw_single = [nlayer, P_ave, T_ave,P_lev,T_lev_dw, T_lev_dw[0], nmol, gas_mr,cflag,ems]
        filename = out_dir+"INPUT_RRTM"
        write_input_rrtm_lw(filename, input_rrtm_lw_single)
        input_rrtm_sw_single = [nlayer, P_ave, T_ave,P_lev,T_lev_dw, nmol,  gas_mr,cflag,alb,juldat,sza]
        filename = out_dir_sw+"INPUT_RRTM"
        write_input_rrtm_sw(filename, input_rrtm_sw_single)

        #### run rrtmg
        bash_cmd2 = f"cd {out_dir}; {rrtmg_lw_dir}"
        tmp=subprocess.run([bash_cmd2], shell=True, capture_output=True)
        #### run rrtmg
        bash_cmd2 = f"cd {out_dir_sw}; {rrtmg_sw_dir}"
        tmp=subprocess.run([bash_cmd2], shell=True, capture_output=True)

        out_dflux[i,j,:] = read_rrtm_lw_data(out_dir,nlayer)[3,:]
        out_dflux_sw[i,j,:] = read_rrtm_sw_data(out_dir_sw,nlayer)[5,:]

        ############# third compute net flux and heating rate #############
        out_nflux[i,j,:] = out_uflux[i,j,:]-out_dflux[i,j,:]
        out_nflux_sw[i,j,:] = out_dflux_sw[i,j,:]-out_uflux_sw[i,j,:]

        # if j==0:
        #     chr_lw[i,:]= read_rrtm_lw_data(out_dir,nlayer)[5,:]
        #     chr_sw[i,:]= read_rrtm_sw_data(out_dir_sw,nlayer)[7,:]
        #     (csw[i,:],clw[i,:]) = compute_cmatrix(out_dflux_sw[i,j,:],out_uflux_sw[i,j,:],\
        #     out_dflux[i,j,:],out_uflux[i,j,:],chr_lw[i,:],chr_sw[i,:])


        # (out_hr_sw[i,j,:],out_hr[i,j,:])=\
        # compute_HR_cmatrix(out_dflux_sw[i,j,:],out_uflux_sw[i,j,:],out_dflux[i,j,:],out_uflux[i,j,:],clw[i,:],csw[i,:])

        (csw,clw) = compute_cmatrix(out_dflux_sw1,out_uflux_sw1,\
        out_dflux1,out_uflux1,hr_lw1,hr_sw1)
        (out_hr_sw[i,j,:],out_hr[i,j,:])=\
        compute_HR_cmatrix(out_dflux_sw[i,j,:],out_uflux_sw[i,j,:],out_dflux[i,j,:],out_uflux[i,j,:],clw,csw)

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
