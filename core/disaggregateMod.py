"""
Module to provide disaggregation functionality.
"""
from pathlib import Path
from datetime import datetime, timedelta
import os.path

import numpy as np
from netCDF4 import Dataset
from strenum import StrEnum
from core import err_handler
from core.enumConfig import SuppForcingPcpEnum

test_enabled = True

def disaggregate_factory(ConfigOptions):
    if len(ConfigOptions.supp_precip_forcings) == 1 and ConfigOptions.supp_precip_forcings[0] == str(SuppForcingPcpEnum.AK_NWS_IV):
        return ext_ana_disaggregate
    #Add new cases here
    #elif condition:
    else:
        return no_op_disaggregate


def no_op_disaggregate(input_forcings, supplemental_precip, config_options, mpi_config):
    pass


def ext_ana_disaggregate(input_forcings, supplemental_precip, config_options, mpi_config):
    """
    Function for disaggregating 6hr SuppPcp data to 1hr Input data
    :param input_forcings:
    :param supplemental_precip:
    :param config_options:
    :param mpi_config:
    :return:
    """
    # Check to make sure we have valid grids.
    if input_forcings.regridded_forcings2 is None or supplemental_precip.regridded_precip2 is None:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Bypassing ext_ana_disaggregation routine due to missing input or supp pcp data"
            err_handler.log_warning(config_options, mpi_config)
        return
            
    if supplemental_precip.ext_ana != "STAGE4":
        #Just copy RAINRATE straight over into the output
        supplemental_precip.regridded_precip2[:,:] = input_forcings.regridded_forcings2[3,:,:]
        if mpi_config.rank == 0:
            config_options.statusMsg = f"Bypassing ext_ana_disaggregation routine due to supplemental_precip.ext_ana = {supplemental_precip.ext_ana}"
            err_handler.log_warning(config_options, mpi_config)
        return
    
    
    #print("ext_ana_disaggregate RAINRATE input_forcings.regridded_forcings2[3,:,:]")
    #print(input_forcings.regridded_forcings2[3,:,:])
    #print("ext_ana_disaggregate supplemental_precip.regridded_precip2[:,:]")
    #print(supplemental_precip.regridded_precip2[:,:])
    #print("supplemental_precip.regridded_precip2[:,:].shape")
    #print(supplemental_precip.regridded_precip2[:,:].shape)

    read_hours = 0
    found_target_hh = False
    ana_data = []
    if mpi_config.rank == 0:
        target_hh = Path(input_forcings.file_in2).stem[-4:-2]
        _,yyyymmddhh,*_ = Path(supplemental_precip.file_in2).stem.split('.')
        #Stage IV files are initialized 6 hours ago
        date_iter = datetime.strptime(f"{yyyymmddhh}", '%Y%m%d%H') - timedelta(hours=6)
        end_date = date_iter + timedelta(hours=6)
        beg_hh,end_hh,yyyymmdd = date_iter.strftime('%H'), end_date.strftime('%H'), date_iter.strftime('%Y%m%d')
        #Advance the date_iter by 1 hour since the beginning of the Stage IV data in date range is excluded, the end is included
        #(begin_date,end_date]
        date_iter += timedelta(hours=1)
        while date_iter <= end_date:
            tmp_file = f"{input_forcings.inDir}/{date_iter.strftime('%Y%m%d%H')}/{date_iter.strftime('%Y%m%d%H')}00.LDASIN_DOMAIN1"
            if os.path.exists(tmp_file):
                config_options.statusMsg = f"Reading {input_forcings.netcdf_var_names[3]} from {tmp_file} for disaggregation"
                err_handler.log_msg(config_options, mpi_config)
                with Dataset(tmp_file,'r') as ds:
                    try:
                        #Read in rainrate
                        data = ds.variables[input_forcings.netcdf_var_names[3]][0, :, :]
                        data[data == config_options.globalNdv] = np.nan
                        ana_data.append(data)
                        read_hours += 1
                        if date_iter.hour == int(target_hh):
                            found_target_hh = True
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to extract: RAINRATE from: {input_forcings.file_in2} ({str(err)})"
                        err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = f"Input file missing {tmp_file}"
                err_handler.log_warning(config_options, mpi_config)

            date_iter += timedelta(hours=1)

    found_target_hh = mpi_config.broadcast_parameter(found_target_hh, config_options, param_type=bool)
    err_handler.check_program_status(config_options, mpi_config)
    if not found_target_hh:
        if mpi_config.rank == 0:
            config_options.statusMsg = f"Could not find AnA target_hh = {target_hh} for disaggregation. Setting output values to {config_options.globalNdv}."
            err_handler.log_warning(config_options, mpi_config)
        supplemental_precip.regridded_precip2[:,:] = config_options.globalNdv
        return

    read_hours = mpi_config.broadcast_parameter(read_hours, config_options, param_type=int)
    err_handler.check_program_status(config_options, mpi_config)
    if read_hours != 6:
        if mpi_config.rank == 0:
            config_options.statusMsg = f"Could not find all 6 AnA files for disaggregation. Only found {read_hours} hours. Setting output values to {config_options.globalNdv}."
            err_handler.log_warning(config_options, mpi_config)
        supplemental_precip.regridded_precip2[:,:] = config_options.globalNdv
        return

    ana_sum = np.array([],dtype=np.float32)
    target_data = np.array([],dtype=np.float32)
    ana_all_zeros = np.array([],dtype=bool)
    ana_no_zeros = np.array([],dtype=bool)
    target_data_no_zeros = np.array([],dtype=bool)
    if mpi_config.rank == 0:
        config_options.statusMsg = f"Performing hourly disaggregation of {supplemental_precip.file_in2}"
        err_handler.log_msg(config_options, mpi_config)

        ana_sum = sum(ana_data)
        target_data = ana_data[(int(target_hh)-1)%6]

        ana_zeros = [(a == 0).astype(int) for a in ana_data]
        target_data_zeros = (target_data == 0)
        target_data_no_zeros = ~target_data_zeros
        ana_zeros_sum = sum(ana_zeros)
        ana_all_zeros = (ana_zeros_sum == 6)
        ana_no_zeros = (ana_zeros_sum == 0)

    err_handler.check_program_status(config_options, mpi_config)
    ana_sum = mpi_config.scatter_array(input_forcings, ana_sum, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    target_data = mpi_config.scatter_array(input_forcings, target_data, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    ana_all_zeros = mpi_config.scatter_array(input_forcings, ana_all_zeros, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    ana_no_zeros = mpi_config.scatter_array(input_forcings, ana_no_zeros, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    target_data_no_zeros = mpi_config.scatter_array(input_forcings, target_data_no_zeros, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    
    orig_err_settings = np.geterr()
    #Ignore Warning: RuntimeWarning: invalid value encountered in true_divide
    np.seterr(invalid='ignore')
    disagg_factors = np.select([ana_all_zeros,(ana_no_zeros | target_data_no_zeros)],
              [1/6.0*np.ones(supplemental_precip.regridded_precip2[:,:].shape),np.clip(target_data/ana_sum,0,1)],
              0)
    np.seterr(**orig_err_settings)

    if mpi_config.comm.Get_size() == 1 and test_enabled:
        test_file = f"{config_options.scratch_dir}/stage_4_acc6h_{yyyymmdd}_{beg_hh}_{end_hh}.txt"
        np.savetxt(test_file,supplemental_precip.regridded_precip2)
    
        test_file = f"{config_options.scratch_dir}/disaggregation_factors_{target_hh}_{yyyymmdd}{beg_hh}_{end_date.strftime('%Y%m%d%H')}.txt"
        np.savetxt(test_file,disagg_factors)

    #supplemental_precip.regridded_precip2[(0.0 < supplemental_precip.regridded_precip2) & (supplemental_precip.regridded_precip2 < 0.00003)] = 0.0
    supplemental_precip.regridded_precip2[:,:] *= disagg_factors
    np.nan_to_num(supplemental_precip.regridded_precip2[:,:], copy=False, nan=config_options.globalNdv) 

    if mpi_config.comm.Get_size() == 1 and test_enabled:
        test_file = f"{config_options.scratch_dir}/stage_4_acc6h_disaggregated_{target_hh}_{yyyymmdd}{beg_hh}_{end_date.strftime('%Y%m%d%H')}.txt"
        np.savetxt(test_file,supplemental_precip.regridded_precip2)
