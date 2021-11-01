"""
Module to provide disaggregation functionality.
"""
from pathlib import Path
from datetime import datetime, timedelta
import os.path

import numpy as np
from netCDF4 import Dataset

from core import err_handler


def disaggregate_factory(ConfigOptions):
    if len(ConfigOptions.supp_precip_forcings) == 1 and ConfigOptions.supp_precip_forcings[0] == 11:
        return ext_ana_disaggregate
    #Add new cases here
    #elif condition:
    else:
        return no_op_disaggregate


def no_op_disaggregate(input_forcings, supplemental_precip, ConfigOptions, MpiConfig):
    pass


def ext_ana_disaggregate(input_forcings, supplemental_precip, ConfigOptions, MpiConfig):
    """
    Function for disaggregating 6hr SuppPcp data to 1hr Input data
    :param input_forcings:
    :param supplemental_precip:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Check to make sure we have valid grids.
    if input_forcings.regridded_forcings2 is None or supplemental_precip.regridded_precip2 is None:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Bypassing ext_ana_disaggregation routine due to missing input or supp pcp data"
            err_handler.log_warning(ConfigOptions, MpiConfig)
        return
            
    if supplemental_precip.ext_ana != "STAGE4":
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = f"Bypassing ext_ana_disaggregation routine due to supplemental_precip.ext_ana = {supplemental_precip.ext_ana}"
            err_handler.log_msg(ConfigOptions, MpiConfig)
        return

    #print(input_forcings.regridded_forcings2[3,:,:])
    #print(supplemental_precip.regridded_precip2[:,:])
 
    target_hh = Path(input_forcings.file_in2).stem[-4:-2]
    _,_,_,beg_hh,end_hh,yyyymmdd = Path(supplemental_precip.file_in2).stem.split('_')
    data_sum = None
    target_data = None
    date_iter = datetime.strptime(f"{yyyymmdd}{beg_hh}", '%Y%m%d%H')
    end_date = date_iter + timedelta(hours=6)
    #Advance the date_iter by 1 hour since the beginning of the Stage IV data in date range is excluded, the end is included
    #(begin_date,end_date]
    date_iter += timedelta(hours=1)
    while date_iter <= end_date:
        tmp_file = f"{input_forcings.inDir}/{date_iter.strftime('%Y%m%d%H')}/{date_iter.strftime('%Y%m%d%H')}00.LDASIN_DOMAIN1"
        if MpiConfig.rank == 0:
            if os.path.exists(tmp_file):
                ConfigOptions.statusMsg = f"Reading {input_forcings.netcdf_var_names[3]} from {tmp_file} for disaggregation"
                err_handler.log_msg(ConfigOptions, MpiConfig)
                with Dataset(tmp_file,'r') as ds:
                    try:
                        #Read in rainrate
                        data = ds.variables[input_forcings.netcdf_var_names[3]][0, :, :]
                        if data_sum is not None:
                            data_sum += data
                        else:
                            data_sum = np.copy(data)
                        if date_iter.hour == int(target_hh):
                            target_data = np.copy(data)
    
                        del data
                    except (ValueError, KeyError, AttributeError) as err:
                        ConfigOptions.errMsg = f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.file_in2} ({str(err)})"
                        err_handler.log_critical(config_options, mpi_config)
            else:
                ConfigOptions.statusMsg = f"Input file missing {tmp_file}"
                err_handler.log_warning(ConfigOptions, MpiConfig)
        err_handler.check_program_status(ConfigOptions, MpiConfig)
        
        date_iter += timedelta(hours=1)

    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = f"Performing disaggregation of {supplemental_precip.file_in2} using the {target_hh} hour fraction of the sum of hourly ExtAnA files RAINRATE"
        err_handler.log_msg(ConfigOptions, MpiConfig)
        if target_data is None:
            target_data = np.empty(data_sum.shape)
            target_data[:] = np.nan
            ConfigOptions.statusMsg = f"Could not find ExtAnA target_hh = {target_hh} for disaggregation. Setting values to {ConfigOptions.globalNdv}."
            err_handler.log_warning(ConfigOptions, MpiConfig)

    #TODO: disable test code
    #Begin test code
    test_file = f"{ConfigOptions.scratch_dir}/stage_4_A_PCP_GDS5_SFC_acc6h_{yyyymmdd}_{beg_hh}_{end_hh}.txt"
    np.savetxt(test_file,supplemental_precip.regridded_precip2)
    #End test code
    #supplemental_precip.regridded_precip2[(0.0 < supplemental_precip.regridded_precip2) & (supplemental_precip.regridded_precip2 < 0.00003)] = 0.0
    #TODO: disable test code
    #Begin test code
    test_file = f"{ConfigOptions.scratch_dir}/disaggregation_factors_{target_hh}_{yyyymmdd}{beg_hh}_{end_date.strftime('%Y%m%d%H')}.txt"
    np.savetxt(test_file,np.nan_to_num(target_data/data_sum,nan=ConfigOptions.globalNdv))
    #End test code
    supplemental_precip.regridded_precip2[:,:] *= target_data/data_sum
    np.nan_to_num(supplemental_precip.regridded_precip2[:,:], copy=False, nan=ConfigOptions.globalNdv) 
     #TODO: disable test code
    #Begin test code
    test_file = f"{ConfigOptions.scratch_dir}/stage_4_A_PCP_GDS5_SFC_acc6_disaggregation_{target_hh}_{yyyymmdd}{beg_hh}_{end_date.strftime('%Y%m%d%H')}.txt"
    np.savetxt(test_file,supplemental_precip.regridded_precip2)
    #End test code
    
