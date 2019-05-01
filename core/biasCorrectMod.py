"""
Module file for incorporating bias correction for various input
forcing variables. Calling tree will be guided based on user-specified
options.
"""
from core import  errMod
import os
from netCDF4 import Dataset
import numpy as np

def run_bias_correction(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    Top level calling routine for initiating bias correction on
    this particular input forcing. This is called prior to downscaling,
    but after regridding.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    # Dictionary for mapping to temperature bias correction.
    bias_correct_temperature = {
        0: no_bias_correct
    }
    bias_correct_temperature[input_forcings.t2dBiasCorrectOpt](input_forcings, GeoMetaWrfHydro,
                                                               ConfigOptions, MpiConfig, 4)
    errMod.check_program_status(ConfigOptions, MpiConfig)


    # Dictionary for mapping to humidity bias correction.
    bias_correct_humidity = {
        0: no_bias_correct
    }
    bias_correct_humidity[input_forcings.q2dBiasCorrectOpt](input_forcings, GeoMetaWrfHydro,
                                                            ConfigOptions, MpiConfig, 5)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to surface pressure bias correction.
    bias_correct_pressure = {
        0: no_bias_correct
    }
    bias_correct_pressure[input_forcings.psfcBiasCorrectOpt](input_forcings, GeoMetaWrfHydro,
                                                             ConfigOptions, MpiConfig, 6)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to incoming shortwave radiation correction.
    bias_correct_sw = {
        0: no_bias_correct
    }
    bias_correct_sw[input_forcings.swBiasCorrectOpt](input_forcings, GeoMetaWrfHydro,
                                                     ConfigOptions, MpiConfig, 7)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to incoming longwave radiation correction.
    bias_correct_lw = {
        0: no_bias_correct
    }
    bias_correct_lw[input_forcings.lwBiasCorrectOpt](input_forcings, GeoMetaWrfHydro,
                                                     ConfigOptions, MpiConfig, 2)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to wind bias correction.
    bias_correct_wind = {
        0: no_bias_correct
    }
    # Run for U-Wind
    bias_correct_wind[input_forcings.windBiasCorrectOpt](input_forcings, GeoMetaWrfHydro,
                                                         ConfigOptions, MpiConfig, 0)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    # Run for V-Wind
    bias_correct_wind[input_forcings.windBiasCorrectOpt](input_forcings, GeoMetaWrfHydro,
                                                         ConfigOptions, MpiConfig, 1)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to precipitation bias correction.
    bias_correct_precip = {
        0: no_bias_correct
    }
    bias_correct_precip[input_forcings.precipBiasCorrectOpt](input_forcings, GeoMetaWrfHydro,
                                                             ConfigOptions, MpiConfig, 3)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Assign the temperature/pressure grids to temporary placeholders here.
    # these will be used if 2-meter specific humidity is downscaled.
    if input_forcings.q2dDownscaleOpt != 0:
        input_forcings.t2dTmp = input_forcings.final_forcings[4,:,:]*1.0
        input_forcings.psfcTmp = input_forcings.final_forcings[6,:,:]*1.0
    else:
        input_forcings.t2dTmp = None
        input_forcings.psfcTmp = None

def no_bias_correct(input_forcings, GeoMetaWrfHydro, ConfigOptions, MpiConfig, force_num):
    """
    Generic routine to simply pass forcing states through without any
    bias correction.
    :param input_forcings:
    :param ConfigOptions:
    :param force_num:
    :return:
    """
    try:
        input_forcings.final_forcings[force_num, :, :] = input_forcings.final_forcings[force_num, :, :]
    except:
        ConfigOptions.errMsg = "Unable to set final forcings during bias correction routine."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

def cfsv2_nldas_nwm_bias_correct(input_forcings, GeoMetaWrfHydro, ConfigOptions, MpiConfig, force_num):
    """
    Routine to run CDF/PDF bias correction parametric corrections
    SPECIFIC to the NWM long-range configuration.
    :param input_forcings:
    :param ConfigOptions:
    :param force_num:
    :return:
    """
    # Create a dictionary that maps forcing numbers to the expected NetCDF variable names, etc.
    nldasParam1Vars = {
        0: 'UGRD10M_PARAM_1',
        1: 'VGRD10M_PARAM_1',
        2: 'LW_PARAM_1',
        3: 'PRATE_PARAM_1',
        4: 'T2M_PARAM_1',
        5: 'Q2M_PARAM_1',
        6: 'PSFC_PARAM_1',
        7: 'SW_PARAM_1'
    }
    nldasParam2Vars = {
        0: 'UGRD10M_PARAM_2',
        1: 'VGRD10M_PARAM_2',
        2: 'LW_PARAM_2',
        3: 'PRATE_PARAM_2',
        4: 'T2M_PARAM_2',
        5: 'Q2M_PARAM_2',
        6: 'PSFC_PARAM_2',
        7: 'SW_PARAM_2'
    }
    cfsParamPathVars = {
        0: 'ugrd',
        1: 'vgrd',
        2: 'dlwsfc',
        3: 'prate',
        4: 'tmp2m',
        5: 'q2m',
        6: 'pressfc',
        7: 'dswsfc'
    }
    # Check to ensure we are running with CFSv2 here....
    if input_forcings.productName != "CFSv2_6Hr_Global_GRIB2":
        ConfigOptions.errMsg = "Attempting to run CFSv2-NLDAS bias correction on: " + input_forcings.productName
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Open the necessary parameter grids, which are on the global CFSv2 grid, then scatter them out
    # to the various processors.
    if MpiConfig.rank == 0:
        nldas_param_file = input_forcings.paramDir + "/NLDAS_Climo/nldas2_" + \
                           ConfigOptions.current_output_date.strftime('%m%d%H') + \
                           "_dist_params.nc"
        if not os.path.isfile(nldas_param_file):
            ConfigOptions.errMsg = "Unable to locate necessary bias correction parameter file: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        # Open the NetCDF file.
        try:
            idNldasParam = Dataset(nldas_param_file, 'r')
        except:
            ConfigOptions.errMsg = "Unable to open parameter file: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        # Ensure dimensions/variables are as expected.
        if 'lat_0' not in idNldasParam.dimensions.keys():
            ConfigOptions.errMsg = "Expected to find lat_0 dimension in: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if 'lon_0' not in idNldasParam.dimensions.keys():
            ConfigOptions.errMsg = "Expected to find lon_0 dimension in: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if idNldasParam.dimensions['lat_0'].size != 190:
            ConfigOptions.errMsg = "Expected lat_0 size is 190 - found size of: " + \
                                   str(idNldasParam.dimensions['lat_0'].size) + " in: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if idNldasParam.dimensions['lon_0'].size != 384:
            ConfigOptions.errMsg = "Expected lon_0 size is 384 - found size of: " + \
                                   str(idNldasParam.dimensions['lon_0'].size) + " in: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if nldasParam1Vars[force_num] not in idNldasParam.variables.keys():
            ConfigOptions.errMsg = "Expected variable: " + nldasParam1Vars[force_num] + " not found " + \
                                   "in: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if nldasParam2Vars[force_num] not in idNldasParam.variables.keys():
            ConfigOptions.errMsg = "Expected variable: " + nldasParam2Vars[force_num] + " not found " + \
                                   "in: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if force_num == 3:
            if 'ZERO_PRECIP_PROB' not in idNldasParam.variables.keys():
                ConfigOptions.errMsg = "Expected variable: ZERO_PRECIP_PROB not found in: " + nldas_param_file
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

        try:
            nldas_param_1 = idNldasParam.variables[nldasParam1Vars[force_num]][:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract: " + nldasParam1Vars[force_num] + " from: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            nldas_param_2 = idNldasParam.variables[nldasParam2Vars[force_num]][:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract: " + nldasParam2Vars[force_num] + " from: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if nldas_param_1.shape[0] != 190 or nldas_param_1.shape[1] != 384:
            ConfigOptions.errMsg = "Parameter variable: " + nldasParam1Vars[force_num] + " from: " + \
                                   nldas_param_file + " not of shape [190,384]."
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if nldas_param_2.shape[0] != 190 or nldas_param_2.shape[1] != 384:
            ConfigOptions.errMsg = "Parameter variable: " + nldasParam2Vars[force_num] + " from: " + \
                                   nldas_param_file + " not of shape [190,384]."
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        # Read in the zero precip prob grids if we are bias correcting precipitation.
        if force_num == 3:
            try:
                nldas_zero_pcp = idNldasParam.variables['ZERO_PRECIP_PROB'][:,:]
            except:
                ConfigOptions.errMsg = "Unable to extract ZERO_PRECIP_PROB from: " + nldas_param_file
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if nldas_zero_pcp.shape[0] != 190 or nldas_zero_pcp.shape[1] != 384:
                ConfigOptions.errMsg = "Parameter variable: ZERO_PRECIP_PROB from: " + nldas_param_file + \
                                       " not of shape [190,384]."
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

        #cfs_data = input_forcings.global_input_forcings1[force_num, ystart:yend, xstart:xend]
        #cfs_data = input_forcings.global_input_forcings2[force_num, ystart:yend, xstart:xend]
    else:
        nldas_param_1 = None
        nldas_param_2 = None
        nldas_zero_pcp = None
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Scatter NLDAS parameters
    nldas_param_1_sub = MpiConfig.scatter_array(input_forcings, nldas_param_1, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    nldas_param_2_sub = MpiConfig.scatter_array(input_forcings, nldas_param_2, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    if force_num == 3:
        nldas_zero_pcp_sub = MpiConfig.scatter_array(input_forcings, nldas_zero_pcp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

    if MpiConfig.rank == 0:
        # Read in the CFSv2 parameter files, based on the previous CFSv2 dates
        cfs_param_path1 = input_forcings.paramDir + "/cfs_" + cfsParamPathVars[force_num] + "_" + \
            input_forcings.fcst_date1.strftime('%m%d') + "_" + input_forcings.fcst_date1.strftime('%H') + \
            '_dist_params.nc'
        cfs_param_path2 = input_forcings.paramDir + "/cfs_" + cfsParamPathVars[force_num] + "_" + \
                          input_forcings.fcst_date2.strftime('%m%d') + "_" + input_forcings.fcst_date2.strftime('%H') + \
                          '_dist_params.nc'

        if not os.path.isfile(cfs_param_path1):
            ConfigOptions.errMsg = "Unable to locate necessary parameter file: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if not os.path.isfile(cfs_param_path2):
            ConfigOptions.errMsg = "Unable to locate necessary parameter file: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        # Open the files and ensure they contain the correct information.
        try:
            idCfsParam1 = Dataset(cfs_param_path1, 'r')
        except:
            ConfigOptions.errMsg = "Unable to open parameter file: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            idCfsParam2 = Dataset(cfs_param_path2, 'r')
        except:
            ConfigOptions.errMsg = "Unable to open parameter file: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if 'DISTRIBUTION_PARAM_1' not in idCfsParam1.variables.keys():
            ConfigOptions.errMsg = "Expected DISTRIBUTION_PARAM_1 variable not found in: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if 'DISTRIBUTION_PARAM_2' not in idCfsParam1.variables.keys():
            ConfigOptions.errMsg = "Expected DISTRIBUTION_PARAM_1 variable not found in: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if 'DISTRIBUTION_PARAM_1' not in idCfsParam2.variables.keys():
            ConfigOptions.errMsg = "Expected DISTRIBUTION_PARAM_1 variable not found in: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if 'DISTRIBUTION_PARAM_2' not in idCfsParam2.variables.keys():
            ConfigOptions.errMsg = "Expected DISTRIBUTION_PARAM_1 variable not found in: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        try:
            param_1 = idCfsParam2.variables['DISTRIBUTION_PARAM_1'][:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract DISTRIBUTION_PARAM_1 from: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            param_2 = idCfsParam2.variables['DISTRIBUTION_PARAM_2'][:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract DISTRIBUTION_PARAM_2 from: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        try:
            lat_0 = idCfsParam2.variables['lat_0'][:]
        except:
            ConfigOptions.errMsg = "Unable to extract lat_0 from: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            lon_0 = idCfsParam2.variables['lon_0'][:]
        except:
            ConfigOptions.errMsg = "Unable to extract lon_0 from: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        try:
            prev_param_1 = idCfsParam1.variables['DISTRIBUTION_PARAM_1'][:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract DISTRIBUTION_PARAM_1 from: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            prev_param_2 = idCfsParam1.variables['DISTRIBUTION_PARAM_2'][:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract DISTRIBUTION_PARAM_2 from: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if param_1.shape[0] != 190 and param_1.shape[1] != 384:
            ConfigOptions.errMsg = "Unexpected DISTRIBUTION_PARAM_1 found in: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if param_2.shape[0] != 190 and param_2.shape[1] != 384:
            ConfigOptions.errMsg = "Unexpected DISTRIBUTION_PARAM_2 found in: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if prev_param_1.shape[0] != 190 and prev_param_1.shape[1] != 384:
            ConfigOptions.errMsg = "Unexpected DISTRIBUTION_PARAM_1 found in: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if prev_param_2.shape[0] != 190 and prev_param_2.shape[1] != 384:
            ConfigOptions.errMsg = "Unexpected DISTRIBUTION_PARAM_2 found in: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        # Read in the zero precip prob grids if we are bias correcting precipitation.
        if force_num == 3:
            try:
                zero_pcp = idCfsParam2.variables['ZERO_PRECIP_PROB'][:, :]
            except:
                ConfigOptions.errMsg = "Unable to locate ZERO_PRECIP_PROB in: " + cfs_param_path2
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            try:
                prev_zero_pcp = idCfsParam2.variables['ZERO_PRECIP_PROB'][:, :]
            except:
                ConfigOptions.errMsg = "Unable to locate ZERO_PRECIP_PROB in: " + cfs_param_path1
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if zero_pcp.shape[0] != 190 and zero_pcp.shape[1] != 384:
                ConfigOptions.errMsg = "Unexpected ZERO_PRECIP_PROB found in: " + cfs_param_path2
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if prev_zero_pcp.shape[0] != 190 and prev_zero_pcp.shape[1] != 384:
                ConfigOptions.errMsg = "Unexpected ZERO_PRECIP_PROB found in: " + cfs_param_path1
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

    else:
        param_1 = None
        param_2 = None
        prev_param_1 = None
        prev_param_2 = None
        zero_pcp = None
        prev_zero_pcp = None
    errMod.check_program_status(MpiConfig, ConfigOptions)

    # Scatter CFS parameters
    cfs_param_1_sub = MpiConfig.scatter_array(input_forcings, param_1, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    cfs_param_2_sub = MpiConfig.scatter_array(input_forcings, param_2, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    cfs_prev_param_1_sub = MpiConfig.scatter_array(input_forcings, prev_param_1, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    cfs_prev_param_2_sub = MpiConfig.scatter_array(input_forcings, prev_param_2, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    if force_num == 3:
        cfs_zero_pcp_sub = MpiConfig.scatter_array(input_forcings, zero_pcp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)
        cfs_prev_zero_pcp_sub = MpiConfig.scatter_array(input_forcings, prev_zero_pcp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

    if MpiConfig.rank == 0:
        # Close the parameter files.
        try:
            idNldasParam.close()
        except:
            ConfigOptions.errMsg = "Unable to close parameter file: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            idCfsParam1.close()
        except:
            ConfigOptions.errMsg = "Unable to close parameter file: " + cfs_param_path1
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            idCfsParam2.close()
        except:
            ConfigOptions.errMsg = "Unable to close parameter file: " + cfs_param_path2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

    else:
        idNldasParam = None
        idCfsParam1 = None
        idCfsParam2 = None
        idGridCorr = None
    errMod.check_program_status(ConfigOptions, MpiConfig)