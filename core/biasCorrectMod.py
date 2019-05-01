"""
Module file for incorporating bias correction for various input
forcing variables. Calling tree will be guided based on user-specified
options.
"""
from core import  errMod
import os
from netCDF4 import Dataset
import numpy as np
import math

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
    # Specify the min/max ranges on CDF/PDF values for each variable
    valRange1 = {
        0: -50.0,
        1: -50.0,
        2: 1.0,
        3: 0.01,
        4: 200.0,
        5: 0.01,
        6: 50000.0,
        7: 0.0
    }
    valRange2 = {
        0: 50.0,
        1: 50.0,
        2: 800.0,
        3: 100.0,
        4: 330.0,
        5: 40.0,
        6: 1100000.0,
        7: 1000.0
    }
    valStep = {
        0: 0.1,
        1: 0.1,
        2: 0.19975,
        3: 0.049995,
        4: 0.1,
        5: 3.9999,
        6: 350.0,
        7: 10.0
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

        # Extract the fill value
        try:
            fillTmp = idNldasParam.variables[nldasParam1Vars[force_num]]._FillValue
        except:
            ConfigOptions.errMsg = "Unable to extract Fill_Value from: " + nldas_param_file
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

        # Set missing values accordingly.
        nldas_param_1[np.where(nldas_param_1 == fillTmp)] = ConfigOptions.globalNdv
        nldas_param_2[np.where(nldas_param_1 == fillTmp)] = ConfigOptions.globalNdv
        if force_num == 3:
            nldas_zero_pcp[np.where(nldas_zero_pcp == fillTmp)] = ConfigOptions.globalNdv

    else:
        nldas_param_1 = None
        nldas_param_2 = None
        nldas_zero_pcp = None
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Reset the temporary fill value
    fillTmp = None

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

        # Reset any missing values. Because the fill values for these files are all over the map, we
        # will just do a gross check here. For the most part, there shouldn't be missing values.
        param_1[np.where(param_1 > 500000.0)] = ConfigOptions.globalNdv
        param_2[np.where(param_2 > 500000.0)] = ConfigOptions.globalNdv
        prev_param_1[np.where(prev_param_1 > 500000.0)] = ConfigOptions.globalNdv
        prev_param_2[np.where(prev_param_2 > 500000.0)] = ConfigOptions.globalNdv
        if force_num == 3:
            zero_pcp[np.where(zero_pcp > 500000.0)] = ConfigOptions.globalNdv
            prev_zero_pcp[np.where(prev_zero_pcp > 500000.0)] = ConfigOptions.globalNdv

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

    # Now.... Loop through the local CFSv2 grid cells and perform the following steps:
    # 1.) Interpolate the six-hour values to the current output timestep.
    # 2.) Calculate the CFSv2 cdf/pdf
    # 3.) Calculate the NLDAS cdf/pdf
    # 4.) Adjust CFSv2 values based on the method of pdf matching.
    # 5.) Regrid the CFSv2 values to the WRF-Hydro domain using the pre-calculated ESMF
    #     regridding object.
    # 6.) Place the data into the final output arrays for further processing (downscaling).
    # 7.) Reset variables for memory efficiency and exit the routine.

    # Establish local arrays of data.
    cfs_data = np.empty([input_forcings.ny_local, input_forcings.nx_local], np.float64)

    # Establish parameters of the CDF matching.
    vals = np.arange(valRange1[force_num], valRange2[force_num], valStep[force_num])

    # Process each of the pixel cells for this local processor on the CFS grid.
    for x_local in range(0,input_forcings.nx_local):
        for y_local in range(0,input_forcings.ny_local):
            cfs_prev_tmp = input_forcings.coarse_input_forcings1[force_num, y_local, x_local]
            cfs_next_tmp = input_forcings.coarse_input_forcings2[force_num, y_local, x_local]

            # Check for any missing parameter values. If any missing values exist,
            # set this flag to False. Further down, if it's False, we will simply
            # set the local CFS adjusted value to the interpolated value.
            correctFlag = True
            if cfs_param_1_sub[y_local,x_local] == ConfigOptions.globalNdv:
                correctFlag = False
            if cfs_param_2_sub[y_local,x_local] == ConfigOptions.globalNdv:
                correctFlag = False
            if cfs_prev_param_1_sub[y_local,x_local] == ConfigOptions.globalNdv:
                correctFlag = False
            if cfs_prev_param_2_sub[y_local,x_local] == ConfigOptions.globalNdv:
                correctFlag = False
            if nldas_param_1_sub[y_local,x_local] == ConfigOptions.globalNdv:
                correctFlag = False
            if nldas_param_2_sub[y_local,x_local] == ConfigOptions.globalNdv:
                correctFlag = False
            if force_num == 3:
                if cfs_prev_zero_pcp_sub[y_local,x_local] == ConfigOptions.globalNdv:
                    correctFlag = False
                if cfs_zero_pcp_sub[y_local,x_local] == ConfigOptions.globalNdv:
                    correctFlag = False
                if nldas_zero_pcp_sub[y_local,x_local] == ConfigOptions.globalNdv:
                    correctFlag = False

            # Interpolate the two CFS values (and parameters) in time.
            dtFromPrevious = ConfigOptions.current_output_date - input_forcings.fcst_date1
            hrFromPrevious = dtFromPrevious.total_seconds()/3600.0
            interpFactor1 = float(1 - (hrFromPrevious / 6.0))
            interpFactor2 = float(hrFromPrevious / 6.0)

            # Since this is only for CFSv2 6-hour data, we will assume 6-hour intervals.
            # This is already checked at the beginning of this routine for the product name.
            cfs_param_1_interp = cfs_prev_param_1_sub[y_local, x_local] * interpFactor1 + \
                                 cfs_param_1_sub[y_local, x_local] * interpFactor2
            cfs_param_2_interp = cfs_prev_param_2_sub[y_local, x_local] * interpFactor1 + \
                                 cfs_param_2_sub[y_local, x_local] * interpFactor2
            cfs_interp_fcst = cfs_prev_tmp * interpFactor1 + cfs_next_tmp * interpFactor2

            pts = (vals - cfs_param_1_interp) / cfs_param_2_interp
            spacing = (vals[2] - vals[1]) / cfs_param_2_interp
            cfs_pdf = (np.exp(-0.5 * (pts ^ 2)) / math.sqrt(2 * 3.141592)) * spacing
            cfs_cdf = np.cumsum(cfs_pdf)

            nldas_nearest_1 = nldas_param_1_sub[y_local, x_local]
            nldas_nearest_2 = nldas_param_2_sub[y_local, x_local]

            pts = (vals - nldas_nearest_1) / nldas_nearest_2
            spacing = (vals[2] - vals[1]) / nldas_nearest_2
            nldas_pdf = (np.exp(-0.5 * (pts ^ 2)) / math.sqrt(2 * 3.141592)) * spacing
            nldas_cdf = np.cumsum(nldas_pdf)

            # compute adjusted value now using the CFSv2 forecast value and the two CDFs
            # find index in vals array


            # Reset variables for memory purposes.
            pts = None

