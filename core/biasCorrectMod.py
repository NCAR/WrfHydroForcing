"""
Module file for incorporating bias correction for various input
forcing variables. Calling tree will be guided based on user-specified
options.
"""
from core import  errMod
import os
from netCDF4 import Dataset

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

    # Since this is NWM-specific, ensure we are actually running the NWM.
    if GeoMetaWrfHydro.nx_global != 4608:
        ConfigOptions.errMsg = "This is a Conus NWM-Specific bias-correction routine. Expected 4608 columns " \
                               "in the modeling domain but found: " + str(GeoMetaWrfHydro.nx_global)
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    if GeoMetaWrfHydro.ny_global != 3840:
        ConfigOptions.errMsg = "This is a Conus NWM-Specific bias-correction routine. Expected 3840 rows " \
                               "in the modeling domain but found: " + str(GeoMetaWrfHydro.ny_global)
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Check to ensure we are running with CFSv2 here....
    if input_forcings.productName != "CFSv2_6Hr_Global_GRIB2":
        ConfigOptions.errMsg = "Attempting to run CFSv2-NLDAS bias correction on: " + input_forcings.productName
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    if MpiConfig.rank == 0:
        # We will be doing the bias correction on rank 0. These grids are coarse enough
        # that it should take a relatively small amount of time to run the calculations.
        # Once bias correction on the global CFSv2 grids has taken place (where NLDAS params exist),
        # THEN CFSv2 data will be interpolated and regridded to the final NWM grid.
        # First determine the expected input NLDAS2 parameteric NetCDF file
        # that contains parameters for this specific output hour.
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
        if idNldasParam.dimensions['lat_0'].size != 30:
            ConfigOptions.errMsg = "Expected lat_0 size is 30 - found size of: " + \
                                   str(idNldasParam.dimensions['lat_0'].size) + " in: " + nldas_param_file
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        if idNldasParam.dimensions['lon_0'].size != 60:
            ConfigOptions.errMsg = "Expected lon_0 size is 60 - found size of: " + \
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

    else:
        idNldasParam = None
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Reset variables for memory efficiency
    idNldasParam = None