"""
Module file for incorporating bias correction for various input
forcing variables. Calling tree will be guided based on user-specified
options.
"""
from core import  errMod
import os
from netCDF4 import Dataset

def run_bias_correction(input_forcings,ConfigOptions,MpiConfig):
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
    bias_correct_temperature[input_forcings.t2dBiasCorrectOpt](input_forcings,
                                                               ConfigOptions, MpiConfig, 4)
    errMod.check_program_status(ConfigOptions, MpiConfig)


    # Dictionary for mapping to humidity bias correction.
    bias_correct_humidity = {
        0: no_bias_correct
    }
    bias_correct_humidity[input_forcings.q2dBiasCorrectOpt](input_forcings,
                                                            ConfigOptions, MpiConfig, 5)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to surface pressure bias correction.
    bias_correct_pressure = {
        0: no_bias_correct
    }
    bias_correct_pressure[input_forcings.psfcBiasCorrectOpt](input_forcings,
                                                             ConfigOptions, MpiConfig, 6)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to incoming shortwave radiation correction.
    bias_correct_sw = {
        0: no_bias_correct
    }
    bias_correct_sw[input_forcings.swBiasCorrectOpt](input_forcings,
                                                     ConfigOptions, MpiConfig, 7)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to incoming longwave radiation correction.
    bias_correct_lw = {
        0: no_bias_correct
    }
    bias_correct_lw[input_forcings.lwBiasCorrectOpt](input_forcings,
                                                     ConfigOptions, MpiConfig, 2)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to wind bias correction.
    bias_correct_wind = {
        0: no_bias_correct
    }
    # Run for U-Wind
    bias_correct_wind[input_forcings.windBiasCorrectOpt](input_forcings,
                                                         ConfigOptions, MpiConfig, 0)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    # Run for V-Wind
    bias_correct_wind[input_forcings.windBiasCorrectOpt](input_forcings,
                                                         ConfigOptions, MpiConfig, 1)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary for mapping to precipitation bias correction.
    bias_correct_precip = {
        0: no_bias_correct
    }
    bias_correct_precip[input_forcings.precipBiasCorrectOpt](input_forcings,
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

def no_bias_correct(input_forcings,ConfigOptions, MpiConfig, force_num):
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

def cfsv2_nldas_nwm_bias_correct(input_forcings,ConfigOptions, MpiConfig, force_num):
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

    errMod.check_program_status(ConfigOptions, MpiConfig)