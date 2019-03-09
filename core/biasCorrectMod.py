"""
Module file for incorporating bias correction for various input
forcing variables. Calling tree will be guided based on user-specified
options.
"""
def run_bias_correction(input_forcings,ConfigOptions):
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
                                                               ConfigOptions)

    # Dictionary for mapping to humidity bias correction.
    bias_correct_humidity = {
        0: no_bias_correct
    }
    bias_correct_humidity[input_forcings.q2dBiasCorrectOpt](input_forcings,
                                                            ConfigOptions)

    # Dictionary for mapping to surface pressure bias correction.
    bias_correct_pressure = {
        0: no_bias_correct
    }
    bias_correct_pressure[input_forcings.psfcBiasCorrectOpt](input_forcings,
                                                             ConfigOptions)

    # Dictionary for mapping to incoming shortwave radiation correction.
    bias_correct_sw = {
        0: no_bias_correct
    }
    bias_correct_sw[input_forcings.swBiasCorrectOpt](input_forcings,
                                                     ConfigOptions)

    # Dictionary for mapping to incoming longwave radiation correction.
    bias_correct_lw = {
        0: no_bias_correct
    }
    bias_correct_lw[input_forcings.lwBiasCorrectOpt](input_forcings,
                                                     ConfigOptions)

    # Dictionary for mapping to wind bias correction.
    bias_correct_wind = {
        0: no_bias_correct
    }
    bias_correct_wind[input_forcings.windBiasCorrectOpt](input_forcings,
                                                         ConfigOptions)

    # Dictionary for mapping to precipitation bias correction.
    bias_correct_precip = {
        0: no_bias_correct
    }
    bias_correct_precip[input_forcings.precipBiasCorrectOpt](input_forcings,
                                                             ConfigOptions)

    # Assign the temperature/pressure grids to temporary placeholders here.
    # these will be used if 2-meter specific humidity is downscaled.
    if input_forcings.q2dDownscaleOpt != 0:
        input_forcings.t2dTmp = input_forcings.final_forcings[4,:,:]*1.0
        input_forcings.psfcTmp = input_forcings.final_forcings[6,:,:]*1.0
    else:
        input_forcings.t2dTmp = None
        input_forcings.psfcTmp = None

def no_bias_correct(input_forcings,ConfigOptions):
    """
    Generic routine to simply pass forcing states through without any
    bias correction.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    input_forcings.final_forcings = input_forcings.final_forcings