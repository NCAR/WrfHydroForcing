"""
Module file for handling of downscaling input regridded (possibly bias-corrected)
forcing fields. Each input forcing product will loop through and determine if
downscaling is needed, based off options specified by the user.
"""
def run_downscaling(input_forcings,ConfigOptions):
    """
    Top level module function that will downscale forcing variables
    for this particular input forcing product.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    # Dictionary mapping to temperature downscaling.
    downscale_temperature = {
        0: no_downscale,
        1: simple_lapse,
        2: param_lapse
    }
    downscale_temperature[input_forcings.t2dDownscaleOpt](input_forcings,ConfigOptions)

    # Dictionary mapping to pressure downscaling.
    downscale_pressure = {
        0: no_downscale,
        1: pressure_down_classic
    }
    downscale_pressure[input_forcings.psfcDownscaleOpt](input_forcings,ConfigOptions)

    # Dictionary mapping to shortwave radiation downscaling
    downscale_sw = {
        0: no_downscale,
        1: ncar_topo_adj
    }
    downscale_sw[input_forcings.swDowscaleOpt](input_forcings,ConfigOptions)

    # Dictionary mapping to specific humidity downscaling
    downscale_q2 = {
        0: no_downscale,
        1: q2_down_classic
    }
    downscale_q2[input_forcings.q2dDownscaleOpt](input_forcings,ConfigOptions)

    # Dictionary mapping to precipitation downscaling.
    downscale_precip = {
        0: no_downscale,
        1: precip_mtn_mapper
    }
    downscale_precip[input_forcings.precipDownscaleOpt](input_forcings,ConfigOptions)

def no_downscale(input_forcings,ConfigOptions):
    """
    Generic function for passing states through without any
    downscaling.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    input_forcings.final_forcings = input_forcings.final_forcings
