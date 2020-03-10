"""
Module file for incorporating bias correction for various input
forcing variables. Calling tree will be guided based on user-specified
options.
"""
import math
from math import tau as TWO_PI
import os
import random
import time

import numpy as np
from netCDF4 import Dataset

from core import err_handler

NumpyExceptions = (IndexError, ValueError, AttributeError, ArithmeticError)

# These come from the netCDF4 module, but they're not exported via __all__ so we put them here:
default_fillvals = {'S1': '\x00', 'i1': -127, 'u1': 255, 'i2': -32767, 'u2': 65535, 'i4': -2147483647,
                    'u4': 4294967295, 'i8': -9223372036854775806, 'u8': 18446744073709551614,
                    'f4': 9.969209968386869e+36, 'f8': 9.969209968386869e+36}


def run_bias_correction(input_forcings, config_options, geo_meta_wrf_hydro, mpi_config):
    """
    Top level calling routine for initiating bias correction on
    this particular input forcing. This is called prior to downscaling,
    but after regridding.
    :param mpi_config:
    :param geo_meta_wrf_hydro:
    :param input_forcings:
    :param config_options:
    :return:
    """
    # Dictionary for mapping to temperature bias correction.
    bias_correct_temperature = {
        0: no_bias_correct,
        1: cfsv2_nldas_nwm_bias_correct,
        2: ncar_tbl_correction,
        3: ncar_temp_gfs_bias_correct,
        4: ncar_temp_hrrr_bias_correct
    }
    bias_correct_temperature[input_forcings.t2dBiasCorrectOpt](input_forcings, config_options, mpi_config, 0)
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary for mapping to humidity bias correction.
    bias_correct_humidity = {
        0: no_bias_correct,
        1: cfsv2_nldas_nwm_bias_correct,
        2: ncar_tbl_correction
    }
    bias_correct_humidity[input_forcings.q2dBiasCorrectOpt](input_forcings, config_options, mpi_config, 1)
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary for mapping to surface pressure bias correction.
    bias_correct_pressure = {
        0: no_bias_correct,
        1: cfsv2_nldas_nwm_bias_correct
    }
    bias_correct_pressure[input_forcings.psfcBiasCorrectOpt](input_forcings, config_options, mpi_config, 7)
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary for mapping to incoming shortwave radiation correction.
    bias_correct_sw = {
        0: no_bias_correct,
        1: cfsv2_nldas_nwm_bias_correct,
        2: ncar_sw_hrrr_bias_correct
    }
    if input_forcings.swBiasCorrectOpt != 2:
        bias_correct_sw[input_forcings.swBiasCorrectOpt](input_forcings, config_options, mpi_config, 5)
    else:
        bias_correct_sw[input_forcings.swBiasCorrectOpt](input_forcings, geo_meta_wrf_hydro,
                                                         config_options, mpi_config, 5)
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary for mapping to incoming longwave radiation correction.
    bias_correct_lw = {
        0: no_bias_correct,
        1: cfsv2_nldas_nwm_bias_correct,
        2: ncar_blanket_adjustment_lw,
        3: ncar_lwdown_gfs_bias_correct
    }
    bias_correct_lw[input_forcings.lwBiasCorrectOpt](input_forcings, config_options, mpi_config, 6)
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary for mapping to wind bias correction.
    bias_correct_wind = {
        0: no_bias_correct,
        1: cfsv2_nldas_nwm_bias_correct,
        2: ncar_tbl_correction,
        3: ncar_wspd_gfs_bias_correct,
        4: ncar_wspd_hrrr_bias_correct
    }
    # Run for U-Wind
    bias_correct_wind[input_forcings.windBiasCorrectOpt](input_forcings, config_options, mpi_config, 2)
    err_handler.check_program_status(config_options, mpi_config)
    # Run for V-Wind
    bias_correct_wind[input_forcings.windBiasCorrectOpt](input_forcings, config_options, mpi_config, 3)
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary for mapping to precipitation bias correction.
    bias_correct_precip = {
        0: no_bias_correct,
        1: cfsv2_nldas_nwm_bias_correct
    }
    bias_correct_precip[input_forcings.precipBiasCorrectOpt](input_forcings, config_options, mpi_config, 4)
    err_handler.check_program_status(config_options, mpi_config)

    # Assign the temperature/pressure grids to temporary placeholders here.
    # these will be used if 2-meter specific humidity is downscaled.
    if input_forcings.q2dDownscaleOpt != 0:
        input_forcings.t2dTmp = input_forcings.final_forcings[4, :, :] * 1.0
        input_forcings.psfcTmp = input_forcings.final_forcings[6, :, :] * 1.0
    else:
        input_forcings.t2dTmp = None
        input_forcings.psfcTmp = None


def no_bias_correct(input_forcings, config_options, mpi_config, force_num):
    """
    Generic routine to simply pass forcing states through without any
    bias correction.
    :param mpi_config:
    :param input_forcings:
    :param config_options:
    :param force_num:
    :return:
    """
    try:
        input_forcings.final_forcings[force_num, :, :] = input_forcings.final_forcings[force_num, :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to set final forcings during bias correction routine: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def ncar_tbl_correction(input_forcings, config_options, mpi_config, force_num):
    """
    Generic NCAR bias correction for forcings based on the forecast hour. A lookup table
    is used for each different forcing variable. NOTE!!!! - This is based on HRRRv3 analysis
    and should be used with extreme caution.
    :param input_forcings:
    :param config_options:
    :param mpi_config:
    :param force_num:
    :return:
    """
    # Establish lookup tables for each forcing, for each forecast hour.
    adj_tbl = {
        0: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0],
        1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0],
        2: [0.35, 0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.01,
            -0.01, -0.02, -0.03, -0.4, -0.05],
        3: [0.35, 0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.01,
            -0.01, -0.02, -0.03, -0.4, -0.05]
    }
    if mpi_config.rank == 0:
        config_options.statusMsg = "Performing table lookup bias correction for: " + \
                                   input_forcings.netcdf_var_names[force_num] + " For input: " + \
                                   input_forcings.productName
        err_handler.log_msg(config_options, mpi_config)

    # First check to make sure we are within the accepted forecast range per the above table. For now, this
    # bias correction only applies to the first 18 forecast hours.
    if int(input_forcings.fcst_hour2) > 18:
        config_options.statusMsg = "Current forecast hour for: " + input_forcings.productName + \
                                   " is greater than allowed forecast range of 18 for table lookup bias correction."
        err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Extract local array of values to perform adjustment on.
    force_tmp = None
    try:
        force_tmp = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_num] + \
                                " from local forcing object for: " + input_forcings.productName + \
                                "(" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    ind_valid = None
    try:
        ind_valid = np.where(force_tmp != config_options.globalNdv)
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to perform valid search for: " + input_forcings.netcdf_var_names[force_num] + \
                                " from local forcing object for: " + input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Apply the bias correction adjustment based on the current forecast hour.
    try:
        force_tmp[ind_valid] = force_tmp[ind_valid] + adj_tbl[force_num][int(input_forcings.fcst_hour2)]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to apply table bias correction for: " + \
                                input_forcings.netcdf_var_names[force_num] + \
                                " from local forcing object for: " + input_forcings.productName + \
                                " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = force_tmp[:, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place temporary LW array back into forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Reset temporary variables to keep low memory footprint.
    del force_tmp
    del ind_valid


def ncar_blanket_adjustment_lw(input_forcings, config_options, mpi_config, force_num):
    """
    Generic NCAR bias correction for incoming longwave radiation fluxes. NOTE!!! - This is based
    off HRRRv3 analysis and should be used with extreme caution.....
    :param input_forcings:
    :param config_options:
    :param mpi_config:
    :param force_num:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = "Performing blanket bias correction on incoming longwave " \
                                   "radiation fluxes for input: " + \
                                   input_forcings.productName
        err_handler.log_msg(config_options, mpi_config)

    # Establish blanket adjustment to apply across the board in W/m^2
    adj_lw = 9.0

    # Perform adjustment.
    lw_tmp = None
    try:
        lw_tmp = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract incoming LW from forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    ind_valid = None
    try:
        ind_valid = np.where(lw_tmp != config_options.globalNdv)
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to calculate valid index in incoming LW for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        lw_tmp[ind_valid] = lw_tmp[ind_valid] + adj_lw
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to perform LW bias correction for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = lw_tmp[:, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place temporary LW array back into forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Reset temporary variables to keep low memory footprint.
    del lw_tmp
    del ind_valid


def ncar_sw_hrrr_bias_correct(input_forcings, geo_meta_wrf_hydro, config_options, mpi_config, force_num):
    """
    Function to implement a bias correction to the forecast incoming shortwave radiation fluxes.
    NOTE!!!! - This bias correction is based on in-situ analysis performed against HRRRv3
    fields. It's high discouraged to use this for any other NWP products, or even with the HRRR
    changes versions in the future.
    :param input_forcings:
    :param geo_meta_wrf_hydro:
    :param config_options:
    :param mpi_config:
    :param force_num:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = "Performing NCAR bias correction on incoming shortwave " \
                                   "radiation fluxes for input: " + \
                                   input_forcings.productName
        err_handler.log_msg(config_options, mpi_config)

    # Establish constant parameters. NOTE!!! - These will change with future HRRR upgrades.
    c1 = -0.159
    c2 = -0.077

    # Establish current datetime information, along wth solar constants.
    f_hr = input_forcings.fcst_hour2
    # For now, hard-coding the total number of forecast hours to be 18, since we
    # are assuming this is HRRR
    n_fcst_hr = 18

    # Trig params
    d2r = math.pi / 180.0
    r2d = 180.0 / math.pi
    date_current = config_options.current_output_date
    hh = float(date_current.hour)
    mm = float(date_current.minute)
    ss = float(date_current.second)
    doy = float(time.strptime(date_current.strftime('%Y.%m.%d'), '%Y.%m.%d').tm_yday)
    frac_year = 2.0 * math.pi / 365.0 * (doy - 1.0 + (hh / 24.0) + (mm / 1440.0) + (ss / 86400.0))
    # eqtime is the difference in minutes between true solar time and that if solar noon was at actual noon.
    # This difference is due to Earth's elliptical orbit around the sun.
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(frac_year) - 0.032077 * math.sin(frac_year) -
                       0.014615 * math.cos(2.0 * frac_year) - 0.040849 * math.sin(2.0 * frac_year))

    # decl is the solar declination angle in radians: how much the Earth is tilted toward or away from the sun
    decl = 0.006918 - 0.399912 * math.cos(frac_year) + 0.070257 * math.sin(frac_year) - \
        0.006758 * math.cos(2.0 * frac_year) + 0.000907 * math.sin(2.0 * frac_year) - \
        0.002697 * math.cos(3.0 * frac_year) + 0.001480 * math.sin(3.0 * frac_year)

    # Create temporary grids for calculating the solar zenith angle, which will be used in the bias correction.
    # time offset in minutes from the prime meridian
    time_offset = eqtime + 4.0 * geo_meta_wrf_hydro.longitude_grid

    # tst is the true solar time: the number of minutes since solar midnight
    tst = hh * 60.0 + mm + ss / 60.0 + time_offset

    # solar hour angle in radians: the amount the sun is off from due south
    ha = d2r * ((tst / 4.0) - 180.0)

    # solar zenith angle is the angle between straight up and the center of the sun's disc
    # the cosine of the sol_zen_ang is proportional to the solar intensity
    # (not accounting for humidity or cloud cover)
    sol_zen_ang = r2d * np.arccos(np.sin(geo_meta_wrf_hydro.latitude_grid * d2r) * math.sin(decl) +
                                  np.cos(geo_meta_wrf_hydro.latitude_grid * d2r) * math.cos(decl) * np.cos(ha))

    # Check for any values greater than 90 degrees.
    sol_zen_ang[np.where(sol_zen_ang > 90.0)] = 90.0

    # Extract the current incoming shortwave field from the forcing object and set it to
    # a local grid. We will perform the bias correction on this grid, based on forecast
    # hour and datetime information. Once a correction has taken place, we will place
    # the corrected field back into the forcing object.
    sw_tmp = None
    try:
        sw_tmp = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract incoming shortwave forcing from object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Calculate where we have valid values.
    ind_valid = None
    try:
        ind_valid = np.where(sw_tmp != config_options.globalNdv)
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to run a search for valid SW values for: " + input_forcings.productName + \
                                " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Perform the bias correction.
    try:
        # The second half of this calculation below is the actual calculation of the incoming SW bias, which is then
        # added (or subtracted if negative) to the original values.
        sw_tmp[ind_valid] = sw_tmp[ind_valid] + \
                          (c1 + (c2 * ((f_hr - 1) / (n_fcst_hr - 1)))) * np.cos(sol_zen_ang[ind_valid] * d2r) * \
                          sw_tmp[ind_valid]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to apply NCAR HRRR bias correction to incoming shortwave radiation: " + \
                                str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Place updated states back into the forcing object.
    try:
        input_forcings.final_forcings[7, :, :] = sw_tmp[:, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place bias-corrected incoming SW radiation fluxes back into the forcing " \
                                "object: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Reset variables to keep memory footprints low.
    del sw_tmp
    del time_offset
    del tst
    del ha
    del sol_zen_ang
    del ind_valid


def ncar_temp_hrrr_bias_correct(input_forcings, config_options, mpi_config, force_num):
    if mpi_config.rank == 0:
        config_options.statusMsg = "Performing NCAR HRRR bias correction on incoming 2m temperature input: " + \
                                   input_forcings.productName
        err_handler.log_msg(config_options, mpi_config)

    date_current = config_options.current_output_date
    hh = float(date_current.hour)
    MM= float(date_current.month)

    # determine if we're in AnA or SR configuration
    if config_options.ana_flag == 1:
        net_bias_AA = 0.13
        diurnal_ampl_AA = -0.18
        diurnal_offs_AA = 2.2
        monthly_ampl_AA = -0.15
        monthly_offs_AA = -2.0

        bias_corr = net_bias_AA + diurnal_ampl_AA * math.sin(diurnal_offs_AA + hh / 24 * 2 * math.pi) + \
                    monthly_ampl_AA * math.sin(monthly_offs_AA + MM / 12 * 2*math.pi)

    else:
        net_bias_SR = 0.060
        diurnal_ampl_SR = -0.31
        diurnal_offs_SR = 2.2
        monthly_ampl_SR = -0.21
        monthly_offs_SR = -2.4
        fhr_mult_SR = -0.016

        fhr = config_options.current_output_step

        bias_corr = net_bias_SR + fhr * fhr_mult_SR + \
                    diurnal_ampl_SR * math.sin(diurnal_offs_SR + hh / 24 * 2*math.pi) + \
                    monthly_ampl_SR * math.sin(monthly_offs_SR + MM / 12 * 2*math.pi)

    if mpi_config.rank == 0:
        config_options.statusMsg = f"\tAnAFlag = {config_options.ana_flag} {bias_corr}"
        err_handler.log_msg(config_options, mpi_config)

    temp_in = None
    try:
        temp_in = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract incoming temperature from forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    ind_valid = None
    try:
        ind_valid = np.where(temp_in != config_options.globalNdv)
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to calculate valid index in incoming temperature for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        temp_in[ind_valid] = temp_in[ind_valid] + bias_corr
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to perform temperature bias correction for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = temp_in[:, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place temporary temperature array back into forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    del temp_in
    del ind_valid


def ncar_temp_gfs_bias_correct(input_forcings, config_options, mpi_config, force_num):
    if mpi_config.rank == 0:
        config_options.statusMsg = "Performing NCAR GFS bias correction on incoming 2m temperature input: " + \
                                   input_forcings.productName
        err_handler.log_msg(config_options, mpi_config)

    date_current = config_options.current_output_date
    hh = float(date_current.hour)

    net_bias_mr = -0.18
    fhr_mult_mr = 0.002
    diurnal_ampl_mr = -1.4
    diurnal_offs_mr = -2.1

    fhr = config_options.current_output_step

    bias_corr = net_bias_mr + fhr_mult_mr * fhr + diurnal_ampl_mr * math.sin(diurnal_offs_mr + hh / 24 * TWO_PI)

    temp_in = None
    try:
        temp_in = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract incoming temperature from forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    ind_valid = None
    try:
        ind_valid = np.where(temp_in != config_options.globalNdv)
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to calculate valid index in incoming temperature for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        temp_in[ind_valid] = temp_in[ind_valid] + bias_corr
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to perform temperature bias correction for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = temp_in[:, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place temporary temperature array back into forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    del temp_in
    del ind_valid


def ncar_lwdown_gfs_bias_correct(input_forcings, config_options, mpi_config, force_num):
    if mpi_config.rank == 0:
        config_options.statusMsg = "Performing NCAR bias correction on incoming longwave " \
                                   "radiation fluxes for input: " + \
                                   input_forcings.productName
        err_handler.log_msg(config_options, mpi_config)

    date_current = config_options.current_output_date
    hh = float(date_current.hour)

    fhr = config_options.current_output_step

    lwdown_net_bias_mr = 9.9
    lwdown_fhr_mult_mr = 0.00
    lwdown_diurnal_ampl_mr = -1.5
    lwdown_diurnal_offs_mr = 2.8

    bias_corr = lwdown_net_bias_mr + lwdown_fhr_mult_mr * fhr + lwdown_diurnal_ampl_mr * \
                math.sin(lwdown_diurnal_offs_mr + hh / 24 * TWO_PI)

    lwdown_in = None
    try:
        lwdown_in = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract incoming longwave from forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    ind_valid = None
    try:
        ind_valid = np.where(lwdown_in != config_options.globalNdv)
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to calculate valid index in incoming longwave for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        lwdown_in[ind_valid] = lwdown_in[ind_valid] + bias_corr
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to perform longwave bias correction for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = lwdown_in[:, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place temporary longwave array back into forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    del lwdown_in
    del ind_valid


def ncar_wspd_hrrr_bias_correct(input_forcings, config_options, mpi_config, force_num):
    if mpi_config.rank == 0:
        config_options.statusMsg = "Performing NCAR bias correction on incoming windspeed for input: " + \
                                   input_forcings.productName + " at step " + str(config_options.current_output_step)
        err_handler.log_msg(config_options, mpi_config)

    date_current = config_options.current_output_date
    hh = float(date_current.hour)

    fhr = config_options.current_output_step

    # need to get wind speed from U, V components
    ugrd_idx = input_forcings.grib_vars.index('UGRD')
    vgrd_idx = input_forcings.grib_vars.index('VGRD')

    ugrid_in = input_forcings.final_forcings[input_forcings.input_map_output[ugrd_idx], :, :]
    vgrid_in = input_forcings.final_forcings[input_forcings.input_map_output[vgrd_idx], :, :]

    wdir = np.arctan2(vgrid_in, ugrid_in)
    wspd = np.sqrt(np.square(ugrid_in) + np.square(vgrid_in))

    if config_options.ana_flag:
        wspd_bias_corr = 0.35       # fixed for AnA
    else:
        wspd_net_bias_sr = [0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05,
                            0.03, 0.02, 0.01, -0.01, -0.02, -0.03, -0.04, -0.05]
        wspd_bias_corr = wspd_net_bias_sr[config_options.current_output_step - 1]

    wspd = wspd + wspd_bias_corr
    wspd = np.where(wspd < 0, 0, wspd)

    ugrid_out = wspd * np.cos(wdir)
    vgrid_out = wspd * np.sin(wdir)
    # TODO: cache the "other" value so we don't repeat this calculation unnecessarily

    bias_corrected = ugrid_out if force_num == ugrd_idx else vgrid_out

    wind_in = None
    try:
        wind_in = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract incoming windspeed from forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    ind_valid = None
    try:
        ind_valid = np.where(wind_in != config_options.globalNdv)
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to calculate valid index in incoming windspeed for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    try:
        wind_in[ind_valid] = bias_corrected[ind_valid]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to perform windspeed bias correction for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = wind_in[:, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place temporary windspeed array back into forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)

    err_handler.check_program_status(config_options, mpi_config)

    del wind_in
    del ind_valid


def ncar_wspd_gfs_bias_correct(input_forcings, config_options, mpi_config, force_num):
    if mpi_config.rank == 0:
        config_options.statusMsg = "Performing NCAR bias correction on incoming windspeed for input: " + \
                                   input_forcings.productName
        err_handler.log_msg(config_options, mpi_config)

    date_current = config_options.current_output_date
    hh = float(date_current.hour)

    fhr = config_options.current_output_step
    wspd_net_bias_mr = -0.20
    wspd_fhr_mult_mr = 0.00
    wspd_diurnal_ampl_mr = -0.32
    wspd_diurnal_offs_mr = -1.1

    # need to get wind speed from U, V components
    ugrd_idx = input_forcings.grib_vars.index('UGRD')
    vgrd_idx = input_forcings.grib_vars.index('VGRD')

    ugrid_in = input_forcings.final_forcings[input_forcings.input_map_output[ugrd_idx], :, :]
    vgrid_in = input_forcings.final_forcings[input_forcings.input_map_output[vgrd_idx], :, :]

    wdir = np.arctan2(vgrid_in, ugrid_in)
    wspd = np.sqrt(np.square(ugrid_in) + np.square(vgrid_in))

    wspd_bias_corr = wspd_net_bias_mr + wspd_fhr_mult_mr * fhr + \
                wspd_diurnal_ampl_mr * math.sin(wspd_diurnal_offs_mr + hh / 24 * TWO_PI)

    wspd = wspd + wspd_bias_corr
    wspd = np.where(wspd < 0, 0, wspd)

    ugrid_out = wspd * np.cos(wdir)
    vgrid_out = wspd * np.sin(wdir)

    # TODO: cache the "other" value so we don't repeat this calculation unnecessarily

    bias_corrected = ugrid_out if force_num == ugrd_idx else vgrid_out

    wind_in = None
    try:
        wind_in = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract incoming windspeed from forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    ind_valid = None
    try:
        ind_valid = np.where(wind_in != config_options.globalNdv)
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to calculate valid index in incoming windspeed for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    try:
        wind_in[ind_valid] = bias_corrected[ind_valid]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to perform windspeed bias correction for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = wind_in[:, :]
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place temporary windspeed array back into forcing object for: " + \
                                input_forcings.productName + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)

    err_handler.check_program_status(config_options, mpi_config)

    del wind_in
    del ind_valid


def cfsv2_nldas_nwm_bias_correct(input_forcings, config_options, mpi_config, force_num):
    """
    Routine to run CDF/PDF bias correction parametric corrections
    SPECIFIC to the NWM long-range configuration.
    :param mpi_config:
    :param input_forcings:
    :param config_options:
    :param force_num:
    :return:
    """

    # TODO: move these into a (.py or .json) configuration file
    # Create a dictionary that maps forcing numbers to the expected NetCDF variable names, etc.
    nldas_param1_vars = {
        2: 'UGRD10M_PARAM_1',
        3: 'VGRD10M_PARAM_1',
        6: 'LW_PARAM_1',
        4: 'PRATE_PARAM_1',
        0: 'T2M_PARAM_1',
        1: 'Q2M_PARAM_1',
        7: 'PSFC_PARAM_1',
        5: 'SW_PARAM_1'
    }

    nldas_param2_vars = {
        2: 'UGRD10M_PARAM_2',
        3: 'VGRD10M_PARAM_2',
        6: 'LW_PARAM_2',
        4: 'PRATE_PARAM_2',
        0: 'T2M_PARAM_2',
        1: 'Q2M_PARAM_2',
        7: 'PSFC_PARAM_2',
        5: 'SW_PARAM_2'
    }

    cfs_param_path_vars = {
        2: 'ugrd',
        3: 'vgrd',
        6: 'dlwsfc',
        4: 'prate',
        0: 'tmp2m',
        1: 'q2m',
        7: 'pressfc',
        5: 'dswsfc'
    }

    # Specify the min/max ranges on CDF/PDF values for each variable
    val_range1 = {
        2: -50.0,
        3: -50.0,
        6: 1.0,
        4: 0.01,
        0: 200.0,
        1: 0.01,
        7: 50000.0,
        5: 0.0
    }

    val_range2 = {
        2: 50.0,
        3: 50.0,
        6: 800.0,
        4: 100.0,
        0: 330.0,
        1: 40.0,
        7: 1100000.0,
        5: 1000.0
    }

    val_step = {
        2: 0.1,
        3: 0.1,
        6: 0.19975,
        4: 0.049995,
        0: 0.1,
        1: 3.9999,
        7: 350.0,
        5: 10.0
    }

    if mpi_config.rank == 0:
        config_options.statusMsg = "Running NLDAS-CFSv2 CDF/PDF bias correction on variable: " + \
                                   input_forcings.netcdf_var_names[force_num]
        err_handler.log_msg(config_options, mpi_config)

    # Check to ensure we are running with CFSv2 here....
    if input_forcings.productName != "CFSv2_6Hr_Global_GRIB2":
        config_options.errMsg = "Attempting to run CFSv2-NLDAS bias correction on: " + input_forcings.productName
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Open the necessary parameter grids, which are on the global CFSv2 grid, then scatter them out
    # to the various processors.
    id_nldas_param = nldas_param_file = None
    if mpi_config.rank == 0:
        nldas_param_file = input_forcings.paramDir + "/NLDAS_Climo/nldas2_" + \
                           config_options.current_output_date.strftime('%m%d%H') + \
                           "_dist_params.nc"
        if not os.path.isfile(nldas_param_file):
            config_options.errMsg = "Unable to locate necessary bias correction parameter file: " + \
                                    nldas_param_file
            err_handler.log_critical(config_options, mpi_config)

        # Open the NetCDF file.
        try:
            id_nldas_param = Dataset(nldas_param_file, 'r')
        except OSError as err:
            config_options.errMsg = "Unable to open parameter file: " + nldas_param_file + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
            raise err

        # Ensure dimensions/variables are as expected.
        if 'lat_0' not in id_nldas_param.dimensions.keys():
            config_options.errMsg = "Expected to find lat_0 dimension in: " + nldas_param_file
            err_handler.log_critical(config_options, mpi_config)

        if 'lon_0' not in id_nldas_param.dimensions.keys():
            config_options.errMsg = "Expected to find lon_0 dimension in: " + nldas_param_file
            err_handler.log_critical(config_options, mpi_config)

        if id_nldas_param.dimensions['lat_0'].size != 190:
            config_options.errMsg = "Expected lat_0 size is 190 - found size of: " + \
                                    str(id_nldas_param.dimensions['lat_0'].size) + " in: " + nldas_param_file
            err_handler.log_critical(config_options, mpi_config)

        if id_nldas_param.dimensions['lon_0'].size != 384:
            config_options.errMsg = "Expected lon_0 size is 384 - found size of: " + \
                                    str(id_nldas_param.dimensions['lon_0'].size) + " in: " + nldas_param_file
            err_handler.log_critical(config_options, mpi_config)

        if nldas_param1_vars[force_num] not in id_nldas_param.variables.keys():
            config_options.errMsg = "Expected variable: " + nldas_param1_vars[force_num] + " not found " + \
                                   "in: " + nldas_param_file
            err_handler.log_critical(config_options, mpi_config)

        if nldas_param2_vars[force_num] not in id_nldas_param.variables.keys():
            config_options.errMsg = "Expected variable: " + nldas_param2_vars[force_num] + " not found " + \
                                   "in: " + nldas_param_file
            err_handler.log_critical(config_options, mpi_config)

        if force_num == 4:
            if 'ZERO_PRECIP_PROB' not in id_nldas_param.variables.keys():
                config_options.errMsg = "Expected variable: ZERO_PRECIP_PROB not found in: " + \
                                        nldas_param_file
                err_handler.log_critical(config_options, mpi_config)

        nldas_param_1 = None
        try:
            nldas_param_1 = id_nldas_param.variables[nldas_param1_vars[force_num]][:, :]
        except NumpyExceptions as npe:
            config_options.errMsg = "Unable to extract: " + nldas_param1_vars[force_num] + \
                                   " from: " + nldas_param_file + " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)

        nldas_param_2 = None
        try:
            nldas_param_2 = id_nldas_param.variables[nldas_param2_vars[force_num]][:, :]
        except NumpyExceptions as npe:
            config_options.errMsg = "Unable to extract: " + nldas_param2_vars[force_num] + \
                                   " from: " + nldas_param_file + " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)

        if nldas_param_1.shape[0] != 190 or nldas_param_1.shape[1] != 384:
            config_options.errMsg = "Parameter variable: " + nldas_param1_vars[force_num] + " from: " + \
                                    nldas_param_file + " not of shape [190,384]."
            err_handler.log_critical(config_options, mpi_config)

        if nldas_param_2.shape[0] != 190 or nldas_param_2.shape[1] != 384:
            config_options.errMsg = "Parameter variable: " + nldas_param2_vars[force_num] + " from: " + \
                                    nldas_param_file + " not of shape [190,384]."
            err_handler.log_critical(config_options, mpi_config)

        # Extract the fill value
        fill_tmp = None
        try:
            try:
                fill_tmp = id_nldas_param.variables[nldas_param1_vars[force_num]].getncattr('_FillValue')
            except AttributeError:
                fill_tmp = default_fillvals[id_nldas_param.variables[nldas_param1_vars[force_num]].dtype.str[1:]]
        except NumpyExceptions as npe:
            config_options.errMsg = "Unable to extract _FillValue from: " + nldas_param_file + " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)

        # Read in the zero precip prob grids if we are bias correcting precipitation.
        nldas_zero_pcp = None
        if force_num == 4:
            try:
                nldas_zero_pcp = id_nldas_param.variables['ZERO_PRECIP_PROB'][:, :]
            except NumpyExceptions as npe:
                config_options.errMsg = "Unable to extract ZERO_PRECIP_PROB from: " + nldas_param_file + \
                                        " (" + str(npe) + ")"
                err_handler.log_critical(config_options, mpi_config)

            if nldas_zero_pcp.shape[0] != 190 or nldas_zero_pcp.shape[1] != 384:
                config_options.errMsg = "Parameter variable: ZERO_PRECIP_PROB from: " + nldas_param_file + \
                                       " not of shape [190,384]."
                err_handler.log_critical(config_options, mpi_config)

        # Set missing values accordingly.
        nldas_param_1[np.where(nldas_param_1 == fill_tmp)] = config_options.globalNdv
        nldas_param_2[np.where(nldas_param_1 == fill_tmp)] = config_options.globalNdv
        if force_num == 4:
            nldas_zero_pcp[np.where(nldas_zero_pcp == fill_tmp)] = config_options.globalNdv

    else:
        nldas_param_1 = None
        nldas_param_2 = None
        nldas_zero_pcp = None
    err_handler.check_program_status(config_options, mpi_config)

    # Scatter NLDAS parameters
    nldas_param_1_sub = mpi_config.scatter_array(input_forcings, nldas_param_1, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    nldas_param_2_sub = mpi_config.scatter_array(input_forcings, nldas_param_2, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    if force_num == 4:
        nldas_zero_pcp_sub = mpi_config.scatter_array(input_forcings, nldas_zero_pcp, config_options)
        err_handler.check_program_status(config_options, mpi_config)
    else:
        nldas_zero_pcp_sub = None

    id_cfs_param1 = id_cfs_param2 = None
    cfs_param_path1 = cfs_param_path2 = None
    if mpi_config.rank == 0:
        # Read in the CFSv2 parameter files, based on the previous CFSv2 dates
        cfs_param_path1 = (input_forcings.paramDir + "/CFSv2_Climo/cfs_" +
                           cfs_param_path_vars[force_num] + "_" +
                           input_forcings.fcst_date1.strftime('%m%d') + "_" +
                           input_forcings.fcst_date1.strftime('%H') + '_dist_params.nc')

        cfs_param_path2 = (input_forcings.paramDir + "/CFSv2_Climo/cfs_" +
                           cfs_param_path_vars[force_num] + "_" +
                           input_forcings.fcst_date2.strftime('%m%d') + "_" +
                           input_forcings.fcst_date2.strftime('%H') + '_dist_params.nc')

        if not os.path.isfile(cfs_param_path1):
            config_options.errMsg = "Unable to locate necessary parameter file: " + cfs_param_path1
            err_handler.log_critical(config_options, mpi_config)
        if not os.path.isfile(cfs_param_path2):
            config_options.errMsg = "Unable to locate necessary parameter file: " + cfs_param_path2
            err_handler.log_critical(config_options, mpi_config)

        # Open the files and ensure they contain the correct information.
        try:
            id_cfs_param1 = Dataset(cfs_param_path1, 'r')
        except OSError as err:
            config_options.errMsg = "Unable to open parameter file: " + cfs_param_path1 + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
        try:
            id_cfs_param2 = Dataset(cfs_param_path2, 'r')
        except OSError as err:
            config_options.errMsg = "Unable to open parameter file: " + cfs_param_path2 + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)

        config_options.statusMsg = "Checking CFS parameter files."
        err_handler.log_msg(config_options, mpi_config)
        if 'DISTRIBUTION_PARAM_1' not in id_cfs_param1.variables.keys():
            config_options.errMsg = "Expected DISTRIBUTION_PARAM_1 variable not found in: " + cfs_param_path1
            err_handler.log_critical(config_options, mpi_config)
        if 'DISTRIBUTION_PARAM_2' not in id_cfs_param1.variables.keys():
            config_options.errMsg = "Expected DISTRIBUTION_PARAM_1 variable not found in: " + cfs_param_path1
            err_handler.log_critical(config_options, mpi_config)

        if 'DISTRIBUTION_PARAM_1' not in id_cfs_param2.variables.keys():
            config_options.errMsg = "Expected DISTRIBUTION_PARAM_1 variable not found in: " + cfs_param_path2
            err_handler.log_critical(config_options, mpi_config)
        if 'DISTRIBUTION_PARAM_2' not in id_cfs_param2.variables.keys():
            config_options.errMsg = "Expected DISTRIBUTION_PARAM_1 variable not found in: " + cfs_param_path2
            err_handler.log_critical(config_options, mpi_config)

        param_1 = param_2 = None
        try:
            param_1 = id_cfs_param2.variables['DISTRIBUTION_PARAM_1'][:, :]
        except NumpyExceptions as npe:
            config_options.errMsg = "Unable to extract DISTRIBUTION_PARAM_1 from: " + cfs_param_path2 + \
                                    " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)
        try:
            param_2 = id_cfs_param2.variables['DISTRIBUTION_PARAM_2'][:, :]
        except NumpyExceptions as npe:
            config_options.errMsg = "Unable to extract DISTRIBUTION_PARAM_2 from: " + cfs_param_path2 + \
                                    " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)

        # try:
        #     lat_0 = id_cfs_param2.variables['lat_0'][:]
        # except NumpyExceptions as npe:
        #     config_options.errMsg = "Unable to extract lat_0 from: " + cfs_param_path2 + " (" + str(npe) + ")"
        #     err_handler.log_critical(config_options, mpi_config)
        # try:
        #     lon_0 = id_cfs_param2.variables['lon_0'][:]
        # except NumpyExceptions as npe:
        #     config_options.errMsg = "Unable to extract lon_0 from: " + cfs_param_path2 + " (" + str(npe) + ")"
        #     err_handler.log_critical(config_options, mpi_config)

        prev_param_1 = prev_param_2 = None
        try:
            prev_param_1 = id_cfs_param1.variables['DISTRIBUTION_PARAM_1'][:, :]
        except NumpyExceptions as npe:
            config_options.errMsg = "Unable to extract DISTRIBUTION_PARAM_1 from: " + cfs_param_path1 + \
                                    " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)
        try:
            prev_param_2 = id_cfs_param1.variables['DISTRIBUTION_PARAM_2'][:, :]
        except NumpyExceptions as npe:
            config_options.errMsg = "Unable to extract DISTRIBUTION_PARAM_2 from: " + cfs_param_path1 + \
                                    " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)

        if param_1.shape[0] != 190 and param_1.shape[1] != 384:
            config_options.errMsg = "Unexpected DISTRIBUTION_PARAM_1 found in: " + cfs_param_path2
            err_handler.log_critical(config_options, mpi_config)
        if param_2.shape[0] != 190 and param_2.shape[1] != 384:
            config_options.errMsg = "Unexpected DISTRIBUTION_PARAM_2 found in: " + cfs_param_path2
            err_handler.log_critical(config_options, mpi_config)

        if prev_param_1.shape[0] != 190 and prev_param_1.shape[1] != 384:
            config_options.errMsg = "Unexpected DISTRIBUTION_PARAM_1 found in: " + cfs_param_path1
            err_handler.log_critical(config_options, mpi_config)
        if prev_param_2.shape[0] != 190 and prev_param_2.shape[1] != 384:
            config_options.errMsg = "Unexpected DISTRIBUTION_PARAM_2 found in: " + cfs_param_path1
            err_handler.log_critical(config_options, mpi_config)

        config_options.statusMsg = "Reading in zero precip probs."
        err_handler.log_msg(config_options, mpi_config)
        # Read in the zero precip prob grids if we are bias correcting precipitation.
        zero_pcp = prev_zero_pcp = None
        if force_num == 4:
            try:
                zero_pcp = id_cfs_param2.variables['ZERO_PRECIP_PROB'][:, :]
            except NumpyExceptions as npe:
                config_options.errMsg = "Unable to locate ZERO_PRECIP_PROB in: " + cfs_param_path2 + \
                                        " (" + str(npe) + ")"
                err_handler.log_critical(config_options, mpi_config)
            try:
                prev_zero_pcp = id_cfs_param2.variables['ZERO_PRECIP_PROB'][:, :]
            except NumpyExceptions as npe:
                config_options.errMsg = "Unable to locate ZERO_PRECIP_PROB in: " + cfs_param_path1 + \
                                        " (" + str(npe) + ")"
                err_handler.log_critical(config_options, mpi_config)

            if zero_pcp.shape[0] != 190 and zero_pcp.shape[1] != 384:
                config_options.errMsg = "Unexpected ZERO_PRECIP_PROB found in: " + cfs_param_path2
                err_handler.log_critical(config_options, mpi_config)
            if prev_zero_pcp.shape[0] != 190 and prev_zero_pcp.shape[1] != 384:
                config_options.errMsg = "Unexpected ZERO_PRECIP_PROB found in: " + cfs_param_path1
                err_handler.log_critical(config_options, mpi_config)

        # Reset any missing values. Because the fill values for these files are all over the map, we
        # will just do a gross check here. For the most part, there shouldn't be missing values.
        param_1[np.where(param_1 > 500000.0)] = config_options.globalNdv
        param_2[np.where(param_2 > 500000.0)] = config_options.globalNdv
        prev_param_1[np.where(prev_param_1 > 500000.0)] = config_options.globalNdv
        prev_param_2[np.where(prev_param_2 > 500000.0)] = config_options.globalNdv
        if force_num == 4:
            zero_pcp[np.where(zero_pcp > 500000.0)] = config_options.globalNdv
            prev_zero_pcp[np.where(prev_zero_pcp > 500000.0)] = config_options.globalNdv

    else:
        param_1 = None
        param_2 = None
        prev_param_1 = None
        prev_param_2 = None
        zero_pcp = None
        prev_zero_pcp = None
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        config_options.statusMsg = "Scattering CFS parameter grids"
        err_handler.log_msg(config_options, mpi_config)
    # Scatter CFS parameters
    cfs_param_1_sub = mpi_config.scatter_array(input_forcings, param_1, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    cfs_param_2_sub = mpi_config.scatter_array(input_forcings, param_2, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    cfs_prev_param_1_sub = mpi_config.scatter_array(input_forcings, prev_param_1, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    cfs_prev_param_2_sub = mpi_config.scatter_array(input_forcings, prev_param_2, config_options)
    err_handler.check_program_status(config_options, mpi_config)
    if force_num == 4:
        cfs_zero_pcp_sub = mpi_config.scatter_array(input_forcings, zero_pcp, config_options)
        err_handler.check_program_status(config_options, mpi_config)
        cfs_prev_zero_pcp_sub = mpi_config.scatter_array(input_forcings, prev_zero_pcp, config_options)
        err_handler.check_program_status(config_options, mpi_config)
    else:
        cfs_prev_zero_pcp_sub = None
        cfs_zero_pcp_sub = None

    if mpi_config.rank == 0:
        config_options.statusMsg = "Closing CFS bias correction parameter files."
        err_handler.log_msg(config_options, mpi_config)
        # Close the parameter files.
        try:
            id_nldas_param.close()
        except OSError as err:
            config_options.errMsg = "Unable to close parameter file: " + nldas_param_file + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
        try:
            id_cfs_param1.close()
        except OSError as err:
            config_options.errMsg = "Unable to close parameter file: " + cfs_param_path1 + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
        try:
            id_cfs_param2.close()
        except OSError as err:
            config_options.errMsg = "Unable to close parameter file: " + cfs_param_path2 + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)

    err_handler.check_program_status(config_options, mpi_config)

    # Now.... Loop through the local CFSv2 grid cells and perform the following steps:
    # 1.) Interpolate the six-hour values to the current output timestep.
    # 2.) Calculate the CFSv2 cdf/pdf
    # 3.) Calculate the NLDAS cdf/pdf
    # 4.) Adjust CFSv2 values based on the method of pdf matching.
    # 5.) Regrid the CFSv2 values to the WRF-Hydro domain using the pre-calculated ESMF
    #     regridding object.
    # 6.) Place the data into the final output arrays for further processing (downscaling).
    # 7.) Reset variables for memory efficiency and exit the routine.

    if mpi_config.rank == 0:
        config_options.statusMsg = "Creating local CFS CDF arrays."
        err_handler.log_msg(config_options, mpi_config)
    # Establish local arrays of data.
    cfs_data = np.empty([input_forcings.ny_local, input_forcings.nx_local], np.float64)

    # Establish parameters of the CDF matching.
    vals = np.arange(val_range1[force_num], val_range2[force_num], val_step[force_num])

    if mpi_config.rank == 0:
        config_options.statusMsg = "Looping over local arrays to calculate bias corrections."
        err_handler.log_msg(config_options, mpi_config)
    # Process each of the pixel cells for this local processor on the CFS grid.
    for x_local in range(0, input_forcings.nx_local):
        for y_local in range(0, input_forcings.ny_local):
            cfs_prev_tmp = input_forcings.coarse_input_forcings1[input_forcings.input_map_output[force_num],
                                                                 y_local, x_local]
            cfs_next_tmp = input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_num],
                                                                 y_local, x_local]

            # Check for any missing parameter values. If any missing values exist,
            # set this flag to False. Further down, if it's False, we will simply
            # set the local CFS adjusted value to the interpolated value.
            correct_flag = True
            if cfs_param_1_sub[y_local, x_local] == config_options.globalNdv:
                correct_flag = False
            if cfs_param_2_sub[y_local, x_local] == config_options.globalNdv:
                correct_flag = False
            if cfs_prev_param_1_sub[y_local, x_local] == config_options.globalNdv:
                correct_flag = False
            if cfs_prev_param_2_sub[y_local, x_local] == config_options.globalNdv:
                correct_flag = False
            if nldas_param_1_sub[y_local, x_local] == config_options.globalNdv:
                correct_flag = False
            if nldas_param_2_sub[y_local, x_local] == config_options.globalNdv:
                correct_flag = False
            if force_num == 4:
                if cfs_prev_zero_pcp_sub[y_local, x_local] == config_options.globalNdv:
                    correct_flag = False
                if cfs_zero_pcp_sub[y_local, x_local] == config_options.globalNdv:
                    correct_flag = False
                if nldas_zero_pcp_sub[y_local, x_local] == config_options.globalNdv:
                    correct_flag = False

            # Interpolate the two CFS values (and parameters) in time.
            dt_from_previous = config_options.current_output_date - input_forcings.fcst_date1
            hr_from_previous = dt_from_previous.total_seconds() / 3600.0
            interp_factor1 = float(1 - (hr_from_previous / 6.0))
            interp_factor2 = float(hr_from_previous / 6.0)

            # Since this is only for CFSv2 6-hour data, we will assume 6-hour intervals.
            # This is already checked at the beginning of this routine for the product name.
            cfs_param_1_interp = (cfs_prev_param_1_sub[y_local, x_local] * interp_factor1 +
                                  cfs_param_1_sub[y_local, x_local] * interp_factor2)
            cfs_param_2_interp = (cfs_prev_param_2_sub[y_local, x_local] * interp_factor1 +
                                  cfs_param_2_sub[y_local, x_local] * interp_factor2)
            cfs_interp_fcst = cfs_prev_tmp * interp_factor1 + cfs_next_tmp * interp_factor2
            nldas_nearest_1 = nldas_param_1_sub[y_local, x_local]
            nldas_nearest_2 = nldas_param_2_sub[y_local, x_local]

            if correct_flag:
                if force_num != 4 and force_num != 5 and force_num != 1:
                    # Not incoming shortwave or precip or specific humidity
                    pts = (vals - cfs_param_1_interp) / cfs_param_2_interp
                    spacing = (vals[2] - vals[1]) / cfs_param_2_interp
                    cfs_pdf = (np.exp(-0.5 * (np.power(pts, 2))) / math.sqrt(2 * 3.141592)) * spacing
                    cfs_cdf = np.cumsum(cfs_pdf)

                    pts = (vals - nldas_nearest_1) / nldas_nearest_2
                    spacing = (vals[2] - vals[1]) / nldas_nearest_2
                    nldas_pdf = (np.exp(-0.5 * (np.power(pts, 2))) / math.sqrt(2 * 3.141592)) * spacing
                    nldas_cdf = np.cumsum(nldas_pdf)

                    # compute adjusted value now using the CFSv2 forecast value and the two CDFs
                    # find index in vals array
                    diff_tmp = np.absolute(vals - cfs_interp_fcst)
                    cfs_ind = np.where(diff_tmp == diff_tmp.min())[0][0]
                    cfs_cdf_val = cfs_cdf[cfs_ind]

                    # now whats the index of the closest cdf value in the nldas array?
                    diff_tmp = np.absolute(cfs_cdf_val - nldas_cdf)
                    cfs_nldas_ind = np.where(diff_tmp == diff_tmp.min())[0][0]

                    # Adjust the CFS data
                    cfs_data[y_local, x_local] = vals[cfs_nldas_ind]

                if force_num == 5:
                    # Incoming shortwave radiation flux.
                    # find nearest nldas grid point and then calculate nldas cdf
                    nldas_nearest_1 = nldas_param_1_sub[y_local, x_local]

                    if cfs_interp_fcst > 2.0 and cfs_param_1_interp > 2.0:
                        factor = nldas_nearest_1 / cfs_param_1_interp
                        cfs_data[y_local, x_local] = cfs_interp_fcst * factor
                    else:
                        cfs_data[y_local, x_local] = 0.0
                if force_num == 1:
                    # Specific humidity
                    # spacing = vals[2] - vals[1]
                    cfs_interp_fcst = cfs_interp_fcst * 1000.0  # units are now g/kg
                    cfs_cdf = 1 - np.exp(-(np.power((vals / cfs_param_1_interp), cfs_param_2_interp)))

                    nldas_cdf = 1 - np.exp(-(np.power((vals / nldas_nearest_1), nldas_nearest_2)))

                    # compute adjusted value now using the CFSv2 forecast value and the two CDFs
                    # find index in vals array
                    diff_tmp = np.absolute(vals - cfs_interp_fcst)
                    cfs_ind = np.where(diff_tmp == diff_tmp.min())[0][0]
                    cfs_cdf_val = cfs_cdf[cfs_ind]

                    # now whats the index of the closest cdf value in the nldas array?
                    diff_tmp = np.absolute(cfs_cdf_val - nldas_cdf)
                    cfs_nldas_ind = np.where(diff_tmp == diff_tmp.min())[0][0]

                    # Adjust the CFS data
                    cfs_data[y_local, x_local] = vals[cfs_nldas_ind] / 1000.0  # convert back to kg/kg
                if force_num == 4:
                    # Precipitation
                    # precipitation is estimated using a Weibull distribution
                    # valid values range from 3e-6 mm/s (0.01 mm/hr) up to 100 mm/hr
                    # spacing = vals[2] - vals[1]
                    cfs_zero_pcp_interp = cfs_prev_zero_pcp_sub[y_local, x_local] * interp_factor1 + \
                                          cfs_zero_pcp_sub[y_local, x_local] * interp_factor2

                    cfs_cdf = 1 - np.exp(-(np.power((vals / cfs_param_1_interp), cfs_param_2_interp)))
                    # cfs_cdf_scaled = ((1 - cfs_zero_pcp_interp) + cfs_cdf) / \
                    #                  (cfs_cdf.max() + (1 - cfs_zero_pcp_interp))

                    nldas_nearest_zero_pcp = nldas_zero_pcp_sub[y_local, x_local]

                    if nldas_nearest_2 == 0.0:
                        # if second Weibull parameter is zero, the
                        # distribution has no width, no precipitation outside first bin
                        nldas_cdf = np.empty([2000], np.float64)
                        nldas_cdf[:] = 1.0
                        nldas_nearest_zero_pcp = 1.0
                    else:
                        # valid point, see if we need to adjust cfsv2 precip
                        nldas_cdf = 1 - np.exp(-(np.power((vals / nldas_nearest_1), nldas_nearest_2)))

                    # compute adjusted value now using the CFSv2 forecast value and the two CDFs
                    # find index in vals array
                    diff_tmp = np.absolute(vals - (cfs_interp_fcst * 3600.0))
                    cfs_ind = np.where(diff_tmp == diff_tmp.min())[0][0]
                    cfs_cdf_val = cfs_cdf[cfs_ind]

                    # now whats the index of the closest cdf value in the nldas array?
                    diff_tmp = np.absolute(cfs_cdf_val - nldas_cdf)
                    cfs_nldas_ind = np.where(diff_tmp == diff_tmp.min())[0][0]

                    if cfs_interp_fcst == 0.0 and nldas_nearest_zero_pcp == 1.0:
                        # if no rain in cfsv2, no rain in bias corrected field
                        cfs_data[y_local, x_local] = 0.0
                    else:
                        # else there is rain in cfs forecast, so adjust it in some manner
                        pcp_pop_diff = nldas_nearest_zero_pcp - cfs_zero_pcp_interp
                        if cfs_zero_pcp_interp <= nldas_nearest_zero_pcp:
                            # if cfsv2 zero precip probability is less than nldas,
                            # then do one adjustment
                            if cfs_cdf_val <= pcp_pop_diff:
                                # if cfsv2 precip cdf is still less than pop
                                # difference, set precip to zero
                                cfs_data[y_local, x_local] = 0.0
                            else:
                                # cfsv2 precip cdf > nldas zero precip probability,
                                # so adjust cfsv2 to nldas2 precip
                                cfs_data[y_local, x_local] = vals[cfs_nldas_ind] / 3600.0  # convert back to mm/s

                                # check for unreasonable corrections of cfs rainfall
                                # ad-hoc setting that cfsv2 precipitation should not be corrected by more than 3x
                                # if it is, this indicated nldas2 distribution is unrealistic
                                # and default back to cfsv2 forecast value
                                if (cfs_data[y_local, x_local] / cfs_interp_fcst) >= 3.0:
                                    cfs_data[y_local, x_local] = cfs_interp_fcst
                        else:
                            if cfs_cdf_val <= abs(pcp_pop_diff):
                                # if cfsv2 cdf value less than pop difference, need to randomly
                                # generate precip, since we're in the zero portion of the nldas
                                # zero precip prob still
                                randn = random.uniform(0.0, abs(pcp_pop_diff))
                                diff_tmp = np.absolute(randn - nldas_cdf)
                                new_nldas_ind = np.where(diff_tmp == diff_tmp.min())[0][0]
                                cfs_data[y_local, x_local] = vals[new_nldas_ind] / 3600.0

                                # ad-hoc setting that cfsv2 precipitation should not be corrected by more than 3x
                                # if it is, this indicated nldas2 distribution is unrealistic
                                # and default back to cfsv2 forecast value
                                if (cfs_data[y_local, x_local] / cfs_interp_fcst) >= 3.0:
                                    cfs_data[y_local, x_local] = cfs_interp_fcst
                            else:
                                cfs_data[y_local, x_local] = vals[cfs_nldas_ind] / 3600.0  # convert back to mm/s
                                # ad-hoc setting that cfsv2 precipitation should not be corrected by more than 3x
                                # if it is, this indicated nldas2 distribution is unrealistic
                                # and default back to cfsv2 forecast value
                                if (cfs_data[y_local, x_local] / cfs_interp_fcst) >= 3.0:
                                    cfs_data[y_local, x_local] = cfs_interp_fcst
            else:
                # No adjustment for this CFS pixel cell as we have missing parameter values.
                cfs_data[y_local, x_local] = cfs_interp_fcst

    # Regrid the local CFS slap to the output array
    try:
        input_forcings.esmf_field_in.data[:, :] = cfs_data
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to place CFSv2 forcing data into temporary ESMF field: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                 input_forcings.esmf_field_out)
    except ValueError as ve:
        config_options.errMsg = "Unable to regrid CFSv2 variable: " + input_forcings.netcdf_var_names[force_num] + \
                                " (" + str(ve) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Set any pixel cells outside the input domain to the global missing value.
    try:
        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
            config_options.globalNdv
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to run mask calculation on CFSv2 variable: " + \
                                input_forcings.netcdf_var_names[force_num] + " (" + str(npe) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = \
            input_forcings.esmf_field_out.data
    except NumpyExceptions as npe:
        config_options.errMsg = "Unable to extract ESMF field data for CFSv2: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)
