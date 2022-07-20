# This is a datetime module for handling datetime
# calculations in the forcing engine.
import datetime
import math
import os

import numpy as np

from core import err_handler
from core.forcingInputMod import input_forcings
NETCDF = input_forcings.NETCDF


def calculate_lookback_window(config_options):
    """
    Calculate the beginning, ending datetime variables
    for a look-back period. Also calculate the processing
    time window delta value.
    :param config_options: Abstract class holding job information.
    :return: Updated abstract class with updated datetime variables.
    """
    # First calculate the current time in UTC.
    if config_options.realtime_flag:
        d_current_utc = datetime.datetime.utcnow()
    else:
        d_current_utc = config_options.b_date_proc

    # Next, subtract the lookup window (specified in minutes) to get a crude window
    # of processing.
    d_lookback = d_current_utc - datetime.timedelta(
        seconds=60 * (config_options.look_back - config_options.output_freq))

    # Determine the first forecast iteration that will be processed on this day
    # based on the forecast frequency and where in the day we are at.
    fcst_step_tmp = math.ceil((d_lookback.hour * 60 + d_lookback.minute) / config_options.fcst_freq)
    d_look_tmp1 = datetime.datetime(d_lookback.year, d_lookback.month, d_lookback.day)
    d_look_tmp1 = d_look_tmp1 + datetime.timedelta(seconds=(60 * config_options.fcst_freq * fcst_step_tmp))

    # If we are offsetting the forecasts, apply here.
    if config_options.fcst_shift > 0:
        d_look_tmp1 = d_look_tmp1 + datetime.timedelta(seconds=60 * config_options.fcst_shift)

    config_options.b_date_proc = d_look_tmp1

    # Now calculate the end of the processing window based on the time from the
    # beginning of the processing window.
    dt_tmp = d_current_utc - config_options.b_date_proc
    n_fcst_steps = math.floor((dt_tmp.days*1440+dt_tmp.seconds/60.0) / config_options.fcst_freq)
    #Special case for HRRR AK when we don't need/want more than one forecast cycle
    if 19 in config_options.input_forcings:
        n_fcst_steps = 0
        config_options.nFcsts = int(n_fcst_steps) + 1
        config_options.e_date_proc = config_options.b_date_proc + datetime.timedelta(seconds=config_options.look_back*60-config_options.fcst_freq*60)
        return 

    config_options.nFcsts = int(n_fcst_steps) + 1
    config_options.e_date_proc = config_options.b_date_proc + datetime.timedelta(
        seconds=n_fcst_steps * config_options.fcst_freq * 60)


def find_nldas_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after hourly NLDAS files for use in processing.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    # Normally, we do a check to make sure the input horizons chosen by the user are not
    # greater than an expected value. However, since these are custom input NetCDF files,
    # we are foregoing that check.
    current_nldas_cycle = config_options.current_fcst_cycle - \
        datetime.timedelta(seconds=input_forcings.userCycleOffset * 60.0)

    # Calculate the current forecast hour within this cycle.
    dt_tmp = d_current - current_nldas_cycle

    current_nldas_hour = int(dt_tmp.days*24) + math.floor(dt_tmp.seconds/3600.0)

    # Calculate the previous file to process.
    min_since_last_output = (current_nldas_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_nldas_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_nldas_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_nldas_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_nldas_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_nldas_date - current_nldas_cycle
    next_nldas_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = next_nldas_forecast_hour
    dt_tmp = prev_nldas_date - current_nldas_cycle
    prev_nldas_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prev_nldas_forecast_hour
    # If we are on the first forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_nldas_forecast_hour == 0:
        prev_nldas_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + "/NLDAS_FORA0125_H.A" + \
                input_forcings.fcst_date1.strftime('%Y%m%d.%H%M') + \
                ".002" + input_forcings.file_ext
    tmp_file2 = input_forcings.inDir + '/NLDAS_FORA0125_H.A' + \
                input_forcings.fcst_date1.strftime('%Y%m%d.%H%M') + \
                ".002" + input_forcings.file_ext

    if mpi_config.rank == 0:
        # Check to see if files are already set. If not, then reset, grids and
        # regridding objects to communicate things need to be re-established.
        if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
            else:
                # Check to see if we are restarting from a previously failed instance. In this case,
                # We are not on the first timestep, but no previous forcings have been processed.
                # We need to process the previous input timestep for temporal interpolation purposes.
                # if not np.any(input_forcings.regridded_forcings1):
                if input_forcings.regridded_forcings1 is None:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                                   input_forcings.productName
                        err_handler.log_msg(config_options, mpi_config)
                    input_forcings.rstFlag = 1
                    input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                    input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                    input_forcings.file_in2 = tmp_file1
                    input_forcings.file_in1 = tmp_file1
                    input_forcings.fcst_date2 = input_forcings.fcst_date1
                    input_forcings.fcst_hour2 = input_forcings.fcst_hour1
                else:
                    # The custom window has shifted. Reset fields 2 to
                    # be fields 1.
                    input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                    input_forcings.file_in1 = tmp_file1
                    input_forcings.file_in2 = tmp_file2
            input_forcings.regridComplete = False
    else:
            input_forcings.file_in2 = tmp_file1
            input_forcings.file_in1 = tmp_file1
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input Custom file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input Custom file: " + \
                                           input_forcings.file_in2 + " not found. Will not use in final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_aorc_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after hourly AORC files for use in processing.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    # Normally, we do a check to make sure the input horizons chosen by the user are not
    # greater than an expected value. However, since these are custom input NetCDF files,
    # we are foregoing that check.
    current_aorc_cycle = config_options.current_fcst_cycle - \
        datetime.timedelta(seconds=input_forcings.userCycleOffset * 60.0)

    # Calculate the current forecast hour within this cycle.
    dt_tmp = d_current - current_aorc_cycle

    current_aorc_hour = int(dt_tmp.days*24) + math.floor(dt_tmp.seconds/3600.0)

    # Calculate the previous file to process.
    min_since_last_output = (current_aorc_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_aorc_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_aorc_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_aorc_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_aorc_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_aorc_date - current_aorc_cycle
    next_aorc_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = next_aorc_forecast_hour
    dt_tmp = prev_aorc_date - current_aorc_cycle
    prev_aorc_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prev_aorc_forecast_hour
    # If we are on the first forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_aorc_forecast_hour == 0:
        prev_aorc_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + "/AORC-OWP_" + \
                input_forcings.fcst_date1.strftime('%Y%m%d%H') + \
                "z_example" + input_forcings.file_ext
    tmp_file2 = input_forcings.inDir + '/AORC-OWP_' + \
                input_forcings.fcst_date1.strftime('%Y%m%d%H') + \
                "z_example" + input_forcings.file_ext

    if mpi_config.rank == 0:
        # Check to see if files are already set. If not, then reset, grids and
        # regridding objects to communicate things need to be re-established.
        if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
            else:
                # Check to see if we are restarting from a previously failed instance. In this case,
                # We are not on the first timestep, but no previous forcings have been processed.
                # We need to process the previous input timestep for temporal interpolation purposes.
                # if not np.any(input_forcings.regridded_forcings1):
                if input_forcings.regridded_forcings1 is None:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                                   input_forcings.productName
                        err_handler.log_msg(config_options, mpi_config)
                    input_forcings.rstFlag = 1
                    input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                    input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                    input_forcings.file_in2 = tmp_file1
                    input_forcings.file_in1 = tmp_file1
                    input_forcings.fcst_date2 = input_forcings.fcst_date1
                    input_forcings.fcst_hour2 = input_forcings.fcst_hour1
                else:
                    # The custom window has shifted. Reset fields 2 to
                    # be fields 1.
                    input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                    input_forcings.file_in1 = tmp_file1
                    input_forcings.file_in2 = tmp_file2
            input_forcings.regridComplete = False
    else:
        input_forcings.file_in2 = tmp_file1
        input_forcings.file_in1 = tmp_file1
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input AORC file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input AORC file: " + \
                                           input_forcings.file_in2 + " not found. Will not use in final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.exists(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_ak_ext_ana_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after Alaska Extended Ana cycles based on the current timestep.
    ExtAna inputs are outputs from a prior FE run.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = "Processing Alaska ExtAnA Data. Calculating neighboring " \
                                   "files for this output timestep"
        err_handler.log_msg(config_options, mpi_config)

    # First find the current ExtAnA forecast cycle that we are using.
    ana_offset = 1 if config_options.ana_flag else 0
    current_ext_ana_cycle = config_options.current_fcst_cycle - datetime.timedelta(
        seconds=(ana_offset + input_forcings.userCycleOffset) * 60.0)
    
    ext_ana_horizon = 32

    # If the user has specified a forcing horizon that is greater than what is available
    # for this time period, throw an error.
    if (input_forcings.userFcstHorizon + input_forcings.userCycleOffset) / 60.0 > ext_ana_horizon:
        config_options.errMsg = "User has specified a ExtAnA conus forecast horizon " + \
                                "that is greater than the maximum allowed hours of: " + str(ext_ana_horizon)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Calculate the current forecast hour within this ExtAnA cycle.
    dt_tmp = d_current - current_ext_ana_cycle
    current_ext_ana_hour = int(dt_tmp.days*24) + int(dt_tmp.seconds/3600.0)

    # Calculate the previous file to process.
    min_since_last_output = (current_ext_ana_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_ext_ana_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_ext_ana_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_ext_ana_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_ext_ana_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_ext_ana_date - current_ext_ana_cycle
    next_ext_ana_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = next_ext_ana_forecast_hour
    dt_tmp = prev_ext_ana_date - current_ext_ana_cycle
    prev_ext_ana_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prev_ext_ana_forecast_hour
    err_handler.check_program_status(config_options, mpi_config)

    # If we are on the first ExtAnA forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_ext_ana_forecast_hour == 0:
        prev_ext_ana_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + '/' + prev_ext_ana_date.strftime('%Y%m%d%H') + \
                "/" + prev_ext_ana_date.strftime('%Y%m%d%H') +  "00" + \
                ".LDASIN_DOMAIN1"
    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous ExtAnA file being used: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)

    tmp_file2 = tmp_file1
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Next ExtAnA file being used: " + tmp_file2
            err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
                # if not np.any(input_forcings.regridded_forcings1):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The ExtAnA window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.exists(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input ExtAnA file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input ExtAnA file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.exists(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_conus_hrrr_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after HRRR conus cycles based on the current timestep.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = "Processing Conus HRRR Data. Calculating neighboring " \
                                   "files for this output timestep"
        err_handler.log_msg(config_options, mpi_config)

    # Fortunately, HRRR data is straightforward compared to GFS in terms of precip values, etc.
    if d_current >= datetime.datetime(2018, 10, 1):
        default_horizon = 18  # 18-hour forecasts.
        six_hr_horizon = 36  # 36-hour forecasts every six hours.
    else:
        default_horizon = 18  # 18-hour forecasts.
        six_hr_horizon = 18  # 18-hour forecasts every six hours.

    # First find the current HRRR forecast cycle that we are using.
    ana_offset = 1 if config_options.ana_flag else 0
    current_hrrr_cycle = config_options.current_fcst_cycle - datetime.timedelta(
        seconds=(ana_offset + input_forcings.userCycleOffset) * 60.0)
    if current_hrrr_cycle.hour % 6 != 0:
        hrrr_horizon = default_horizon
    else:
        hrrr_horizon = six_hr_horizon

    # If the user has specified a forcing horizon that is greater than what is available
    # for this time period, throw an error.
    if (input_forcings.userFcstHorizon + input_forcings.userCycleOffset) / 60.0 > hrrr_horizon:
        config_options.errMsg = "User has specified a HRRR conus forecast horizon " + \
                                "that is greater than the maximum allowed hours of: " + str(hrrr_horizon)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Calculate the current forecast hour within this HRRR cycle.
    dt_tmp = d_current - current_hrrr_cycle
    current_hrrr_hour = int(dt_tmp.days*24) + int(dt_tmp.seconds/3600.0)

    # Calculate the previous file to process.
    min_since_last_output = (current_hrrr_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_hrrr_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_hrrr_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_hrrr_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_hrrr_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_hrrr_date - current_hrrr_cycle
    next_hrrr_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = next_hrrr_forecast_hour
    dt_tmp = prev_hrrr_date - current_hrrr_cycle
    prev_hrrr_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prev_hrrr_forecast_hour
    err_handler.check_program_status(config_options, mpi_config)

    # If we are on the first HRRR forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_hrrr_forecast_hour == 0:
        prev_hrrr_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + '/hrrr.' + current_hrrr_cycle.strftime(
        '%Y%m%d') + "/conus/hrrr.t" + current_hrrr_cycle.strftime('%H') + 'z.wrfsfcf' + \
        str(prev_hrrr_forecast_hour).zfill(2) + input_forcings.file_ext
    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous HRRR file being used: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)

    tmp_file2 = input_forcings.inDir + '/hrrr.' + current_hrrr_cycle.strftime(
        '%Y%m%d') + "/conus/hrrr.t" + current_hrrr_cycle.strftime('%H') + 'z.wrfsfcf' \
        + str(next_hrrr_forecast_hour).zfill(2) + input_forcings.file_ext
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Next HRRR file being used: " + tmp_file2
            err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
                # if not np.any(input_forcings.regridded_forcings1):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The HRRR window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.exists(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input HRRR file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input HRRR file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.exists(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_ak_hrrr_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after HRRR Alaska cycles based on the current timestep.
    This data differs from the HRRR conus because there is a forecast cycle every 3 hrs instead of
    every hour.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = "Processing Conus HRRR AK Data. Calculating neighboring " \
                                   "files for this output timestep"
        err_handler.log_msg(config_options, mpi_config)

    default_horizon = 18  # 18-hour forecasts.
    six_hr_horizon = 48  # 48-hour forecasts every six hours.

    # First find the current HRRR AK forecast cycle that we are using.
    if config_options.ana_flag:
        shift = config_options.first_fcst_cycle.hour % 3
        current_hrrr_cycle = config_options.first_fcst_cycle - datetime.timedelta(seconds=3600*shift)
        if shift == 0:
            current_hrrr_cycle -= datetime.timedelta(hours=6)
        else:
            current_hrrr_cycle -= datetime.timedelta(hours=3)

        # Calculate the current forecast hour within this HRRR cycle.
        dt_tmp = d_current - current_hrrr_cycle
        current_hrrr_hour = int(dt_tmp.days * 24) + int(dt_tmp.seconds / 3600.0)
    else:
        current_hrrr_cycle = config_options.current_fcst_cycle - \
            datetime.timedelta(seconds=input_forcings.userCycleOffset * 60.0)

        # Map the native forecast hour to the shifted HRRR cycles
        hrrr_cycle = (current_hrrr_cycle.hour // 3 * 3) - 3
        if hrrr_cycle < 0:
            hrrr_cycle += 24

        # throw out the first 3 hours of the cycle
        current_hrrr_hour = (current_hrrr_cycle.hour % 3) + 3
    
    #if current_hrrr_cycle.hour % 6 == 0:
    #    hrrr_horizon = 48
    #else:
    #    hrrr_horizon = 18
    hrrr_horizon = 48

    # print(f"HRRR cycle is {current_hrrr_cycle}, hour is {current_hrrr_hour}")

    # If the user has specified a forcing horizon that is greater than what is available
    # for this time period, throw an error.
    spec_horizon = (input_forcings.userFcstHorizon + input_forcings.userCycleOffset) / 60.0
    if spec_horizon > hrrr_horizon:
        config_options.errMsg = f"User has specified a HRRR Alaska forecast horizon {spec_horizon} " + \
                                f"that is greater than the maximum allowed hours of: {hrrr_horizon}"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Calculate the previous file to process.
    min_since_last_output = (current_hrrr_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_hrrr_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_hrrr_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_hrrr_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_hrrr_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_hrrr_date - current_hrrr_cycle
    next_hrrr_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    if config_options.ana_flag:
        next_hrrr_forecast_hour -= 1    # for analysis vs forecast
    input_forcings.fcst_hour2 = next_hrrr_forecast_hour
    dt_tmp = prev_hrrr_date - current_hrrr_cycle
    prev_hrrr_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    if config_options.ana_flag:
        prev_hrrr_forecast_hour -= 1    # for analysis vs forecast
    input_forcings.fcst_hour1 = prev_hrrr_forecast_hour
    err_handler.check_program_status(config_options, mpi_config)

    # If we are on the first HRRR forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_hrrr_forecast_hour == 0:
        prev_hrrr_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + '/hrrr.' + current_hrrr_cycle.strftime(
        '%Y%m%d') + "/hrrr.t" + current_hrrr_cycle.strftime('%H') + 'z.wrfsfcf' + \
        str(prev_hrrr_forecast_hour).zfill(2) + ".ak" + input_forcings.file_ext
    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous HRRR file being used: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)

    tmp_file2 = input_forcings.inDir + '/hrrr.' + current_hrrr_cycle.strftime(
        '%Y%m%d') + "/hrrr.t" + current_hrrr_cycle.strftime('%H') + 'z.wrfsfcf' \
        + str(next_hrrr_forecast_hour).zfill(2) + ".ak" + input_forcings.file_ext
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Next HRRR file being used: " + tmp_file2
            err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
                # if not np.any(input_forcings.regridded_forcings1):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The HRRR window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.exists(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input HRRR file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input HRRR file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.exists(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_conus_rap_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after RAP conus 13km cycles based on the current timestep.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    if d_current >= datetime.datetime(2018, 10, 1):
        default_horizon = 21   # 21-hour forecasts.
        extra_hr_horizon = 39  # 39-hour forecasts at 3,9,15,21 UTC.
    else:
        default_horizon = 18   # 18-hour forecasts.
        extra_hr_horizon = 18  # 18-hour forecasts every six hours.

    # First find the current RAP forecast cycle that we are using.
    ana_offset = 1 if config_options.ana_flag else 0
    current_rap_cycle = config_options.current_fcst_cycle - datetime.timedelta(
            seconds=(ana_offset + input_forcings.userCycleOffset) * 60.0)
    if current_rap_cycle.hour == 3 or current_rap_cycle.hour == 9 or \
            current_rap_cycle.hour == 15 or current_rap_cycle.hour == 21:
        rap_horizon = default_horizon
    else:
        rap_horizon = extra_hr_horizon

    # If the user has specified a forcing horizon that is greater than what is available
    # for this time period, throw an error.
    if (input_forcings.userFcstHorizon + input_forcings.userCycleOffset) / 60.0 > rap_horizon:
        config_options.errMsg = "User has specified a RAP CONUS 13km forecast horizon " + \
                                "that is greater than the maximum allowed hours of: " + str(rap_horizon)
        return

    # Calculate the current forecast hour within this HRRR cycle.
    dt_tmp = d_current - current_rap_cycle
    current_rap_hour = int(dt_tmp.days*24) + int(dt_tmp.seconds/3600.0)

    # Calculate the previous file to process.
    min_since_last_output = (current_rap_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_rap_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_rap_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_rap_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_rap_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_rap_date - current_rap_cycle
    next_rap_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = next_rap_forecast_hour
    dt_tmp = prev_rap_date - current_rap_cycle
    prev_rap_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prev_rap_forecast_hour
    # If we are on the first GFS forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_rap_forecast_hour < 1:
        prev_rap_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + '/rap.' + \
        current_rap_cycle.strftime('%Y%m%d') + "/rap.t" + \
        current_rap_cycle.strftime('%H') + 'z.awp130bgrbf' + \
        str(prev_rap_forecast_hour).zfill(2) + input_forcings.file_ext
    tmp_file2 = input_forcings.inDir + '/rap.' + \
        current_rap_cycle.strftime('%Y%m%d') + "/rap.t" + \
        current_rap_cycle.strftime('%H') + 'z.awp130bgrbf' + \
        str(next_rap_forecast_hour).zfill(2) + input_forcings.file_ext

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
                # if not np.any(input_forcings.regridded_forcings1):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The Rapid Refresh window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if not os.path.isfile(input_forcings.file_in2):
        # look for the `pgrb` equivalent:
        input_forcings.file_in2 = input_forcings.file_in2.replace("bgrb", "pgrb")
        if mpi_config.rank == 0:
            if not os.path.isfile(input_forcings.file_in2):
                missing = input_forcings.file_in2.replace("pgrb", "[bgrb|pgrb]")
                if input_forcings.enforce == 1:
                    config_options.errMsg = "Expected input RAP file: " + missing + " not found."
                    err_handler.log_critical(config_options, mpi_config)
                else:
                    config_options.statusMsg = "Expected input RAP file: " + missing + \
                                               " not found, and will not be used in final layering."
                    err_handler.log_warning(config_options, mpi_config)
            else:
                missing = input_forcings.file_in2.replace("pgrb", "bgrb")
                config_options.statusMsg = "Expected input RAP file: {} was not found; {} will be used instead".format(
                    missing, input_forcings.file_in2)
                err_handler.log_warning(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv

    err_handler.check_program_status(config_options, mpi_config)


def find_gfs_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after GFS cycles based on the current timestep.
    :param mpi_config:
    :param d_current:
    :param input_forcings:
    :param config_options:
    :return:
    """
    # First calculate how the GFS files are structured based on our current processing date.
    # This will change in the future, and should be modified as the GFS system evolves.
    if d_current >= datetime.datetime(2018, 10, 1):
        gfs_out_horizons = [120, 240, 384]
        gfs_out_freq = {
            120: 60,
            240: 180,
            384: 720
        }
        gfs_precip_delineators = {
            120: [360, 60],
            240: [360, 180],
            284: [720, 720]
        }
    else:
        gfs_out_horizons = [120, 240, 384]
        gfs_out_freq = {
            120: 60,
            240: 180,
            384: 720
        }
        gfs_precip_delineators = {
            120: [360, 60],
            240: [360, 180],
            284: [720, 720]
        }

    # If the user has specified a forcing horizon that is greater than what
    # is available here, return an error.
    if (input_forcings.userFcstHorizon+input_forcings.userCycleOffset)/60.0 > max(gfs_out_horizons):
        config_options.errMsg = "User has specified a GFS forecast horizon " \
                                "that is greater than maximum allowed hours of: " \
                                + str(max(gfs_out_horizons))
        return

    # First determine if we need a previous file or not. For hourly files,
    # only need the previous file if it's not the first hour after a new GFS cycle
    # (I.E. 7z, 13z, 19z, 1z). Any other hourly timestep requires the previous hourly data
    # (I.E. 8z needs 7z file). For 3-hourly files, we will look for files for the previous
    # three hours and next three hourly output. For example, a timestep that falls 125 hours
    # after the beginning of the GFS cycle will require data from the f123 file and 126 file.
    # For any files after 240 hours out from the beginning of a GFS forecast cycle, the intervals
    # move to 12 hours.

    # First find the current GFS forecast cycle that we are using.
    current_gfs_cycle = config_options.current_fcst_cycle - \
        datetime.timedelta(seconds=input_forcings.userCycleOffset * 60.0)

    # Calculate the current forecast hour within this GFS cycle.
    dt_tmp = d_current - current_gfs_cycle
    current_gfs_hour = int(dt_tmp.days*24) + int(dt_tmp.seconds/3600.0)

    # Calculate the GFS output frequency based on our current GFS forecast hour.
    current_gfs_freq = None
    for horizon_tmp in gfs_out_horizons:
        if current_gfs_hour <= horizon_tmp:
            current_gfs_freq = gfs_out_freq[horizon_tmp]
            input_forcings.outFreq = gfs_out_freq[horizon_tmp]
            break

    # Calculate the previous file to process.
    min_since_last_output = (current_gfs_hour*60) % current_gfs_freq
    if min_since_last_output == 0:
        min_since_last_output = current_gfs_freq
        # current_gfs_hour = current_gfs_hour
        # previousGfsHour = current_gfs_hour - int(current_gfs_freq/60.0)
    prev_gfs_date = d_current - datetime.timedelta(seconds=min_since_last_output*60)
    input_forcings.fcst_date1 = prev_gfs_date
    if min_since_last_output == current_gfs_freq:
        min_until_next_output = 0
    else:
        min_until_next_output = current_gfs_freq - min_since_last_output
    next_gfs_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_gfs_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_gfs_date - current_gfs_cycle
    next_gfs_forecast_hour = int(dt_tmp.days*24.0) + int(dt_tmp.seconds/3600.0)
    input_forcings.fcst_hour2 = next_gfs_forecast_hour
    dt_tmp = prev_gfs_date - current_gfs_cycle
    prev_gfs_forecast_hour = int(dt_tmp.days*24.0) + int(dt_tmp.seconds/3600.0)
    input_forcings.fcst_hour1 = prev_gfs_forecast_hour
    # If we are on the first GFS forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_gfs_forecast_hour == 0:
        prev_gfs_forecast_hour = 1

    # Calculate expected file paths.
    if current_gfs_cycle < datetime.datetime(2019, 6, 12, 12):
        tmp_file1 = input_forcings.inDir + '/gfs.' + \
            current_gfs_cycle.strftime('%Y%m%d%H') + "/gfs.t" + \
            current_gfs_cycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(prev_gfs_forecast_hour).zfill(3) + input_forcings.file_ext
        tmp_file2 = input_forcings.inDir + '/gfs.' + \
            current_gfs_cycle.strftime('%Y%m%d%H') + "/gfs.t" + \
            current_gfs_cycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(next_gfs_forecast_hour).zfill(3) + input_forcings.file_ext
    else:
        # FV3 change on June 12th, 2019
        tmp_file1 = input_forcings.inDir + '/gfs.' + \
            current_gfs_cycle.strftime('%Y%m%d') + "/" + \
            current_gfs_cycle.strftime('%H') + \
            ('/' if input_forcings.fileType != NETCDF else '/atmos/') + "gfs.t" + \
            current_gfs_cycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(prev_gfs_forecast_hour).zfill(3) + input_forcings.file_ext
        tmp_file2 = input_forcings.inDir + '/gfs.' + \
            current_gfs_cycle.strftime('%Y%m%d') + "/" + \
            current_gfs_cycle.strftime('%H') + \
            ('/' if input_forcings.fileType != NETCDF else '/atmos/') + "gfs.t" + \
            current_gfs_cycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(next_gfs_forecast_hour).zfill(3) + input_forcings.file_ext

    # If needed, initialize the globalPcpRate1 array (this is for when we have the initial grid, or we need to change
    # grids.
    # if np.any(input_forcings.globalPcpRate2) and not np.any(input_forcings.globalPcpRate1):
    if input_forcings.globalPcpRate2 is not None and input_forcings.globalPcpRate1 is None:
        input_forcings.globalPcpRate1 = np.empty([input_forcings.globalPcpRate2.shape[0],
                                                  input_forcings.globalPcpRate2.shape[1]], np.float32)
    # if np.any(input_forcings.globalPcpRate2) and np.any(input_forcings.globalPcpRate1):
    if input_forcings.globalPcpRate2 is not None and input_forcings.globalPcpRate1 is not None:
        if input_forcings.globalPcpRate2.shape[0] != input_forcings.globalPcpRate1.shape[0]:
            # The grid has changed, we need to re-initialize the globalPcpRate1 array.
            input_forcings.globalPcpRate1 = None
            input_forcings.globalPcpRate1 = np.empty([input_forcings.globalPcpRate2.shape[0],
                                                      input_forcings.globalPcpRate2.shape[1]], np.float32)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            # if not np.any(input_forcings.regridded_forcings1):
            if input_forcings.regridded_forcings1 is None:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                if input_forcings.productName == "GFS_Production_GRIB2":
                    if mpi_config.rank == 0:
                        input_forcings.globalPcpRate1 = input_forcings.globalPcpRate1
                        input_forcings.globalPcpRate2 = input_forcings.globalPcpRate2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The forcing window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                if input_forcings.productName == "GFS_Production_GRIB2":
                    if mpi_config.rank == 0:
                        input_forcings.globalPcpRate1[:, :] = input_forcings.globalPcpRate2[:, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input GFS file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input GFS file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_nam_nest_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after NAM Nest cycles based on the current timestep.
    This function was written to be generic enough to accommodate any of the NAM nest flavors.
    :param mpi_config:
    :param d_current:
    :param input_forcings:
    :param config_options:
    :return:
    """
    # Establish how often our NAM nest cycles occur.
    nam_nest_out_horizons = []
    nam_nest_out_freq = {}
    if d_current >= datetime.datetime(2012, 10, 1):
        nam_nest_out_horizons = [60]
        nam_nest_out_freq = {
            60: 60
        }

    # If the user has specified a forcing horizon that is greater than what
    # is available here, return an error.
    if (input_forcings.userFcstHorizon+input_forcings.userCycleOffset)/60.0 > max(nam_nest_out_horizons):
        config_options.errMsg = "User has specified a NAM nest forecast horizon " \
                                "that is greater than maximum allowed hours of: " \
                                + str(max(nam_nest_out_horizons))
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # First find the current NAM nest forecast cycle that we are using.
    if config_options.ana_flag:
        # find nearest previous cycle, and always use the first cycle for consistency
        shift = config_options.first_fcst_cycle.hour % 6
        current_nam_nest_cycle = config_options.first_fcst_cycle - datetime.timedelta(seconds=3600*shift)

        # avoid forecast hours 0-3, shift back if necessary
        if config_options.first_fcst_cycle.hour % 6 < 4:
            current_nam_nest_cycle -= datetime.timedelta(seconds=21600)     # shift back 6 hours
    else:
        current_nam_nest_cycle = config_options.current_fcst_cycle - \
            datetime.timedelta(seconds=input_forcings.userCycleOffset * 60.0)

    if mpi_config.rank == 0:
        config_options.statusMsg = "Current NAM nest cycle being used: " + \
                                   current_nam_nest_cycle.strftime('%Y-%m-%d %H')
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Calculate the current forecast hour within this NAM nest cycle.
    dt_tmp = d_current - current_nam_nest_cycle
    current_nam_nest_hour = int(dt_tmp.days*24) + int(dt_tmp.seconds/3600.0)

    # Calculate the NAM Nest output frequency based on our current NAM Nest forecast hour.
    current_nam_nest_freq = float('nan')
    for horizonTmp in nam_nest_out_horizons:
        if current_nam_nest_hour <= horizonTmp:
            current_nam_nest_freq = nam_nest_out_freq[horizonTmp]
            input_forcings.outFreq = nam_nest_out_freq[horizonTmp]
            break

    # Calculate the previous file to process.
    min_since_last_output = (current_nam_nest_hour*60) % current_nam_nest_freq
    if min_since_last_output == 0:
        min_since_last_output = current_nam_nest_freq
    prev_nam_nest_date = d_current - datetime.timedelta(seconds=min_since_last_output*60)
    input_forcings.fcst_date1 = prev_nam_nest_date
    if min_since_last_output == current_nam_nest_freq:
        min_until_next_output = 0
    else:
        min_until_next_output = current_nam_nest_freq - min_since_last_output
    next_nam_nest_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_nam_nest_date
    err_handler.check_program_status(config_options, mpi_config)

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_nam_nest_date - current_nam_nest_cycle
    next_nam_nest_forecast_hour = int(dt_tmp.days*24.0) + int(dt_tmp.seconds/3600.0)
    if config_options.ana_flag:
        next_nam_nest_forecast_hour -= 1    # for analysis vs forecast

    input_forcings.fcst_hour2 = next_nam_nest_forecast_hour
    dt_tmp = prev_nam_nest_date - current_nam_nest_cycle
    prev_nam_nest_forecast_hour = int(dt_tmp.days*24.0) + int(dt_tmp.seconds/3600.0)

    input_forcings.fcst_hour1 = prev_nam_nest_forecast_hour
    # If we are on the first NAM nest forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_nam_nest_forecast_hour == 0:
        prev_nam_nest_forecast_hour = 1

    if input_forcings.keyValue == 13 or input_forcings.keyValue == 16:
        domain_string = "hawaiinest"
    elif input_forcings.keyValue == 14 or input_forcings.keyValue == 17:
        domain_string = "priconest"
    elif input_forcings.keyValue == 15:
        domain_string = "alaskanest"
    else:
        domain_string = ''

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + '/nam.' + \
        current_nam_nest_cycle.strftime('%Y%m%d') + "/nam.t" + \
        current_nam_nest_cycle.strftime('%H') + 'z.' + domain_string + '.hiresf' + \
        str(prev_nam_nest_forecast_hour).zfill(2) + '.tm00' + input_forcings.file_ext
    tmp_file2 = input_forcings.inDir + '/nam.' + \
        current_nam_nest_cycle.strftime('%Y%m%d') + "/nam.t" + \
        current_nam_nest_cycle.strftime('%H') + 'z.' + domain_string + '.hiresf' + \
        str(next_nam_nest_forecast_hour).zfill(2) + '.tm00' + input_forcings.file_ext
    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous NAM nest file being used: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Next NAM nest file being used: " + tmp_file2
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            # if not np.any(input_forcings.regridded_forcings1):
            if input_forcings.regridded_forcings1 is None:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The NAM nest window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input NAM Nest file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input NAM Nest file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_cfsv2_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function that will calculate the neighboring CFSv2 global GRIB2 files for a given
    forcing output timestep.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    ens_str = str(config_options.cfsv2EnsMember)
    ens_str = ens_str.zfill(2)
    cfs_out_horizons = [6480]  # Forecast cycles go out 9 months.
    cfs_out_freq = {
        6480: 360
    }

    # If the user has specified a forcing horizon that is greater than what
    # is available here, return an error.
    if (input_forcings.userFcstHorizon + input_forcings.userCycleOffset) / 60.0 > max(cfs_out_horizons):
        config_options.errMsg = "User has specified a CFSv2 forecast horizon " \
                                "that is greater than maximum allowed hours of: " \
                                + str(max(cfs_out_horizons))
        return

    # First find the current CFS forecast cycle that we are using.
    current_cfs_cycle = config_options.current_fcst_cycle - \
        datetime.timedelta(seconds=input_forcings.userCycleOffset * 60.0)

    # Calculate the current forecast hour within this CFSv2 cycle.
    dt_tmp = d_current - current_cfs_cycle
    current_cfs_hour = int(dt_tmp.days * 24) + int(dt_tmp.seconds / 3600.0)

    # Calculate the CFS output frequency based on our current CFS forecast hour.
    current_cfs_freq = float('nan')
    for horizonTmp in cfs_out_horizons:
        if current_cfs_hour <= horizonTmp:
            current_cfs_freq = cfs_out_freq[horizonTmp]
            input_forcings.outFreq = cfs_out_freq[horizonTmp]
            break

    # Calculate the previous file to process.
    min_since_last_output = (current_cfs_hour * 60) % current_cfs_freq
    if min_since_last_output == 0:
        min_since_last_output = current_cfs_freq
        # current_cfs_hour = current_cfs_hour
        # previousCfsHour = current_cfs_hour - int(current_cfs_freq/60.0)
    prev_cfs_date = d_current - \
        datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_cfs_date
    if min_since_last_output == current_cfs_freq:
        min_until_next_output = 0
    else:
        min_until_next_output = current_cfs_freq - min_since_last_output
    next_cfs_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_cfs_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_cfs_date - current_cfs_cycle
    next_cfs_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = next_cfs_forecast_hour
    dt_tmp = prev_cfs_date - current_cfs_cycle
    prev_cfs_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prev_cfs_forecast_hour
    # If we are on the first CFS forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_cfs_forecast_hour == 0:
        prev_cfs_date = next_cfs_date

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + "/cfs." + \
        current_cfs_cycle.strftime('%Y%m%d') + "/" + \
        current_cfs_cycle.strftime('%H') + "/" + \
        "6hrly_grib_" + ens_str + "/flxf" + \
        prev_cfs_date.strftime('%Y%m%d%H') + "." + \
        ens_str + "." + current_cfs_cycle.strftime('%Y%m%d%H') + \
        input_forcings.file_ext
    tmp_file2 = input_forcings.inDir + "/cfs." + \
        current_cfs_cycle.strftime('%Y%m%d') + "/" + \
        current_cfs_cycle.strftime('%H') + "/" + \
        "6hrly_grib_" + ens_str + "/flxf" + \
        next_cfs_date.strftime('%Y%m%d%H') + "." + \
        ens_str + "." + current_cfs_cycle.strftime('%Y%m%d%H') + \
        input_forcings.file_ext

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            # if not np.any(input_forcings.regridded_forcings1):
            if input_forcings.regridded_forcings1 is None:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The CFS window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
                if config_options.runCfsNldasBiasCorrect:
                    # Reset our global CFSv2 grids.
                    input_forcings.coarse_input_forcings1[:, :, :] = input_forcings.coarse_input_forcings2[:, :, :]
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input CFSv2 file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input CFSv2 file: " + \
                                           input_forcings.file_in2 + " not found. Will not use in final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_custom_hourly_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after hourly custom NetCDF files for use in processing.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    # Normally, we do a check to make sure the input horizons chosen by the user are not
    # greater than an expected value. However, since these are custom input NetCDF files,
    # we are foregoing that check.
    current_custom_cycle = config_options.current_fcst_cycle - \
        datetime.timedelta(seconds=input_forcings.userCycleOffset * 60.0)

    # Calculate the current forecast hour within this cycle.
    dt_tmp = d_current - current_custom_cycle

    current_custom_hour = int(dt_tmp.days*24) + math.floor(dt_tmp.seconds/3600.0)
    current_custom_min = math.floor((dt_tmp.seconds % 3600.0)/60.0)

    # Calculate the previous file to process.
    min_since_last_output = (current_custom_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_custom_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_custom_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_custom_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_custom_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_custom_date - current_custom_cycle
    next_custom_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = next_custom_forecast_hour
    dt_tmp = prev_custom_date - current_custom_cycle
    prev_custom_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prev_custom_forecast_hour
    # If we are on the first forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_custom_forecast_hour == 0:
        prev_custom_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + "/custom_hourly." + \
        current_custom_cycle.strftime('%Y%m%d%H') + '.f' + \
        str(prev_custom_forecast_hour).zfill(2) + '.nc'
    tmp_file2 = input_forcings.inDir + '/custom_hourly.' + \
        current_custom_cycle.strftime('%Y%m%d%H') + '.f' + \
        str(next_custom_forecast_hour).zfill(2) + '.nc'
    if mpi_config.rank == 0:
        # Check to see if files are already set. If not, then reset, grids and
        # regridding objects to communicate things need to be re-established.
        if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
            else:
                # Check to see if we are restarting from a previously failed instance. In this case,
                # We are not on the first timestep, but no previous forcings have been processed.
                # We need to process the previous input timestep for temporal interpolation purposes.
                # if not np.any(input_forcings.regridded_forcings1):
                if input_forcings.regridded_forcings1 is None:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                                   input_forcings.productName
                        err_handler.log_msg(config_options, mpi_config)
                    input_forcings.rstFlag = 1
                    input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                    input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                    input_forcings.file_in2 = tmp_file1
                    input_forcings.file_in1 = tmp_file1
                    input_forcings.fcst_date2 = input_forcings.fcst_date1
                    input_forcings.fcst_hour2 = input_forcings.fcst_hour1
                else:
                    # The custom window has shifted. Reset fields 2 to
                    # be fields 1.
                    input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                    input_forcings.file_in1 = tmp_file1
                    input_forcings.file_in2 = tmp_file2
            input_forcings.regridComplete = False
        err_handler.check_program_status(config_options, mpi_config)

        # Ensure we have the necessary new file
        if mpi_config.rank == 0:
            if not os.path.isfile(input_forcings.file_in2):
                if input_forcings.enforce == 1:
                    config_options.errMsg = "Expected input Custom file: " + input_forcings.file_in2 + " not found."
                    err_handler.log_critical(config_options, mpi_config)
                else:
                    config_options.statusMsg = "Expected input Custom file: " + \
                                               input_forcings.file_in2 + " not found. Will not use in final layering."
                    err_handler.log_warning(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


def find_hourly_mrms_radar_neighbors(supplemental_precip, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and next MRMS radar-only QPE files. This
    will also calculate the neighboring radar quality index (RQI) files as well.
    :param supplemental_precip:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    # First we need to find the nearest previous and next hour, which is
    # the previous/next MRMS files we will be using.
    current_yr = d_current.year
    current_mo = d_current.month
    current_day = d_current.day
    current_hr = d_current.hour
    current_min = d_current.minute

    # Set the input file frequency to be hourly.
    supplemental_precip.input_frequency = 60.0

    prev_date1 = datetime.datetime(current_yr, current_mo, current_day, current_hr)
    dt_tmp = d_current - prev_date1
    if dt_tmp.total_seconds() == 0:
        # We are on the hour, we can set this date to the be the "next" date.
        next_mrms_date = d_current
        prev_mrms_date = d_current - datetime.timedelta(seconds=3600.0)
    else:
        # We are between two MRMS hours.
        prev_mrms_date = prev_date1
        next_mrms_date = prev_mrms_date + datetime.timedelta(seconds=3600.0)

    supplemental_precip.pcp_date1 = prev_mrms_date
    #supplemental_precip.pcp_date2 = next_mrms_date
    supplemental_precip.pcp_date2 = prev_mrms_date

    # Calculate expected file paths.
    if supplemental_precip.keyValue == 1:
        tmp_file1 = supplemental_precip.inDir + "/RadarOnly_QPE/" + \
            "RadarOnly_QPE_01H_00.00_" + \
            supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
            "-" + supplemental_precip.pcp_date1.strftime('%H') + \
            "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
        tmp_file2 = supplemental_precip.inDir + "/RadarOnly_QPE/" + \
            "RadarOnly_QPE_01H_00.00_" + \
            supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
            "-" + supplemental_precip.pcp_date2.strftime('%H') + \
            "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')

    elif supplemental_precip.keyValue == 2:
        tmp_file1 = supplemental_precip.inDir + "/GaugeCorr_QPE/" + \
                   "GaugeCorr_QPE_01H_00.00_" + \
                   supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
                   "-" + supplemental_precip.pcp_date1.strftime('%H') + \
                   "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
        tmp_file2 = supplemental_precip.inDir + "/GaugeCorr_QPE/" + \
                   "GaugeCorr_QPE_01H_00.00_" + \
                   supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
                   "-" + supplemental_precip.pcp_date2.strftime('%H') + \
                   "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')

    elif supplemental_precip.keyValue == 5 or supplemental_precip.keyValue == 6:
        tmp_file1 = supplemental_precip.inDir + "/MultiSensor_QPE_01H_Pass1/" + \
                    "MRMS_MultiSensor_QPE_01H_Pass1_00.00_" + \
                    supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
                    "-" + supplemental_precip.pcp_date1.strftime('%H') + \
                    "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
        tmp_file2 = supplemental_precip.inDir + "/MultiSensor_QPE_01H_Pass2/" + \
                    "MRMS_MultiSensor_QPE_01H_Pass2_00.00_" + \
                    supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
                    "-" + supplemental_precip.pcp_date2.strftime('%H') + \
                    "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
    elif supplemental_precip.keyValue == 10:
        tmp_file1 = supplemental_precip.inDir + "/MultiSensor_QPE_01H_Pass1/" + \
                    supplemental_precip.pcp_date1.strftime('%Y%m%d') + "/" + \
                    "MRMS_MultiSensor_QPE_01H_Pass1_00.00_" + \
                    supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
                    "-" + supplemental_precip.pcp_date1.strftime('%H') + \
                    "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
        tmp_file2 = supplemental_precip.inDir + "/MultiSensor_QPE_01H_Pass2/" + \
                    supplemental_precip.pcp_date1.strftime('%Y%m%d') + "/" + \
                    "MRMS_MultiSensor_QPE_01H_Pass2_00.00_" + \
                    supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
                    "-" + supplemental_precip.pcp_date2.strftime('%H') + \
                    "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
    else:
        tmp_file1 = tmp_file2 = ""

    # Compose the RQI paths.
    if supplemental_precip.keyValue == 1 or supplemental_precip.keyValue == 2:
       tmp_rqi_file1 = supplemental_precip.inDir + "/RadarQualityIndex/" + \
           "RadarQualityIndex_00.00_" + \
           supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
           "-" + supplemental_precip.pcp_date1.strftime('%H') + \
           "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
       tmp_rqi_file2 = supplemental_precip.inDir + "/RadarQualityIndex/" + \
           "RadarQualityIndex_00.00_" + \
           supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
           "-" + supplemental_precip.pcp_date2.strftime('%H') + \
           "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
    #elif supplemental_precip.keyValue == 5:
    #   tmp_rqi_file1 = supplemental_precip.inDir + "/RadarQualityIndex/" + \
    #       "MRMS_EXP_RadarQualityIndex_00.00_" + \
    #       supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
    #       "-" + supplemental_precip.pcp_date1.strftime('%H') + \
    #       "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
    #   tmp_rqi_file2 = supplemental_precip.inDir + "/RadarQualityIndex/" + \
    #       "MRMS_EXP_RadarQualityIndex_00.00_" + \
    #       supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
    #       "-" + supplemental_precip.pcp_date2.strftime('%H') + \
    #       "0000" + supplemental_precip.file_ext + ('.gz' if supplemental_precip.fileType != NETCDF else '')
    else:
       tmp_rqi_file1 = tmp_rqi_file2 = ""

    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous MRMS supplemental file: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Next MRMS supplemental file: " + tmp_file2
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Previous MRMS RQI supplemental file: " + tmp_rqi_file1
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Next MRMS RQI supplemental file: " + tmp_rqi_file2
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if supplemental_precip.file_in1 != tmp_file1 or supplemental_precip.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
            supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
            supplemental_precip.regridded_rqi1 = supplemental_precip.regridded_rqi1
            supplemental_precip.regridded_rqi2 = supplemental_precip.regridded_rqi2
        else:
            # The forecast window has shifted. Reset fields 2 to
            # be fields 1.
            supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
            supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
            supplemental_precip.regridded_rqi1 = supplemental_precip.regridded_rqi1
            supplemental_precip.regridded_rqi2 = supplemental_precip.regridded_rqi2
            # supplemental_precip.regridded_precip1[:,:] = supplemental_precip.regridded_precip2[:,:]
            # supplemental_precip.regridded_rqi1[:, :] = supplemental_precip.regridded_rqi2[:, :]
        supplemental_precip.file_in1 = tmp_file1
        supplemental_precip.file_in2 = tmp_file2
        supplemental_precip.rqi_file_in1 = tmp_rqi_file1
        supplemental_precip.rqi_file_in2 = tmp_rqi_file2
        supplemental_precip.regridComplete = False

    # If either file does not exist, set to None. This will instruct downstream regridding steps to
    # set the regridded states to the global NDV. That ensures no supplemental precipitation will be
    # added to the final output grids.

    # if not os.path.isfile(tmp_file1) or not os.path.isfile(tmp_file2):
    #    if MpiConfig.rank == 0:
    #        ConfigOptions.statusMsg = "MRMS files are missing. Will not process " \
    #                                  "supplemental precipitation"
    #        errMod.log_warning(ConfigOptions,MpiConfig)
    #    supplemental_precip.file_in2 = None
    #    supplemental_precip.file_in1 = None

    # errMod.check_program_status(ConfigOptions, MpiConfig)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(supplemental_precip.file_in2) and (supplemental_precip.keyValue == 5 or supplemental_precip.keyValue == 6):
            config_options.statusMsg = "MRMS file {} not found, will attempt to use {} instead.".format(
                    supplemental_precip.file_in2, supplemental_precip.file_in1)
            err_handler.log_warning(config_options, mpi_config)
            supplemental_precip.file_in2 = supplemental_precip.file_in1
        if not os.path.isfile(supplemental_precip.file_in2):
            if supplemental_precip.enforce == 1:
                config_options.errMsg = "Expected input MRMS file: " + supplemental_precip.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input MRMS file: " + supplemental_precip.file_in2 + \
                                           " not found. " + "Will not use in final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(supplemental_precip.file_in2):
        if supplemental_precip.regridded_precip2 is not None:
            supplemental_precip.regridded_precip2[:, :] = config_options.globalNdv


def find_hourly_wrf_arw_neighbors(supplemental_precip, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and next WRF-ARW Hi-Res Window files for the previous
    and next output timestep. These files will be used for supplemental precipitation.
    :param supplemental_precip:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    if d_current >= datetime.datetime(2008, 10, 1):
        default_horizon = 48  # Current forecasts go out to 48 hours.

    # First find the current ARW forecast cycle that we are using.
    if config_options.ana_flag:
        current_arw_cycle = config_options.current_fcst_cycle

        # find nearest previous cycle
        shift = config_options.current_fcst_cycle.hour % 12
        # use the ForecastInputOffsets from the configuration to offset non-00z cycling
        if shift < supplemental_precip.userCycleOffset:
            shift += supplemental_precip.userCycleOffset
        else:
            shift -= supplemental_precip.userCycleOffset
        current_arw_cycle -= datetime.timedelta(seconds=3600*shift)

        # avoid forecast hours 0-3, shift back if necessary
        if 3 < (config_options.first_fcst_cycle.hour % 12) <= 9 and shift < 6:
            current_arw_cycle -= datetime.timedelta(seconds=43200)  # shift back 12 hours

    else:
        current_arw_cycle = config_options.current_fcst_cycle

    if mpi_config.rank == 0:
        config_options.statusMsg = "Processing WRF-ARW supplemental precipitation from " \
                                   "forecast cycle: " + current_arw_cycle.strftime('%Y-%m-%d %H')
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Apply some logic here to account for the ARW weirdness associated with the
    # odd forecast cycles.
    fcst_horizon = -1
    if supplemental_precip.keyValue == 3:
        # Data only available for 00/12 UTC cycles.
        if current_arw_cycle.hour != 0 and current_arw_cycle.hour != 12:
            if mpi_config.rank == 0:
                config_options.statusMsg = "No Hawaii WRF-ARW found for this cycle. " \
                                           "No supplemental ARW precipitation will be used."
                err_handler.log_msg(config_options, mpi_config)
            supplemental_precip.file_in1 = None
            supplemental_precip.file_in2 = None
            return
        fcst_horizon = 48
    if supplemental_precip.keyValue == 4:
        # Data only available for 06/18 UTC cycles.
        cycle = current_arw_cycle
        if cycle.hour != 6 and cycle.hour != 18:
            if mpi_config.rank == 0:
                config_options.statusMsg = "No Puerto Rico WRF-ARW found for cycle {}. " \
                                           "No supplemental ARW precipitation will be used.".format(cycle)
                err_handler.log_msg(config_options, mpi_config)
            supplemental_precip.file_in1 = None
            supplemental_precip.file_in2 = None
            return
        fcst_horizon = 48

    if mpi_config.rank == 0:
        config_options.statusMsg = "Current ARW forecast cycle being used: " + \
                                   current_arw_cycle.strftime('%Y-%m-%d %H')
        err_handler.log_msg(config_options, mpi_config)

    # Calculate the current forecast hour within this ARW cycle.
    dt_tmp = d_current - current_arw_cycle
    current_arw_hour = int(dt_tmp.days * 24) + int(dt_tmp.seconds / 3600.0)

    if current_arw_hour > fcst_horizon:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Current ARW forecast hour greater than max allowed " \
                                       "of: " + str(fcst_horizon)
            err_handler.log_msg(config_options, mpi_config)
            supplemental_precip.file_in2 = None
            supplemental_precip.file_in1 = None
            pass
    err_handler.check_program_status(config_options, mpi_config)

    # Calculate the previous file to process.
    min_since_last_output = (current_arw_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_arw_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    supplemental_precip.pcp_date1 = prev_arw_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_arw_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    supplemental_precip.pcp_date2 = next_arw_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_arw_date - current_arw_cycle
    next_arw_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    if next_arw_forecast_hour > fcst_horizon:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Next ARW forecast hour greater than max allowed " \
                                      "of: " + str(fcst_horizon)
            err_handler.log_msg(config_options, mpi_config)
            supplemental_precip.file_in2 = None
            supplemental_precip.file_in1 = None
            pass
    err_handler.check_program_status(config_options, mpi_config)

    supplemental_precip.fcst_hour2 = next_arw_forecast_hour
    dt_tmp = prev_arw_date - current_arw_cycle
    prev_arw_forecast_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    supplemental_precip.fcst_hour1 = prev_arw_forecast_hour

    # If we are on the first ARW forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_arw_forecast_hour == 0:
        prev_arw_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = tmp_file2 = "(none)"
    if supplemental_precip.keyValue == 3 or supplemental_precip.keyValue == 8:
        tmp_file1 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(prev_arw_forecast_hour).zfill(2) + '.hi' + supplemental_precip.file_ext
        tmp_file2 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(next_arw_forecast_hour).zfill(2) + '.hi' + supplemental_precip.file_ext
    elif supplemental_precip.keyValue == 4 or supplemental_precip.keyValue == 18:
        tmp_file1 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(prev_arw_forecast_hour).zfill(2) + '.pr' + supplemental_precip.file_ext
        tmp_file2 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(next_arw_forecast_hour).zfill(2) + '.pr' + supplemental_precip.file_ext

    err_handler.check_program_status(config_options, mpi_config)
    if mpi_config.rank == 0:
        config_options.statusMsg = "Current ARW input file: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Next ARW input file: " + tmp_file2
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if supplemental_precip.file_in1 != tmp_file1 or supplemental_precip.file_in2 != tmp_file2:
        # if config_options.current_output_step == 1:
        #     supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
        #     supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
        # else:
        #     # The forecast window has shifted. Reset fields 2 to
        #     # be fields 1.
        #     supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
        #     supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
        supplemental_precip.file_in1 = tmp_file1
        supplemental_precip.file_in2 = tmp_file2
        supplemental_precip.regridComplete = False

    # If either file does not exist, set to None. This will instruct downstream regridding steps to
    # set the regridded states to the global NDV. That ensures no supplemental precipitation will be
    # added to the final output grids.
    # if not os.path.isfile(tmp_file1) or not os.path.isfile(tmp_file2):
    #    if MpiConfig.rank == 0:
    #        ConfigOptions.statusMsg = "Found missing ARW file. Will not process supplemental precipitation for " \
    #                                  "this time step."
    #        errMod.log_msg(ConfigOptions, MpiConfig)
    #    supplemental_precip.file_in2 = None
    #    supplemental_precip.file_in1 = None

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(supplemental_precip.file_in2):
            if supplemental_precip.enforce == 1:
                config_options.errMsg = "Expected input ARW file: " + supplemental_precip.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input ARW file: " + supplemental_precip.file_in2 + \
                                          " not found. " + "Will not use in final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(supplemental_precip.file_in2):
        if supplemental_precip.regridded_precip2 is not None:
            supplemental_precip.regridded_precip2[:, :] = config_options.globalNdv

    # supplemental_precip.file_in2 = supplemental_precip.file_in1


def find_sbcv2_lwf_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and after hourly MRMS SBCV2_LWF files for use in processing.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """

    # Set the input file frequency to be hourly.
    input_forcings.input_frequency = 60.0

    # First we need to find the nearest previous and next hour, which is
    # the previous/next MRMS files we will be using.

    prev_date1 = datetime.datetime(d_current.year, d_current.month, d_current.day, d_current.hour)
    dt_tmp = d_current - prev_date1

    if dt_tmp.total_seconds() == 0:
        # We are on the hour, we can set this date to the be the "next" date.
        next_mrms_date = d_current
        prev_mrms_date = d_current - datetime.timedelta(seconds=3600.0)
    else:
        # We are between two MRMS hours.
        prev_mrms_date = prev_date1
        next_mrms_date = prev_mrms_date + datetime.timedelta(seconds=3600.0)

    input_forcings.fcst_date1 = prev_mrms_date
    # supplemental_precip.pcp_date2 = next_mrms_date
    input_forcings.fcst_date2 = prev_mrms_date

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + "/SBC_LWF/" + \
                input_forcings.fcst_date1.strftime("%Y%m") + "/" + input_forcings.fcst_date1.strftime('%Y%m%d') +\
                "/SBCV2_LWF." + input_forcings.fcst_date1.strftime('%Y%m%d') + \
                "-" + input_forcings.fcst_date1.strftime('%H') + \
                "0000.netcdf"
    tmp_file2 = tmp_file1  # NO TEMPORAL INTERPOLATION HERE, YO!

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_precip1 = input_forcings.regridded_precip1
            input_forcings.regridded_precip2 = input_forcings.regridded_precip2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            # if not np.any(input_forcings.regridded_forcings1):
            if input_forcings.regridded_precip1 is None:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_precip1 = input_forcings.regridded_precip1
                input_forcings.regridded_precip2 = input_forcings.regridded_precip2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The forcing window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_precip1[:, :] = input_forcings.regridded_precip2[:, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input SBCV2_LWF file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input SBCV2_LWF file: " + \
                                           input_forcings.file_in2 + " not found. Will not use in final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(input_forcings.file_in2):
        if input_forcings.regridded_precip2 is not None:
            input_forcings.regridded_precip2[:, :] = config_options.globalNdv


def _find_ak_ext_ana_precip_stage4(supplemental_precip, config_options, d_current, mpi_config):
    # First we need to find the nearest previous and next hour, which is
    # the previous/next Stage IV files we will be using.
    d_current_epoch = int(d_current.strftime("%s"))
    six_hr_sec = 21600
    d_current_epoch -= 3600
    d_current_epoch += six_hr_sec
    #if we're at an even 6 hour multiple move the time back 6 hours as d_current is included at the end of the prior range
    #(begin_date,end_date]
    if d_current_epoch%six_hr_sec == 0:
        d_current_epoch -= six_hr_sec
    #next_stage4_date = datetime.datetime.fromtimestamp(d_current_epoch - d_current_epoch%six_hr_sec)
    #d_prev_epoch = d_current_epoch-six_hr_sec
    #prev_stage4_date = datetime.datetime.fromtimestamp(d_prev_epoch - d_prev_epoch%six_hr_sec)
    prev_stage4_date = datetime.datetime.fromtimestamp(d_current_epoch - d_current_epoch%six_hr_sec)
    next_stage4_date = prev_stage4_date
    # Set the input file frequency to be six-hourly.
    supplemental_precip.input_frequency = 360.0

    supplemental_precip.pcp_date1 = prev_stage4_date
    supplemental_precip.pcp_date2 = next_stage4_date

    try:
        #Use comma delimited string with first part containing Stage IV data and second part containing MRMS
        stage4_in_dir = supplemental_precip.inDir.split(',')[0]
    except IndexError:
        stage4_in_dir = None

    # Calculate expected file paths.
    tmp_file_ext = ".grb2" if supplemental_precip.fileType == 'GRIB2' else ".grb2.nc"
    if stage4_in_dir and supplemental_precip.keyValue == 11:
        tmp_file1 = f"{stage4_in_dir}/st4_ak.{supplemental_precip.pcp_date1.strftime('%Y%m%d%H.06h')}{tmp_file_ext}"
        #if d_current_epoch%six_hr_sec == 0:
        #    tmp_file2 = f"{stage4_in_dir}/st4_ak.{supplemental_precip.pcp_date2.strftime('%Y%m%d%H.06h')}{tmp_file_ext}"
        #else:
        tmp_file2 = f"{stage4_in_dir}/st4_ak.{supplemental_precip.pcp_date2.strftime('%Y%m%d%H.06h')}{tmp_file_ext}"
    else:
        tmp_file1 = tmp_file2 = ""

    # Compose the RQI paths.
    tmp_rqi_file1 = tmp_rqi_file2 = ""

    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous Stage IV supplemental file: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Next Stage IV supplemental file: " + tmp_file2
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if supplemental_precip.file_in1 != tmp_file1 or supplemental_precip.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
            supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
            supplemental_precip.regridded_rqi1 = supplemental_precip.regridded_rqi1
            supplemental_precip.regridded_rqi2 = supplemental_precip.regridded_rqi2
        else:
            # The forecast window has shifted. Reset fields 2 to
            # be fields 1.
            supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
            supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
            supplemental_precip.regridded_rqi1 = supplemental_precip.regridded_rqi1
            supplemental_precip.regridded_rqi2 = supplemental_precip.regridded_rqi2
            # supplemental_precip.regridded_precip1[:,:] = supplemental_precip.regridded_precip2[:,:]
            # supplemental_precip.regridded_rqi1[:, :] = supplemental_precip.regridded_rqi2[:, :]
        supplemental_precip.file_in1 = tmp_file1
        supplemental_precip.file_in2 = tmp_file2
        supplemental_precip.rqi_file_in1 = tmp_rqi_file1
        supplemental_precip.rqi_file_in2 = tmp_rqi_file2
        supplemental_precip.regridComplete = False

    # If either file does not exist, set to None. This will instruct downstream regridding steps to
    # set the regridded states to the global NDV. That ensures no supplemental precipitation will be
    # added to the final output grids.

    # if not os.path.isfile(tmp_file1) or not os.path.isfile(tmp_file2):
    #    if MpiConfig.rank == 0:
    #        ConfigOptions.statusMsg = "Stage IV files are missing. Will not process " \
    #                                  "supplemental precipitation"
    #        errMod.log_warning(ConfigOptions,MpiConfig)
    #    supplemental_precip.file_in2 = None
    #    supplemental_precip.file_in1 = None

    # errMod.check_program_status(ConfigOptions, MpiConfig)

    # Ensure we have the necessary new file
    if not os.path.isfile(supplemental_precip.file_in2) and supplemental_precip.keyValue == 11:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Stage IV file {} not found, will attempt to use {} instead.".format(
                supplemental_precip.file_in2, supplemental_precip.file_in1)
            err_handler.log_warning(config_options, mpi_config)
        supplemental_precip.file_in2 = supplemental_precip.file_in1
    if mpi_config.rank == 0 and not os.path.isfile(supplemental_precip.file_in2):
        if supplemental_precip.enforce == 1:
            config_options.errMsg = "Expected input Stage IV file: " + supplemental_precip.file_in2 + " not found."
            err_handler.log_critical(config_options, mpi_config)
        else:
            config_options.statusMsg = "Expected input Stage IV file: " + supplemental_precip.file_in2 + \
                                       " not found. " + "Will not use in final layering."
            err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(supplemental_precip.file_in2):
        if supplemental_precip.regridded_precip2 is not None:
            supplemental_precip.regridded_precip2[:, :] = config_options.globalNdv
    
    supplemental_precip.ext_ana = "STAGE4"


def _find_ak_ext_ana_precip_mrms(supplemental_precip, config_options, d_current, mpi_config):
    # First we need to find the nearest previous and next hour, which is
    # the previous/next MRMS files we will be using.
    #next_mrms_date = d_current
    prev_mrms_date = d_current - datetime.timedelta(hours=1)
    next_mrms_date = prev_mrms_date

    # Set the input file frequency to be hourly.
    supplemental_precip.input_frequency = 60.0

    supplemental_precip.pcp_date1 = prev_mrms_date
    supplemental_precip.pcp_date2 = next_mrms_date

    try:
        #Use comma delimited string with first part containing Stage IV data and second part containing MRMS
        mrms_in_dir = supplemental_precip.inDir.split(',')[1]
    except IndexError:
        mrms_in_dir = None

    # Calculate expected file paths.
    if supplemental_precip.keyValue == 11:
        tmp_file1 = mrms_in_dir + "/MultiSensor_QPE_01H_Pass1/" + \
                    supplemental_precip.pcp_date1.strftime('%Y%m%d') + "/" + \
                    "MRMS_MultiSensor_QPE_01H_Pass1_00.00_" + \
                    supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
                    "-" + supplemental_precip.pcp_date1.strftime('%H') + \
                    "0000.grib2.gz"
        tmp_file2 = mrms_in_dir + "/MultiSensor_QPE_01H_Pass2/" + \
                    supplemental_precip.pcp_date1.strftime('%Y%m%d') + "/" + \
                    "MRMS_MultiSensor_QPE_01H_Pass2_00.00_" + \
                    supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
                    "-" + supplemental_precip.pcp_date2.strftime('%H') + \
                    "0000.grib2.gz"
    else:
        tmp_file1 = tmp_file2 = ""

    # Compose the RQI paths.
    tmp_rqi_file1 = tmp_rqi_file2 = ""

    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous MRMS supplemental file: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Next MRMS supplemental file: " + tmp_file2
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if supplemental_precip.file_in1 != tmp_file1 or supplemental_precip.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
            supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
            supplemental_precip.regridded_rqi1 = supplemental_precip.regridded_rqi1
            supplemental_precip.regridded_rqi2 = supplemental_precip.regridded_rqi2
        else:
            # The forecast window has shifted. Reset fields 2 to
            # be fields 1.
            supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
            supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
            supplemental_precip.regridded_rqi1 = supplemental_precip.regridded_rqi1
            supplemental_precip.regridded_rqi2 = supplemental_precip.regridded_rqi2
            # supplemental_precip.regridded_precip1[:,:] = supplemental_precip.regridded_precip2[:,:]
            # supplemental_precip.regridded_rqi1[:, :] = supplemental_precip.regridded_rqi2[:, :]
        supplemental_precip.file_in1 = tmp_file1
        supplemental_precip.file_in2 = tmp_file2
        supplemental_precip.rqi_file_in1 = tmp_rqi_file1
        supplemental_precip.rqi_file_in2 = tmp_rqi_file2
        supplemental_precip.regridComplete = False

    # If either file does not exist, set to None. This will instruct downstream regridding steps to
    # set the regridded states to the global NDV. That ensures no supplemental precipitation will be
    # added to the final output grids.

    # if not os.path.isfile(tmp_file1) or not os.path.isfile(tmp_file2):
    #    if MpiConfig.rank == 0:
    #        ConfigOptions.statusMsg = "MRMS files are missing. Will not process " \
    #                                  "supplemental precipitation"
    #        errMod.log_warning(ConfigOptions,MpiConfig)
    #    supplemental_precip.file_in2 = None
    #    supplemental_precip.file_in1 = None

    # errMod.check_program_status(ConfigOptions, MpiConfig)

    # Ensure we have the necessary new file
    if not os.path.isfile(supplemental_precip.file_in2) and supplemental_precip.keyValue == 11:
        if mpi_config.rank == 0:
            config_options.statusMsg = "MRMS file {} not found, will attempt to use {} instead.".format(
                supplemental_precip.file_in2, supplemental_precip.file_in1)
            err_handler.log_warning(config_options, mpi_config)
        supplemental_precip.file_in2 = supplemental_precip.file_in1
    if mpi_config.rank == 0 and not os.path.isfile(supplemental_precip.file_in2):
        if supplemental_precip.enforce == 1:
            config_options.errMsg = "Expected input MRMS file: " + supplemental_precip.file_in2 + " not found."
            err_handler.log_critical(config_options, mpi_config)
        else:
            config_options.statusMsg = "Expected input MRMS file: " + supplemental_precip.file_in2 + \
                                       " not found. " + "Will not use in final layering."
            err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if not os.path.isfile(supplemental_precip.file_in2):
        if mpi_config.rank == 0:
            config_options.statusMsg = f"{supplemental_precip.file_in2} not found. Attempting to use Stage IV in place of MRMS"
            err_handler.log_warning(config_options, mpi_config)
        _find_ak_ext_ana_precip_stage4(supplemental_precip, config_options, d_current, mpi_config)
    else:
        supplemental_precip.ext_ana = "MRMS"


def find_ak_ext_ana_precip_neighbors(supplemental_precip, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and next Stage IV MPE files. 
    :param supplemental_precip:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    if d_current > config_options.e_date_proc - datetime.timedelta(hours=7):
        #print(f"Using MRMS hour {d_current}")
        #_find_ak_ext_ana_precip_mrms(supplemental_precip, config_options, d_current, mpi_config)
        supplemental_precip.ext_ana = "MRMS"
    else:
        #print(f"Using StageIV hour {d_current}")
        _find_ak_ext_ana_precip_stage4(supplemental_precip, config_options, d_current, mpi_config)


def find_hourly_nbm_apcp_neighbors(supplemental_precip, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and next NBM/CORE/CONUS files. This
    will also calculate the neighboring radar quality index (RQI) files as well.
    :param supplemental_precip:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    nbm_out_freq = {
    36: 60,
    240: 360
    }
        
    # First we need to find the nearest previous and next hour, which is
    # the previous/next NBM files we will be using.
    current_yr = d_current.year
    current_mo = d_current.month
    current_day = d_current.day
    current_hr = d_current.hour
    current_min = d_current.minute
 
    # First find the current NBM forecast cycle that we are using.
    current_nbm_cycle = config_options.current_fcst_cycle - \
        datetime.timedelta(seconds=supplemental_precip.userCycleOffset * 60.0)

    # Calculate the current forecast hour within this NBM cycle.
    dt_tmp = d_current - current_nbm_cycle
    current_nbm_hour = int(dt_tmp.days*24) + int(dt_tmp.seconds/3600.0)

    # Set the input file frequency to be hourly for f001-f036 and 6-hourly beyond f036.
    if current_nbm_hour <= 36:
        supplemental_precip.input_frequency = 60.0
    else:
        supplemental_precip.input_frequency = 360.0
            
    # Calculate the previous file to process.
    min_since_last_output = (current_nbm_hour*60) % supplemental_precip.input_frequency
    if min_since_last_output == 0:
        min_since_last_output = supplemental_precip.input_frequency
    prev_nbm_date = d_current - datetime.timedelta(seconds=min_since_last_output*60)
    supplemental_precip.fcst_date1 = prev_nbm_date
    if min_since_last_output == supplemental_precip.input_frequency:
        min_until_next_output = 0
    else:
        min_until_next_output = supplemental_precip.input_frequency - min_since_last_output
    next_nbm_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    supplemental_precip.fcst_date2 = next_nbm_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_nbm_date - current_nbm_cycle
    next_nbm_forecast_hour = int(dt_tmp.days*24.0) + int(dt_tmp.seconds/3600.0)
    supplemental_precip.fcst_hour2 = next_nbm_forecast_hour
    dt_tmp = prev_nbm_date - current_nbm_cycle
    prev_nbm_forecast_hour = int(dt_tmp.days*24.0) + int(dt_tmp.seconds/3600.0)
    supplemental_precip.fcst_hour1 = prev_nbm_forecast_hour
    # If we are on the first NBM forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prev_nbm_forecast_hour == 0:
        prev_nbm_forecast_hour = 1

    # Calculate expected file paths.
    if supplemental_precip.keyValue == 8:
        tmp_file1 = supplemental_precip.inDir + "/blend." + \
            current_nbm_cycle.strftime('%Y%m%d') + \
            "/" + current_nbm_cycle.strftime('%H') + \
            "/core/blend.t" + current_nbm_cycle.strftime('%H') + \
            "z.core.f" + str(next_nbm_forecast_hour).zfill(3) + ".co" \
            + supplemental_precip.file_ext
        tmp_file2 = supplemental_precip.inDir + "/blend." + \
            current_nbm_cycle.strftime('%Y%m%d') + \
            "/" + current_nbm_cycle.strftime('%H') + \
            "/core/blend.t" + current_nbm_cycle.strftime('%H') + \
            "z.core.f" + str(prev_nbm_forecast_hour).zfill(3) + ".co" \
            + supplemental_precip.file_ext
    elif supplemental_precip.keyValue == 9:
        tmp_file1 = supplemental_precip.inDir + "/blend." + \
            current_nbm_cycle.strftime('%Y%m%d') + \
            "/" + current_nbm_cycle.strftime('%H') + \
            "/core/blend.t" + current_nbm_cycle.strftime('%H') + \
            "z.core.f" + str(next_nbm_forecast_hour).zfill(3) + ".ak" \
            + supplemental_precip.file_ext
        tmp_file2 = supplemental_precip.inDir + "/blend." + \
            current_nbm_cycle.strftime('%Y%m%d') + \
            "/" + current_nbm_cycle.strftime('%H') + \
            "/core/blend.t" + current_nbm_cycle.strftime('%H') + \
            "z.core.f" + str(prev_nbm_forecast_hour).zfill(3) + ".ak" \
            + supplemental_precip.file_ext
    else:
        tmp_file1 = tmp_file2 = ""

    if mpi_config.rank == 0:
        config_options.statusMsg = "Prev NBM supplemental file: " + tmp_file2
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Next NBM supplemental file: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if supplemental_precip.file_in1 != tmp_file1 or supplemental_precip.file_in2 != tmp_file2:
        supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
        supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
        supplemental_precip.file_in1 = tmp_file1
        supplemental_precip.file_in2 = tmp_file2
        supplemental_precip.regridComplete = False

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.isfile(supplemental_precip.file_in2) and ((supplemental_precip.keyValue == 8) or (supplemental_precip.keyValue == 9)):
            config_options.statusMsg = "NBM file {} not found, will attempt to use {} instead.".format(
                    supplemental_precip.file_in2, supplemental_precip.file_in1)
            err_handler.log_warning(config_options, mpi_config)
            supplemental_precip.file_in2 = supplemental_precip.file_in1
        if not os.path.isfile(supplemental_precip.file_in2):
            if supplemental_precip.enforce == 1:
                config_options.errMsg = "Expected input NBM file: " + supplemental_precip.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input NBM file: " + supplemental_precip.file_in2 + \
                                           " not found. " + "Will not use in final layering."
                err_handler.log_warning(config_options, mpi_config)
                config_options.statusMsg = "You can use Util/pull_s3_grib_vars.py to Download NBM data from AWS-S3 archive."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(supplemental_precip.file_in2):
        if supplemental_precip.regridded_precip2 is not None:
            supplemental_precip.regridded_precip2[:, :] = config_options.globalNdv

    # Do we want to use NBM data at this timestep? If not, set the local slab of arrays to missing.
    if not config_options.use_data_at_current_time:
        if supplemental_precip.regridded_precip2 is not None:
            supplemental_precip.regridded_precip2[:, :] = config_options.globalNdv
        if supplemental_precip.regridded_precip1 is not None:
            supplemental_precip.regridded_precip1[:, :] = config_options.globalNdv
