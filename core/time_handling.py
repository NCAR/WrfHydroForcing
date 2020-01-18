# This is a datetime module for handling datetime
# calculations in the forcing engine.
import datetime
import math
import os

import numpy as np

from core import err_handler


def calculate_lookback_window(config_options):
    """
    Calculate the beginning, ending datetime variables
    for a look-back period. Also calculate the processing
    time window delta value.
    :param config_options: Abstract class holding job information.
    :return: Updated abstract class with updated datetime variables.
    """
    # First calculate the current time in UTC.
    d_current_utc = datetime.datetime.utcnow()

    # Next, subtract the lookup window (specified in minutes) to get a crude window
    # of processing.
    d_lookback = d_current_utc - datetime.timedelta(seconds=60 * config_options.look_back)

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
    config_options.n_fcsts = int(n_fcst_steps) + 1
    config_options.e_date_proc = config_options.b_date_proc + datetime.timedelta(
        seconds=n_fcst_steps * config_options.fcst_freq * 60)


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
    current_hrrr_cycle = config_options.current_fcst_cycle - datetime.timedelta(
        seconds=input_forcings.userCycleOffset * 60.0)
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
        '%Y%m%d') + "/hrrr.t" + current_hrrr_cycle.strftime('%H') + 'z.wrfsfcf' + \
        str(prev_hrrr_forecast_hour).zfill(2) + '.grib2'
    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous HRRR file being used: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)

    tmp_file2 = input_forcings.inDir + '/hrrr.' + current_hrrr_cycle.strftime(
        '%Y%m%d') + "/hrrr.t" + current_hrrr_cycle.strftime('%H') + 'z.wrfsfcf' \
        + str(next_hrrr_forecast_hour).zfill(2) + '.grib2'
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
        if not os.path.isfile(input_forcings.file_in2):
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
    if not os.path.isfile(input_forcings.file_in2):
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
    current_rap_cycle = config_options.current_fcst_cycle - datetime.timedelta(
            seconds=input_forcings.userCycleOffset * 60.0)
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
    if prev_rap_forecast_hour == 0:
        prev_rap_forecast_hour = 1

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + '/rap.' + \
        current_rap_cycle.strftime('%Y%m%d') + "/rap.t" + \
        current_rap_cycle.strftime('%H') + 'z.awp130bgrbf' + \
        str(prev_rap_forecast_hour).zfill(2) + '.grib2'
    tmp_file2 = input_forcings.inDir + '/rap.' + \
        current_rap_cycle.strftime('%Y%m%d') + "/rap.t" + \
        current_rap_cycle.strftime('%H') + 'z.awp130bgrbf' + \
        str(next_rap_forecast_hour).zfill(2) + '.grib2'

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
    if mpi_config.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input RAP file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input RAP file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.isfile(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv


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
            str(prev_gfs_forecast_hour).zfill(3) + '.grib2'
        tmp_file2 = input_forcings.inDir + '/gfs.' + \
            current_gfs_cycle.strftime('%Y%m%d%H') + "/gfs.t" + \
            current_gfs_cycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(next_gfs_forecast_hour).zfill(3) + '.grib2'
    else:
        # FV3 change on June 12th, 2019
        tmp_file1 = input_forcings.inDir + '/gfs.' + \
            current_gfs_cycle.strftime('%Y%m%d') + "/" + \
            current_gfs_cycle.strftime('%H') + "/gfs.t" + \
            current_gfs_cycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(prev_gfs_forecast_hour).zfill(3) + '.grib2'
        tmp_file2 = input_forcings.inDir + '/gfs.' + \
            current_gfs_cycle.strftime('%Y%m%d') + "/" + \
            current_gfs_cycle.strftime('%H') + "/gfs.t" + \
            current_gfs_cycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(next_gfs_forecast_hour).zfill(3) + '.grib2'

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
    elif input_forcings.keyValue == 14:
        domain_string = "priconest"
    elif input_forcings.keyValue == 15:
        domain_string = "alaskanest"
    else:
        domain_string = ''

    # Calculate expected file paths.
    tmp_file1 = input_forcings.inDir + '/nam.' + \
        current_nam_nest_cycle.strftime('%Y%m%d') + "/nam.t" + \
        current_nam_nest_cycle.strftime('%H') + 'z.' + domain_string + '.hiresf' + \
        str(prev_nam_nest_forecast_hour).zfill(2) + '.tm00.grib2'
    tmp_file2 = input_forcings.inDir + '/nam.' + \
        current_nam_nest_cycle.strftime('%Y%m%d') + "/nam.t" + \
        current_nam_nest_cycle.strftime('%H') + 'z.' + domain_string + '.hiresf' + \
        str(next_nam_nest_forecast_hour).zfill(2) + '.tm00.grib2'
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
        ".grb2"
    tmp_file2 = input_forcings.inDir + "/cfs." + \
        current_cfs_cycle.strftime('%Y%m%d') + "/" + \
        current_cfs_cycle.strftime('%H') + "/" + \
        "6hrly_grib_" + ens_str + "/flxf" + \
        next_cfs_date.strftime('%Y%m%d%H') + "." + \
        ens_str + "." + current_cfs_cycle.strftime('%Y%m%d%H') + \
        ".grb2"

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
    supplemental_precip.pcp_date2 = next_mrms_date

    # Calculate expected file paths.
    if supplemental_precip.keyValue == 1:
        tmp_file1 = supplemental_precip.inDir + "/RadarOnly_QPE_01H/" + \
            "MRMS_RadarOnly_QPE_01H_00.00_" + \
            supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
            "-" + supplemental_precip.pcp_date1.strftime('%H') + \
            "0000.grib2.gz"
        tmp_file2 = supplemental_precip.inDir + "/RadarOnly_QPE_01H/" + \
            "MRMS_RadarOnly_QPE_01H_00.00_" + \
            supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
            "-" + supplemental_precip.pcp_date2.strftime('%H') + \
            "0000.grib2.gz"
    elif supplemental_precip.keyValue == 2:
        tmp_file1 = supplemental_precip.inDir + "/GaugeCorr_QPE_01H/" + \
            "MRMS_GaugeCorr_QPE_01H_00.00_" + \
            supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
            "-" + supplemental_precip.pcp_date1.strftime('%H') + \
            "0000.grib2.gz"
        tmp_file2 = supplemental_precip.inDir + "/GaugeCorr_QPE_01H/" + \
            "MRMS_GaugeCorr_QPE_01H_00.00_" + \
            supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
            "-" + supplemental_precip.pcp_date2.strftime('%H') + \
            "0000.grib2.gz"
    else:
        tmp_file1 = tmp_file2 = ""

    # Compose the RQI paths.
    tmp_rqi_file1 = supplemental_precip.inDir + "/RadarQualityIndex/" + \
        "MRMS_RadarQualityIndex_00.00_" + \
        supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
        "-" + supplemental_precip.pcp_date1.strftime('%H') + \
        "0000.grib2.gz"
    tmp_rqi_file2 = supplemental_precip.inDir + "/RadarQualityIndex/" + \
        "MRMS_RadarQualityIndex_00.00_" + \
        supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
        "-" + supplemental_precip.pcp_date2.strftime('%H') + \
        "0000.grib2.gz"
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


def find_hourly_wrf_arw_hi_res_pcp_neighbors(supplemental_precip, config_options, d_current, mpi_config):
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
        if current_arw_cycle.hour != 6 and current_arw_cycle.hour != 18:
            if mpi_config.rank == 0:
                config_options.statusMsg = "No Puerto Rico WRF-ARW found for this cycle. " \
                                           "No supplemental ARW precipitation will be used."
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
    if supplemental_precip.keyValue == 3:
        tmp_file1 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(prev_arw_forecast_hour).zfill(2) + '.hi.grib2'
        tmp_file2 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(next_arw_forecast_hour).zfill(2) + '.hi.grib2'
    elif supplemental_precip.keyValue == 4:
        tmp_file1 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(prev_arw_forecast_hour).zfill(2) + '.pr.grib2'
        tmp_file2 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(next_arw_forecast_hour).zfill(2) + '.pr.grib2'
    else:
        tmp_file1 = tmp_file2 = supplemental_precip.inDir + '/hiresw.' + \
            current_arw_cycle.strftime('%Y%m%d') + '/hiresw.t' + \
            current_arw_cycle.strftime('%H') + 'z.arw_2p5km.f' + \
            str(next_arw_forecast_hour).zfill(2) + '.hi.grib2'

    err_handler.check_program_status(config_options, mpi_config)
    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous ARW supplemental precipitation file: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)
        config_options.statusMsg = "Next ARW supplemental precipitation file: " + tmp_file2
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
