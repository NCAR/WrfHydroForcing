# This is a datetime module for handling datetime
# calculations in the forcing engine.
from core import errMod
import datetime
import math
import os
import numpy as np

def calculate_lookback_window(ConfigOptions):
    """
    Calculate the beginning, ending datetime variables
    for a look-back period. Also calculate the processing
    time window delta value.
    :param ConfigOptions: Abstract class holding job information.
    :return: Updated abstract class with updated datetime variables.
    """
    # First calculate the current time in UTC.
    dCurrentUtc = datetime.datetime.utcnow()

    # Next, subtract the lookup window (specified in minutes) to get a crude window
    # of processing.
    dLookback = dCurrentUtc - datetime.timedelta(seconds=60*ConfigOptions.look_back)

    # Determine the first forecast iteration that will be processed on this day
    # based on the forecast frequency and where in the day we are at.
    fcstStepTmp = math.ceil((dLookback.hour*60 + dLookback.minute)/ConfigOptions.fcst_freq)
    dLookTmp1 = datetime.datetime(dLookback.year,dLookback.month,dLookback.day)
    dLookTmp1 = dLookTmp1 + datetime.timedelta(seconds=60*ConfigOptions.fcst_freq*fcstStepTmp)

    # If we are offsetting the forecasts, apply here.
    if ConfigOptions.fcst_shift > 0:
        dLookTmp1 = dLookTmp1 + datetime.timedelta(seconds=60*ConfigOptions.fcst_shift)

    ConfigOptions.b_date_proc = dLookTmp1


    # Now calculate the end of the processing window based on the time from the
    # beginning of the processing window.
    dtTmp = dCurrentUtc - ConfigOptions.b_date_proc
    nFcstSteps = math.floor((dtTmp.days*1440+dtTmp.seconds/60.0)/ConfigOptions.fcst_freq)
    ConfigOptions.nFcsts = int(nFcstSteps) + 1
    ConfigOptions.e_date_proc = ConfigOptions.b_date_proc + datetime.timedelta(seconds=nFcstSteps*ConfigOptions.fcst_freq*60)

def find_conus_hrrr_neighbors(input_forcings,ConfigOptions,dCurrent,MpiConfg):
    """
    Function to calculate the previous and after HRRR conus cycles based on the current timestep.
    :param input_forcings:
    :param ConfigOptions:
    :param dCurrent:
    :param MpiConfg:
    :return:
    """
    if MpiConfg.rank == 0:
        ConfigOptions.statusMsg = "Processing Conus HRRR Data. Calculating neighboring " \
                                  "files for this output timestep"
        errMod.log_msg(ConfigOptions,MpiConfg)

    # Fortunately, HRRR data is straightforward compared to GFS in terms of precip values, etc.
    if dCurrent >= datetime.datetime(2018,10,1):
        defaultHorizon = 18 # 18-hour forecasts.
        sixHrHorizon = 36 # 36-hour forecasts every six hours.
    else:
        defaultHorizon = 18  # 18-hour forecasts.
        sixHrHorizon = 18  # 18-hour forecasts every six hours.

    # First find the current HRRR forecast cycle that we are using.
    currentHrrrCycle = ConfigOptions.current_fcst_cycle - \
                       datetime.timedelta(seconds=
                                          (input_forcings.userCycleOffset) * 60.0)
    if currentHrrrCycle.hour%6 != 0:
        hrrrHorizon = defaultHorizon
    else:
        hrrrHorizon = sixHrHorizon

    # If the user has specified a forcing horizon that is greater than what is available
    # for this time period, throw an error.
    if (input_forcings.userFcstHorizon + input_forcings.userCycleOffset) / 60.0 > hrrrHorizon:
        ConfigOptions.errMsg = "User has specified a HRRR conus forecast horizon " + \
            "that is greater than the maximum allowed hours of: " + \
            str(hrrrHorizon)
        errMod.log_critical(ConfigOptions,MpiConfg)
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Calculate the current forecast hour within this HRRR cycle.
    dtTmp = dCurrent - currentHrrrCycle
    currentHrrrHour = int(dtTmp.days*24) + int(dtTmp.seconds/3600.0)

    # Calculate the previous file to process.
    minSinceLastOutput = (currentHrrrHour * 60) % 60
    if minSinceLastOutput == 0:
        minSinceLastOutput = 60
    prevHrrrDate = dCurrent - datetime.timedelta(seconds=minSinceLastOutput * 60)
    input_forcings.fcst_date1 = prevHrrrDate
    if minSinceLastOutput == 60:
        minUntilNextOutput = 0
    else:
        minUntilNextOutput = 60 - minSinceLastOutput
    nextHrrrDate = dCurrent + datetime.timedelta(seconds=minUntilNextOutput * 60)
    input_forcings.fcst_date2 = nextHrrrDate

    # Calculate the output forecast hours needed based on the prev/next dates.
    dtTmp = nextHrrrDate - currentHrrrCycle
    nextHrrrForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = nextHrrrForecastHour
    dtTmp = prevHrrrDate - currentHrrrCycle
    prevHrrrForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prevHrrrForecastHour
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # If we are on the first HRRR forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prevHrrrForecastHour == 0:
        prevHrrrForecastHour = 1

    # Calculate expected file paths.
    tmpFile1 = input_forcings.inDir + '/hrrr.' + \
               currentHrrrCycle.strftime('%Y%m%d') + "/hrrr.t" + \
               currentHrrrCycle.strftime('%H') + 'z.wrfsfcf' + \
               str(prevHrrrForecastHour).zfill(2) + '.grib2'
    if MpiConfg.rank == 0:
        ConfigOptions.statusMsg = "Previous HRRR file being used: " + tmpFile1
        errMod.log_msg(ConfigOptions,MpiConfg)

    tmpFile2 = input_forcings.inDir + '/hrrr.' + \
               currentHrrrCycle.strftime('%Y%m%d') + "/hrrr.t" + \
               currentHrrrCycle.strftime('%H') + 'z.wrfsfcf' + \
               str(nextHrrrForecastHour).zfill(2) + '.grib2'
    if MpiConfg.rank == 0:
        if MpiConfg.rank == 0:
            ConfigOptions.statusMsg = "Next HRRR file being used: " + tmpFile2
            errMod.log_msg(ConfigOptions, MpiConfg)
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmpFile1 or input_forcings.file_in2 != tmpFile2:
        if ConfigOptions.current_output_step == 1:
            print('We are on the first output timestep.')
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmpFile1
            input_forcings.file_in2 = tmpFile2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
            #if not np.any(input_forcings.regridded_forcings1):
                if MpiConfg.rank == 0:
                    ConfigOptions.statusMsg = "Restarting forecast cyle. Will regrid previous: " + \
                                              input_forcings.productName
                    errMod.log_msg(ConfigOptions, MpiConfg)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmpFile1
                input_forcings.file_in1 = tmpFile1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The HRRR window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmpFile1
                input_forcings.file_in2 = tmpFile2
        input_forcings.regridComplete = False
    errMod.check_program_status(ConfigOptions, MpiConfg)

    # Ensure we have the necessary new file
    if MpiConfg.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                ConfigOptions.errMsg = "Expected input HRRR file: " + input_forcings.file_in2 + " not found."
                errMod.log_critical(ConfigOptions,MpiConfg)
            else:
                ConfigOptions.statusMsg = "Expected input HRRR file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                errMod.log_msg(ConfigOptions, MpiConfg)
                if input_forcings.regridded_forcings2 is not None:
                #if np.any(input_forcings.regridded_forcings2):
                    input_forcings.regridded_forcings2[:, :, :] = ConfigOptions.globalNdv

    errMod.check_program_status(ConfigOptions, MpiConfg)

def find_conus_rap_neighbors(input_forcings,ConfigOptions,dCurrent,MpiConfg):
    """
    Function to calculate the previous and after RAP conus 13km cycles based on the current timestep.
    :param input_forcings:
    :param ConfigOptions:
    :param dCurrent:
    :param MpiConfg:
    :return:
    """
    if MpiConfg.rank == 0:
        if input_forcings.keyValue == 5:
            print("PROCESSING conus RAP")

    if dCurrent >= datetime.datetime(2018,10,1):
        defaultHorizon = 21 # 21-hour forecasts.
        extraHrHorizon = 39 # 39-hour forecasts at 3,9,15,21 UTC.
    else:
        defaultHorizon = 18  # 18-hour forecasts.
        extraHrHorizon = 18  # 18-hour forecasts every six hours.

    # First find the current RAP forecast cycle that we are using.
    currentRapCycle = ConfigOptions.current_fcst_cycle - \
                      datetime.timedelta(seconds=
                                         (input_forcings.userCycleOffset) * 60.0)
    if currentRapCycle.hour == 3 or currentRapCycle.hour == 9 or \
            currentRapCycle.hour == 15 or currentRapCycle.hour == 21:
        rapHorizon = defaultHorizon
    else:
        rapHorizon = extraHrHorizon

    # If the user has specified a forcing horizon that is greater than what is available
    # for this time period, throw an error.
    if (input_forcings.userFcstHorizon + input_forcings.userCycleOffset) / 60.0 > rapHorizon:
        ConfigOptions.errMsg = "User has specified a RAP conus 13km forecast horizon " + \
            "that is greater than the maximum allowed hours of: " + \
            str(rapHorizon)
        print(ConfigOptions.errMsg)
        raise Exception

    if MpiConfg.rank == 0:
        print("CURRENT RAP CYCLE BEING USED = " + currentRapCycle.strftime('%Y-%m-%d %H'))

    # Calculate the current forecast hour within this HRRR cycle.
    dtTmp = dCurrent - currentRapCycle
    currentRapHour = int(dtTmp.days*24) + int(dtTmp.seconds/3600.0)
    if MpiConfg.rank == 0:
        print("Current RAP Forecast Hour = " + str(currentRapHour))

    # Calculate the previous file to process.
    minSinceLastOutput = (currentRapHour * 60) % 60
    if MpiConfg.rank == 0:
        print(currentRapHour)
        print(minSinceLastOutput)
    if minSinceLastOutput == 0:
        minSinceLastOutput = 60
    prevRapDate = dCurrent - datetime.timedelta(seconds=minSinceLastOutput * 60)
    input_forcings.fcst_date1 = prevRapDate
    if MpiConfg.rank == 0:
        print(prevRapDate)
    if minSinceLastOutput == 60:
        minUntilNextOutput = 0
    else:
        minUntilNextOutput = 60 - minSinceLastOutput
    nextRapDate = dCurrent + datetime.timedelta(seconds=minUntilNextOutput * 60)
    input_forcings.fcst_date2 = nextRapDate
    if MpiConfg.rank == 0:
        print(nextRapDate)

    # Calculate the output forecast hours needed based on the prev/next dates.
    dtTmp = nextRapDate - currentRapCycle
    if MpiConfg.rank == 0:
        print(currentRapCycle)
    nextRapForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    if MpiConfg.rank == 0:
        print(nextRapForecastHour)
    input_forcings.fcst_hour2 = nextRapForecastHour
    dtTmp = prevRapDate - currentRapCycle
    prevRapForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    if MpiConfg.rank == 0:
        print(prevRapForecastHour)
    input_forcings.fcst_hour1 = prevRapForecastHour
    # If we are on the first GFS forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prevRapForecastHour == 0:
        prevRapForecastHour = 1

    # Calculate expected file paths.
    tmpFile1 = input_forcings.inDir + '/rap.' + \
               currentRapCycle.strftime('%Y%m%d') + "/rap.t" + \
               currentRapCycle.strftime('%H') + 'z.awp130bgrbf' + \
               str(prevRapForecastHour).zfill(2) + '.grib2'
    if MpiConfg.rank == 0:
        print(tmpFile1)
    tmpFile2 = input_forcings.inDir + '/rap.' + \
               currentRapCycle.strftime('%Y%m%d') + "/rap.t" + \
               currentRapCycle.strftime('%H') + 'z.awp130bgrbf' + \
               str(nextRapForecastHour).zfill(2) + '.grib2'
    if MpiConfg.rank == 0:
        print(tmpFile2)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmpFile1 or input_forcings.file_in2 != tmpFile2:
        if ConfigOptions.current_output_step == 1:
            print('We are on the first output timestep.')
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmpFile1
            input_forcings.file_in2 = tmpFile2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
            #if not np.any(input_forcings.regridded_forcings1):
                if MpiConfg.rank == 0:
                    ConfigOptions.statusMsg = "Restarting forecast cyle. Will regrid previous: " + \
                                              input_forcings.productName
                    errMod.log_msg(ConfigOptions, MpiConfg)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmpFile1
                input_forcings.file_in1 = tmpFile1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The Rapid Refresh window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmpFile1
                input_forcings.file_in2 = tmpFile2
        input_forcings.regridComplete = False
    errMod.check_program_status(ConfigOptions, MpiConfg)

    # Ensure we have the necessary new file
    if MpiConfg.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                ConfigOptions.errMsg = "Expected input RAP file: " + input_forcings.file_in2 + " not found."
                errMod.log_critical(ConfigOptions, MpiConfg)
            else:
                ConfigOptions.statusMsg = "Expected input RAP file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                errMod.log_msg(ConfigOptions, MpiConfg)
                if input_forcings.regridded_forcings2 is not None:
                #if np.any(input_forcings.regridded_forcings2):
                    input_forcings.regridded_forcings2[:, :, :] = ConfigOptions.globalNdv

    errMod.check_program_status(ConfigOptions, MpiConfg)

def find_gfs_neighbors(input_forcings,ConfigOptions,dCurrent,MpiConfg):
    """
    Function to calcualte the previous and after GFS cycles based on the current timestep.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    if MpiConfg.rank == 0:
        if input_forcings.keyValue == 9:
            print("PROCESSING 0.25 Degree GFS")
        if input_forcings.keyValue == 3:
            print("PROCESSING Production GFS")

    # First calculate how the GFS files are structured based on our current processing date.
    # This will change in the future, and should be modified as the GFS system evolves.
    if dCurrent >= datetime.datetime(2018,10,1):
        gfsOutHorizons = [120, 240, 384]
        gfsOutFreq = {
            120: 60,
            240: 180,
            384: 720
        }
        gfsPrecipDelineators = {
            120: [360,60],
            240: [360,180],
            284: [720,720]
        }
    else:
        gfsOutHorizons = [120, 240, 384]
        gfsOutFreq = {
            120: 60,
            240: 180,
            384: 720
        }
        gfsPrecipDelineators = {
            120: [360, 60],
            240: [360, 180],
            284: [720, 720]
        }

    # If the user has specified a forcing horizon that is greater than what
    # is available here, return an error.
    if (input_forcings.userFcstHorizon+input_forcings.userCycleOffset)/60.0 > max(gfsOutHorizons):
        ConfigOptions.errMsg = "User has specified a GFS forecast horizon " \
                               "that is greater than maximum allowed hours of: " \
                               + str(max(gfsOutHorizons))
        print(ConfigOptions.errMsg)
        raise Exception

    if MpiConfg.rank == 0:
        print('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
    # First determine if we need a previous file or not. For hourly files,
    # only need the previous file if it's not the first hour after a new GFS cycle
    # (I.E. 7z, 13z, 19z, 1z). Any other hourly timestep requires the previous hourly data
    # (I.E. 8z needs 7z file). For 3-hourly files, we will look for files for the previous
    # three hours and next three hourly output. For example, a timestep that falls 125 hours
    # after the beginning of the GFS cycle will require data from the f123 file and 126 file.
    # For any files after 240 hours out from the beginning of a GFS forecast cycle, the intervals
    # move to 12 hours.

    # First find the current GFS forecast cycle that we are using.
    currentGfsCycle = ConfigOptions.current_fcst_cycle - \
                      datetime.timedelta(seconds=
                                         (input_forcings.userCycleOffset)*60.0)
    if MpiConfg.rank == 0:
        print("CURRENT GFS CYCLE BEING USED = " + currentGfsCycle.strftime('%Y-%m-%d %H'))

    # Calculate the current forecast hour within this GFS cycle.
    dtTmp = dCurrent - currentGfsCycle
    currentGfsHour = int(dtTmp.days*24) + int(dtTmp.seconds/3600.0)
    if MpiConfg.rank == 0:
        print("Current GFS Forecast Hour = " + str(currentGfsHour))

    # Calculate the GFS output frequency based on our current GFS forecast hour.
    for horizonTmp in gfsOutHorizons:
        if currentGfsHour <= horizonTmp:
            currentGfsFreq = gfsOutFreq[horizonTmp]
            input_forcings.outFreq = gfsOutFreq[horizonTmp]
            break
    if MpiConfg.rank == 0:
        print("Current GFS output frequency = " + str(currentGfsFreq))

    # Calculate the previous file to process.
    minSinceLastOutput = (currentGfsHour*60)%currentGfsFreq
    if MpiConfg.rank == 0:
        print(currentGfsHour)
        print(currentGfsFreq)
        print(minSinceLastOutput)
    if minSinceLastOutput == 0:
        minSinceLastOutput = currentGfsFreq
        #currentGfsHour = currentGfsHour
        #previousGfsHour = currentGfsHour - int(currentGfsFreq/60.0)
    prevGfsDate = dCurrent - \
                  datetime.timedelta(seconds=minSinceLastOutput*60)
    input_forcings.fcst_date1 = prevGfsDate
    if MpiConfg.rank == 0:
        print(prevGfsDate)
    if minSinceLastOutput == currentGfsFreq:
        minUntilNextOutput = 0
    else:
        minUntilNextOutput = currentGfsFreq - minSinceLastOutput
    nextGfsDate = dCurrent + datetime.timedelta(seconds=minUntilNextOutput*60)
    input_forcings.fcst_date2 = nextGfsDate
    if MpiConfg.rank == 0:
        print(nextGfsDate)

    # Calculate the output forecast hours needed based on the prev/next dates.
    dtTmp = nextGfsDate - currentGfsCycle
    if MpiConfg.rank == 0:
        print(currentGfsCycle)
    nextGfsForecastHour = int(dtTmp.days*24.0) + int(dtTmp.seconds/3600.0)
    if MpiConfg.rank == 0:
        print(nextGfsForecastHour)
    input_forcings.fcst_hour2 = nextGfsForecastHour
    dtTmp = prevGfsDate - currentGfsCycle
    prevGfsForecastHour = int(dtTmp.days*24.0) + int(dtTmp.seconds/3600.0)
    if MpiConfg.rank == 0:
        print(prevGfsForecastHour)
    input_forcings.fcst_hour1 = prevGfsForecastHour
    # If we are on the first GFS forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prevGfsForecastHour == 0:
        prevGfsForecastHour = 1

    # Calculate expected file paths.
    if currentGfsCycle < datetime.datetime(2019,6,12,12):
        tmpFile1 = input_forcings.inDir + '/gfs.' + \
            currentGfsCycle.strftime('%Y%m%d%H') + "/gfs.t" + \
            currentGfsCycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(prevGfsForecastHour).zfill(3) + '.grib2'
        if MpiConfg.rank == 0:
            print(tmpFile1)
        tmpFile2 = input_forcings.inDir + '/gfs.' + \
            currentGfsCycle.strftime('%Y%m%d%H') + "/gfs.t" + \
            currentGfsCycle.strftime('%H') + 'z.sfluxgrbf' + \
            str(nextGfsForecastHour).zfill(3) + '.grib2'
    else:
        # FV3 change on June 12th, 2019
        tmpFile1 = input_forcings.inDir + '/gfs.' + \
                   currentGfsCycle.strftime('%Y%m%d') + "/" + \
                   currentGfsCycle.strftime('%H') + "/gfs.t" + \
                   currentGfsCycle.strftime('%H') + 'z.sfluxgrbf' + \
                   str(prevGfsForecastHour).zfill(3) + '.grib2'
        if MpiConfg.rank == 0:
            print(tmpFile1)
        tmpFile2 = input_forcings.inDir + '/gfs.' + \
                   currentGfsCycle.strftime('%Y%m%d') + "/" + \
                   currentGfsCycle.strftime('%H') + "/gfs.t" + \
                   currentGfsCycle.strftime('%H') + 'z.sfluxgrbf' + \
                   str(nextGfsForecastHour).zfill(3) + '.grib2'
    if MpiConfg.rank == 0:
        print(tmpFile2)
        print('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')

    # If needed, initialize the globalPcpRate1 array (this is for when we have the initial grid, or we need to change
    # grids.
    if input_forcings.globalPcpRate2 is not None and input_forcings.globalPcpRate1 is None:
    #if np.any(input_forcings.globalPcpRate2) and not np.any(input_forcings.globalPcpRate1):
        input_forcings.globalPcpRate1 = np.empty([input_forcings.globalPcpRate2.shape[0],
                                                  input_forcings.globalPcpRate2.shape[1]], np.float32)
    if input_forcings.globalPcpRate2 is not None and input_forcings.globalPcpRate1 is not None:
    #if np.any(input_forcings.globalPcpRate2) and np.any(input_forcings.globalPcpRate1):
        if input_forcings.globalPcpRate2.shape[0] != input_forcings.globalPcpRate1.shape[0]:
            # The grid has changed, we need to re-initialize the globalPcpRate1 array.
            input_forcings.globalPcpRate1 = None
            input_forcings.globalPcpRate1 = np.empty([input_forcings.globalPcpRate2.shape[0],
                                                      input_forcings.globalPcpRate2.shape[1]], np.float32)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmpFile1 or input_forcings.file_in2 != tmpFile2:
        if ConfigOptions.current_output_step == 1:
            print('We are on the first output timestep.')
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmpFile1
            input_forcings.file_in2 = tmpFile2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
            #if not np.any(input_forcings.regridded_forcings1):
                if MpiConfg.rank == 0:
                    ConfigOptions.statusMsg = "Restarting forecast cyle. Will regrid previous: " + \
                                              input_forcings.productName
                    errMod.log_msg(ConfigOptions, MpiConfg)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                if input_forcings.productName == "GFS_Production_GRIB2":
                    if MpiConfg.rank == 0:
                        input_forcings.globalPcpRate1 = input_forcings.globalPcpRate1
                        input_forcings.globalPcpRate2 = input_forcings.globalPcpRate2
                input_forcings.file_in2 = tmpFile1
                input_forcings.file_in1 = tmpFile1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The forcing window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                if input_forcings.productName == "GFS_Production_GRIB2":
                    if MpiConfg.rank == 0:
                        input_forcings.globalPcpRate1[:, :] = input_forcings.globalPcpRate2[:, :]
                input_forcings.file_in1 = tmpFile1
                input_forcings.file_in2 = tmpFile2
        input_forcings.regridComplete = False
    errMod.check_program_status(ConfigOptions, MpiConfg)

    # Ensure we have the necessary new file
    if MpiConfg.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                ConfigOptions.errMsg = "Expected input GFS file: " + input_forcings.file_in2 + " not found."
                errMod.log_critical(ConfigOptions, MpiConfg)
            else:
                ConfigOptions.statusMsg = "Expected input GFS file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                errMod.log_msg(ConfigOptions, MpiConfg)
                if input_forcings.regridded_forcings2 is not None:
                #if np.any(input_forcings.regridded_forcings2):
                    input_forcings.regridded_forcings2[:, :, :] = ConfigOptions.globalNdv

    errMod.check_program_status(ConfigOptions, MpiConfg)

def find_nam_nest_neighbors(input_forcings,ConfigOptions,dCurrent,MpiConfg):
    """
    Function to calcualte the previous and after NAM Nest cycles based on the current timestep.
    This function was written to be generic enough to accomodate any of the NAM nest flavors.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    # Establish how often our NAM nest cycles occur.
    if dCurrent >= datetime.datetime(2012,10,1):
        namNestOutHorizons = [60]
        namNestOutFreq = {
            60: 60
        }

    # If the user has specified a forcing horizon that is greater than what
    # is available here, return an error.
    if (input_forcings.userFcstHorizon+input_forcings.userCycleOffset)/60.0 > max(namNestOutHorizons):
        ConfigOptions.errMsg = "User has specified a NAM nest forecast horizon " \
                               "that is greater than maximum allowed hours of: " \
                               + str(max(namNestOutHorizons))
        errMod.log_critical(ConfigOptions,MpiConfg)
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # First find the current NAM nest forecast cycle that we are using.
    currentNamNestCycle = ConfigOptions.current_fcst_cycle - \
                          datetime.timedelta(seconds=
                                             (input_forcings.userCycleOffset)*60.0)
    if MpiConfg.rank == 0:
        ConfigOptions.statusMsg = "Current NAM nest cycle being used: " + currentNamNestCycle.strftime('%Y-%m-%d %H')
        errMod.log_msg(ConfigOptions,MpiConfg)
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Calculate the current forecast hour within this NAM nest cycle.
    dtTmp = dCurrent - currentNamNestCycle
    currentNamNestHour = int(dtTmp.days*24) + int(dtTmp.seconds/3600.0)

    # Calculate the NAM Nest output frequency based on our current NAM Nest forecast hour.
    for horizonTmp in namNestOutHorizons:
        if currentNamNestHour <= horizonTmp:
            currentNamNestFreq = namNestOutFreq[horizonTmp]
            input_forcings.outFreq = namNestOutFreq[horizonTmp]
            break

    # Calculate the previous file to process.
    minSinceLastOutput = (currentNamNestHour*60)%currentNamNestFreq
    if minSinceLastOutput == 0:
        minSinceLastOutput = currentNamNestFreq
    prevNamNestDate = dCurrent - \
                  datetime.timedelta(seconds=minSinceLastOutput*60)
    input_forcings.fcst_date1 = prevNamNestDate
    if minSinceLastOutput == currentNamNestFreq:
        minUntilNextOutput = 0
    else:
        minUntilNextOutput = currentNamNestFreq - minSinceLastOutput
    nextNamNestDate = dCurrent + datetime.timedelta(seconds=minUntilNextOutput*60)
    input_forcings.fcst_date2 = nextNamNestDate
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Calculate the output forecast hours needed based on the prev/next dates.
    dtTmp = nextNamNestDate - currentNamNestCycle
    nextNamNestForecastHour = int(dtTmp.days*24.0) + int(dtTmp.seconds/3600.0)
    input_forcings.fcst_hour2 = nextNamNestForecastHour
    dtTmp = prevNamNestDate - currentNamNestCycle
    prevNamNestForecastHour = int(dtTmp.days*24.0) + int(dtTmp.seconds/3600.0)
    input_forcings.fcst_hour1 = prevNamNestForecastHour
    # If we are on the first NAM nest forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prevNamNestForecastHour == 0:
        prevNamNestForecastHour = 1

    if input_forcings.keyValue == 13:
        domainString = "hawaiinest"
    elif input_forcings.keyValue == 14:
        domainString = "priconest"
    elif input_forcings.keyValue == 15:
        domainString = "alaskanest"

    # Calculate expected file paths.
    tmpFile1 = input_forcings.inDir + '/nam.' + \
        currentNamNestCycle.strftime('%Y%m%d') + "/nam.t" + \
        currentNamNestCycle.strftime('%H') + 'z.' + domainString + '.hiresf' + \
        str(prevNamNestForecastHour).zfill(2) + '.tm00.grib2'
    tmpFile2 = input_forcings.inDir + '/nam.' + \
               currentNamNestCycle.strftime('%Y%m%d') + "/nam.t" + \
               currentNamNestCycle.strftime('%H') + 'z.' + domainString + '.hiresf' + \
               str(nextNamNestForecastHour).zfill(2) + '.tm00.grib2'
    if MpiConfg.rank == 0:
        ConfigOptions.statusMsg = "Previous NAM nest file being used: " + tmpFile1
        errMod.log_msg(ConfigOptions,MpiConfg)
        ConfigOptions.statusMsg = "Next NAM nest file being used: " + tmpFile2
        errMod.log_msg(ConfigOptions, MpiConfg)
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmpFile1 or input_forcings.file_in2 != tmpFile2:
        if ConfigOptions.current_output_step == 1:
            print('We are on the first output timestep.')
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmpFile1
            input_forcings.file_in2 = tmpFile2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
            #if not np.any(input_forcings.regridded_forcings1):
                if MpiConfg.rank == 0:
                    ConfigOptions.statusMsg = "Restarting forecast cyle. Will regrid previous: " + \
                                              input_forcings.productName
                    errMod.log_msg(ConfigOptions, MpiConfg)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmpFile1
                input_forcings.file_in1 = tmpFile1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The NAM nest window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmpFile1
                input_forcings.file_in2 = tmpFile2
        input_forcings.regridComplete = False
    errMod.check_program_status(ConfigOptions, MpiConfg)

    # Ensure we have the necessary new file
    if MpiConfg.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                ConfigOptions.errMsg = "Expected input NAM Nest file: " + input_forcings.file_in2 + " not found."
                errMod.log_critical(ConfigOptions, MpiConfg)
            else:
                ConfigOptions.statusMsg = "Expected input NAM Nest file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                errMod.log_msg(ConfigOptions, MpiConfg)
                if input_forcings.regridded_forcings2 is not None:
                #if np.any(input_forcings.regridded_forcings2):
                    input_forcings.regridded_forcings2[:, :, :] = ConfigOptions.globalNdv

    errMod.check_program_status(ConfigOptions, MpiConfg)

def find_cfsv2_neighbors(input_forcings,ConfigOptions,dCurrent,MpiConfg):
    """
    Function that will calculate the neighboring CFSv2 global GRIB2 files for a given
    forcing output timestep.
    :param input_forcings:
    :param ConfigOptions:
    :param dCurrent:
    :param MpiConfg:
    :return:
    """
    if MpiConfg.rank == 0:
        print("PROCESSING CFSv2")

    ensStr = str(ConfigOptions.cfsv2EnsMember)
    ensStr = ensStr.zfill(2)
    cfsOutHorizons = [6480] # Forecast cycles go out 9 months.
    cfsOutFreq = {
        6480: 360
    }

    # If the user has specified a forcing horizon that is greater than what
    # is available here, return an error.
    if (input_forcings.userFcstHorizon + input_forcings.userCycleOffset) / 60.0 > max(cfsOutHorizons):
        ConfigOptions.errMsg = "User has specified a CFSv2 forecast horizon " \
                               "that is greater than maximum allowed hours of: " \
                               + str(max(cfsOutHorizons))
        print(ConfigOptions.errMsg)
        raise Exception

    if MpiConfg.rank == 0:
        print('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')

    # First find the current GFS forecast cycle that we are using.
    currentCfsCycle = ConfigOptions.current_fcst_cycle - \
                      datetime.timedelta(seconds=
                                         (input_forcings.userCycleOffset) * 60.0)
    if MpiConfg.rank == 0:
        print("CURRENT CFSv2 CYCLE BEING USED = " + currentCfsCycle.strftime('%Y-%m-%d %H'))

    # Calculate the current forecast hour within this CFSv2 cycle.
    dtTmp = dCurrent - currentCfsCycle
    currentCfsHour = int(dtTmp.days * 24) + int(dtTmp.seconds / 3600.0)
    if MpiConfg.rank == 0:
        print("Current CFSv2 Forecast Hour = " + str(currentCfsHour))

    # Calculate the CFS output frequency based on our current CFS forecast hour.
    for horizonTmp in cfsOutHorizons:
        if currentCfsHour <= horizonTmp:
            currentCfsFreq = cfsOutFreq[horizonTmp]
            input_forcings.outFreq = cfsOutFreq[horizonTmp]
            break
    if MpiConfg.rank == 0:
        print("Current CFSv2 output frequency = " + str(currentCfsFreq))

    # Calculate the previous file to process.
    minSinceLastOutput = (currentCfsHour * 60) % currentCfsFreq
    if MpiConfg.rank == 0:
        print(currentCfsHour)
        print(currentCfsFreq)
        print(minSinceLastOutput)
    if minSinceLastOutput == 0:
        minSinceLastOutput = currentCfsFreq
        # currentCfsHour = currentCfsHour
        # previousCfsHour = currentCfsHour - int(currentCfsFreq/60.0)
    prevCfsDate = dCurrent - \
                  datetime.timedelta(seconds=minSinceLastOutput * 60)
    input_forcings.fcst_date1 = prevCfsDate
    if MpiConfg.rank == 0:
        print(prevCfsDate)
    if minSinceLastOutput == currentCfsFreq:
        minUntilNextOutput = 0
    else:
        minUntilNextOutput = currentCfsFreq - minSinceLastOutput
    nextCfsDate = dCurrent + datetime.timedelta(seconds=minUntilNextOutput * 60)
    input_forcings.fcst_date2 = nextCfsDate
    if MpiConfg.rank == 0:
        print(nextCfsDate)

    # Calculate the output forecast hours needed based on the prev/next dates.
    dtTmp = nextCfsDate - currentCfsCycle
    if MpiConfg.rank == 0:
        print(currentCfsCycle)
    nextCfsForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    if MpiConfg.rank == 0:
        print(nextCfsForecastHour)
    input_forcings.fcst_hour2 = nextCfsForecastHour
    dtTmp = prevCfsDate - currentCfsCycle
    prevCfsForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    if MpiConfg.rank == 0:
        print(prevCfsForecastHour)
    input_forcings.fcst_hour1 = prevCfsForecastHour
    # If we are on the first CFS forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prevCfsForecastHour == 0:
        prevCfsDate = nextCfsDate

    # Calculate expected file paths.
    tmpFile1 = input_forcings.inDir + "/cfs." + \
        currentCfsCycle.strftime('%Y%m%d') + "/" + \
        currentCfsCycle.strftime('%H') + "/" + \
        "6hrly_grib_" + ensStr + "/flxf" + \
        prevCfsDate.strftime('%Y%m%d%H') + "." + \
        ensStr + "." + currentCfsCycle.strftime('%Y%m%d%H') + \
        ".grb2"
    if MpiConfg.rank == 0:
        print(tmpFile1)
    tmpFile2 = input_forcings.inDir + "/cfs." + \
               currentCfsCycle.strftime('%Y%m%d') + "/" + \
               currentCfsCycle.strftime('%H') + "/" + \
               "6hrly_grib_" + ensStr + "/flxf" + \
               nextCfsDate.strftime('%Y%m%d%H') + "." + \
               ensStr + "." + currentCfsCycle.strftime('%Y%m%d%H') + \
               ".grb2"
    if MpiConfg.rank == 0:
        print(tmpFile2)
        print('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmpFile1 or input_forcings.file_in2 != tmpFile2:
        if ConfigOptions.current_output_step == 1:
            print('We are on the first output timestep.')
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmpFile1
            input_forcings.file_in2 = tmpFile2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
            #if not np.any(input_forcings.regridded_forcings1):
                if MpiConfg.rank == 0:
                    ConfigOptions.statusMsg = "Restarting forecast cyle. Will regrid previous: " + \
                                              input_forcings.productName
                    errMod.log_msg(ConfigOptions, MpiConfg)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmpFile1
                input_forcings.file_in1 = tmpFile1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The CFS window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmpFile1
                input_forcings.file_in2 = tmpFile2
                if ConfigOptions.runCfsNldasBiasCorrect:
                    # Reset our global CFSv2 grids.
                    input_forcings.coarse_input_forcings1[:, :, :] = input_forcings.coarse_input_forcings2[:, :, :]
        input_forcings.regridComplete = False
    errMod.check_program_status(ConfigOptions, MpiConfg)

    # Ensure we have the necessary new file
    if MpiConfg.rank == 0:
        if not os.path.isfile(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                ConfigOptions.errMsg = "Expected input CFSv2 file: " + input_forcings.file_in2 + " not found."
                errMod.log_critical(ConfigOptions, MpiConfg)
            else:
                ConfigOptions.statusMsg = "Expected input CFSv2 file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                errMod.log_msg(ConfigOptions, MpiConfg)
                if input_forcings.regridded_forcings2 is not None:
                #if np.any(input_forcings.regridded_forcings2):
                    input_forcings.regridded_forcings2[:, :, :] = ConfigOptions.globalNdv

    errMod.check_program_status(ConfigOptions, MpiConfg)

def find_custom_hourly_neighbors(input_forcings,ConfigOptions,dCurrent,MpiConfg):
    """
    Function to calculate the previous and after hourly custom NetCDF files for use in processing.
    :param input_forcings:
    :param ConfigOptions:
    :param dCurrent:
    :param MpiConfg:
    :return:
    """
    if MpiConfg.rank == 0:
        if input_forcings.keyValue == 5:
            print("PROCESSING custom hourly NetCDF files.")

    # Normally, we do a check to make sure the input horizons chosen by the user are not
    # greater than an expeted value. However, since these are custom input NetCDF files,
    # we are foregoing that check.
    currentCustomCycle = ConfigOptions.current_fcst_cycle - \
                         datetime.timedelta(seconds=
                                            (input_forcings.userCycleOffset) * 60.0)

    if MpiConfg.rank == 0:
        print("CURRENT CUSTOM CYCLE BEING USED = " + currentCustomCycle.strftime('%Y-%m-%d %H'))

    # Calculate the current forecast hour within this cycle.
    dtTmp = dCurrent - currentCustomCycle

    currentCustomHour = int(dtTmp.days*24) + math.floor(dtTmp.seconds/3600.0)
    currentCustomMin = math.floor((dtTmp.seconds%3600.0)/60.0)

    if MpiConfg.rank == 0:
        print("Current CUSTOM Forecast Hour = " + str(currentCustomHour))
        print("Current CUSTOM Forecast Minute = " + str(currentCustomMin))

    # Calculate the previous file to process.
    minSinceLastOutput = (currentCustomHour * 60) % 60
    if MpiConfg.rank == 0:
        print(currentCustomHour)
        print(minSinceLastOutput)
    if minSinceLastOutput == 0:
        minSinceLastOutput = 60
    prevCustomDate = dCurrent - datetime.timedelta(seconds=minSinceLastOutput * 60)
    input_forcings.fcst_date1 = prevCustomDate
    if MpiConfg.rank == 0:
        print(prevCustomDate)
    if minSinceLastOutput == 60:
        minUntilNextOutput = 0
    else:
        minUntilNextOutput = 60 - minSinceLastOutput
    nextCustomDate = dCurrent + datetime.timedelta(seconds=minUntilNextOutput * 60)
    input_forcings.fcst_date2 = nextCustomDate
    if MpiConfg.rank == 0:
        print(nextCustomDate)

    # Calculate the output forecast hours needed based on the prev/next dates.
    dtTmp = nextCustomDate - currentCustomCycle
    if MpiConfg.rank == 0:
        print(currentCustomCycle)
    nextCustomForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    if MpiConfg.rank == 0:
        print(nextCustomForecastHour)
    input_forcings.fcst_hour2 = nextCustomForecastHour
    dtTmp = prevCustomDate - currentCustomCycle
    prevCustomForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    if MpiConfg.rank == 0:
        print(prevCustomForecastHour)
    input_forcings.fcst_hour1 = prevCustomForecastHour
    # If we are on the first forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prevCustomForecastHour == 0:
        prevCustomForecastHour = 1

    # Calculate expected file paths.
    tmpFile1 = input_forcings.inDir + "/custom_hourly." + \
                currentCustomCycle.strftime('%Y%m%d%H') + '.f' + \
                str(prevCustomForecastHour).zfill(2) + '.nc'
    if MpiConfg.rank == 0:
        print(tmpFile1)
    tmpFile2 = input_forcings.inDir + '/custom_hourly.' + \
                currentCustomCycle.strftime('%Y%m%d%H') + '.f' + \
                str(nextCustomForecastHour).zfill(2) + '.nc'
    if MpiConfg.rank == 0:
        print(tmpFile2)
        # Check to see if files are already set. If not, then reset, grids and
        # regridding objects to communicate things need to be re-established.
        if input_forcings.file_in1 != tmpFile1 or input_forcings.file_in2 != tmpFile2:
            if ConfigOptions.current_output_step == 1:
                print('We are on the first output timestep.')
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in1 = tmpFile1
                input_forcings.file_in2 = tmpFile2
            else:
                # Check to see if we are restarting from a previously failed instance. In this case,
                # We are not on the first timestep, but no previous forcings have been processed.
                # We need to process the previous input timestep for temporal interpolation purposes.
                if input_forcings.regridded_forcings1 is None:
                #if not np.any(input_forcings.regridded_forcings1):
                    if MpiConfg.rank == 0:
                        ConfigOptions.statusMsg = "Restarting forecast cyle. Will regrid previous: " + \
                                                  input_forcings.productName
                        errMod.log_msg(ConfigOptions, MpiConfg)
                    input_forcings.rstFlag = 1
                    input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                    input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                    input_forcings.file_in2 = tmpFile1
                    input_forcings.file_in1 = tmpFile1
                    input_forcings.fcst_date2 = input_forcings.fcst_date1
                    input_forcings.fcst_hour2 = input_forcings.fcst_hour1
                else:
                    # The custom window has shifted. Reset fields 2 to
                    # be fields 1.
                    input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                    input_forcings.file_in1 = tmpFile1
                    input_forcings.file_in2 = tmpFile2
            input_forcings.regridComplete = False
        errMod.check_program_status(ConfigOptions, MpiConfg)

        # Ensure we have the necessary new file
        if MpiConfg.rank == 0:
            if not os.path.isfile(input_forcings.file_in2):
                if input_forcings.enforce == 1:
                    ConfigOptions.errMsg = "Expected input Custom file: " + input_forcings.file_in2 + " not found."
                    errMod.log_critical(ConfigOptions, MpiConfg)
                else:
                    ConfigOptions.statusMsg = "Expected input Custom file: " + input_forcings.file_in2 + " not found. " \
                                                                                                        "Will not use in " \
                                                                                                        "final layering."
                    errMod.log_msg(ConfigOptions, MpiConfg)
                    if input_forcings.regridded_forcings2 is not None:
                    #if np.any(input_forcings.regridded_forcings2):
                        input_forcings.regridded_forcings2[:, :, :] = ConfigOptions.globalNdv

        errMod.check_program_status(ConfigOptions, MpiConfg)

def find_hourly_MRMS_radar_neighbors(supplemental_precip,ConfigOptions,dCurrent,MpiConfg):
    """
    Function to calculate the previous and next MRMS radar-only QPE files. This
    will also calculate the neighboring radar quality index (RQI) files as well.
    :param supplemental_precip:
    :param ConfigOptions:
    :param dCurrent:
    :param MpiConfg:
    :return:
    """
    # First we need to find the nearest previous and next hour, which is
    # the previous/next MRMS files we will be using.
    currentYr = dCurrent.year
    currentMo = dCurrent.month
    currentDay = dCurrent.day
    currentHr = dCurrent.hour
    currentMin = dCurrent.minute

    # Set the input file frequency to be hourly.
    supplemental_precip.input_frequency = 60.0

    prevDate1 = datetime.datetime(currentYr,currentMo,currentDay,currentHr)
    dtTmp = dCurrent - prevDate1
    if dtTmp.total_seconds() == 0:
        # We are on the hour, we can set this date to the be the "next" date.
        nextMrmsDate = dCurrent
        prevMrmsDate = dCurrent - datetime.timedelta(seconds=3600.0)
    else:
        # We are between two MRMS hours.
        prevMrmsDate = prevDate1
        nextMrmsDate = prevMrmsDate + datetime.timedelta(seconds=3600.0)

    supplemental_precip.pcp_date1 = prevMrmsDate
    supplemental_precip.pcp_date2 = nextMrmsDate

    # Calculate expected file paths.
    if supplemental_precip.keyValue == 1:
        tmpFile1 = supplemental_precip.inDir + "/RadarOnly_QPE_01H/" + \
                   "MRMS_RadarOnly_QPE_01H_00.00_" + \
                   supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
                   "-" + supplemental_precip.pcp_date1.strftime('%H') + \
                   "0000.grib2.gz"
        tmpFile2 = supplemental_precip.inDir + "/RadarOnly_QPE_01H/" + \
                   "MRMS_RadarOnly_QPE_01H_00.00_" + \
                   supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
                   "-" + supplemental_precip.pcp_date2.strftime('%H') + \
                   "0000.grib2.gz"
    if supplemental_precip.keyValue == 2:
        tmpFile1 = supplemental_precip.inDir + "/GaugeCorr_QPE_01H/" + \
                   "MRMS_GaugeCorr_QPE_01H_00.00_" + \
                   supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
                   "-" + supplemental_precip.pcp_date1.strftime('%H') + \
                   "0000.grib2.gz"
        tmpFile2 = supplemental_precip.inDir + "/GaugeCorr_QPE_01H/" + \
                   "MRMS_GaugeCorr_QPE_01H_00.00_" + \
                   supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
                   "-" + supplemental_precip.pcp_date2.strftime('%H') + \
                   "0000.grib2.gz"

    # Compose the RQI paths.
    tmpRqiFile1 = supplemental_precip.inDir + "/RadarQualityIndex/" + \
                  "MRMS_RadarQualityIndex_00.00_" + \
                  supplemental_precip.pcp_date1.strftime('%Y%m%d') + \
                  "-" + supplemental_precip.pcp_date1.strftime('%H') + \
                  "0000.grib2.gz"
    tmpRqiFile2 = supplemental_precip.inDir + "/RadarQualityIndex/" + \
                  "MRMS_RadarQualityIndex_00.00_" + \
                  supplemental_precip.pcp_date2.strftime('%Y%m%d') + \
                  "-" + supplemental_precip.pcp_date2.strftime('%H') + \
                  "0000.grib2.gz"
    if MpiConfg.rank == 0:
        ConfigOptions.statusMsg = "Previous MRMS supplemental " \
                                  "file: " + tmpFile1
        errMod.log_msg(ConfigOptions,MpiConfg)
        ConfigOptions.statusMsg = "Next MRMS supplemental " \
                                  "file: " + tmpFile2
        errMod.log_msg(ConfigOptions, MpiConfg)
        ConfigOptions.statusMsg = "Previous MRMS RQI supplemental " \
                                  "file: " + tmpRqiFile1
        errMod.log_msg(ConfigOptions, MpiConfg)
        ConfigOptions.statusMsg = "Next MRMS RQI supplemental " \
                                  "file: " + tmpRqiFile2
        errMod.log_msg(ConfigOptions, MpiConfg)
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if supplemental_precip.file_in1 != tmpFile1 or supplemental_precip.file_in2 != tmpFile2:
        if ConfigOptions.current_output_step == 1:
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
            #supplemental_precip.regridded_precip1[:,:] = supplemental_precip.regridded_precip2[:,:]
            #supplemental_precip.regridded_rqi1[:, :] = supplemental_precip.regridded_rqi2[:, :]
        supplemental_precip.file_in1 = tmpFile1
        supplemental_precip.file_in2 = tmpFile2
        supplemental_precip.rqi_file_in1 = tmpRqiFile1
        supplemental_precip.rqi_file_in2 = tmpRqiFile2
        supplemental_precip.regridComplete = False

    # If either file does not exist, set to None. This will instruct downstream regridding steps to
    # set the regridded states to the global NDV. That ensures no supplemental precipitation will be
    # added to the final output grids.

    #if not os.path.isfile(tmpFile1) or not os.path.isfile(tmpFile2):
    #    if MpiConfg.rank == 0:
    #        ConfigOptions.statusMsg = "MRMS files are missing. Will not process " \
    #                                  "supplemental precipitation"
    #        errMod.log_warning(ConfigOptions,MpiConfg)
    #    supplemental_precip.file_in2 = None
    #    supplemental_precip.file_in1 = None

    #errMod.check_program_status(ConfigOptions, MpiConfg)

    # Ensure we have the necessary new file
    if MpiConfg.rank == 0:
        if not os.path.isfile(supplemental_precip.file_in2):
            if supplemental_precip.enforce == 1:
                ConfigOptions.errMsg = "Expected input MRMS file: " + supplemental_precip.file_in2 + " not found."
                errMod.log_critical(ConfigOptions, MpiConfg)
            else:
                ConfigOptions.statusMsg = "Expected input MRMS file: " + supplemental_precip.file_in2 + \
                                          " not found. " + "Will not use in final layering."
                errMod.log_msg(ConfigOptions, MpiConfg)
                if supplemental_precip.regridded_precip2 is not None:
                #if np.any(supplemental_precip.regridded_precip2):
                    supplemental_precip.regridded_precip2[:, :, :] = ConfigOptions.globalNdv

    errMod.check_program_status(ConfigOptions, MpiConfg)

def find_hourly_WRF_ARW_HiRes_PCP_neighbors(supplemental_precip,ConfigOptions,dCurrent,MpiConfg):
    """
    Function to calculate the previous and next WRF-ARW Hi-Res Window files for the previous
    and next output timestep. These files will be used for supplemental precipitation.
    :param supplemental_precip:
    :param ConfigOptions:
    :param dCurrent:
    :param MpiConfg:
    :return:
    """
    if dCurrent >= datetime.datetime(2008, 10, 1):
        defaultHorizon = 48 # Current forecasts go out to 48 hours.

    # First find the current ARW forecast cycle that we are using.
    currentARWCycle = ConfigOptions.current_fcst_cycle

    if MpiConfg.rank == 0:
        ConfigOptions.statusMsg = "Processing WRF-ARW supplemental precipitation from " \
                                  "forecast cycle: " + currentARWCycle.strftime('%Y-%m-%d %H')
        errMod.log_msg(ConfigOptions,MpiConfg)
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Apply some logic here to account for the ARW weirdness associated with the
    # odd forecast cycles.
    if supplemental_precip.keyValue == 3:
        # Data only available for 00/12 UTC cycles.
        if currentARWCycle.hour != 0 and currentARWCycle.hour != 12:
            if MpiConfg.rank == 0:
                ConfigOptions.statusMsg = "No Hawaii WRF-ARW found for this cycle. " \
                                          "No supplemental ARW " \
                                          "precipitation will be used."
                errMod.log_msg(ConfigOptions,MpiConfg)
            supplemental_precip.file_in1 = None
            supplemental_precip.file_in2 = None
            return
        fcstHorizon = 48
    if supplemental_precip.keyValue == 4:
        # Data only available for 06/18 UTC cycles.
        if currentARWCycle.hour != 6 and currentARWCycle.hour != 18:
            if MpiConfg.rank == 0:
                ConfigOptions.statusMsg = "No Puerto Rico WRF-ARW found for this cycle. " \
                                          "No supplemental ARW precipitation will be used."
                errMod.log_msg(ConfigOptions,MpiConfg)
            supplemental_precip.file_in1 = None
            supplemental_precip.file_in2 = None
            return
        fcstHorizon = 48

    if MpiConfg.rank == 0:
        ConfigOptions.statusMsg = "Current ARW forecast cycle being used: " + \
                                  currentARWCycle.strftime('%Y-%m-%d %H')
        errMod.log_msg(ConfigOptions,MpiConfg)

    # Calculate the current forecast hour within this ARW cycle.
    dtTmp = dCurrent - currentARWCycle
    currentARWHour = int(dtTmp.days * 24) + int(dtTmp.seconds / 3600.0)

    if currentARWHour > fcstHorizon:
        if MpiConfg.rank == 0:
            ConfigOptions.statusMsg = "Current ARW forecast hour greater than max allowed " \
                                      "of: " + str(fcstHorizon)
            errMod.log_msg(ConfigOptions,MpiConfg)
            supplemental_precip.file_in2 = None
            supplemental_precip.file_in1 = None
            return
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Calculate the previous file to process.
    minSinceLastOutput = (currentARWHour * 60) % 60
    if MpiConfg.rank == 0:
        print(currentARWHour)
        print(minSinceLastOutput)
    if minSinceLastOutput == 0:
        minSinceLastOutput = 60
    prevARWDate = dCurrent - datetime.timedelta(seconds=minSinceLastOutput * 60)
    supplemental_precip.pcp_date1 = prevARWDate
    if MpiConfg.rank == 0:
        print(prevARWDate)
    if minSinceLastOutput == 60:
        minUntilNextOutput = 0
    else:
        minUntilNextOutput = 60 - minSinceLastOutput
    nextARWDate = dCurrent + datetime.timedelta(seconds=minUntilNextOutput * 60)
    supplemental_precip.pcp_date2 = nextARWDate
    if MpiConfg.rank == 0:
        print(nextARWDate)

    # Calculate the output forecast hours needed based on the prev/next dates.
    dtTmp = nextARWDate - currentARWCycle
    nextARWForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    if nextARWForecastHour > fcstHorizon:
        if MpiConfg.rank == 0:
            ConfigOptions.statusMsg = "Next ARW forecast hour greater than max allowed " \
                                      "of: " + str(fcstHorizon)
            errMod.log_msg(ConfigOptions, MpiConfg)
            supplemental_precip.file_in2 = None
            supplemental_precip.file_in1 = None
            return
    errMod.check_program_status(ConfigOptions,MpiConfg)

    supplemental_precip.fcst_hour2 = nextARWForecastHour
    dtTmp = prevARWDate - currentARWCycle
    prevARWForecastHour = int(dtTmp.days * 24.0) + int(dtTmp.seconds / 3600.0)
    supplemental_precip.fcst_hour1 = prevARWForecastHour

    # If we are on the first ARW forecast hour (1), and we have calculated the previous forecast
    # hour to be 0, simply set both hours to be 1. Hour 0 will not produce the fields we need, and
    # no interpolation is required.
    if prevARWForecastHour == 0:
        prevARWForecastHour = 1

    # Calculate expected file paths.
    if supplemental_precip.keyValue == 3:
        tmpFile1 = supplemental_precip.inDir + '/hiresw.' + \
                   currentARWCycle.strftime('%Y%m%d') + '/hiresw.t' + \
                   currentARWCycle.strftime('%H') + 'z.arw_2p5km.f' + \
                   str(prevARWForecastHour).zfill(2) + '.hi.grib2'
        tmpFile2 = supplemental_precip.inDir + '/hiresw.' + \
                   currentARWCycle.strftime('%Y%m%d') + '/hiresw.t' + \
                   currentARWCycle.strftime('%H') + 'z.arw_2p5km.f' + \
                   str(nextARWForecastHour).zfill(2) + '.hi.grib2'
    if supplemental_precip.keyValue == 4:
        tmpFile1 = supplemental_precip.inDir + '/hiresw.' + \
                   currentARWCycle.strftime('%Y%m%d') + '/hiresw.t' + \
                   currentARWCycle.strftime('%H') + 'z.arw_2p5km.f' + \
                   str(prevARWForecastHour).zfill(2) + '.pr.grib2'
        tmpFile2 = supplemental_precip.inDir + '/hiresw.' + \
                   currentARWCycle.strftime('%Y%m%d') + '/hiresw.t' + \
                   currentARWCycle.strftime('%H') + 'z.arw_2p5km.f' + \
                   str(nextARWForecastHour).zfill(2) + '.pr.grib2'
    errMod.check_program_status(ConfigOptions,MpiConfg)
    if MpiConfg.rank == 0:
        ConfigOptions.statusMsg = "Previous ARW supplemental precipitation file: " + tmpFile1
        errMod.log_msg(ConfigOptions, MpiConfg)
        ConfigOptions.statusMsg = "Next ARW supplemental precipitation file: " + tmpFile2
        errMod.log_msg(ConfigOptions, MpiConfg)
    errMod.check_program_status(ConfigOptions,MpiConfg)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if supplemental_precip.file_in1 != tmpFile1 or supplemental_precip.file_in2 != tmpFile2:
        if ConfigOptions.current_output_step == 1:
            supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
            supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
        else:
            # The forecast window has shifted. Reset fields 2 to
            # be fields 1.
            supplemental_precip.regridded_precip1 = supplemental_precip.regridded_precip1
            supplemental_precip.regridded_precip2 = supplemental_precip.regridded_precip2
        supplemental_precip.file_in1 = tmpFile1
        supplemental_precip.file_in2 = tmpFile2
        supplemental_precip.regridComplete = False

    # If either file does not exist, set to None. This will instruct downstream regridding steps to
    # set the regridded states to the global NDV. That ensures no supplemental precipitation will be
    # added to the final output grids.
    #if not os.path.isfile(tmpFile1) or not os.path.isfile(tmpFile2):
    #    if MpiConfg.rank == 0:
    #        ConfigOptions.statusMsg = "Found missing ARW file. Will not process supplemental precipitation for " \
    #                                  "this time step."
    #        errMod.log_msg(ConfigOptions, MpiConfg)
    #    supplemental_precip.file_in2 = None
    #    supplemental_precip.file_in1 = None

    # Ensure we have the necessary new file
    if MpiConfg.rank == 0:
        if not os.path.isfile(supplemental_precip.file_in2):
            if supplemental_precip.enforce == 1:
                ConfigOptions.errMsg = "Expected input ARW file: " + supplemental_precip.file_in2 + " not found."
                errMod.log_critical(ConfigOptions, MpiConfg)
            else:
                ConfigOptions.statusMsg = "Expected input ARW file: " + supplemental_precip.file_in2 + \
                                          " not found. " + "Will not use in final layering."
                errMod.log_msg(ConfigOptions, MpiConfg)
                if supplemental_precip.regridded_precip2 is not None:
                #if np.any(supplemental_precip.regridded_precip2):
                    supplemental_precip.regridded_precip2[:, :, :] = ConfigOptions.globalNdv

    errMod.check_program_status(ConfigOptions, MpiConfg)

