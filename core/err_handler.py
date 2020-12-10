import logging
import os
import sys
import traceback

import numpy as np
from mpi4py import MPI


def err_out_screen(err_msg):
    """
    Generic routine to exit the program gracefully. This specific error function does not log
    error messages to the log file, but simply prints them out to the screen. This is function
    is designed specifically for early in the program execution where a log file hasn't been
    established yet.
    Logan Karsten - National Center for Atmospheric Research, karsten@ucar.edu
    """

    err_msg_out = 'ERROR: ' + err_msg
    print(err_msg_out)
    sys.exit(1)

def err_out_screen_para(err_msg,MpiConfig):
    """
    Generic function for printing an error message to the screen and aborting MPI.
    This should only be called if logging cannot occur and an abrupt end the program
    is neded.
    :param err_msg:
    :param MpiConfig:
    :return:
    """
    err_msg_out = 'ERROR: RANK - ' + str(MpiConfig.rank) + ' : ' + err_msg
    print(err_msg_out)
    # MpiConfig.comm.Abort()
    sys.exit(1)

def check_program_status(ConfigOptions, MpiConfig):
    """
    Generic function to check the err statuses for each processor in the program.
    If any flags come back, gracefully exit the program.
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Sync up processors to ensure everyone is on the same page.
    #MpiConfig.comm.barrier()

    # Collect values from each processor.
    # data = MpiConfig.comm.gather(ConfigOptions.errFlag, root=0)
    # if MpiConfig.rank == 0:
    #     for i in range(MpiConfig.size):
    #         if data[i] != 0:
    #             MpiConfig.comm.Abort()
    #             sys.exit(1)
    # else:
    #     assert data is None

    # Reduce version:
    any_error = MpiConfig.comm.reduce(ConfigOptions.errFlag)
    if MpiConfig.rank == 0:
        if ConfigOptions.errFlag:
            # print("any_error: ", any_error, type(any_error), flush=True)
            stack = traceback.format_stack()[:-1]
            [print(frame, flush=True, end='') for frame in stack]
            MpiConfig.comm.Abort()
            sys.exit(1)

    # Sync up processors.
    # MpiConfig.comm.barrier()

def init_log(ConfigOptions,MpiConfig):
    """
    Function for initializing log file for individual forecast cycles. Each
    log file is unique to the instant the program was initialized.
    :param ConfigOptions:
    :return:
    """
    try:
        logObj = logging.getLogger('logForcing')
    except:
        ConfigOptions.errMsg = "Unable to create logging object " \
                               "for: " + ConfigOptions.logFile
        err_out_screen_para(ConfigOptions.errMsg,MpiConfig)
    try:
        formatter = logging.Formatter('[%(asctime)s]: %(levelname)s '
                                      '- %(message)s', '%m/%d %H:%M:%S')
    except:
        ConfigOptions.errMsg = "Unable to establish formatting for logger."
        err_out_screen_para(ConfigOptions.errMsg,MpiConfig)
    try:
        ConfigOptions.logHandle = logging.FileHandler(ConfigOptions.logFile,mode='a')
    except:
        ConfigOptions.errMsg = "Unable to create log file handle for: " + \
            ConfigOptions.logFile
        err_out_screen_para(ConfigOptions.errMsg,MpiConfig)
    try:
        ConfigOptions.logHandle.setFormatter(formatter)
    except:
        ConfigOptions.errMsg = "Unable to set formatting for: " + \
            ConfigOptions.logFile
        err_out_screen_para(ConfigOptions.errMsg,MpiConfig)
    try:
        logObj.addHandler(ConfigOptions.logHandle)
    except:
        ConfigOptions.errMsg = "ERROR: Unable to add log handler for: " + \
            ConfigOptions.logFile
        err_out_screen_para(ConfigOptions.errMsg,MpiConfig)

def err_out(ConfigOptions):
    """
    Function to error out after an error message has been logged for a
    forecast cycle. We will exit with a non-zero exit status.
    :param ConfigOptions:
    :return:
    """
    try:
        logObj = logging.getLogger('logForcing')
    except:
        ConfigOptions.errMsg = "Unable to obtain a logger object for: " + \
            ConfigOptions.logFile
        raise Exception()
    try:
        logObj.setLevel(logging.ERROR)
    except:
        ConfigOptions.errMsg = "Unable to set ERROR logger level for: " + \
            ConfigOptions.logFile
        raise Exception()
    try:
        logObj.error(ConfigOptions.errMsg)
    except:
        ConfigOptions.errMsg = "Unable to write error message to: " + \
            ConfigOptions.logFile
        raise Exception()
    MPI.Finalize()
    sys.exit(1)

def log_error(ConfigOptions,MpiConfig):
    """
    Function to log an error message to the log file.
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    try:
        logObj = logging.getLogger('logForcing')
    except:
        err_out_screen_para(('Unable to obtain logger object on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    try:
        logObj.setLevel(logging.ERROR)
    except:
        err_out_screen_para(('Unable to set ERROR logger level on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    try:
        logObj.error("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.errMsg)
    except:
        err_out_screen_para(('Unable to write ERROR message on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    ConfigOptions.errFlag = 1

def log_critical(ConfigOptions,MpiConfig):
    """
    Function for logging an error message without exiting without a
    non-zero exit status.
    :param ConfigOptions:
    :return:
    """
    try:
        logObj = logging.getLogger('logForcing')
    except:
        err_out_screen_para(('Unable to obtain logger object on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    try:
        logObj.setLevel(logging.CRITICAL)
    except:
        err_out_screen_para(('Unable to set CRITICAL logger level on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    try:
        logObj.critical("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.errMsg)
    except:
        err_out_screen_para(('Unable to write CRITICAL message on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    ConfigOptions.errFlag = 1

def log_warning(ConfigOptions,MpiConfig):
    """
    Function to log warning messages to the log file.
    :param ConfigOptions:
    :return:
    """
    try:
        logObj = logging.getLogger('logForcing')
    except:
        err_out_screen_para(('Unable to obtain logger object on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    try:
        logObj.setLevel(logging.WARNING)
    except:
        err_out_screen_para(('Unable to set WARNING logger level on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    try:
        logObj.warning("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.statusMsg)
    except:
        err_out_screen_para(('Unable to write WARNING message on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)

def log_msg(ConfigOptions,MpiConfig):
    """
    Function to log INFO messages to a specified log file.
    :param ConfigOptions:
    :return:
    """
    try:
        logObj = logging.getLogger('logForcing')
    except:
        err_out_screen_para(('Unable to obtain logger object on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    try:
        logObj.setLevel(logging.INFO)
    except:
        err_out_screen_para(('Unable to set INFO logger level on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)
    try:
        logObj.info("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.statusMsg)
    except:
        err_out_screen_para(('Unable to write INFO message on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile),MpiConfig)

def close_log(ConfigOptions,MpiConfig):
    """
    Function for closing a log file.
    :param ConfigOptions:
    :return:
    """
    try:
        logObj = logging.getLogger('logForcing')
    except:
        err_out_screen_para(('Unable to obtain logger object on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile), MpiConfig)
    try:
        logObj.removeHandler(ConfigOptions.logHandle)
    except:
        err_out_screen_para(('Unable to remove logging file handle on RANK: ' + str(MpiConfig.rank) +
                             ' for log file: ' + ConfigOptions.logFile), MpiConfig)
    try:
        ConfigOptions.logHandle.close()
    except:
        err_out_screen_para(('Unable to close looging file: ' + ConfigOptions.logFile +
                             ' on RANK: ' + str(MpiConfig.rank)),MpiConfig)
    ConfigOptions.logHandle = None

def check_forcing_bounds(ConfigOptions, input_forcings, MpiConfig):
    """
    Function for running a reasonable value check for individual forcing
    variables. This one check type with the other checking for final missing
    values in the final output grid.
    :param ConfigOptions:
    :param input_forcings:
    :param MpiConfig:
    :return:
    """
    # Establish a range of values for each output variable.
    variable_range = {
        'U2D': [0, -500.0, 500.0],
        'V2D': [1, -500.0, 500.0],
        'LWDOWN': [2, -1000.0, 10000.0],
        'RAINRATE': [3, 0.0, 100.0],
        'T2D': [4, 0.0, 400.0],
        'Q2D': [5, -100.0, 100.0],
        'PSFC': [6, 0.0, 2000000.0],
        'SWDOWN': [7, 0.0, 5000.0]
    }
    fvars = ['U2D', 'V2D', 'LWDOWN', 'RAINRATE', 'T2D', 'Q2D', 'PSFC', 'SWDOWN']

    # If the regridded field is None type, return to the main program as this means no forcings
    # were found for this timestep.
    if input_forcings.regridded_forcings2 is None:
        return

    # Loop over all the variables. Check for reasonable ranges. If any values are
    # exceeded, shut the forcing engine down.
    for varTmp in variable_range:
        if fvars.index(varTmp) not in input_forcings.input_map_output:
            continue

        # First check to see if we have any data that is not missing.
        # indCheck = np.where(input_forcings.regridded_forcings2[variable_range[varTmp][0]]
        #                    != ConfigOptions.globalNdv)

        #if len(indCheck[0]) == 0:
        #    ConfigOptions.errMsg = "No valid data found for " + varTmp + " in " + input_forcings.file_in2
        #    log_critical(ConfigOptions, MpiConfig)
        #    indCheck = None
        #    return

        # Check to see if any pixel cells are below the minimum value.
        indCheck = np.where((input_forcings.regridded_forcings2[variable_range[varTmp][0]] != ConfigOptions.globalNdv) &
                            (input_forcings.regridded_forcings2[variable_range[varTmp][0]] < variable_range[varTmp][1]))
        numCells = len(indCheck[0])
        if numCells > 0:
            ConfigOptions.errMsg = "Data below minimum threshold for: " + varTmp + " in " + input_forcings.file_in2 + \
                                   " for " + str(numCells) + " regridded pixel cells."
            log_critical(ConfigOptions, MpiConfig)
            indCheck = None
            return

        # Check to see if any pixel cells are above the maximum value.
        indCheck = np.where((input_forcings.regridded_forcings2[variable_range[varTmp][0]] != ConfigOptions.globalNdv) &
                            (input_forcings.regridded_forcings2[variable_range[varTmp][0]] > variable_range[varTmp][2]))
        numCells = len(indCheck[0])
        if numCells > 0:
            ConfigOptions.errMsg = "Data above maximum threshold for: " + varTmp + " in " + input_forcings.file_in2 + \
                                   " for " + str(numCells) + " regridded pixel cells."
            log_critical(ConfigOptions, MpiConfig)
            indCheck = None
            return

    indCheck = None
    return

def check_supp_pcp_bounds(ConfigOptions, supplemental_precip, MpiConfig):
    """
    Function for running a reasonable value check on supplemental precipitation
    values. This is one check type with the other checking for final missing
    values in the final output grid.
    :param ConfigOptions:
    :param supplemental_precip:
    :param MpiConfig:
    :return:
    """
    # If the regridded field is None type, return to the main program as this means no forcings
    # were found for this timestep.
    if supplemental_precip.regridded_precip2 is None:
        return

    # First check to see if we have any data that is not missing.
    # indCheck = np.where(supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv)

    # if len(indCheck[0]) == 0:
    #     ConfigOptions.errMsg = "No valid supplemental precip found in " + supplemental_precip.file_in2
    #     log_critical(ConfigOptions, MpiConfig)
    #     indCheck = None
    #     return

    # Check to see if any pixel cells are below the minimum value.
    indCheck = np.where((supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv) &
                        (supplemental_precip.regridded_precip2 < 0.0))
    numCells = len(indCheck[0])
    if numCells > 0:
        ConfigOptions.errMsg = "Supplemental precip data below minimum threshold for in " + \
                               supplemental_precip.file_in2 + \
                               " for " + str(numCells) + " regridded pixel cells."
        log_critical(ConfigOptions, MpiConfig)
        indCheck = None
        return

    # Check to see if any pixel cells are above the maximum value.
    indCheck = np.where((supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv) &
                        (supplemental_precip.regridded_precip2 > 100.0))
    numCells = len(indCheck[0])
    if numCells > 0:
        ConfigOptions.errMsg = "Supplemental precip data above maximum threshold for in " + \
                               supplemental_precip.file_in2 + \
                               " for " + str(numCells) + " regridded pixel cells."
        log_critical(ConfigOptions, MpiConfig)
        indCheck = None
        return

    indCheck = None
    return

def check_missing_final(outPath, ConfigOptions, output_grid, var_name, MpiConfig):
    """
    Function that checks the final output grids to ensure no missing values
    are in place. Final output grids cannot contain any missing values per
    WRF-Hydro requirements.
    ;:param outPath:
    :param ConfigOptions:
    :param output_grid:
    :param var_name:
    :param MpiConfig:
    :return:
    """
    # Run NDV check. If ANY values are found, throw an error and shut the
    # forcing engine down.
    indCheck = np.where(output_grid == ConfigOptions.globalNdv)

    if len(indCheck[0]) > 0:
        ConfigOptions.errMsg = "Found " + str(len(indCheck[0])) + " NDV pixel cells in output grid for: " + \
                               var_name
        log_critical(ConfigOptions, MpiConfig)
        indCheck = None
        # If the output file has been created, remove it as it will be empty.
        if os.path.isfile(outPath):
            os.remove(outPath)
        return
    else:
        indCheck = None
        return
