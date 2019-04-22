import sys
import logging
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
    err_msg_out = 'ERROR: ' + err_msg
    print(err_msg_out)
    MpiConfig.comm.Abort()
    sys.exit(1)

def check_program_status(ConfigOptions,MpiConfig):
    """
    Generic function to check the err statuses for each processor in the program.
    If any flags come back, gracefully exit the program.
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Collect values from each processor.
    data = MpiConfig.comm.gather(ConfigOptions.errFlag, root=0)
    if MpiConfig.rank == 0:
        for i in range(MpiConfig.size):
            print('RANK NUMBER: ' + str(i) + ' - STATUS: ' + str(data[i]))
            if data[i] != 0:
                print('FOUND ERROR ON RANK: ' + str(i))
                MpiConfig.comm.Abort()
                sys.exit(1)
    else:
        assert data is None

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
    print(ConfigOptions.errMsg)
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
        logObj.error(ConfigOptions.errMsg)
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
        logObj.critical(ConfigOptions.errMsg)
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
        logObj.warning(ConfigOptions.statusMsg)
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
        logObj.info(ConfigOptions.statusMsg)
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