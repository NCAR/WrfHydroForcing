import sys
import logging

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
    sys.exit(-1)

def init_log(ConfigOptions):
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
        err_out_screen(ConfigOptions.errMsg)
    try:
        formatter = logging.Formatter('[%(asctime)s]: %(levelname)s '
                                      '- %(message)s', '%m/%d %H:%M:%S')
    except:
        ConfigOptions.errMsg = "Unable to establish formatting for logger."
        err_out_screen(ConfigOptions.errMsg)
    try:
        ConfigOptions.logHandle = logging.FileHandler(ConfigOptions.logFile,mode='a')
    except:
        ConfigOptions.errMsg = "Unable to create log file handle for: " + \
            ConfigOptions.logFile
        err_out_screen(ConfigOptions.errMsg)
    try:
        ConfigOptions.logHandle.setFormatter(formatter)
    except:
        ConfigOptions.errMsg = "Unable to set formatting for: " + \
            ConfigOptions.logFile
        err_out_screen(ConfigOptions.errMsg)
    try:
        logObj.addHandler(ConfigOptions.logHandle)
    except:
        ConfigOptions.errMsg = "ERROR: Unable to add log handler for: " + \
            ConfigOptions.logFile
        err_out_screen(ConfigOptions.errMsg)

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
    sys.exit(1)

def log_error(ConfigOptions):
    """
    Function for logging an error message without exiting without a
    non-zero exit status.
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
        logObj.setLevel(logging.CRITICAL)
    except:
        ConfigOptions.errMsg = "Unable to set CRITICAL logger level for: " + \
            ConfigOptions.logFile
        raise Exception()
    try:
        logObj.critical(ConfigOptions.errMsg)
    except:
        ConfigOptions.errMsg = "Unable to write critical message to: " + \
            ConfigOptions.logFile
        raise Exception()

def log_warning(ConfigOptions):
    """
    Function to log warning messages to the log file.
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
        logObj.setLevel(logging.WARNING)
    except:
        ConfigOptions.errMsg = "Unable to set WARNING logger level for: " + \
            ConfigOptions.logFile
        raise Exception()
    try:
        logObj.warning(ConfigOptions.statusMsg)
    except:
        ConfigOptions.errMsg = "Unable to write warning message to: " + \
            ConfigOptions.logFile
        raise Exception()

def log_msg(ConfigOptions):
    """
    Function to log INFO messages to a specified log file.
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
        logObj.setLevel(logging.INFO)
    except:
        ConfigOptions.errMsg = "Unable to set INFO logger level for: " + \
            ConfigOptions.logFile
        raise Exception()
    try:
        logObj.info(ConfigOptions.statusMsg)
    except:
        ConfigOptions.errMsg = "Unable to write info message to: " + \
            ConfigOptions.logFile
        raise Exception()

def close_log(ConfigOptions):
    """
    Function for closing a log file.
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
        logObj.removeHandler(ConfigOptions.logHandle)
    except:
        ConfigOptions.errMsg = "Unable to remove logging file handle for: " + \
            ConfigOptions.logFile
        raise Exception()
    try:
        ConfigOptions.logHandle.close()
    except:
        ConfigOptions.errMsg = "Unable to close logging file: " + \
            ConfigOptions.logFile
        raise Exception()
    try:
        ConfigOptions.logHandle = None
    except:
        ConfigOptions.errMsg = "Unable to reset logging handle object to " \
                               "None after closing log file: " + \
            ConfigOptions.logFile
        raise Exception()