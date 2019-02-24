# This is a datetime module for handling datetime
# calculations in the forcing engine.
from core import errMod
import datetime
import math

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
    ConfigOptions.nFcsts = int(nFcstSteps)
    ConfigOptions.e_date_proc = ConfigOptions.b_date_proc + datetime.timedelta(seconds=nFcstSteps*ConfigOptions.fcst_freq*60)

def find_gfs_neighbors(input_forcings,ConfigOptions,dCurrent):
    """
    Function to calcualte the previous and after GFS cycles based on the current timestep.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    if input_forcings.keyValue == 1:
        print("She is processing NLDAS GLURL.......")
    if input_forcings.keyValue == 3:
        print("We are processing regular GFS data....")
    if input_forcings.keyValue == 9:
        print("We are processing 0.25 degree GFS data.....")
