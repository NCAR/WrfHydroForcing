# This is a datetime module for handling datetime
# calculations in the forcing engine.
from core import errmod
import datetime

def calculate_lookback_window(ConfigOptions):
    """
    Calculate the beginning, ending datetime variables
    for a look-back period. Also calculate the processing
    time window delta value.
    :param ConfigOptions: Abstract class holding job information.
    :return: Updated abstract class with updated datetime variables.
    """
    dCurrentUtc = datetime.datetime.utcnow()
    current_year = dCurrentUtc.year
    current_month = dCurrentUtc.month
    current_day = dCurrentUtc.day
    current_hour = dCurrentUtc.hour
    current_minute = dCurrentUtc.minute

    # First calculate the end of the processing window. If the
    # forecast frequency is greater than or equal to an hour,
    # we will set the end of the processing window to be the beginning
    # of the current window.
    if ConfigOptions.fcst_freq >= 60:
        ConfigOptions.e_date_proc = datetime.datetime(current_year,current_month,
                                                      current_day,current_hour)
    else:
        dProcTmp = datetime.datetime(current_year,current_month,current_day,
                                     current_hour)
        numSubIntervals = int(current_minute/ConfigOptions.fcst_freq)
        ConfigOptions.e_date_proc = dProcTmp + datetime.timedelta(
            seconds=3600.0*numSubIntervals*ConfigOptions.fcst_freq
        )

    # Calculate the beginning of the processing window.
    ConfigOptions.e_date_proc - datetime.timedelta(
        seconds=3600.0*:calculate_lookback_window()
    )