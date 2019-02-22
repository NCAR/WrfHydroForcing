from core import errmod

def process_forecasts(jobMeta):
    """
    Main calling module for running realtime forecasts and re-forecasts.
    :param jobMeta:
    :return:
    """

    # Before any processing begins, we must first initialize a geospatial
    # abstract class for the WRF-Hydro grid we are processing our forcings to.
    # This will depend on the geogrid file specified by the user.
    #geoMetaWrfHydro =