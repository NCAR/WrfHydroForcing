"""
Regridding module file for regridding input forcing files.
"""
#import ESMF

def regrid_gfs(input_forcings,ConfigOptions):
    """
    Function for handing regridding of input GFS data
    fro GRIB2 files.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        return