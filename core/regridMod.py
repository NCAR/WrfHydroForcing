"""
Regridding module file for regridding input forcing files.
"""
#import ESMF
import os
from core import errMod
import subprocess

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
    #print("REGRID STATUS = " + str(input_forcings.regridComplete))
    if input_forcings.regridComplete:
        return

    # Create a path for a temporary NetCDF file that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = ConfigOptions.scratch_dir + "/" + \
        input_forcings.file_in2 + "_TMP.nc"

    #print(input_forcings.tmpFile)
    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if os.path.isfile(input_forcings.file_in2):
        ConfigOptions.statusMsg = "Found old temporary file: " + \
            input_forcings.file_in2 + " - Removing....."
        errMod.log_warning(ConfigOptions)
        try:
            os.remove(input_forcings.file_in2)
        except:
            errMod.err_out(ConfigOptions)

    # We will process each variable at a time. Unfortunately, wgrib2 makes it a bit
    # difficult to handle forecast strings, otherwise this could be done in one commmand.
    # This makes a compelling case for the use of a GRIB Python API in the future....
    # Incoming shortwave radiation flux.....
    cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(DLWRF):(surface):" + \
          "(" + str(input_forcings.fcst_hour2) + " hour fcst):\""
    try:
        cmdOutput = subprocess.Popen([cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
        out,err = cmdOutput.communicate()
        exitcode = cmdOutput.returncode
    except:
        ConfigOptions.errMsg = "Unable to open GFS file: " + input_forcings.file_in2
        errMod.err_out()


    print(cmd)