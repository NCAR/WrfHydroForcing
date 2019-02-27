"""
Regridding module file for regridding input forcing files.
"""
import ESMF
import os
from core import errMod
import subprocess
from netCDF4 import Dataset
import numpy as np

def regrid_gfs(input_forcings,ConfigOptions,wrfHydroGeoMeta,MpiConfig):
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

    print("REGRID STATUS = " + str(input_forcings.regridComplete))
    # Create a path for a temporary NetCDF file that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = ConfigOptions.scratch_dir + "/" + \
        "GFS_TMP.nc"

    MpiConfig.comm.barrier()

    #print(input_forcings.tmpFile)
    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if MpiConfig.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                input_forcings.tmpFile + " - Removing....."
            errMod.log_warning(ConfigOptions)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                errMod.err_out(ConfigOptions)

    # We will process each variable at a time. Unfortunately, wgrib2 makes it a bit
    # difficult to handle forecast strings, otherwise this could be done in one commmand.
    # This makes a compelling case for the use of a GRIB Python API in the future....
    # Incoming shortwave radiation flux.....

    MpiConfig.comm.barrier()

    cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(DLWRF):(surface):" + \
          "(" + str(input_forcings.fcst_hour2) + " hour fcst):\" -netcdf " + input_forcings.tmpFile
    if MpiConfig.rank == 0:
        try:
            cmdOutput = subprocess.Popen([cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
            out,err = cmdOutput.communicate()
            exitcode = cmdOutput.returncode
        except:
            ConfigOptions.errMsg = "Unable to open GFS file: " + input_forcings.file_in2
            errMod.err_out()
        # Ensure the temporary file has been created.
        if not os.path.isfile(input_forcings.tmpFile):
            ConfigOptions.errMsg = "Unable to find expected temporary file: " + input_forcings.tmpFile
            errMod.err_out()

        # Open the temporary file, ensure expected variables are in the file.
        try:
            idTmp = Dataset(input_forcings.tmpFile,'r')
        except:
            ConfigOptions.errMsg = "Unable to open temporary NetCDF file: " + input_forcings.tmpFile
            errMod.err_out()
        if 'latitude' not in idTmp.variables.keys():
            ConfigOptions.errMsg = "Unable to locate expected 'latitude' variable in: " + input_forcings.tmpFile
            errMod.err_out()
        if 'longitude' not in idTmp.variables.keys():
            ConfigOptions.errMsg = "Unable to locate expected 'longitude' variable in: " + input_forcings.tmpFile
            errMod.err_out()
        if 'DLWRF_surface' not in idTmp.variables.keys():
            ConfigOptions.errMsg = "Unable to locate expected 'DLWRF_surface' variable in: " + input_forcings.tmpFile
            errMod.err_out()

    MpiConfig.comm.barrier()

    # If the destination ESMF field hasn't been created, create it here.
    if not input_forcings.esmf_field_out:
        try:
            input_forcings.esmf_field_out = ESMF.Field(wrfHydroGeoMeta.esmf_grid,name='GFS_REGRIDDED')
        except:
            ConfigOptions.errMsg = "Unable to create GFS destination ESMF field object."
            errMod.err_out(ConfigOptions)

    # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
    # a new weight file:
    # 1.) This is the first output time step, so we need to calculate a weight file.
    # 2.) The input forcing grid has changed.
    calcRegridFlag = False

    MpiConfig.comm.barrier()

    if input_forcings.nxGlobal == None or input_forcings.nyGlobal == None:
        # This is the first timestep.
        calcRegridFlag = True
        # Create out regridded numpy arrays to hold the regridded data.
        input_forcings.regridded_forcings1 = np.empty([8, wrfHydroGeoMeta.nx_local, wrfHydroGeoMeta.ny_local],
                                                      np.float32)
        input_forcings.regridded_forcings2 = np.empty([8, wrfHydroGeoMeta.nx_local, wrfHydroGeoMeta.ny_local],
                                                       np.float32)
    else:
        if idTmp.variables['DLWRF_surface'].shape[1] != input_forcings.nyGlobal and \
                idTmp.variables['DLWRF_surface'].shape[0] != input_forcings.nxGlobal:
            calcRegridFlag = True
            input_forcings.regridded_forcings2 = np.empty([8, wrfHydroGeoMeta.nx_local, wrfHydroGeoMeta.ny_local],
                                                          np.float32)

    if calcRegridFlag:
        try:
            input_forcings.nyGlobal = idTmp.variables['DLWRF_surface'].shape[1]
        except:
            ConfigOptions.errMsg = "Unable to extract Y from DLWRF_surface in: " + input_forcings.tmpFile
            errMod.err_out(ConfigOptions)
        try:
            input_forcings.nxGlobal = idTmp.variables['DLWRF_surface'].shape[2]
        except:
            ConfigOptions.errMsg = "Unable to extract X from DLWRF_surface in: " + input_forcings.tmpFile
            errMod.err_out(ConfigOptions)
        try:
            input_forcings.esmf_grid_in = ESMF.Grid(np.array([input_forcings.nyGlobal,input_forcings.nxGlobal]),
                                                   staggerloc=ESMF.StaggerLoc.CENTER,
                                                   coord_sys=ESMF.CoordSys.SPH_DEG)
        except:
            ConfigOptions.errMsg = "Unable to create source GFS ESMF grid from temporary file: " + \
                                   input_forcings.tmpFile
            errMod.err_out(ConfigOptions)
        try:
            input_forcings.x_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
            input_forcings.x_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
            input_forcings.y_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
            input_forcings.y_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
            input_forcings.nx_local = input_forcings.x_upper_bound - input_forcings.x_lower_bound
            input_forcings.ny_local = input_forcings.y_upper_bound - input_forcings.y_upper_bound
        except:
            ConfigOptions.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary " + \
                "file: " + input_forcings.tmpFile
            errMod.err_out(ConfigOptions)

        # Process lat/lon values from the GFS grid.
        if len(idTmp.variables['latitude']) == 3:
            # We have 2D grids already in place.
            latTmp = id.variables['latitude'][0, :, :]
            lonTmp = id.variables['longitude'][0, :, :]
        elif len(idTmp.variables['longitude']) == 2:
            # We have 2D grids already in place.
            latTmp = id.variables['latitude'][0, :, :]
            lonTmp = id.variables['longitude'][0, :, :]
        elif len(idTmp.variables['latitude']) == 1:
            # We have 1D lat/lons we need to translate into
            # 2D grids.
            latTmp = np.repeat(idTmp.variables['latitude'][:],input_forcings.nxGlobal,axis=1)
            lonTmp = np.tile(idTmp.variables['longitude'][:],(input_forcings.nyGlobal,1))

        try:
            input_forcings.esmf_lats = input_forcings.esmf_grid_in.get_coords(0)
        except:
            ConfigOptions.errMsg = "Unable to locate latitude coordinate object within input GFS ESMF grid."
            errMod.err_out(ConfigOptions)

        try:
            input_forcings.esmf_lons = input_forcings.esmf_grid_in.get_coords(1)
        except:
            ConfigOptions.errMsg = "Unable to locate longitude coordinate object within input GFS ESMF grid."
            errMod.err_out(ConfigOptions)

        input_forcings.esmf_lats[:,:] = latTmp[input_forcings.y_lower_bound:input_forcings.y_upper_bound,
                                        input_forcings.x_lower_bound:input_forcings.x_upper_bound]
        input_forcings.esmf_lons[:, :] = lonTmp[input_forcings.y_lower_bound:input_forcings.y_upper_bound,
                                         input_forcings.x_lower_bound:input_forcings.x_upper_bound]

        #Create a ESMF field to hold the incoming data.
        input_forcings.esmf_field_in = ESMF.Field(input_forcings.esmf_grid_in,name="GFS_NATIVE")

        # Place temporary data into the field array for generating the regridding object.
        input_forcings.esmf_field_in[:,:] = idTmp[input_forcings.y_lower_bound:input_forcings.y_upper_bound,
                                        input_forcings.x_lower_bound:input_forcings.x_upper_bound]

        # Create out regridded numpy arrays to hold the regridded data.
        input_forcings.regridded_forcings1 = np.empty([8,wrfHydroGeoMeta.nx_local,wrfHydroGeoMeta.ny_local],
                                                      np.float32)
        input_forcings.regridded_forcings2 = np.empty([8,wrfHydroGeoMeta.nx_local,wrfHydroGeoMeta.ny_local],
                                                      np.float32)

        input_forcings.regridObj = ESMF.Regrid(input_forcings.lw_field_in2,
                                               input_forcings.lw_field_in2,
                                               regrid_method=ESMF.RegridMethod.BILINEAR,
                                               unmapped_action=ESMF.UnmappedAction.IGNORE)

    # Go through and regrid all the input variables.
    input_forcings.esmf_field_in[:,:] = idTmp.variables['DLWRF_surface'][input_forcings.y_lower_bound:input_forcings.y_upper_bound,
                                        input_forcings.x_lower_bound:input_forcings.x_upper_bound]
    input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                             input_forcings.esmf_field_out)
    input_forcings.regridded_forcings2[0,:,:] = input_forcings.esmf_field_out[:,:]


    print(cmd)