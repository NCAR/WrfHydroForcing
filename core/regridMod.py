"""
Regridding module file for regridding input forcing files.
"""
import ESMF
import os
import sys
from core import errMod
import subprocess
from netCDF4 import Dataset
import numpy as np
from core import ioMod

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

    if MpiConfig.rank == 0:
        print("REGRID STATUS = " + str(input_forcings.regridComplete))
    # Create a path for a temporary NetCDF file that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = ConfigOptions.scratch_dir + "/" + \
        "GFS_TMP.nc"

    MpiConfig.comm.barrier()

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

    # Loop through all of the input forcings in GFS data. Convert the GRIB2 files
    # to NetCDF, read in the data, regrid it, then map it to the apropriate
    # array slice in the output arrays.
    forceCount = 0
    for forceTmp in input_forcings.netcdf_var_names:
        # Create a temporary NetCDF file from the GRIB2 file.
        if forceTmp == "PRATE":
            # By far the most complicated of output variables. We need to calculate
            # our 'average' PRATE based on our current hour.
            if input_forcings.fcst_hour2 <= 240:
                tmpHrCurrent = input_forcings.fcst_hour2
                diffTmp = tmpHrCurrent%6
                if diffTmp == 6:
                    tmpHrPrevious = tmpHrCurrent - 6
                else:
                    tmpHrPrevious = tmpHrCurrent - diffTmp
            else:
                tmpHrPrevious = input_forcings.fcst_hour1

            cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(" + \
                  input_forcings.grib_vars[forceCount] + "):(" + \
                  input_forcings.grib_levels[forceCount] + "):(" + str(tmpHrPrevious) + \
                  "-" + str(input_forcings.fcst_hour2) + \
                  " hour ave fcst):\" -netcdf " + input_forcings.tmpFile
        else:
            cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(" + \
                  input_forcings.grib_vars[forceCount] + "):(" + \
                  input_forcings.grib_levels[forceCount] + "):(" + str(input_forcings.fcst_hour2) + \
                  " hour fcst):\" -netcdf " + input_forcings.tmpFile

        idTmp = ioMod.open_grib2(input_forcings.file_in2,input_forcings.tmpFile,cmd,
                                 ConfigOptions,MpiConfig,input_forcings.netcdf_var_names[forceCount])
        MpiConfig.comm.barrier()

        calcRegridFlag = check_regrid_status(idTmp,forceCount,input_forcings,
                                             ConfigOptions,MpiConfig,wrfHydroGeoMeta)

        if calcRegridFlag:
            if MpiConfig.rank == 0:
                print('CALCULATING WEIGHTS')
            calculate_weights(MpiConfig, ConfigOptions,
                              forceCount, input_forcings, idTmp)
        MpiConfig.comm.barrier()

        # Regrid the input variables.
        if MpiConfig.rank == 0:
            print("REGRIDDING: " + input_forcings.netcdf_var_names[forceCount])
            varTmp = idTmp.variables[input_forcings.netcdf_var_names[forceCount]][0,:,:]
        else:
            varTmp = None
        MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
        MpiConfig.comm.barrier

        input_forcings.esmf_field_in.data[:,:] = varSubTmp
        MpiConfig.comm.barrier

        input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                 input_forcings.esmf_field_out)
        MpiConfig.comm.barrier

        input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount],:,:] = \
            input_forcings.esmf_field_out.data
        MpiConfig.comm.barrier

        # Close the temporary NetCDF file and remove it.
        if MpiConfig.rank == 0:
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                errMod.err_out(ConfigOptions)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                errMod.err_out()

        forceCount = forceCount + 1

def check_regrid_status(idTmp,forceCount,input_forcings,ConfigOptions,MpiConfig,wrfHydroGeoMeta):
    """
    Function for checking to see if regridding weights need to be
    calculated (or recalculated).
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # If the destination ESMF field hasn't been created, create it here.
    if not input_forcings.esmf_field_out:
        try:
            input_forcings.esmf_field_out = ESMF.Field(wrfHydroGeoMeta.esmf_grid, name='GFS_REGRIDDED')
        except:
            ConfigOptions.errMsg = "Unable to create GFS destination ESMF field object."
            errMod.err_out(ConfigOptions)

    # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
    # a new weight file:
    # 1.) This is the first output time step, so we need to calculate a weight file.
    # 2.) The input forcing grid has changed.
    calcRegridFlag = False

    MpiConfig.comm.barrier()

    if input_forcings.nx_global == None or input_forcings.ny_global == None:
        # This is the first timestep.
        # Create out regridded numpy arrays to hold the regridded data.
        input_forcings.regridded_forcings1 = np.empty([8, wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                      np.float32)
        input_forcings.regridded_forcings2 = np.empty([8, wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                      np.float32)

    if MpiConfig.rank == 0:
        if input_forcings.nx_global == None or input_forcings.ny_global == None:
            # This is the first timestep.
            calcRegridFlag = True
        else:
            if MpiConfig.rank == 0:
                if idTmp.variables[input_forcings.netcdf_var_names[forceCount]].shape[1] \
                        != input_forcings.ny_global and \
                        idTmp.variables[input_forcings.netcdf_var_names[forceCount]].shape[2] \
                        != input_forcings.nx_global:
                    calcRegridFlag = True

    MpiConfig.comm.barrier()

    # Broadcast the flag to the other processors.
    calcRegridFlag = MpiConfig.broadcast_parameter(calcRegridFlag, ConfigOptions)

    MpiConfig.comm.barrier()
    return calcRegridFlag

def calculate_weights(MpiConfig,ConfigOptions,forceCount,input_forcings,idTmp):
    """
    Function to calculate ESMF weights based on the output ESMF
    field previously calculated, along with input lat/lon grids,
    and a sample dataset.
    :param MpiConfig:
    :param ConfigOptions:
    :param forceCount:
    :return:
    """
    if MpiConfig.rank == 0:
        try:
            input_forcings.ny_global = \
                idTmp.variables[input_forcings.netcdf_var_names[forceCount]].shape[1]
        except:
            ConfigOptions.errMsg = "Unable to extract Y from: " + \
                                   input_forcings.netcdf_var_names[forceCount] + " from: " + \
                                   input_forcings.tmpFile
            errMod.err_out(ConfigOptions)
        try:
            input_forcings.nx_global = \
                idTmp.variables[input_forcings.netcdf_var_names[forceCount]].shape[2]
        except:
            ConfigOptions.errMsg = "Unable to extract Y from: " + \
                                   input_forcings.netcdf_var_names[forceCount] + " from: " + \
                                   input_forcings.tmpFile
            errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    # Broadcast the forcing nx/ny values
    input_forcings.ny_global = MpiConfig.broadcast_parameter(input_forcings.ny_global,
                                                             ConfigOptions)
    input_forcings.nx_global = MpiConfig.broadcast_parameter(input_forcings.nx_global,
                                                             ConfigOptions)
    MpiConfig.comm.barrier()

    try:
        input_forcings.esmf_grid_in = ESMF.Grid(np.array([input_forcings.ny_global, input_forcings.nx_global]),
                                                staggerloc=ESMF.StaggerLoc.CENTER,
                                                coord_sys=ESMF.CoordSys.SPH_DEG)
    except:
        ConfigOptions.errMsg = "Unable to create source GFS ESMF grid from temporary file: " + \
                               input_forcings.tmpFile
        errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    try:
        input_forcings.x_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.x_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.y_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.y_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        # print('PROC: ' + str(MpiConfig.rank) + ' GFS XBOUND1 = ' + str(input_forcings.x_lower_bound))
        # print('PROC: ' + str(MpiConfig.rank) + ' GFS XBOUND2 = ' + str(input_forcings.x_upper_bound))
        # print('PROC: ' + str(MpiConfig.rank) + ' GFS YBOUND1 = ' + str(input_forcings.y_lower_bound))
        # print('PROC: ' + str(MpiConfig.rank) + ' GFS YBOUND2 = ' + str(input_forcings.y_upper_bound))
        input_forcings.nx_local = input_forcings.x_upper_bound - input_forcings.x_lower_bound
        input_forcings.ny_local = input_forcings.y_upper_bound - input_forcings.y_lower_bound
    except:
        ConfigOptions.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary " + \
                               "file: " + input_forcings.tmpFile
        errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        # Process lat/lon values from the GFS grid.
        if len(idTmp.variables['latitude'].shape) == 3:
            # We have 2D grids already in place.
            latTmp = id.variables['latitude'][0, :, :]
            lonTmp = id.variables['longitude'][0, :, :]
        elif len(idTmp.variables['longitude'].shape) == 2:
            # We have 2D grids already in place.
            latTmp = id.variables['latitude'][0, :, :]
            lonTmp = id.variables['longitude'][0, :, :]
        elif len(idTmp.variables['latitude'].shape) == 1:
            # We have 1D lat/lons we need to translate into
            # 2D grids.
            latTmp = np.repeat(idTmp.variables['latitude'][:][:, np.newaxis], input_forcings.nx_global, axis=1)
            lonTmp = np.tile(idTmp.variables['longitude'][:], (input_forcings.ny_global, 1))
    MpiConfig.comm.barrier()

    # Scatter global GFS latitude grid to processors..
    if MpiConfig.rank == 0:
        varTmp = latTmp
    else:
        varTmp = None
    varSubLatTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
    MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        varTmp = lonTmp
    else:
        varTmp = None
    varSubLonTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
    MpiConfig.comm.barrier()

    try:
        input_forcings.esmf_lats = input_forcings.esmf_grid_in.get_coords(0)
    except:
        ConfigOptions.errMsg = "Unable to locate latitude coordinate object within input GFS ESMF grid."
        errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    try:
        input_forcings.esmf_lons = input_forcings.esmf_grid_in.get_coords(1)
    except:
        ConfigOptions.errMsg = "Unable to locate longitude coordinate object within input GFS ESMF grid."
        errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    input_forcings.esmf_lats[:, :] = varSubLatTmp
    input_forcings.esmf_lons[:, :] = varSubLonTmp
    varSubLatTmp = None
    varSubLonTmp = None
    latTmp = None
    lonTmp = None

    # Create a ESMF field to hold the incoming data.
    input_forcings.esmf_field_in = ESMF.Field(input_forcings.esmf_grid_in, name="GFS_NATIVE")

    MpiConfig.comm.barrier()

    # Scatter global grid to processors..
    if MpiConfig.rank == 0:
        varTmp = idTmp[input_forcings.netcdf_var_names[forceCount]][0, :, :]
    else:
        varTmp = None
    varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
    MpiConfig.comm.barrier()

    # Place temporary data into the field array for generating the regridding object.
    input_forcings.esmf_field_in.data[:, :] = varSubTmp
    MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        print("CREATING GFS REGRID OBJECT")
    input_forcings.regridObj = ESMF.Regrid(input_forcings.esmf_field_in,
                                           input_forcings.esmf_field_out,
                                           regrid_method=ESMF.RegridMethod.BILINEAR,
                                           unmapped_action=ESMF.UnmappedAction.IGNORE)