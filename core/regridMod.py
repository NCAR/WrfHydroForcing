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

def regrid_conus_hrrr(input_forcings,ConfigOptions,wrfHydroGeoMeta,MpiConfig):
    """
    Function for handling regridding of HRRR data.
    :param input_forcings:
    :param ConfigOptions:
    :param wrfHydroGeoMeta:
    :param MpiConfig:
    :return:
    """
    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "No HRRR regridding required for this timesetp."
            errMod.log_msg(ConfigOptions,MpiConfig)
        errMod.log_msg(ConfigOptions,MpiConfig)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = ConfigOptions.scratch_dir + "/" + \
        "HRRR_CONUS_TMP.nc"
    input_forcings.tmpFileHeight = ConfigOptions.scratch_dir + "/" + \
                                   "HRRR_CONUS_TMP_HEIGHT.nc"

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if MpiConfig.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                                      input_forcings.tmpFile + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove temporary file: " + \
                                       input_forcings.tmpFile
                errMod.log_critical(ConfigOptions,MpiConfig)
                pass
    errMod.check_program_status(ConfigOptions,MpiConfig)

    forceCount = 0
    for forceTmp in input_forcings.grib_vars:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Processing Conus HRRR Variable: " + input_forcings.grib_vars[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        # Create a temporary NetCDF file from the GRIB2 file.
        cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(" + \
              input_forcings.grib_vars[forceCount] + "):(" + \
              input_forcings.grib_levels[forceCount] + "):(" + str(input_forcings.fcst_hour2) + \
              " hour fcst):\" -netcdf " + input_forcings.tmpFile

        idTmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                 ConfigOptions, MpiConfig, input_forcings.netcdf_var_names[forceCount])
        errMod.check_program_status(ConfigOptions,MpiConfig)

        calcRegridFlag = check_regrid_status(idTmp, forceCount, input_forcings,
                                             ConfigOptions, MpiConfig, wrfHydroGeoMeta)
        errMod.check_program_status(ConfigOptions,MpiConfig)

        if calcRegridFlag:
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Calculating HRRR regridding weights."
                errMod.log_msg(ConfigOptions,MpiConfig)
            calculate_weights(MpiConfig, ConfigOptions,
                              forceCount, input_forcings, idTmp)
            errMod.check_program_status(ConfigOptions,MpiConfig)

            # Read in the HRRR height field, which is used for downscaling purposes.
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Reading in HRRR elevation data."
                errMod.log_msg(ConfigOptions,MpiConfig)
            cmd = "wgrib2 " + input_forcings.file_in2 + " -match " + \
                "\":(HGT):(surface):\" " + \
                " -netcdf " + input_forcings.tmpFileHeight
            idTmpHeight = ioMod.open_grib2(input_forcings.file_in2,input_forcings.tmpFileHeight,
                                           cmd,ConfigOptions,MpiConfig,'HGT_surface')
            errMod.check_program_status(ConfigOptions,MpiConfig)

            # Regrid the height variable.
            if MpiConfig.rank == 0:
                try:
                    varTmp = idTmpHeight.variables['HGT_surface'][0,:,:]
                except:
                    ConfigOptions.errMsg = "Unable to extract HRRR elevation from: " + input_forcings.tmpFileHeight
                    pass
            else:
                varTmp = None
            errMod.check_program_status(ConfigOptions,MpiConfig)

            varSubTmp = MpiConfig.scatter_array(input_forcings,varTmp,ConfigOptions)
            errMod.check_program_status(ConfigOptions,MpiConfig)

            try:
                input_forcings.esmf_field_in.data[:,:] = varSubTmp
            except:
                ConfigOptions.errMsg = "Unable to place input NetCDF HRRR data into the ESMF field object."
                errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions,MpiConfig)

            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                errMod.log_msg(ConfigOptions,MpiConfig)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except:
                ConfigOptions.errMsg = "Unable to regrid HRRR surface elevation using ESMF."
                errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    ConfigOptions.globalNdv
            except:
                ConfigOptions.errMsg = "Unable to perform HRRR mask search on elevation data."
                errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.height[:,:] = input_forcings.esmf_field_out.data
            except:
                ConfigOptions.errMsg = "Unable to extract regridded HRRR elevation data from ESMF."
                errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Close the temporary NetCDF file and remove it.
            if MpiConfig.rank == 0:
                try:
                    idTmpHeight.close()
                except:
                    ConfigOptions.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions,MpiConfig)

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except:
                    ConfigOptions.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Regrid the input variables.
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Processing input HRRR variable: " + input_forcings.netcdf_var_names[forceCount]
            errMod.log_msg(ConfigOptions,MpiConfig)
            try:
                varTmp = idTmp.variables[input_forcings.netcdf_var_names[forceCount]][0,:,:]
            except:
                ConfigOptions.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[forceCount] + \
                                       " from: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            varTmp = None
        errMod.check_program_status(ConfigOptions, MpiConfig)

        varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)


        try:
            input_forcings.esmf_field_in.data[:,:] = varSubTmp
        except:
            ConfigOptions.errMsg = "Unable to place input HRRR data into ESMF field."
            errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Regridding Input HRRR Field: " + input_forcings.netcdf_var_names[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except:
            ConfigOptions.errMsg = "Unable to regrid input HRRR forcing data."
            errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                ConfigOptions.globalNdv
        except:
            ConfigOptions.errMsg = "Unable to perform mask test on regridded HRRR forcings."
            errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount],:,:] = \
                input_forcings.esmf_field_out.data
        except:
            ConfigOptions.errMsg = "Unable to extract regridded HRRR forcing data from the ESMF field."
            errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if ConfigOptions.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[forceCount], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount], :, :]
        MpiConfig.comm.barrier()

        # Close the temporary NetCDF file and remove it.
        if MpiConfig.rank == 0:
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions,MpiConfig)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions,MpiConfig)
        MpiConfig.comm.barrier()

        forceCount = forceCount + 1

def regrid_conus_rap(input_forcings,ConfigOptions,wrfHydroGeoMeta,MpiConfig):
    """
    Function for handling regridding of RAP 13km conus data.
    :param input_forcings:
    :param ConfigOptions:
    :param wrfHydroGeoMeta:
    :param MpiConfig:
    :return:
    """
    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "No RAP regridding required for this timesetp."
            errMod.log_msg(ConfigOptions, MpiConfig)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = ConfigOptions.scratch_dir + "/" + \
        "RAP_CONUS_TMP.nc"
    input_forcings.tmpFileHeight = ConfigOptions.scratch_dir + "/" + \
                                   "RAP_CONUS_TMP_HEIGHT.nc"

    errMod.check_program_status(ConfigOptions,MpiConfig)

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if MpiConfig.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                                      input_forcings.tmpFile + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
    errMod.check_program_status(ConfigOptions,MpiConfig)

    forceCount = 0
    for forceTmp in input_forcings.grib_vars:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Processing Conus RAP Variable: " + input_forcings.grib_vars[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        # Create a temporary NetCDF file from the GRIB2 file.
        cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(" + \
              input_forcings.grib_vars[forceCount] + "):(" + \
              input_forcings.grib_levels[forceCount] + "):(" + str(input_forcings.fcst_hour2) + \
              " hour fcst):\" -netcdf " + input_forcings.tmpFile

        idTmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                 ConfigOptions, MpiConfig, input_forcings.netcdf_var_names[forceCount])
        errMod.check_program_status(ConfigOptions,MpiConfig)

        calcRegridFlag = check_regrid_status(idTmp, forceCount, input_forcings,
                                             ConfigOptions, MpiConfig, wrfHydroGeoMeta)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        if calcRegridFlag:
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Calculating RAP regridding weights."
            calculate_weights(MpiConfig, ConfigOptions,
                              forceCount, input_forcings, idTmp)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Read in the RAP height field, which is used for downscaling purposes.
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Reading in RAP elevation data."
                errMod.log_msg(ConfigOptions, MpiConfig)
            cmd = "wgrib2 " + input_forcings.file_in2 + " -match " + \
                "\":(HGT):(surface):\" " + \
                " -netcdf " + input_forcings.tmpFileHeight
            idTmpHeight = ioMod.open_grib2(input_forcings.file_in2,input_forcings.tmpFileHeight,
                                           cmd,ConfigOptions,MpiConfig,'HGT_surface')
            errMod.check_program_status(ConfigOptions,MpiConfig)

            # Regrid the height variable.
            if MpiConfig.rank == 0:
                try:
                    varTmp = idTmpHeight.variables['HGT_surface'][0,:,:]
                except:
                    ConfigOptions.errMsg = "Unable to extract HGT_surface from : " + idTmpHeight
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    pass
            else:
                varTmp = None
            errMod.check_program_status(ConfigOptions,MpiConfig)

            varSubTmp = MpiConfig.scatter_array(input_forcings,varTmp,ConfigOptions)
            errMod.check_program_status(ConfigOptions,MpiConfig)

            try:
                input_forcings.esmf_field_in.data[:,:] = varSubTmp
            except:
                ConfigOptions.errMsg = "Unable to place temporary RAP elevation variable into ESMF field."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                errMod.log_msg(ConfigOptions, MpiConfig)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                        input_forcings.esmf_field_out)
            except:
                ConfigOptions.errMsg = "Unable to regrid RAP elevation data using ESMF."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    ConfigOptions.globalNdv
            except:
                ConfigOptions.errMsg = "Unable to perform mask search on RAP elevation data."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.height[:,:] = input_forcings.esmf_field_out.data
            except:
                ConfigOptions.errMsg = "Unable to place RAP ESMF elevation field into local array."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Close the temporary NetCDF file and remove it.
            if MpiConfig.rank == 0:
                try:
                    idTmpHeight.close()
                except:
                    ConfigOptions.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    pass

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except:
                    ConfigOptions.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    pass
            errMod.check_program_status(ConfigOptions, MpiConfig)

        # Regrid the input variables.
        if MpiConfig.rank == 0:
            print("REGRIDDING: " + input_forcings.netcdf_var_names[forceCount])
            try:
                varTmp = idTmp.variables[input_forcings.netcdf_var_names[forceCount]][0,:,:]
            except:
                ConfigOptions.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[forceCount] + \
                                       " from: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            varTmp = None
        errMod.check_program_status(ConfigOptions, MpiConfig)

        varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        try:
            input_forcings.esmf_field_in.data[:,:] = varSubTmp
        except:
            ConfigOptions.errMsg = "Unable to place local RAP array into ESMF field."
            errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Regridding Input RAP Field: " + input_forcings.netcdf_var_names[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except:
            ConfigOptions.errMsg = "Unable to regrid RAP variable: " + input_forcings.netcdf_var_names[forceCount]
            errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                ConfigOptions.globalNdv
        except:
            ConfigOptions.errMsg = "Unable to run mask calculation on RAP variable: " + \
                                   input_forcings.netcdf_var_names[forceCount]
            errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount],:,:] = \
                input_forcings.esmf_field_out.data
        except:
            ConfigOptions.errMsg = "Unable to place RAP ESMF data into local array."
            errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if ConfigOptions.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[forceCount], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount], :, :]
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if ConfigOptions.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[forceCount], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount], :, :]
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Close the temporary NetCDF file and remove it.
        if MpiConfig.rank == 0:
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        errMod.check_program_status(ConfigOptions, MpiConfig)

        forceCount = forceCount + 1

def regrid_cfsv2(input_forcings,ConfigOptions,wrfHydroGeoMeta,MpiConfig):
    """
    Function for handling regridding of global CFSv2 forecast data.
    :param input_forcings:
    :param ConfigOptions:
    :param wrfHydroGeoMeta:
    :param MpiConfig:
    :return:
    """
    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        # Check to see if we are running NWM-custom interpolation/bias
        # correction on incoming CFSv2 data. Because of the nature, we
        # need to regrid bias-corrected data every hour.
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "No need to read in new CFSv2 data at this time."
            errMod.log_msg(ConfigOptions, MpiConfig)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = ConfigOptions.scratch_dir + "/" + \
        "CFSv2_TMP.nc"
    input_forcings.tmpFileHeight = ConfigOptions.scratch_dir + "/" + \
                                   "CFSv2_TMP_HEIGHT.nc"
    errMod.check_program_status(ConfigOptions,MpiConfig)

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if MpiConfig.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                                      input_forcings.tmpFile + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove previous temporary file: " \
                                       " " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
    errMod.check_program_status(ConfigOptions, MpiConfig)

    forceCount = 0
    for forceTmp in input_forcings.grib_vars:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Processing CFSv2 Variable: " + input_forcings.grib_vars[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        # Create a temporary NetCDF file from the GRIB2 file.
        cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(" + \
              input_forcings.grib_vars[forceCount] + "):(" + \
              input_forcings.grib_levels[forceCount] + "):(" + str(input_forcings.fcst_hour2) + \
              " hour fcst):\" -netcdf " + input_forcings.tmpFile

        idTmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                 ConfigOptions, MpiConfig, input_forcings.netcdf_var_names[forceCount])
        errMod.check_program_status(ConfigOptions, MpiConfig)

        calcRegridFlag = check_regrid_status(idTmp, forceCount, input_forcings,
                                             ConfigOptions, MpiConfig, wrfHydroGeoMeta)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        if calcRegridFlag:
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Calculate CFSv2 regridding weights."
                errMod.log_msg(ConfigOptions, MpiConfig)

            calculate_weights(MpiConfig, ConfigOptions,
                              forceCount, input_forcings, idTmp)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Read in the RAP height field, which is used for downscaling purposes.
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Reading in CFSv2 elevation data."
                errMod.log_msg(ConfigOptions, MpiConfig)

            cmd = "wgrib2 " + input_forcings.file_in2 + " -match " + \
                "\":(HGT):(surface):\" " + \
                " -netcdf " + input_forcings.tmpFileHeight
            idTmpHeight = ioMod.open_grib2(input_forcings.file_in2,input_forcings.tmpFileHeight,
                                           cmd,ConfigOptions,MpiConfig,'HGT_surface')
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Regrid the height variable.
            if MpiConfig.rank == 0:
                try:
                    varTmp = idTmpHeight.variables['HGT_surface'][0,:,:]
                except:
                    ConfigOptions.errMsg = "Unable to extract HGT_surface from file:" \
                                           " " + input_forcings.file_in2
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    pass
            else:
                varTmp = None
            errMod.check_program_status(ConfigOptions, MpiConfig)

            varSubTmp = MpiConfig.scatter_array(input_forcings,varTmp,ConfigOptions)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.esmf_field_in.data[:,:] = varSubTmp
            except:
                ConfigOptions.errMsg = "Unable to place CFSv2 elevation data into the ESMF field object."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                errMod.log_msg(ConfigOptions, MpiConfig)

            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except:
                ConfigOptions.errMsg = "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    ConfigOptions.globalNdv
            except:
                ConfigOptions.errMsg = "Unable to run mask calculation on CFSv2 elevtation data."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)


            try:
                input_forcings.height[:,:] = input_forcings.esmf_field_out.data
            except:
                ConfigOptions.errMsg = "Unable to extract CFSv2 regridded elevation data from ESMF field."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Close the temporary NetCDF file and remove it.
            if MpiConfig.rank == 0:
                try:
                    idTmpHeight.close()
                except:
                    ConfigOptions.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            if MpiConfig.rank == 0:
                try:
                    os.remove(input_forcings.tmpFileHeight)
                except:
                    ConfigOptions.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

        # Regrid the input variables.
        if MpiConfig.rank == 0:
            if not ConfigOptions.runCfsNldasBiasCorrect:
                ConfigOptions.statusMsg = "Regridding CFSv2 variable: " + input_forcings.netcdf_var_names[forceCount]
                errMod.log_msg(ConfigOptions, MpiConfig)
            try:
                varTmp = idTmp.variables[input_forcings.netcdf_var_names[forceCount]][0,:,:]
            except:
                ConfigOptions.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[forceCount] + \
                                       " from file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            varTmp = None
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Scatter the global CFSv2 data to the local processors.
        varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Assign local CFSv2 data to the input forcing object.. IF..... we are running the
        # bias correction. These grids are interpolated in a separate routine, AFTER bias
        # correction has taken place.
        if ConfigOptions.runCfsNldasBiasCorrect:
            if not input_forcings.coarse_input_forcings1 and not input_forcings.coarse_input_forcings2 and \
                    ConfigOptions.current_output_step == 1:
                # We need to create NumPy arrays to hold the CFSv2 global data.
                input_forcings.coarse_input_forcings2 = np.empty([8, varSubTmp.shape[0], varSubTmp.shape[1]],
                                                                 np.float64)
                input_forcings.coarse_input_forcings1 = np.empty([8, varSubTmp.shape[0], varSubTmp.shape[1]],
                                                                 np.float64)
            try:
                input_forcings.coarse_input_forcings2[input_forcings.input_map_output[forceCount],:,:] = varSubTmp
            except:
                ConfigOptions.errMsg = "Unable to place local CFSv2 input variable: " + \
                                        input_forcings.netcdf_var_names[forceCount] + \
                                        " into local numpy array."
                pass
            if ConfigOptions.current_output_step == 1:
                input_forcings.coarse_input_forcings1[input_forcings.input_map_output[forceCount], :, :] = \
                    input_forcings.coarse_input_forcings2[input_forcings.input_map_output[forceCount], :, :]
        else:
            input_forcings.coarse_input_forcings2 = None
            input_forcings.coarse_input_forcings1 = None
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Only regrid the current files if we did not specify the NLDAS2 NWM bias correction, which needs to take place
        # first before any regridding can take place. That takes place in the bias-correction routine.
        if not ConfigOptions.runCfsNldasBiasCorrect:
            try:
                input_forcings.esmf_field_in.data[:,:] = varSubTmp
            except:
                ConfigOptions.errMsg = "Unable to place CFSv2 forcing data into temporary ESMF field."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except:
                ConfigOptions.errMsg = "Unable to regrid CFSv2 variable: " + input_forcings.netcdf_var_names[forceCount]
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    ConfigOptions.globalNdv
            except:
                ConfigOptions.errMsg = "Unable to run mask calculation on CFSv2 variable: " + \
                                       input_forcings.netcdf_var_names[forceCount]
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount],:,:] = \
                    input_forcings.esmf_field_out.data
            except:
                ConfigOptions.errMsg = "Unable to extract ESMF field data for CFSv2."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if ConfigOptions.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[forceCount], :, :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount], :, :]
            errMod.check_program_status(ConfigOptions, MpiConfig)

        # Close the temporary NetCDF file and remove it.
        if MpiConfig.rank == 0:
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        if MpiConfig.rank == 0:
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        forceCount = forceCount + 1

def regrid_custom_hourly_netcdf(input_forcings,ConfigOptions,wrfHydroGeoMeta,MpiConfig):
    """
    Function for handling regridding of custom input NetCDF hourly forcing files.
    :param input_forcings:
    :param ConfigOptions:
    :param wrfHydroGeoMeta:
    :param MpiConfig:
    :return:
    """
    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        return

    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "No Custom Hourly NetCDF regridding required for this timesetp."
        errMod.log_msg(ConfigOptions, MpiConfig)
    MpiConfig.comm.barrier()

    # Open the input NetCDF file containing necessary data.
    idTmp = ioMod.open_netcdf_forcing(input_forcings.file_in2,ConfigOptions,MpiConfig)
    MpiConfig.comm.barrier()

    forceCount = 0
    for forceTmp in input_forcings.netcdf_var_names:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Processing Custom NetCDF Forcing Variable: " + input_forcings.grib_vars[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        calcRegridFlag = check_regrid_status(idTmp, forceCount, input_forcings,
                                             ConfigOptions, MpiConfig, wrfHydroGeoMeta)

        if calcRegridFlag:
            if MpiConfig.rank == 0:
                print('CALCULATING WEIGHTS')
            calculate_weights(MpiConfig, ConfigOptions,
                              forceCount, input_forcings, idTmp)

            # Read in the RAP height field, which is used for downscaling purposes.
            if MpiConfig.rank == 0:
                print("READING IN CUSTOM HEIGHT FIELD")
            if 'HGT_surface' not in idTmp.variables.keys():
                ConfigOptions.errMsg = "Unable to locate HGT_surface in: " + input_forcings.file_in2
                raise Exception()
            MpiConfig.comm.barrier()

            # Regrid the height variable.
            if MpiConfig.rank == 0:
                varTmp = idTmp.variables['HGT_surface'][0,:,:]
            else:
                varTmp = None
            MpiConfig.comm.barrier()

            varSubTmp = MpiConfig.scatter_array(input_forcings,varTmp,ConfigOptions)
            MpiConfig.comm.barrier()

            input_forcings.esmf_field_in.data[:,:] = varSubTmp
            MpiConfig.comm.barrier()

            if MpiConfig.rank == 0:
                print("REGRIDDING CUSTOM HEIGHT FIELD")
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
            # Set any pixel cells outside the input domain to the global missing value.
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                ConfigOptions.globalNdv
            MpiConfig.comm.barrier()

            input_forcings.height[:,:] = input_forcings.esmf_field_out.data
            MpiConfig.comm.barrier()

        MpiConfig.comm.barrier()

        # Regrid the input variables.
        if MpiConfig.rank == 0:
            print("REGRIDDING: " + input_forcings.netcdf_var_names[forceCount])
            varTmp = idTmp.variables[input_forcings.netcdf_var_names[forceCount]][0,:,:]
        else:
            varTmp = None
        MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
        MpiConfig.comm.barrier()

        input_forcings.esmf_field_in.data[:,:] = varSubTmp
        MpiConfig.comm.barrier()

        input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                 input_forcings.esmf_field_out)
        # Set any pixel cells outside the input domain to the global missing value.
        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
            ConfigOptions.globalNdv
        MpiConfig.comm.barrier()

        input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount],:,:] = \
            input_forcings.esmf_field_out.data
        MpiConfig.comm.barrier()

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if ConfigOptions.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[forceCount], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount], :, :]
        MpiConfig.comm.barrier()

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
    if input_forcings.regridComplete:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "No 13km GFS regridding required for this timesetp."
            errMod.log_msg(ConfigOptions, MpiConfig)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = ConfigOptions.scratch_dir + "/" + \
        "GFS_TMP.nc"
    input_forcings.tmpFileHeight = ConfigOptions.scratch_dir + "/" + \
                                   "GFS_TMP_HEIGHT.nc"
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if MpiConfig.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                input_forcings.tmpFile + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # We will process each variable at a time. Unfortunately, wgrib2 makes it a bit
    # difficult to handle forecast strings, otherwise this could be done in one commmand.
    # This makes a compelling case for the use of a GRIB Python API in the future....
    # Incoming shortwave radiation flux.....

    # Loop through all of the input forcings in GFS data. Convert the GRIB2 files
    # to NetCDF, read in the data, regrid it, then map it to the apropriate
    # array slice in the output arrays.
    forceCount = 0
    for forceTmp in input_forcings.grib_vars:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Processing 13km GFS Variable: " + input_forcings.grib_vars[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        # Create a temporary NetCDF file from the GRIB2 file.
        if forceTmp == "PRATE":
            # By far the most complicated of output variables. We need to calculate
            # our 'average' PRATE based on our current hour.
            if input_forcings.fcst_hour2 <= 240:
                tmpHrCurrent = input_forcings.fcst_hour2
                diffTmp = tmpHrCurrent%6
                if diffTmp == 0:
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
        errMod.check_program_status(ConfigOptions, MpiConfig)

        calcRegridFlag = check_regrid_status(idTmp,forceCount,input_forcings,
                                             ConfigOptions,MpiConfig,wrfHydroGeoMeta)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        if calcRegridFlag:
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Calculating 13km GFS regridding weights."
            calculate_weights(MpiConfig, ConfigOptions,
                              forceCount, input_forcings, idTmp)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Read in the GFS height field, which is used for downscaling purposes.
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Reading in 13km GFS elevation data."
                errMod.log_msg(ConfigOptions, MpiConfig)
            cmd = "wgrib2 " + input_forcings.file_in2 + " -match " + \
                "\":(HGT):(surface):\" " + \
                " -netcdf " + input_forcings.tmpFileHeight
            idTmpHeight = ioMod.open_grib2(input_forcings.file_in2,input_forcings.tmpFileHeight,
                                           cmd,ConfigOptions,MpiConfig,'HGT_surface')
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Regrid the height variable.
            if MpiConfig.rank == 0:
                try:
                    varTmp = idTmpHeight.variables['HGT_surface'][0,:,:]
                except:
                    ConfigOptions.errMsg = "Unable to extract GFS elevation from: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    pass
            else:
                varTmp = None
            errMod.check_program_status(ConfigOptions, MpiConfig)

            varSubTmp = MpiConfig.scatter_array(input_forcings,varTmp,ConfigOptions)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.esmf_field_in.data[:,:] = varSubTmp
            except:
                ConfigOptions.errMsg = "Unable to place local GFS array into an ESMF field."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Regridding 13km GFS surface elevation data to the WRF-Hydro domain."
                errMod.log_msg(ConfigOptions, MpiConfig)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except:
                ConfigOptions.errMsg = "Unable to regrid GFS elevation data."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    ConfigOptions.globalNdv
            except:
                ConfigOptions.errMsg = "Unable to perform mask search on GFS elevation data."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.height[:,:] = input_forcings.esmf_field_out.data
            except:
                ConfigOptions.errMsg = "Unable to extract GFS elevation array from ESMF field."
                errMod.log_critical(ConfigOptions, MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Close the temporary NetCDF file and remove it.
            if MpiConfig.rank == 0:
                try:
                    idTmpHeight.close()
                except:
                    ConfigOptions.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    pass

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except:
                    ConfigOptions.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    pass
            errMod.check_program_status(ConfigOptions, MpiConfig)

        # Regrid the input variables.
        if MpiConfig.rank == 0:
            try:
                varTmp = idTmp.variables[input_forcings.netcdf_var_names[forceCount]][0,:,:]
            except:
                ConfigOptions.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[forceCount] + \
                                       " from: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            varTmp = None
        errMod.check_program_status(ConfigOptions, MpiConfig)

        varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        try:
            input_forcings.esmf_field_in.data[:,:] = varSubTmp
        except:
            ConfigOptions.errMsg = "Unable to place GFS local array into ESMF field object."
            errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Regridding Input 13km GFS Field: " + input_forcings.netcdf_var_names[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except:
            ConfigOptions.errMsg = "Unable to regrid GFS variable: " + input_forcings.netcdf_var_names[forceCount]
            errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                ConfigOptions.globalNdv
        except:
            ConfigOptions.errMsg = "Unable to run mask search on GFS variable: " + \
                                   input_forcings.netcdf_var_names[forceCount]
            errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount],:,:] = \
                input_forcings.esmf_field_out.data
        except:
            ConfigOptions.errMsg = "Unable to extract GFS ESMF field data to local array."
            errMod.log_critical(ConfigOptions, MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if ConfigOptions.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[forceCount], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount], :, :]
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Close the temporary NetCDF file and remove it.
        if MpiConfig.rank == 0:
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        errMod.check_program_status(ConfigOptions, MpiConfig)

        forceCount = forceCount + 1

def regrid_nam_nest(input_forcings,ConfigOptions,wrfHydroGeoMeta,MpiConfig):
    """
    Function for handing regridding of input NAM nest data
    fro GRIB2 files.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        ConfigOptions.statusMsg = "No regridding of NAM nest data necessary for this timestep - already completed."
        errMod.log_msg(ConfigOptions,MpiConfig)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = ConfigOptions.scratch_dir + "/" + \
        "NAM_NEST_TMP.nc"
    input_forcings.tmpFileHeight = ConfigOptions.scratch_dir + "/" + \
                                   "NAM_NEST_TMP_HEIGHT.nc"
    errMod.check_program_status(ConfigOptions,MpiConfig)


    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if MpiConfig.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                input_forcings.tmpFile + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                errMod.err_out(ConfigOptions)
    errMod.check_program_status(ConfigOptions,MpiConfig)

    # Loop through all of the input forcings in NAM nest data. Convert the GRIB2 files
    # to NetCDF, read in the data, regrid it, then map it to the apropriate
    # array slice in the output arrays.
    forceCount = 0
    for forceTmp in input_forcings.grib_vars:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Processing NAM Nest Variable: " + input_forcings.grib_vars[forceCount]
            errMod.log_msg(ConfigOptions, MpiConfig)
        # Create a temporary NetCDF file from the GRIB2 file.
        cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(" + \
              input_forcings.grib_vars[forceCount] + "):(" + \
              input_forcings.grib_levels[forceCount] + "):(" + str(input_forcings.fcst_hour2) + \
              " hour fcst):\" -netcdf " + input_forcings.tmpFile

        idTmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                 ConfigOptions, MpiConfig, input_forcings.netcdf_var_names[forceCount])
        errMod.check_program_status(ConfigOptions,MpiConfig)

        calcRegridFlag = check_regrid_status(idTmp, forceCount, input_forcings,
                                             ConfigOptions, MpiConfig, wrfHydroGeoMeta)
        errMod.check_program_status(ConfigOptions,MpiConfig)

        if calcRegridFlag:
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Calculating NAM nest regridding weights...."
                errMod.log_msg(ConfigOptions,MpiConfig)
            calculate_weights(MpiConfig, ConfigOptions,
                              forceCount, input_forcings, idTmp)
            errMod.check_program_status(ConfigOptions,MpiConfig)

            # Read in the RAP height field, which is used for downscaling purposes.
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Reading in NAM nest elevation data from GRIB2."
                errMod.log_msg(ConfigOptions,MpiConfig)
            cmd = "wgrib2 " + input_forcings.file_in2 + " -match " + \
                  "\":(HGT):(surface):\" " + \
                  " -netcdf " + input_forcings.tmpFileHeight
            idTmpHeight = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                                           cmd, ConfigOptions, MpiConfig, 'HGT_surface')
            errMod.check_program_status(ConfigOptions,MpiConfig)

            # Regrid the height variable.
            if MpiConfig.rank == 0:
                varTmp = idTmpHeight.variables['HGT_surface'][0, :, :]
            else:
                varTmp = None
            errMod.check_program_status(ConfigOptions,MpiConfig)

            varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.esmf_field_in.data[:, :] = varSubTmp
            except:
                ConfigOptions.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object."
                errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                errMod.log_msg(ConfigOptions,MpiConfig)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except:
                ConfigOptions.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain " \
                                       "using ESMF."
                errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    ConfigOptions.globalNdv
            except:
                ConfigOptions.errMsg = "Unable to compute mask on NAM nest elevation data."
                errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            try:
                input_forcings.height[:, :] = input_forcings.esmf_field_out.data
            except:
                ConfigOptions.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local " \
                                       "array."
                errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

            # Close the temporary NetCDF file and remove it.
            if MpiConfig.rank == 0:
                try:
                    idTmpHeight.close()
                except:
                    ConfigOptions.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions,MpiConfig)

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except:
                    ConfigOptions.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    errMod.log_critical(ConfigOptions,MpiConfig)
            errMod.check_program_status(ConfigOptions, MpiConfig)

        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Regrid the input variables.
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Regrdding NAM nest input variable: " + input_forcings.netcdf_var_names[forceCount]
            errMod.log_msg(ConfigOptions,MpiConfig)
            try:
                varTmp = idTmp.variables[input_forcings.netcdf_var_names[forceCount]][0, :, :]
            except:
                ConfigOptions.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[forceCount] + \
                                       " from: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions,MpiConfig)
        else:
            varTmp = None
        errMod.check_program_status(ConfigOptions, MpiConfig)

        varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        try:
            input_forcings.esmf_field_in.data[:, :] = varSubTmp
        except:
            ConfigOptions.errMsg = "Unable to place local array into local ESMF field."
            errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except:
            ConfigOptions.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF."
            errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                ConfigOptions.globalNdv
        except:
            ConfigOptions.errMsg = "Unable to calculate mask from input NAM nest regridded forcings."
            errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount], :, :] = \
                input_forcings.esmf_field_out.data
        except:
            ConfigOptions.errMsg = "Unable to place local ESMF regridded data into local array."
            errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if ConfigOptions.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[forceCount], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[forceCount], :, :]
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Close the temporary NetCDF file and remove it.
        if MpiConfig.rank == 0:
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions,MpiConfig)
            try:
                os.remove(input_forcings.tmpFile)
            except:
                ConfigOptions.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                errMod.log_critical(ConfigOptions,MpiConfig)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        forceCount = forceCount + 1

def regrid_mrms_hourly(supplemental_precip,ConfigOptions,wrfHydroGeoMeta,MpiConfig):
    """
    Function for handling regridding hourly MRMS precipitation. An RQI mask file
    Is necessary to filter out poor precipitation estimates.
    :param supplemental_precip:
    :param ConfigOptions:
    :param wrfHydroGeoMeta:
    :param MpiConfig:
    :return:
    """
    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "No MRMS regridding required for this timesetp."
            errMod.log_msg(ConfigOptions, MpiConfig)
        return
    # MRMS data originally is stored as .gz files. We need to compose a series
    # of temporary paths.
    # 1.) The unzipped GRIB2 precipitation file.
    # 2.) The unzipped GRIB2 RQI file.
    # 3.) A temporary NetCDF file that stores the precipitation grid.
    # 4.) A temporary NetCDF file that stores the RQI grid.
    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    mrms_tmp_grib2 = ConfigOptions.scratch_dir + "/MRMS_PCP_TMP.grib2"
    mrms_tmp_nc = ConfigOptions.scratch_dir + "/MRMS_PCP_TMP.nc"
    mrms_tmp_rqi_grib2 = ConfigOptions.scratch_dir + "/MRMS_RQI_TMP.grib2"
    mrms_tmp_rqi_nc = ConfigOptions.scratch_dir + "/MRMS_RQI_TMP.nc"
    MpiConfig.comm.barrier()

    # If the input paths have been set to None, this means input is missing. We will
    # alert the user, and set the final output grids to be the global NDV and return.
    if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "No MRMS Precipitation available. Supplemental precipitation will " \
                                      "not be used."
            errMod.log_msg(ConfigOptions, MpiConfig)
        supplemental_precip.regridded_precip2 = None
        supplemental_precip.regridded_precip1 = None
        return

    # These files shouldn't exist. If they do, remove them.
    if MpiConfig.rank == 0:
        if os.path.isfile(mrms_tmp_grib2):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                                      mrms_tmp_grib2 + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(mrms_tmp_grib2)
            except:
                ConfigOptions.errMsg = "Unable to remove file: " + mrms_tmp_grib2
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        if os.path.isfile(mrms_tmp_nc):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                                      mrms_tmp_nc + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(mrms_tmp_nc)
            except:
                ConfigOptions.errMsg = "Unable to remove file: " + mrms_tmp_nc
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        if os.path.isfile(mrms_tmp_rqi_grib2):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                                      mrms_tmp_rqi_grib2 + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(mrms_tmp_rqi_grib2)
            except:
                ConfigOptions.errMsg = "Unable to remove file: " + mrms_tmp_rqi_grib2
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        if os.path.isfile(mrms_tmp_rqi_nc):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                                      mrms_tmp_rqi_nc + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(mrms_tmp_rqi_nc)
            except:
                ConfigOptions.errMsg = "Unable to remove file: " + mrms_tmp_rqi_nc
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # If the input paths have been set to None, this means input is missing. We will
    # alert the user, and set the final output grids to be the global NDV and return.
    if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "No MRMS Precipitation available. Supplemental precipitation will " \
                                      "not be used."
            errMod.log_msg(ConfigOptions, MpiConfig)
        supplemental_precip.regridded_precip2 = None
        supplemental_precip.regridded_precip1 = None
        return

    # Unzip MRMS files to temporary locations.
    try:
        ioMod.unzip_file(supplemental_precip.file_in2,mrms_tmp_grib2,
                         ConfigOptions,MpiConfig)
    except:
        errMod.log_critical(ConfigOptions, MpiConfig)
    try:
        ioMod.unzip_file(supplemental_precip.rqi_file_in2,mrms_tmp_rqi_grib2,
                         ConfigOptions,MpiConfig)
    except:
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Perform a GRIB dump to NetCDF for the MRMS precip and RQI data.
    cmd1 = "wgrib2 " + mrms_tmp_grib2 + " -netcdf " + mrms_tmp_nc
    idMrms = ioMod.open_grib2(mrms_tmp_grib2,mrms_tmp_nc,cmd1,ConfigOptions,
                              MpiConfig,supplemental_precip.netcdf_var_names[0])
    errMod.check_program_status(ConfigOptions, MpiConfig)

    cmd2 = "wgrib2 " + mrms_tmp_rqi_grib2 + " -netcdf " + mrms_tmp_rqi_nc
    idMrmsRqi = ioMod.open_grib2(mrms_tmp_rqi_grib2, mrms_tmp_rqi_nc, cmd2, ConfigOptions,
                                 MpiConfig, supplemental_precip.rqi_netcdf_var_names[0])
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Remove temporary GRIB2 files
    if MpiConfig.rank == 0:
        try:
            os.remove(mrms_tmp_grib2)
        except:
            ConfigOptions.errMsg = "Unable to remove GRIB2 file: " + mrms_tmp_grib2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            os.remove(mrms_tmp_rqi_grib2)
        except:
            ConfigOptions.errMsg = "Unable to remove GRIB2 file: " + mrms_tmp_rqi_grib2
            errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Check to see if we need to calculate regridding weights.
    calcRegridFlag = check_supp_pcp_regrid_status(idMrms, supplemental_precip,ConfigOptions,
                                                  MpiConfig, wrfHydroGeoMeta)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    if calcRegridFlag:
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Calculating MRMS regridding weights."
            errMod.log_msg(ConfigOptions, MpiConfig)
        calculate_supp_pcp_weights(MpiConfig, ConfigOptions,
                                   supplemental_precip, idMrms,mrms_tmp_nc)
        errMod.check_program_status(ConfigOptions, MpiConfig)

    # Regrid the RQI grid.
    if MpiConfig.rank == 0:
        try:
            varTmp = idMrmsRqi.variables[supplemental_precip.rqi_netcdf_var_names[0]][0, :, :]
        except:
            ConfigOptions.errMsg = "Unable to extract: " + supplemental_precip.rqi_netcdf_var_names[0] + \
                                   " from: " + mrms_tmp_rqi_grib2
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
    else:
        varTmp = None
    errMod.check_program_status(ConfigOptions, MpiConfig)

    varSubTmp = MpiConfig.scatter_array(supplemental_precip, varTmp, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        supplemental_precip.esmf_field_in.data[:, :] = varSubTmp
    except:
        ConfigOptions.errMsg = "Unable to place MRMS data into local ESMF field."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Regridding MRMS RQI Field."
        errMod.log_msg(ConfigOptions, MpiConfig)
    try:
        supplemental_precip.esmf_field_out = supplemental_precip.regridObj(supplemental_precip.esmf_field_in,
                                                                           supplemental_precip.esmf_field_out)
    except:
        ConfigOptions.errMsg = "Unable to regrid MRMS RQI field."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Set any pixel cells outside the input domain to the global missing value.
    try:
        supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = \
            ConfigOptions.globalNdv
    except:
        ConfigOptions.errMsg = "Unable to run mask calculation for MRMS RQI data."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    supplemental_precip.regridded_rqi2[:, :] = supplemental_precip.esmf_field_out.data
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Close the temporary NetCDF file and remove it.
    if MpiConfig.rank == 0:
        try:
            idMrmsRqi.close()
        except:
            ConfigOptions.errMsg = "Unable to close NetCDF file: " + mrms_tmp_rqi_nc
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            os.remove(mrms_tmp_rqi_nc)
        except:
            ConfigOptions.errMsg = "Unable to remove NetCDF file: " + mrms_tmp_rqi_nc
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Regrid the input variables.
    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
        errMod.log_msg(ConfigOptions, MpiConfig)
        try:
            varTmp = idMrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
        except:
            ConfigOptions.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + \
                                   " from: " + mrms_tmp_nc
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
    else:
        varTmp = None
    errMod.check_program_status(ConfigOptions, MpiConfig)

    varSubTmp = MpiConfig.scatter_array(supplemental_precip, varTmp, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    supplemental_precip.esmf_field_in.data[:, :] = varSubTmp
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        supplemental_precip.esmf_field_out = supplemental_precip.regridObj(supplemental_precip.esmf_field_in,
                                                                           supplemental_precip.esmf_field_out)
    except:
        ConfigOptions.errMsg = "Unable to regrid MRMS precipitation"
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Set any pixel cells outside the input domain to the global missing value.
    try:
        supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = \
            ConfigOptions.globalNdv
    except:
        ConfigOptions.errMsg = "Unable to run mask search on MRMS supplemental precip."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    supplemental_precip.regridded_precip2[:, :] = \
        supplemental_precip.esmf_field_out.data
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Check for any RQI values below the threshold specified by the user.
    # Set these values to global NDV.
    try:
        indFilter = np.where(supplemental_precip.regridded_rqi2 <= ConfigOptions.rqiThresh)
    except:
        ConfigOptions.errMsg = "Unable to run MRMS RQI threshold search."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    supplemental_precip.regridded_precip2[indFilter] = ConfigOptions.globalNdv

    # Convert the hourly precipitation total to a rate of mm/s
    try:
        indValid = np.where(supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv)
    except:
        ConfigOptions.errMsg = "Unable to run global NDV search on MRMS regridded precip."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    supplemental_precip.regridded_precip2[indValid] = supplemental_precip.regridded_precip2[indValid]/3600.0
    # Reset index variables to free up memory
    indValid = None
    indFilter = None

    # If we are on the first timestep, set the previous regridded field to be
    # the latest as there are no states for time 0.
    if ConfigOptions.current_output_step == 1:
        supplemental_precip.regridded_precip1[:, :] = \
            supplemental_precip.regridded_precip2[:, :]
        supplemental_precip.regridded_rqi1[:, :] = \
            supplemental_precip.regridded_rqi2[:, :]
    MpiConfig.comm.barrier()

    # Close the temporary NetCDF file and remove it.
    if MpiConfig.rank == 0:
        try:
            idMrms.close()
        except:
            ConfigOptions.errMsg = "Unable to close NetCDF file: " + mrms_tmp_nc
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        try:
            os.remove(mrms_tmp_nc)
        except:
            ConfigOptions.errMsg = "Unable to remove NetCDF file: " + mrms_tmp_nc
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
    errMod.check_program_status(ConfigOptions, MpiConfig)

def regrid_hourly_WRF_ARW_HiRes_PCP(supplemental_precip,ConfigOptions,wrfHydroGeoMeta,MpiConfig):
    """
    Function for handling regridding hourly forecasted ARW precipitation for hi-res nests.
    :param supplemental_precip:
    :param ConfigOptions:
    :param wrfHydroGeoMeta:
    :param MpiConfig:
    :return:
    """
    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        return

    if MpiConfig.rank == 0:
        print("REGRID STATUS = " + str(supplemental_precip.regridComplete))
    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    arw_tmp_nc = ConfigOptions.scratch_dir + "/ARW_PCP_TMP.nc"

    # These files shouldn't exist. If they do, remove them.
    if MpiConfig.rank == 0:
        if os.path.isfile(arw_tmp_nc):
            ConfigOptions.statusMsg = "Found old temporary file: " + \
                                      arw_tmp_nc + " - Removing....."
            errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                os.remove(arw_tmp_nc)
            except:
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # If the input paths have been set to None, this means input is missing. We will
    # alert the user, and set the final output grids to be the global NDV and return.
    if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
        if MpiConfig.rank == 0:
            "NO ARW PRECIP AVAILABLE. SETTING FINAL SUPP GRIDS TO NDV"
        supplemental_precip.regridded_precip2 = None
        supplemental_precip.regridded_precip1 = None
        return
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Create a temporary NetCDF file from the GRIB2 file.
    cmd = "wgrib2 " + supplemental_precip.file_in2 + " -match \":(" + \
          "APCP):(surface):(" + str(supplemental_precip.fcst_hour1) + \
          "-" + str(supplemental_precip.fcst_hour2) + " hour acc fcst):\"" + \
          " -netcdf " + arw_tmp_nc

    idTmp = ioMod.open_grib2(supplemental_precip.file_in2, arw_tmp_nc, cmd,
                             ConfigOptions, MpiConfig, "APCP_surface")
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Check to see if we need to calculate regridding weights.
    calcRegridFlag = check_supp_pcp_regrid_status(idTmp, supplemental_precip,ConfigOptions,
                                                  MpiConfig, wrfHydroGeoMeta)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    if calcRegridFlag:
        if MpiConfig.rank == 0:
            print('CALCULATING ARW WEIGHTS')
        calculate_supp_pcp_weights(MpiConfig, ConfigOptions,
                                   supplemental_precip, idTmp, arw_tmp_nc)
        errMod.check_program_status(ConfigOptions, MpiConfig)

    # Regrid the input variables.
    if MpiConfig.rank == 0:
        print("REGRIDDING: APCP_surface")
        try:
            varTmp = idTmp.variables['APCP_surface'][0, :, :]
        except:
            ConfigOptions.errMsg = "Unable to extract precipitation from WRF ARW file: " + \
                                   supplemental_precip.file_in2
            errMod.log_critical(ConfigOptions, MpiConfig)
    else:
        varTmp = None
    errMod.check_program_status(ConfigOptions, MpiConfig)

    varSubTmp = MpiConfig.scatter_array(supplemental_precip, varTmp, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        supplemental_precip.esmf_field_in.data[:, :] = varSubTmp
    except:
        ConfigOptions.errMsg = "Unable to place WRF ARW precipitation into local ESMF field."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        supplemental_precip.esmf_field_out = supplemental_precip.regridObj(supplemental_precip.esmf_field_in,
                                                                           supplemental_precip.esmf_field_out)
    except:
        ConfigOptions.errMsg = "Unable to regrid WRF ARW supplemental precipitation."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Set any pixel cells outside the input domain to the global missing value.
    try:
        supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = \
            ConfigOptions.globalNdv
    except:
        ConfigOptions.errMsg = "Unable to run mask search on WRF ARW supplemental precipitation."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    supplemental_precip.regridded_precip2[:, :] = \
        supplemental_precip.esmf_field_out.data
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Convert the hourly precipitation total to a rate of mm/s
    try:
        indValid = np.where(supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv)
    except:
        ConfigOptions.errMsg = "Unable to run NDV search on WRF ARW supplemental precipitation."
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    supplemental_precip.regridded_precip2[indValid] = supplemental_precip.regridded_precip2[indValid]/3600.0
    # Reset index variables to free up memory
    indValid = None

    # If we are on the first timestep, set the previous regridded field to be
    # the latest as there are no states for time 0.
    if ConfigOptions.current_output_step == 1:
        supplemental_precip.regridded_precip1[:, :] = \
            supplemental_precip.regridded_precip2[:, :]
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Close the temporary NetCDF file and remove it.
    if MpiConfig.rank == 0:
        try:
            idTmp.close()
        except:
            ConfigOptions.errMsg = "Unable to close NetCDF file: " + arw_tmp_nc
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
        try:
            os.remove(arw_tmp_nc)
        except:
            ConfigOptions.errMsg = "Unable to remove NetCDF file: " + arw_tmp_nc
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass
    errMod.check_program_status(ConfigOptions, MpiConfig)

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
            input_forcings.esmf_field_out = ESMF.Field(wrfHydroGeoMeta.esmf_grid, name=input_forcings.productName + \
                                                                                       'FORCING_REGRIDDED')
        except:
            ConfigOptions.errMsg = "Unable to create " + input_forcings.productName + \
                                   " destination ESMF field object."
            errMod.log_critical(ConfigOptions,MpiConfig)
            return

    # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
    # a new weight file:
    # 1.) This is the first output time step, so we need to calculate a weight file.
    # 2.) The input forcing grid has changed.
    calcRegridFlag = False
    MpiConfig.comm.barrier()

    if input_forcings.nx_global is None or input_forcings.ny_global is None:
        # This is the first timestep.
        # Create out regridded numpy arrays to hold the regridded data.
        input_forcings.regridded_forcings1 = np.empty([8, wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                      np.float32)
        input_forcings.regridded_forcings2 = np.empty([8, wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                      np.float32)

    if MpiConfig.rank == 0:
        if input_forcings.nx_global is None or input_forcings.ny_global is None:
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
    errMod.check_program_status(ConfigOptions,MpiConfig)

    return calcRegridFlag

def check_supp_pcp_regrid_status(idTmp,supplemental_precip,ConfigOptions,MpiConfig,wrfHydroGeoMeta):
    """
    Function for checking to see if regridding weights need to be
    calculated (or recalculated).
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # If the destination ESMF field hasn't been created, create it here.
    if not supplemental_precip.esmf_field_out:
        try:
            supplemental_precip.esmf_field_out = ESMF.Field(wrfHydroGeoMeta.esmf_grid,
                                                            name=supplemental_precip.productName + \
                                                                 'SUPP_PCP_REGRIDDED')
        except:
            ConfigOptions.errMsg = "Unable to create " + supplemental_precip.productName + \
                                   " destination ESMF field object."
            errMod.err_out(ConfigOptions)

    # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
    # a new weight file:
    # 1.) This is the first output time step, so we need to calculate a weight file.
    # 2.) The input forcing grid has changed.
    calcRegridFlag = False

    MpiConfig.comm.barrier()

    if supplemental_precip.nx_global is None or supplemental_precip.ny_global is None:
        # This is the first timestep.
        # Create out regridded numpy arrays to hold the regridded data.
        supplemental_precip.regridded_precip1 = np.empty([wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                      np.float32)
        supplemental_precip.regridded_precip2 = np.empty([wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                      np.float32)
        supplemental_precip.regridded_rqi1 = np.empty([wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                      np.float32)
        supplemental_precip.regridded_rqi2 = np.empty([wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                      np.float32)

    if MpiConfig.rank == 0:
        if supplemental_precip.nx_global is None or supplemental_precip.ny_global is None:
            # This is the first timestep.
            calcRegridFlag = True
        else:
            if MpiConfig.rank == 0:
                if idTmp.variables[supplemental_precip.netcdf_var_names[0]].shape[1] \
                        != supplemental_precip.ny_global and \
                        idTmp.variables[supplemental_precip.netcdf_var_names[0]].shape[2] \
                        != supplemental_precip.nx_global:
                    calcRegridFlag = True

    # We will now check to see if the regridded arrays are still None. This means the fields were set to None
    # earlier for missing data. We need to reset them to nx_global/ny_global where the calcRegridFlag is False.
    if supplemental_precip.regridded_precip2 is None:
        supplemental_precip.regridded_precip2 = np.empty([wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                         np.float32)
    if supplemental_precip.regridded_precip1 is None:
        supplemental_precip.regridded_precip1 = np.empty([wrfHydroGeoMeta.ny_local, wrfHydroGeoMeta.nx_local],
                                                         np.float32)

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
            ConfigOptions.errMsg = "Unable to extract Y shape size from: " + \
                                   input_forcings.netcdf_var_names[forceCount] + " from: " + \
                                   input_forcings.tmpFile
            errMod.log_critical(ConfigOptions,MpiConfig)
        try:
            input_forcings.nx_global = \
                idTmp.variables[input_forcings.netcdf_var_names[forceCount]].shape[2]
        except:
            ConfigOptions.errMsg = "Unable to extract X shape size from: " + \
                                   input_forcings.netcdf_var_names[forceCount] + " from: " + \
                                   input_forcings.tmpFile
            errMod.log_critical(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions,MpiConfig)

    # Broadcast the forcing nx/ny values
    input_forcings.ny_global = MpiConfig.broadcast_parameter(input_forcings.ny_global,
                                                             ConfigOptions)
    errMod.check_program_status(ConfigOptions,MpiConfig)
    input_forcings.nx_global = MpiConfig.broadcast_parameter(input_forcings.nx_global,
                                                             ConfigOptions)
    errMod.check_program_status(ConfigOptions,MpiConfig)

    try:
        input_forcings.esmf_grid_in = ESMF.Grid(np.array([input_forcings.ny_global, input_forcings.nx_global]),
                                                staggerloc=ESMF.StaggerLoc.CENTER,
                                                coord_sys=ESMF.CoordSys.SPH_DEG)
    except:
        ConfigOptions.errMsg = "Unable to create source ESMF grid from temporary file: " + \
                               input_forcings.tmpFile
        errMod.log_critical(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        input_forcings.x_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.x_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.y_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.y_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.nx_local = input_forcings.x_upper_bound - input_forcings.x_lower_bound
        input_forcings.ny_local = input_forcings.y_upper_bound - input_forcings.y_lower_bound
    except:
        ConfigOptions.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary " + \
                               "file: " + input_forcings.tmpFile
        errMod.log_critical(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    if MpiConfig.rank == 0:
        # Process lat/lon values from the GFS grid.
        if len(idTmp.variables['latitude'].shape) == 3:
            # We have 2D grids already in place.
            latTmp = idTmp.variables['latitude'][0, :, :]
            lonTmp = idTmp.variables['longitude'][0, :, :]
        elif len(idTmp.variables['longitude'].shape) == 2:
            # We have 2D grids already in place.
            latTmp = idTmp.variables['latitude'][:, :]
            lonTmp = idTmp.variables['longitude'][:, :]
        elif len(idTmp.variables['latitude'].shape) == 1:
            # We have 1D lat/lons we need to translate into
            # 2D grids.
            latTmp = np.repeat(idTmp.variables['latitude'][:][:, np.newaxis], input_forcings.nx_global, axis=1)
            lonTmp = np.tile(idTmp.variables['longitude'][:], (input_forcings.ny_global, 1))
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Scatter global GFS latitude grid to processors..
    if MpiConfig.rank == 0:
        varTmp = latTmp
    else:
        varTmp = None
    varSubLatTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    if MpiConfig.rank == 0:
        varTmp = lonTmp
    else:
        varTmp = None
    varSubLonTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        input_forcings.esmf_lats = input_forcings.esmf_grid_in.get_coords(1)
    except:
        ConfigOptions.errMsg = "Unable to locate latitude coordinate object within input ESMF grid."
        errMod.log_critical(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        input_forcings.esmf_lons = input_forcings.esmf_grid_in.get_coords(0)
    except:
        ConfigOptions.errMsg = "Unable to locate longitude coordinate object within input ESMF grid."
        errMod.log_critical(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    input_forcings.esmf_lats[:, :] = varSubLatTmp
    input_forcings.esmf_lons[:, :] = varSubLonTmp
    varSubLatTmp = None
    varSubLonTmp = None
    latTmp = None
    lonTmp = None

    # Create a ESMF field to hold the incoming data.
    try:
        input_forcings.esmf_field_in = ESMF.Field(input_forcings.esmf_grid_in,
                                                  name=input_forcings.productName + "_NATIVE")
    except:
        ConfigOptions.errMsg = "Unable to create ESMF field object."
        errMod.log_critical(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Scatter global grid to processors..
    if MpiConfig.rank == 0:
        varTmp = idTmp[input_forcings.netcdf_var_names[forceCount]][0, :, :]
        # Set all valid values to 1.0, and all missing values to 0.0. This will
        # be used to generate an output mask that is used later on in downscaling, layering,
        # etc.
        varTmp[:,:] = 1.0
    else:
        varTmp = None
    varSubTmp = MpiConfig.scatter_array(input_forcings, varTmp, ConfigOptions)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Place temporary data into the field array for generating the regridding object.
    input_forcings.esmf_field_in.data[:, :] = varSubTmp
    MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Creating weight object from ESMF"
        errMod.log_msg(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)
    try:
        input_forcings.regridObj = ESMF.Regrid(input_forcings.esmf_field_in,
                                               input_forcings.esmf_field_out,
                                               src_mask_values=np.array([0]),
                                               regrid_method=ESMF.RegridMethod.BILINEAR,
                                               unmapped_action=ESMF.UnmappedAction.IGNORE)
    except:
        ConfigOptions.errMsg = "Unable to regrid input data from ESMF"
        errMod.log_critical(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Run the regridding object on this test dataset. Check the output grid for
    # any 0 values.
    try:
        input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                 input_forcings.esmf_field_out)
    except:
        ConfigOptions.errMsg = "Unable to extract regridded data from ESMF regridded field."
        errMod.log_critical(ConfigOptions,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    input_forcings.regridded_mask[:, :] = input_forcings.esmf_field_out.data[:, :]

def calculate_supp_pcp_weights(MpiConfig,ConfigOptions,supplemental_precip,idTmp,tmpFile):
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
            supplemental_precip.ny_global = \
                idTmp.variables[supplemental_precip.netcdf_var_names[0]].shape[1]
        except:
            ConfigOptions.errMsg = "Unable to extract Y shape size from: " + \
                                   supplemental_precip.netcdf_var_names[0] + " from: " + \
                                   tmpFile
            errMod.err_out(ConfigOptions)
        try:
            supplemental_precip.nx_global = \
                idTmp.variables[supplemental_precip.netcdf_var_names[0]].shape[2]
        except:
            ConfigOptions.errMsg = "Unable to extract X shape size from: " + \
                                   supplemental_precip.netcdf_var_names[0] + " from: " + \
                                   tmpFile
            errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    # Broadcast the forcing nx/ny values
    supplemental_precip.ny_global = MpiConfig.broadcast_parameter(supplemental_precip.ny_global,
                                                             ConfigOptions)
    supplemental_precip.nx_global = MpiConfig.broadcast_parameter(supplemental_precip.nx_global,
                                                             ConfigOptions)
    MpiConfig.comm.barrier()

    try:
        supplemental_precip.esmf_grid_in = ESMF.Grid(np.array([supplemental_precip.ny_global,
                                                               supplemental_precip.nx_global]),
                                                     staggerloc=ESMF.StaggerLoc.CENTER,
                                                     coord_sys=ESMF.CoordSys.SPH_DEG)
    except:
        ConfigOptions.errMsg = "Unable to create source ESMF grid from temporary file: " + \
                               tmpFile
        errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    try:
        supplemental_precip.x_lower_bound = supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        supplemental_precip.x_upper_bound = supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        supplemental_precip.y_lower_bound = supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        supplemental_precip.y_upper_bound = supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        supplemental_precip.nx_local = supplemental_precip.x_upper_bound - supplemental_precip.x_lower_bound
        supplemental_precip.ny_local = supplemental_precip.y_upper_bound - supplemental_precip.y_lower_bound
    except:
        ConfigOptions.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary " + \
                               "file: " + tmpFile
        errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        # Process lat/lon values from the GFS grid.
        if len(idTmp.variables['latitude'].shape) == 3:
            # We have 2D grids already in place.
            latTmp = idTmp.variables['latitude'][0, :, :]
            lonTmp = idTmp.variables['longitude'][0, :, :]
        elif len(idTmp.variables['longitude'].shape) == 2:
            # We have 2D grids already in place.
            latTmp = idTmp.variables['latitude'][:, :]
            lonTmp = idTmp.variables['longitude'][:, :]
        elif len(idTmp.variables['latitude'].shape) == 1:
            # We have 1D lat/lons we need to translate into
            # 2D grids.
            latTmp = np.repeat(idTmp.variables['latitude'][:][:, np.newaxis], supplemental_precip.nx_global, axis=1)
            lonTmp = np.tile(idTmp.variables['longitude'][:], (supplemental_precip.ny_global, 1))
    MpiConfig.comm.barrier()

    # Scatter global GFS latitude grid to processors..
    if MpiConfig.rank == 0:
        varTmp = latTmp
    else:
        varTmp = None
    varSubLatTmp = MpiConfig.scatter_array(supplemental_precip, varTmp, ConfigOptions)
    MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        varTmp = lonTmp
    else:
        varTmp = None
    varSubLonTmp = MpiConfig.scatter_array(supplemental_precip, varTmp, ConfigOptions)
    MpiConfig.comm.barrier()

    try:
        supplemental_precip.esmf_lats = supplemental_precip.esmf_grid_in.get_coords(1)
    except:
        ConfigOptions.errMsg = "Unable to locate latitude coordinate object within supplemental precip ESMF grid."
        errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    try:
        supplemental_precip.esmf_lons = supplemental_precip.esmf_grid_in.get_coords(0)
    except:
        ConfigOptions.errMsg = "Unable to locate longitude coordinate object within supplemental precip ESMF grid."
        errMod.err_out(ConfigOptions)
    MpiConfig.comm.barrier()

    supplemental_precip.esmf_lats[:, :] = varSubLatTmp
    supplemental_precip.esmf_lons[:, :] = varSubLonTmp
    varSubLatTmp = None
    varSubLonTmp = None
    latTmp = None
    lonTmp = None

    # Create a ESMF field to hold the incoming data.
    supplemental_precip.esmf_field_in = ESMF.Field(supplemental_precip.esmf_grid_in,
                                                   name=supplemental_precip.productName + \
                                                        "_NATIVE")

    MpiConfig.comm.barrier()

    # Scatter global grid to processors..
    if MpiConfig.rank == 0:
        varTmp = idTmp[supplemental_precip.netcdf_var_names[0]][0, :, :]
        # Set all valid values to 1.0, and all missing values to 0.0. This will
        # be used to generate an output mask that is used later on in downscaling, layering,
        # etc.
        varTmp[:,:] = 1.0
    else:
        varTmp = None
    varSubTmp = MpiConfig.scatter_array(supplemental_precip, varTmp, ConfigOptions)
    MpiConfig.comm.barrier()

    # Place temporary data into the field array for generating the regridding object.
    supplemental_precip.esmf_field_in.data[:, :] = varSubTmp
    MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        print("CREATING SUPPLEMENTAL PRECIP REGRID OBJECT")
    supplemental_precip.regridObj = ESMF.Regrid(supplemental_precip.esmf_field_in,
                                                supplemental_precip.esmf_field_out,
                                                src_mask_values=np.array([0]),
                                                regrid_method=ESMF.RegridMethod.BILINEAR,
                                                unmapped_action=ESMF.UnmappedAction.IGNORE)

    # Run the regridding object on this test dataset. Check the output grid for
    # any 0 values.
    supplemental_precip.esmf_field_out = supplemental_precip.regridObj(supplemental_precip.esmf_field_in,
                                                                       supplemental_precip.esmf_field_out)
    supplemental_precip.regridded_mask[:, :] = supplemental_precip.esmf_field_out.data[:, :]