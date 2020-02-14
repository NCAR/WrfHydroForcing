"""
Regridding module file for regridding input forcing files.
"""
import os
import sys
import traceback

import ESMF
import numpy as np

from core import err_handler
from core import ioMod
from core import timeInterpMod


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def regrid_conus_hrrr(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handling regridding of HRRR data.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No HRRR regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config)
        err_handler.log_msg(config_options, mpi_config)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = config_options.scratch_dir + "/" + "HRRR_CONUS_TMP.nc"
    input_forcings.tmpFileHeight = config_options.scratch_dir + "/" + "HRRR_CONUS_TMP_HEIGHT.nc"

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if mpi_config.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            config_options.statusMsg = "Found old temporary file: " + input_forcings.tmpFile + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFile
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    fields = []
    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Converting CONUS HRRR Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)
        fields.append(':' + grib_var + ':' +
                      input_forcings.grib_levels[force_count] + ':'
                      + str(input_forcings.fcst_hour2) + " hour fcst:")

    # Create a temporary NetCDF file from the GRIB2 file.
    cmd = '$WGRIB2 -match "(' + '|'.join(fields) + ')" ' + input_forcings.file_in2 + \
          " -netcdf " + input_forcings.tmpFile
    id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                              config_options, mpi_config, inputVar=None)
    err_handler.check_program_status(config_options, mpi_config)

    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing Conus HRRR Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)

        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating HRRR regridding weights."
                err_handler.log_msg(config_options, mpi_config)
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Read in the HRRR height field, which is used for downscaling purposes.
            if mpi_config.rank == 0:
                config_options.statusMsg = "Reading in HRRR elevation data."
                err_handler.log_msg(config_options, mpi_config)
            cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                "\":(HGT):(surface):\" " + \
                " -netcdf " + input_forcings.tmpFileHeight
            id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                                             cmd, config_options, mpi_config, 'HGT_surface')
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the height variable.
            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_tmp_height.variables['HGT_surface'][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract HRRR elevation from " + \
                                            input_forcings.tmpFileHeight + ": " + str(err)

            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place input NetCDF HRRR data into the ESMF field object: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                err_handler.log_msg(config_options, mpi_config)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid HRRR surface elevation using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to perform HRRR mask search on elevation data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.height[:, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract regridded HRRR elevation data from ESMF: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Close the temporary NetCDF file and remove it.
            if mpi_config.rank == 0:
                try:
                    id_tmp_height.close()
                except OSError:
                    config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except OSError:
                    config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Regrid the input variables.
        var_tmp = None
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing input HRRR variable: " + \
                                       input_forcings.netcdf_var_names[force_count]
            err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + \
                                        " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place input HRRR data into ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding Input HRRR Field: " + input_forcings.netcdf_var_names[force_count]
            err_handler.log_msg(config_options, mpi_config)
        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid input HRRR forcing data: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                config_options.globalNdv
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to perform mask test on regridded HRRR forcings: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.esmf_field_out.data
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract regridded HRRR forcing data from the ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
        # mpi_config.comm.barrier()

    # Close the temporary NetCDF file and remove it.
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
        try:
            os.remove(input_forcings.tmpFile)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
    # mpi_config.comm.barrier()


def regrid_conus_rap(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handling regridding of RAP 13km conus data.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No RAP regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = config_options.scratch_dir + "/" + "RAP_CONUS_TMP.nc"
    input_forcings.tmpFileHeight = config_options.scratch_dir + "/" + "RAP_CONUS_TMP_HEIGHT.nc"

    err_handler.check_program_status(config_options, mpi_config)

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if mpi_config.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            config_options.statusMsg = "Found old temporary file: " + \
                                       input_forcings.tmpFile + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + input_forcings.tmpFile
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    fields = []
    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Converting CONUS RAP Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)
        fields.append(':' + grib_var + ':' +
                      input_forcings.grib_levels[force_count] + ':'
                      + str(input_forcings.fcst_hour2) + " hour fcst:")

    # Create a temporary NetCDF file from the GRIB2 file.
    cmd = '$WGRIB2 -match "(' + '|'.join(fields) + ')" ' + input_forcings.file_in2 + \
          " -netcdf " + input_forcings.tmpFile
    id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                              config_options, mpi_config, inputVar=None)
    err_handler.check_program_status(config_options, mpi_config)

    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing Conus RAP Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)

        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating RAP regridding weights."
                err_handler.log_msg(config_options, mpi_config)
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Read in the RAP height field, which is used for downscaling purposes.
            if mpi_config.rank == 0:
                config_options.statusMsg = "Reading in RAP elevation data."
                err_handler.log_msg(config_options, mpi_config)
            cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                "\":(HGT):(surface):\" " + \
                " -netcdf " + input_forcings.tmpFileHeight
            id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                                             cmd, config_options, mpi_config, 'HGT_surface')
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the height variable.
            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_tmp_height.variables['HGT_surface'][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract HGT_surface from : " + id_tmp_height + \
                                            " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place temporary RAP elevation variable into ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                err_handler.log_msg(config_options, mpi_config)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to perform mask search on RAP elevation data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.height[:, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place RAP ESMF elevation field into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Close the temporary NetCDF file and remove it.
            if mpi_config.rank == 0:
                try:
                    id_tmp_height.close()
                except OSError:
                    config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except OSError:
                    config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        # Regrid the input variables.
        var_tmp = None
        if mpi_config.rank == 0:
            try:
                var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + \
                                        " from: " + input_forcings.tmpFile + \
                                        " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place local RAP array into ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding Input RAP Field: " + input_forcings.netcdf_var_names[force_count]
            err_handler.log_msg(config_options, mpi_config)
        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid RAP variable: " + input_forcings.netcdf_var_names[force_count] \
                                    + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                config_options.globalNdv
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask calculation on RAP variable: " + \
                                    input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.esmf_field_out.data
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place RAP ESMF data into local array: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
        err_handler.check_program_status(config_options, mpi_config)

    # Close the temporary NetCDF file and remove it.
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
        try:
            os.remove(input_forcings.tmpFile)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_cfsv2(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handling regridding of global CFSv2 forecast data.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        # Check to see if we are running NWM-custom interpolation/bias
        # correction on incoming CFSv2 data. Because of the nature, we
        # need to regrid bias-corrected data every hour.
        if mpi_config.rank == 0:
            config_options.statusMsg = "No need to read in new CFSv2 data at this time."
            err_handler.log_msg(config_options, mpi_config)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = config_options.scratch_dir + "/" + "CFSv2_TMP.nc"
    input_forcings.tmpFileHeight = config_options.scratch_dir + "/" + "CFSv2_TMP_HEIGHT.nc"
    err_handler.check_program_status(config_options, mpi_config)

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if mpi_config.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            config_options.statusMsg = "Found old temporary file: " + \
                                       input_forcings.tmpFile + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError as err:
                config_options.errMsg = "Unable to remove previous temporary file: " \
                                        + input_forcings.tmpFile + str(err)
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    fields = []
    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Converting CFSv2 Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)
        fields.append(':' + grib_var + ':' +
                      input_forcings.grib_levels[force_count] + ':'
                      + str(input_forcings.fcst_hour2) + " hour fcst:")

    # Create a temporary NetCDF file from the GRIB2 file.
    cmd = '$WGRIB2 -match "(' + '|'.join(fields) + ')" ' + input_forcings.file_in2 + \
          " -netcdf " + input_forcings.tmpFile
    id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                              config_options, mpi_config, inputVar=None)
    err_handler.check_program_status(config_options, mpi_config)

    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing CFSv2 Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)

        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculate CFSv2 regridding weights."
                err_handler.log_msg(config_options, mpi_config)

            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Read in the RAP height field, which is used for downscaling purposes.
            if mpi_config.rank == 0:
                config_options.statusMsg = "Reading in CFSv2 elevation data."
                err_handler.log_msg(config_options, mpi_config)

            cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                "\":(HGT):(surface):\" " + \
                " -netcdf " + input_forcings.tmpFileHeight
            id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                                             cmd, config_options, mpi_config, 'HGT_surface')
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the height variable.
            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_tmp_height.variables['HGT_surface'][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract HGT_surface from file: " \
                                            + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place CFSv2 elevation data into the ESMF field object: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                err_handler.log_msg(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask calculation on CFSv2 elevation data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.height[:, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract CFSv2 regridded elevation data from ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Close the temporary NetCDF file and remove it.
            if mpi_config.rank == 0:
                try:
                    id_tmp_height.close()
                except OSError:
                    config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                try:
                    os.remove(input_forcings.tmpFileHeight)
                except OSError:
                    config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        # Regrid the input variables.
        var_tmp = None
        if mpi_config.rank == 0:
            if not config_options.runCfsNldasBiasCorrect:
                config_options.statusMsg = "Regridding CFSv2 variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + \
                                       " from file: " + input_forcings.tmpFile + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Scatter the global CFSv2 data to the local processors.
        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        # Assign local CFSv2 data to the input forcing object.. IF..... we are running the
        # bias correction. These grids are interpolated in a separate routine, AFTER bias
        # correction has taken place.
        if config_options.runCfsNldasBiasCorrect:
            if input_forcings.coarse_input_forcings1 is None and input_forcings.coarse_input_forcings2 is None \
                    and config_options.current_output_step == 1:
                # if not np.any(input_forcings.coarse_input_forcings1) and not \
                #        np.any(input_forcings.coarse_input_forcings2) and \
                #        ConfigOptions.current_output_step == 1:
                # We need to create NumPy arrays to hold the CFSv2 global data.
                input_forcings.coarse_input_forcings2 = np.empty([8, var_sub_tmp.shape[0], var_sub_tmp.shape[1]],
                                                                 np.float64)
                input_forcings.coarse_input_forcings1 = np.empty([8, var_sub_tmp.shape[0], var_sub_tmp.shape[1]],
                                                                 np.float64)
            try:
                input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_count], :, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local CFSv2 input variable: " + \
                                        input_forcings.netcdf_var_names[force_count] + \
                                        " into local numpy array. (" + str(err) + ")"

            if config_options.current_output_step == 1:
                input_forcings.coarse_input_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                    input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_count], :, :]
        else:
            input_forcings.coarse_input_forcings2 = None
            input_forcings.coarse_input_forcings1 = None
        err_handler.check_program_status(config_options, mpi_config)

        # Only regrid the current files if we did not specify the NLDAS2 NWM bias correction, which needs to take place
        # first before any regridding can take place. That takes place in the bias-correction routine.
        if not config_options.runCfsNldasBiasCorrect:
            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place CFSv2 forcing data into temporary ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid CFSv2 variable: " + \
                                        input_forcings.netcdf_var_names[force_count] + " (" + str(ve) + ")"
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask calculation on CFSv2 variable: " + \
                                        input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract ESMF field data for CFSv2: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
            err_handler.check_program_status(config_options, mpi_config)
        else:
            # Set regridded arrays to dummy values as they are regridded later in the bias correction routine.
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                config_options.globalNdv
            input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                config_options.globalNdv

    # Close the temporary NetCDF file and remove it.
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            os.remove(input_forcings.tmpFile)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_custom_hourly_netcdf(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handling regridding of custom input NetCDF hourly forcing files.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        return

    if mpi_config.rank == 0:
        config_options.statusMsg = "No Custom Hourly NetCDF regridding required for this timestep."
        err_handler.log_msg(config_options, mpi_config)
    # mpi_config.comm.barrier()

    # Open the input NetCDF file containing necessary data.
    id_tmp = ioMod.open_netcdf_forcing(input_forcings.file_in2, config_options, mpi_config)
    # mpi_config.comm.barrier()

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing Custom NetCDF Forcing Variable: " + \
                                       nc_var
            err_handler.log_msg(config_options, mpi_config)
        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)

        if calc_regrid_flag:
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config)

            # Read in the RAP height field, which is used for downscaling purposes.
            if 'HGT_surface' not in id_tmp.variables.keys():
                config_options.errMsg = "Unable to locate HGT_surface in: " + input_forcings.file_in2
                raise Exception()
            # mpi_config.comm.barrier()

            # Regrid the height variable.
            if mpi_config.rank == 0:
                var_tmp = id_tmp.variables['HGT_surface'][0, :, :]
            else:
                var_tmp = None
            # mpi_config.comm.barrier()

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            # mpi_config.comm.barrier()

            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            # mpi_config.comm.barrier()

            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
            # Set any pixel cells outside the input domain to the global missing value.
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                config_options.globalNdv
            # mpi_config.comm.barrier()

            input_forcings.height[:, :] = input_forcings.esmf_field_out.data
            # mpi_config.comm.barrier()

        # mpi_config.comm.barrier()

        # Regrid the input variables.
        if mpi_config.rank == 0:
            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
        else:
            var_tmp = None
        # mpi_config.comm.barrier()

        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        # mpi_config.comm.barrier()

        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        # mpi_config.comm.barrier()

        input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                 input_forcings.esmf_field_out)
        # Set any pixel cells outside the input domain to the global missing value.
        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
            config_options.globalNdv
        # mpi_config.comm.barrier()

        input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
            input_forcings.esmf_field_out.data
        # mpi_config.comm.barrier()

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
        # mpi_config.comm.barrier()

        # Close the temporary NetCDF file and remove it.
        if mpi_config.rank == 0:
            try:
                id_tmp.close()
            except OSError:
                config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                err_handler.err_out(config_options)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                err_handler.err_out(config_options)

@static_vars(last_file=None)
def regrid_gfs(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handing regridding of input GFS data
    fro GRIB2 files.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No 13km GFS regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = config_options.scratch_dir + "/" + "GFS_TMP.nc"
    input_forcings.tmpFileHeight = config_options.scratch_dir + "/" + "GFS_TMP_HEIGHT.nc"
    err_handler.check_program_status(config_options, mpi_config)

    # check / set previous file to see if we're going to reuse
    reuse_prev_file = (input_forcings.file_in2 == regrid_gfs.last_file)
    regrid_gfs.last_file = input_forcings.file_in2

    # This file may exist. If it does, and we don't need it again, remove it.....
    if not reuse_prev_file and mpi_config.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            config_options.statusMsg = "Found old temporary file: " + \
                                       input_forcings.tmpFile + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + input_forcings.tmpFile
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # We will process each variable at a time. Unfortunately, wgrib2 makes it a bit
    # difficult to handle forecast strings, otherwise this could be done in one command.
    # This makes a compelling case for the use of a GRIB Python API in the future....
    # Incoming shortwave radiation flux.....

    # Loop through all of the input forcings in GFS data. Convert the GRIB2 files
    # to NetCDF, read in the data, regrid it, then map it to the appropriate
    # array slice in the output arrays.

    if reuse_prev_file:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Reusing previous input file: " + input_forcings.file_in2
            err_handler.log_msg(config_options, mpi_config)
        id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)
    else:
        fields = []
        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Converting 13km GFS Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config)
            # Create a temporary NetCDF file from the GRIB2 file.
            if grib_var == "PRATE":
                # By far the most complicated of output variables. We need to calculate
                # our 'average' PRATE based on our current hour.
                if input_forcings.fcst_hour2 <= 240:
                    tmp_hr_current = input_forcings.fcst_hour2

                    diff_tmp = tmp_hr_current % 6 if tmp_hr_current % 6 > 0 else 6
                    tmp_hr_previous = tmp_hr_current - diff_tmp

                else:
                    tmp_hr_previous = input_forcings.fcst_hour1

                fields.append(':' + grib_var + ':' +
                              input_forcings.grib_levels[force_count] + ':' +
                              str(tmp_hr_previous) + '-' + str(input_forcings.fcst_hour2) + " hour ave fcst:")
            else:
                fields.append(':' + grib_var + ':' +
                              input_forcings.grib_levels[force_count] + ':'
                              + str(input_forcings.fcst_hour2) + " hour fcst:")

        cmd = '$WGRIB2 -match "(' + '|'.join(fields) + ')" ' + input_forcings.file_in2 + \
              " -netcdf " + input_forcings.tmpFile
        id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                  config_options, mpi_config, inputVar=None)
        err_handler.check_program_status(config_options, mpi_config)

    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing 13km GFS Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)

        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating 13km GFS regridding weights."
                err_handler.log_msg(config_options, mpi_config)
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Read in the GFS height field, which is used for downscaling purposes.
            if mpi_config.rank == 0:
                config_options.statusMsg = "Reading in 13km GFS elevation data."
                err_handler.log_msg(config_options, mpi_config)
            cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                "\":(HGT):(surface):\" " + \
                " -netcdf " + input_forcings.tmpFileHeight
            id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                                             cmd, config_options, mpi_config, 'HGT_surface')
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the height variable.
            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_tmp_height.variables['HGT_surface'][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract GFS elevation from: " + input_forcings.tmpFileHeight + \
                                            " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local GFS array into an ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding 13km GFS surface elevation data to the WRF-Hydro domain."
                err_handler.log_msg(config_options, mpi_config)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid GFS elevation data: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to perform mask search on GFS elevation data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.height[:, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract GFS elevation array from ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Close the temporary NetCDF file and remove it.
            if mpi_config.rank == 0:
                try:
                    id_tmp_height.close()
                except OSError:
                    config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except OSError:
                    config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        # Regrid the input variables.
        var_tmp = None
        if mpi_config.rank == 0:
            try:
                var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + \
                                       " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # If we are regridding GFS data, and this is precipitation, we need to run calculations
        # on the global precipitation average rates to calculate instantaneous global rates.
        # This is due to GFS's weird nature of doing average rates over different periods.
        if input_forcings.productName == "GFS_Production_GRIB2":
            if grib_var == "PRATE":
                if mpi_config.rank == 0:
                    input_forcings.globalPcpRate2 = var_tmp
                    var_tmp = timeInterpMod.gfs_pcp_time_interp(input_forcings, config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        mpi_config.comm.barrier()
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place GFS local array into ESMF field object: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding Input 13km GFS Field: " + \
                                       input_forcings.netcdf_var_names[force_count]
            err_handler.log_msg(config_options, mpi_config)
        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid GFS variable: " + input_forcings.netcdf_var_names[force_count] \
                                    + " (" + str(ve) + ")"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                config_options.globalNdv
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on GFS variable: " + \
                                    input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.esmf_field_out.data
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract GFS ESMF field data to local array: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
        err_handler.check_program_status(config_options, mpi_config)

    # Close the temporary NetCDF file and remove it.
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)

        # DON'T REMOVE THE FILE, IT WILL EITHER BE REUSED or OVERWRITTEN
        # try:
        #     os.remove(input_forcings.tmpFile)
        # except OSError:
        #     config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
        #     err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_nam_nest(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handing regridding of input NAM nest data
    fro GRIB2 files.
    :param mpi_config:
    :param wrf_hydro_geo_meta:
    :param input_forcings:
    :param config_options:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        config_options.statusMsg = "No regridding of NAM nest data necessary for this timestep - already completed."
        err_handler.log_msg(config_options, mpi_config)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = config_options.scratch_dir + "/" + "NAM_NEST_TMP.nc"
    input_forcings.tmpFileHeight = config_options.scratch_dir + "/" + "NAM_NEST_TMP_HEIGHT.nc"
    err_handler.check_program_status(config_options, mpi_config)

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if mpi_config.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            config_options.statusMsg = "Found old temporary file: " + \
                                       input_forcings.tmpFile + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                err_handler.err_out(config_options)
    err_handler.check_program_status(config_options, mpi_config)

    # Loop through all of the input forcings in NAM nest data. Convert the GRIB2 files
    # to NetCDF, read in the data, regrid it, then map it to the appropriate
    # array slice in the output arrays.
    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing NAM Nest Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)
        # Create a temporary NetCDF file from the GRIB2 file.
        cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match \":(" + \
              grib_var + "):(" + \
              input_forcings.grib_levels[force_count] + "):(" + str(input_forcings.fcst_hour2) + \
              " hour fcst):\" -netcdf " + input_forcings.tmpFile

        id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                  config_options, mpi_config, input_forcings.netcdf_var_names[force_count])
        err_handler.check_program_status(config_options, mpi_config)

        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating NAM nest regridding weights...."
                err_handler.log_msg(config_options, mpi_config)
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Read in the RAP height field, which is used for downscaling purposes.
            if mpi_config.rank == 0:
                config_options.statusMsg = "Reading in NAM nest elevation data from GRIB2."
                err_handler.log_msg(config_options, mpi_config)
            cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                  "\":(HGT):(surface):\" " + \
                  " -netcdf " + input_forcings.tmpFileHeight
            id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                                             cmd, config_options, mpi_config, 'HGT_surface')
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the height variable.
            if mpi_config.rank == 0:
                var_tmp = id_tmp_height.variables['HGT_surface'][0, :, :]
            else:
                var_tmp = None
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " \
                                        + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                err_handler.log_msg(config_options, mpi_config)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain " \
                                        "using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.height[:, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local " \
                                        "array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Close the temporary NetCDF file and remove it.
            if mpi_config.rank == 0:
                try:
                    id_tmp_height.close()
                except OSError:
                    config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except OSError:
                    config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        err_handler.check_program_status(config_options, mpi_config)

        # Regrid the input variables.
        var_tmp = None
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding NAM nest input variable: " + \
                                       input_forcings.netcdf_var_names[force_count]
            err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                       " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                config_options.globalNdv
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.esmf_field_out.data
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
        err_handler.check_program_status(config_options, mpi_config)

        # Close the temporary NetCDF file and remove it.
        if mpi_config.rank == 0:
            try:
                id_tmp.close()
            except OSError:
                config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                err_handler.log_critical(config_options, mpi_config)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_mrms_hourly(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handling regridding hourly MRMS precipitation. An RQI mask file
    Is necessary to filter out poor precipitation estimates.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(supplemental_precip.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MRMS regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config)
        return
    # MRMS data originally is stored as .gz files. We need to compose a series
    # of temporary paths.
    # 1.) The unzipped GRIB2 precipitation file.
    # 2.) The unzipped GRIB2 RQI file.
    # 3.) A temporary NetCDF file that stores the precipitation grid.
    # 4.) A temporary NetCDF file that stores the RQI grid.
    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    mrms_tmp_grib2 = config_options.scratch_dir + "/MRMS_PCP_TMP.grib2"
    mrms_tmp_nc = config_options.scratch_dir + "/MRMS_PCP_TMP.nc"
    mrms_tmp_rqi_grib2 = config_options.scratch_dir + "/MRMS_RQI_TMP.grib2"
    mrms_tmp_rqi_nc = config_options.scratch_dir + "/MRMS_RQI_TMP.nc"
    # mpi_config.comm.barrier()

    # If the input paths have been set to None, this means input is missing. We will
    # alert the user, and set the final output grids to be the global NDV and return.
    if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MRMS Precipitation available. Supplemental precipitation will " \
                                       "not be used."
            err_handler.log_msg(config_options, mpi_config)
        supplemental_precip.regridded_precip2 = None
        supplemental_precip.regridded_precip1 = None
        return

    # These files shouldn't exist. If they do, remove them.
    if mpi_config.rank == 0:
        if os.path.isfile(mrms_tmp_grib2):
            config_options.statusMsg = "Found old temporary file: " + \
                                       mrms_tmp_grib2 + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(mrms_tmp_grib2)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + mrms_tmp_grib2
                err_handler.log_critical(config_options, mpi_config)
        if os.path.isfile(mrms_tmp_nc):
            config_options.statusMsg = "Found old temporary file: " + \
                                       mrms_tmp_nc + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(mrms_tmp_nc)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + mrms_tmp_nc
                err_handler.log_critical(config_options, mpi_config)
        if os.path.isfile(mrms_tmp_rqi_grib2):
            config_options.statusMsg = "Found old temporary file: " + \
                                       mrms_tmp_rqi_grib2 + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(mrms_tmp_rqi_grib2)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + mrms_tmp_rqi_grib2
                err_handler.log_critical(config_options, mpi_config)
        if os.path.isfile(mrms_tmp_rqi_nc):
            config_options.statusMsg = "Found old temporary file: " + \
                                       mrms_tmp_rqi_nc + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(mrms_tmp_rqi_nc)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + mrms_tmp_rqi_nc
                err_handler.log_critical(config_options, mpi_config)

    err_handler.check_program_status(config_options, mpi_config)

    # If the input paths have been set to None, this means input is missing. We will
    # alert the user, and set the final output grids to be the global NDV and return.
    # if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
    #    if MpiConfig.rank == 0:
    #        ConfigOptions.statusMsg = "No MRMS Precipitation available. Supplemental precipitation will " \
    #                                  "not be used."
    #        errMod.log_msg(ConfigOptions, MpiConfig)
    #    supplemental_precip.regridded_precip2 = None
    #    supplemental_precip.regridded_precip1 = None
    #    return

    # Unzip MRMS files to temporary locations.
    ioMod.unzip_file(supplemental_precip.file_in2, mrms_tmp_grib2, config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.rqiMethod == 1:
        ioMod.unzip_file(supplemental_precip.rqi_file_in2, mrms_tmp_rqi_grib2, config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    # Perform a GRIB dump to NetCDF for the MRMS precip and RQI data.
    cmd1 = "$WGRIB2 " + mrms_tmp_grib2 + " -netcdf " + mrms_tmp_nc
    id_mrms = ioMod.open_grib2(mrms_tmp_grib2, mrms_tmp_nc, cmd1, config_options,
                               mpi_config, supplemental_precip.netcdf_var_names[0])
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.rqiMethod == 1:
        cmd2 = "$WGRIB2 " + mrms_tmp_rqi_grib2 + " -netcdf " + mrms_tmp_rqi_nc
        id_mrms_rqi = ioMod.open_grib2(mrms_tmp_rqi_grib2, mrms_tmp_rqi_nc, cmd2, config_options,
                                       mpi_config, supplemental_precip.rqi_netcdf_var_names[0])
        err_handler.check_program_status(config_options, mpi_config)
    else:
        id_mrms_rqi = None

    # Remove temporary GRIB2 files
    if mpi_config.rank == 0:
        try:
            os.remove(mrms_tmp_grib2)
        except OSError:
            config_options.errMsg = "Unable to remove GRIB2 file: " + mrms_tmp_grib2
            err_handler.log_critical(config_options, mpi_config)

        if config_options.rqiMethod == 1:
            try:
                os.remove(mrms_tmp_rqi_grib2)
            except OSError:
                config_options.errMsg = "Unable to remove GRIB2 file: " + mrms_tmp_rqi_grib2
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if we need to calculate regridding weights.
    calc_regrid_flag = check_supp_pcp_regrid_status(id_mrms, supplemental_precip, config_options,
                                                    wrf_hydro_geo_meta, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Calculating MRMS regridding weights."
            err_handler.log_msg(config_options, mpi_config)
        calculate_supp_pcp_weights(supplemental_precip, id_mrms, mrms_tmp_nc, config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    # Regrid the RQI grid.
    if config_options.rqiMethod == 1:
        var_tmp = None
        if mpi_config.rank == 0:
            try:
                var_tmp = id_mrms_rqi.variables[supplemental_precip.rqi_netcdf_var_names[0]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + supplemental_precip.rqi_netcdf_var_names[0] + \
                                        " from: " + mrms_tmp_rqi_grib2 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place MRMS data into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding MRMS RQI Field."
            err_handler.log_msg(config_options, mpi_config)
        try:
            supplemental_precip.esmf_field_out = supplemental_precip.regridObj(supplemental_precip.esmf_field_in,
                                                                               supplemental_precip.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid MRMS RQI field: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = \
                config_options.globalNdv
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask calculation for MRMS RQI data: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    if config_options.rqiMethod == 0:
        # We will set the RQI field to 1.0 here so no MRMS data gets masked out.
        supplemental_precip.regridded_rqi2[:, :] = 1.0

        if mpi_config.rank == 0:
            config_options.statusMsg = "MRMS Will not be filtered using RQI values."
            err_handler.log_msg(config_options, mpi_config)

    elif config_options.rqiMethod == 2:
        # Read in the RQI field from monthly climatological files.
        ioMod.read_rqi_monthly_climo(config_options, mpi_config, supplemental_precip, wrf_hydro_geo_meta)
    elif config_options.rqiMethod == 1:
        # We are using the MRMS RQI field in realtime
        supplemental_precip.regridded_rqi2[:, :] = supplemental_precip.esmf_field_out.data
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.rqiMethod == 1:
        # Close the temporary NetCDF file and remove it.
        if mpi_config.rank == 0:
            try:
                id_mrms_rqi.close()
            except OSError:
                config_options.errMsg = "Unable to close NetCDF file: " + mrms_tmp_rqi_nc
                err_handler.log_critical(config_options, mpi_config)
            try:
                os.remove(mrms_tmp_rqi_nc)
            except OSError:
                config_options.errMsg = "Unable to remove NetCDF file: " + mrms_tmp_rqi_nc
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    # Regrid the input variables.
    var_tmp = None
    if mpi_config.rank == 0:
        config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
        err_handler.log_msg(config_options, mpi_config)
        try:
            var_tmp = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + \
                                    " from: " + mrms_tmp_nc + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.esmf_field_out = supplemental_precip.regridObj(supplemental_precip.esmf_field_in,
                                                                           supplemental_precip.esmf_field_out)
    except ValueError as ve:
        config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Set any pixel cells outside the input domain to the global missing value.
    try:
        supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = \
            config_options.globalNdv
    except (ValueError, ArithmeticError) as npe:
        config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    supplemental_precip.regridded_precip2[:, :] = \
        supplemental_precip.esmf_field_out.data
    err_handler.check_program_status(config_options, mpi_config)

    # Check for any RQI values below the threshold specified by the user.
    # Set these values to global NDV.
    try:
        ind_filter = np.where(supplemental_precip.regridded_rqi2 < config_options.rqiThresh)
        supplemental_precip.regridded_precip2[ind_filter] = config_options.globalNdv
        del ind_filter
    except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
        config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Convert the hourly precipitation total to a rate of mm/s
    try:
        ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
        supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
        del ind_valid
    except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
        config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If we are on the first timestep, set the previous regridded field to be
    # the latest as there are no states for time 0.
    if config_options.current_output_step == 1:
        supplemental_precip.regridded_precip1[:, :] = \
            supplemental_precip.regridded_precip2[:, :]
        supplemental_precip.regridded_rqi1[:, :] = \
            supplemental_precip.regridded_rqi2[:, :]
    # mpi_config.comm.barrier()

    # Close the temporary NetCDF file and remove it.
    if mpi_config.rank == 0:
        try:
            id_mrms.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + mrms_tmp_nc
            err_handler.log_critical(config_options, mpi_config)

        try:
            os.remove(mrms_tmp_nc)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + mrms_tmp_nc
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_hourly_wrf_arw(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
       Function for handing regridding of input NAM nest data
       fro GRIB2 files.
       :param mpi_config:
       :param wrf_hydro_geo_meta:
       :param input_forcings:
       :param config_options:
       :return:
       """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        config_options.statusMsg = "No regridding of WRF-ARW nest data necessary for this timestep - already completed."
        err_handler.log_msg(config_options, mpi_config)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    input_forcings.tmpFile = config_options.scratch_dir + "/" + "ARW_TMP.nc"
    input_forcings.tmpFileHeight = config_options.scratch_dir + "/" + "ARW_TMP_HEIGHT.nc"
    err_handler.check_program_status(config_options, mpi_config)

    # This file shouldn't exist.... but if it does (previously failed
    # execution of the program), remove it.....
    if mpi_config.rank == 0:
        if os.path.isfile(input_forcings.tmpFile):
            config_options.statusMsg = "Found old temporary file: " + \
                                       input_forcings.tmpFile + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                err_handler.err_out(config_options)
    err_handler.check_program_status(config_options, mpi_config)

    # Loop through all of the input forcings in NAM nest data. Convert the GRIB2 files
    # to NetCDF, read in the data, regrid it, then map it to the appropriate
    # array slice in the output arrays.
    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing WRF-ARW Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)
        # Create a temporary NetCDF file from the GRIB2 file.
        var_str = "{}-{} hour acc fcst".format(input_forcings.fcst_hour1, input_forcings.fcst_hour2) \
            if grib_var == 'APCP' else str(input_forcings.fcst_hour2) + " hour fcst"
        cmd = "wgrib2 " + input_forcings.file_in2 + " -match \":(" + \
              grib_var + "):(" + \
              input_forcings.grib_levels[force_count] + "):(" + var_str + \
              '):" -netcdf ' + input_forcings.tmpFile

        id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                  config_options, mpi_config, input_forcings.netcdf_var_names[force_count])
        err_handler.check_program_status(config_options, mpi_config)

        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating WRF-ARW regridding weights...."
                err_handler.log_msg(config_options, mpi_config)
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Read in the RAP height field, which is used for downscaling purposes.
            if mpi_config.rank == 0:
                config_options.statusMsg = "Reading in WRF-ARW elevation data from GRIB2."
                err_handler.log_msg(config_options, mpi_config)
            cmd = "wgrib2 " + input_forcings.file_in2 + " -match " + \
                  "\":(HGT):(surface):\" " + \
                  " -netcdf " + input_forcings.tmpFileHeight
            id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                                             cmd, config_options, mpi_config, 'HGT_surface')
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the height variable.
            if mpi_config.rank == 0:
                var_tmp = id_tmp_height.variables['HGT_surface'][0, :, :]
            else:
                var_tmp = None
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " \
                                        + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                err_handler.log_msg(config_options, mpi_config)
            try:
                input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                         input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain " \
                                        "using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.height[:, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local " \
                                        "array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Close the temporary NetCDF file and remove it.
            if mpi_config.rank == 0:
                try:
                    id_tmp_height.close()
                except OSError:
                    config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)

                try:
                    os.remove(input_forcings.tmpFileHeight)
                except OSError:
                    config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        err_handler.check_program_status(config_options, mpi_config)

        # Regrid the input variables.
        var_tmp = None
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding WRF-ARW input variable: " + \
                                       input_forcings.netcdf_var_names[force_count]
            err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                        " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                config_options.globalNdv
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Convert the hourly precipitation total to a rate of mm/s
        if grib_var == 'APCP':
            try:
                ind_valid = np.where(input_forcings.esmf_field_out.data != config_options.globalNdv)
                input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on WRF ARW precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.esmf_field_out.data
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
        err_handler.check_program_status(config_options, mpi_config)

        # Close the temporary NetCDF file and remove it.
        if mpi_config.rank == 0:
            try:
                id_tmp.close()
            except OSError:
                config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                err_handler.log_critical(config_options, mpi_config)
            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_hourly_wrf_arw_hi_res_pcp(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handling regridding hourly forecasted ARW precipitation for hi-res nests.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(supplemental_precip.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No ARW regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config)
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    arw_tmp_nc = config_options.scratch_dir + "/ARW_PCP_TMP.nc"

    # These files shouldn't exist. If they do, remove them.
    if mpi_config.rank == 0:
        if os.path.isfile(arw_tmp_nc):
            config_options.statusMsg = "Found old temporary file: " + \
                                       arw_tmp_nc + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os.remove(arw_tmp_nc)
            except IOError:
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the input paths have been set to None, this means input is missing. We will
    # alert the user, and set the final output grids to be the global NDV and return.
    # if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
    #    if MpiConfig.rank == 0:
    #        "NO ARW PRECIP AVAILABLE. SETTING FINAL SUPP GRIDS TO NDV"
    #    supplemental_precip.regridded_precip2 = None
    #    supplemental_precip.regridded_precip1 = None
    #    return
    # errMod.check_program_status(ConfigOptions, MpiConfig)

    # Create a temporary NetCDF file from the GRIB2 file.
    cmd = "$WGRIB2 " + supplemental_precip.file_in2 + " -match \":(" + \
          "APCP):(surface):(" + str(supplemental_precip.fcst_hour1) + \
          "-" + str(supplemental_precip.fcst_hour2) + " hour acc fcst):\"" + \
          " -netcdf " + arw_tmp_nc

    id_tmp = ioMod.open_grib2(supplemental_precip.file_in2, arw_tmp_nc, cmd,
                              config_options, mpi_config, "APCP_surface")
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if we need to calculate regridding weights.
    calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options,
                                                    wrf_hydro_geo_meta, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Calculating WRF ARW regridding weights."
            err_handler.log_msg(config_options, mpi_config)
        calculate_supp_pcp_weights(supplemental_precip, id_tmp, arw_tmp_nc, config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    # Regrid the input variables.
    var_tmp = None
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding WRF ARW APCP Precipitation."
            err_handler.log_msg(config_options, mpi_config)
        try:
            var_tmp = id_tmp.variables['APCP_surface'][0, :, :]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract precipitation from WRF ARW file: " + \
                                    supplemental_precip.file_in2 + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to place WRF ARW precipitation into local ESMF field: " + str(err)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.esmf_field_out = supplemental_precip.regridObj(supplemental_precip.esmf_field_in,
                                                                           supplemental_precip.esmf_field_out)
    except ValueError as ve:
        config_options.errMsg = "Unable to regrid WRF ARW supplemental precipitation: " + str(ve)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Set any pixel cells outside the input domain to the global missing value.
    try:
        supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = \
            config_options.globalNdv
    except (ValueError, ArithmeticError) as npe:
        config_options.errMsg = "Unable to run mask search on WRF ARW supplemental precipitation: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    supplemental_precip.regridded_precip2[:, :] = supplemental_precip.esmf_field_out.data
    err_handler.check_program_status(config_options, mpi_config)

    # Convert the hourly precipitation total to a rate of mm/s
    try:
        ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
        supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
        del ind_valid
    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
        config_options.errMsg = "Unable to run NDV search on WRF ARW supplemental precipitation: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If we are on the first timestep, set the previous regridded field to be
    # the latest as there are no states for time 0.
    if config_options.current_output_step == 1:
        supplemental_precip.regridded_precip1[:, :] = \
            supplemental_precip.regridded_precip2[:, :]
    err_handler.check_program_status(config_options, mpi_config)

    # Close the temporary NetCDF file and remove it.
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + arw_tmp_nc
            err_handler.log_critical(config_options, mpi_config)
        try:
            os.remove(arw_tmp_nc)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + arw_tmp_nc
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for checking to see if regridding weights need to be
    calculated (or recalculated).
    :param wrf_hydro_geo_meta:
    :param force_count:
    :param id_tmp:
    :param input_forcings:
    :param config_options:
    :param mpi_config:
    :return:
    """
    # If the destination ESMF field hasn't been created, create it here.
    if not input_forcings.esmf_field_out:
        try:
            input_forcings.esmf_field_out = ESMF.Field(wrf_hydro_geo_meta.esmf_grid,
                                                       name=input_forcings.productName + 'FORCING_REGRIDDED')
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = "Unable to create " + input_forcings.productName + \
                                    " destination ESMF field object: " + str(esmf_error)
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
    # a new weight file:
    # 1.) This is the first output time step, so we need to calculate a weight file.
    # 2.) The input forcing grid has changed.
    calc_regrid_flag = False
    # mpi_config.comm.barrier()

    if input_forcings.nx_global is None or input_forcings.ny_global is None:
        # This is the first timestep.
        # Create out regridded numpy arrays to hold the regridded data.
        input_forcings.regridded_forcings1 = np.empty([8, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                                                      np.float32)
        input_forcings.regridded_forcings2 = np.empty([8, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                                                      np.float32)

    if mpi_config.rank == 0:
        if input_forcings.nx_global is None or input_forcings.ny_global is None:
            # This is the first timestep.
            calc_regrid_flag = True
        else:
            if mpi_config.rank == 0:
                if id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[1] \
                        != input_forcings.ny_global and \
                        id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[2] \
                        != input_forcings.nx_global:
                    calc_regrid_flag = True
    # mpi_config.comm.barrier()

    # Broadcast the flag to the other processors.
    calc_regrid_flag = mpi_config.broadcast_parameter(calc_regrid_flag, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    return calc_regrid_flag


def check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for checking to see if regridding weights need to be
    calculated (or recalculated).
    :param supplemental_precip:
    :param id_tmp:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the destination ESMF field hasn't been created, create it here.
    if not supplemental_precip.esmf_field_out:
        try:
            supplemental_precip.esmf_field_out = ESMF.Field(wrf_hydro_geo_meta.esmf_grid,
                                                            name=supplemental_precip.productName + 'SUPP_PCP_REGRIDDED')
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = "Unable to create " + supplemental_precip.productName + \
                                    " destination ESMF field object: " + str(esmf_error)
            err_handler.err_out(config_options)

    # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
    # a new weight file:
    # 1.) This is the first output time step, so we need to calculate a weight file.
    # 2.) The input forcing grid has changed.
    calc_regrid_flag = False

    # mpi_config.comm.barrier()

    if supplemental_precip.nx_global is None or supplemental_precip.ny_global is None:
        # This is the first timestep.
        # Create out regridded numpy arrays to hold the regridded data.
        supplemental_precip.regridded_precip1 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                                                         np.float32)
        supplemental_precip.regridded_precip2 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                                                         np.float32)
        supplemental_precip.regridded_rqi1 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                                                      np.float32)
        supplemental_precip.regridded_rqi2 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                                                      np.float32)
        supplemental_precip.regridded_rqi1[:, :] = config_options.globalNdv
        supplemental_precip.regridded_rqi2[:, :] = config_options.globalNdv

    if mpi_config.rank == 0:
        if supplemental_precip.nx_global is None or supplemental_precip.ny_global is None:
            # This is the first timestep.
            calc_regrid_flag = True
        else:
            if mpi_config.rank == 0:
                if id_tmp.variables[supplemental_precip.netcdf_var_names[0]].shape[1] \
                        != supplemental_precip.ny_global and \
                        id_tmp.variables[supplemental_precip.netcdf_var_names[0]].shape[2] \
                        != supplemental_precip.nx_global:
                    calc_regrid_flag = True

    # We will now check to see if the regridded arrays are still None. This means the fields were set to None
    # earlier for missing data. We need to reset them to nx_global/ny_global where the calc_regrid_flag is False.
    if supplemental_precip.regridded_precip2 is None:
        supplemental_precip.regridded_precip2 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                                                         np.float32)
    if supplemental_precip.regridded_precip1 is None:
        supplemental_precip.regridded_precip1 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                                                         np.float32)

    # mpi_config.comm.barrier()

    # Broadcast the flag to the other processors.
    calc_regrid_flag = mpi_config.broadcast_parameter(calc_regrid_flag, config_options)

    mpi_config.comm.barrier()
    return calc_regrid_flag


def calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config):
    """
    Function to calculate ESMF weights based on the output ESMF
    field previously calculated, along with input lat/lon grids,
    and a sample dataset.
    :param input_forcings:
    :param id_tmp:
    :param mpi_config:
    :param config_options:
    :param force_count:
    :return:
    """
    if mpi_config.rank == 0:
        try:
            input_forcings.ny_global = id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[1]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract Y shape size from: " + \
                                    input_forcings.netcdf_var_names[force_count] + " from: " + \
                                    input_forcings.tmpFile + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
        try:
            input_forcings.nx_global = id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[2]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract X shape size from: " + \
                                    input_forcings.netcdf_var_names[force_count] + " from: " + \
                                    input_forcings.tmpFile + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Broadcast the forcing nx/ny values
    input_forcings.ny_global = mpi_config.broadcast_parameter(input_forcings.ny_global,
                                                              config_options)
    err_handler.check_program_status(config_options, mpi_config)
    input_forcings.nx_global = mpi_config.broadcast_parameter(input_forcings.nx_global,
                                                              config_options)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_grid_in = ESMF.Grid(np.array([input_forcings.ny_global, input_forcings.nx_global]),
                                                staggerloc=ESMF.StaggerLoc.CENTER,
                                                coord_sys=ESMF.CoordSys.SPH_DEG)
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = "Unable to create source ESMF grid from temporary file: " + \
                                input_forcings.tmpFile + " (" + str(esmf_error) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.x_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.x_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.y_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.y_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.nx_local = input_forcings.x_upper_bound - input_forcings.x_lower_bound
        input_forcings.ny_local = input_forcings.y_upper_bound - input_forcings.y_lower_bound
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary " + \
                                "file: " + input_forcings.tmpFile + " (" + str(err) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to make sure we have enough dimensionality to run regridding. ESMF requires both grids
    # to have a size of at least 2.
    if input_forcings.nx_local < 2 or input_forcings.ny_local < 2:
        config_options.errMsg = "You have either specified too many cores for: " + input_forcings.productName + \
                               ", or  your input forcing grid is too small to process. Local grid must " \
                               "have x/y dimension size of 2."
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    lat_tmp = None
    lon_tmp = None
    if mpi_config.rank == 0:
        # Process lat/lon values from the GFS grid.
        if len(id_tmp.variables['latitude'].shape) == 3:
            # We have 2D grids already in place.
            lat_tmp = id_tmp.variables['latitude'][0, :, :]
            lon_tmp = id_tmp.variables['longitude'][0, :, :]
        elif len(id_tmp.variables['longitude'].shape) == 2:
            # We have 2D grids already in place.
            lat_tmp = id_tmp.variables['latitude'][:, :]
            lon_tmp = id_tmp.variables['longitude'][:, :]
        elif len(id_tmp.variables['latitude'].shape) == 1:
            # We have 1D lat/lons we need to translate into
            # 2D grids.
            lat_tmp = np.repeat(id_tmp.variables['latitude'][:][:, np.newaxis], input_forcings.nx_global, axis=1)
            lon_tmp = np.tile(id_tmp.variables['longitude'][:], (input_forcings.ny_global, 1))
    err_handler.check_program_status(config_options, mpi_config)

    # Scatter global GFS latitude grid to processors..
    if mpi_config.rank == 0:
        var_tmp = lat_tmp
    else:
        var_tmp = None
    var_sub_lat_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        var_tmp = lon_tmp
    else:
        var_tmp = None
    var_sub_lon_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_lats = input_forcings.esmf_grid_in.get_coords(1)
    except ESMF.GridException as ge:
        config_options.errMsg = "Unable to locate latitude coordinate object within input ESMF grid: " + str(ge)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_lons = input_forcings.esmf_grid_in.get_coords(0)
    except ESMF.GridException as ge:
        config_options.errMsg = "Unable to locate longitude coordinate object within input ESMF grid: " + str(ge)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    input_forcings.esmf_lats[:, :] = var_sub_lat_tmp
    input_forcings.esmf_lons[:, :] = var_sub_lon_tmp
    del var_sub_lat_tmp
    del var_sub_lon_tmp
    del lat_tmp
    del lon_tmp

    # Create a ESMF field to hold the incoming data.
    try:
        input_forcings.esmf_field_in = ESMF.Field(input_forcings.esmf_grid_in,
                                                  name=input_forcings.productName + "_NATIVE")
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = "Unable to create ESMF field object: " + str(esmf_error)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Scatter global grid to processors..
    if mpi_config.rank == 0:
        var_tmp = id_tmp[input_forcings.netcdf_var_names[force_count]][0, :, :]
        # Set all valid values to 1.0, and all missing values to 0.0. This will
        # be used to generate an output mask that is used later on in downscaling, layering,
        # etc.
        var_tmp[:, :] = 1.0
    else:
        var_tmp = None
    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    # Place temporary data into the field array for generating the regridding object.
    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
    # mpi_config.comm.barrier()

    if mpi_config.rank == 0:
        config_options.statusMsg = "Creating weight object from ESMF"
        err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)
    try:
        input_forcings.regridObj = ESMF.Regrid(input_forcings.esmf_field_in,
                                               input_forcings.esmf_field_out,
                                               src_mask_values=np.array([0]),
                                               regrid_method=ESMF.RegridMethod.BILINEAR,
                                               unmapped_action=ESMF.UnmappedAction.IGNORE)
    except (RuntimeError, ImportError, ESMF.ESMPyException) as esmf_error:
        config_options.errMsg = "Unable to regrid input data from ESMF: " + str(esmf_error)
        err_handler.log_critical(config_options, mpi_config)
        etype, value, tb = sys.exc_info()
        traceback.print_exception(etype, value, tb)
        print(input_forcings.esmf_field_in)
        print(input_forcings.esmf_field_out)
        print(np.array([0]))  
        
    err_handler.check_program_status(config_options, mpi_config)

    # Run the regridding object on this test dataset. Check the output grid for
    # any 0 values.
    try:
        input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                 input_forcings.esmf_field_out)
    except ValueError as ve:
        config_options.errMsg = "Unable to extract regridded data from ESMF regridded field: " + str(ve)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    input_forcings.regridded_mask[:, :] = input_forcings.esmf_field_out.data[:, :]


def calculate_supp_pcp_weights(supplemental_precip, id_tmp, tmp_file, config_options, mpi_config):
    """
    Function to calculate ESMF weights based on the output ESMF
    field previously calculated, along with input lat/lon grids,
    and a sample dataset.
    :param tmp_file:
    :param id_tmp:
    :param supplemental_precip:
    :param mpi_config:
    :param config_options:
    :return:
    """
    if mpi_config.rank == 0:
        try:
            supplemental_precip.ny_global = id_tmp.variables[supplemental_precip.netcdf_var_names[0]].shape[1]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract Y shape size from: " + \
                                    supplemental_precip.netcdf_var_names[0] + " from: " + \
                                    tmp_file + " (" + str(err) + ")"
            err_handler.err_out(config_options)
        try:
            supplemental_precip.nx_global = id_tmp.variables[supplemental_precip.netcdf_var_names[0]].shape[2]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract X shape size from: " + \
                                    supplemental_precip.netcdf_var_names[0] + " from: " + \
                                    tmp_file + " (" + str(err) + ")"
            err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    # Broadcast the forcing nx/ny values
    supplemental_precip.ny_global = mpi_config.broadcast_parameter(supplemental_precip.ny_global,
                                                                   config_options)
    supplemental_precip.nx_global = mpi_config.broadcast_parameter(supplemental_precip.nx_global,
                                                                   config_options)
    # mpi_config.comm.barrier()

    try:
        supplemental_precip.esmf_grid_in = ESMF.Grid(np.array([supplemental_precip.ny_global,
                                                               supplemental_precip.nx_global]),
                                                     staggerloc=ESMF.StaggerLoc.CENTER,
                                                     coord_sys=ESMF.CoordSys.SPH_DEG)
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = "Unable to create source ESMF grid from temporary file: " + \
                                tmp_file + " (" + str(esmf_error) + ")"
        err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    try:
        supplemental_precip.x_lower_bound = supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        supplemental_precip.x_upper_bound = supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        supplemental_precip.y_lower_bound = supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        supplemental_precip.y_upper_bound = supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        supplemental_precip.nx_local = supplemental_precip.x_upper_bound - supplemental_precip.x_lower_bound
        supplemental_precip.ny_local = supplemental_precip.y_upper_bound - supplemental_precip.y_lower_bound
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary " + \
                               "file: " + tmp_file + " (" + str(err) + ")"
        err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    # Check to make sure we have enough dimensionality to run regridding. ESMF requires both grids
    # to have a size of at least 2.
    if supplemental_precip.nx_local < 2 or supplemental_precip.ny_local < 2:
        config_options.errMsg = "You have either specified too many cores for: " + supplemental_precip.productName + \
                               ", or  your input forcing grid is too small to process. Local grid " \
                               "must have x/y dimension size of 2."
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    lat_tmp = lon_tmp = None
    if mpi_config.rank == 0:
        # Process lat/lon values from the GFS grid.
        if len(id_tmp.variables['latitude'].shape) == 3:
            # We have 2D grids already in place.
            lat_tmp = id_tmp.variables['latitude'][0, :, :]
            lon_tmp = id_tmp.variables['longitude'][0, :, :]
        elif len(id_tmp.variables['longitude'].shape) == 2:
            # We have 2D grids already in place.
            lat_tmp = id_tmp.variables['latitude'][:, :]
            lon_tmp = id_tmp.variables['longitude'][:, :]
        elif len(id_tmp.variables['latitude'].shape) == 1:
            # We have 1D lat/lons we need to translate into
            # 2D grids.
            lat_tmp = np.repeat(id_tmp.variables['latitude'][:][:, np.newaxis], supplemental_precip.nx_global, axis=1)
            lon_tmp = np.tile(id_tmp.variables['longitude'][:], (supplemental_precip.ny_global, 1))
    # mpi_config.comm.barrier()

    # Scatter global GFS latitude grid to processors..
    if mpi_config.rank == 0:
        var_tmp = lat_tmp
    else:
        var_tmp = None
    var_sub_lat_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
    # mpi_config.comm.barrier()

    if mpi_config.rank == 0:
        var_tmp = lon_tmp
    else:
        var_tmp = None
    var_sub_lon_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
    # mpi_config.comm.barrier()

    try:
        supplemental_precip.esmf_lats = supplemental_precip.esmf_grid_in.get_coords(1)
    except ESMF.GridException as ge:
        config_options.errMsg = "Unable to locate latitude coordinate object within supplemental precip ESMF grid: " \
                                + str(ge)
        err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    try:
        supplemental_precip.esmf_lons = supplemental_precip.esmf_grid_in.get_coords(0)
    except ESMF.GridException as ge:
        config_options.errMsg = "Unable to locate longitude coordinate object within supplemental precip ESMF grid: " \
                                + str(ge)
        err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    supplemental_precip.esmf_lats[:, :] = var_sub_lat_tmp
    supplemental_precip.esmf_lons[:, :] = var_sub_lon_tmp
    del var_sub_lat_tmp
    del var_sub_lon_tmp
    del lat_tmp
    del lon_tmp

    # Create a ESMF field to hold the incoming data.
    supplemental_precip.esmf_field_in = ESMF.Field(supplemental_precip.esmf_grid_in,
                                                   name=supplemental_precip.productName + "_NATIVE")

    # mpi_config.comm.barrier()

    # Scatter global grid to processors..
    if mpi_config.rank == 0:
        var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][0, :, :]
        # Set all valid values to 1.0, and all missing values to 0.0. This will
        # be used to generate an output mask that is used later on in downscaling, layering,
        # etc.
        var_tmp[:, :] = 1.0
    else:
        var_tmp = None
    var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
    mpi_config.comm.barrier()

    # Place temporary data into the field array for generating the regridding object.
    supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
    # mpi_config.comm.barrier()

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
