import os

import numpy as np

from .regrid_base import Regridder
from core import file_io


class RegridGRIB2(Regridder):
    def __init__(self, regridder_name: str, config_options, wrf_hydro_geo_meta, mpi_config):
        super().__init__(regridder_name, config_options, wrf_hydro_geo_meta, mpi_config)

        # TODO: Move this to regrid_base if possible
        self._did_init_regrid = True

    def regrid_input(self, input_forcings):
        datasource = self.setup_input(input_forcings)
        self.init_regrid_objects(datasource, input_forcings)
        self.regrid_fields(datasource, input_forcings)
        self.release_netcdf(datasource)

    def setup_input(self, input_forcings):
        if self.already_processed(input_forcings):
            return

        # Create a path for a temporary NetCDF files that will
        # be created through the wgrib2 process.
        input_forcings.tmpFile = os.path.join(self.config.scratch_dir, "{}_TMP.nc".format(self.name,
                                                                                          self.mkfilename()))

        # This file shouldn't exist, but if it does (previously failed execution of the program), remove it...
        if self.IS_MPI_ROOT:
            if os.path.isfile(input_forcings.tmpFile):
                self.log_status("Found old temporary file: {} - Removing...".format(input_forcings.tmpFile))
                try:
                    os.remove(input_forcings.tmpFile)
                except OSError:
                    self.log_critical("Unable to remove temporary file: " + input_forcings.tmpFile)

        fields = []
        for forcing_idx, grib_var in enumerate(input_forcings.grib_vars):
            self.log_status("Converting {} Variable: {}".format(self.name, grib_var), root_only=True)
            fields.append(':' + grib_var + ':' +
                          input_forcings.grib_levels[forcing_idx] + ':'
                          + str(input_forcings.fcst_hour2) + " hour fcst:")
        fields.append(":(HGT):(surface):")  # add HGT variable to main set (no separate file)

        with self.parallel_block():
            # Create a temporary NetCDF file from the GRIB2 file.
            cmd = '$WGRIB2 -match "(' + '|'.join(fields) + ')" ' + input_forcings.file_in2 + \
                  " -netcdf " + input_forcings.tmpFile

            datasource = file_io.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                            self.config, self.mpi, inputVar=None)

        return datasource

    def init_regrid_objects(self, datasource, input_forcings):
        # TODO: why is there a loop here? Don't we only need one Regrid() object?
        # TODO: move this to regrid_base if possible
        for forcing_idx, grib_var in enumerate(input_forcings.grib_vars):
            # self.log_status("Processing {} Variable: {}".format(self.name, grib_var), root_only=True)

            if self.need_regrid_object(datasource, forcing_idx, input_forcings):
                with self.parallel_block():
                    self.log_status("Calculating {} regridding weights".format(self.name), root_only=True)
                    self.calculate_weights(datasource, forcing_idx, input_forcings)

                # Regrid the height variable.
                with self.parallel_block():
                    var_tmp = None
                    if self.IS_MPI_ROOT:
                        try:
                            var_tmp = datasource.variables['HGT_surface'][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            self.log_critical("Unable to extract {} elevation from {}: {}".format(
                                    self.name, input_forcings.tmpFile, err))

                        var_local_tmp = self.mpi.scatter_array(input_forcings, var_tmp, self.config)

                with self.parallel_block():
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_local_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        self.log_critical("Unable to place input {} data into ESMF field: {}".format(self.name, err))

                with self.parallel_block():
                    self.log_status("Regridding {} surface elevation data to the WRF-Hydro domain.".format(self.name),
                                    root_only=True)
                    try:
                        input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                                 input_forcings.esmf_field_out)
                    except ValueError as ve:
                        self.log_critical("Unable to regrid {} surface elevation using ESMF: {}".format(self.name, ve))

                # Set any pixel cells outside the input domain to the global missing value.
                with self.parallel_block():
                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                            self.config.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        self.log_critical(
                                "Unable to perform {} mask search on elevation data: {}".format(self.name, npe))

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        self.log_critical(
                                "Unable to extract regridded {} elevation data from ESMF: {}".format(self.name, err))

                self._did_init_regrid = True

    def regrid_fields(self, datasource, input_forcings):

        if not self._did_init_regrid:
            raise RuntimeError("Attempting to regrid {} without ESMF regridding objects initialized".format(self.name))

        for forcing_idx, nc_var in enumerate(input_forcings.netcdf_var_names):
            # EXTRACT AND REGRID THE INPUT FIELDS:
            var_local_tmp = self.extract_field(datasource, input_forcings, nc_var)

            with self.parallel_block():
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_local_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    self.log_critical("Unable to place input {} data into ESMF field: {}".format(self.name, err))

            with self.parallel_block():
                self.log_status("Regridding {} input field: {}".format(self.name, nc_var), root_only=True)
                try:
                    input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                             input_forcings.esmf_field_out)
                except ValueError as ve:
                    self.log_critical("Unable to regrid {} input forcing data: {}".format(self.name, ve))

            # Set any pixel cells outside the input domain to the global missing value.
            with self.parallel_block():
                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        self.config.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    self.log_critical("Unable to perform mask test on regridded {} forcings: {}".format(self.name, npe))

            with self.parallel_block():
                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[forcing_idx], :, :] = \
                        input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    self.log_critical(
                            "Unable to extract regridded {} forcing data from ESMF field: {}".format(self.name, err))

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if self.config.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[forcing_idx], :, :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[forcing_idx], :, :]

    def release_netcdf(self, datasource):
        # Close the temporary NetCDF file and remove it.
        if self.IS_MPI_ROOT:
            ds_path = datasource.filepath()
            try:
                datasource.close()
            except OSError:
                self.log_critical("Unable to close NetCDF file: {}".format(ds_path))

            try:
                os.remove(ds_path)
            except OSError:
                # TODO: could this be just a warning?
                self.log_critical("Unable to remove NetCDF file: {}".format(ds_path))

    def extract_field(self, datasource, input_forcings, nc_var):
        var_tmp = None
        with self.parallel_block():
            if self.IS_MPI_ROOT:
                self.log_status("Processing input {} variable: {}".format(self.name, nc_var))
                try:
                    var_tmp = datasource.variables[nc_var][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    self.log_critical(
                            "Unable to extract {} from {} ({})".format(nc_var, input_forcings.tmpFile, err))

            var_local_tmp = self.mpi.scatter_array(input_forcings, var_tmp, self.config)
        return var_local_tmp
