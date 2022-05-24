import os

from core import file_io
from core.regridding.regrid_base import Regridder


class RegridCustomHourlyNetCDF(Regridder):
    def __init__(self, config_options, wrf_hydro_geo_meta, mpi_config):
        super().__init__("Custom Hourly NetCDF", config_options, wrf_hydro_geo_meta, mpi_config)

        # TODO: Move this to regrid_base if possible
        self._did_init_regrid = True

    def regrid_input(self, input_forcings):
        if self.already_processed(input_forcings):
            return

        # Open the input NetCDF file containing necessary data.
        datasource = file_io.open_netcdf_forcing(input_forcings.file_in2, self.config, self.mpi)

        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            self.log_status("Processing Custom NetCDF Forcing Variable: ".format(nc_var))

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
