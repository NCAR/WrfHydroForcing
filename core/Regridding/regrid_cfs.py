import numpy as np

from .regrid_grib2 import RegridGRIB2


class RegridCFS(RegridGRIB2):
    def __init__(self, config_options, wrf_hydro_geo_meta, mpi_config):
        super().__init__("CFSv2", config_options, wrf_hydro_geo_meta, mpi_config)

    def regrid_input(self, input_forcings):
        datasource = self.setup_input(input_forcings)
        self.init_regrid_objects(datasource, input_forcings)

        # only run regrid if NOT using bias correction
        if not self.config.runCfsNldasBiasCorrect:
            self.regrid_fields(datasource, input_forcings)
            input_forcings.coarse_input_forcings1 = input_forcings.coarse_input_forcings2 = None
        else:
            # bias correction will be run later:
            for idx, nc_var in enumerate(input_forcings.netcdf_var_names):
                var_local_tmp = self.extract_field(datasource, input_forcings, nc_var)

                imo_idx = input_forcings.input_map_output[idx]
                input_forcings.regridded_forcings1[imo_idx, :, :] = self.config.globalNdv
                input_forcings.regridded_forcings2[imo_idx, :, :] = self.config.globalNdv
                if input_forcings.coarse_input_forcings1 is None:
                    input_forcings.coarse_input_forcings1 = np.empty([input_forcings.NUM_OUTPUTS,
                                                                      var_local_tmp.shape[0],
                                                                      var_local_tmp.shape[1]],
                                                                     np.float64)

                if input_forcings.coarse_input_forcings2 is None:
                    input_forcings.coarse_input_forcings2 = np.empty_like(input_forcings.coarse_input_forcings1)

                try:
                    input_forcings.coarse_input_forcings2[imo_idx, :, :] = var_local_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    self.log_critical(
                        "Unable to place local {} input field {} into local numpy array:\n\t({})".format(self.name,
                                                                                                         nc_var, err))

                if self.config.current_output_step == 1:
                    input_forcings.coarse_input_forcings1[imo_idx, :, :] = \
                        input_forcings.coarse_input_forcings2[imo_idx, :, :]

        self.release_netcdf(datasource)
