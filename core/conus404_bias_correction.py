from datetime import datetime
from os import path

from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma

from core import err_handler


def conus404_daymet_temp_bias_correction(input_forcings, config_options, mpi_config, force_num, geo_meta):
    apply_daymet_bias_correction(input_forcings, config_options, mpi_config, force_num, geo_meta,
                                 "temperature", "Tmean", lambda x, y: x + y)


def conus404_daymet_precip_bias_correction(input_forcings, config_options, mpi_config, force_num, geo_meta):
    apply_daymet_bias_correction(input_forcings, config_options, mpi_config, force_num, geo_meta,
                                 "precipitation", "precip", lambda x, y: x * (1 + (y/100)))


def apply_daymet_bias_correction(input_forcings, config_options, mpi_config, force_num, geo_meta, description, variable, operator):
    '''
    Apply CONUS404 Daymet-based temperature bias correction to native WRF-Hydro CONUS grid
    '''

    if mpi_config.rank == 0:
        # determine day-of-year
        jday = input_forcings.fcst_date1.timetuple().tm_yday

        # adjust as necessary for leap year
        adjusted = False
        leap_year = datetime(input_forcings.fcst_date1.year, 12, 31).timetuple().tm_yday == 366
        if (not leap_year) and jday >= 60:
            adjusted = True
            jday -= 1

        if mpi_config.rank == 0:
            config_options.statusMsg = f"Performing CONUS404/Daymet {description} bias corrrection " \
                                       f"for day {jday + 1 if adjusted else jday} of year {input_forcings.fcst_date1.year}"
            err_handler.log_msg(config_options, mpi_config)

        # open appropriate offset file
        try:
            offset_file_path = path.join(config_options.dScaleParamDirs[0], 'bias_corr_1km',
                                         f'CONUS404_Daymet_1km_{variable}_bias_corr_doy_{jday:03}.nc')

            # read entire grid to root processor
            offset_ds = Dataset(offset_file_path)
            offset_var = offset_ds[f'conus404_daymet_{variable}_bias_corr']
            offset_global = offset_var[:]
            offset_missing = offset_var.getncattr('_FillValue')
        except Exception as e:
            # likely here due to missing file
            # TODO: add specific exception(s)
            config_options.errMsg = f"Failed to load Daymet {description} bias offsets - check that file/directory " + \
                                    f"{offset_file_path} exists and is a valid offset file ({str(e)})"
            err_handler.log_critical(config_options, mpi_config)
    else:
        offset_global = None
        offset_missing = -9999
    err_handler.check_program_status(config_options, mpi_config)

    try:
        # distribute missing value and global offsets to each local processor
        offset_missing = mpi_config.broadcast_parameter(offset_missing, config_options, param_type=np.dtype('float32'))
        offset_local = mpi_config.scatter_array(geo_meta, offset_global, config_options)
        offset_local = ma.masked_where(offset_local == offset_missing, offset_local)

        # add offsets to local temperatures

        var_local = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
        input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :] = operator(var_local, offset_local)
    except Exception as e:
        # likely here due to mismatch sizes in slabs, due to grid changing
        # TODO: add specific exception(s)
        config_options.errMsg = f"Failed to apply Daymet {description} bias offsets - check that grids are the same size ({str(e)})"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)
