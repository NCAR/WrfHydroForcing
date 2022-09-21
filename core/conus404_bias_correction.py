from datetime import datetime
from os import path

from netCDF4 import Dataset
import numpy as np

from core import err_handler


def zoom(a, n):
    zoomed = np.empty(shape=[e*n for e in a.shape], dtype=a.dtype)
    for i in range(a.shape[0]):
        base_i = i * n
        for j in range(a.shape[1]):
            base_j = j * n
            for ii in range(n):
                for jj in range(n):
                    zoomed[base_i+ii, base_j+jj] = a[i, j]

    return zoomed


def conus404_daymet_temp_bias_correction(input_forcings, config_options, mpi_config, force_num, geo_meta):
    '''
    Apply CONUS404 Daymet-based temperature bias correction to native WRF-Hydro CONUS grid
    '''

    if mpi_config.rank == 0:
        # determine day-of-year
        jday = input_forcings.fcst_date2.timetuple().tm_yday

        # adjust as necessary for leap year
        is_leap_year = datetime(jday, 12, 31).timetuple().tm_yday == 366
        if not is_leap_year and jday >= 60:
            jday -= 1

        if mpi_config.rank == 0:
            config_options.statusMsg = f"Performing CONUS404/Daymet temperature bias corrrection " \
                                       f"for day {jday} of year {input_forcings.fcst_date2.year}"
            err_handler.log_msg(config_options, mpi_config)

        # open appropriate offset file
        try:
            offset_file_path = path.join(config_options.dScaleParamDirs[0], 'bias_corr_1km',
                                         f'CONUS404_Daymet_1km_Tmean_bias_corr_doy_{jday:03}.nc')

            # read entire grid to root processor
            offset_ds = Dataset(offset_file_path)
            offset_global = offset_ds['conus404_daymet_T_bias_corr'][:]
        except Exception as e:
            # likely here due to missing file
            # TODO: add specific exception(s)
            config_options.errMsg = f"Failed to load Daymet temperature bias offsets - check that file/directory" + \
                                    f"{offset_file_path} exists and is a valid offset file" + str(e)
            err_handler.log_critical(config_options, mpi_config)
    else:
        offset_global = None
    err_handler.check_program_status(config_options, mpi_config)

    # distribute global offsets to each local processor
    offset_local = mpi_config.scatter_array(geo_meta, offset_global, config_options)

    # add offsets to local temperatures
    try:
        temp_local = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
        temp_local += offset_local
    except Exception as e:
        # likely here due to mismatch sizes in slabs, due to grid changing
        # TODO: add specific exception(s)
        config_options.errMsg = "Failed to apply Daymet temperature bias offsets - check that grids are the same size: " + str(e)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def conus404_daymet_precip_bias_correction(input_forcings, config_options, mpi_config, force_num, geo_meta):
    '''
    Apply CONUS404 Daymet-based precipitation bias correction to native WRF-Hydro CONUS grid
    '''

    if mpi_config.rank == 0:
        # determine day-of-year
        jday = input_forcings.fcst_date2.timetuple().tm_yday

        # adjust as necessary for leap year
        is_leap_year = datetime(jday, 12, 31).timetuple().tm_yday == 366
        if not is_leap_year and jday >= 60:
            jday -= 1

        if mpi_config.rank == 0:
            config_options.statusMsg = f"Performing CONUS404/Daymet precipitation bias corrrection " \
                                       f"for day {jday} of year {input_forcings.fcst_date2.year}"
            err_handler.log_msg(config_options, mpi_config)

        # open appropriate offset file
        offset_file_path = path.join(config_options.dScaleParamDirs[0],
                                     'bias_corr_1km',
                                     f'CONUS404_precip_bias_corr_1km_day_of_year_{jday:03}.nc')

        # read entire grid to root processor, enlarging by 4x to map 4km to 1km
        offset_ds = Dataset(offset_file_path)
        offset_global = offset_ds['conus_precip_bias_corr_day_1km'][:]
    else:
        offset_global = None

    # distribute global offsets to each local processor
    offset_local = mpi_config.scatter_array(geo_meta, offset_global, config_options)

    # add offsets to local temperatures
    try:
        precip_local = input_forcings.final_forcings[input_forcings.input_map_output[force_num], :, :]
        precip_local *= 1 + (offset_local/100)
    except Exception as e:
        # likely here due to mismatch sizes in slabs, due to grid changing
        # TODO: add specific exception(s)
        config_options.errMsg = "Failed to apply Daymet precip bias offsets - check that grids are the same ratio:\n" + e
        err_handler.err_out(config_options)
