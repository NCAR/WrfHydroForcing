from .regrid_grib2 import RegridGRIB2


class RegridHRRR(RegridGRIB2):
    def __init__(self, config_options, wrf_hydro_geo_meta, mpi_config):
        super().__init__("CONUS HRRR", config_options, wrf_hydro_geo_meta, mpi_config)
