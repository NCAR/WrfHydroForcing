import pytest
from netCDF4 import Dataset

from core.parallel import MpiConfig
from core.config import ConfigOptions
from core.geoMod import GeoMetaWrfHydro

from fixtures import *

import numpy as np


@pytest.mark.mpi
def test_get_processor_bounds(geo_meta,mpi_meta):
    geo_meta.get_processor_bounds()

    assert geo_meta.x_lower_bound == 0
    assert geo_meta.x_upper_bound == geo_meta.nx_global

    assert abs((geo_meta.ny_global / mpi_meta.size) - geo_meta.ny_local) < mpi_meta.size
    assert abs(geo_meta.y_lower_bound - (mpi_meta.rank * geo_meta.ny_local)) < mpi_meta.size
    assert abs(geo_meta.y_upper_bound - ((mpi_meta.rank + 1) * geo_meta.ny_local)) < mpi_meta.size

@pytest.mark.mpi
def test_calc_slope(geo_meta,config_options):
    config_options.read_config()

    idTmp = Dataset(config_options.geogrid,'r')
    (slopeOut,slp_azi) = geo_meta.calc_slope(idTmp,config_options)

    assert slopeOut.shape == slp_azi.shape
    assert slopeOut.size == slp_azi.size

    # test the sums of these grids are as expected,
    # within floating point rounding tolerance
    assert abs(np.sum(slopeOut) - 1527.7666) < 0.001
    assert abs(np.sum(slp_azi) - 58115.258) < 0.001