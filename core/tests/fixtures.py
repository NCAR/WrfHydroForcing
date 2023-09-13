import pytest

from core.parallel import MpiConfig
from core.config import ConfigOptions
from core.geoMod import GeoMetaWrfHydro
from core.forcingInputMod import input_forcings, initDict

import numpy as np


def isWithinTolerance(val1, val2, tolerance=None):
    if tolerance is None:
        tolerance = 0.0
    
    diff = abs(val1 - val2)
    return diff <= tolerance

@pytest.fixture
def mpi_meta(config_options):
    config_options.read_config()

    mpi_meta = MpiConfig()
    mpi_meta.initialize_comm(config_options)

    return mpi_meta

@pytest.fixture
def geo_meta(config_options, mpi_meta):
    config_options.read_config()

    WrfHydroGeoMeta = GeoMetaWrfHydro()
    WrfHydroGeoMeta.initialize_destination_geo(config_options, mpi_meta)
    WrfHydroGeoMeta.initialize_geospatial_metadata(config_options, mpi_meta)

    return WrfHydroGeoMeta

@pytest.fixture
def config_options():
    config_path = './yaml/configOptions_valid.yaml'
    yield ConfigOptions(config_path)

@pytest.fixture
def input_forcings_final(config_options,geo_meta):
    config_options.read_config()

    inputForcingMod = initDict(config_options,geo_meta)
    input_forcings = inputForcingMod[config_options.input_forcings[0]]

    input_forcings.final_forcings = np.full((10,geo_meta.ny_local, geo_meta.nx_local),1.5)
    input_forcings.fcst_hour2 = 12

    return input_forcings

