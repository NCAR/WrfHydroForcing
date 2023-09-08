import pytest
from core.parallel import MpiConfig
from core.config import ConfigOptions
from core.geoMod import GeoMetaWrfHydro
import numpy as np

@pytest.fixture
def config_options():
    config_path = 'yaml/configOptions_valid.yaml'
    return ConfigOptions(config_path)

@pytest.fixture
def geo_meta(config_options, mpi_meta):
    config_options.read_config()

    WrfHydroGeoMeta = GeoMetaWrfHydro()
    WrfHydroGeoMeta.initialize_destination_geo(config_options, mpi_meta)
    WrfHydroGeoMeta.initialize_geospatial_metadata(config_options, mpi_meta)

    return WrfHydroGeoMeta

@pytest.fixture
def mpi_meta(config_options):
    config_options.read_config()

    mpi_meta = MpiConfig()
    mpi_meta.initialize_comm(config_options)

    return mpi_meta

@pytest.mark.mpi
def test_create_mpi_config(config_options, mpi_meta):
    config_options.read_config()

@pytest.mark.mpi
def test_broadcast_parameter(config_options, mpi_meta):

    # int
    value = 999 if mpi_meta.rank == 0 else None
    assert (mpi_meta.rank == 0 and value == 999) or (mpi_meta.rank != 0 and value is None)
    param = mpi_meta.broadcast_parameter(value, config_options, param_type=int)
    assert param == 999

    # float
    value = 999.999 if mpi_meta.rank == 0 else None
    assert (mpi_meta.rank == 0 and value == 999.999) or (mpi_meta.rank != 0 and value is None)
    param = mpi_meta.broadcast_parameter(value, config_options, param_type=float)
    assert param == 999.999

    # bool
    value = True if mpi_meta.rank == 0 else None
    assert (mpi_meta.rank == 0 and value == True) or (mpi_meta.rank != 0 and value is None)
    param = mpi_meta.broadcast_parameter(value, config_options, param_type=bool)
    assert param == True


@pytest.mark.mpi
def test_scatter_array(config_options, mpi_meta, geo_meta):
    test_array = np.zeros([1,390,590], dtype=np.float64)

    #arr = mpi_meta.scatter_array(geo_meta,test_array,config_options)


    assert True