import pytest

from core.parallel import MpiConfig
from core.config import ConfigOptions
from core.geoMod import GeoMetaWrfHydro
from core.forcingInputMod import initDict

import numpy as np


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
    config_options = ConfigOptions(config_path)
    config_options.read_config()
    yield config_options

@pytest.fixture
def config_options_cfs():
    config_path = './yaml/configOptions_cfs.yaml'
    config_options = ConfigOptions(config_path)
    config_options.read_config()
    yield config_options

@pytest.fixture
def config_options_downscale():
    config_path = './yaml/configOptions_downscale.yaml'
    config_options = ConfigOptions(config_path)
    config_options.read_config()
    yield config_options

@pytest.fixture
def unbiased_input_forcings(config_options,geo_meta):
    """
    Prepare the mock input forcing grid
    """
    config_options.read_config()

    inputForcingMod = initDict(config_options,geo_meta)
    input_forcings = inputForcingMod[config_options.input_forcings[0]]

    # put a value of 1.5 at every gridpoint, for 10 mock variables
    input_forcings.final_forcings = np.full((10,geo_meta.ny_local, geo_meta.nx_local),1.5)
    input_forcings.fcst_hour2 = 12

    return input_forcings

def isWithinTolerance(val1, val2, tolerance=None):
    """
    Make sure val1==val2, within the provided tolerance
    """
    if tolerance is None:
        tolerance = 0.0
    
    diff = abs(val1 - val2)
    return diff <= tolerance


def run_and_compare(funct, config_options, unbiased_input_forcings, mpi_meta, geo_meta=None, 
                    compval=None, tolerance=None, equal=False, index=(0,0), var='T2D', isDownscaleFunct=False):
    """
    Run the provided function, and sample the output to compare
    against the expected value. Also ensure the overall original and modified grid sums are different.
    :param funct: The function to call
    :param config_options:
    :param unbiased_input_forcings:
    :param mpi_meta:
    :param geo_meta: Can be None if not needed for function call
    :param compval: Expected corrected value. Can be None to skip this test
    :tolerance: Ensure that abs(compval-sampledval) <= tolerance. If None, the values must match exactly
    :param equal: If true, the modified grid should be equivalent to the original grid
    """
    #config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc
    config_options.current_output_step = 60

    unbiased_input_forcings.define_product()
    final_forcings = np.copy(unbiased_input_forcings.final_forcings)
    if isDownscaleFunct:
        funct(unbiased_input_forcings, config_options, geo_meta, mpi_meta)
    else:
        varnum = unbiased_input_forcings.input_map_output.index(var)
        if geo_meta:
            funct(unbiased_input_forcings, geo_meta, config_options, mpi_meta, varnum)
        else:
            funct(unbiased_input_forcings, config_options, mpi_meta, varnum)

    if equal:
        assert np.sum(final_forcings) == np.sum(unbiased_input_forcings.final_forcings)
    else:
        assert np.sum(final_forcings) != np.sum(unbiased_input_forcings.final_forcings)

    if compval is not None:
        outId = getOutputId(config_options, unbiased_input_forcings, var)
        idx = (outId, index[0], index[1])
        
        pointval = unbiased_input_forcings.final_forcings[idx]
        # TODO set acceptable tolerance
        print("expected value: %s, got: %s" % (compval, pointval))
        assert isWithinTolerance(pointval, compval, tolerance)

def getOutputId(config_options, input_forcings, varname):
    OutputEnum = config_options.OutputEnum
    varnum = input_forcings.input_map_output.index(varname)
    outId = 0
    for ind, xvar in enumerate(OutputEnum):
        if xvar.name == input_forcings.input_map_output[varnum]:
            outId = ind
            break

    return outId



