import pytest
from core.config import ConfigOptions
from core.forcingInputMod import input_forcings
from gdrive_download import maybe_download_testdata


@pytest.fixture
def config_options():
    maybe_download_testdata()
    config_path = './yaml/configOptions_valid.yaml'
    yield ConfigOptions(config_path)

@pytest.fixture
def input_forcings_sample(config_options):
    maybe_download_testdata()
    config_options.read_config()
    force_key = config_options.input_forcings[0]
    InputDict = {}
    InputDict[force_key] = input_forcings()
    InputDict[force_key].keyValue = force_key
    yield InputDict

def test_define_product(config_options, input_forcings_sample):
    force_key = config_options.input_forcings[0]
    input_forcings_sample[force_key].forcingInputModYaml = "./yaml/inputForcings_valid.yaml"
    input_forcings_sample[force_key].define_product()
    # Assert expected values for valid configuration
    assert input_forcings_sample[force_key].cycleFreq == 360
    assert input_forcings_sample[force_key].input_map_output == ['T2D','Q2D','U2D','V2D','RAINRATE','SWDOWN','LWDOWN','PSFC']

    # Add more assertions for other attributes

@pytest.mark.parametrize('invalid_config', ['./yaml/inputForcings_invalid.yaml'
    # Add more invalid configurations
])

def test_define_config_invalid(invalid_config):
    with pytest.raises(Exception):
        
        force_key = config_options.input_forcings[0]
        input_forcings_sample[force_key].forcingInputModYaml = invalid_config
        input_forcings_sample[force_key].define_product()

@pytest.mark.parametrize('missing_key', [
    # Missing required keys
    'GFS_GLOBAL',
    'HRRR',
    # Add more missing keys
])
def test_define_config_missing(missing_key):
    config = './yaml/inputForcings_missing.yaml'
    with pytest.raises(Exception):
        config_options = ConfigOptions(config)
        config_options.read_config()

@pytest.mark.parametrize('empty_value', [
    # Empty values
    './yaml/inputForcings_empty.yaml'
    # Add more empty values
])
def test_define_config_empty(empty_value):
    with pytest.raises(Exception):
        config_options = ConfigOptions(empty_value)
        config_options.read_config()

