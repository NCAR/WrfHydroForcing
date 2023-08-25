import pytest
from core.config import ConfigOptions
from gdrive_download import maybe_download_testdata

@pytest.mark.parametrize('invalid_config', [
    # Invalid values or types
     'yaml/configOptions_invalid1.yaml'
    # Add more invalid configurations
])
def test_read_config_invalid(invalid_config):
    maybe_download_testdata()
    with pytest.raises(Exception):
        config_options = ConfigOptions(invalid_config)
        config_options.read_config()

@pytest.mark.parametrize('missing_key', [
    # Missing required keys
    'Forecast',
    'Output'
    # Add more missing keys
])
def test_read_config_missing(missing_key):
    maybe_download_testdata()
    config = 'yaml/configOptions_missing.yaml'
    with pytest.raises(Exception):
        config_options = ConfigOptions(config)
        config_options.read_config()

@pytest.mark.parametrize('empty_value', [
    # Empty values
    'yaml/configOptions_empty.yaml'
    # Add more empty values
])
def test_read_config_empty(empty_value):
    maybe_download_testdata()
    with pytest.raises(Exception):
        config_options = ConfigOptions(empty_value)
        config_options.read_config()

@pytest.fixture
def config_options():
    config_path = 'yaml/configOptions_valid.yaml'
    return ConfigOptions(config_path)

def test_read_config_valid(config_options):
    config_options.read_config()

    # Assert expected values for valid configuration
    assert config_options.output_dir == 'output/Hawaii_AnA_withMRMS_test'
    assert config_options.fcst_freq == 60

    # Add more assertions for other attributes
