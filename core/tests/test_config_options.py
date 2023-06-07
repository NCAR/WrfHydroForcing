import pytest
from WrfHydroForcing.core.config import ConfigOptions

@pytest.mark.parametrize('invalid_config', [
    # Invalid values or types
     '/glade/work/ishitas/FE_Development/merged_master_code/WrfHydroForcing/core/tests/configOptions_invalid1.yaml'
    # Add more invalid configurations
])
def test_read_config_invalid(invalid_config):
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
    config = '/glade/work/ishitas/FE_Development/merged_master_code/WrfHydroForcing/core/tests/configOptions_missing.yaml'
    with pytest.raises(Exception):
        config_options = ConfigOptions(config)
        config_options.read_config()

@pytest.mark.parametrize('empty_value', [
    # Empty values
    '/glade/work/ishitas/FE_Development/merged_master_code/WrfHydroForcing/core/tests/configOptions_empty.yaml'
    # Add more empty values
])
def test_read_config_empty(empty_value):
    with pytest.raises(Exception):
        config_options = ConfigOptions(empty_value)
        config_options.read_config()

@pytest.fixture
def config_options():
    config_path = '/glade/work/ishitas/FE_Development/merged_master_code/WrfHydroForcing/core/tests/configOptions_valid.yaml'
    return ConfigOptions(config_path)

def test_read_config_valid(config_options):
    config_options.read_config()

    # Assert expected values for valid configuration
    assert config_options.output_dir == '/glade/scratch/ishitas/ForcingEngine/Hawaii_AnA_withMRMS_test'
    assert config_options.fcst_freq == 60

    # Add more assertions for other attributes
