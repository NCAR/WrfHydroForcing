import pytest
import numpy as np

from core.parallel import MpiConfig
from core.config import ConfigOptions
from core.geoMod import GeoMetaWrfHydro
from core.forcingInputMod import *

from fixtures import *
import core.bias_correction as BC

@pytest.mark.mpi
def test_no_bias_correct(config_options, mpi_meta, input_forcings_final):
    config_options.read_config()
    final_forcings = np.copy(input_forcings_final.final_forcings)

    BC.no_bias_correct(input_forcings_final, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) == np.sum(input_forcings_final.final_forcings)
    assert input_forcings_final.final_forcings[4][0][0] == 1.5

@pytest.mark.mpi
def test_ncar_tbl_correction(config_options, mpi_meta, input_forcings_final):
    config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc

    final_forcings = np.copy(input_forcings_final.final_forcings)
    BC.ncar_tbl_correction(input_forcings_final, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings)

    pointval = input_forcings_final.final_forcings[4][0][0]
    # TODO set acceptable tolerance
    assert isWithinTolerance(pointval,2.5, None)

@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw(config_options, mpi_meta, input_forcings_final):
    config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc

    final_forcings = np.copy(input_forcings_final.final_forcings)
    BC.ncar_blanket_adjustment_lw(input_forcings_final, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings)
   
    pointval = input_forcings_final.final_forcings[4][0][0]
    # TODO set acceptable tolerance
    assert isWithinTolerance(pointval, 6.768526039220487, None )
    
@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct(config_options, geo_meta, mpi_meta, input_forcings_final):
    config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc

    final_forcings = np.copy(input_forcings_final.final_forcings)
    BC.ncar_sw_hrrr_bias_correct(input_forcings_final, geo_meta, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings)
    
    pointval = input_forcings_final.final_forcings[4][0][0]
    # TODO set acceptable tolerance
    assert isWithinTolerance(pointval, 1.4999999948202003, None )

@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct(config_options, mpi_meta, input_forcings_final):
    config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc

    final_forcings = np.copy(input_forcings_final.final_forcings)
    BC.ncar_temp_hrrr_bias_correct(input_forcings_final, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings)   
    
    pointval = input_forcings_final.final_forcings[4][0][0]
    # TODO set acceptable tolerance
    assert isWithinTolerance(pointval, 1.558319002854755, None)

@pytest.mark.mpi
def test_ncar_temp_gfs_bias_correct(config_options, mpi_meta, input_forcings_final):
    config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc
    config_options.current_output_step = 60

    final_forcings = np.copy(input_forcings_final.final_forcings)
    BC.ncar_temp_gfs_bias_correct(input_forcings_final, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings)   
    
    pointval = input_forcings_final.final_forcings[4][0][0]
    # TODO set acceptable tolerance
    assert isWithinTolerance(pointval, 0.7468065367962597, None)

    config_options.current_output_step = 180
    input_forcings_final.final_forcings = np.copy(final_forcings)
    BC.ncar_temp_gfs_bias_correct(input_forcings_final, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings) 

    pointval = input_forcings_final.final_forcings[4][0][0]
    # TODO set acceptable tolerance
    assert isWithinTolerance(pointval, 0.9868065367962597, None)

@pytest.mark.mpi
def test_ncar_lwdown_gfs_bias_correct(config_options, mpi_meta, input_forcings_final):
    config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc
    config_options.current_output_step = 60

    final_forcings = np.copy(input_forcings_final.final_forcings)
    BC.ncar_lwdown_gfs_bias_correct(input_forcings_final, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings)
   
    pointval = input_forcings_final.final_forcings[4][0][0]
    # TODO set acceptable tolerance
    assert isWithinTolerance(pointval, 12.54182912750415, None)

@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct(config_options, mpi_meta, input_forcings_final):
    config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc
    config_options.current_output_step = 60

    final_forcings = np.copy(input_forcings_final.final_forcings)
    BC.ncar_wspd_hrrr_bias_correct(input_forcings_final, config_options, mpi_meta, 0)

    assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings)
   
    pointval = input_forcings_final.final_forcings[4][0][0]
    # TODO set acceptable tolerance
    assert isWithinTolerance(pointval, 1.5722859839330992, None)

# @pytest.mark.mpi
# def test_ncar_wspd_gfs_bias_correct(config_options, mpi_meta, input_forcings_final):
#     config_options.read_config()
#     config_options.current_output_date = config_options.b_date_proc
#     config_options.current_output_step = 60

#     input_forcings_final.define_product()
#     final_forcings = np.copy(input_forcings_final.final_forcings)
#     BC.ncar_wspd_gfs_bias_correct(input_forcings_final, config_options, mpi_meta, 0)

#     assert np.sum(final_forcings) != np.sum(input_forcings_final.final_forcings)
   
#     pointval = input_forcings_final.final_forcings[4][0][0]
#     # TODO set acceptable tolerance
#     assert isWithinTolerance(pointval, 1.5722859839330992, None)