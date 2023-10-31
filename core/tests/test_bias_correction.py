import pytest
import numpy as np
from datetime import datetime

from core.parallel import MpiConfig
from core.config import ConfigOptions
from core.geoMod import GeoMetaWrfHydro
from core.forcingInputMod import *
from core.enumConfig import *

from fixtures import *
import core.bias_correction as BC


def date(datestr):
    return datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')

var_list = {
    'T2D': 300, 
    'Q2D': 50.0,
    'U2D': 2.5, 
    'V2D': 1.5,
    'RAINRATE': 2.5,
    'SWDOWN': 400,
    'LWDOWN': 400,
    'PSFC': 90000
}

expected_values = {
    'no_bias_correct': [
        { 'index': (0,0), 'value': 1.5, 'tolerance': None}
    ],
    'ncar_tbl_correction': [
        { 'index': (0,0), 'value': 2.5, 'tolerance': None, 'var': 'T2D', 'fcst_hour': 0},
        { 'index': (0,0), 'value': 2.5, 'tolerance': None, 'var': 'Q2D', 'fcst_hour': 0},
        { 'index': (0,0), 'value': 1.85, 'tolerance': None, 'var': 'U2D', 'fcst_hour': 0},
        { 'index': (0,0), 'value': 1.85, 'tolerance': None, 'var': 'V2D', 'fcst_hour': 0},
        { 'index': (0,0), 'value': 2.5, 'tolerance': None, 'var': 'T2D', 'fcst_hour': 6},
        { 'index': (0,0), 'value': 2.5, 'tolerance': None, 'var': 'Q2D', 'fcst_hour': 6},
        { 'index': (0,0), 'value': 1.6, 'tolerance': None, 'var': 'U2D', 'fcst_hour': 6},
        { 'index': (0,0), 'value': 1.6, 'tolerance': None, 'var': 'V2D', 'fcst_hour': 6},
        { 'index': (0,0), 'value': 2.5, 'tolerance': None, 'var': 'T2D', 'fcst_hour': 12},
        { 'index': (0,0), 'value': 2.5, 'tolerance': None, 'var': 'Q2D', 'fcst_hour': 12},
        { 'index': (0,0), 'value': 1.52, 'tolerance': None, 'var': 'U2D', 'fcst_hour': 12},
        { 'index': (0,0), 'value': 1.52, 'tolerance': None, 'var': 'V2D', 'fcst_hour': 12},
        { 'index': (0,0), 'value': 2.5, 'tolerance': None, 'var': 'T2D', 'fcst_hour': 18},
        { 'index': (0,0), 'value': 2.5, 'tolerance': None, 'var': 'Q2D', 'fcst_hour': 18},
        { 'index': (0,0), 'value': 1.45, 'tolerance': None, 'var': 'U2D', 'fcst_hour': 18},
        { 'index': (0,0), 'value': 1.45, 'tolerance': None, 'var': 'V2D', 'fcst_hour': 18}
    ],
    'ncar_blanket_adjustment_lw': [
        { 'index': (0,0), 'value': 9.82194214668789, 'tolerance': None, 'date': date('2023-01-01 00:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 7.763851170382089, 'tolerance': None, 'date': date('2023-04-01 06:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 6.8580578533121095, 'tolerance': None, 'date': date('2023-07-01 12:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 8.91614882961791, 'tolerance': None, 'date': date('2023-10-01 18:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 7.479010060854788, 'tolerance': None, 'date': date('2023-01-01 00:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 4.6895996292499, 'tolerance': None, 'date': date('2023-04-01 06:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 6.160989939145213, 'tolerance': None, 'date': date('2023-07-01 12:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 8.950400370750101, 'tolerance': None, 'date': date('2023-10-01 18:00:00'), 'ana': 0}
    ],
    'ncar_sw_hrrr_bias_correct': [
        { 'index': (0,0), 'value': 1.5826259963214397, 'tolerance': None, 'date': date('2023-01-01 00:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.4999999948202003, 'tolerance': None, 'date': date('2023-04-01 06:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.4999999948202003, 'tolerance': None, 'date': date('2023-07-01 12:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.540613193064928, 'tolerance': None, 'date': date('2023-10-01 18:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.4394609276205301, 'tolerance': None, 'date': date('2023-01-01 00:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 1.500000003795177, 'tolerance': None, 'date': date('2023-04-01 06:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 1.500000003795177, 'tolerance': None, 'date': date('2023-07-01 12:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 1.4702432015910745, 'tolerance': None, 'date': date('2023-10-01 18:00:00'), 'ana': 0}        
    ],
    'ncar_temp_hrrr_bias_correct': [
        { 'index': (0,0), 'value': 1.462288117950048, 'tolerance': None, 'date': date('2023-01-01 00:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.49941014460614, 'tolerance': None, 'date': date('2023-04-01 06:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.5757118820499518, 'tolerance': None, 'date': date('2023-07-01 12:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.53858985539386, 'tolerance': None, 'date': date('2023-10-01 18:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 0.9518785484037021, 'tolerance': None, 'date': date('2023-01-01 00:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 0.8684798631054194, 'tolerance': None, 'date': date('2023-04-01 06:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 0.884121451596298, 'tolerance': None, 'date': date('2023-07-01 12:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 0.9675201368945807, 'tolerance': None, 'date': date('2023-10-01 18:00:00'), 'ana': 0}
    ],
    'ncar_temp_gfs_bias_correct': [
        { 'index': (0,0), 'value': 2.6484931133084233, 'tolerance': None, 'date': date('2023-01-01 00:00:00')},
        { 'index': (0,0), 'value': 2.1467845464398003, 'tolerance': None, 'date': date('2023-04-01 06:00:00')},
        { 'index': (0,0), 'value': 0.23150688669157682, 'tolerance': None, 'date': date('2023-07-01 12:00:00')},
        { 'index': (0,0), 'value': 0.7332154535601993, 'tolerance': None, 'date': date('2023-10-01 18:00:00')}
    ],
    'ncar_lwdown_gfs_bias_correct': [
        { 'index': (0,0), 'value': 10.897517774766143, 'tolerance': None, 'date': date('2023-01-01 00:00:00')},
        { 'index': (0,0), 'value': 12.813333511002988, 'tolerance': None, 'date': date('2023-04-01 06:00:00')},
        { 'index': (0,0), 'value': 11.902482225233857, 'tolerance': None, 'date': date('2023-07-01 12:00:00')},
        { 'index': (0,0), 'value': 9.986666488997013, 'tolerance': None, 'date': date('2023-10-01 18:00:00')}
    ],
    'ncar_wspd_hrrr_bias_correct': [
        { 'index': (0,0), 'value': 1.7324061943600948, 'tolerance': None, 'date': date('2023-01-01 00:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.6027854243986395, 'tolerance': None, 'date': date('2023-04-01 06:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.592862924985717, 'tolerance': None, 'date': date('2023-07-01 12:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.7224836949471725, 'tolerance': None, 'date': date('2023-10-01 18:00:00'), 'ana': 1},
        { 'index': (0,0), 'value': 1.2131465997822362, 'tolerance': None, 'date': date('2023-01-01 00:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 1.080473211999704, 'tolerance': None, 'date': date('2023-04-01 06:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 1.1504572971498712, 'tolerance': None, 'date': date('2023-07-01 12:00:00'), 'ana': 0},
        { 'index': (0,0), 'value': 1.2831306849324031, 'tolerance': None, 'date': date('2023-10-01 18:00:00'), 'ana': 0}
    ],
     'ncar_wspd_gfs_bias_correct': [
        { 'index': (0,0), 'value': 1.560235849440387, 'tolerance': None, 'date': date('2023-01-01 00:00:00')},
        { 'index': (0,0), 'value': 1.2559415578811088, 'tolerance': None, 'date': date('2023-04-01 06:00:00')},
        { 'index': (0,0), 'value': 1.1569214380849937, 'tolerance': None, 'date': date('2023-07-01 12:00:00')},
        { 'index': (0,0), 'value': 1.4612157296442718, 'tolerance': None, 'date': date('2023-10-01 18:00:00')}
    ],
    'cfsv2_nldas_nwm_bias_correct': [
        { 'index': (0,0), 'value': 285, 'tolerance': None, 'var': 'T2D' },
        { 'index': (0,0), 'value': 0.009524456432886543, 'tolerance': None, 'var': 'Q2D' },
        { 'index': (0,0), 'value': 3.562956632266148, 'tolerance': None, 'var': 'U2D' },
        { 'index': (0,0), 'value': 0.11510974533140078, 'tolerance': None, 'var': 'V2D' }, 
        { 'index': (0,0), 'value': 0.01622283237712793, 'tolerance': None, 'var': 'RAINRATE' },
        { 'index': (0,0), 'value': 0, 'tolerance': None, 'var': 'SWDOWN' },
        { 'index': (0,0), 'value': 272, 'tolerance': None, 'var': 'LWDOWN' },
        { 'index': (0,0), 'value': 49999 , 'tolerance': None, 'var': 'PSFC' }
    ]
}

#######################################################
#
# This test suite creates a mock 'forcing grid' filled
# with a known value, calls each bias_correction function,
# and samples a gridpoint to compare against the expected
# bias-corrected value for that function.
#
# The functions currently expect an exact (full precision)
# match, but a tolerance can be set for each test or
# globally in the run_and_compare() function
#
#######################################################

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

@pytest.fixture
def unbiased_input_forcings_cfs(config_options_cfs,geo_meta,mpi_meta):
    """
    Prepare the mock input forcing grid for CFS
    """
    # Don't run for mpi processes > 0, since the
    # CFS and NLDAS parameters are spatially dependent and we're not sure
    # how we are splitting up the grid.
    if mpi_meta.size > 1:
        return
    config_options_cfs.read_config()

    inputForcingMod = initDict(config_options_cfs,geo_meta)
    input_forcings = inputForcingMod[config_options_cfs.input_forcings[0]]
    # TODO Can we construct this path from the config file? Parts of the path are currently hard-coded
    # im time_handling.py, however
    input_forcings.file_in2 = "config_data/CFSv2/cfs.20190923/06/6hrly_grib_01/flxf2019100106.01.2019092306.grb2"
    input_forcings.fcst_hour2 = 192
    input_forcings.fcst_date1 = datetime.strptime("2019100106", "%Y%m%d%H%M")
    input_forcings.fcst_date2 = input_forcings.fcst_date1

    input_forcings.regrid_inputs(config_options_cfs,geo_meta,mpi_meta)

    # put a value of 1.5 at every gridpoint, for 10 mock variables
    input_forcings.final_forcings = np.full((10,geo_meta.ny_local, geo_meta.nx_local),1.5)
    input_forcings.fcst_hour2 = 12

    return input_forcings

#####  No bias correction tests  #########
@pytest.mark.mpi
def test_no_bias_correct(config_options, mpi_meta, unbiased_input_forcings):
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='no_bias_correct', equal=True)


#####  NCAR table bias correction tests  #########
@pytest.mark.mpi
def test_ncar_tbl_correction_0z_temp(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='T2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_0z_rh(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='Q2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_0z_u(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='U2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_0z_v(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='V2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_6z_temp(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 6
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='T2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_6z_rh(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 6
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='Q2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_6z_u(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 6
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='U2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_6z_v(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 6
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='V2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_12z_temp(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 12
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='T2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_12z_rh(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 12
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='Q2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_12z_u(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 12
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='U2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_12z_v(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 12
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='V2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_18z_temp(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 18
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='T2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_18z_rh(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 18
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='Q2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_18z_u(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 18
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='U2D')

@pytest.mark.mpi
def test_ncar_tbl_correction_18z_v(config_options, mpi_meta, unbiased_input_forcings):
    unbiased_input_forcings.fcst_hour2 = 18
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_tbl_correction', var='V2D')

#####  NCAR blanket LW Bias Correction tests  #########
@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw_0z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_blanket_adjustment_lw', var='LWDOWN')

@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw_6z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_blanket_adjustment_lw', var='LWDOWN')
        
@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw_12z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_blanket_adjustment_lw', var='LWDOWN')
        
@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw_18z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_blanket_adjustment_lw', var='LWDOWN')

@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw_0z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_blanket_adjustment_lw', var='LWDOWN')

@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw_6z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_blanket_adjustment_lw', var='LWDOWN')
        
@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw_12z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_blanket_adjustment_lw', var='LWDOWN')
        
@pytest.mark.mpi
def test_ncar_blanket_adjustment_lw_18z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_blanket_adjustment_lw', var='LWDOWN')
        
#####  NCAR HRRR SW bias correction tests  #########
@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct_0z(config_options, geo_meta, mpi_meta, unbiased_input_forcings):
    # This bias correction is affected by spatial position. Since the spatial point is different depending on how
    # many processes we have, we skip this test for n_processes > 1
    if mpi_meta.size > 1: return
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=geo_meta, type='ncar_sw_hrrr_bias_correct', var='SWDOWN')

@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct_6z(config_options, geo_meta, mpi_meta, unbiased_input_forcings):
    # This bias correction is affected by spatial position. Since the spatial point is different depending on how
    # many processes we have, we skip this test for n_processes > 1
    if mpi_meta.size > 1: return
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=geo_meta, type='ncar_sw_hrrr_bias_correct', var='SWDOWN')

@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct_12z(config_options, geo_meta, mpi_meta, unbiased_input_forcings):
    # This bias correction is affected by spatial position. Since the spatial point is different depending on how
    # many processes we have, we skip this test for n_processes > 1
    if mpi_meta.size > 1: return
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=geo_meta, type='ncar_sw_hrrr_bias_correct', var='SWDOWN')

@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct_18z(config_options, geo_meta, mpi_meta, unbiased_input_forcings):
    # This bias correction is affected by spatial position. Since the spatial point is different depending on how
    # many processes we have, we skip this test for n_processes > 1
    if mpi_meta.size > 1: return
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=geo_meta, type='ncar_sw_hrrr_bias_correct', var='SWDOWN')
 
@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct_0z_noana(config_options, geo_meta, mpi_meta, unbiased_input_forcings):
    # This bias correction is affected by spatial position. Since the spatial point is different depending on how
    # many processes we have, we skip this test for n_processes > 1
    if mpi_meta.size > 1: return
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=geo_meta, type='ncar_sw_hrrr_bias_correct', var='SWDOWN')

@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct_6z_noana(config_options, geo_meta, mpi_meta, unbiased_input_forcings):
    # This bias correction is affected by spatial position. Since the spatial point is different depending on how
    # many processes we have, we skip this test for n_processes > 1
    if mpi_meta.size > 1: return
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=geo_meta, type='ncar_sw_hrrr_bias_correct', var='SWDOWN')

@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct_12z_noana(config_options, geo_meta, mpi_meta, unbiased_input_forcings):
    # This bias correction is affected by spatial position. Since the spatial point is different depending on how
    # many processes we have, we skip this test for n_processes > 1
    if mpi_meta.size > 1: return
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=geo_meta, type='ncar_sw_hrrr_bias_correct', var='SWDOWN')

@pytest.mark.mpi
def test_ncar_sw_hrrr_bias_correct_18z_noana(config_options, geo_meta, mpi_meta, unbiased_input_forcings):
    # This bias correction is affected by spatial position. Since the spatial point is different depending on how
    # many processes we have, we skip this test for n_processes > 1
    if mpi_meta.size > 1: return
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=geo_meta, type='ncar_sw_hrrr_bias_correct', var='SWDOWN')

#####  NCAR HRRR temperature bias correction tests  #########  
@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct_0z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_hrrr_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct_6z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_hrrr_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct_12z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_hrrr_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct_18z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_hrrr_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct_0z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_hrrr_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct_6z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_hrrr_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct_12z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_hrrr_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_hrrr_bias_correct_18z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_hrrr_bias_correct', var='T2D')

#####  NCAR GFS temperature bias correction tests  #########  
@pytest.mark.mpi
def test_ncar_temp_gfs_bias_correct_0z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_gfs_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_gfs_bias_correct_6z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_gfs_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_gfs_bias_correct_12z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_gfs_bias_correct', var='T2D')

@pytest.mark.mpi
def test_ncar_temp_gfs_bias_correct_18z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_temp_gfs_bias_correct', var='T2D')

#####  NCAR GFS LW bias correction tests  ######### 
@pytest.mark.mpi
def test_ncar_lwdown_gfs_bias_correct_0z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_lwdown_gfs_bias_correct', var='LWDOWN')

@pytest.mark.mpi
def test_ncar_lwdown_gfs_bias_correct_6z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_lwdown_gfs_bias_correct', var='LWDOWN')

@pytest.mark.mpi
def test_ncar_lwdown_gfs_bias_correct_12z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_lwdown_gfs_bias_correct', var='LWDOWN')

@pytest.mark.mpi
def test_ncar_lwdown_gfs_bias_correct_18z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_lwdown_gfs_bias_correct', var='LWDOWN')

#####  NCAR HRRR wind speed bias correction tests  #########
@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct_0z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_hrrr_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct_6z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_hrrr_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct_12z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_hrrr_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct_18z(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    config_options.ana_flag = 1
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_hrrr_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct_0z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_hrrr_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct_6z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_hrrr_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct_12z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_hrrr_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_hrrr_bias_correct_18z_noana(config_options, mpi_meta, unbiased_input_forcings):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    config_options.ana_flag = 0
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_hrrr_bias_correct', var='U2D')

#####  NCAR GFS Wind Speed bias correction tests  ######### 
@pytest.mark.mpi
def test_ncar_wspd_gfs_bias_correct_0z(config_options, unbiased_input_forcings, mpi_meta):
    config_options.b_date_proc = date('2023-01-01 00:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_gfs_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_gfs_bias_correct_6z(config_options, unbiased_input_forcings, mpi_meta):
    config_options.b_date_proc = date('2023-04-01 06:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_gfs_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_gfs_bias_correct_12z(config_options, unbiased_input_forcings, mpi_meta):
    config_options.b_date_proc = date('2023-07-01 12:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_gfs_bias_correct', var='U2D')

@pytest.mark.mpi
def test_ncar_wspd_gfs_bias_correct_18z(config_options, unbiased_input_forcings, mpi_meta):
    config_options.b_date_proc = date('2023-10-01 18:00:00')
    run_unit_test(config_options, mpi_meta, unbiased_input_forcings, type='ncar_wspd_gfs_bias_correct', var='U2D')

#####  CFS bias correction tests  #########
@pytest.mark.mpi
def test_cfsv2_nldas_nwm_bias_correct(config_options_cfs, unbiased_input_forcings_cfs, mpi_meta,geo_meta):
    # Don't run for mpi processes > 0, since the
    # CFS and NLDAS parameters are spatially dependent and we're not sure
    # how we are splitting up the grid.
    if mpi_meta.size > 1:
        return
    for var in var_list:
        val = var_list[var]
        unbiased_input_forcings_cfs.final_forcings = np.full((10,geo_meta.ny_local, geo_meta.nx_local),val)
        unbiased_input_forcings_cfs.coarse_input_forcings1 = np.full((10,geo_meta.ny_local, geo_meta.nx_local),val)
        unbiased_input_forcings_cfs.coarse_input_forcings2 = np.full((10,geo_meta.ny_local, geo_meta.nx_local),val)
        run_unit_test(config_options_cfs, mpi_meta, unbiased_input_forcings_cfs, type='cfsv2_nldas_nwm_bias_correct', var=var)

##### Tests for the top-level 'run_bias_correction()' function #######
@pytest.mark.mpi   
def test_run_bias_correction1(unbiased_input_forcings, config_options, geo_meta, mpi_meta):
    unbiased_input_forcings.t2dBiasCorrectOpt = BiasCorrTempEnum.HRRR.name

    run_bias_correction(unbiased_input_forcings, config_options, geo_meta, mpi_meta, "ncar_temp_hrrr_bias_correct", varname='T2D')

@pytest.mark.mpi
def test_run_bias_correction2(unbiased_input_forcings, config_options, geo_meta, mpi_meta):
    unbiased_input_forcings.windBiasCorrectOpt = BiasCorrWindEnum.HRRR.name

    run_bias_correction(unbiased_input_forcings, config_options, geo_meta, mpi_meta, "ncar_wspd_hrrr_bias_correct", varname='U2D')

@pytest.mark.mpi
def test_run_bias_correction3(unbiased_input_forcings, config_options, geo_meta, mpi_meta):
    unbiased_input_forcings.q2dBiasCorrectOpt = BiasCorrHumidEnum.CUSTOM.name

    run_bias_correction(unbiased_input_forcings, config_options, geo_meta, mpi_meta, "ncar_tbl_correction", varname='Q2D')


############ Helper functions ################

def run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta=None, 
                  type='ncar_sw_hrrr_bias_correct', var='SWDOWN', equal=False):
    for spec in expected_values[type]:
        if 'fcst_hour' in spec and spec['fcst_hour'] != unbiased_input_forcings.fcst_hour2:
            continue
        if 'var' in spec and spec['var'] != var:
            continue
        if 'ana' in spec and spec['ana'] != config_options.ana_flag:
            continue
        if 'date' in spec and spec['date'] != config_options.b_date_proc:
            continue

        funct = getattr(BC, type)

        run_and_compare(funct, config_options, unbiased_input_forcings, mpi_meta, geo_meta=geo_meta,
                        compval=spec['value'], tolerance=spec['tolerance'], index=spec['index'], 
                        var=var, equal=equal)

def run_bias_correction(unbiased_input_forcings, config_options, geo_meta, mpi_meta, specname, varname="T2D"):
    config_options.current_output_step = 60
    config_options.current_output_date = config_options.b_date_proc

    final_forcings = np.copy(unbiased_input_forcings.final_forcings)

    BC.run_bias_correction(unbiased_input_forcings,config_options,geo_meta,mpi_meta)
    assert np.sum(final_forcings) != np.sum(unbiased_input_forcings.final_forcings)
    for spec in expected_values[specname]:
        if 'fcst_hour' in spec and spec['fcst_hour'] != unbiased_input_forcings.fcst_hour2:
            continue
        if 'var' in spec and spec['var'] != varname:
            continue
        if 'ana' in spec and spec['ana'] != config_options.ana_flag:
            continue
        if 'date' in spec and spec['date'] != config_options.b_date_proc:
            continue
        
        outId = getOutputId(config_options,unbiased_input_forcings, varname)
        index = (outId, spec['index'][0], spec['index'][1])
        pointval = unbiased_input_forcings.final_forcings[index]
        assert isWithinTolerance(pointval, spec['value'], spec['tolerance'])

def run_and_compare(funct, config_options, unbiased_input_forcings, mpi_meta, geo_meta=None, 
                    compval=None, tolerance=None, equal=False, index=(0,0), var='T2D'):
    """
    Run the provided bias_correction function, and sample the output to compare
    against the expected value. Also ensure the overall original and bias_corrected grid sums are different.
    :param funct: The function to call
    :param config_options:
    :param unbiased_input_forcings:
    :param mpi_meta:
    :param geo_meta: Can be None if not needed for function call
    :param compval: Expected corrected value. Can be None to skip this test
    :tolerance: Ensure that abs(compval-sampledval) <= tolerance. If None, the values must match exactly
    :param equal: If true, the bias_corrected grid should be equivalent to the original grid
    """
    #config_options.read_config()
    config_options.current_output_date = config_options.b_date_proc
    config_options.current_output_step = 60

    unbiased_input_forcings.define_product()
    final_forcings = np.copy(unbiased_input_forcings.final_forcings)
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

def isWithinTolerance(val1, val2, tolerance=None):
    """
    Make sure val1==val2, within the provided tolerance
    """
    if tolerance is None:
        tolerance = 0.0
    
    diff = abs(val1 - val2)
    return diff <= tolerance
