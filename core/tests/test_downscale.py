import pytest
import numpy as np
from datetime import datetime

from core.parallel import MpiConfig
from core.config import ConfigOptions
from core.geoMod import GeoMetaWrfHydro
from core.forcingInputMod import *
from core.enumConfig import *

from fixtures import *
import core.downscale as DS


expected_values = {
    'no_downscale': [
        { 'index': (0,0), 'value': 1.5, 'tolerance': None}
    ],
    'simple_lapse': [
        { 'index': (0,0), 'value': 2.1489999890327454, 'tolerance': None}
    ],
    'param_lapse': [
        { 'index': (0,0), 'value': 3.0, 'tolerance': None}
    ],
    'pressure_down_classic': [
        { 'index': (0,0), 'value': 4.9140393659641175, 'tolerance': None}
    ],
    'q2_down_classic': [
        { 'index': (0,0), 'value': 0.9992289527938468, 'tolerance': None}
    ],
    'nwm_monthly_PRISM_downscale': [
        { 'index': (0,0), 'value': 2.357142857142857, 'tolerance': None}
    ],
    'ncar_topo_adj': [
        { 'index': (0,0), 'value': 1400.0, 'tolerance': None}
    ]
}


############### Downscale function tests ##########################


@pytest.mark.mpi
def test_no_downscale(unbiased_input_forcings, config_options_downscale, geo_meta, mpi_meta):
    run_unit_test(config_options_downscale, mpi_meta, unbiased_input_forcings, geo_meta, type='no_downscale', equal=True)

@pytest.mark.mpi
def test_simple_lapse(unbiased_input_forcings, config_options_downscale, geo_meta, mpi_meta):
    unbiased_input_forcings.q2dDownscaleOpt = DownScaleHumidEnum.REGRID_TEMP_PRESS
    unbiased_input_forcings.height = geo_meta.height + 100
    run_unit_test(config_options_downscale, mpi_meta, unbiased_input_forcings, geo_meta, type='simple_lapse')

@pytest.mark.mpi
def test_param_lapse(unbiased_input_forcings, config_options_downscale, geo_meta, mpi_meta):
    unbiased_input_forcings.q2dDownscaleOpt = DownScaleHumidEnum.REGRID_TEMP_PRESS
    unbiased_input_forcings.height = geo_meta.height + 100
    unbiased_input_forcings.paramDir = config_options_downscale.dScaleParamDirs[0]
    run_unit_test(config_options_downscale, mpi_meta, unbiased_input_forcings, geo_meta, type='param_lapse')

@pytest.mark.mpi
def test_pressure_down_classic(unbiased_input_forcings, config_options_downscale, geo_meta, mpi_meta):
    unbiased_input_forcings.height = geo_meta.height + 100
    run_unit_test(config_options_downscale, mpi_meta, unbiased_input_forcings, geo_meta, 
                  type='pressure_down_classic',var='PSFC')
    
@pytest.mark.mpi
def test_q2_down_classic(unbiased_input_forcings, config_options_downscale, geo_meta, mpi_meta):
    unbiased_input_forcings.t2dTmp = unbiased_input_forcings.t2dTmp * 0 + 273.15
    unbiased_input_forcings.psfcTmp = unbiased_input_forcings.psfcTmp * 0 + 100000
    run_unit_test(config_options_downscale, mpi_meta, unbiased_input_forcings, geo_meta, 
                  type='q2_down_classic',var='Q2D')
    
@pytest.mark.mpi
def test_nwm_monthly_PRISM_downscale(unbiased_input_forcings, config_options_downscale, geo_meta, mpi_meta):
    config_options_downscale.current_output_date = datetime.strptime("20101001", "%Y%m%d")
    config_options_downscale.prev_output_date = datetime.strptime("20100901", "%Y%m%d")
    unbiased_input_forcings.paramDir = config_options_downscale.dScaleParamDirs[0]
    run_unit_test(config_options_downscale, mpi_meta, unbiased_input_forcings, geo_meta, 
                  type='nwm_monthly_PRISM_downscale',var='RAINRATE')
    
@pytest.mark.mpi
def test_ncar_topo_adj(unbiased_input_forcings, config_options_downscale, geo_meta, mpi_meta):
    config_options_downscale.current_output_date = datetime.strptime("2010122000", "%Y%m%d%H")
    geo_meta.latitude_grid = geo_meta.latitude_grid + 20.0
    unbiased_input_forcings.final_forcings = np.full((10,geo_meta.ny_local, geo_meta.nx_local),1500.0)
    run_unit_test(config_options_downscale, mpi_meta, unbiased_input_forcings, geo_meta, 
                  type='ncar_topo_adj',var='SWDOWN')
    

############### Helper function tests ##########################


@pytest.mark.mpi
def test_radconst(config_options_downscale):
    config_options_downscale.current_output_date = datetime.strptime("20230427","%Y%m%d")
    DECLIN, SOLCON = DS.radconst(config_options_downscale)

    assert isWithinTolerance(DECLIN, 0.23942772252574912, None)
    assert isWithinTolerance(SOLCON, 1350.9227993143058, None)

    config_options_downscale.current_output_date = datetime.strptime("20231027","%Y%m%d")
    DECLIN, SOLCON = DS.radconst(config_options_downscale)

    assert isWithinTolerance(DECLIN, -0.24225978751208468, None)
    assert isWithinTolerance(SOLCON, 1388.352238489423, None)


@pytest.mark.mpi
def test_calc_coszen(config_options_downscale, geo_meta, mpi_meta):
    # Skip this test for nprocs > 1, since the latitudes will be
    # different for each scattered array
    if mpi_meta.size > 1 and mpi_meta.rank != 0:
        return True

    config_options_downscale.current_output_date = datetime.strptime("2023042706","%Y%m%d%H")
    declin, solcon = DS.radconst(config_options_downscale)
    coszen, hrang = DS.calc_coszen(config_options_downscale, declin, geo_meta)

    assert isWithinTolerance(coszen[0][0], -0.24323696, 1e-7)
    assert isWithinTolerance(hrang[0][0], -4.3573823, 1e-7)

    config_options_downscale.current_output_date = datetime.strptime("2023102718","%Y%m%d%H")
    declin, solcon = DS.radconst(config_options_downscale)
    coszen, hrang = DS.calc_coszen(config_options_downscale, declin, geo_meta)
 
    assert isWithinTolerance(coszen[0][0], 0.29333305, 1e-7)
    assert isWithinTolerance(hrang[0][0], -1.1556603, 1e-7)


@pytest.mark.mpi
def test_rel_hum(config_options_downscale, unbiased_input_forcings, geo_meta):
    unbiased_input_forcings.t2dTmp = np.full((geo_meta.ny_local, geo_meta.nx_local),300)
    unbiased_input_forcings.psfcTmp = np.full((geo_meta.ny_local, geo_meta.nx_local),100000)
    unbiased_input_forcings.final_forcings = np.full((10, geo_meta.ny_local, geo_meta.nx_local),0.015)

    rh = DS.rel_hum(unbiased_input_forcings, config_options_downscale)

    assert isWithinTolerance(rh[0][0], 68.33107966724083, None)


@pytest.mark.mpi
def test_mixhum_ptrh(config_options_downscale, unbiased_input_forcings, geo_meta):
    unbiased_input_forcings.t2dTmp = np.full((geo_meta.ny_local, geo_meta.nx_local),300)
    unbiased_input_forcings.psfcTmp = np.full((geo_meta.ny_local, geo_meta.nx_local),100000)
    unbiased_input_forcings.final_forcings = np.full((10, geo_meta.ny_local, geo_meta.nx_local),0.015)

    relHum = DS.rel_hum(unbiased_input_forcings, config_options_downscale)

    rh = DS.mixhum_ptrh(unbiased_input_forcings, relHum, 2, config_options_downscale)

    assert isWithinTolerance(rh[0][0], 9.039249311422024, None)


def run_unit_test(config_options, mpi_meta, unbiased_input_forcings, geo_meta, type=None, equal=False, var='T2D'):
    funct = getattr(DS, type)
    for spec in expected_values[type]:
        run_and_compare(funct, config_options, unbiased_input_forcings, mpi_meta, geo_meta=geo_meta, equal=equal,
                compval=spec['value'], tolerance=spec['tolerance'], index=spec['index'], var=var, isDownscaleFunct=True)

