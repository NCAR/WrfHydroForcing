"""
Module file for handling of downscaling input regridded (possibly bias-corrected)
forcing fields. Each input forcing product will loop through and determine if
downscaling is needed, based off options specified by the user.
"""
import math
import time
import numpy as np
from core import errMod

def run_downscaling(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    Top level module function that will downscale forcing variables
    for this particular input forcing product.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    # Dictionary mapping to temperature downscaling.
    downscale_temperature = {
        0: no_downscale,
        1: simple_lapse
        #2: param_lapse
    }
    downscale_temperature[input_forcings.t2dDownscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)

    # Dictionary mapping to pressure downscaling.
    downscale_pressure = {
        0: no_downscale,
        1: pressure_down_classic
    }
    downscale_pressure[input_forcings.psfcDownscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)

    # Dictionary mapping to shortwave radiation downscaling
    downscale_sw = {
        0: no_downscale,
        1: ncar_topo_adj
    }
    downscale_sw[input_forcings.swDowscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)

    # Dictionary mapping to specific humidity downscaling
    downscale_q2 = {
        0: no_downscale,
        1: q2_down_classic
    }
    downscale_q2[input_forcings.q2dDownscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)

    # Dictionary mapping to precipitation downscaling.
    downscale_precip = {
        0: no_downscale,
        1: no_downscale
        #1: precip_mtn_mapper
    }
    downscale_precip[input_forcings.precipDownscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)

def no_downscale(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    Generic function for passing states through without any
    downscaling.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    input_forcings.final_forcings = input_forcings.final_forcings

def simple_lapse(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    Function that applies a single lapse rate adjustment to modeled
    2-meter temperature by taking the difference of the native
    input elevation and the WRF-hydro elevation.
    :param inpute_forcings:
    :param ConfigOptions:
    :param GeoMetaWrfHydro:
    :return:
    """
    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Applying simple lapse rate to temperature downscaling"
        errMod.log_msg(ConfigOptions,MpiConfig)

    # Calculate the elevation difference.
    elevDiff = input_forcings.height - GeoMetaWrfHydro.height

    # Assign existing, un-downscaled temperatures to a temporary placeholder, which
    # will be used for specific humidity downscaling.
    if input_forcings.q2dDownscaleOpt > 0:
        input_forcings.t2dTmp[:,:] = input_forcings.final_forcings[4,:,:]

    # Apply single lapse rate value to the input 2-meter
    # temperature values.
    try:
        indNdv = np.where(input_forcings.final_forcings == ConfigOptions.globalNdv)
    except:
        ConfigOptions.errMsg = "Unable to perform NDV search on input forcings"
        errMod.log_critical(ConfigOptions,MpiConfig)
        return
    try:
        input_forcings.final_forcings[4,:,:] = input_forcings.final_forcings[4,:,:] + \
                                               (6.49/1000.0)*elevDiff
    except:
        ConfigOptions.errMsg = "Unable to apply lapse rate to input 2-meter temperatures."
        errMod.log_critical(ConfigOptions,MpiConfig)
        return

    input_forcings.final_forcings[indNdv] = ConfigOptions.globalNdv

    # Reset for memory efficiency
    indNdv = None

def pressure_down_classic(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    Generic function to downscale surface pressure to the WRF-Hydro domain.
    :param input_forcings:
    :param ConfigOptions:
    :param GeoMetaWrfHydro:
    :return:
    """
    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Performing topographic adjustment to surface pressure."
        errMod.log_msg(ConfigOptions,MpiConfig)

    # Calculate the elevation difference.
    elevDiff = input_forcings.height - GeoMetaWrfHydro.height

    # Assign existing, un-downscaled pressure values to a temporary placeholder, which
    # will be used for specific humidity downscaling.
    if input_forcings.q2dDownscaleOpt > 0:
        input_forcings.psfcTmp[:, :] = input_forcings.final_forcings[6, :, :]

    try:
        indNdv = np.where(input_forcings.final_forcings == ConfigOptions.globalNdv)
    except:
        ConfigOptions.errMsg = "Unable to perform NDV search on input forcings"
        errMod.log_critical(ConfigOptions, MpiConfig)
        return
    try:
        input_forcings.final_forcings[6,:,:] = input_forcings.final_forcings[6,:,:] +\
                                               (input_forcings.final_forcings[6,:,:]*elevDiff*9.8)/\
                                               (input_forcings.final_forcings[4,:,:]*287.05)
    except:
        ConfigOptions.errMsg = "Unable to downscale surface pressure to input forcings."
        errMod.log_critical(ConfigOptions, MpiConfig)
        return

    input_forcings.final_forcings[indNdv] = ConfigOptions.globalNdv

    # Reset for memory efficiency
    indNdv = None

def q2_down_classic(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    NCAR function for downscaling 2-meter specific humidity using already downscaled
    2-meter temperature, unadjusted surface pressure, and downscaled surface
    pressure.
    :param input_forcings:
    :param ConfigOptions:
    :param GeoMetaWrfHydro:
    :return:
    """
    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Performing topographic adjustment to specific humidity."
        errMod.log_msg(ConfigOptions,MpiConfig)

    # Establish where we have missing values.
    try:
        indNdv = np.where(input_forcings.final_forcings == ConfigOptions.globalNdv)
    except:
        ConfigOptions.errMsg = "Unable to perform NDV search on input forcings"
        errMod.log_critical(ConfigOptions, MpiConfig)
        return

    # First calculate relative humidity given original surface pressure and 2-meter
    # temperature
    try:
        relHum = rel_hum(input_forcings,ConfigOptions,MpiConfig)
    except:
        ConfigOptions.errMsg = "Unable to perform topographic downscaling of incoming " \
                               "specific humidity to relative humidity"
        errMod.log_critical(ConfigOptions, MpiConfig)
        return

    # Downscale 2-meter specific humidity
    try:
        q2Tmp = mixhum_ptrh(input_forcings,relHum,2,ConfigOptions)
    except:
        ConfigOptions.errMsg = "Unable to perform topographic downscaling of " \
                               "incoming specific humidity"
        errMod.log_critical(ConfigOptions, MpiConfig)
        return
    input_forcings.final_forcings[5,:,:] = q2Tmp
    input_forcings.final_forcings[indNdv] = ConfigOptions.globalNdv
    q2Tmp = None
    indNdv = None

def ncar_topo_adj(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    Topographic adjustment of incoming shortwave radiation fluxes,
    given input parameters.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Performing topographic adjustment to incoming " \
                                  "shortwave radiation flux."
        errMod.log_msg(ConfigOptions,MpiConfig)

    # Establish where we have missing values.
    try:
        indNdv = np.where(input_forcings.final_forcings == ConfigOptions.globalNdv)
    except:
        ConfigOptions.errMsg = "Unable to perform NDV search on input forcings"
        errMod.log_critical(ConfigOptions, MpiConfig)
        return

    # By the time this function has been called, necessary input static grids (height, slope, etc),
    # should have been calculated for each local slab of data.
    DEGRAD = math.pi/180.0
    DPD = 360.0/365.0
    try:
        DECLIN, SOLCON = radconst(ConfigOptions)
    except:
        ConfigOptions.errMsg = "Unable to calculate solar constants based on datetime information."
        errMod.log_critical(ConfigOptions,MpiConfig)
        return

    try:
        coszen_loc, hrang_loc = calc_coszen(ConfigOptions,DECLIN,GeoMetaWrfHydro)
    except:
        ConfigOptions.errMsg = "Unable to calculate COSZEN or HRANG variables for topographic adjustment " \
                               "of incoming shortwave radiation"
        errMod.log_critical(ConfigOptions,MpiConfig)
        return

    try:
        TOPO_RAD_ADJ_DRVR(GeoMetaWrfHydro,input_forcings,coszen_loc,DECLIN,SOLCON,
                          hrang_loc)
    except:
        ConfigOptions.errMsg = "Unable to perform final topographic adjustment of incoming " \
                               "shortwave radiation fluxes."
        errMod.log_critical(ConfigOptions,MpiConfig)
        return

    # Assign missing values based on our mask.
    input_forcings.final_forcings[indNdv] = ConfigOptions.globalNdv

    # Reset variables to free up memory
    DECLIN = None
    SOLCON = None
    coszen_loc = None
    hrang_loc = None
    indNdv = None

def radconst(ConfigOptions):
    """
    Function to calculate the current incoming solar constant.
    :param ConfigOptions:
    :return:
    """
    dCurrent = ConfigOptions.current_output_date
    DEGRAD = math.pi/180.0
    DPD = 360.0/365.0

    # For short wave radiation
    DECLIN = 0.0
    SOLCON = 0.0

    # Calculate the current julian day.
    JULIAN = time.strptime(dCurrent.strftime('%Y.%m.%d'), '%Y.%m.%d').tm_yday

    # OBECL : OBLIQUITY = 23.5 DEGREE
    OBECL = 23.5 * DEGRAD
    SINOB = math.sin(OBECL)

    # Calculate longitude of the sun from vernal equinox
    if JULIAN >= 80:
        SXLONG = DPD * (JULIAN - 80)
    if JULIAN < 80:
        SXLONG = DPD * (JULIAN + 285)
    SXLONG = SXLONG*DEGRAD
    ARG = SINOB * math.sin(SXLONG)
    DECLIN = math.asin(ARG)
    DECDEG = DECLIN / DEGRAD

    # Solar constant eccentricity factor (Paltridge and Platt 1976)
    DJUL = JULIAN * 360.0 / 365.0
    RJUL = DJUL * DEGRAD
    ECCFAC = 1.000110 + (0.034221 * math.cos(RJUL)) + (0.001280 * math.sin(RJUL)) + \
             (0.000719 * math.cos(2 * RJUL)) + (0.000077 * math.sin(2 * RJUL))
    SOLCON = 1370.0 * ECCFAC

    return DECLIN, SOLCON

def calc_coszen(ConfigOptions,declin,GeoMetaWrfHydro):
    """
    Downscaling function to compute radiation terms based on current datetime
    information and lat/lon grids.
    :param ConfigOptions:
    :param input_forcings:
    :param declin:
    :return:
    """
    degrad = math.pi / 180.0
    gmt = 0

    # Calculate the current julian day.
    dCurrent = ConfigOptions.current_output_date
    julian = time.strptime(dCurrent.strftime('%Y.%m.%d'), '%Y.%m.%d').tm_yday

    da = 6.2831853071795862 * ((julian - 1) / 365.0)
    eot = ((0.000075 + 0.001868 * math.cos(da)) - (0.032077 * math.sin(da)) - \
           (0.014615 * math.cos(2 * da)) - (0.04089 * math.sin(2 * da))) * 229.18
    xtime = dCurrent.hour * 60.0  # Minutes of day
    xt24 = int(xtime) % 1440 + eot
    tloctm = GeoMetaWrfHydro.longitude_grid/15.0 + gmt + xt24/60.0
    hrang = ((tloctm - 12.0) * degrad) * 15.0
    xxlat = GeoMetaWrfHydro.latitude_grid * degrad
    coszen = np.sin(xxlat) * math.sin(declin) + np.cos(xxlat) * math.cos(declin) * np.cos(hrang)

    # Reset temporary variables to free up memory.
    tloctm = None
    xxlat = None

    return coszen, hrang

def TOPO_RAD_ADJ_DRVR(GeoMetaWrfHydro,input_forcings,COSZEN,declin,solcon,hrang2d):
    """
    Downscaling driver for correcting incoming shortwave radiation fluxes from a low
    resolution to a a higher resolution.
    :param GeoMetaWrfHydro:
    :param input_forcings:
    :param COSZEN:
    :param declin:
    :param solcon:
    :param hrang2d:
    :return:
    """
    degrad = math.pi / 180.0

    ny = GeoMetaWrfHydro.ny_local
    nx = GeoMetaWrfHydro.nx_local

    xxlat = GeoMetaWrfHydro.latitude_grid*degrad

    # Sanity checking on incoming shortwave grid.
    SWDOWN = input_forcings.final_forcings[7,:,:]
    SWDOWN[np.where(SWDOWN < 0.0)] = 0.0
    SWDOWN[np.where(SWDOWN >= 1400.0)] = 1400.0

    COSZEN[np.where(COSZEN < 1E-4)] = 1E-4

    corr_frac = np.empty([ny, nx], np.int)
    # shadow_mask = np.empty([ny,nx],np.int)
    diffuse_frac = np.empty([ny, nx], np.int)
    corr_frac[:, :] = 0
    diffuse_frac[:, :] = 0
    # shadow_mask[:,:] = 0

    indTmp = np.where((GeoMetaWrfHydro.slope[:,:] == 0.0) &
                      (SWDOWN <= 10.0))
    corr_frac[indTmp] = 1

    term1 = np.sin(xxlat) * np.cos(hrang2d)
    term2 = ((0 - np.cos(GeoMetaWrfHydro.slp_azi)) *
             np.sin(GeoMetaWrfHydro.slope))
    term3 = np.sin(hrang2d) * (np.sin(GeoMetaWrfHydro.slp_azi) *
                               np.sin(GeoMetaWrfHydro.slope))
    term4 = (np.cos(xxlat) * np.cos(hrang2d)) * np.cos(GeoMetaWrfHydro.slope)
    term5 = np.cos(xxlat) * (np.cos(GeoMetaWrfHydro.slp_azi) *
                             np.sin(GeoMetaWrfHydro.slope))
    term6 = np.sin(xxlat) * np.cos(GeoMetaWrfHydro.slope)

    csza_slp = (term1 * term2 - term3 + term4) * math.cos(declin) + \
               (term5 + term6) * math.sin(declin)

    csza_slp[np.where(csza_slp <= 1E-4)] = 1E-4
    # Topographic shading
    # csza_slp[np.where(shadow == 1)] = 1E-4

    # Correction factor for sloping topographic: the diffuse fraction of solar
    # radiation is assumed to be unaffected by the slope.
    corr_fac = diffuse_frac + ((1 - diffuse_frac) * csza_slp) / COSZEN
    corr_fac[np.where(corr_fac > 1.3)] = 1.3

    # Peform downscaling
    SWDOWN_OUT = SWDOWN * corr_fac

    # Reset variables to free up memory
    # corr_frac = None
    diffuse_frac = None
    term1 = None
    term2 = None
    term3 = None
    term4 = None
    term5 = None
    term6 = None

    input_forcings.final_forcings[7,:,:] = SWDOWN_OUT

    # Reset variables to free up memory
    SWDOWN = None
    SWDOWN_OUT = None

def rel_hum(input_forcings,ConfigOptions):
    """
    Function to calculate relative humidity given
    original, undownscaled surface pressure and 2-meter
    temperature.
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    tmpHumidity = input_forcings.final_forcings[5,:,:]/(1-input_forcings.final_forcings[5,:,:])

    T0 = 273.15
    EP = 0.622
    ONEMEP = 0.378
    ES0 = 6.11
    A = 17.269
    B = 35.86

    EST = ES0 * np.exp((A * (input_forcings.t2dTmp - T0)) / (input_forcings.t2dTmp - B))
    QST = (EP * EST) / ((input_forcings.psfcTmp * 0.01) - ONEMEP * EST)
    RH = 100 * (tmpHumidity / QST)

    # Reset variables to free up memory
    tmpHumidity = None

    return RH

def mixhum_ptrh(input_forcings,relHum,iswit,ConfigOptions):
    """
    Functionto convert relative humidity back to a downscaled
    2-meter specific humidity
    :param input_forcings:
    :param ConfigOptions:
    :return:
    """
    T0 = 273.15
    EP = 0.622
    ONEMEP = 0.378
    ES0 = 6.11
    A = 17.269
    B = 35.86

    term1 = A * (input_forcings.final_forcings[4,:,:] - T0)
    term2 = input_forcings.final_forcings[4,:,:] - B
    EST = np.exp(term1 / term2) * ES0

    QST = (EP * EST) / ((input_forcings.final_forcings[6,:,:]/100.0) - ONEMEP * EST)
    QW = QST * (relHum * 0.01)
    if iswit == 2:
        QW = QW / (1.0 + QW)
    if iswit < 0:
        QW = QW * 1000.0

    # Reset variables to free up memory
    term1 = None
    term2 = None
    EST = None
    QST = None
    psfcTmp = None

    return QW
