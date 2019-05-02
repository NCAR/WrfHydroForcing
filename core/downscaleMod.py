"""
Module file for handling of downscaling input regridded (possibly bias-corrected)
forcing fields. Each input forcing product will loop through and determine if
downscaling is needed, based off options specified by the user.
"""
import math
import time
import numpy as np
from core import errMod
from netCDF4 import Dataset
import os

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
        1: simple_lapse,
        2: param_lapse
    }
    downscale_temperature[input_forcings.t2dDownscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary mapping to pressure downscaling.
    downscale_pressure = {
        0: no_downscale,
        1: pressure_down_classic
    }
    downscale_pressure[input_forcings.psfcDownscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary mapping to shortwave radiation downscaling
    downscale_sw = {
        0: no_downscale,
        1: ncar_topo_adj
    }
    downscale_sw[input_forcings.swDowscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary mapping to specific humidity downscaling
    downscale_q2 = {
        0: no_downscale,
        1: q2_down_classic
    }
    downscale_q2[input_forcings.q2dDownscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    # Dictionary mapping to precipitation downscaling.
    downscale_precip = {
        0: no_downscale,
        1: nwm_monthly_PRISM_downscale
        #1: precip_mtn_mapper
    }
    downscale_precip[input_forcings.precipDownscaleOpt](input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

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

def param_lapse(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    Function that applies a apriori lapse rate adjustment to modeled
    2-meter temperature by taking the difference of the native
    input elevation and the WRF-hydro elevation. It's assumed this lapse
    rate grid has already been regridded to the final output WRF-Hydro
    grid.
    :param inpute_forcings:
    :param ConfigOptions:
    :param GeoMetaWrfHydro:
    :return:
    """
    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Applying aprior lapse rate grid to temperature downscaling"
        errMod.log_msg(ConfigOptions,MpiConfig)

    # Calculate the elevation difference.
    elevDiff = input_forcings.height - GeoMetaWrfHydro.height

    if not input_forcings.lapseGrid:
        # We have not read in our lapse rate file. Read it in, do extensive checks,
        # scatter the lapse rate grid out to individual processors, then apply the
        # lapse rate to the 2-meter temperature grid.
        if MpiConfig.rank == 0:
            # First ensure we have a parameter directory
            if input_forcings.paramDir == "NONE":
                ConfigOptions.errMsg = "User has specified spatial temperature lapse rate " \
                                       "downscaling while no downscaling parameter directory " \
                                       "exists."
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            # Compose the path to the lapse rate grid file.
            lapsePath = input_forcings.paramDir + "/lapse_param.nc"
            if not os.path.isfile(lapsePath):
                ConfigOptions.errMsg = "Expected lapse rate parameter file: " + \
                                       lapsePath + " does not exist."
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            # Open the lapse rate file. Check for the expected variable, along with
            # the dimension size to make sure everything matches up.
            try:
                idTmp = Dataset(lapsePath,'r')
            except:
                ConfigOptions.errMsg = "Unable to open parameter file: " + lapsePath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if not 'lapse' in idTmp.variables.keys():
                ConfigOptions.errMsg = "Expected 'lapse' variable not located in parameter " \
                                       "file: " + lapsePath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            try:
                lapseTmp = idTmp.variables['lapse'][:,:]
            except:
                ConfigOptions.errMsg = "Unable to extracte 'lapse' variable from parameter: " \
                                       "file: " + lapsePath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            # Check dimensions to ensure they match up to the output grid.
            if lapseTmp.shape[0] != GeoMetaWrfHydro.nx_global:
                ConfigOptions.errMsg = "X-Dimension size mismatch between output grid and lapse " \
                                       "rate from parameter file: " + lapsePath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if lapseTmp.shape[1] != GeoMetaWrfHydro.ny_global:
                ConfigOptions.errMsg = "Y-Dimension size mismatch between output grid and lapse " \
                                       "rate from parameter file: " + lapsePath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            # Perform a quick search to ensure we don't have radical values.
            indTmp = np.where(lapseTmp < 0.0)
            if len(indTmp[0]) > 0:
                ConfigOptions.errMsg = "Found missing values in the lapse rate grid from " \
                                       "parameter file: " + lapsePath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            indTmp = np.where(lapseTmp > 100.0)
            if len(indTmp[0]) > 0:
                ConfigOptions.errMsg = "Found excessively high values in the lapse rate grid from " \
                                       "parameter file: " + lapsePath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            # Close the parameter lapse rate file.
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close parameter file: " + lapsePath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            lapseTmp = None
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Scatter the lapse rate grid to the other processors.
        input_forcings.lapseGrid = MpiConfig.scatter_array(GeoMetaWrfHydro,lapseTmp,ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

    # Apply the local lapse rate grid to our local slab of 2-meter temperature data.
    temperature_grid_tmp = input_forcings.final_forcings[4, :, :]
    try:
        indNdv = np.where(input_forcings.final_forcings == ConfigOptions.globalNdv)
    except:
        ConfigOptions.errMsg = "Unable to perform NDV search on input " + \
                               input_forcings.productName + " regridded forcings."
        errMod.log_critical(ConfigOptions, MpiConfig)
        return
    try:
        indValid = np.where(temperature_grid_tmp != ConfigOptions.globalNdv)
    except:
        ConfigOptions.errMsg = "Unable to perform search for valid values on input " + \
                               input_forcings.productName + " regridded temperature forcings."
        errMod.log_critical(ConfigOptions, MpiConfig)
        return
    try:
        temperature_grid_tmp[indValid] = temperature_grid_tmp[indValid] + \
                                         ((input_forcings.lapseGrid[indValid]/1000.0) * elevDiff)
    except:
        ConfigOptions.errMsg = "Unable to apply spatial lapse rate values to input " + \
                               input_forcings.productName + " regridded temperature forcings."
        errMod.log_critical(ConfigOptions, MpiConfig)
        return

    input_forcings.final_forcings[4,:,:] = temperature_grid_tmp
    input_forcings.final_forcings[indNdv] = ConfigOptions.globalNdv

    # Reset for memory efficiency
    indTmp = None
    indNdv = None
    indValid = None
    elevDiff = None
    temperature_grid_tmp = None

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
        relHum = rel_hum(input_forcings,ConfigOptions)
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

def nwm_monthly_PRISM_downscale(input_forcings,ConfigOptions,GeoMetaWrfHydro,MpiConfig):
    """
    NCAR/OWP function for downscaling precipitation using monthly PRISM climatology in a
    mountain-mapper like fashion.
    :param input_forcings:
    :param ConfigOptions:
    :param GeoMetaWrfHydro:
    :return:
    """
    if MpiConfig.rank == 0:
        ConfigOptions.statusMsg = "Performing NWM Monthly PRISM Mountain Mapper " \
                                  "Downscaling of Precipitation"
        errMod.log_msg(ConfigOptions,MpiConfig)
        print(ConfigOptions.statusMsg)

    # Establish whether or not we need to read in new PRISM monthly climatology:
    # 1.) This is the first output timestep, and no grids have been initialized.
    # 2.) We have switched months from the last timestep. In this case, we need
    #     to re-initialize the grids for the current month.
    initializeFlag = False
    if input_forcings.nwmPRISM_denGrid is None and input_forcings.nwmPRISM_numGrid is None:
        # We are on situation 1 - This is the first output step.
        initializeFlag = True
        print('WE NEED TO READ IN PRISM GRIDS')
    if ConfigOptions.current_output_date.month != ConfigOptions.prev_output_date.month:
        # We are on situation #2 - The month has changed so we need to reinitialize the
        # PRISM grids.
        initializeFlag = True
        print('MONTH CHANGE.... NEED TO READ IN NEW PRISM GRIDS.')

    if initializeFlag == True:
        # First reset the local PRISM grids to be safe.
        input_forcings.nwmPRISM_numGrid = None
        input_forcings.nwmPRISM_denGrid = None

        # Compose paths to the expected files.
        numeratorPath = input_forcings.paramDir + "/PRISM_Precip_Clim_" + \
                        ConfigOptions.current_output_date.strftime('%h') + '_NWM_Mtn_Mapper_Numer.nc'
        denominatorPath = input_forcings.paramDir + "/PRISM_Precip_Clim_" + \
                          ConfigOptions.current_output_date.strftime('%h') + '_NWM_Mtn_Mapper_Denom.nc'
        print(numeratorPath)
        print(denominatorPath)

        # Make sure files exist.
        if not os.path.isfile(numeratorPath):
            ConfigOptions.errMsg = "Expected parameter file: " + numeratorPath + " for mountain mapper downscaling " \
                                                                                 "of precipitation not found."
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if not os.path.isfile(denominatorPath):
            ConfigOptions.errMsg = "Expected parameter file: " + denominatorPath + " for mountain mapper downscaling " \
                                                                                   "of precipitation not found."
            errMod.log_critical(ConfigOptions, MpiConfig)
            pass

        if MpiConfig.rank == 0:
            # Open the NetCDF parameter files. Check to make sure expected dimension sizes are in place, along with
            # variable names, etc.
            try:
                idNum = Dataset(numeratorPath,'r')
            except:
                ConfigOptions.errMsg = "Unable to open parameter file: " + numeratorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            try:
                idDenom = Dataset(denominatorPath,'r')
            except:
                ConfigOptions.errMsg = "Unable to open parameter file: " + denominatorPath
                errMod.log_critical(ConfigOptions,MpiConfig)
                pass

            # Check to make sure expected names, dimension sizes are present.
            if 'x' not in idNum.variables.keys():
                ConfigOptions.errMsg = "Expected 'x' variable not found in parameter file: " + numeratorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if 'x' not in idDenom.variables.keys():
                ConfigOptions.errMsg = "Expected 'x' variable not found in parameter file: " + denominatorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            if 'y' not in idNum.variables.keys():
                ConfigOptions.errMsg = "Expected 'y' variable not found in parameter file: " + numeratorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if 'y' not in idDenom.variables.keys():
                ConfigOptions.errMsg = "Expected 'y' variable not found in parameter file: " + denominatorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            if 'Data' not in idNum.variables.keys():
                ConfigOptions.errMsg = "Expected 'Data' variable not found in parameter file: " + numeratorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if 'Data' not in idDenom.variables.keys():
                ConfigOptions.errMsg = "Expected 'Data' variable not found in parameter file: " + denominatorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            if idNum.variables['Data'].shape[0] != GeoMetaWrfHydro.ny_global:
                ConfigOptions.errMsg = "Input Y dimension for: " + numeratorPath + " does not match the output WRF-Hydro " \
                                                                                   " Y dimension size."
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if idDenom.variables['Data'].shape[0] != GeoMetaWrfHydro.ny_global:
                ConfigOptions.errMsg = "Input Y dimension for: " + denominatorPath + " does not match the output WRF-Hydro " \
                                                                                 " Y dimension size."
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            if idNum.variables['Data'].shape[1] != GeoMetaWrfHydro.nx_global:
                ConfigOptions.errMsg = "Input X dimension for: " + numeratorPath + " does not match the output WRF-Hydro " \
                                                                               " X dimension size."
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            if idDenom.variables['Data'].shape[1] != GeoMetaWrfHydro.nx_global:
                ConfigOptions.errMsg = "Input X dimension for: " + denominatorPath + " does not match the output WRF-Hydro " \
                                                                                 " X dimension size."
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            # Read in the PRISM grid on the output grid. Then scatter the array out to the processors.
            try:
                numDataTmp = idNum.variables['Data'][:,:]
            except:
                ConfigOptions.errMsg = "Unable to extract 'Data' from parameter file: " + numeratorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            try:
                denDataTmp = idDenom.variables['Data'][:,:]
            except:
                ConfigOptions.errMsg = "Unable to extract 'Data' from parameter file: " + denominatorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass

            # Close the parameter files.
            try:
                idNum.close()
            except:
                ConfigOptions.errMsg = "Unable to close parameter file: " + numeratorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
            try:
                idDenom.close()
            except:
                ConfigOptions.errMsg = "Unable to close parameter file: " + denominatorPath
                errMod.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            numDataTmp = None
            denDataTmp = None
        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Scatter the array out to the local processors
        input_forcings.nwmPRISM_numGrid = MpiConfig.scatter_array(GeoMetaWrfHydro, numDataTmp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

        input_forcings.nwmPRISM_denGrid = MpiConfig.scatter_array(GeoMetaWrfHydro, denDataTmp, ConfigOptions)
        errMod.check_program_status(ConfigOptions, MpiConfig)

    # Create temporary grids from the local slabs of params/precip forcings.
    localRainRate = input_forcings.final_forcings[5,:,:]
    numLocal = input_forcings.nwmPRISM_numGrid[:,:]
    denLocal = input_forcings.nwmPRISM_denGrid[:,:]

    # Establish index of where we have valid data.
    try:
        indValid = np.where((localRainRate > 0.0) & (denLocal > 0.0) & (numLocal > 0.0))
    except:
        ConfigOptions.errMsg = "Unable to run numpy search for valid values on precip and " \
                               "param grid in mountain mapper downscaling"
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        localRainRate[indValid] = localRainRate[indValid] * numLocal[indValid]
    except:
        ConfigOptions.errMsg = "Unable to multiply precip by numerator in mountain mapper downscaling"
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    try:
        localRainRate[indValid] = localRainRate[indValid] / denLocal[indValid]
    except:
        ConfigOptions.errMsg = "Unable to divide precip by denominator in mountain mapper downscaling"
        errMod.log_critical(ConfigOptions, MpiConfig)
    errMod.check_program_status(ConfigOptions, MpiConfig)

    input_forcings.final_forcings[5, :, :] = localRainRate

    # Reset variables for memory efficiency
    idDenom = None
    idNum = None
    localRainRate = None
    numLocal = None
    denLocal = None

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
