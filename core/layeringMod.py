"""
Layering module for implementing various layering schemes in the WRF-Hydro forcing engine.
Future functionality may include blenidng, etc.
"""
import numpy as np
from core import err_handler

def layer_final_forcings(OutputObj,input_forcings,ConfigOptions,MpiConfig):
    """
    Function to perform basic layering of input forcings as they are processed. The logic
    works as following:
    1.) As the parent calling program loops through the forcings for each layer
        for this timestep, forcings are placed onto the output grid by shear brute
        replacement. However, this only occurs where valid data exists.
        Supplemental precipitation will be layered in separately.
    :param OutputObj:
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Loop through the 8 (or 9) forcing products to layer in:
    # 0.) U-Wind (m/s)
    # 1.) V-Wind (m/s)
    # 2.) Surface incoming longwave radiation flux (W/m^2)
    # 3.) Precipitation rate (mm/s)
    # 4.) 2-meter temperature (K)
    # 5.) 2-meter specific humidity (kg/kg)
    # 6.) Surface pressure (Pa)
    # 7.) Surface incoming shortwave radiation flux (W/m^2)
    # 8.) Liquid fraction of precipitation ([0..1])

    force_count = 9 if ConfigOptions.include_lqfrac else 8
    for force_idx in range(0, force_count):
        if force_idx in input_forcings.input_map_output:
            outLayerCurrent = OutputObj.output_local[force_idx,:,:]
            layerIn = input_forcings.final_forcings[force_idx,:,:]
            indSet = np.where(layerIn != ConfigOptions.globalNdv)
            outLayerCurrent[indSet] = layerIn[indSet]

            if force_idx == 8:
                calcSet = np.where(np.logical_and(outLayerCurrent != ConfigOptions.globalNdv, np.logical_or(outLayerCurrent < 0, outLayerCurrent > 1)))
                # if calcSet[0].size > 0:
                #     print(f"WARNING: Liquid fraction of precipitation outside of valid range [0..1] detected. {indSet[0].size} values found.")
                #     print(f"{np.histogram(np.where(outLayerCurrent[calcSet] < 0, outLayerCurrent[calcSet], 0), 2)}")
                outLayerCurrent[calcSet] =  np.where(OutputObj.output_local[4,:,:] >= 273.15+2.2, 1.0, 0.0)[calcSet] # 2.2C threshold for rain/snow

            OutputObj.output_local[force_idx, :, :] = outLayerCurrent
            # Reset for next iteration and memory efficiency.
            indSet = None


def layer_supplemental_forcing(OutputObj, supplemental_precip, ConfigOptions, MpiConfig):
    """
    Function to layer in supplemental precipitation where we have valid values. Any pixel
    cells that contain missing values will not be layered in, and background input forcings
    will be used instead.
    :param OutputObj:
    :param supplemental_precip:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    indSet = np.where(supplemental_precip.final_supp_precip != ConfigOptions.globalNdv)
    layerIn = supplemental_precip.final_supp_precip
    layerOut = OutputObj.output_local[supplemental_precip.output_var_idx, :, :]

    #TODO: review test layering for ExtAnA calculation to replace FE QPE with MPE RAINRATE
    #If this isn't sufficient, replace QPE with MPE here:
    #if supplemental_precip.keyValue == 11:
    #    ConfigOptions.statusMsg = "Performing ExtAnA calculation"
    #    err_handler.log_msg(ConfigOptions, MpiConfig)

    if len(indSet[0]) != 0:
        layerOut[indSet] = layerIn[indSet]
    else:
        # We have all missing data for the supplemental precip for this step.
        layerOut = layerOut

    # TODO: test that even does anything...?s
    OutputObj.output_local[supplemental_precip.output_var_idx, :, :] = layerOut

