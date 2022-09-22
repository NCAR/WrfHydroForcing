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
    OutputEnum = ConfigOptions.OutputEnum
    # Loop through the forcing products to layer in

    for force_id in OutputEnum: #len of OutputEnum class
        force_name = force_id.name
        if force_name in input_forcings.input_map_output:
            force_idx = input_forcings.input_map_output.index(force_name)
            outLayerCurrent = OutputObj.output_local[force_idx,:,:]
            layerIn = input_forcings.final_forcings[force_idx,:,:]
            indSet = np.where(layerIn != ConfigOptions.globalNdv)
            outLayerCurrent[indSet] = layerIn[indSet]
            OutputObj.output_local[force_idx, :, :] = outLayerCurrent

            # Reset for next iteration and memory efficiency.
            indSet = None
    # MpiConfig.comm.barrier()


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

    OutputEnum = ConfigOptions.OutputEnum
    indSet = np.where(supplemental_precip.final_supp_precip != ConfigOptions.globalNdv)
    layerIn = supplemental_precip.final_supp_precip
    for ind, xvar in enumerate(OutputEnum):
        if xvar.name == supplemental_precip.output_var_idx:
            outId = ind
            break
        
    
    layerOut = OutputObj.output_local[outId, :, :]

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
    OutputObj.output_local[outId, :, :] = layerOut

