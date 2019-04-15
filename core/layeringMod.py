"""
Layering module for implementing various layering schemes in the WRF-Hydro forcing engine.
Future functionality may include blenidng, etc.
"""
import numpy as np

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
    # Loop through the 8 forcing products to layer in:
    # 1.) U-Wind (m/s)
    # 2.) V-Wind (m/s)
    # 3.) Surface incoming longwave radiation flux (W/m^2)
    # 4.) Precipitation rate (mm/s)
    # 5.) 2-meter temperature (K)
    # 6.) 2-meter specific humidity (kg/kg)
    # 7.) Surface pressure (Pa)
    # 8.) Surface incoming shortwave radiation flux (W/m^2)

    for forceTmp in range(0,8):
        outLayerCurrent = OutputObj.output_local[forceTmp,:,:]
        layerIn = input_forcings.final_forcings[forceTmp,:,:]
        indSet = np.where(layerIn != ConfigOptions.globalNdv)
        outLayerCurrent[indSet] = layerIn[indSet]
        OutputObj.output_local[forceTmp,:,:] = outLayerCurrent

        # Reset for next iteration and memory efficiency.
        indSet = None
    MpiConfig.comm.barrier()

def layer_supplemental_precipitation(OutputObj,supplemental_precip,ConfigOptions,MpiConfig):
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
    layerOut = OutputObj.output_local[3,:,:]
    layerOut[indSet] = layerIn[indSet]

    OutputObj.output_local[3,:,:] = layerOut

    # Reset local variables to reduce memory usage.
    layerIn = None
    layerOut = None
    indSet = None

    MpiConfig.comm.barrier()
