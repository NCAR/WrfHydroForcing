"""
Temporal interpolation input forcings to the current output timestep.
"""
import sys
from core import errMod
import numpy as np

def no_interpolation(input_forcings,ConfigOptions,MpiConfig):
    """
    Function for simply setting the final regridded fields to the
    input forcings that are from the next input forcing frequency.
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    input_forcings.final_forcings[:,:,:] = input_forcings.regridded_forcings2[:,:,:]

def no_interpolation_supp_pcp(supplemental_precip,ConfigOptions,MpiConfig):
    """
    Function for simply setting the final regridded supplemental precipitation
    to the supplemental precipitation grids from the next precip frequency that
    is available.
    :param supplemental_precip:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    supplemental_precip.final_supp_precip[:,:] = supplemental_precip.regridded_precip2[:,:]

def nearest_neighbor(input_forcings,ConfigOptions,MpiConfig):
    """
    Function for setting the current output regridded forcings to the nearest
    input forecast step.
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Calculate the difference between the current output timestep,
    # and the previous input forecast output step.
    dtFromPrevious = ConfigOptions.current_output_date - input_forcings.fcst_date1

    # Calculate the difference between the current output timesetp,
    # and the next forecast output step.
    dtFromNext = ConfigOptions.current_output_date - input_forcings.fcst_date2

    if abs(dtFromNext.total_seconds()) <= abs(dtFromPrevious.total_seconds()):
        # Default to the regridded states from the next forecast output step.
        input_forcings.final_forcings[:,:,:] = input_forcings.regridded_forcings2[:,:,:]
    else:
        # Default to the regridded states from the previous forecast output
        # step.
        input_forcings.final_forcings[:,:,:] = input_forcings.regridded_forcings1[:,:,:]

def nearest_neighbor_supp_pcp(supplemental_precip,ConfigOptions,MpiConfig):
    """
    Function for setting the current output regridded supplemental precipitation
    to the nearest supplemental precipitation input step.
    :param supplemental_precip:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Calculate the difference between the current ouptut timestep,
    # and the previous supplemental input step.
    dtFromPrevious = ConfigOptions.current_output_step - supplemental_precip.pcp_date1

    # Calculate the difference between the current output timestep,
    # and the next supplemental input step.
    dtFromNext = ConfigOptions.current_output_date - supplemental_precip.pcp_date2

    if abs(dtFromNext.total_seconds()) <= abs(dtFromPrevious.total_seconds()):
        # Default to the regridded states from the next forecast output step.
        supplemental_precip.final_supp_precip[:,:] = supplemental_precip.regridded_precip2[:,:]
    else:
        # Default to the regridded states from the previous forecast output
        # step.
        supplemental_precip.final_supp_precip[:,:] = supplemental_precip.regridded_precip1[:,:]

def weighted_average(input_forcings,ConfigOptions,MpiConfig):
    """
    Function for setting the current output regridded fields as a weighted
    average between the previous output step and the next output step.
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Calculate the difference between the current output timestep,
    # and the previous input forecast output step. Use this to calculate a fraction
    # of the previous forcing output to use in the final output for this step.
    dtFromPrevious = ConfigOptions.current_output_date - input_forcings.fcst_date1
    weight1 = 1-(abs(dtFromPrevious.total_seconds())/(input_forcings.outFreq*60.0))

    # Calculate the difference between the current output timesetp,
    # and the next forecast output step. Use this to calculate a fraction of
    # the next forcing output to use in the final output for this step.
    dtFromNext = ConfigOptions.current_output_date - input_forcings.fcst_date2
    weight2 = 1-(abs(dtFromNext.total_seconds())/(input_forcings.outFreq*60.0))

    # Calculate where we have missing data in either the previous or next forcing dataset.
    ind1Ndv = np.where(input_forcings.regridded_forcings1 == ConfigOptions.globalNdv)
    ind2Ndv = np.where(input_forcings.regridded_forcings2 == ConfigOptions.globalNdv)

    input_forcings.final_forcings[:,:,:] = input_forcings.regridded_forcings1[:,:,:]*weight1 + \
        input_forcings.regridded_forcings2[:,:,:]*weight2

    # Set any pixel cells that were missing for either window to missing value.
    input_forcings.final_forcings[ind1Ndv] = ConfigOptions.globalNdv
    input_forcings.final_forcings[ind2Ndv] = ConfigOptions.globalNdv

    # Reset for memory efficiency.
    ind1Ndv = None
    ind2Ndv = None

def weighted_average_supp_pcp(supplemental_precip,ConfigOptions,MpiConfig):
    """
    Function for setting the current output regridded supplemental precipitation fields
    as an average between the previous and next input supplemental precipitation timesteps.
    :param supplemental_precip:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Calculate the difference between the current output timestep,
    # and the previous input supp pcp step. Use this to calculate a fraction
    # of the previous supp pcp to use in the final output for this step.
    dtFromPrevious = ConfigOptions.current_output_date - supplemental_precip.pcp_date1
    weight1 = 1 - (abs(dtFromPrevious.total_seconds()) / (supplemental_precip.input_frequency * 60.0))

    # Calculate the difference between the current output timesetp,
    # and the next input supp pcp step. Use this to calculate a fraction of
    # the next forcing supp pcp to use in the final output for this step.
    dtFromNext = ConfigOptions.current_output_date - supplemental_precip.pcp_date2
    weight2 = 1 - (abs(dtFromNext.total_seconds()) / (supplemental_precip.input_frequency * 60.0))

    # Calculate where we have missing data in either the previous or next forcing dataset.
    ind1Ndv = np.where(supplemental_precip.regridded_precip1 == ConfigOptions.globalNdv)
    ind2Ndv = np.where(supplemental_precip.regridded_precip2 == ConfigOptions.globalNdv)

    supplemental_precip.final_supp_precip[:,:] = supplemental_precip.regridded_precip1[:,:] * weight1 + \
                                                 supplemental_precip.regridded_precip2[:,:] * weight2

    # Set any pixel cells that were missing for either window to missing value.
    supplemental_precip.final_supp_precip[ind1Ndv] = ConfigOptions.globalNdv
    supplemental_precip.final_supp_precip[ind2Ndv] = ConfigOptions.globalNdv

    # Reset for memory efficiency.
    ind1Ndv = None
    ind2Ndv = None

