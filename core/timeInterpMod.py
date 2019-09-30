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
    # Check to make sure we have valid grids.
    if input_forcings.regridded_forcings2 is None:
        input_forcings.final_forcings[:, :, :] = ConfigOptions.globalNdv
    else:
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
    if supplemental_precip.regridded_precip2 is not None:
        supplemental_precip.final_supp_precip[:,:] = supplemental_precip.regridded_precip2[:,:]
    else:
        # We have missing files.
        supplemental_precip.final_supp_precip[:,:] = ConfigOptions.globalNdv

def nearest_neighbor(input_forcings,ConfigOptions,MpiConfig):
    """
    Function for setting the current output regridded forcings to the nearest
    input forecast step.
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # If we are running CFSv2 with bias correction, bypass as temporal interpolation is done
    # internally (NWM-only).
    if ConfigOptions.runCfsNldasBiasCorrect and input_forcings.productName == "CFSv2_6Hr_Global_GRIB2":
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Bypassing temporal interpolation routine due to NWM bias correction for CFSv2"
            errMod.log_msg(ConfigOptions, MpiConfig)
            return

    # Calculate the difference between the current output timestep,
    # and the previous input forecast output step.
    dtFromPrevious = ConfigOptions.current_output_date - input_forcings.fcst_date1

    # Calculate the difference between the current output timesetp,
    # and the next forecast output step.
    dtFromNext = ConfigOptions.current_output_date - input_forcings.fcst_date2

    if abs(dtFromNext.total_seconds()) <= abs(dtFromPrevious.total_seconds()):
        # Default to the regridded states from the next forecast output step.
        if input_forcings.regridded_forcings2 is None:
            input_forcings.final_forcings[:, :, :] = ConfigOptions.globalNdv
        else:
            input_forcings.final_forcings[:,:,:] = input_forcings.regridded_forcings2[:,:,:]
    else:
        # Default to the regridded states from the previous forecast output
        # step.
        if input_forcings.regridded_forcings1 is None:
            input_forcings.final_forcings[:, :, :] = ConfigOptions.globalNdv
        else:
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
    if supplemental_precip.regridded_precip2 is not None and supplemental_precip.regridded_precip1 is not None:
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
    else:
        # We have missing files.
        supplemental_precip.final_supp_precip[:, :] = ConfigOptions.globalNdv

def weighted_average(input_forcings,ConfigOptions,MpiConfig):
    """
    Function for setting the current output regridded fields as a weighted
    average between the previous output step and the next output step.
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Check to make sure we have valid grids.
    if input_forcings.regridded_forcings2 is None:
        input_forcings.final_forcings[:, :, :] = ConfigOptions.globalNdv
        return
    if input_forcings.regridded_forcings1 is None:
        input_forcings.final_forcings[:, :, :] = ConfigOptions.globalNdv
        return

    # If we are running CFSv2 with bias correction, bypass as temporal interpolation is done
    # internally (NWM-only).
    if ConfigOptions.runCfsNldasBiasCorrect and input_forcings.productName == "CFSv2_6Hr_Global_GRIB2":
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Bypassing temporal interpolation routine due to NWM bias correction for CFSv2"
            errMod.log_msg(ConfigOptions, MpiConfig)
            return

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
    if supplemental_precip.regridded_precip2 is not None and supplemental_precip.regridded_precip1 is not None:
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
    else:
        # We have missing files.
        supplemental_precip.final_supp_precip[:, :] = ConfigOptions.globalNdv

def gfs_pcp_time_interp(input_forcings,ConfigOptions,MpiConfig):
    """
    Function that will calculate an instantaneous precipitation rate, representative
    of the latest forecast GFS hour, or range of GFS forecast hours. This is
    done as GFS has a quirky way of outputting precipitation rates.
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return: instPcpGlobal
    """
    # There is a chain of logic we will follow here for GFS data.
    # Forecast Hours <= 120:
    # 1.) For first hour in every six hour period, the avg precip rate
    #     will be treated as the instantaneous rate. No processing
    #     needed. So 0-1, 6-7, 12-13, etc.
    # 2.) For remaining hours, we need to calculate the difference between
    #     this avg precip rate, and the previous one. So,
    #     [(0-4)-(0-3)] gives us a precipitation rate for hour 4.
    # Forecast Hours > 120:
    # 1.) For the first three hours of every six hour period, we
    #     will treat the avg precip rate coming in as the instantaneous
    #     value at hour 3. This is because there are no values for
    #     individual hours within this horizon. So, 120-123, 126-129, etc
    # 2.) For other three hours, we need to calculate the difference
    #     between the previous three hourly average rate, and the average
    #     rate for this entire six hour time frame. Why is this the case
    #     with GFS? Ask someone at NCEP.......
    #     [(120-126)-(120-123)] gives us a precipitation rate representative
    #     of the second three hour period. It's not necessarily instantaneous,
    #     but we will treat it as such.
    if input_forcings.fcst_hour2 <= 120:
        if input_forcings.fcst_hour2%6 == 1:
            # We are on the first hour of a six-hour period. We can treat
            # the precipitation rate as instantaneous for this hour.
            instPcpGlobal = input_forcings.globalPcpRate2
            total1 = None
            total2 = None
        else:
            # We need to calculate the difference from the previous
            # avg window to get an instantaneous value for this hour.
            total1 = input_forcings.globalPcpRate1 * (3600.0 * (input_forcings.fcst_hour1%6))
            if input_forcings.fcst_hour2%6 == 0:
                # We have a 0-6, 6-12, 12-18 avg rate....
                total2 = input_forcings.globalPcpRate2 * (3600.0 * 6)
            else:
                # We have 0-5, 0-4, etc
                total2 = input_forcings.globalPcpRate2 * (3600.0 * (input_forcings.fcst_hour2%6))
            instPcpGlobal = (total2 - total1)/3600.0
            # Reset variables to free up memory
            total1 = None
            total2 = None
    else:
        # We are in Situation #2 which currently runs out until the end of the
        # end of the GFS forecast cycle of 384 hours.
        if input_forcings.fcst_hour2%6 == 3:
            # We are on the first 3 hours of a six hour period. Simply treat the average
            # precipitation rate for this time period as the instantaneous precipitation
            # rate.
            instPcpGlobal = input_forcings.globalPcpRate2
            total2 = None
            total1 = None
        else:
            total1 = input_forcings.globalPcpRate1 * (3600.0 * 3.0)
            total2 = input_forcings.globalPcpRate2 * (3600.0 * 6.0)
            instPcpGlobal = (total2 - total1)/(3600.0 * 3.0)
            # Reset variables to free up memory.
            total1 = None
            total2 = None
    # Return the interpolated grid back to the regridding program.

    # Set any negative values to 0.0
    instPcpGlobal[np.where(instPcpGlobal < 0.0)] = 0.0

    return instPcpGlobal

