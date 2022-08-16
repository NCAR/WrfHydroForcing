import datetime
import os

from core import bias_correction
from core import downscale
from core import err_handler
from core import layeringMod
from core import disaggregateMod


def process_forecasts(ConfigOptions, wrfHydroGeoMeta, inputForcingMod, suppPcpMod, MpiConfig, OutputObj):
    """
    Main calling module for running realtime forecasts and re-forecasts.
    :param jobMeta:
    :return:
    """
    # Loop through each WRF-Hydro forecast cycle being processed. Within
    # each cycle, perform the following tasks:
    # 1.) Loop over each output frequency
    # 2.) Determine the input forcing cycle dates (both before and after)
    #     for temporal interpolation, downscaling, and bias correction reasons.
    # 3.) If the input forcings haven't been opened and read into memory,
    #     open them.
    # 4.) Check to see if the ESMF objects for input forcings have been
    #     created. If not, create them, including the regridding object.
    # 5.) Regrid forcing grids for input cycle dates surrounding the
    #     current output timestep if they haven't been regridded.
    # 6.) Perform bias correction and/or downscaling.
    # 7.) Output final grids to LDASIN NetCDF files with associated
    #     WRF-Hydro geospatial metadata to the final output directories.
    # Throughout this entire process, log progress being made into LOG
    # files. Once a forecast cycle is complete, we will touch an empty
    # 'WrfHydroForcing.COMPLETE' flag in the directory. This will be
    # checked upon the beginning of this program to see if we
    # need to process any files.

    disaggregate_fun = disaggregateMod.disaggregate_factory(ConfigOptions)

    for fcstCycleNum in range(ConfigOptions.nFcsts):
        ConfigOptions.current_fcst_cycle = ConfigOptions.b_date_proc + datetime.timedelta(
            seconds=ConfigOptions.fcst_freq * 60 * fcstCycleNum)
        if ConfigOptions.first_fcst_cycle is None:
            ConfigOptions.first_fcst_cycle = ConfigOptions.current_fcst_cycle

        if ConfigOptions.ana_flag:
            fcstCycleOutDir = ConfigOptions.output_dir + "/" + ConfigOptions.e_date_proc.strftime('%Y%m%d%H')
        else:
            fcstCycleOutDir = ConfigOptions.output_dir + "/" + ConfigOptions.current_fcst_cycle.strftime('%Y%m%d%H')

        # reset skips if present
        for forceKey in ConfigOptions.input_forcings:
            inputForcingMod[forceKey].skip = False

        # put all AnA output in the same directory
        if ConfigOptions.ana_flag:
            if ConfigOptions.ana_out_dir is None:
                ConfigOptions.ana_out_dir = fcstCycleOutDir
            fcstCycleOutDir = ConfigOptions.ana_out_dir

        # completeFlag = ConfigOptions.scratch_dir + "/WrfHydroForcing.COMPLETE"
        completeFlag = fcstCycleOutDir + "/WrfHydroForcing.COMPLETE"
        if os.path.isfile(completeFlag):
            ConfigOptions.statusMsg = "Forecast Cycle: " + \
                                      ConfigOptions.current_fcst_cycle.strftime('%Y-%m-%d %H:%M') + \
                                      " has already completed."
            err_handler.log_msg(ConfigOptions, MpiConfig)
            # We have already completed processing this cycle,
            # move on.
            continue

        if (not ConfigOptions.ana_flag) or (ConfigOptions.logFile is None):
            if MpiConfig.rank == 0:
                # If the cycle directory doesn't exist, create it.
                if not os.path.isdir(fcstCycleOutDir):
                    try:
                        os.mkdir(fcstCycleOutDir)
                    except:
                        ConfigOptions.errMsg = "Unable to create output " \
                                               "directory: " + fcstCycleOutDir
                        err_handler.err_out_screen_para(ConfigOptions.errMsg, MpiConfig)
            err_handler.check_program_status(ConfigOptions, MpiConfig)

            # Compose a path to a log file, which will contain information
            # about this forecast cycle.
            # ConfigOptions.logFile = ConfigOptions.output_dir + "/LOG_" + \

            if ConfigOptions.ana_flag:
                log_time = ConfigOptions.e_date_proc
            else:
                log_time = ConfigOptions.current_fcst_cycle

            ConfigOptions.logFile = ConfigOptions.scratch_dir + "/LOG_" + ConfigOptions.nwmConfig + \
                                    ('_' if ConfigOptions.nwmConfig != "long_range" else "_mem" + str(ConfigOptions.cfsv2EnsMember)+ "_") + \
                                    ConfigOptions.d_program_init.strftime('%Y%m%d%H%M') + \
                                    "_" + log_time.strftime('%Y%m%d%H%M')

            # Initialize the log file.
            try:
                err_handler.init_log(ConfigOptions, MpiConfig)
            except:
                err_handler.err_out_screen_para(ConfigOptions.errMsg, MpiConfig)
            err_handler.check_program_status(ConfigOptions, MpiConfig)

        # Log information about this forecast cycle
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
            err_handler.log_msg(ConfigOptions, MpiConfig)
            ConfigOptions.statusMsg = 'Processing Forecast Cycle: ' + \
                                      ConfigOptions.current_fcst_cycle.strftime('%Y-%m-%d %H:%M')
            err_handler.log_msg(ConfigOptions, MpiConfig)
            ConfigOptions.statusMsg = 'Forecast Cycle Length is: ' + \
                                      str(ConfigOptions.cycle_length_minutes) + " minutes"
            err_handler.log_msg(ConfigOptions, MpiConfig)
        # MpiConfig.comm.barrier()

        # Loop through each output timestep. Perform the following functions:
        # 1.) Calculate all necessary input files per user options.
        # 2.) Read in input forcings from GRIB/NetCDF files.
        # 3.) Regrid the forcings, and temporally interpolate.
        # 4.) Downscale.
        # 5.) Layer, and output as necessary.
        ana_factor = 1 if ConfigOptions.ana_flag is False else 0
        show_message = True
        for outStep in range(1, ConfigOptions.num_output_steps + 1):
            # Reset out final grids to missing values.
            OutputObj.output_local[:, :, :] = -9999.0

            ConfigOptions.current_output_step = outStep
            OutputObj.outDate = ConfigOptions.current_fcst_cycle + datetime.timedelta(
                    seconds=ConfigOptions.output_freq * 60 * outStep
            )
            ConfigOptions.current_output_date = OutputObj.outDate

            # if AnA, adjust file date for analysis vs forecast
            if ConfigOptions.ana_flag:
                file_date = OutputObj.outDate - datetime.timedelta(seconds=ConfigOptions.output_freq * 60)
            else:
                file_date = OutputObj.outDate

            # Calculate the previous output timestep. This is used in potential downscaling routines.
            if outStep == ana_factor:
                ConfigOptions.prev_output_date = ConfigOptions.current_output_date
            else:
                ConfigOptions.prev_output_date = ConfigOptions.current_output_date - datetime.timedelta(
                        seconds=ConfigOptions.output_freq * 60
                )
            if MpiConfig.rank == 0 and show_message:
                ConfigOptions.statusMsg = '========================================='
                err_handler.log_msg(ConfigOptions, MpiConfig)
                ConfigOptions.statusMsg = "Processing for output timestep: " + \
                                          file_date.strftime('%Y-%m-%d %H:%M')
                err_handler.log_msg(ConfigOptions, MpiConfig)
            # MpiConfig.comm.barrier()

            # Compose the expected path to the output file. Check to see if the file exists,
            # if so, continue to the next time step. Also initialize our output arrays if necessary.
            OutputObj.outPath = fcstCycleOutDir + "/" + file_date.strftime('%Y%m%d%H%M') + \
                                ".LDASIN_DOMAIN1"
            # MpiConfig.comm.barrier()

            if os.path.isfile(OutputObj.outPath):
                if MpiConfig.rank == 0:
                    ConfigOptions.statusMsg = "Output file: " + OutputObj.outPath + " exists. Moving " + \
                                              " to the next output timestep."
                    err_handler.log_msg(ConfigOptions, MpiConfig)
                err_handler.check_program_status(ConfigOptions, MpiConfig)
                continue
            else:
                ConfigOptions.currentForceNum = 0
                ConfigOptions.currentCustomForceNum = 0
                # Loop over each of the input forcings specifed.
                for forceKey in ConfigOptions.input_forcings:
                    input_forcings = inputForcingMod[forceKey]
                    # Calculate the previous and next input cycle files from the inputs.
                    input_forcings.calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    # break loop if done early
                    if input_forcings.skip is True:
                        show_message = False            # just to avoid confusion
                        break

                    # Regrid forcings.
                    input_forcings.regrid_inputs(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    # Run check on regridded fields for reasonable values that are not missing values.
                    err_handler.check_forcing_bounds(ConfigOptions, input_forcings, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    # If we are restarting a forecast cycle, re-calculate the neighboring files, and regrid the
                    # next set of forcings as the previous step just regridded the previous forcing.
                    if input_forcings.rstFlag == 1:
                        if input_forcings.regridded_forcings1 is not None and \
                                input_forcings.regridded_forcings2 is not None:
                            # Set the forcings back to reflect we just regridded the previous set of inputs, not the next.
                            input_forcings.regridded_forcings1[:, :, :] = \
                                input_forcings.regridded_forcings2[:, :, :]

                        # Re-calculate the neighbor files.
                        input_forcings.calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                        err_handler.check_program_status(ConfigOptions, MpiConfig)

                        # Regrid the forcings for the end of the window.
                        input_forcings.regrid_inputs(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                        err_handler.check_program_status(ConfigOptions, MpiConfig)

                        input_forcings.rstFlag = 0

                    # Run temporal interpolation on the grids.
                    input_forcings.temporal_interpolate_inputs(ConfigOptions, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    # Run bias correction.
                    bias_correction.run_bias_correction(input_forcings, ConfigOptions,
                                                        wrfHydroGeoMeta, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    # Run downscaling on grids for this output timestep.
                    downscale.run_downscaling(input_forcings, ConfigOptions,
                                              wrfHydroGeoMeta, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    # Layer in forcings from this product.
                    layeringMod.layer_final_forcings(OutputObj, input_forcings, ConfigOptions, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    ConfigOptions.currentForceNum = ConfigOptions.currentForceNum + 1

                    if forceKey == 10:
                        ConfigOptions.currentCustomForceNum = ConfigOptions.currentCustomForceNum + 1
                
                else:
                    # Process supplemental precipitation if we specified in the configuration file.
                    if ConfigOptions.number_supp_pcp > 0:
                        for suppPcpKey in ConfigOptions.supp_precip_forcings:
                            # Like with input forcings, calculate the neighboring files to use.
                            suppPcpMod[suppPcpKey].calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

                            # Regrid the supplemental precipitation.
                            suppPcpMod[suppPcpKey].regrid_inputs(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

                            if suppPcpMod[suppPcpKey].regridded_precip1 is not None \
                                    and suppPcpMod[suppPcpKey].regridded_precip2 is not None:
                                # if np.any(suppPcpMod[suppPcpKey].regridded_precip1) and \
                                #        np.any(suppPcpMod[suppPcpKey].regridded_precip2):
                                # Run check on regridded fields for reasonable values that are not missing values.
                                err_handler.check_supp_pcp_bounds(ConfigOptions, suppPcpMod[suppPcpKey], MpiConfig)
                                err_handler.check_program_status(ConfigOptions, MpiConfig)

                                disaggregate_fun(input_forcings, suppPcpMod[suppPcpKey], ConfigOptions, MpiConfig)
                                err_handler.check_program_status(ConfigOptions, MpiConfig)

                                # Run temporal interpolation on the grids.
                                suppPcpMod[suppPcpKey].temporal_interpolate_inputs(ConfigOptions, MpiConfig)
                                err_handler.check_program_status(ConfigOptions, MpiConfig)

                                # Layer in the supplemental precipitation into the current output object.
                                layeringMod.layer_supplemental_forcing(OutputObj, suppPcpMod[suppPcpKey],
                                                                    ConfigOptions, MpiConfig)
                                err_handler.check_program_status(ConfigOptions, MpiConfig)

                    # Call the output routines
                    #   adjust date for AnA if necessary
                    if ConfigOptions.ana_flag:
                        OutputObj.outDate = file_date

                    OutputObj.output_final_ldasin(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

        if (not ConfigOptions.ana_flag) or (fcstCycleNum == (ConfigOptions.nFcsts - 1)):
            if MpiConfig.rank == 0:
                ConfigOptions.statusMsg = "Forcings complete for forecast cycle: " + \
                                          ConfigOptions.current_fcst_cycle.strftime('%Y-%m-%d %H:%M')
                err_handler.log_msg(ConfigOptions, MpiConfig)
            err_handler.check_program_status(ConfigOptions, MpiConfig)

            if MpiConfig.rank == 0:
                # Close the log file.
                try:
                    err_handler.close_log(ConfigOptions, MpiConfig)
                except:
                    err_handler.err_out_screen_para(ConfigOptions.errMsg, MpiConfig)

            # Success.... Now touch an empty complete file for this forecast cycle to indicate
            # completion in case the code is re-ran.
            try:
                open(completeFlag, 'a').close()
            except:
                ConfigOptions.errMsg = "Unable to create completion file: " + completeFlag
                err_handler.log_critical(ConfigOptions, MpiConfig)
            err_handler.check_program_status(ConfigOptions, MpiConfig)
