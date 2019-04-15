from core import errMod
import datetime
import os
import sys
from core import downscaleMod
from core import biasCorrectMod
from core import layeringMod

def process_forecasts(ConfigOptions,wrfHydroGeoMeta,inputForcingMod,suppPcpMod,MpiConfig,OutputObj):
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

    if MpiConfig.rank == 0:
        print('Forecast Cycle Length is: ' + str(ConfigOptions.cycle_length_minutes))
    for fcstCycleNum in range(ConfigOptions.nFcsts):
        ConfigOptions.current_fcst_cycle = ConfigOptions.b_date_proc + \
                                           datetime.timedelta(
            seconds=ConfigOptions.fcst_freq*60*fcstCycleNum
        )
        if MpiConfig.rank == 0:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('Processing Forecast Cycle: ' + ConfigOptions.current_fcst_cycle.strftime('%Y-%m-%d %H:%M'))
        fcstCycleOutDir = ConfigOptions.output_dir + "/" + \
            ConfigOptions.current_fcst_cycle.strftime('%Y%m%d%H%M')
        completeFlag = fcstCycleOutDir + "/WrfHydroForcing.COMPLETE"
        if os.path.isfile(completeFlag):
            # We have already completed processing this cycle,
            # move on.
            continue

        if MpiConfig.rank == 0:
            # If the cycle directory doesn't exist, create it.
            if not os.path.isdir(fcstCycleOutDir):
                try:
                    os.mkdir(fcstCycleOutDir)
                except:
                    ConfigOptions.errMsg = "Unable to create output " \
                                           "directory: " + fcstCycleOutDir
                    errMod.err_out_screen(ConfigOptions.errMsg)

        # Compose a path to a log file, which will contain information
        # about this forecast cycle.
        ConfigOptions.logFile = fcstCycleOutDir + "/LOG_" + \
            ConfigOptions.d_program_init.strftime('%Y%m%d%H%M') + \
            "_" + ConfigOptions.current_fcst_cycle.strftime('%Y%m%d%H%M')

        if MpiConfig.rank == 0:
            # Initialize the log file.
            try:
                errMod.init_log(ConfigOptions)
            except:
                errMod.err_out_screen(ConfigOptions.errMsg)


        # Loop through each output timestep. Perform the following functions:
        # 1.) Calculate all necessary input files per user options.
        # 2.) Read in input forcings from GRIB/NetCDF files.
        # 3.) Regrid the forcings, and temporally interpolate.
        # 4.) Downscale.
        # 5.) Layer, and output as necessary.
        for outStep in range(1,ConfigOptions.num_output_steps+1):
            # Reset out final grids to missing values.
            OutputObj.output_local[:,:,:] = -9999.0

            ConfigOptions.current_output_step = outStep
            OutputObj.outDate = ConfigOptions.current_fcst_cycle + datetime.timedelta(
                seconds=ConfigOptions.output_freq*60*outStep
            )
            ConfigOptions.current_output_date = OutputObj.outDate
            if MpiConfig.rank == 0:
                print('=========================================')
                print("Processing for output timestep: " + OutputObj.outDate.strftime('%Y-%m-%d %H:%M'))

            # Compose the expected path to the output file. Check to see if the file exists,
            # if so, continue to the next time step. Also initialize our output arrays if necessary.
            OutputObj.outPath = fcstCycleOutDir + "/" + OutputObj.outDate.strftime('%Y%m%d%H%M') + \
                ".LDASIN_DOMAIN1"

            MpiConfig.comm.barrier()

            if os.path.isfile(OutputObj.outPath):
                ConfigOptions.statusMsg = "Output file: " + OutputObj.outPath + " exists. Moving " + \
                    " to the next output timestep."
                continue
            else:
                ConfigOptions.currentForceNum = 0
                ConfigOptions.currentCustomForceNum = 0
                # Loop over each of the input forcings specifed.
                for forceKey in ConfigOptions.input_forcings:
                    # Calculate the previous and next input cycle files from the inputs.
                    inputForcingMod[forceKey].calc_neighbor_files(ConfigOptions, OutputObj.outDate,MpiConfig)
                    if MpiConfig.rank == 0:
                        print('Previous INPUT File = ' + inputForcingMod[forceKey].file_in1)
                        print('Next INPUT File = ' + inputForcingMod[forceKey].file_in2)
                    #try:
                    #    inputForcingMod[forceKey].calc_neighbor_files(ConfigOptions,outputObj.outDate,MpiConfig)
                    #except:
                    #    errMod.err_out(ConfigOptions)
                    MpiConfig.comm.barrier()

                    # Regrid forcings.
                    inputForcingMod[forceKey].regrid_inputs(ConfigOptions,wrfHydroGeoMeta,MpiConfig)
                    #try:
                    #    inputForcingMod[forceKey].regrid_inputs(ConfigOptions,wrfHydroGeoMeta,MpiConfig)
                    #except:
                    #    errMod.err_out(ConfigOptions)
                    MpiConfig.comm.barrier()

                    # Run temporal interpolation on the grids.
                    inputForcingMod[forceKey].temporal_interpolate_inputs(ConfigOptions,MpiConfig)
                    #try:
                    #    inputForcingMod[forceKey].temporal_interpolate_inputs(ConfigOptions, MpiConfig)
                    #except:
                    #    errMod.err_out(ConfigOptions)
                    MpiConfig.comm.barrier()

                    # Run bias correction.
                    biasCorrectMod.run_bias_correction(inputForcingMod[forceKey],ConfigOptions)
                    #try:
                    #    biasCorrectMod.run_bias_correction(inputForcingMod[forceKey],ConfigOptions)
                    #except:
                    #    errMod.err_out(ConfigOptions)

                    # Run downscaling on grids for this output timestep.
                    downscaleMod.run_downscaling(inputForcingMod[forceKey],ConfigOptions,wrfHydroGeoMeta)
                    #try:
                    #    downscaleMod.run_downscaling(inputForcingMod[forceKey],ConfigOptions,wrfHydroGeoMeta)
                    #except:
                    #    errMod.err_out(ConfigOptions)
                    MpiConfig.comm.barrier()

                    # Layer in forcings from this product.
                    layeringMod.layer_final_forcings(OutputObj,inputForcingMod[forceKey],ConfigOptions,MpiConfig)
                    try:
                        layeringMod.layer_final_forcings(OutputObj, inputForcingMod[forceKey], ConfigOptions, MpiConfig)
                    except:
                        errMod.err_out(ConfigOptions)
                    MpiConfig.comm.barrier()

                    ConfigOptions.currentForceNum = ConfigOptions.currentForceNum + 1

                    if forceKey == 10:
                        ConfigOptions.currentCustomForceNum = ConfigOptions.currentCustomForceNum + 1

                # Process supplemental precipitation if we specified in the configuration file.
                if ConfigOptions.number_supp_pcp > 0:
                    for suppPcpKey in ConfigOptions.supp_precip_forcings:
                        # Like with input forcings, calculate the neighboring files to use.
                        suppPcpMod[suppPcpKey].calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                        if MpiConfig.rank == 0:
                            print('Previous SUPP PCP File = ' + suppPcpMod[suppPcpKey].file_in1)
                            print('Next SUPP PCP File = ' + suppPcpMod[suppPcpKey].file_in2)
                        # try:
                        #    suppPcpMod[suppPcpKey].calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                        # except:
                        #    errMod.err_out(ConfigOptions)
                        MpiConfig.comm.barrier()

                        # Regrid the supplemental precipitation.
                        suppPcpMod[suppPcpKey].regrid_inputs(ConfigOptions,wrfHydroGeoMeta,MpiConfig)
                        #try:
                        #except:
                        #    errMod.err_out(ConfigOptions)
                        MpiConfig.comm.barrier()

                # Call the output routines
                OutputObj.output_final_ldasin(ConfigOptions,wrfHydroGeoMeta,MpiConfig)
                MpiConfig.comm.barrier()


            #sys.exit(1)

        #sys.exit(1)

        if MpiConfig.rank == 0:
            # Close the log file.
            try:
                errMod.close_log(ConfigOptions)
            except:
                errMod.err_out_screen(ConfigOptions.errMsg)

