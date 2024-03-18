"""
Module file to handle both reading in input files:
1.) GRIB2 format
2.) NetCDF format
Also, creating output files.
"""
import datetime
import gzip
import math
import os
import shutil
import subprocess

import numpy as np
from netCDF4 import Dataset

from core import err_handler


class OutputObj:
    """
    Abstract class to hold local "slabs" of final output
    grids.
    """
    def __init__(self,ConfigOptions,GeoMetaWrfHydro):
        self.output_local = None
        self.outPath = None
        self.outDate = None
        self.out_ndv = -999999

        # Create local "slabs" to hold final output grids. These
        # will be collected during the output routine below.
        force_count = 9 if ConfigOptions.include_lqfrac else 8
        self.output_local = np.empty([force_count, GeoMetaWrfHydro.ny_local, GeoMetaWrfHydro.nx_local])
        #self.output_local[:,:,:] = self.out_ndv

    def output_final_ldasin(self, ConfigOptions, geoMetaWrfHydro, MpiConfig):
        """
        Output routine to produce final LDASIN files for the WRF-Hydro
        modeling system. This function is assuming all regridding,
        interpolation, downscaling, and bias correction has occurred
        on the necessary input forcings to generate a final set of
        outputs on the output grid. Since this program is ran in parallel,
        all work is done on local "slabs" of data for each processor to
        make the code more efficient. On this end, this function will
        collect the "slabs" into final output grids that go into the
        output files. In addition, detailed geospatial metadata is translated
        from the input geogrid file, to the final output files.
        :param ConfiguOptions:
        :param geoMetaWrfHydro:
        :param MpiConfig:
        :return:
        """
        output_variable_attribute_dict = {
            'U2D': [0,'m s-1','x_wind','10-m U-component of wind','time: point',0.001,0.0,3],
            'V2D': [1,'m s-1','y_wind','10-m V-component of wind','time: point',0.001,0.0,3],
            'LWDOWN': [2,'W m-2','surface_downward_longwave_flux',
                       'Surface downward long-wave radiation flux','time: point',0.001,0.0,3],
            'RAINRATE': [3,'mm s^-1','precipitation_flux','Surface Precipitation Rate','time: mean',1.0,0.0,0],
            'T2D': [4,'K','air_temperature','2-m Air Temperature','time: point',0.01,100.0,2],
            'Q2D': [5,'kg kg-1','surface_specific_humidity','2-m Specific Humidity','time: point',0.000001,0.0,6],
            'PSFC': [6,'Pa','air_pressure','Surface Pressure','time: point',0.1,0.0,1],
            'SWDOWN': [7,'W m-2','surface_downward_shortwave_flux',
                       'Surface downward short-wave radiation flux','time: point',0.001,0.0,3]
        }

        if ConfigOptions.include_lqfrac:
            output_variable_attribute_dict['LQFRAC'] = [8, '%', 'liquid_water_fraction',
                                                        'Fraction of precipitation that is liquid vs. frozen',
                                                        'time: point', 0.01, 0.0, 3]

        # Compose the ESMF remapped string attribute based on the regridding option chosen by the user.
        # We will default to the regridding method chosen for the first input forcing selected.
        if ConfigOptions.regrid_opt[0] == 1:
            regrid_att = "remapped via ESMF regrid_with_weights: Bilinear"
        elif ConfigOptions.regrid_opt[0] == 2:
            regrid_att = "remapped via ESMF regrid_with_weights: Nearest Neighbor"
        elif ConfigOptions.regrid_opt[0] == 3:
            regrid_att = "remapped via ESMF regrid_with_weights: Conservative Bilinear"

        # Ensure all processors are synced up before outputting.
        #MpiConfig.comm.barrier()

        idOut = None
        if MpiConfig.rank == 0:
            while (True):
                # Only output on the master processor.
                try:
                    idOut = Dataset(self.outPath,'w')
                except Exception as e:
                    ConfigOptions.errMsg = "Unable to create output file: " + self.outPath + "\n" + str(e)
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Create dimensions.
                try:
                    idOut.createDimension("time", None)
                except:
                    ConfigOptions.errMsg = "Unable to create time dimension in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.createDimension("y",geoMetaWrfHydro.ny_global)
                except:
                    ConfigOptions.errMsg = "Unable to create y dimension in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.createDimension("x",geoMetaWrfHydro.nx_global)
                except:
                    ConfigOptions.errMsg = "Unable to create x dimension in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.createDimension("reference_time", 1)
                except:
                    ConfigOptions.errMsg = "Unable to create reference_time dimension in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Set global attributes
                try:
                    idOut.model_output_valid_time = self.outDate.strftime("%Y-%m-%d_%H:%M:00")
                except:
                    ConfigOptions.errMsg = "Unable to set the model_output_valid_time attribute in :" + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    if ConfigOptions.ana_flag:
                        model_init = ConfigOptions.b_date_proc - datetime.timedelta(minutes=ConfigOptions.output_freq)
                    else:
                        model_init = ConfigOptions.current_fcst_cycle
                    idOut.model_initialization_time = model_init.strftime("%Y-%m-%d_%H:%M:00")
                except:
                    ConfigOptions.errMsg = "Unable to set the model_initialization_time global " \
                                           "attribute in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                if ConfigOptions.nwmVersion is not None:
                    try:
                        idOut.NWM_version_number = "v" + str(ConfigOptions.nwmVersion)
                    except:
                        ConfigOptions.errMsg = "Unable to set the NWM_version_number global attribute in: " \
                                               + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                if ConfigOptions.nwmConfig is not None:
                    try:
                        idOut.model_configuration = ConfigOptions.nwmConfig
                    except:
                        ConfigOptions.errMsg = "Unable to set the model_configuration global attribute in: " + \
                                               self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                try:
                    idOut.model_output_type = "forcing"
                except:
                    ConfigOptions.errMsg = "Unable to put model_output_type global attribute in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                try:
                    idOut.model_total_valid_times = np.int32(ConfigOptions.actual_output_steps)
                except:
                    ConfigOptions.errMsg = "Unable to create total_valid_times global attribute in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                if ConfigOptions.spatial_meta is not None:
                    # apply spatial_global_atts to output globals attrs
                    for k, v in geoMetaWrfHydro.spatial_global_atts.items():
                        if k not in ("Source_Software", "history", "processing_notes", "version"):     # don't add these
                            idOut.setncattr(k, v)

                # Create variables.
                try:
                    idOut.createVariable('time','i4',('time'))
                except:
                    ConfigOptions.errMsg = "Unable to create time variable in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.createVariable('reference_time','i4',('reference_time'))
                except:
                    ConfigOptions.errMsg = "Unable to create reference_time variable in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Populate time and reference time variables with appropriate attributes and time values.
                try:
                    idOut.variables['time'].units = "minutes since 1970-01-01 00:00:00 UTC"
                except:
                    ConfigOptions.errMsg = "Unable to create time units attribute in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.variables['time'].standard_name = "time"
                except:
                    ConfigOptions.errMsg = "Unable to create time standard_name attribute in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.variables['time'].long_name = "valid output time"
                except:
                    ConfigOptions.errMsg = "Unable to create time long_name attribute in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                try:
                    idOut.variables['reference_time'].units = "minutes since 1970-01-01 00:00:00 UTC"
                except:
                    ConfigOptions.errMsg = "Unable to create reference_time units attribute in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.variables['reference_time'].standard_name = "forecast_reference_time"
                except:
                    ConfigOptions.errMsg = "Unable to create reference_time standard_name attribute in: " + \
                                           self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.variables['reference_time'].long_name = "model initialization time"
                except:
                    ConfigOptions.errMsg = "Unable to create reference_time long_name attribute in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Populate time variables
                dEpoch = datetime.datetime(1970, 1, 1)
                dtValid = self.outDate - dEpoch
                dtRef = ConfigOptions.current_fcst_cycle - dEpoch

                try:
                    idOut.variables['time'][0] = int(dtValid.days * 24.0 * 60.0) + \
                                                 int(math.floor(dtValid.seconds / 60.0))
                except:
                    ConfigOptions.errMsg = "Unable to populate the time variable in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                try:
                    idOut.variables['reference_time'][0] = int(dtRef.days * 24.0 * 60.0) + \
                                                           int(math.floor(dtRef.seconds / 60.0))
                except:
                    ConfigOptions.errMsg = "Unable to populate the time variable in: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Create geospatial metadata coordinate variables if data was read in from an optional
                # spatial metadata file.
                if ConfigOptions.spatial_meta is not None:
                    # Create coordinate variables and populate with attributes read in.
                    try:
                        if ConfigOptions.useCompression == 1:
                            idOut.createVariable('x', 'f8', ('x'), zlib=True, complevel=2)
                        else:
                            idOut.createVariable('x','f8', ('x'))
                    except:
                        ConfigOptions.errMsg = "Unable to create x variable in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['x'].setncatts(geoMetaWrfHydro.x_coord_atts)
                    except:
                        ConfigOptions.errMsg = "Unable to establish x coordinate attributes in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['x'][:] = geoMetaWrfHydro.x_coords
                    except:
                        ConfigOptions.errMsg = "Unable to place x coordinate values into output variable " \
                                               "for output file: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        if ConfigOptions.useCompression == 1:
                            idOut.createVariable('y','f8',('y'), zlib=True, complevel=2)
                        else:
                            idOut.createVariable('y', 'f8', ('y'))
                    except:
                        ConfigOptions.errMsg = "Unable to create y variable in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['y'].setncatts(geoMetaWrfHydro.y_coord_atts)
                    except:
                        ConfigOptions.errMsg = "Unable to establish y coordinate attributes in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['y'][:] = geoMetaWrfHydro.y_coords
                    except:
                        ConfigOptions.errMsg = "Unable to place y coordinate values into output variable " \
                                               "for output file: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        idOut.createVariable('crs','S1')
                    except:
                        ConfigOptions.errMsg = "Unable to create crs in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['crs'].setncatts(geoMetaWrfHydro.crs_atts)
                    except:
                        ConfigOptions.errMsg = "Unable to establish crs attributes in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                # Loop through and create each variable, along with expected attributes.
                for varTmp in output_variable_attribute_dict:
                    try:
                        if ConfigOptions.useCompression:
                            zlib = True
                            complevel = 2
                            least_significant_digit = None if varTmp == 'RAINRATE' else \
                                output_variable_attribute_dict[varTmp][7]       # use all digits in RAINRATE
                        else:
                            zlib = False
                            complevel = 0
                            least_significant_digit = None

                        if ConfigOptions.useFloats or varTmp == 'RAINRATE':     # RAINRATE always a float
                            fill_value = ConfigOptions.globalNdv
                            dtype = 'f4'
                        else:
                            fill_value = int(ConfigOptions.globalNdv)
                            #fill_value = int((ConfigOptions.globalNdv - output_variable_attribute_dict[varTmp][6]) /
                            #                 output_variable_attribute_dict[varTmp][5])
                            dtype = 'i4'

                        idOut.createVariable(varTmp, dtype, ('time', 'y', 'x'),
                                             fill_value=fill_value,
                                             zlib=zlib,
                                             complevel=complevel,
                                             least_significant_digit=least_significant_digit)

                    except:
                        ConfigOptions.errMsg = "Unable to create " + varTmp + " variable in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp].cell_methods = output_variable_attribute_dict[varTmp][4]
                    except:
                        ConfigOptions.errMsg = "Unable to create cell_methods attribute for: " + varTmp + \
                            " in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp].remap = regrid_att
                    except:
                        ConfigOptions.errMsg = "Unable to create remap attribute for: " + varTmp + \
                            " in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    # Place geospatial metadata attributes in if we have them.
                    if ConfigOptions.spatial_meta is not None:
                        try:
                            idOut.variables[varTmp].grid_mapping = 'crs'
                        except:
                            ConfigOptions.errMsg = "Unable to create grid_mapping attribute for: " + \
                                                   varTmp + " in: " + self.outPath
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break
                        if 'esri_pe_string' in geoMetaWrfHydro.crs_atts.keys():
                            try:
                                idOut.variables[varTmp].esri_pe_string = geoMetaWrfHydro.crs_atts['esri_pe_string']
                            except:
                                ConfigOptions.errMsg = "Unable to create esri_pe_string attribute for: " + \
                                                       varTmp + " in: " + self.outPath
                                err_handler.log_critical(ConfigOptions, MpiConfig)
                                break
                        if 'proj4' in geoMetaWrfHydro.spatial_global_atts.keys():
                            try:
                                idOut.variables[varTmp].proj4 = geoMetaWrfHydro.spatial_global_atts['proj4']
                            except:
                                ConfigOptions.errMsg = "Unable to create proj4 attribute for: " + varTmp + \
                                    " in: " + self.outPath
                                err_handler.log_critical(ConfigOptions, MpiConfig)
                                break

                    try:
                        idOut.variables[varTmp].units = output_variable_attribute_dict[varTmp][1]
                    except:
                        ConfigOptions.errMsg = "Unable to create units attribute for: " + varTmp + " in: " + \
                            self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp].standard_name = output_variable_attribute_dict[varTmp][2]
                    except:
                        ConfigOptions.errMsg = "Unable to create standard_name attribute for: " + varTmp + \
                            " in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp].long_name = output_variable_attribute_dict[varTmp][3]
                    except:
                        ConfigOptions.errMsg = "Unable to create long_name attribute for: " + varTmp + \
                            " in: " + self.outPath
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    # If we are using scale_factor / add_offset, create here.
                    if not ConfigOptions.useFloats:
                        if varTmp != 'RAINRATE':
                            try:
                                idOut.variables[varTmp].scale_factor = output_variable_attribute_dict[varTmp][5]
                            except (ValueError, IOError):
                                ConfigOptions.errMsg = "Unable to create scale_factor attribute for: " + varTmp + \
                                                       " in: " + self.outPath
                                err_handler.log_critical(ConfigOptions, MpiConfig)
                                break
                            try:
                                idOut.variables[varTmp].add_offset = output_variable_attribute_dict[varTmp][6]
                            except (ValueError, IOError):
                                ConfigOptions.errMsg = "Unable to create add_offset attribute for: " + varTmp + \
                                                       " in: " + self.outPath
                                err_handler.log_critical(ConfigOptions, MpiConfig)
                                break
                break

        err_handler.check_program_status(ConfigOptions, MpiConfig)

        # Now loop through each variable, collect the data (call on each processor), assemble into the final
        # output grid, and place into the output file (if on processor 0).
        for varTmp in output_variable_attribute_dict:
            # First run a check for missing values. There should be none at this point.
            # err_handler.check_missing_final(self.outPath, ConfigOptions, self.output_local[output_variable_attribute_dict[varTmp][0], :, :],
            #                                 varTmp, MpiConfig)
            # if ConfigOptions.errFlag == 1:
            #     continue

            # Collect data from the various processors, and place into the output file.
            try:
                # TODO change communication call from comm.gather() to comm.Gather for efficency
                # final = MpiConfig.comm.gather(self.output_local[output_variable_attribute_dict[varTmp][0],:,:],root=0)

                # Use gatherv to merge the data slabs
                dataOutTmp = MpiConfig.merge_slabs_gatherv(self.output_local[output_variable_attribute_dict[varTmp][0],:,:], ConfigOptions)
            except Exception as e:
                print(e)
                ConfigOptions.errMsg = "Unable to gather final grids for: " + varTmp
                err_handler.log_critical(ConfigOptions, MpiConfig)
                continue

            if MpiConfig.rank == 0:
                try:
                    idOut.variables[varTmp][0, :, :] = dataOutTmp
                except (ValueError, IOError):
                    ConfigOptions.errMsg = "Unable to place final output grid for: " + varTmp
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                # Reset temporary data objects to keep memory usage down.
                del dataOutTmp

            # Reset temporary data objects to keep memory usage down.
            final = None

            err_handler.check_program_status(ConfigOptions, MpiConfig)

        if MpiConfig.rank == 0:
            while (True):
                # Close the NetCDF file
                try:
                    idOut.close()
                except (ValueError, IOError):
                    ConfigOptions.errMsg = "Unable to close output file: " + self.outPath
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                break

        err_handler.check_program_status(ConfigOptions, MpiConfig)


def open_grib2(GribFileIn,NetCdfFileOut,Wgrib2Cmd,ConfigOptions,MpiConfig,
               inputVar):
    """
    Generic function to convert a GRIB2 file into a NetCDF file. Function
    will also open the NetCDF file, and ensure all necessary inputs are
    in file.
    :param GribFileIn:
    :param NetCdfFileOut:
    :param ConfigOptions:
    :return:
    """
    # Ensure all processors are synced up before outputting.
    # MpiConfig.comm.barrier()

    # Run wgrib2 command to convert GRIB2 file to NetCDF.
    if MpiConfig.rank == 0:
        # Check to see if output file already exists. If so, delete it and
        # override.
        ConfigOptions.statusMsg = "Reading in GRIB2 file: " + GribFileIn
        err_handler.log_msg(ConfigOptions, MpiConfig)
        if os.path.isfile(NetCdfFileOut):
            ConfigOptions.statusMsg = "Overwriting temporary NetCDF file: " + NetCdfFileOut
            err_handler.log_msg(ConfigOptions, MpiConfig)
        try:
            # WCOSS fix for WGRIB2 crashing when called on the same file twice in python
            if not os.environ.get('MFE_SILENT'):
                print("command: " + Wgrib2Cmd)

            # set up GRIB2TABLE if needed:
            if not os.environ.get('GRIB2TABLE'):
                g2path = os.path.join(ConfigOptions.scratch_dir, "grib2.tbl")
                with open(g2path, 'wt') as g2t:
                    g2t.write(
                            "209:1:0:0:161:1:6:30:MultiSensorQPE01H:"
                            "Multi-sensor estimated precipitation accumulation 1-hour:mm\n"
                            "209:1:0:0:161:1:6:37:MultiSensorQPE01H:"
                            "Multi-sensor estimated precipitation accumulation 1-hour:mm\n"
                    )
                os.environ['GRIB2TABLE'] = g2path

            exitcode = subprocess.call(Wgrib2Cmd, shell=True)

            #print("exitcode: " + str(exitcode))
            # Call WGRIB2 with subprocess.Popen
            #cmdOutput = subprocess.Popen([Wgrib2Cmd], stdout=subprocess.PIPE,
            #                             stderr=subprocess.PIPE, shell=True)
            #out, err = cmdOutput.communicate()
            #exitcode = cmdOutput.returncode
        except:
            ConfigOptions.errMsg = "Unable to convert: " + GribFileIn + " to " + \
                                   NetCdfFileOut
            err_handler.log_critical(ConfigOptions, MpiConfig)
            idTmp = None
            pass

        # Reset temporary subprocess variables.
        out = None
        err = None
        exitcode = None

        # Ensure file exists.
        if not os.path.isfile(NetCdfFileOut):
            ConfigOptions.errMsg = "Expected NetCDF file: " + NetCdfFileOut + \
                                   " not found. It's possible the GRIB2 variable was not found."
            err_handler.log_critical(ConfigOptions, MpiConfig)
            idTmp = None
            pass

        # Open the NetCDF file.
        try:
            idTmp = Dataset(NetCdfFileOut,'r')
        except:
            ConfigOptions.errMsg = "Unable to open input NetCDF file: " + \
                                   NetCdfFileOut
            err_handler.log_critical(ConfigOptions, MpiConfig)
            idTmp = None
            pass

        if idTmp is not None:
            # Check for expected lat/lon variables.
            if 'latitude' not in idTmp.variables.keys():
                ConfigOptions.statusMsg = "Unable to locate latitude from: " + \
                                       GribFileIn
                err_handler.log_warning(ConfigOptions, MpiConfig)
                # idTmp = None
                pass
        if idTmp is not None:
            if 'longitude' not in idTmp.variables.keys():
                ConfigOptions.statusMsg = "Unable to locate longitude from: " + \
                                       GribFileIn
                err_handler.log_warning(ConfigOptions, MpiConfig)
                # idTmp = None
                pass

        if idTmp is not None and inputVar is not None:
            # Loop through all the expected variables.
            if inputVar not in idTmp.variables.keys():
                ConfigOptions.errMsg = "Unable to locate expected variable: " + \
                                       inputVar + " in: " + NetCdfFileOut
                err_handler.log_critical(ConfigOptions, MpiConfig)
                idTmp = None
                pass
    else:
        idTmp = None

    # Ensure all processors are synced up before outputting.
    # MpiConfig.comm.barrier()  ## THIS HAPPENS IN check_program_status

    err_handler.check_program_status(ConfigOptions, MpiConfig)

    # Return the NetCDF file handle back to the user.
    return idTmp


def open_netcdf_forcing(NetCdfFileIn, ConfigOptions, MpiConfig, open_on_all_procs=False, lat_var="latitude", lon_var="longitude"):
    """
    Generic function to convert a NetCDF forcing file given a list of input forcing
    variables.
    :param GribFileIn:
    :param NetCdfFileOut:
    :param ConfigOptions:
    :return:
    """
    # Ensure all processors are synced up before outputting.
    #MpiConfig.comm.barrier()

    # Open the NetCDF file on the master processor and read in data.
    if MpiConfig.rank == 0 or open_on_all_procs:
        # Ensure file exists.
        if not os.path.isfile(NetCdfFileIn):
            ConfigOptions.errMsg = "Expected NetCDF file: " + NetCdfFileIn + \
                                    " not found."
            err_handler.log_critical(ConfigOptions, MpiConfig)
            idTmp = None

        # Open the NetCDF file.
        try:
            idTmp = Dataset(NetCdfFileIn, 'r')
        except:
            ConfigOptions.errMsg = "Unable to open input NetCDF file: " + \
                                    NetCdfFileIn
            err_handler.log_critical(ConfigOptions, MpiConfig)
            idTmp = None

        if idTmp is not None:
            # Check for expected lat/lon variables.
            if lat_var not in idTmp.variables.keys():
                ConfigOptions.errMsg = f"Unable to locate {lat_var} from: " + \
                                        NetCdfFileIn
                err_handler.log_critical(ConfigOptions, MpiConfig)
                idTmp = None
        if idTmp is not None:
            if lon_var not in idTmp.variables.keys():
                ConfigOptions.errMsg = f"Unable to locate {lon_var} from: " + \
                                        NetCdfFileIn
                err_handler.log_critical(ConfigOptions, MpiConfig)
                idTmp = None
    else:
        idTmp = None

    err_handler.check_program_status(ConfigOptions, MpiConfig)
    # Return the NetCDF file handle back to the user.
    return idTmp


def unzip_file(GzFileIn,FileOut,ConfigOptions,MpiConfig):
    """
    Generic I/O function to unzip a .gz file to a new location.
    :param GzFileIn:
    :param FileOut:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Ensure all processors are synced up before outputting.
    #MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        # Unzip the file in place.
        try:
            ConfigOptions.statusMsg = "Unzipping file: {}".format(GzFileIn)
            err_handler.log_msg(ConfigOptions, MpiConfig)
            with gzip.open(GzFileIn, 'rb') as fTmpGz:
                with open(FileOut, 'wb') as fTmp:
                    shutil.copyfileobj(fTmpGz, fTmp)
        except:
            ConfigOptions.errMsg = "Unable to unzip: " + GzFileIn
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return

        if not os.path.isfile(FileOut):
            ConfigOptions.errMsg = "Unable to locate expected unzipped file: " + \
                                   FileOut
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return
    else:
        return

def read_rqi_monthly_climo(ConfigOptions, MpiConfig, supplemental_precip, GeoMetaWrfHydro):
    """
    Function to read in monthly RQI grids on the NWM grid. This is an NWM ONLY
    option. Please do not activate if not executing on the NWM conus grid.
    :param ConfigOptions:
    :param MpiConfig:
    :param supplemental_precip:
    :return:
    """
    # Ensure all processors are synced up before proceeding.
    #MpiConfig.comm.barrier()

    # First check to see if the RQI grids have valid values in them. There should
    # be NO NDV values if the grids have properly been read in.
    indTmp = np.where(supplemental_precip.regridded_rqi2 != ConfigOptions.globalNdv)

    rqiPath = ConfigOptions.supp_precip_param_dir + "/MRMS_WGT_RQI0.9_m" + \
              supplemental_precip.pcp_date2.strftime('%m') + '_v1.1_geosmth.nc'

    if len(indTmp[0]) == 0:
        # We haven't initialized the RQI fields. We need to do this.....
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Reading in RQI Parameter File: " + rqiPath
            err_handler.log_msg(ConfigOptions, MpiConfig)
            # First make sure the RQI file exists.
            if not os.path.isfile(rqiPath):
                ConfigOptions.errMsg = "Expected RQI parameter file: " + rqiPath + " not found."
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass

            # Open the Parameter file.
            try:
                idTmp = Dataset(rqiPath, 'r')
            except:
                ConfigOptions.errMsg = "Unable to open parameter file: " + rqiPath
                pass

            # Extract out the RQI grid.
            try:
                varTmp = idTmp.variables['POP_0mabovemeansealevel'][0, :, :]
            except:
                ConfigOptions.errMsg = "Unable to extract POP_0mabovemeansealevel from parameter file: " + rqiPath
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass

            # Sanity checking on grid size.
            if varTmp.shape[0] != GeoMetaWrfHydro.ny_global or varTmp.shape[1] != GeoMetaWrfHydro.nx_global:
                ConfigOptions.errMsg = "Improper dimension sizes for POP_0mabovemeansealevel " \
                                       "in parameter file: " + rqiPath
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            idTmp = None
            varTmp = None
        err_handler.check_program_status(ConfigOptions, MpiConfig)

        # Scatter the array out to the local processors
        varSubTmp = MpiConfig.scatter_array(GeoMetaWrfHydro, varTmp, ConfigOptions)
        err_handler.check_program_status(ConfigOptions, MpiConfig)

        supplemental_precip.regridded_rqi2[:, :] = varSubTmp

        # Reset variables for memory purposes
        varSubTmp = None
        varTmp = None

        # Close the RQI NetCDF file
        if MpiConfig.rank == 0:
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close parameter file: " + rqiPath
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass
        err_handler.check_program_status(ConfigOptions, MpiConfig)


    # Also check to see if we have switched to a new month based on the previous
    # MRMS step and the current one.
    if supplemental_precip.pcp_date2.month != supplemental_precip.pcp_date1.month:
        # We need to read in a new RQI monthly grid.
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = "Reading in RQI Parameter File: " + rqiPath
            err_handler.log_msg(ConfigOptions, MpiConfig)
            # First make sure the RQI file exists.
            if not os.path.isfile(rqiPath):
                ConfigOptions.errMsg = "Expected RQI parameter file: " + rqiPath + " not found."
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass

            # Open the Parameter file.
            try:
                idTmp = Dataset(rqiPath, 'r')
            except:
                ConfigOptions.errMsg = "Unable to open parameter file: " + rqiPath
                pass

            # Extract out the RQI grid.
            try:
                varTmp = idTmp.variables['POP_0mabovemeansealevel'][0, :, :]
            except:
                ConfigOptions.errMsg = "Unable to extract POP_0mabovemeansealevel from parameter file: " + rqiPath
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass

            # Sanity checking on grid size.
            if varTmp.shape[0] != GeoMetaWrfHydro.ny_global or varTmp.shape[1] != GeoMetaWrfHydro.nx_global:
                ConfigOptions.errMsg = "Improper dimension sizes for POP_0mabovemeansealevel " \
                                        "in parameter file: " + rqiPath
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            idTmp = None
            varTmp = None
        err_handler.check_program_status(ConfigOptions, MpiConfig)

        # Scatter the array out to the local processors
        varSubTmp = MpiConfig.scatter_array(GeoMetaWrfHydro, varTmp, ConfigOptions)
        err_handler.check_program_status(ConfigOptions, MpiConfig)

        supplemental_precip.regridded_rqi2[:, :] = varSubTmp

        # Reset variables for memory purposes
        varSubTmp = None
        varTmp = None

        # Close the RQI NetCDF file
        if MpiConfig.rank == 0:
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close parameter file: " + rqiPath
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass
        err_handler.check_program_status(ConfigOptions, MpiConfig)

