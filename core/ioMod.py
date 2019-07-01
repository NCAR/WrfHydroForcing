"""
Module file to handle both reading in input files:
1.) GRIB2 format
2.) NetCDF format
Also, creating output files.
"""
from netCDF4 import Dataset
from core import errMod
import numpy as np
import os
import subprocess
import gzip
import shutil

class OutputObj:
    """
    Abstract class to hold local "slabs" of final output
    grids.
    """
    def __init__(self,GeoMetaWrfHydro):
        self.output_local = None
        self.outPath = None
        self.outDate = None
        self.out_ndv = -9999

        # Create local "slabs" to hold final output grids. These
        # will be collected during the output routine below.
        self.output_local = np.empty([8,GeoMetaWrfHydro.ny_local,GeoMetaWrfHydro.nx_local])
        #self.output_local[:,:,:] = self.out_ndv

    def output_final_ldasin(self,ConfigOptions,geoMetaWrfHydro,MpiConfig):
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
            'U2D': [0,'m s-1','x_wind','10-m U-component of wind','time: point'],
            'V2D': [1,'m s-1','y_wind','10-m V-component of wind','time: point'],
            'LWDOWN': [2,'W m-2','surface downward longwave_flux','Surface downward long-wave radiation flux','time: point'],
            'RAINRATE': [3,'mm s^-1','precipitation_flux','Surface Precipitation Rate','time: mean'],
            'T2D': [4,'K','air temperature','2-m Air Temperature','time: point'],
            'Q2D': [5,'kg kg-1','surface_specific_humidity','2-m Specific Humidity','time: point'],
            'PSFC': [6,'Pa','air_pressure','Surface Pressure','time: point'],
            'SWDOWN': [7,'W m-2','surface_downward_shortwave_flux','Surface downward short-wave radiation flux','time point']
        }

        # Compose the ESMF remapped string attribute based on the regridding option chosen by the user.
        # We will default to the regridding method chosen for the first input forcing selected.
        if ConfigOptions.regrid_opt[0] == 1:
            regrid_att = "remapped via ESMF regrid_with_weights: Bilinear"
        elif ConfigOptions.regrid_opt[0] == 2:
            regrid_att = "remapped via ESMF regrid_with_weights: Nearest Neighbor"
        elif ConfigOptions.regrid_opt[0] == 3:
            regrid_att = "remapped via ESMF regrid_with_weights: Conservative Bilinear"

        # Ensure all processors are synced up before outputting.
        MpiConfig.comm.barrier()

        if MpiConfig.rank == 0:
            while (True):
                # Only output on the master processor.
                try:
                    idOut = Dataset(self.outPath,'w')
                except:
                    ConfigOptions.errMsg = "Unable to create output file: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break

                # Create dimensions.
                try:
                    idOut.createDimension("time",1)
                except:
                    ConfigOptions.errMsg = "Unable to create time dimension in: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.createDimension("y",geoMetaWrfHydro.ny_global)
                except:
                    ConfigOptions.errMsg = "Unable to create y dimension in: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.createDimension("x",geoMetaWrfHydro.nx_global)
                except:
                    ConfigOptions.errMsg = "Unable to create x dimension in: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.createDimension("reference_time",1)
                except:
                    ConfigOptions.errMsg = "Unable to create reference_time dimension in: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break

                # Set global attributes
                try:
                    idOut.model_output_valid_time = self.outDate.strftime("%Y-%m-%d_%H:%M:00")
                except:
                    ConfigOptions.errMsg = "Unable to set the model_output_valid_time attribute in :" + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.model_initialization_time = ConfigOptions.current_fcst_cycle.strftime("%Y-%m-%d_%H:%M:00")
                except:
                    ConfigOptions.errMsg = "Unable to set the model_initialization_time global " \
                                           "attribute in: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break

                # Create variables.
                try:
                    idOut.createVariable('time','i4',('time'))
                except:
                    ConfigOptions.errMsg = "Unable to create time variable in: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    idOut.createVariable('reference_time','i4',('reference_time'))
                except:
                    ConfigOptions.errMsg = "Unable to create reference_time variable in: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break

                # Create geospatial metadata coordinate variables if data was read in from an optional
                # spatial metadata file.
                if ConfigOptions.spatial_meta is not None:
                    # Create coordinate variables and populate with attributes read in.
                    try:
                        idOut.createVariable('x','f8',('x'))
                    except:
                        ConfigOptions.errMsg = "Unable to create x variable in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['x'].setncatts(geoMetaWrfHydro.x_coord_atts)
                    except:
                        ConfigOptions.errMsg = "Unable to establish x coordinate attributes in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['x'][:] = geoMetaWrfHydro.x_coords
                    except:
                        ConfigOptions.errMsg = "Unable to place x coordinate values into output variable " \
                                               "for output file: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        idOut.createVariable('y','f8',('y'))
                    except:
                        ConfigOptions.errMsg = "Unable to create y variable in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['y'].setncatts(geoMetaWrfHydro.y_coord_atts)
                    except:
                        ConfigOptions.errMsg = "Unable to establish y coordinate attributes in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['y'][:] = geoMetaWrfHydro.y_coords
                    except:
                        ConfigOptions.errMsg = "Unable to place y coordinate values into output variable " \
                                               "for output file: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        idOut.createVariable('crs','S1')
                    except:
                        ConfigOptions.errMsg = "Unable to create crs in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables['crs'].setncatts(geoMetaWrfHydro.crs_atts)
                    except:
                        ConfigOptions.errMsg = "Unable to establish crs attributes in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break

                # Loop through and create each variable, along with expected attributes.
                for varTmp in output_variable_attribute_dict:
                    try:
                        idOut.createVariable(varTmp,'f4',('time','y','x'),fill_value=ConfigOptions.globalNdv)
                    except:
                        ConfigOptions.errMsg = "Unable to create " + varTmp + " variable in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp].cell_methods = output_variable_attribute_dict[varTmp][4]
                    except:
                        ConfigOptions.errMsg = "Unable to create cell_methods attribute for: " + varTmp + \
                            " in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp].remap = regrid_att
                    except:
                        ConfigOptions.errMsg = "Unable to create remap attribute for: " + varTmp + \
                            " in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    # Place geospatial metadata attributes in if we have them.
                    if ConfigOptions.spatial_meta is not None:
                        try:
                            idOut.variables[varTmp].grid_mapping = 'crs'
                        except:
                            ConfigOptions.errMsg = "Unable to create grid_mapping attribute for: " + \
                                                   varTmp + " in: " + self.outPath
                            errMod.log_critical(ConfigOptions, MpiConfig)
                            break
                        if 'esri_pe_string' in geoMetaWrfHydro.crs_atts.keys():
                            try:
                                idOut.variables[varTmp].esri_pe_string = geoMetaWrfHydro.crs_atts['esri_pe_string']
                            except:
                                ConfigOptions.errMsg = "Unable to create esri_pe_string attribute for: " + \
                                                       varTmp + " in: " + self.outPath
                                errMod.log_critical(ConfigOptions, MpiConfig)
                                break
                        if 'proj4' in geoMetaWrfHydro.spatial_global_atts.keys():
                            try:
                                idOut.variables[varTmp].proj4 = geoMetaWrfHydro.spatial_global_atts['proj4']
                            except:
                                ConfigOptions.errMsg = "Unable to create proj4 attribute for: " + varTmp + \
                                    " in: " + self.outPath
                                errMod.log_critical(ConfigOptions, MpiConfig)
                                break

                    try:
                        idOut.variables[varTmp].units = output_variable_attribute_dict[varTmp][1]
                    except:
                        ConfigOptions.errMsg = "Unable to create units attribute for: " + varTmp + " in: " + \
                            self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp].standard_name = output_variable_attribute_dict[varTmp][2]
                    except:
                        ConfigOptions.errMsg = "Unable to create standard_name attribute for: " + varTmp + \
                            " in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp].long_name = output_variable_attribute_dict[varTmp][3]
                    except:
                        ConfigOptions.errMsg = "Unable to create long_name attribute for: " + varTmp + \
                            " in: " + self.outPath
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                break

        errMod.check_program_status(ConfigOptions, MpiConfig)

        # Now loop through each variable, collect the data (call on each processor), assemble into the final
        # output grid, and place into the output file (if on processor 0).
        for varTmp in output_variable_attribute_dict:
            # Collect data from the various processors, and place into the output file.
            try:
                final = MpiConfig.comm.gather(self.output_local[output_variable_attribute_dict[varTmp][0],:,:],root=0)
            except:
                ConfigOptions.errMsg = "Unable to gather final grids for: " + varTmp
                errMod.log_critical(ConfigOptions, MpiConfig)
                continue
            MpiConfig.comm.barrier()
            if MpiConfig.rank == 0:
                while (True):
                    try:
                        dataOutTmp = np.concatenate([final[i] for i in range(MpiConfig.size)],axis=0)
                    except:
                        ConfigOptions.errMsg = "Unable to finalize collection of output grids for: " + varTmp
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        idOut.variables[varTmp][0,:,:] = dataOutTmp
                    except:
                        ConfigOptions.errMsg = "Unable to place final output grid for: " + varTmp
                        errMod.log_critical(ConfigOptions, MpiConfig)
                        break
                    # Reset temporary data objects to keep memory usage down.
                    dataOutTmp = None
                    break

            # Reset temporary data objects to keep memory usage down.
            final = None

            errMod.check_program_status(ConfigOptions, MpiConfig)

        if MpiConfig.rank == 0:
            while (True):
                # Close the NetCDF file
                try:
                    idOut.close()
                except:
                    ConfigOptions.errMsg = "Unable to close output file: " + self.outPath
                    errMod.log_critical(ConfigOptions, MpiConfig)
                    break
                break

        errMod.check_program_status(ConfigOptions, MpiConfig)

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
    # Run wgrib2 command to convert GRIB2 file to NetCDF.
    if MpiConfig.rank == 0:
        while (True):
            # Check to see if output file already exists. If so, delete it and
            # override.
            ConfigOptions.statusMsg = "Reading in GRIB2 file: " + GribFileIn
            errMod.log_msg(ConfigOptions, MpiConfig)
            if os.path.isfile(NetCdfFileOut):
                ConfigOptions.statusMsg = "Overriding temporary NetCDF file: " + NetCdfFileOut
                errMod.log_warning(ConfigOptions,MpiConfig)
            try:
                cmdOutput = subprocess.Popen([Wgrib2Cmd], stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True)
                out, err = cmdOutput.communicate()
                exitcode = cmdOutput.returncode
                #subprocess.run([Wgrib2Cmd],shell=True)
            except:
                ConfigOptions.errMsg = "Unable to convert: " + GribFileIn + " to " + \
                    NetCdfFileOut
                errMod.log_critical(ConfigOptions,MpiConfig)
                idTmp = None
                break

            # Reset temporary subprocess variables.
            out = None
            err = None
            exitcode = None

            # Ensure file exists.
            if not os.path.isfile(NetCdfFileOut):
                ConfigOptions.errMsg = "Expected NetCDF file: " + NetCdfFileOut + \
                    " not found. It's possible the GRIB2 variable was not found."
                errMod.log_critical(ConfigOptions,MpiConfig)
                idTmp = None
                break

            # Open the NetCDF file.
            try:
                idTmp = Dataset(NetCdfFileOut,'r')
            except:
                ConfigOptions.errMsg = "Unable to open input NetCDF file: " + \
                                        NetCdfFileOut
                errMod.log_critical(ConfigOptions,MpiConfig)
                idTmp = None
                break

            if idTmp is not None:
                # Check for expected lat/lon variables.
                if 'latitude' not in idTmp.variables.keys():
                    ConfigOptions.errMsg = "Unable to locate latitude from: " + \
                                           GribFileIn
                    errMod.log_critical(ConfigOptions,MpiConfig)
                    idTmp = None
                    break
            if idTmp is not None:
                if 'longitude' not in idTmp.variables.keys():
                    ConfigOptions.errMsg = "Unable t locate longitude from: " + \
                                           GribFileIn
                    errMod.log_critical(ConfigOptions,MpiConfig)
                    idTmp = None
                    break

            if idTmp is not None:
                # Loop through all the expected variables.
                if inputVar not in idTmp.variables.keys():
                    ConfigOptions.errMsg = "Unable to locate expected variable: " + \
                                           inputVar + " in: " + NetCdfFileOut
                    errMod.log_critical(ConfigOptions,MpiConfig)
                    idTmp = None
                    break

            break
    else:
        idTmp = None

    # Return the NetCDF file handle back to the user.
    return idTmp

def open_netcdf_forcing(NetCdfFileIn,ConfigOptions,MpiConfig):
    """
    Generic function to convert a NetCDF forcing file given a list of input forcing
    variables.
    :param GribFileIn:
    :param NetCdfFileOut:
    :param ConfigOptions:
    :return:
    """
    # Open the NetCDF file on the master processor and read in data.
    if MpiConfig.rank == 0:
        while (True):
            # Ensure file exists.
            if not os.path.isfile(NetCdfFileIn):
                ConfigOptions.errMsg = "Expected NetCDF file: " + NetCdfFileIn + \
                                        " not found."
                errMod.log_critical(ConfigOptions,MpiConfig)
                idTmp = None
                break

            # Open the NetCDF file.
            try:
                idTmp = Dataset(NetCdfFileIn, 'r')
            except:
                ConfigOptions.errMsg = "Unable to open input NetCDF file: " + \
                                        NetCdfFileIn
                errMod.log_critical(ConfigOptions,MpiConfig)
                idTmp = None
                break

            if idTmp is not None:
                # Check for expected lat/lon variables.
                if 'latitude' not in idTmp.variables.keys():
                    ConfigOptions.errMsg = "Unable to locate latitude from: " + \
                                            NetCdfFileIn
                    errMod.log_critical(ConfigOptions,MpiConfig)
                    idTmp = None
                    break
            if idTmp is not None:
                if 'longitude' not in idTmp.variables.keys():
                    ConfigOptions.errMsg = "Unable t locate longitude from: " + \
                                            NetCdfFileIn
                    errMod.log_critical(ConfigOptions,MpiConfig)
                    idTmp = None
                    break
            break
    else:
        idTmp = None
    errMod.check_program_status(ConfigOptions, MpiConfig)

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
    if MpiConfig.rank == 0:
        # Unzip the file in place.
        try:
            with gzip.open(GzFileIn, 'rb') as fTmpGz:
                with open(FileOut, 'wb') as fTmp:
                    shutil.copyfileobj(fTmpGz, fTmp)
        except:
            ConfigOptions.errMsg = "Unable to unzip: " + GzFileIn
            raise Exception()

        if not os.path.isfile(FileOut):
            ConfigOptions.errMsg = "Unable to locate expected unzipped file: " + \
                                   FileOut
            raise Exception()
    else:
        return