"""
This module will guide the forcing engine in defining
parameters in all input forcing products. These parameters
include things such as file types, grid definitions (including
initializing ESMF grids and regrid objects), etc
"""
import numpy as np
from strenum import StrEnum
from core import time_handling
#from core import regrid
#from core import timeInterpMod
import yaml
from core.enumConfig import TemporalInterpEnum
from core.enumConfig import DownScaleHumidEnum

class input_forcings:
    """
    This is an abstract class that will define all the parameters
    of a single input forcing product.
    """

    # Constants
    GRIB2  = "GRIB2"
    GRIB1  = "GRIB1"
    NETCDF = "NETCDF"

    def __init__(self):
        """
        Initializing all attributes and objects to None.
        """
        self.keyValue               = None
        self.inDir                  = None
        self.enforce                = None
        self.paramDir               = None
        self.userFcstHorizon        = None
        self.userCycleOffset        = None
        self.productName            = None
        self.fileType               = None
        self.file_ext               = None
        self.nx_global              = None
        self.ny_global              = None
        self.nx_local               = None
        self.ny_local               = None
        self.x_lower_bound          = None
        self.x_upper_bound          = None
        self.y_lower_bound          = None
        self.y_upper_bound          = None
        self.cycleFreq              = None
        self.outFreq                = None
        self.regridOpt              = None
        self.timeInterpOpt          = None
        self.t2dDownscaleOpt        = None
        self.lapseGrid              = None
        self.rqiClimoGrid           = None
        self.swDowscaleOpt          = None
        self.precipDownscaleOpt     = None
        self.nwmPRISM_numGrid       = None
        self.nwmPRISM_denGrid       = None
        self.q2dDownscaleOpt        = None
        self.psfcDownscaleOpt       = None
        self.t2dBiasCorrectOpt      = None
        self.swBiasCorrectOpt       = None
        self.precipBiasCorrectOpt   = None
        self.q2dBiasCorrectOpt      = None
        self.windBiasCorrectOpt     = None
        self.psfcBiasCorrectOpt     = None
        self.lwBiasCorrectOpt       = None
        self.esmf_lats              = None
        self.esmf_lons              = None
        self.esmf_grid_in           = None
        self.regridComplete         = False
        self.regridObj              = None
        self.esmf_field_in          = None
        self.esmf_field_out         = None
        # --------------------------------
        # Only used for CFSv2 bias correction
        # as bias correction needs to take
        # place prior to regridding.
        self.coarse_input_forcings1 = None
        self.coarse_input_forcings2 = None
        # --------------------------------
        self.skip                   = False
        self.regridded_forcings1    = None
        self.regridded_forcings2    = None
        self.globalPcpRate1         = None
        self.globalPcpRate2         = None
        self.regridded_mask         = None
        self.final_forcings         = None
        self.ndv                    = None
        self.file_in1               = None
        self.file_in2               = None
        self.fcst_hour1             = None
        self.fcst_hour2             = None
        self.fcst_date1             = None
        self.fcst_date2             = None
        self.height                 = None
        self.netcdf_var_names       = None
        self.grib_mes_idx           = None
        self.input_map_output       = None
        self.grib_levels            = None
        self.grib_vars              = None
        self.tmpFile                = None
        self.tmpFileHeight          = None
        self.psfcTmp                = None
        self.t2dTmp                 = None
        self.rstFlag                = 0
        self.regridded_precip1      = None
        self.regridded_precip2      = None
        self.border                 = None
        self.forcingInputModYaml    = None

    def define_product(self):
        """
        Function to define the product name based on the mapping
        forcing key value.
        :return:
        """
        GRIB1  = self.GRIB1
        GRIB2  = self.GRIB2
        NETCDF = self.NETCDF

        yaml_stream = None
        try:
            yaml_stream = open(self.forcingInputModYaml)
            config = yaml.safe_load(yaml_stream)
        except yaml.YAMLError as yaml_exc:
            err_handler.err_out_screen('Error parsing the configuration file: %s\n%s' % (self.forcingInputModYaml,yaml_exc))
        except IOError:
            err_handler.err_out_screen('Unable to open the configuration file: %s' % self.forcingInputModYaml)
        finally:
            if yaml_stream:
                yaml_stream.close()
        try:
            inputs = config[self.keyValue]
        except KeyError:
            err_handler.err_out_screen('Unable to locate Input map in configuration file.')
        self.productName = inputs["product_name"]

        if self.fileType   == 'GRIB1':
            self.file_ext  = '.grb'
        elif self.fileType == 'GRIB2':
            self.file_ext  = '.grib2'
        elif self.fileType == 'NETCDF':
            self.file_ext  = '.nc'

        self.cycleFreq        = inputs["cycle_freq_minutes"]

        self.grib_vars        = inputs["grib_vars_in"]

        self.grib_levels      = inputs["grib_levels_in"]

        self.netcdf_var_names = inputs["netcdf_variables"]

        self.grib_mes_idx     = inputs["grib_message_idx"]

        self.input_map_output = inputs["input_map_to_outputs"]

    def calc_neighbor_files(self,ConfigOptions,dCurrent,MpiConfig):
        """
        Function that will calculate the last/next expected
        input forcing file based on the current time step that
        is being processed.
        :param ConfigOptions:
        :param dCurrent:
        :return:
        """
        # First calculate the current input cycle date this
        # WRF-Hydro output timestep corresponds to.
        ForcingEnum = ConfigOptions.ForcingEnum

        find_neighbor_files = {
            str(ForcingEnum.NLDAS.name)          : time_handling.find_nldas_neighbors,
            str(ForcingEnum.GFS_GLOBAL.name)     : time_handling.find_gfs_neighbors,
            str(ForcingEnum.HRRR.name)           : time_handling.find_conus_hrrr_neighbors,
            str(ForcingEnum.RAP.name)            : time_handling.find_conus_rap_neighbors,
            str(ForcingEnum.CFS_V2.name)         : time_handling.find_cfsv2_neighbors,
            str(ForcingEnum.WRF_NEST_HI.name)    : time_handling.find_hourly_wrf_arw_neighbors,
            str(ForcingEnum.GFS_GLOBAL_25.name)  : time_handling.find_gfs_neighbors,
            str(ForcingEnum.CUSTOM_1.name)       : time_handling.find_custom_hourly_neighbors,
            str(ForcingEnum.CUSTOM_2.name)       : time_handling.find_custom_hourly_neighbors,
            str(ForcingEnum.CUSTOM_3.name)       : time_handling.find_aorc_neighbors,
            str(ForcingEnum.NAM_NEST_HI.name)    : time_handling.find_nam_nest_neighbors,
            str(ForcingEnum.NAM_NEST_PR.name)    : time_handling.find_nam_nest_neighbors,
            str(ForcingEnum.NAM_NEST_AK.name)    : time_handling.find_nam_nest_neighbors,
            str(ForcingEnum.NAM_NEST_HI_RAD.name): time_handling.find_nam_nest_neighbors,
            str(ForcingEnum.NAM_NEST_PR_RAD.name): time_handling.find_nam_nest_neighbors,
            str(ForcingEnum.WRF_ARW_PR.name)     : time_handling.find_hourly_wrf_arw_neighbors,
            str(ForcingEnum.HRRR_AK.name)        : time_handling.find_ak_hrrr_neighbors,
            str(ForcingEnum.HRRR_AK_EXT.name)    : time_handling.find_ak_ext_ana_neighbors,
        }

        find_neighbor_files[self.keyValue](self, ConfigOptions, dCurrent,MpiConfig)

    def regrid_inputs(self, ConfigOptions, wrfHyroGeoMeta, MpiConfig):
        """
        Polymorphic function that will regrid input forcings to the
        final output grids for this particular timestep. For
        timesteps that require interpolation, two sets of input
        forcing grids will be regridded IF we have come across new
        files and the process flag has been reset.
        :param ConfigOptions:
        :return:
        """
        # Establish a mapping dictionary that will point the
        # code to the functions to that will regrid the data.
        from core import regrid
        ForcingEnum = ConfigOptions.ForcingEnum
           
        regrid_inputs = {
            str(ForcingEnum.NLDAS.name)          : regrid.regrid_conus_rap,
            str(ForcingEnum.GFS_GLOBAL.name)     : regrid.regrid_gfs,
            str(ForcingEnum.HRRR.name)           : regrid.regrid_conus_hrrr,
            str(ForcingEnum.RAP.name)            : regrid.regrid_conus_rap,
            str(ForcingEnum.CFS_V2.name)         : regrid.regrid_cfsv2,
            str(ForcingEnum.WRF_NEST_HI.name)    : regrid.regrid_hourly_wrf_arw,
            str(ForcingEnum.GFS_GLOBAL_25.name)  : regrid.regrid_gfs,
            str(ForcingEnum.CUSTOM_1.name)       : regrid.regrid_custom_hourly_netcdf,
            str(ForcingEnum.CUSTOM_2.name)       : regrid.regrid_custom_hourly_netcdf,
            str(ForcingEnum.CUSTOM_3.name)       : regrid.regrid_custom_hourly_netcdf,
            str(ForcingEnum.NAM_NEST_HI.name)    : regrid.regrid_nam_nest,
            str(ForcingEnum.NAM_NEST_PR.name)    : regrid.regrid_nam_nest,
            str(ForcingEnum.NAM_NEST_AK.name)    : regrid.regrid_nam_nest,
            str(ForcingEnum.NAM_NEST_HI_RAD.name): regrid.regrid_nam_nest,
            str(ForcingEnum.NAM_NEST_PR_RAD.name): regrid.regrid_nam_nest,
            str(ForcingEnum.WRF_ARW_PR.name)     : regrid.regrid_hourly_wrf_arw,
            str(ForcingEnum.HRRR_AK.name)        : regrid.regrid_conus_hrrr,
            str(ForcingEnum.HRRR_AK_EXT.name)    : regrid.regrid_ak_ext_ana
        }
        regrid_inputs[self.keyValue](self,ConfigOptions,wrfHyroGeoMeta,MpiConfig)

    def temporal_interpolate_inputs(self,ConfigOptions,MpiConfig):
        """
        Polymorphic function that will run temporal interpolation of
        the input forcing grids that have been regridded. This is
        especially important for forcings that have large output
        frequencies. This is also important for frequent WRF-Hydro
        input timesteps.
        :param ConfigOptions:
        :param MpiConfig:
        :return:
        """
        from core import timeInterpMod
        temporal_interpolate_inputs = {
            str(TemporalInterpEnum.NONE.name)             : timeInterpMod.no_interpolation,
            str(TemporalInterpEnum.NEAREST_NEIGHBOR.name) : timeInterpMod.nearest_neighbor,
            str(TemporalInterpEnum.LINEAR_WEIGHT_AVG.name): timeInterpMod.weighted_average
        }
        temporal_interpolate_inputs[self.timeInterpOpt](self,ConfigOptions,MpiConfig)

def initDict(ConfigOptions,GeoMetaWrfHydro):
    """
    Initial function to create an input forcing dictionary, which
    will contain an abstract class for each input forcing product.
    This gets called one time by the parent calling program.
    :param ConfigOptions:
    :return: InputDict - A dictionary defining our inputs.
    """
    # Initialize an empty dictionary
    InputDict = {}

    # Loop through and initialize the empty class for each product.
    custom_count = 0
    for force_tmp in range(0,ConfigOptions.number_inputs):
        force_key = ConfigOptions.input_forcings[force_tmp]
        InputDict[force_key] = input_forcings()
        InputDict[force_key].keyValue = force_key
        InputDict[force_key].regridOpt = ConfigOptions.regrid_opt[force_tmp]
        InputDict[force_key].enforce = ConfigOptions.input_force_mandatory[force_tmp]
        InputDict[force_key].timeInterpOpt = ConfigOptions.forceTemoralInterp[force_tmp]

        InputDict[force_key].q2dDownscaleOpt = ConfigOptions.q2dDownscaleOpt[force_tmp]
        InputDict[force_key].t2dDownscaleOpt = ConfigOptions.t2dDownscaleOpt[force_tmp]
        InputDict[force_key].precipDownscaleOpt = ConfigOptions.precipDownscaleOpt[force_tmp]
        InputDict[force_key].swDowscaleOpt = ConfigOptions.swDownscaleOpt[force_tmp]
        InputDict[force_key].psfcDownscaleOpt = ConfigOptions.psfcDownscaleOpt[force_tmp]
        # Check to make sure the necessary input files for downscaling are present.
        #if InputDict[force_key].t2dDownscaleOpt == 2:
        #    # We are using a pre-calculated lapse rate on the WRF-Hydro grid.
        #    pathCheck = ConfigOptions.downscaleParamDir = "/T2M_Lapse_Rate_" + \
        #        InputDict[force_key].productName + ".nc"
        #    if not os.path.isfile(pathCheck):
        #        ConfigOptions.errMsg = "Expected temperature lapse rate grid: " + \
        #            pathCheck + " not found."
        #        raise Exception

        InputDict[force_key].t2dBiasCorrectOpt = ConfigOptions.t2BiasCorrectOpt[force_tmp]
        InputDict[force_key].q2dBiasCorrectOpt = ConfigOptions.q2BiasCorrectOpt[force_tmp]
        InputDict[force_key].precipBiasCorrectOpt = ConfigOptions.precipBiasCorrectOpt[force_tmp]
        InputDict[force_key].swBiasCorrectOpt = ConfigOptions.swBiasCorrectOpt[force_tmp]
        InputDict[force_key].lwBiasCorrectOpt = ConfigOptions.lwBiasCorrectOpt[force_tmp]
        InputDict[force_key].windBiasCorrectOpt = ConfigOptions.windBiasCorrect[force_tmp]
        InputDict[force_key].psfcBiasCorrectOpt = ConfigOptions.psfcBiasCorrectOpt[force_tmp]

        InputDict[force_key].inDir = ConfigOptions.input_force_dirs[force_tmp]
        InputDict[force_key].paramDir = ConfigOptions.dScaleParamDirs[force_tmp]
        InputDict[force_key].fileType = ConfigOptions.input_force_types[force_tmp]
        InputDict[force_key].forcingInputModYaml = ConfigOptions.forcingInputModYaml
        InputDict[force_key].define_product()
        InputDict[force_key].userFcstHorizon = ConfigOptions.fcst_input_horizons[force_tmp]
        InputDict[force_key].userCycleOffset = ConfigOptions.fcst_input_offsets[force_tmp]

        InputDict[force_key].border = ConfigOptions.ignored_border_widths[force_tmp]

        # If we have specified specific humidity downscaling, establish arrays to hold
        # temporary temperature arrays that are un-downscaled.
        if InputDict[force_key].q2dDownscaleOpt != DownScaleHumidEnum.NONE.name:
            InputDict[force_key].t2dTmp = np.empty([GeoMetaWrfHydro.ny_local,
                                                    GeoMetaWrfHydro.nx_local],
                                                   np.float32)
            InputDict[force_key].psfcTmp = np.empty([GeoMetaWrfHydro.ny_local,
                                                    GeoMetaWrfHydro.nx_local],
                                                   np.float32)

        # Initialize the local final grid of values. This is represntative
        # of the local grid for this forcing, for a specific output timesetp.
        # This grid will be updated from one output timestep to another, and
        # also through downscaling and bias correction.
        InputDict[force_key].final_forcings = np.empty([len(ConfigOptions.OutputEnum),GeoMetaWrfHydro.ny_local,
                                                        GeoMetaWrfHydro.nx_local],
                                                       np.float64)
        InputDict[force_key].height = np.empty([GeoMetaWrfHydro.ny_local,
                                                GeoMetaWrfHydro.nx_local],np.float32)
        InputDict[force_key].regridded_mask = np.empty([GeoMetaWrfHydro.ny_local,
                                                GeoMetaWrfHydro.nx_local],np.float32)

        # Obtain custom input cycle frequencies
        ForcingEnum = ConfigOptions.ForcingEnum
        if force_key == ForcingEnum.CUSTOM_1.name or force_key == ForcingEnum.CUSTOM_2.name:
            InputDict[force_key].cycleFreq = ConfigOptions.customFcstFreq[custom_count]
            custom_count = custom_count + 1

    return InputDict
