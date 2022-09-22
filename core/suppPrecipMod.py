"""
High-level module file that will handle supplemental analysis/observed precipitation grids
that will replace precipitation in the final output files.
"""
import numpy as np

from core import time_handling
from core import regrid
from core import timeInterpMod
from strenum import StrEnum
from core.enumConfig import TemporalInterpEnum
import yaml

class supplemental_precip:
    """
    This is an abstract class that will define all the parameters
    of a single supplemental precipitation product.
    """
    def __init__(self):
        """
        Initializing all attributes and objects to None.
        """
        self.keyValue = None
        self.inDir = None
        self.enforce = None
        self.productName = None
        self.fileType = None
        self.nx_global = None
        self.ny_global = None
        self.nx_local = None
        self.ny_local = None
        self.x_lower_bound = None
        self.x_upper_bound = None
        self.y_lower_bound = None
        self.y_upper_bound = None
        self.regridOpt = None
        self.timeInterpOpt = None
        self.esmf_lats = None
        self.esmf_lons = None
        self.esmf_grid_in = None
        self.regridComplete = False
        self.regridObj = None
        self.esmf_field_in = None
        self.esmf_field_out = None
        self.regridded_precip1 = None
        self.regridded_precip2 = None
        self.regridded_rqi1 = None
        self.regridded_rqi2 = None
        self.regridded_mask = None
        self.final_supp_precip = None
        self.file_in1 = None
        self.file_in2 = None
        self.rqiMethod = None
        self.rqiThresh = None
        self.rqi_file_in1 = None
        self.rqi_file_in2 = None
        self.pcp_hour1 = None
        self.pcp_hour2 = None
        self.pcp_date1 = None
        self.pcp_date2 = None
        self.fcst_hour1 = None
        self.fcst_hour2 = None
        self.input_frequency = None
        self.netcdf_var_names = None
        self.rqi_netcdf_var_names = None
        self.grib_levels = None
        self.grib_vars = None
        self.tmpFile = None
        self.userCycleOffset = None

        self.global_x_lower = None
        self.global_y_lower = None
        self.global_x_upper = None
        self.global_y_upper = None
        self.has_cache = False
        self.suppPrecipModYaml = None

    def define_product(self):
        """
        Function to define the product name based on the mapping
        forcing key value.
        :return:
        """

        yaml_stream = None
        try:
            yaml_stream = open(self.suppPrecipModYaml)
            config = yaml.safe_load(yaml_stream)
        except yaml.YAMLError as yaml_exc:
            err_handler.err_out_screen('Error parsing the configuration file: %s\n%s' % (self.suppPrecipModYaml,yaml_exc))
        except IOError:
            err_handler.err_out_screen('Unable to open the configuration file: %s' % self.suppPrecipModYaml)
        finally:
            if yaml_stream:
                yaml_stream.close()

        try:
            inputs = config[self.keyValue]
        except KeyError:
            err_handler.err_out_screen('Unable to locate Input map in configuration file.')

        self.productName = inputs["product_name"]

        if self.fileType == 'GRIB1':
            self.file_ext = '.grb'
        elif self.fileType == 'GRIB2':
            self.file_ext = '.grib2'
        elif self.fileType == 'NETCDF':
            self.file_ext = '.nc'

        self.grib_vars = inputs["grib_vars_in"]

        self.grib_levels = inputs["grib_levels_in"]

        self.netcdf_var_names = inputs["netcdf_variables"]

        self.rqi_netcdf_var_names = inputs["netcdf_rqi_variables"]

        self.output_var_idx = inputs["output_variables"]

    def calc_neighbor_files(self,ConfigOptions,dCurrent,MpiConfig):
        """
        Function that will calculate the last/next expected
        supplemental precipitation file based on the current time step that
        is being processed.
        :param ConfigOptions:
        :param dCurrent:
        :return:
        """
        # First calculate the current input cycle date this
        # WRF-Hydro output timestep corresponds to.
        SuppForcingPcpEnum = ConfigOptions.SuppForcingPcpEnum
        find_neighbor_files = {
            str(SuppForcingPcpEnum.MRMS.name)         : time_handling.find_hourly_mrms_radar_neighbors,
            str(SuppForcingPcpEnum.MRMS_GAGE.name)    : time_handling.find_hourly_mrms_radar_neighbors,
            str(SuppForcingPcpEnum.WRF_ARW_HI.name)   : time_handling.find_hourly_wrf_arw_neighbors,
            str(SuppForcingPcpEnum.WRF_ARW_PR.name)   : time_handling.find_hourly_wrf_arw_neighbors,
            str(SuppForcingPcpEnum.MRMS_CONUS_MS.name): time_handling.find_hourly_mrms_radar_neighbors,
            str(SuppForcingPcpEnum.MRMS_HI_MS.name)   : time_handling.find_hourly_mrms_radar_neighbors,
            str(SuppForcingPcpEnum.MRMS_SBCV2.name)   : time_handling.find_sbcv2_lwf_neighbors,
            str(SuppForcingPcpEnum.AK_OPT1.name)      : time_handling.find_hourly_nbm_apcp_neighbors,
            str(SuppForcingPcpEnum.AK_OPT2.name)      : time_handling.find_hourly_nbm_apcp_neighbors,
            str(SuppForcingPcpEnum.AK_MRMS.name)      : time_handling.find_hourly_mrms_radar_neighbors,
            str(SuppForcingPcpEnum.AK_NWS_IV.name)    : time_handling.find_ak_ext_ana_precip_neighbors
        }

        find_neighbor_files[SuppForcingPcpEnum(self.keyValue).name](self, ConfigOptions, dCurrent, MpiConfig)

    def regrid_inputs(self,ConfigOptions,wrfHyroGeoMeta,MpiConfig):
        """
        Polymorphic function that will regrid input forcings to the
        supplemental precipitation grids for this particular timestep. For
        timesteps that require interpolation, two sets of input
        forcing grids will be regridded IF we have come across new
        files and the process flag has been reset.
        :param ConfigOptions:
        :return:
        """
        # Establish a mapping dictionary that will point the
        # code to the functions to that will regrid the data.
        SuppForcingPcpEnum = ConfigOptions.SuppForcingPcpEnum
        regrid_inputs = {
            str(SuppForcingPcpEnum.MRMS.name)         : regrid.regrid_mrms_hourly,
            str(SuppForcingPcpEnum.MRMS_GAGE.name)    : regrid.regrid_mrms_hourly,
            str(SuppForcingPcpEnum.WRF_ARW_HI.name)   : regrid.regrid_hourly_wrf_arw_hi_res_pcp,
            str(SuppForcingPcpEnum.WRF_ARW_PR.name)   : regrid.regrid_hourly_wrf_arw_hi_res_pcp,
            str(SuppForcingPcpEnum.MRMS_CONUS_MS.name): regrid.regrid_mrms_hourly,
            str(SuppForcingPcpEnum.MRMS_HI_MS.name)   : regrid.regrid_mrms_hourly,
            str(SuppForcingPcpEnum.MRMS_SBCV2.name)   : regrid.regrid_sbcv2_liquid_water_fraction,
            str(SuppForcingPcpEnum.AK_OPT1.name)      : regrid.regrid_hourly_nbm_apcp,
            str(SuppForcingPcpEnum.AK_OPT2.name)      : regrid.regrid_hourly_nbm_apcp,
            str(SuppForcingPcpEnum.AK_MRMS.name)      : regrid.regrid_mrms_hourly,
            str(SuppForcingPcpEnum.AK_NWS_IV.name)    : regrid.regrid_ak_ext_ana_pcp
        }
        regrid_inputs[self.keyValue](self,ConfigOptions,wrfHyroGeoMeta,MpiConfig)

    def temporal_interpolate_inputs(self,ConfigOptions,MpiConfig):
        """
        Polymorphic function that will run temporal interpolation of
        the supplemental precipitation grids that have been regridded. This is
        especially important for supplemental precips that have large output
        frequencies. This is also important for frequent WRF-Hydro
        input timesteps.
        :param ConfigOptions:
        :param MpiConfig:
        :return:
        """
        temporal_interpolate_inputs = {
            str(TemporalInterpEnum.NONE.name)             : timeInterpMod.no_interpolation_supp_pcp,
            str(TemporalInterpEnum.NEAREST_NEIGHBOR.name) : timeInterpMod.nearest_neighbor_supp_pcp,
            str(TemporalInterpEnum.LINEAR_WEIGHT_AVG.name): timeInterpMod.weighted_average_supp_pcp
        }
        temporal_interpolate_inputs[TemporalInterpEnum(self.timeInterpOpt).name](self,ConfigOptions,MpiConfig)
    
def initDict(ConfigOptions,GeoMetaWrfHydro):
    """
    Initial function to create an supplemental dictionary, which
    will contain an abstract class for each supplemental precip product.
    This gets called one time by the parent calling program.
    :param ConfigOptions:
    :return: InputDict - A dictionary defining our inputs.
    """
    # Initialize an empty dictionary
    InputDict = {}

    for supp_pcp_tmp in range(0,ConfigOptions.number_supp_pcp):
        supp_pcp_key = ConfigOptions.supp_precip_forcings[supp_pcp_tmp]
        InputDict[supp_pcp_key] = supplemental_precip()
        InputDict[supp_pcp_key].keyValue = supp_pcp_key
        InputDict[supp_pcp_key].regridOpt = ConfigOptions.regrid_opt_supp_pcp[supp_pcp_tmp]
        InputDict[supp_pcp_key].enforce = ConfigOptions.supp_precip_mandatory[supp_pcp_tmp]
        InputDict[supp_pcp_key].timeInterpOpt = ConfigOptions.suppTemporalInterp[supp_pcp_tmp]

        InputDict[supp_pcp_key].inDir = ConfigOptions.supp_precip_dirs[supp_pcp_tmp]
        InputDict[supp_pcp_key].fileType = ConfigOptions.supp_precip_file_types[supp_pcp_tmp]
        InputDict[supp_pcp_key].suppPrecipModYaml = ConfigOptions.suppPrecipModYaml
        InputDict[supp_pcp_key].define_product()

        # Initialize the local final grid of values
        InputDict[supp_pcp_key].final_supp_precip = np.empty([GeoMetaWrfHydro.ny_local,
                                                              GeoMetaWrfHydro.nx_local],
                                                             np.float64)
        InputDict[supp_pcp_key].regridded_mask = np.empty([GeoMetaWrfHydro.ny_local,
                                                           GeoMetaWrfHydro.nx_local], np.float32)

        InputDict[supp_pcp_key].userCycleOffset = ConfigOptions.supp_input_offsets[supp_pcp_tmp]

        if ConfigOptions.rqiMethod is not None:
            InputDict[supp_pcp_key].rqiMethod = ConfigOptions.rqiMethod[supp_pcp_tmp]
            InputDict[supp_pcp_key].rqiThresh = ConfigOptions.rqiThresh[supp_pcp_tmp]

    return InputDict
