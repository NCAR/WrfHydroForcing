"""
High-level module file that will handle supplemental analysis/observed precipitation grids
that will replace precipitation in the final output files.
"""
import numpy as np

from core import time_handling
from core import regrid
from core import timeInterpMod


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

    def define_product(self):
        """
        Function to define the product name based on the mapping
        forcing key value.
        :return:
        """
        product_names = {
            1: "MRMS_1HR_Radar_Only",
            2: "MRMS_1HR_Gage_Corrected",
            3: "WRF_ARW_Hawaii_2p5km_PCP",
            4: "WRF_ARW_PuertoRico_2p5km_PCP",
            5: "CONUS_MRMS_1HR_MultiSensor",
            6: "Hawaii_MRMS_1HR_MultiSensor",
            7: "MRMS_LiquidWaterFraction",
            8: "NBM_CORE_CONUS_APCP",
            9: "NBM_CORE_ALASKA_APCP"
        }
        self.productName = product_names[self.keyValue]

        ## DEFINED IN CONFIG
        # product_types = {
        #     1: "GRIB2",
        #     2: "GRIB2",
        #     3: "GRIB2",
        #     4: "GRIB2",
        #     5: "GRIB2"
        # }
        # self.fileType = product_types[self.keyValue]
        if self.fileType == 'GRIB1':
            self.file_ext = '.grb'
        elif self.fileType == 'GRIB2':
            self.file_ext = '.grib2'
        elif self.fileType == 'NETCDF':
            self.file_ext = '.nc'

        grib_vars_in = {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None
        }
        self.grib_vars = grib_vars_in[self.keyValue]

        grib_levels_in = {
            1: ['BLAH'],
            2: ['BLAH'],
            3: ['BLAH'],
            4: ['BLAH'],
            5: ['BLAH'],
            6: ['BLAH'],
            7: ['BLAH'],
            8: ['BLAH'],
            9: ['BLAH']
        }
        self.grib_levels = grib_levels_in[self.keyValue]

        netcdf_variables = {
            1: ['RadarOnlyQPE01H_0mabovemeansealevel'],
            2: ['GaugeCorrQPE01H_0mabovemeansealevel'],
            3: ['APCP_surface'],
            4: ['APCP_surface'],
            5: ['MultiSensorQPE01H_0mabovemeansealevel'],
            6: ['MultiSensorQPE01H_0mabovemeansealevel'],
            7: ['sbcv2_lwf'],
            8: ['APCP_surface'],
            9: ['APCP_surface']
        }
        self.netcdf_var_names = netcdf_variables[self.keyValue]

        netcdf_rqi_variables = {
            1: ['RadarQualityIndex_0mabovemeansealevel'],
            2: ['RadarQualityIndex_0mabovemeansealevel'],
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None
        }
        self.rqi_netcdf_var_names = netcdf_rqi_variables[self.keyValue]

        output_variables = {
            1: 3,       # RAINRATE
            2: 3,
            3: 3,
            4: 3,
            5: 3,
            6: 3,
            7: 8,        # LQFRAC
            8: 3,
            9: 3
        }
        self.output_var_idx = output_variables[self.keyValue]

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
        find_neighbor_files = {
            1: time_handling.find_hourly_mrms_radar_neighbors,
            2: time_handling.find_hourly_mrms_radar_neighbors,
            3: time_handling.find_hourly_wrf_arw_neighbors,
            4: time_handling.find_hourly_wrf_arw_neighbors,
            5: time_handling.find_hourly_mrms_radar_neighbors,
            6: time_handling.find_hourly_mrms_radar_neighbors,
            7: time_handling.find_sbcv2_lwf_neighbors,
            8: time_handling.find_hourly_nbm_apcp_neighbors,
            9: time_handling.find_hourly_nbm_apcp_neighbors
        }

        find_neighbor_files[self.keyValue](self, ConfigOptions, dCurrent, MpiConfig)
        # try:
        #    find_neighbor_files[self.keyValue](self,ConfigOptions,dCurrent,MpiConfig)
        # except TypeError:
        #    ConfigOptions.errMsg = "Unable to execute find_neighbor_files for " \
        #                           "supplemental precipitation: " + self.productName
        #    raise
        # except:
        #    raise

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
        regrid_inputs = {
            1: regrid.regrid_mrms_hourly,
            2: regrid.regrid_mrms_hourly,
            3: regrid.regrid_hourly_wrf_arw_hi_res_pcp,
            4: regrid.regrid_hourly_wrf_arw_hi_res_pcp,
            5: regrid.regrid_mrms_hourly,
            6: regrid.regrid_mrms_hourly,
            7: regrid.regrid_sbcv2_liquid_water_fraction,
            8: regrid.regrid_hourly_nbm_apcp,
            9: regrid.regrid_hourly_nbm_apcp
        }
        regrid_inputs[self.keyValue](self,ConfigOptions,wrfHyroGeoMeta,MpiConfig)
        #try:
        #    regrid_inputs[self.keyValue](self,ConfigOptions,MpiConfig)
        #except:
        #    ConfigOptions.errMsg = "Unable to execute regrid_inputs for " + \
        #        "input forcing: " + self.productName
        #    raise

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
            0: timeInterpMod.no_interpolation_supp_pcp,
            1: timeInterpMod.nearest_neighbor_supp_pcp,
            2: timeInterpMod.weighted_average_supp_pcp
        }
        temporal_interpolate_inputs[self.timeInterpOpt](self,ConfigOptions,MpiConfig)
        #temporal_interpolate_inputs[self.keyValue](self,ConfigOptions,MpiConfig)
        #try:
        #    temporal_interpolate_inputs[self.timeInterpOpt](self,ConfigOptions,MpiConfig)
        #except:
        #    ConfigOptions.errMsg = "Unable to execute temporal_interpolate_inputs " + \
        #        " for input forcing: " + self.productName
        #    raise

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
        InputDict[supp_pcp_key].define_product()

        # Initialize the local final grid of values
        InputDict[supp_pcp_key].final_supp_precip = np.empty([GeoMetaWrfHydro.ny_local,
                                                              GeoMetaWrfHydro.nx_local],
                                                             np.float64)
        InputDict[supp_pcp_key].regridded_mask = np.empty([GeoMetaWrfHydro.ny_local,
                                                           GeoMetaWrfHydro.nx_local], np.float32)

        InputDict[supp_pcp_key].userCycleOffset = ConfigOptions.supp_input_offsets[supp_pcp_tmp]

    return InputDict

