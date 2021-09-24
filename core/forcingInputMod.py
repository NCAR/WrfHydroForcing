"""
This module will guide the forcing engine in defining
parameters in all input forcing products. These parameters
include things such as file types, grid definitions (including
initializing ESMF grids and regrid objects), etc
"""
import numpy as np

from core import time_handling
from core import regrid
from core import timeInterpMod


class input_forcings:
    """
    This is an abstract class that will define all the parameters
    of a single input forcing product.
    """
    
    # Constants
    GRIB2 = "GRIB2"
    GRIB1 = "GRIB1"
    NETCDF = "NETCDF"
    
    def __init__(self):
        """
        Initializing all attributes and objects to None.
        """
        self.keyValue = None
        self.inDir = None
        self.enforce = None
        self.paramDir = None
        self.userFcstHorizon = None
        self.userCycleOffset = None
        self.productName = None
        self.fileType = None
        self.file_ext = None
        self.nx_global = None
        self.ny_global = None
        self.nx_local = None
        self.ny_local = None
        self.x_lower_bound = None
        self.x_upper_bound = None
        self.y_lower_bound = None
        self.y_upper_bound = None
        self.cycleFreq = None
        self.outFreq = None
        self.regridOpt = None
        self.timeInterpOpt = None
        self.t2dDownscaleOpt = None
        self.lapseGrid = None
        self.rqiClimoGrid = None
        self.swDowscaleOpt = None
        self.precipDownscaleOpt = None
        self.nwmPRISM_numGrid = None
        self.nwmPRISM_denGrid = None
        self.q2dDownscaleOpt = None
        self.psfcDownscaleOpt = None
        self.t2dBiasCorrectOpt = None
        self.swBiasCorrectOpt = None
        self.precipBiasCorrectOpt = None
        self.q2dBiasCorrectOpt = None
        self.windBiasCorrectOpt = None
        self.psfcBiasCorrectOpt = None
        self.lwBiasCorrectOpt = None
        self.esmf_lats = None
        self.esmf_lons = None
        self.esmf_grid_in = None
        self.regridComplete = False
        self.regridObj = None
        self.esmf_field_in = None
        self.esmf_field_out = None
        # --------------------------------
        # Only used for CFSv2 bias correction
        # as bias correction needs to take
        # place prior to regridding.
        self.coarse_input_forcings1 = None
        self.coarse_input_forcings2 = None
        # --------------------------------
        self.regridded_forcings1 = None
        self.regridded_forcings2 = None
        self.globalPcpRate1 = None
        self.globalPcpRate2 = None
        self.regridded_mask = None
        self.final_forcings = None
        self.ndv = None
        self.file_in1 = None
        self.file_in2 = None
        self.fcst_hour1 = None
        self.fcst_hour2 = None
        self.fcst_date1 = None
        self.fcst_date2 = None
        self.height = None
        self.netcdf_var_names = None
        self.grib_mes_idx = None
        self.input_map_output = None
        self.grib_levels = None
        self.grib_vars = None
        self.tmpFile = None
        self.tmpFileHeight = None
        self.psfcTmp = None
        self.t2dTmp = None
        self.rstFlag = 0
        self.regridded_precip1 = None
        self.regridded_precip2 = None
        self.border = None

    def define_product(self):
        """
        Function to define the product name based on the mapping
        forcing key value.
        :return:
        """
        GRIB1 = self.GRIB1
        GRIB2 = self.GRIB2
        NETCDF = self.NETCDF

        product_names = {
            1: "NLDAS2_GRIB1",
            2: "NARR_GRIB1",
            3: "GFS_Production_GRIB2",
            4: "NAM_Conus_Nest_GRIB2",
            5: "HRRR_Conus_GRIB2",
            6: "RAP_Conus_GRIB2",
            7: "CFSv2_6Hr_Global_GRIB2",
            8: "WRF_ARW_Hawaii_GRIB2",
            9: "GFS_Production_025d_GRIB2",
            10: "Custom_NetCDF_Hourly",
            11: "Custom_NetCDF_Hourly",
            12: "AORC",
            13: "NAM_Nest_3km_Hawaii",
            14: "NAM_Nest_3km_PuertoRico",
            15: "NAM_Nest_3km_Alaska",
            16: "NAM_Nest_3km_Hawaii_Radiation-Only",
            17: "NAM_Nest_3km_PuertoRico_Radiation-Only",
            18: "WRF_ARW_PuertoRico_GRIB2",
            19: "HRRR_Alaska_GRIB2",
            20: "Alaska_ExtAnA"
        }
        self.productName = product_names[self.keyValue]

        ## DEFINED BY CONFIG
        # product_types = {
        #     1: GRIB1,
        #     2: GRIB1,
        #     3: GRIB2,
        #     4: GRIB2,
        #     5: GRIB2,
        #     6: GRIB2,
        #     7: GRIB2,
        #     8: GRIB2,
        #     9: GRIB2,
        #     10: NETCDF,
        #     11: NETCDF,
        #     12: NETCDF,
        #     13: GRIB2,
        #     14: GRIB2,
        #     15: GRIB2,
        #     16: GRIB2,
        #     17: GRIB2,
        #     18: GRIB2,
        #     19: GRIB2,
        #     20: NETCDF
        # }
        # self.fileType = product_types[self.keyValue]
        if self.fileType == 'GRIB1':
            self.file_ext = '.grb'
        elif self.fileType == 'GRIB2':
            self.file_ext = '.grib2'
        elif self.fileType == 'NETCDF':
            self.file_ext = '.nc'

        cycle_freq_minutes = {
            1: 60,
            2: 180,
            3: 360,
            4: 360,
            5: 60,
            6: 60,
            7: 360,
            8: 1440,
            9: 360,
            10: -9999,
            11: -9999,
            12: -9999,
            13: 360,
            14: 360,
            15: 360,
            16: 360,
            17: 360,
            18: 1440,
            19: 180,
            20: 60
        }
        self.cycleFreq = cycle_freq_minutes[self.keyValue]

        grib_vars_in = {
            1: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'PRATE', 'DSWRF',
                'DLWRF', 'PRES'],
            2: None,
            3: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'PRATE', 'DSWRF',
                'DLWRF', 'PRES'],
            4: None,
            5: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'APCP', 'DSWRF',
                'DLWRF', 'PRES'],
            6: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'APCP', 'DSWRF',
                'DLWRF', 'PRES'],
            7: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'PRATE', 'DSWRF',
                'DLWRF', 'PRES'],
            8: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'APCP', 'PRES'],
            9: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'PRATE', 'DSWRF',
                'DLWRF', 'PRES'],
            10: None,
            11: None,
            12: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'APCP',
                 'DSWRF', 'DLWRF', 'PRES'],
            13: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'PRATE', 'DSWRF',
                 'DLWRF', 'PRES'],
            14: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'PRATE', 'DSWRF',
                 'DLWRF', 'PRES'],
            15: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'PRATE', 'DSWRF',
                 'DLWRF', 'PRES'],
            16: ['DSWRF', 'DLWRF'],
            17: ['DSWRF', 'DLWRF'],
            18: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'APCP', 'PRES'],
            19: ['TMP', 'SPFH', 'UGRD', 'VGRD', 'APCP', 'DSWRF',
                'DLWRF', 'PRES'],
            20: ['RAINRATE']
        }
        self.grib_vars = grib_vars_in[self.keyValue]

        grib_levels_in = {
            1: ['2 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface', 'surface', 'surface'],
            2: None,
            3: ['2 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface', 'surface', 'surface'],
            4: None,
            5: ['2 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface', 'surface', 'surface'],
            6: ['2 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface', 'surface', 'surface'],
            7: ['2 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface', 'surface', 'surface'],
            8: ['80 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface'],
            9: ['2 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface', 'surface', 'surface'],
            10: None,
            11: None,
            12: None,
            13: ['2 m above ground', '2 m above ground',
                 '10 m above ground', '10 m above ground',
                 'surface', 'surface', 'surface', 'surface'],
            14: ['2 m above ground', '2 m above ground',
                 '10 m above ground', '10 m above ground',
                 'surface', 'surface', 'surface', 'surface'],
            15: ['2 m above ground', '2 m above ground',
                 '10 m above ground', '10 m above ground',
                 'surface', 'surface', 'surface', 'surface'],
            16: ['surface', 'surface'],
            17: ['surface', 'surface'],
            18: ['80 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface'],
            19: ['2 m above ground', '2 m above ground',
                '10 m above ground', '10 m above ground',
                'surface', 'surface', 'surface', 'surface'],
            20: None
        }
        self.grib_levels = grib_levels_in[self.keyValue]

        netcdf_variables = {
            1: ['TMP_2maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'APCP_surface', 'DSWRF_surface', 'DLWRF_surface',
                'PRES_surface'],
            2: None,
            3: ['TMP_2maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'PRATE_surface', 'DSWRF_surface', 'DLWRF_surface',
                'PRES_surface'],
            4: None,
            5: ['TMP_2maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'APCP_surface', 'DSWRF_surface', 'DLWRF_surface',
                'PRES_surface'],
            6: ['TMP_2maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'APCP_surface', 'DSWRF_surface', 'DLWRF_surface',
                'PRES_surface'],
            7: ['TMP_2maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'PRATE_surface', 'DSWRF_surface', 'DLWRF_surface',
                'PRES_surface'],
            8: ['TMP_80maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'APCP_surface', 'PRES_surface'],
            9: ['TMP_2maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'PRATE_surface', 'DSWRF_surface', 'DLWRF_surface',
                'PRES_surface'],
            10: ['T2D', 'Q2D', 'U10', 'V10', 'RAINRATE', 'DSWRF',
                 'DLWRF', 'PRES'],
            11: ['T2D', 'Q2D', 'U10', 'V10', 'RAINRATE', 'DSWRF',
                 'DLWRF', 'PRES'],
            12: ['TMP_2maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'APCP_surface', 'DSWRF_surface', 'DLWRF_surface',
                'PRES_surface'],
            13: ['TMP_2maboveground', 'SPFH_2maboveground',
                 'UGRD_10maboveground', 'VGRD_10maboveground',
                 'PRATE_surface', 'DSWRF_surface', 'DLWRF_surface',
                 'PRES_surface'],
            14: ['TMP_2maboveground', 'SPFH_2maboveground',
                 'UGRD_10maboveground', 'VGRD_10maboveground',
                 'PRATE_surface', 'DSWRF_surface', 'DLWRF_surface',
                 'PRES_surface'],
            15: ['TMP_2maboveground', 'SPFH_2maboveground',
                 'UGRD_10maboveground', 'VGRD_10maboveground',
                 'PRATE_surface', 'DSWRF_surface', 'DLWRF_surface',
                 'PRES_surface'],
            16: ['DSWRF_surface', 'DLWRF_surface'],
            17: ['DSWRF_surface', 'DLWRF_surface'],
            18: ['TMP_80maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'APCP_surface', 'PRES_surface'],
            19: ['TMP_2maboveground', 'SPFH_2maboveground',
                'UGRD_10maboveground', 'VGRD_10maboveground',
                'APCP_surface', 'DSWRF_surface', 'DLWRF_surface',
                'PRES_surface'],
            20: ['RAINRATE']
        }
        self.netcdf_var_names = netcdf_variables[self.keyValue]

        # arrays that store the message ids of required forcing variables for each forcing type
        # TODO fill these arrays for forcing types other than GFS
        grib_message_idx = {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: [33,34,39,40,43,88,91,6],
            10: None,
            11: None,
            12: None,
            13: None,
            14: None,
            15: None,
            16: None,
            17: None,
            18: None,
            19: None,
            20: None
        }
        self.grib_mes_idx = grib_message_idx[self.keyValue] 

        input_map_to_outputs = {
            1: [4,5,0,1,3,7,2,6],
            2: None,
            3: [4,5,0,1,3,7,2,6],
            4: None,
            5: [4,5,0,1,3,7,2,6],
            6: [4,5,0,1,3,7,2,6],
            7: [4,5,0,1,3,7,2,6],
            8: [4,5,0,1,3,6],
            9: [4,5,0,1,3,7,2,6],
            10: [4,5,0,1,3,7,2,6],
            11: [4,5,0,1,3,7,2,6],
            12: [4,5,0,1,3,7,2,6],
            13: [4,5,0,1,3,7,2,6],
            14: [4,5,0,1,3,7,2,6],
            15: [4,5,0,1,3,7,2,6],
            16: [7,2],
            17: [7, 2],
            18: [4, 5, 0, 1, 3, 6],
            19: [4,5,0,1,3,7,2,6],
            20: [3]
        }
        self.input_map_output = input_map_to_outputs[self.keyValue]

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
        find_neighbor_files = {
            1: time_handling.find_nldas_neighbors,
            3: time_handling.find_gfs_neighbors,
            5: time_handling.find_conus_hrrr_neighbors,
            6: time_handling.find_conus_rap_neighbors,
            7: time_handling.find_cfsv2_neighbors,
            8: time_handling.find_hourly_wrf_arw_neighbors,
            9: time_handling.find_gfs_neighbors,
            10: time_handling.find_custom_hourly_neighbors,
            11: time_handling.find_custom_hourly_neighbors,
            12: time_handling.find_aorc_neighbors,
            13: time_handling.find_nam_nest_neighbors,
            14: time_handling.find_nam_nest_neighbors,
            15: time_handling.find_nam_nest_neighbors,
            16: time_handling.find_nam_nest_neighbors,
            17: time_handling.find_nam_nest_neighbors,
            18: time_handling.find_hourly_wrf_arw_neighbors,
            19: time_handling.find_ak_hrrr_neighbors,
            20: time_handling.find_ak_ext_ana_neighbors,
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
        regrid_inputs = {
            1: regrid.regrid_conus_rap,
            3: regrid.regrid_gfs,
            5: regrid.regrid_conus_hrrr,
            6: regrid.regrid_conus_rap,
            7: regrid.regrid_cfsv2,
            8: regrid.regrid_hourly_wrf_arw,
            9: regrid.regrid_gfs,
            10: regrid.regrid_custom_hourly_netcdf,
            11: regrid.regrid_custom_hourly_netcdf,
            12: regrid.regrid_custom_hourly_netcdf,
            13: regrid.regrid_nam_nest,
            14: regrid.regrid_nam_nest,
            15: regrid.regrid_nam_nest,
            16: regrid.regrid_nam_nest,
            17: regrid.regrid_nam_nest,
            18: regrid.regrid_hourly_wrf_arw,
            19: regrid.regrid_conus_hrrr,
            20: regrid.regrid_ak_ext_ana
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
        temporal_interpolate_inputs = {
            0: timeInterpMod.no_interpolation,
            1: timeInterpMod.nearest_neighbor,
            2: timeInterpMod.weighted_average
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
        InputDict[force_key].define_product()
        InputDict[force_key].userFcstHorizon = ConfigOptions.fcst_input_horizons[force_tmp]
        InputDict[force_key].userCycleOffset = ConfigOptions.fcst_input_offsets[force_tmp]

        InputDict[force_key].border = ConfigOptions.ignored_border_widths[force_tmp]

        # If we have specified specific humidity downscaling, establish arrays to hold
        # temporary temperature arrays that are un-downscaled.
        if InputDict[force_key].q2dDownscaleOpt > 0:
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
        InputDict[force_key].final_forcings = np.empty([8,GeoMetaWrfHydro.ny_local,
                                                        GeoMetaWrfHydro.nx_local],
                                                       np.float64)
        InputDict[force_key].height = np.empty([GeoMetaWrfHydro.ny_local,
                                                GeoMetaWrfHydro.nx_local],np.float32)
        InputDict[force_key].regridded_mask = np.empty([GeoMetaWrfHydro.ny_local,
                                                GeoMetaWrfHydro.nx_local],np.float32)

        # Obtain custom input cycle frequencies
        if force_key == 10 or force_key == 11:
            InputDict[force_key].cycleFreq = ConfigOptions.customFcstFreq[custom_count]
            custom_count = custom_count + 1

    return InputDict
