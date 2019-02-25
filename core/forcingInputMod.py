"""
This module will guide the forcing engine in defining
parameters in all input forcing products. These parameters
include things such as file types, grid definitions (including
initializing ESMF grids and regrid objects), etc
"""
#import ESMF
from core import dateMod

class input_forcings:
    """
    This is an abstract class that will define all the parameters
    of a single input forcing product.
    """
    def __init__(self):
        """
        Initializing all attributes and objects to None.
        """
        self.keyValue = None
        self.inDir = None
        self.userFcstHorizon = None
        self.userCycleOffset = None
        self.productName = None
        self.fileType = None
        self.nxGlobal = None
        self.nyGlobal = None
        #self.cycleDate1 = None
        #self.cycleDate2 = None
        self.cycleFreq = None
        self.esmf_lats = None
        self.esmf_lons = None
        self.esmf_grid_in = None
        self.regridComplete = False
        self.esmf_field_in1 = None
        self.esmf_field_in2 = None
        self.prate_field_out = None
        self.u10m_field_out = None
        self.v10m_field_out = None
        self.psfc_field_out = None
        self.sw_field_out = None
        self.lw_field_out = None
        self.t2m_field_out = None
        self.q2m_field_out = None
        self.prate_field_in1 = None
        self.prate_field_in2 = None
        self.u10m_field_in1 = None
        self.u10m_field_in2 = None
        self.v10m_field_in1 = None
        self.v10m_field_in2 = None
        self.psfc_field_in1 = None
        self.psfc_field_in2 = None
        self.sw_field_in1 = None
        self.sw_field_in2 = None
        self.lw_field_in1 = None
        self.lw_field_in2 = None
        self.t2m_field_in1 = None
        self.t2m_field_in2 = None
        self.q2m_field_in1 = None
        self.q2m_field_in2 = None
        self.ndv = None
        self.file_in1 = None
        self.file_in2 = None
        self.tmpFile1 = None
        self.tmpFile2 = None

    #def read_file(self):
    #    """
    #    Function for opening an input file and reading in
    #    data to a grid.
    #    :return:
    #    """

    def define_product(self):
        """
        Function to define the product name based on the mapping
        forcing key value.
        :return:
        """
        product_names = {
            1: "NLDAS2_GRIB1",
            2: "NARR_GRIB1",
            3: "GFS_Production_GRIB2",
            4: "NAM_Conus_Nest_GRIB2",
            5: "HRRR_Conus_GRIB2",
            6: "RAP_Conus_GRIB2",
            7: "CFSv2_GRIB2",
            8: "WRF_ARW_Hawaii_GRIB2",
            9: "GFS_Production_025d_GRIB2",
            10: "Custom_NetCDF"
        }
        self.productName = product_names[self.keyValue]

        product_types = {
            1: "GRIB1",
            2: "GRIB1",
            3: "GRIB2",
            4: "GRIB2",
            5: "GRIB2",
            6: "GRIB2",
            7: "GRIB2",
            8: "GRIB2",
            9: "GRIB2",
            10: "NetCDF"
        }
        self.fileType = product_types[self.keyValue]

        cycle_freq_minutes = {
            1: 60,
            2: 180,
            3: 360,
            4: 360,
            5: 60,
            6: 60,
            7: 360,
            8: 1440,
            10: 360,
            9: 60
        }
        self.cycleFreq = cycle_freq_minutes[self.keyValue]

    def calc_neighbor_files(self,ConfigOptions,dCurrent):
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
            3: dateMod.find_gfs_neighbors,
            9: dateMod.find_gfs_neighbors
        }

        find_neighbor_files[self.keyValue](self, ConfigOptions, dCurrent)
        #try:
        #    find_neighbor_files[self.keyValue](self,ConfigOptions,dCurrent)
        #except TypeError:
        #    ConfigOptions.errMsg = "Unable to execute find_neighbor_files for " \
        #                           "input forcing: " + self.productName
        #    raise
        #except:
        #    raise

    def regrid_inputs(self,ConfigOptions):
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
        #regrid_inputs = {
        #    3: regridMod.regrid_gfs
        #}

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
    for force_tmp in range(0,ConfigOptions.number_inputs):
        force_key = ConfigOptions.input_forcings[force_tmp]
        InputDict[force_key] = input_forcings()
        InputDict[force_key].keyValue = force_key
        InputDict[force_key].inDir = ConfigOptions.input_force_dirs[force_tmp]
        InputDict[force_key].define_product()
        InputDict[force_key].userFcstHorizon = ConfigOptions.fcst_input_horizons[force_tmp]
        InputDict[force_key].userCycleOffset = ConfigOptions.fcst_input_offsets[force_tmp]
        #InputDict[force_key].t2m_field_out = ESMF.Field(GeoMetaWrfHydro.esmf_grid,
        #                                                name='T2D')
        #InputDict[force_key].q2m_field_out = ESMF.Field(GeoMetaWrfHydro.esmf_grid,
        #                                                name='Q2D')
        #InputDict[force_key].u10m_field_out = ESMF.Field(GeoMetaWrfHydro.esmf_grid,
        #                                                 name='U10')
        #InputDict[force_key].v10m_field_out = ESMF.Field(GeoMetaWrfHydro.esmf_grid,
        #                                                 name='V10')
        #InputDict[force_key].psfc_field_out = ESMF.Field(GeoMetaWrfHydro.esmf_grid,
        #                                                 name='PSFC')
        #InputDict[force_key].prate_field_out = ESMF.Field(GeoMetaWrfHydro.esmf_grid,
        #                                                  name = 'RAINRATE')
        #InputDict[force_key].sw_field_out = ESMF.Field(GeoMetaWrfHydro.esmf_grid,
        #                                               name='SWDOWN')
        #InputDict[force_key].lw_field_out = ESMF.Field(GeoMetaWrfHydro.esmf_grid,
        #                                               name='LWDOWN')

    return InputDict