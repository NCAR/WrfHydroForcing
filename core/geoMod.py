from core import errMod
from netCDF4 import Dataset
import ESMF
import numpy as np

class GeoMetaWrfHydro:
    """
    Abstract class for handling information about the WRF-Hydro domain
    we are processing forcings too.
    """
    def __init__(self):
        self.nx_global = None
        self.ny_global = None
        self.nx_local = None
        self.ny_local = None
        self.x_lower_bound = None
        self.x_upper_bound = None
        self.y_lower_bound = None
        self.y_upper_bound = None
        self.latitude_grid = None
        self.longitude_grid = None
        self.esmf_grid = None
        self.esmf_lat = None
        self.esmf_lon = None

    def get_processor_bounds(self):
        """
        Calculate the local grid boundaries for this processor.
        ESMF operates under the hood and the boundary values
        are calculated within the ESMF software.
        :return:
        """
        self.x_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        self.x_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        self.y_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        self.y_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        self.nx_local = self.x_upper_bound - self.x_lower_bound
        self.ny_local = self.y_upper_bound - self.y_lower_bound

        print("WRF-HYDRO LOCAL X BOUND 1 = " + str(self.x_lower_bound))
        print("WRF-HYDRO LOCAL X BOUND 2 = " + str(self.x_upper_bound))
        print("WRF-HYDRO LOCAL Y BOUND 1 = " + str(self.y_lower_bound))
        print("WRF-HYDRO LOCAL Y BOUND 2 = " + str(self.y_upper_bound))

    def initialize_destination_geo(self,ConfigOptions,MpiConfig):
        """
        Initialization function to initialize ESMF through ESMPy,
        calculate the global parameters of the WRF-Hydro grid
        being processed to, along with the local parameters
        for this particular processor.
        :return:
        """
        # Open the geogrid file and extract necessary information
        # to create ESMF fields.
        if MpiConfig.rank == 0:
            try:
                idTmp = Dataset(ConfigOptions.geogrid,'r')
            except:
                ConfigOptions.errMsg = "Unable to open the WRF-Hydro " \
                                       "geogrid file: " + ConfigOptions.geogrid
                raise

            try:
                self.nx_global = idTmp.variables['XLAT_M'].shape[2]
            except:
                ConfigOptions.errMsg = "Unable to extract X dimension size " \
                                       "from XLAT_M in: " + ConfigOptions.geogrid
                raise

            try:
                self.ny_global = idTmp.variables['XLAT_M'].shape[1]
            except:
                ConfigOptions.errMsg = "Unable to extract Y dimension size " \
                                       "from XLAT_M in: " + ConfigOptions.geogrid
                raise

        MpiConfig.comm.barrier()

        print("PROC ID: " + MpiConfig.rank + " GLOBAL BEFORE BROADCAST = " + str(self.nx_global))
        # Broadcast global dimensions to the other processors.
        MpiConfig.comm.Bcast(self.nx_global,root=0)
        print("PROC ID: " + MpiConfig.rank + " GLOBAL AFTER BROADCAST = " + str(self.nx_global))

        try:
            self.esmf_grid = ESMF.Grid(np.array([self.ny_global,self.nx_global]),
                                       staggerloc=ESMF.StaggerLoc.CENTER,
                                       coord_sys=ESMF.CoordSys.SPH_DEG)
        except:
            ConfigOptions.errMsg = "Unable to create ESMF grid for WRF-Hydro " \
                                   "geogrid: " + ConfigOptions.geogrid
            raise

        self.esmf_lat = self.esmf_grid.get_coords(0)
        self.esmf_lon = self.esmf_grid.get_coords(1)

        # Obtain the local boundaries for this processor.
        self.get_processor_bounds()

        # Place the local lat/lon grid slices from the parent geogrid file into
        # the ESMF lat/lon grids.
        try:
            self.esmf_lat[:,:] = idTmp.variables['XLAT_M'][0,
                                 self.y_lower_bound:self.y_upper_bound,
                                 self.x_lower_bound:self.x_upper_bound]
        except:
            ConfigOptions.errMsg = "Unable to subset XLAT_M from geogrid file into ESMF object"
            raise
        try:
            self.esmf_lon[:,:] = idTmp.variables['XLONG_M'][0,
                                 self.y_lower_bound:self.y_upper_bound,
                                 self.x_lower_bound:self.x_upper_bound]
        except:
            ConfigOptions.errMsg = "Unable to subset XLONG_M from geogrid file into ESMF object"
            raise

        # Close the geogrid file
        try:
            idTmp.close()
        except:
            ConfigOptions.errMsg = "Unable to close geogrid file: " + ConfigOptions.geogrid
            raise