from core import errMod
from netCDF4 import Dataset
import ESMF
import numpy as np
import sys
import math

class GeoMetaWrfHydro:
    """
    Abstract class for handling information about the WRF-Hydro domain
    we are processing forcings too.
    """
    def __init__(self):
        self.nx_global = None
        self.ny_global = None
        self.dx_meters = None
        self.dy_meters = None
        self.nx_local = None
        self.ny_local = None
        self.x_lower_bound = None
        self.x_upper_bound = None
        self.y_lower_bound = None
        self.y_upper_bound = None
        self.latitude_grid = None
        self.longitude_grid = None
        self.sina_grid = None
        self.cosa_grid = None
        self.slope = None
        self.slp_azi = None
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

        #print("WRF-HYDRO LOCAL X BOUND 1 = " + str(self.x_lower_bound))
        #print("WRF-HYDRO LOCAL X BOUND 2 = " + str(self.x_upper_bound))
        #print("WRF-HYDRO LOCAL Y BOUND 1 = " + str(self.y_lower_bound))
        #print("WRF-HYDRO LOCAL Y BOUND 2 = " + str(self.y_upper_bound))

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
                ConfigOptions.errMsg = "Unable to open the WRF-Hydro " + \
                                       "geogrid file: " + ConfigOptions.geogrid
                raise

            try:
                self.nx_global = idTmp.variables['XLAT_M'].shape[2]
            except:
                ConfigOptions.errMsg = "Unable to extract X dimension size " + \
                                       "from XLAT_M in: " + ConfigOptions.geogrid
                raise

            try:
                self.ny_global = idTmp.variables['XLAT_M'].shape[1]
            except:
                ConfigOptions.errMsg = "Unable to extract Y dimension size " + \
                                       "from XLAT_M in: " + ConfigOptions.geogrid
                raise

            try:
                self.dx_meters = idTmp.DX
            except:
                ConfigOptions.errMsg = "Unable to extract DX global attribute " + \
                                       " in: " + ConfigOptions.geogrid
                raise

            try:
                self.dy_meters = idTmp.DY
            except:
                ConfigOptions.errMsg = "Unable to extract DY global attribute " + \
                                       " in: " + ConfigOptions.geogrid
                raise

        MpiConfig.comm.barrier()

        # Broadcast global dimensions to the other processors.
        self.nx_global = MpiConfig.broadcast_parameter(self.nx_global,ConfigOptions)
        self.ny_global = MpiConfig.broadcast_parameter(self.ny_global,ConfigOptions)
        self.dx_meters = MpiConfig.broadcast_parameter(self.dx_meters,ConfigOptions)
        self.dy_meters = MpiConfig.broadcast_parameter(self.dy_meters,ConfigOptions)

        MpiConfig.comm.barrier()

        try:
            self.esmf_grid = ESMF.Grid(np.array([self.ny_global,self.nx_global]),
                                       staggerloc=ESMF.StaggerLoc.CENTER,
                                       coord_sys=ESMF.CoordSys.SPH_DEG)
        except:
            ConfigOptions.errMsg = "Unable to create ESMF grid for WRF-Hydro " \
                                   "geogrid: " + ConfigOptions.geogrid
            raise

        MpiConfig.comm.barrier()

        self.esmf_lat = self.esmf_grid.get_coords(0)
        self.esmf_lon = self.esmf_grid.get_coords(1)

        MpiConfig.comm.barrier()

        # Obtain the local boundaries for this processor.
        self.get_processor_bounds()

        # Scatter global XLAT_M grid to processors..
        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['XLAT_M'][0,:,:]
        else:
            varTmp = None

        MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self,varTmp,ConfigOptions)

        MpiConfig.comm.barrier()

        # Place the local lat/lon grid slices from the parent geogrid file into
        # the ESMF lat/lon grids.
        try:
            self.esmf_lat[:,:] = varSubTmp
            varSubTmp = None
            varTmp = None
        except:
            ConfigOptions.errMsg = "Unable to subset XLAT_M from geogrid file into ESMF object"
            raise

        MpiConfig.comm.barrier()

        # Scatter global XLONG_M grid to processors..
        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['XLONG_M'][0, :, :]
        else:
            varTmp = None

        MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self,varTmp,ConfigOptions)

        MpiConfig.comm.barrier()

        try:
            self.esmf_lon[:,:] = varSubTmp
            varSubTmp = None
            varTmp = None
        except:
            ConfigOptions.errMsg = "Unable to subset XLONG_M from geogrid file into ESMF object"
            raise

        MpiConfig.comm.barrier()

        # Scatter the COSALPHA,SINALPHA grids to the processors.
        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['COSALPHA'][0,:,:]
        else:
            varTmp = None
        MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self,varTmp,ConfigOptions)
        MpiConfig.comm.barrier()
        self.cosa_grid[:,:] = varSubTmp[:,:]
        varSubTmp = None
        varTmp = None

        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['SINALPHA'][0, :, :]
        else:
            varTmp = None
        MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self, varTmp, ConfigOptions)
        MpiConfig.comm.barrier()
        self.sina_grid[:, :] = varSubTmp[:, :]
        varSubTmp = None
        varTmp = None

        # Calculate the slope from the domain using elevation on the WRF-Hydro domain. This will
        # be used for downscaling purposes.
        if MpiConfig.rank == 0:
            try:
                slopeTmp, slp_azi_tmp = self.calc_slope(idTmp,ConfigOptions)
            except:
                raise
        else:
            slopeTmp = None
            slp_azi_tmp = None
        MpiConfig.comm.barrier()

        slopeSubTmp = MpiConfig.scatter_array(self,slopeTmp,ConfigOptions)
        self.slope[:,:] = slopeSubTmp[:,:]
        slopeSubTmp = None
        MpiConfig.comm.barrier()

        slp_azi_sub = MpiConfig.scatter_array(self,slp_azi_tmp,ConfigOptions)
        self.slp_azi[:,:] = slp_azi_sub[:,:]
        slp_azi_tmp = None
        MpiConfig.comm.barrier()

        if MpiConfig.rank == 0:
            # Close the geogrid file
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close geogrid file: " + ConfigOptions.geogrid
                raise


    def calc_slope(self,idTmp,ConfigOptions):
        """
        Function to calculate slope grids needed for incoming shortwave radiation downscaling
        later during the program.
        :param idTmp:
        :param ConfigOptions:
        :return:
        """
        # First extract the sina,cosa, and elevation variables from the geogrid file.
        try:
            sinaGrid = idTmp.variables['SINALPHA'][0,:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract SINALPHA from: " + ConfigOptions.geogrid
            raise

        try:
            cosaGrid = idTmp.variables['COSALPH'][0,:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract COSALPHA from: " + ConfigOptions.geogrid
            raise

        try:
            heightDest = idTmp.variables['HGT_M'][0,:,:]
        except:
            ConfigOptions.errMsg = "Unable to extract HGT_M from: " + ConfigOptions.geogrid
            raise

        # Ensure cosa/sina are correct dimensions
        if sinaGrid.shape[0] != self.ny_global or sinaGrid.shape[1] != self.nx_global:
            ConfigOptions.errMsg = "SINALPHA dimensions mismatch in: " + ConfigOptions.geogrid
            raise
        if cosaGrid.shape[0] != self.ny_global or cosaGrid.shape[1] != self.nx_global:
            ConfigOptions.errMsg = "COSALPHA dimensions mismatch in: " + ConfigOptions.geogrid
            raise
        if heightDest.shape[0] != self.ny_global or heightDest.shape[1] != self.nx_global:
            ConfigOptions.errMsg = "HGT_M dimension mismatch in: " + ConfigOptions.geogrid
            raise

        # Establish constants
        rdx = 1.0/self.dx_meters
        rdy = 1.0/self.dy_meters
        msftx = 1.0
        msfty = 1.0

        slopeOut = np.empty([self.ny_global,self.nx_global],np.float32)
        toposlpx = np.empty([self.ny_global,self.nx_global],np.float32)
        toposlpy = np.empty([self.ny_global,self.nx_global],np.float32)
        slp_azi = np.empty([ny, nx], np.float32)

        for j in range(0,self.ny_global):
            for i in range(0,self.nx_global):
                im1 = i - 1
                ip1 = i + 1
                jm1 = j - 1
                jp1 = j + 1
                if im1 < 0:
                    im1 = 0
                if jm1 < 0:
                    jm1 = 0
                if ip1 >= self.nx_global:
                    ip1 = self.nx_global - 1
                if jp1 >= self.ny_global:
                    jp1 = self.ny_global - 1
                toposlpx[j, i] = ((heightDest[j, ip1] - heightDest[j, im1]) * msftx * rdx) / (ip1 - im1)
                toposlpy[j, i] = ((heightDest[jp1, i] - heightDest[jm1, i]) * msfty * rdy) / (jp1 - jm1)
                hx = toposlpx[j, i]
                hy = toposlpy[j, i]
                slopeOut[j, i] = math.atan((hx ** 2 + hy ** 2) ** 0.5)
                if slopeOut[j, i] < 1E-4:
                    slopeOut[j,i] = 0.0
                    slp_azi[j,i] = 0.0
                else:
                    slp_azi[j,i] = math.atan2(hx,hy) + math.pi
                if cosaGrid[j,i] >= 0.0:
                    slp_azi[j, i] = slp_azi[j, i] - math.asin(sinaGrid[j, i])
                    slp_azi[j, i] = slp_azi[j, i] - (math.pi - math.asin(sinaGrid[j, i]))

        # Reset temporary arrays to None to free up memory
        toposlpx = None
        toposlpy = None
        heightDest = None
        sinaGrid = None
        cosaGrid = None

        return slopeOut,slp_azi


