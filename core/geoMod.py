import math

from netCDF4 import Dataset
import numpy as np

try:
    import ESMF
except ImportError:
    import esmpy as ESMF

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
        self.height = None
        self.sina_grid = None
        self.cosa_grid = None
        self.slope = None
        self.slp_azi = None
        self.esmf_grid = None
        self.esmf_lat = None
        self.esmf_lon = None
        self.crs_atts = None
        self.x_coord_atts = None
        self.x_coords = None
        self.y_coord_atts = None
        self.y_coords = None
        self.spatial_global_atts = None

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
                raise Exception

            try:
                self.nx_global = idTmp.variables['XLAT_M'].shape[2]
            except:
                ConfigOptions.errMsg = "Unable to extract X dimension size " + \
                                       "from XLAT_M in: " + ConfigOptions.geogrid
                raise Exception

            try:
                self.ny_global = idTmp.variables['XLAT_M'].shape[1]
            except:
                ConfigOptions.errMsg = "Unable to extract Y dimension size " + \
                                       "from XLAT_M in: " + ConfigOptions.geogrid
                raise Exception

            try:
                self.dx_meters = idTmp.DX
            except:
                ConfigOptions.errMsg = "Unable to extract DX global attribute " + \
                                       " in: " + ConfigOptions.geogrid
                raise Exception

            try:
                self.dy_meters = idTmp.DY
            except:
                ConfigOptions.errMsg = "Unable to extract DY global attribute " + \
                                       " in: " + ConfigOptions.geogrid
                raise Exception

        #MpiConfig.comm.barrier()

        # Broadcast global dimensions to the other processors.
        self.nx_global = MpiConfig.broadcast_parameter(self.nx_global,ConfigOptions, param_type=int)
        self.ny_global = MpiConfig.broadcast_parameter(self.ny_global,ConfigOptions, param_type=int)
        self.dx_meters = MpiConfig.broadcast_parameter(self.dx_meters,ConfigOptions, param_type=float)
        self.dy_meters = MpiConfig.broadcast_parameter(self.dy_meters,ConfigOptions, param_type=float)

        #MpiConfig.comm.barrier()

        try:
            self.esmf_grid = ESMF.Grid(np.array([self.ny_global,self.nx_global]),
                                       staggerloc=ESMF.StaggerLoc.CENTER,
                                       coord_sys=ESMF.CoordSys.SPH_DEG)
        except:
            ConfigOptions.errMsg = "Unable to create ESMF grid for WRF-Hydro " \
                                   "geogrid: " + ConfigOptions.geogrid
            raise Exception

        #MpiConfig.comm.barrier()

        self.esmf_lat = self.esmf_grid.get_coords(1)
        self.esmf_lon = self.esmf_grid.get_coords(0)

        #MpiConfig.comm.barrier()

        # Obtain the local boundaries for this processor.
        self.get_processor_bounds()

        # Scatter global XLAT_M grid to processors..
        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['XLAT_M'][0,:,:]
        else:
            varTmp = None

        #MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self,varTmp,ConfigOptions)

        #MpiConfig.comm.barrier()

        # Place the local lat/lon grid slices from the parent geogrid file into
        # the ESMF lat/lon grids.
        try:
            self.esmf_lat[:,:] = varSubTmp
            self.latitude_grid = varSubTmp
            varSubTmp = None
            varTmp = None
        except:
            ConfigOptions.errMsg = "Unable to subset XLAT_M from geogrid file into ESMF object"
            raise Exception

        #MpiConfig.comm.barrier()

        # Scatter global XLONG_M grid to processors..
        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['XLONG_M'][0, :, :]
        else:
            varTmp = None

        #MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self,varTmp,ConfigOptions)

        #MpiConfig.comm.barrier()

        try:
            self.esmf_lon[:,:] = varSubTmp
            self.longitude_grid = varSubTmp
            varSubTmp = None
            varTmp = None
        except:
            ConfigOptions.errMsg = "Unable to subset XLONG_M from geogrid file into ESMF object"
            raise Exception

        #MpiConfig.comm.barrier()

        # Scatter the COSALPHA,SINALPHA grids to the processors.
        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['COSALPHA'][0,:,:]
        else:
            varTmp = None
        #MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self,varTmp,ConfigOptions)
        #MpiConfig.comm.barrier()

        self.cosa_grid = varSubTmp[:,:]
        varSubTmp = None
        varTmp = None

        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['SINALPHA'][0, :, :]
        else:
            varTmp = None
        #MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self, varTmp, ConfigOptions)
        #MpiConfig.comm.barrier()
        self.sina_grid = varSubTmp[:, :]
        varSubTmp = None
        varTmp = None

        # Read in a scatter the WRF-Hydro elevation, which is used for downscaling
        # purposes.
        if MpiConfig.rank == 0:
            varTmp = idTmp.variables['HGT_M'][0, :, :]
        else:
            varTmp = None
        #MpiConfig.comm.barrier()

        varSubTmp = MpiConfig.scatter_array(self, varTmp, ConfigOptions)
        #MpiConfig.comm.barrier()
        self.height = varSubTmp
        varSubTmp = None
        varTmp = None

        # Calculate the slope from the domain using elevation on the WRF-Hydro domain. This will
        # be used for downscaling purposes.
        if MpiConfig.rank == 0:
            try:
                slopeTmp, slp_azi_tmp = self.calc_slope(idTmp,ConfigOptions)
            except Exception:
                raise Exception
        else:
            slopeTmp = None
            slp_azi_tmp = None
        #MpiConfig.comm.barrier()

        slopeSubTmp = MpiConfig.scatter_array(self,slopeTmp,ConfigOptions)
        self.slope = slopeSubTmp[:,:]
        slopeSubTmp = None
        #MpiConfig.comm.barrier()

        slp_azi_sub = MpiConfig.scatter_array(self,slp_azi_tmp,ConfigOptions)
        self.slp_azi = slp_azi_sub[:,:]
        slp_azi_tmp = None
        #MpiConfig.comm.barrier()

        if MpiConfig.rank == 0:
            # Close the geogrid file
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close geogrid file: " + ConfigOptions.geogrid
                raise Exception

        # Reset temporary variables to free up memory
        slopeTmp = None
        slp_azi_tmp = None
        varTmp = None

    def initialize_geospatial_metadata(self,ConfigOptions,MpiConfig):
        """
        Function that will read in crs/x/y geospatial metadata and coordinates
        from the optional geospatial metadata file IF it was specified by the user in
        the configuration file.
        :param ConfigOptions:
        :return:
        """
        # We will only read information on processor 0. This data is not necessary for the
        # other processors, and is only used in the output routines.
        if MpiConfig.rank == 0:
            # Open the geospatial metadata file.
            try:
                idTmp = Dataset(ConfigOptions.spatial_meta,'r')
            except:
                ConfigOptions.errMsg = "Unable to open spatial metadata file: " + ConfigOptions.spatial_meta
                raise Exception

            # Make sure the expected variables are present in the file.
            if 'crs' not in idTmp.variables.keys():
                ConfigOptions.errMsg = "Unable to locate crs variable in: " + ConfigOptions.spatial_meta
                raise Exception
            if 'x' not in idTmp.variables.keys():
                ConfigOptions.errMsg = "Unable to locate x variable in: " + ConfigOptions.spatial_meta
                raise Exception
            if 'y' not in idTmp.variables.keys():
                ConfigOptions.errMsg = "Unable to locate y variable in: " + ConfigOptions.spatial_meta
                raise Exception
            # Extract names of variable attributes from each of the input geospatial variables. These
            # can change, so we are making this as flexible as possible to accomodate future changes.
            try:
                crs_att_names = idTmp.variables['crs'].ncattrs()
            except:
                ConfigOptions.errMsg = "Unable to extract crs attribute names from: " + ConfigOptions.spatial_meta
                raise Exception
            try:
                x_coord_att_names = idTmp.variables['x'].ncattrs()
            except:
                ConfigOptions.errMsg = "Unable to extract x attribute names from: " + ConfigOptions.spatial_meta
                raise Exception
            try:
                y_coord_att_names = idTmp.variables['y'].ncattrs()
            except:
                ConfigOptions.errMsg = "Unable to extract y attribute names from: " + ConfigOptions.spatial_meta
                raise Exception
            # Extract attribute values
            try:
                self.x_coord_atts = {item: idTmp.variables['x'].getncattr(item) for item in x_coord_att_names}
            except:
                ConfigOptions.errMsg = "Unable to extract x coordinate attributes from: " + ConfigOptions.spatial_meta
                raise Exception
            try:
                self.y_coord_atts = {item: idTmp.variables['y'].getncattr(item) for item in y_coord_att_names}
            except:
                ConfigOptions.errMsg = "Unable to extract y coordinate attributes from: " + ConfigOptions.spatial_meta
                raise Exception
            try:
                self.crs_atts = {item: idTmp.variables['crs'].getncattr(item) for item in crs_att_names}
            except:
                ConfigOptions.errMsg = "Unable to extract crs coordinate attributes from: " + ConfigOptions.spatial_meta
                raise Exception

            # Extract global attributes
            try:
                global_att_names = idTmp.ncattrs()
            except:
                ConfigOptions.errMsg = "Unable to extract global attribute names from: " + ConfigOptions.spatial_meta
                raise Exception
            try:
                self.spatial_global_atts = {item: idTmp.getncattr(item) for item in global_att_names}
            except:
                ConfigOptions.errMsg = "Unable to extract global attributes from: " + ConfigOptions.spatial_meta
                raise Exception

            # Extract x/y coordinate values
            if len(idTmp.variables['x'].shape) == 1:
                try:
                    self.x_coords = idTmp.variables['x'][:].data
                except:
                    ConfigOptions.errMsg = "Unable to extract x coordinate values from: " + ConfigOptions.spatial_meta
                    raise Exception
                try:
                    self.y_coords = idTmp.variables['y'][:].data
                except:
                    ConfigOptions.errMsg = "Unable to extract y coordinate values from: " + ConfigOptions.spatial_meta
                    raise Exception
                # Check to see if the Y coordinates are North-South. If so, flip them.
                if self.y_coords[1] < self.y_coords[0]:
                    self.y_coords[:] = np.flip(self.y_coords[:], axis=0)

            if len(idTmp.variables['x'].shape) == 2:
                try:
                    self.x_coords = idTmp.variables['x'][:,:].data
                except:
                    ConfigOptions.errMsg = "Unable to extract x coordinate values from: " + ConfigOptions.spatial_meta
                    raise Exception
                try:
                    self.y_coords = idTmp.variables['y'][:,:].data
                except:
                    ConfigOptions.errMsg = "Unable to extract y coordinate values from: " + ConfigOptions.spatial_meta
                    raise Exception
                # Check to see if the Y coordinates are North-South. If so, flip them.
                if self.y_coords[1,0] > self.y_coords[0,0]:
                    self.y_coords[:,:] = np.flipud(self.y_coords[:,:])


            # Close the geospatial metadata file.
            try:
                idTmp.close()
            except:
                ConfigOptions.errMsg = "Unable to close spatial metadata file: " + ConfigOptions.spatial_meta
                raise Exception

        #MpiConfig.comm.barrier()

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
            cosaGrid = idTmp.variables['COSALPHA'][0,:,:]
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
            raise Exception
        if cosaGrid.shape[0] != self.ny_global or cosaGrid.shape[1] != self.nx_global:
            ConfigOptions.errMsg = "COSALPHA dimensions mismatch in: " + ConfigOptions.geogrid
            raise Exception
        if heightDest.shape[0] != self.ny_global or heightDest.shape[1] != self.nx_global:
            ConfigOptions.errMsg = "HGT_M dimension mismatch in: " + ConfigOptions.geogrid
            raise Exception

        # Establish constants
        rdx = 1.0/self.dx_meters
        rdy = 1.0/self.dy_meters
        msftx = 1.0
        msfty = 1.0

        slopeOut = np.empty([self.ny_global,self.nx_global],np.float32)
        toposlpx = np.empty([self.ny_global,self.nx_global],np.float32)
        toposlpy = np.empty([self.ny_global,self.nx_global],np.float32)
        slp_azi = np.empty([self.ny_global, self.nx_global], np.float32)
        ipDiff = np.empty([self.ny_global, self.nx_global], np.int32)
        jpDiff = np.empty([self.ny_global, self.nx_global], np.int32)
        hx = np.empty([self.ny_global,self.nx_global],np.float32)
        hy = np.empty([self.ny_global,self.nx_global],np.float32)

        # Create index arrays that will be used to calculate slope.
        xTmp = np.arange(self.nx_global)
        yTmp = np.arange(self.ny_global)
        xGrid = np.tile(xTmp[:], (self.ny_global, 1))
        yGrid = np.repeat(yTmp[:, np.newaxis], self.nx_global, axis=1)
        indOrig = np.where(heightDest == heightDest)
        indIp1 = ((indOrig[0]), (indOrig[1] + 1))
        indIm1 = ((indOrig[0]), (indOrig[1] - 1))
        indJp1 = ((indOrig[0] + 1), (indOrig[1]))
        indJm1 = ((indOrig[0] - 1), (indOrig[1]))
        indIp1[1][np.where(indIp1[1] >= self.nx_global)] = self.nx_global - 1
        indJp1[0][np.where(indJp1[0] >= self.ny_global)] = self.ny_global - 1
        indIm1[1][np.where(indIm1[1] < 0)] = 0
        indJm1[0][np.where(indJm1[0] < 0)] = 0

        ipDiff[indOrig] = xGrid[indIp1] - xGrid[indIm1]
        jpDiff[indOrig] = yGrid[indJp1] - yGrid[indJm1]

        toposlpx[indOrig] = ((heightDest[indIp1] - heightDest[indIm1]) * msftx * rdx)/ipDiff[indOrig]
        toposlpy[indOrig] = ((heightDest[indJp1] - heightDest[indJm1]) * msfty * rdy)/jpDiff[indOrig]
        hx[indOrig] = toposlpx[indOrig]
        hy[indOrig] = toposlpy[indOrig]
        slopeOut[indOrig] = np.arctan((hx[indOrig] ** 2 + hy[indOrig] **2) ** 0.5)
        slopeOut[np.where(slopeOut < 1E-4)] = 0.0
        slp_azi[np.where(slopeOut < 1E-4)] = 0.0
        indValidTmp = np.where(slopeOut >= 1E-4)
        slp_azi[indValidTmp] = np.arctan2(hx[indValidTmp],hy[indValidTmp]) + math.pi
        indValidTmp = np.where(cosaGrid >= 0.0)
        slp_azi[indValidTmp] = slp_azi[indValidTmp] - np.arcsin(sinaGrid[indValidTmp])
        indValidTmp = np.where(cosaGrid < 0.0)
        slp_azi[indValidTmp] = slp_azi[indValidTmp] - (math.pi - np.arcsin(sinaGrid[indValidTmp]))

        # Reset temporary arrays to None to free up memory
        toposlpx = None
        toposlpy = None
        heightDest = None
        sinaGrid = None
        cosaGrid = None
        indValidTmp = None
        xTmp = None
        yTmp = None
        xGrid = None
        ipDiff = None
        jpDiff = None
        indOrig = None
        indJm1 = None
        indJp1 = None
        indIm1 = None
        indIp1 = None
        hx = None
        hy = None

        return slopeOut,slp_azi


