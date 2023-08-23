import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager

import ESMF
import numpy as np

from core import err_handler


class Regridder(ABC):
    def __init__(self, name:str, config_options, wrf_hydro_geo_meta, mpi_config):
        self.name = name
        self.config = config_options
        self.geo_meta = wrf_hydro_geo_meta
        self.mpi = mpi_config
        self.IS_MPI_ROOT = mpi_config.rank == 0

        self._next_file_number = 0

    '''
    Abstract methods overridden in subclasses:
    '''

    @abstractmethod
    def regrid_input(self, input_forcings):
        pass

    '''
    Utility methods:
    '''

    def mkfilename(self):
        self._next_file_number += 1
        return '{}'.format(self._next_file_number)

    @contextmanager
    def parallel_block(self):
        yield
        err_handler.check_program_status(self.config, self.mpi)

    def log_critical(self, msg, root_only=False):
        if not root_only or self.IS_MPI_ROOT:
            # TODO: rewrite this with cleaned-up OO err_handler
            self.config.errMsg = msg
            err_handler.log_critical(self.config, self.mpi)

    def log_status(self, msg, root_only=False):
        if not root_only or self.IS_MPI_ROOT:
            # TODO: rewrite this with cleaned-up OO err_handler
            self.config.statusMsg = msg
            err_handler.log_msg(self.config, self.mpi)

    def calculate_weights(self, dataset, forcings, forcing_idx):
        """
        Function to calculate ESMF weights based on the output ESMF
        field previously calculated, along with input lat/lon grids,
        and a sample dataset.
        :param dataset:
        :param forcings:
        :param forcing_idx:
        """
        # TODO: should this be moved into the InputForcing object?

        with self.parallel_block():
            if self.IS_MPI_ROOT:
                try:
                    forcings.ny_global = dataset.variables[forcings.netcdf_var_names[forcing_idx]].shape[1]
                except (ValueError, KeyError, AttributeError) as err:
                    self.log_critical("Unable to extract Y shape size from: " +
                                      forcings.netcdf_var_names[forcing_idx] + " from: " +
                                      forcings.tmpFile + " (" + str(err) + ")")
                try:
                    forcings.nx_global = dataset.variables[forcings.netcdf_var_names[forcing_idx]].shape[2]
                except (ValueError, KeyError, AttributeError) as err:
                    self.log_critical("Unable to extract X shape size from: " +
                                      forcings.netcdf_var_names[forcing_idx] + " from: " +
                                      forcings.tmpFile + " (" + str(err) + ")")

        # Broadcast the forcing nx/ny values
        forcings.ny_global = self.mpi.broadcast_parameter(forcings.ny_global, self.config)
        forcings.nx_global = self.mpi.broadcast_parameter(forcings.nx_global, self.config)

        with self.parallel_block():
            try:
                # noinspection PyTypeChecker
                forcings.esmf_grid_in = ESMF.Grid(np.array([forcings.ny_global, forcings.nx_global]),
                                                  staggerloc=ESMF.StaggerLoc.CENTER,
                                                  coord_sys=ESMF.CoordSys.SPH_DEG)

                forcings.x_lower_bound = forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
                forcings.x_upper_bound = forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
                forcings.y_lower_bound = forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
                forcings.y_upper_bound = forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
                forcings.nx_local = forcings.x_upper_bound - forcings.x_lower_bound
                forcings.ny_local = forcings.y_upper_bound - forcings.y_lower_bound

                # Check to make sure we have enough dimensionality to run regridding.
                # ESMF requires both grids to have a size of at least 2.
                if forcings.nx_local < 2 or forcings.ny_local < 2:
                    self.log_critical("You have either specified too many cores for: {}," +
                                      "or your input forcing grid is too small to process. " +
                                      "Local grid must have x/y dimension size of 2.".format(forcings.productName))

            except ESMF.ESMPyException as esmf_error:
                self.log_critical("Unable to create source ESMF grid from " +
                                  "temporary file: {} ({})".format(forcings.tmpFile, esmf_error))

            except (ValueError, KeyError, AttributeError) as err:
                self.log_critical("Unable to extract local X/Y boundaries from global grid from " +
                                  "temporary file: {} ({})".format(forcings.tmpFile, err))

        with self.parallel_block():
            if self.IS_MPI_ROOT:
                # Process lat/lon values from the GFS grid.
                if len(dataset.variables['latitude'].shape) == 3:
                    # We have 3D grids already in place.
                    global_latitude_array = dataset.variables['latitude'][0, :, :]
                    global_longitude_array = dataset.variables['longitude'][0, :, :]
                elif len(dataset.variables['longitude'].shape) == 2:
                    # We have 2D grids already in place.
                    global_latitude_array = dataset.variables['latitude'][:, :]
                    global_longitude_array = dataset.variables['longitude'][:, :]
                elif len(dataset.variables['latitude'].shape) == 1:
                    # We have 1D lat/lons we need to translate into 2D grids.
                    global_latitude_array = np.repeat(dataset.variables['latitude'][:][:, np.newaxis],
                                                      forcings.nx_global, axis=1)
                    global_longitude_array = np.tile(dataset.variables['longitude'][:], (forcings.ny_global, 1))
            else:
                global_latitude_array = global_longitude_array = None

            # Scatter global GFS latitude grid to processors..
            local_latitude_array = self.mpi.scatter_array(forcings, global_latitude_array, self.config)
            local_longitude_array = self.mpi.scatter_array(forcings, global_longitude_array, self.config)

        with self.parallel_block():
            try:
                forcings.esmf_lats = forcings.esmf_grid_in.get_coords(1)
            except ESMF.GridException as ge:
                self.log_critical("Unable to locate latitude coordinate object within input ESMF grid: " + str(ge))

            try:
                forcings.esmf_lons = forcings.esmf_grid_in.get_coords(0)
            except ESMF.GridException as ge:
                self.log_critical("Unable to locate longitude coordinate object within input ESMF grid: " + str(ge))

        forcings.esmf_lats[:, :] = local_latitude_array
        forcings.esmf_lons[:, :] = local_longitude_array
        # del local_latitude_array
        # del local_longitude_array
        del global_latitude_array
        del global_longitude_array

        # Create a ESMF field to hold the incoming data.
        with self.parallel_block():
            try:
                forcings.esmf_field_in = ESMF.Field(forcings.esmf_grid_in, name=forcings.productName + "_NATIVE")

                # Scatter global grid to processors..
                if self.IS_MPI_ROOT:
                    var_tmp_global = dataset[forcings.netcdf_var_names[forcing_idx]][0, :, :]
                    # Set all valid values to 1.0, and all missing values to 0.0. This will
                    # be used to generate an output mask that is used later on in downscaling, layering,
                    # etc.
                    var_tmp_global[:, :] = 1.0
                else:
                    var_tmp_global = None
                var_tmp_local = self.mpi.scatter_array(forcings, var_tmp_global, self.config)
            except ESMF.ESMPyException as esmf_error:
                self.log_critical("Unable to create ESMF field object: " + str(esmf_error))

        # Place temporary data into the field array for generating the regridding object.
        forcings.esmf_field_in.data[:, :] = var_tmp_local

        # ## CALCULATE WEIGHT ## #
        # Try to find a pre-existing weight file, if available

        weight_file = None
        if self.config.weightsDir is not None:
            grid_key = forcings.productName
            weight_file = os.path.join(self.config.weightsDir, "ESMF_weight_{}.nc4".format(grid_key))

            # check if file exists:
            if os.path.exists(weight_file):
                # read the data
                try:
                    self.log_status("Loading cached ESMF weight object for {} from {}".format(
                                forcings.productName, weight_file), root_only=True)

                    begin = time.monotonic()
                    forcings.regridObj = ESMF.RegridFromFile(forcings.esmf_field_in,
                                                             forcings.esmf_field_out,
                                                             weight_file)
                    end = time.monotonic()

                    self.log_status("Finished loading weight object with ESMF, took {} seconds".format(end - begin),
                                    root_only=True)

                except (IOError, ValueError, ESMF.ESMPyException) as esmf_error:
                    self.log_critical("Unable to load cached ESMF weight file: {}".format(esmf_error))

        if forcings.regridObj is None:
            self.log_status("Creating weight object from ESMF", root_only=True)

            with self.parallel_block():
                try:
                    begin = time.monotonic()
                    forcings.regridObj = ESMF.Regrid(forcings.esmf_field_in,
                                                     forcings.esmf_field_out,
                                                     src_mask_values=np.array([0]),
                                                     regrid_method=ESMF.RegridMethod.BILINEAR,
                                                     unmapped_action=ESMF.UnmappedAction.IGNORE,
                                                     filename=weight_file)
                    end = time.monotonic()

                    self.log_status("Finished generating weight object with ESMF, took {} seconds".format(end - begin),
                                    root_only=True)
                except (RuntimeError, ImportError, ESMF.ESMPyException) as esmf_error:
                    self.log_critical("Unable to regrid input data from ESMF: " + str(esmf_error))
                    # etype, value, tb = sys.exc_info()
                    # traceback.print_exception(etype, value, tb)
                    # print(forcings.esmf_field_in)
                    # print(forcings.esmf_field_out)
                    # print(np.array([0]))

            with self.parallel_block():
                # Run the regridding object on this test dataset. Check the output grid for
                # any 0 values.
                try:
                    forcings.esmf_field_out = forcings.regridObj(forcings.esmf_field_in,
                                                                 forcings.esmf_field_out)
                except ValueError as ve:
                    self.log_critical("Unable to extract regridded data from ESMF regridded field: " + str(ve))
                    # delete bad cached file if it exists
                    if weight_file is not None:
                        if os.path.exists(weight_file):
                            os.remove(weight_file)

        forcings.regridded_mask[:, :] = forcings.esmf_field_out.data[:, :]

    def need_regrid_object(self, datasource, forcing_idx: int, input_forcings):
        """
        Function for checking to see if regridding weights need to be
        calculated (or recalculated).
        :param datasource:
        :param forcing_idx:
        :param input_forcings:
        :return:
        """

        with self.parallel_block():
            # If the destination ESMF field hasn't been created, create it here.
            if not input_forcings.esmf_field_out:
                try:
                    input_forcings.esmf_field_out = ESMF.Field(self.geo_meta.esmf_grid,
                                                               name=input_forcings.productName + '_FORCING_REGRIDDED')
                except ESMF.ESMPyException as esmf_error:
                    self.log_critical(
                        "Unable to create {} destination ESMF field object: {}".format(input_forcings.productName,
                                                                                       esmf_error))

        # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
        # a new weight file:
        # 1.) This is the first output time step, so we need to calculate a weight file.
        # 2.) The input forcing grid has changed.
        calc_regrid_flag = False

        if input_forcings.nx_global is None or input_forcings.ny_global is None:
            # This is the first timestep.
            # Create out regridded numpy arrays to hold the regridded data.
            # TODO: could this be moved to where the Regrid is actually created?
            input_forcings.regridded_forcings1 = np.empty([input_forcings.NUM_OUTPUTS,
                                                           self.geo_meta.ny_local, self.geo_meta.nx_local],
                                                          np.float32)
            input_forcings.regridded_forcings2 = np.empty([input_forcings.NUM_OUTPUTS,
                                                           self.geo_meta.ny_local, self.geo_meta.nx_local],
                                                          np.float32)

        with self.parallel_block():
            if self.IS_MPI_ROOT:
                if input_forcings.nx_global is None or input_forcings.ny_global is None:
                    # This is the first timestep.
                    calc_regrid_flag = True
                else:
                    if datasource.variables[input_forcings.netcdf_var_names[forcing_idx]].shape[1] \
                            != input_forcings.ny_global and \
                            datasource.variables[input_forcings.netcdf_var_names[forcing_idx]].shape[2] \
                            != input_forcings.nx_global:
                        calc_regrid_flag = True

            # Broadcast the flag to the other processors.
            calc_regrid_flag = self.mpi.broadcast_parameter(calc_regrid_flag, self.config, param_type=bool)

        return calc_regrid_flag

    def already_processed(self, input_forcings):
        # If the expected file is missing, this means we are allowing missing files, simply
        # exit out of this routine as the regridded fields have already been set to NDV.
        if not os.path.isfile(input_forcings.file_in2):
            self.log_status("No {} input file found for this timestep".format(self.name), root_only=True)
            return

        # Check to see if the regrid complete flag for this
        # output time step is true. This entails the necessary
        # inputs have already been regridded and we can move on.
        if input_forcings.regridComplete:
            self.log_status("No {} regridding required for this timestep".format(self.name), root_only=True)
            return
