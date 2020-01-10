import numpy as np
from mpi4py import MPI

from core import err_handler


class MpiConfig:
    """
    Abstract class for defining the MPI parameters,
    along with initialization of the MPI communication
    handle from mpi4py.
    """
    def __init__(self):
        """
        Initialize the MPI abstract class that will contain basic
        information and communication handles.
        """
        self.comm = None
        self.rank = None
        self.size = None

    def initialize_comm(self, config_options):
        """
        Initial function to initialize MPI.
        :return:
        """
        try:
            self.comm = MPI.COMM_WORLD
            self.comm.Set_errhandler(MPI.ERRORS_RETURN)
        except AttributeError as ae:
            config_options.errMsg = "Unable to initialize the MPI Communicator object"
            raise ae

        try:
            self.size = self.comm.Get_size()
        except MPI.Exception as mpi_exception:
            config_options.errMsg = "Unable to retrieve the MPI size."
            raise mpi_exception

        try:
            self.rank = self.comm.Get_rank()
        except MPI.Exception as mpi_exception:
            config_options.errMsg = "Unable to retrieve the MPI processor rank."
            raise mpi_exception

    def broadcast_parameter(self, value_broadcast, config_options):
        """
        Generic function for sending a parameter value out to the processors.
        :param value_broadcast:
        :param config_options:
        :return:
        """
        # Create dictionary to hold value.
        if self.rank == 0:
            tmp_dict = {'varTmp': value_broadcast}
        else:
            tmp_dict = None
        try:
            tmp_dict = self.comm.bcast(tmp_dict, root=0)
        except MPI.Exception:
            config_options.errMsg = "Unable to broadcast single value from rank 0."
            err_handler.log_critical(config_options, MpiConfig)
            return None
        return tmp_dict['varTmp']

    def scatter_array(self, geo_meta, array_broadcast, config_options):
        """
        Generic function for calling scatter functons based on
        the input dataset type.
        :param geo_meta:
        :param array_broadcast:
        :param config_options:
        :return:
        """
        # Determine which type of input array we have based on the
        # type of numpy array.
        data_type_flag = -1
        if self.rank == 0:
            if array_broadcast.dtype == np.float32:
                data_type_flag = 1
            if array_broadcast.dtype == np.float64:
                data_type_flag = 2

        # Broadcast the numpy datatype to the other processors.
        if self.rank == 0:
            tmp_dict = {'varTmp': data_type_flag}
        else:
            tmp_dict = None
        try:
            tmp_dict = self.comm.bcast(tmp_dict, root=0)
        except MPI.Exception:
            config_options.errMsg = "Unable to broadcast numpy datatype value from rank 0"
            err_handler.log_critical(config_options, MpiConfig)
            return None
        data_type_flag = tmp_dict['varTmp']

        # Broadcast the global array to the child processors, then
        if self.rank == 0:
            array_global_tmp = array_broadcast
        else:
            if data_type_flag == 1:
                array_global_tmp = np.empty([geo_meta.ny_global, geo_meta.nx_global], np.float32)
            elif data_type_flag == 2:
                array_global_tmp = np.empty([geo_meta.ny_global, geo_meta.nx_global], np.float64)
            else:
                array_global_tmp = None
        try:
            self.comm.Bcast(array_global_tmp, root=0)
        except MPI.Exception:
            config_options.errMsg = "Unable to broadcast a global numpy array from rank 0"
            err_handler.log_critical(config_options, MpiConfig)
            return None
        array_sub = array_global_tmp[geo_meta.y_lower_bound: geo_meta.y_upper_bound,
                                     geo_meta.x_lower_bound: geo_meta.x_upper_bound]
        return array_sub

    # def gather_array(self,array_gather,ConfigOptions):
    #    """
    #    Generic function for gathering local arrays from each processor
    #    to a global array on processor 0.
    #    :param array_gather:
    #    :param ConfigOptions:
    #    :return:
    #    """
    #    final = MpiConfig.comm.gather(array_gather[:, :], root=0)
    #
    #    MpiConfig.comm.barrier()
    #
    #
    #    if self.rank == 0:
    #        arrayGlobal = np.concatenate([final[i] for i in range(MpiConfig.size)].axis=0)
    #    else:
    #        arrayGlobal = None
    #
    #    return arrayGlobal
