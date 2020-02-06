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

    def scatter_array(self, geoMeta, array_broadcast, ConfigOptions):
        #return scatter_array_logan(self, geoMeta, array_broadcast, ConfigOptions)
        return scatter_array_scatterv_no_cache(self, geoMeta, arrayOptions, ConfigOptions)

    def scatter_array_logan(self, geoMeta, array_broadcast, ConfigOptions):
        """
        Generic function for calling scatter functons based on
        the input dataset type.
        :param geoMeta:
        :param array_broadcast:
        :param ConfigOptions:
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
            tmpDict = {'varTmp': data_type_flag}
        else:
            tmpDict = None
        try:
            tmpDict = self.comm.bcast(tmpDict, root=0)
        except:
            ConfigOptions.errMsg = "Unable to broadcast numpy datatype value from rank 0"
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return None
        data_type_flag = tmpDict['varTmp']

        # Broadcast the global array to the child processors, then
        if self.rank == 0:
            arrayGlobalTmp = array_broadcast
        else:
            if data_type_flag == 1:
                arrayGlobalTmp = np.empty([geoMeta.ny_global,
                                           geoMeta.nx_global],
                                          np.float32)
            if data_type_flag == 2:
                arrayGlobalTmp = np.empty([geoMeta.ny_global,
                                           geoMeta.nx_global],
                                          np.float64)
        try:
            self.comm.Bcast(arrayGlobalTmp, root=0)
        except:
            ConfigOptions.errMsg = "Unable to broadcast a global numpy array from rank 0"
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return None
        arraySub = arrayGlobalTmp[geoMeta.y_lower_bound:geoMeta.y_upper_bound,
                   geoMeta.x_lower_bound:geoMeta.x_upper_bound]
        return arraySub

    def scatter_array_scatterv_no_cache(self,geoMeta,src_array,ConfigOptions):
        """
            Generic function for calling scatter functons based on
            the input dataset type.
            :param geoMeta:
            :param array_broadcast:
            :param ConfigOptions:
            :return:
        """


        # Determine which type of input array we have based on the
        # type of numpy array.
        data_type_flag = -1
        if self.rank == 0:
            if src_array.dtype == np.float32:
                data_type_flag = 1
            if src_array.dtype == np.float64:
                data_type_flag = 2

        # Broadcast the numpy datatype to the other processors.
        if self.rank == 0:
            tmpDict = {'varTmp': data_type_flag}
        else:
            tmpDict = None
        try:
            tmpDict = self.comm.bcast(tmpDict, root=0)
        except:
            ConfigOptions.errMsg = "Unable to broadcast numpy datatype value from rank 0"
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return None
        data_type_flag = tmpDict['varTmp']


        # gather buffer offsets and bounds to rank 0
        bounds = np.array(
            [np.int32(geoMeta.x_lower_bound), np.int32(geoMeta.y_lower_bound),
             np.int32(geoMeta.x_upper_bound), np.int32(geoMeta.y_upper_bound)])
        global_bounds = np.zeros((self.comm.size*4),np.int32)

        try:
            self.comm.Allgather([bounds, MPI.INTEGER], [global_bounds, MPI.INTEGER])
        except:
            ConfigOptions.errMsg = "Failed all gathering global bounds at rank" + str(self.comm.rank)
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return None

        # create slices for x and y bounds arrays
        x_lower = global_bounds[0:(self.comm.size * 4) + 0:4]
        y_lower = global_bounds[1:(self.comm.size * 4) + 1:4]
        x_upper = global_bounds[2:(self.comm.size * 4) + 2:4]
        y_upper = global_bounds[3:(self.comm.size * 4) + 3:4]

        # we know know the local region for each rank
        if self.rank == 0:
            temp = []
            for i in range(0,self.comm.size):
                temp.append(src_array[y_lower[i]:y_upper[i],
                                         x_lower[i]:x_upper[i]].flatten())
            sendbuf = np.concatenate(tuple(temp))
        else:
            sendbuf = None

        # generate counts
        counts = [ (y_upper[i] - y_lower[i]) * (x_upper[i] - x_lower[i])
                   for i in range(0,self.comm.size)]

        #generate offsets:
        offsets = [0]
        for i in range(1, len(counts)):
            offsets.append(offsets[i - 1] + counts[i])

        #create the recvbuffer
        if data_type_flag == 1:
            data_type = MPI.FLOAT
            recvbuf=np.empty([counts[self.comm.rank]],np.float32)
        else:
            data_type = MPI.DOUBLE
            recvbuf = np.empty([counts[self.comm.rank]], np.float64)

        #scatter the data
        try:
            self.comm.Scatterv( [sendbuf, counts, offsets, data_type], recvbuf, root=0)
        except:
            ConfigOptions.errMsg = "Failed Scatterv from rank 0"
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return None

        subarray = np.reshape(recvbuf,[y_upper[self.rank] -y_lower[self.rank],x_upper[self.rank]- x_lower[self.rank]])
        return subarray

