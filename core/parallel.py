import numpy as np
import mpi4py
mpi4py.rc.threaded = False
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
            self.comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
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

    def broadcast_parameter(self, value_broadcast, config_options, param_type=int):
        """
        Generic function for sending a parameter value out to the processors.
        :param value_broadcast:
        :param config_options:
        :return:
        """

        dtype = np.dtype(param_type)

        if self.rank == 0:
            param = np.asarray(value_broadcast, dtype=dtype)
        else:
            param = np.empty(dtype=dtype, shape=())

        try:
            self.comm.Bcast(param, root=0)
        except MPI.Exception:
            config_options.errMsg = "Unable to broadcast single value from rank 0."
            err_handler.log_critical(config_options, self)
            return None
        return param.item(0)

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
            err_handler.log_critical(ConfigOptions, self)
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
            else:                                            #data_type_flag == 2:
                arrayGlobalTmp = np.empty([geoMeta.ny_global,
                                           geoMeta.nx_global],
                                          np.float64)
        try:
            self.comm.Bcast(arrayGlobalTmp, root=0)
        except:
            ConfigOptions.errMsg = "Unable to broadcast a global numpy array from rank 0"
            err_handler.log_critical(ConfigOptions, self)
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
            if src_array.dtype == bool:
                data_type_flag = 3

        # Broadcast the data_type_flag to other processors
        if self.rank == 0:
            data_type_buffer = np.array([data_type_flag],np.int32)
        else:
            data_type_buffer = np.empty(1,np.int32)

        try:
            self.comm.Bcast(data_type_buffer, root=0)
        except:
            ConfigOptions.errMsg = "Unable to broadcast numpy datatype value from rank 0"
            err_handler.err_out(ConfigOptions)
            return None

        data_type_flag = data_type_buffer[0]
        data_type_buffer = None

        # gather buffer offsets and bounds to rank 0
        bounds = np.array(
            [np.int32(geoMeta.x_lower_bound), np.int32(geoMeta.y_lower_bound),
             np.int32(geoMeta.x_upper_bound), np.int32(geoMeta.y_upper_bound)])
        global_bounds = np.zeros((self.size*4),np.int32)

        try:
            self.comm.Allgather([bounds, MPI.INTEGER], [global_bounds, MPI.INTEGER])
        except:
            ConfigOptions.errMsg = "Failed all gathering global bounds at rank" + str(self.rank)
            err_handler.err_out(ConfigOptions)
            return None

        # create slices for x and y bounds arrays
        x_lower = global_bounds[0:(self.size * 4) + 0:4]
        y_lower = global_bounds[1:(self.size * 4) + 1:4]
        x_upper = global_bounds[2:(self.size * 4) + 2:4]
        y_upper = global_bounds[3:(self.size * 4) + 3:4]

        # generate counts
        counts = [ (y_upper[i] - y_lower[i]) * (x_upper[i] - x_lower[i])
                   for i in range(0,self.size)]

        #generate offsets:
        offsets = [0]
        for i in range(0, self.size - 1):
            offsets.append(offsets[i] + counts[i])

        # create the send buffer
        if self.rank == 0:
            sendbuf = np.empty([src_array.size],src_array.dtype)

            # fill the send buffer
            for i in range(0,self.size):
                start = offsets[i]
                stop  = offsets[i]+counts[i]
                sendbuf[start:stop] = src_array[y_lower[i]:y_upper[i],
                                                x_lower[i]:x_upper[i]].flatten()
        else:
            sendbuf = None

        #create the recvbuffer
        if data_type_flag == 1:
            data_type = MPI.FLOAT
            recvbuf=np.empty([counts[self.rank]],np.float32)
        elif data_type_flag == 3: 
            data_type = MPI.BOOL
            recvbuf = np.empty([counts[self.rank]], bool)
        else:
            data_type = MPI.DOUBLE
            recvbuf = np.empty([counts[self.rank]], np.float64)

        #scatter the data
        try:
            self.comm.Scatterv( [sendbuf, counts, offsets, data_type], recvbuf, root=0)
        except:
            ConfigOptions.errMsg = "Failed Scatterv from rank 0"
            err_handler.error_out(ConfigOptions)
            return None

        subarray = np.reshape(recvbuf,[y_upper[self.rank] -y_lower[self.rank],x_upper[self.rank]- x_lower[self.rank]]).copy()
        return subarray

    # use scatterv based scatter_array
    scatter_array = scatter_array_scatterv_no_cache

    def merge_slabs_gatherv(self, local_slab, options):

        # gather buffer offsets and bounds to rank 0
        shapes = np.array([np.int32(local_slab.shape[0]), np.int32(local_slab.shape[1])])
        global_shapes = np.zeros((self.size * 2), np.int32)

        try:
            self.comm.Allgather([shapes, MPI.INTEGER], [global_shapes, MPI.INTEGER])
        except:
            options.errMsg ="Failed all gathering slab shapes at rank" + str(self.rank)
            err_handler.log_critical(options,self)
            global_bounds = None
        
        #options.errMsg = "All gather for global shapes complete"
        #err_handler.log_msg(options,self)

        width = global_shapes[1]

        # check that all slabes are the same width and sum the number of rows
        total_rows = 0
        for i in range(0,self.size):
            total_rows += global_shapes[2*i]
            if global_shapes[(2*i)+1] != width:
                options.errMsg = "Error: slabs with differing widths detected on slab for rank" + str(i)
                err_handler.log_critical(options,self)
                self.comm.abort()

        #options.errMsg = "Checking of Rows and Columns complete"
        #err_handler.log_msg(options,self)

        # generate counts
        counts = [ global_shapes[i*2] * global_shapes[(i*2)+1]
                   for i in range(0,self.size)]

        #generate offsets:
        offsets = [0]
        for i in range(0, len(counts) -1 ):
            offsets.append(offsets[i] + counts[i])

        #options.errMsg = "Counts and Offsets generated"
        #err_handler.log_msg(options,self)

        # create the receive buffer
        if self.rank == 0:
            recvbuf = np.empty([total_rows, width], local_slab.dtype)
        else:
            recvbuf = None

        # set the MPI data type
        data_type = MPI.BYTE
        if local_slab.dtype == np.float32:
            data_type = MPI.FLOAT
        elif local_slab.dtype == np.float64:
            data_type = MPI.DOUBLE
        elif data_type == np.int32:
            data_type = MPI.INT

        # get the data with Gatherv
        try:
            self.comm.Gatherv(sendbuf=local_slab, recvbuf=[recvbuf, counts, offsets, data_type], root=0)
        except:
            options.errMsg = "Failed to Gatherv to rank 0 from rank " + str(self.rank)
            err_handler.log_critical(options,self)
            return None

        #options.errMsg = "Gatherv complete"
        #err_handler.log_msg(options,self)

        return recvbuf

