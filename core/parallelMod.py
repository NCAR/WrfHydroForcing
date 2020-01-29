from mpi4py import MPI
import os
from core import errMod
import numpy as np

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

    def initialize_comm(self,ConfigOptions):
        """
        Initial function to initialize MPI.
        :return:
        """
        try:
            self.comm = MPI.COMM_WORLD
        except:
            ConfigOptions.errMsg = "Unable to initialize the MPI Communicator object"
            raise Exception()

        try:
            self.size = self.comm.Get_size()
        except:
            ConfigOptions.errMsg = "Unable to retrieve the MPI size."
            raise Exception()

        try:
            self.rank = self.comm.Get_rank()
        except:
            ConfigOptions.errMsg = "Unable to retrieve the MPI processor rank."
            raise Exception()

    def broadcast_parameter(self,value_broadcast,ConfigOptions):
        """
        Generic function for sending a parameter value out to the processors.
        :param ConfigOptions:
        :return:
        """
        # Create dictionary to hold value.
        if self.rank == 0:
            tmpDict = {'varTmp':value_broadcast}
        else:
            tmpDict = None
        try:
            tmpDict = self.comm.bcast(tmpDict,root=0)
        except:
            ConfigOptions.errMsg = "Unable to broadcast single value from rank 0."
            errMod.log_critical(ConfigOptions,MpiConfig)
            return None
        return tmpDict['varTmp']

    def scatter_array(self,geoMeta,src_array,ConfigOptions):
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
            errMod.log_critical(ConfigOptions,MpiConfig)
            return None
        data_type_flag = tmpDict['varTmp']


        # gather buffer offsets and bounds to rank 0
        if geoMeta.has_cache:
            x_lower = geoMeta.global_x_lower
            y_lower = geoMeta.global_y_lower
            x_upper = geoMeta.global_x_upper
            y_upper = geoMeta.global_y_upper
        else:
            try:
                x_lower = np.asarray(self.comm.allgather(np.int32(geoMeta.x_lower_bound)))
            except:
               ConfigOptions.errMsg("Failed all gathering buffer x lower at rank " + str(self.comm.rank))
               errMod.log_critical(ConfigOptions,MpiConfig)
               return None

            try:
                y_lower = np.asarray(self.comm.allgather(np.int32(geoMeta.y_lower_bound)))
            except:
                ConfigOptions.errMsg("Failed all gathering buffer y lower at rank " + str(self.comm.rank))
                errMod.log_critical(ConfigOptions,MpiConfig)
                return None

            try:
                x_upper = np.asarray(self.comm.allgather(np.int32(geoMeta.x_upper_bound)))
            except:
                ConfigOptions.errMsg("Failed all gathering buffer x upper at rank " + str(self.comm.rank))
                errMod.log_critical(ConfigOptions,MpiConfig)
                return None

            try:
                y_upper = np.asarray(self.comm.allgather(np.int32(geoMeta.y_upper_bound)))
            except:
                ConfigOptions.errMsg("Failed all gathering buffer x upper at rank " + str(self.comm.rank))
                errMod.log_critical(ConfigOptions,MpiConfig)
                return None

            # all ranks records global intervals all ranks mark existance of cache
            geoMeta.global_x_lower = x_lower
            geoMeta.global_y_lower = y_lower
            geoMeta.global_x_upper = x_upper
            geoMeta.global_y_upper = y_upper
            geoMeta.has_cache = True


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
        counts = [ (y_upper[i] -y_lower[i]) *(x_upper[i]- x_lower[i])
                   for i in range(0,self.comm.size)]

        #generate offsets
        offsets = [0]
        for i in range(1, len(counts)):
            offsets.append(offsets[i - 1] + counts[i])
        i = None

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
            ConfigOptions.errMsg("Failed to scatter from rank 0")
            errMod.log_critical(ConfigOptions,MpiConfig)
            return None

        try:
            subarray = np.reshape(recvbuf,[y_upper[self.rank] -y_lower[self.rank],x_upper[self.rank]- x_lower[self.rank]])
            return subarray
        except:
            ConfigOptions.errMsg("Reshape failed for dimensions ["+
                                 str(y_upper[self.rank]-y_lower[self.rank])+
                                 ","+str(x_upper[self.rank]-x_upper[self.rank])+
                                 "] at rank:" +str(self.rank))
            errMod.log_critical(ConfigOptions,MpiConfig)
            return None
            

    #def gather_array(self,array_gather,ConfigOptions):
    #    """
    #    Generic function for gathering local arrays from each processor
    #    to a global array on processor 0.
    #    :param array_gather:
    #    :param ConfigOptions:
    #    :return:
    #    """
    #    final = MpiConfig.comm.gather(array_gather[:, :], root=0)

    #    MpiConfig.comm.barrier()


    #    if self.rank == 0:
    #        arrayGlobal = np.concatenate([final[i] for i in range(MpiConfig.size)].axis=0)
    #    else:
    #        arrayGlobal = None

    #    return arrayGlobal
