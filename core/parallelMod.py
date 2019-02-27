from mpi4py import MPI
import os
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
        tmpDict = self.comm.bcast(tmpDict,root=0)
        return tmpDict['varTmp']

    def scatter_array(self,nx_local,ny_local,array_broadcast,ConfigOptions):
        """
        Generic function for breaking up an array to processors
        from rank 0.
        :param array_broadcast:
        :param ConfigOptions:
        :return:
        """
        recvbuf = np.empty([ny_local,nx_local])
        self.comm.Scatter(array_broadcast,recvbuf,root=0)
        np.allclose(recvbuf,self.rank)
        return recvbuf