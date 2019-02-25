from mpi4py import MPI
import os

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