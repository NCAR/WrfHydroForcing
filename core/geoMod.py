from core import errmod

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

    def initialize_destination_geo(self):
        """
        Initialization function to initialize ESMF through ESMPy,
        calculate the global parameters of the WRF-Hydro grid
        being processed to, along with the local parameters
        for this particular processor.
        :return:
        """