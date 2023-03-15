import argparse
import os

import ESMF

from core import config
from core import err_handler
from core import forcingInputMod
from core import forecastMod
from core import geoMod
from core import ioMod
from core import parallel
from core import suppPrecipMod


def main():
    """ Main program to process LDASIN forcing files for WRF-Hydro.
        The main execution of the program requires the user to compose
        a forcing configuration file that is parsed by this workflow.
        Please see documentation for further instructions.
        Logan Karsten - National Center for Atmospheric Research
                        karsten@ucar.edu
                        303-497-2693
    """
    # Parse out the path to the configuration file.
    parser = argparse.ArgumentParser(description='Main calling program to generate WRF-Hydro Forcing')
    parser.add_argument('config_file', metavar='config_file', type=str,
                        help='Configuration file for the forcing engine')
    parser.add_argument('nwm_version', metavar='nwm_version', type=str, nargs='?',
                        help='National Water Model Version Number Specification')
    parser.add_argument('nwm_config', metavar='nwm_config', type=str, nargs='?',
                        help='National Water Model Configuration')

    # Process the input arguments into the program.
    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        err_handler.err_out_screen('Specified configuration file: ' + args.config_file + ' not found.')

    # Initialize the configuration object that will contain all
    # user-specified options.
    job_meta = config.ConfigOptions(args.config_file)

    # Place NWM version number (if provided by the user). This will be placed into the final
    # output files as a global attribute.
    if args.nwm_version is not None:
        job_meta.nwmVersion = args.nwm_version

    # Place NWM configuration (if provided by the user). This will be placed into the final
    # output files as a global attribute.
    if args.nwm_config is not None:
        job_meta.nwmConfig = args.nwm_config

    # Parse the configuration options
    try:
        job_meta.read_config()
    except KeyboardInterrupt:
        err_handler.err_out_screen('User keyboard interrupt')
    except ImportError:
        err_handler.err_out_screen('Missing Python packages')
    except InterruptedError:
        err_handler.err_out_screen('External kill signal detected')

    # Initialize our MPI communication
    mpi_meta = parallel.MpiConfig()
    try:
        mpi_meta.initialize_comm(job_meta)
    except:
        err_handler.err_out_screen(job_meta.errMsg)

    # ESMF.Manager(debug=True)
    # Initialize our WRF-Hydro geospatial object, which contains
    # information about the modeling domain, local processor
    # grid boundaries, and ESMF grid objects/fields to be used
    # in regridding.
    WrfHydroGeoMeta = geoMod.GeoMetaWrfHydro()
    try:
        WrfHydroGeoMeta.initialize_destination_geo(job_meta, mpi_meta)
    except Exception:
        err_handler.err_out_screen_para(job_meta.errMsg, mpi_meta)
    if job_meta.spatial_meta is not None:
        try:
            WrfHydroGeoMeta.initialize_geospatial_metadata(job_meta, mpi_meta)
        except Exception:
            err_handler.err_out_screen_para(job_meta.errMsg, mpi_meta)
    err_handler.check_program_status(job_meta, mpi_meta)

    # Check to make sure we have enough dimensionality to run regridding. ESMF requires both grids
    # to have a size of at least 2.
    if WrfHydroGeoMeta.nx_local < 2 or WrfHydroGeoMeta.ny_local < 2:
        job_meta.errMsg = "You have specified too many cores for your WRF-Hydro grid. " \
                         "Local grid Must have x/y dimension size of 2."
        err_handler.err_out_screen_para(job_meta.errMsg, mpi_meta)
    err_handler.check_program_status(job_meta, mpi_meta)
    # Initialize our output object, which includes local slabs from the output grid.
    try:
        OutputObj = ioMod.OutputObj(WrfHydroGeoMeta, job_meta)
    except Exception:
        err_handler.err_out_screen_para(job_meta, mpi_meta)
    err_handler.check_program_status(job_meta, mpi_meta)

    # Next, initialize our input forcing classes. These objects will contain
    # information about our source products (I.E. data type, grid sizes, etc).
    # Information will be mapped via the options specified by the user.
    # In addition, input ESMF grid objects will be created to hold data for
    # downscaling and regridding purposes.
    try:
        inputForcingMod = forcingInputMod.initDict(job_meta,WrfHydroGeoMeta)
    except Exception:
        err_handler.err_out_screen_para(job_meta, mpi_meta)
    err_handler.check_program_status(job_meta, mpi_meta)

   
    # If we have specified supplemental precipitation products, initialize
    # the supp class.
    if job_meta.number_supp_pcp > 0:
        suppPcpMod = suppPrecipMod.initDict(job_meta,WrfHydroGeoMeta)
    else:
        suppPcpMod = None
    err_handler.check_program_status(job_meta, mpi_meta)

    # There are three run modes (retrospective/realtime/reforecast). We are breaking the main
    # workflow into either retrospective or forecasts (realtime/reforecasts)
    #if jobMeta.retro_flag:
    #    # Place code into here for calling the retro run mod.
    #if job_meta.refcst_flag or job_meta.realtime_flag:
    # Place code in here for calling the forecasting module.
    forecastMod.process_forecasts(job_meta, WrfHydroGeoMeta,inputForcingMod,suppPcpMod,mpi_meta,OutputObj)
    #try:
    #    forecastMod.process_forecasts(jobMeta,WrfHydroGeoMeta,
    #                                  inputForcingMod,suppPcpMod,mpiMeta,OutputObj)
    #except Exception:
    #    errMod.log_critical(jobMeta, mpiMeta)
    err_handler.check_program_status(job_meta, mpi_meta)


if __name__ == "__main__":
    main()
