from core import errMod
import argparse
import os
import sys
from core import configMod
from core import ioMod
from core import parallelMod
from core import geoMod
from core import suppPrecipMod
from core import forcingInputMod
from core import forecastMod

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
        errMod.err_out_screen('Specified configuration file: ' + args.config_file + ' not found.')

    # Initialize the configuration object that will contain all
    # user-specified options.
    jobMeta = configMod.ConfigOptions(args.config_file)

    # Place NWM version number (if provided by the user). This will be placed into the final
    # output files as a global attribute.
    if args.nwm_version is not None:
        jobMeta.nwmVersion = args.nwm_version

    # Place NWM configuration (if provided by the user). This will be placed into the final
    # output files as a global attribute.
    if args.nwm_config is not None:
        jobMeta.nwmConfig = args.nwm_config

    # Parse the configuration options
    try:
        jobMeta.read_config()
    except KeyboardInterrupt:
        errMod.err_out_screen('User keyboard interrupt')
    except ImportError:
        errMod.err_out_screen('Missing Python packages')
    except InterruptedError:
        errMod.err_out_screen('External kill signal detected')

    # Initialize our MPI communication
    mpiMeta = parallelMod.MpiConfig()
    try:
        mpiMeta.initialize_comm(jobMeta)
    except:
        errMod.err_out_screen(jobMeta.errMsg)

    # Initialize our WRF-Hydro geospatial object, which contains
    # information about the modeling domain, local processor
    # grid boundaries, and ESMF grid objects/fields to be used
    # in regridding.
    WrfHydroGeoMeta = geoMod.GeoMetaWrfHydro()
    try:
        WrfHydroGeoMeta.initialize_destination_geo(jobMeta,mpiMeta)
    except Exception:
        errMod.err_out_screen_para(jobMeta.errMsg,mpiMeta)
    if jobMeta.spatial_meta is not None:
        try:
            WrfHydroGeoMeta.initialize_geospatial_metadata(jobMeta,mpiMeta)
        except Exception:
            errMod.err_out_screen_para(jobMeta.errMsg,mpiMeta)
    errMod.check_program_status(jobMeta, mpiMeta)

    # Check to make sure we have enough dimensionality to run regridding. ESMF requires both grids
    # to have a size of at least 2.
    if WrfHydroGeoMeta.nx_local < 2 or WrfHydroGeoMeta.ny_local < 2:
        jobMeta.errMsg = "You have specified too many cores for your WRF-Hydro grid. " \
                         "Local grid Must have x/y dimension size of 2."
        errMod.err_out_screen_para(jobMeta.errMsg,mpiMeta)
    errMod.check_program_status(jobMeta, mpiMeta)

    # Initialize our output object, which includes local slabs from the output grid.
    try:
        OutputObj = ioMod.OutputObj(WrfHydroGeoMeta)
    except Exception:
        errMod.err_out_screen_para(jobMeta, mpiMeta)
    errMod.check_program_status(jobMeta, mpiMeta)

    # Next, initialize our input forcing classes. These objects will contain
    # information about our source products (I.E. data type, grid sizes, etc).
    # Information will be mapped via the options specified by the user.
    # In addition, input ESMF grid objects will be created to hold data for
    # downscaling and regridding purposes.
    try:
        inputForcingMod = forcingInputMod.initDict(jobMeta,WrfHydroGeoMeta)
    except Exception:
        errMod.err_out_screen_para(jobMeta, mpiMeta)
    errMod.check_program_status(jobMeta, mpiMeta)

    # If we have specified supplemental precipitation products, initialize
    # the supp class.
    if jobMeta.number_supp_pcp > 0:
        suppPcpMod = suppPrecipMod.initDict(jobMeta,WrfHydroGeoMeta)
    else:
        suppPcpMod = None
    errMod.check_program_status(jobMeta, mpiMeta)

    # There are three run modes (retrospective/realtime/reforecast). We are breaking the main
    # workflow into either retrospective or forecasts (realtime/reforecasts)
    #if jobMeta.retro_flag:
    #    # Place code into here for calling the retro run mod.
    if jobMeta.refcst_flag or jobMeta.realtime_flag:
        # Place code in here for calling the forecasting module.
        forecastMod.process_forecasts(jobMeta, WrfHydroGeoMeta,inputForcingMod,suppPcpMod,mpiMeta,OutputObj)
        #try:
        #    forecastMod.process_forecasts(jobMeta,WrfHydroGeoMeta,
        #                                  inputForcingMod,suppPcpMod,mpiMeta,OutputObj)
        #except Exception:
        #    errMod.log_critical(jobMeta, mpiMeta)
        errMod.check_program_status(jobMeta, mpiMeta)


if __name__ == "__main__":
    main()
