from core import errMod
import argparse
import os
from core import configMod
from core import parallelMod
import sys


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
    parser.add_argument('config_file', metavar='config_file', type=str, nargs='+',
                        help='Configuration file for the forcing engine')

    # Process the input arguments into the program.
    args = parser.parse_args()

    if len(args.config_file) > 1:
        errMod.err_out_screen('Improper arguments passed to main calling program')

    if not os.path.isfile(args.config_file[0]):
        errMod.err_out_screen('Specified configuration file: ' + args.config_file[0] + ' not found.')

    # Initialize the configuration object that will contain all
    # user-specified options.
    jobMeta = configMod.ConfigOptions(args.config_file[0])

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

    # There are three run modes (retrospective/realtime/reforecast). We are breaking the main
    # workflow into either retrospective or forecasts (realtime/reforecasts)
    #if jobMeta.retro_flag:
    #    # Place code into here for calling the retro run mod.
    #if jobMeta.refcst_flag or jobMeta.realtime_flag:
    #    # Place code in here for calling the forecasting module.


if __name__ == "__main__":
    main()
