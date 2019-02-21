from core import errmod
import argparse
import os
from core import configmod
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
        errmod.err_out_screen('Improper arguments passed to main calling program')

    if not os.path.isfile(args.config_file[0]):
        errmod.err_out_screen('Specified configuration file: ' + args.config_file[0] + ' not found.')

    # Initialize the configuration object that will contain all
    # user-specified options.
    force_options = configmod.ConfigOptions(args.config_file[0])

    # Parse the configuration options
    try:
        force_options.read_config()
    except KeyboardInterrupt:
        errmod.err_out_screen('User keyboard interrupt')
    except ImportError:
        errmod.err_out_screen('Missing Python packages')
    except InterruptedError:
        errmod.err_out_screen('External kill signal detected')


if __name__ == "__main__":
    main()
