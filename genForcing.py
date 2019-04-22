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

    # Initialize our WRF-Hydro geospatial object, which contains
    # information about the modeling domain, local processor
    # grid boundaries, and ESMF grid objects/fields to be used
    # in regridding.
    WrfHydroGeoMeta = geoMod.GeoMetaWrfHydro()
    try:
        WrfHydroGeoMeta.initialize_destination_geo(jobMeta,mpiMeta)
    except Exception:
        errMod.err_out_screen_para(jobMeta.errMsg,mpiMeta)
    except KeyboardInterrupt:
        errMod.err_out_screen_para('User keyboard interrupt',mpiMeta)
    if jobMeta.spatial_meta is not None:
        try:
            WrfHydroGeoMeta.initialize_geospatial_metadata(jobMeta,mpiMeta)
        except:
            errMod.err_out_screen_para(jobMeta.errMsg,mpiMeta)

    mpiMeta.comm.barrier()

    # Initialize our output object, which includes local slabs from the output grid.
    OutputObj = ioMod.OutputObj(WrfHydroGeoMeta)

    # Next, initialize our input forcing classes. These objects will contain
    # information about our source products (I.E. data type, grid sizes, etc).
    # Information will be mapped via the options specified by the user.
    # In addition, input ESMF grid objects will be created to hold data for
    # downscaling and regridding purposes.
    inputForcingMod = forcingInputMod.initDict(jobMeta,WrfHydroGeoMeta)

    # If we have specified supplemental precipitation products, initialize
    # the supp class.
    if jobMeta.number_supp_pcp > 0:
        suppPcpMod = suppPrecipMod.initDict(jobMeta,WrfHydroGeoMeta)
    else:
        suppPcpMod = None

    for fTmp in jobMeta.input_forcings:
        if mpiMeta.rank == 0:
            print('------------------------------')
            print(inputForcingMod[fTmp].keyValue)
            print(inputForcingMod[fTmp].productName)
            print(inputForcingMod[fTmp].fileType)
            print('------------------------------')

    mpiMeta.comm.barrier()

    # There are three run modes (retrospective/realtime/reforecast). We are breaking the main
    # workflow into either retrospective or forecasts (realtime/reforecasts)
    #if jobMeta.retro_flag:
    #    # Place code into here for calling the retro run mod.
    if jobMeta.refcst_flag or jobMeta.realtime_flag:
        # Place code in here for calling the forecasting module.
        forecastMod.process_forecasts(jobMeta,WrfHydroGeoMeta,inputForcingMod,suppPcpMod,mpiMeta,OutputObj)


if __name__ == "__main__":
    main()
