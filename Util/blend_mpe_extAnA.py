# This is a standalone program that will blend in OWP-Generated
# MPE precipitation into an existing Analysis and Assimilation
# (AnA) cycle forcing file to create a final Extended Analysis
# and Assimilation cycle forcing file. NOTE - This is inteded
# for use with the National Water Model ONLY.

# Logan Karsten
# National Center for Atmospheric Research
# Research Applications Laboratory
# karsten@ucar.edu
# 303-497-2693

import argparse
import os
import sys
import shutil
from netCDF4 import Dataset
import numpy as np

def main():
    """
    Main calling program that requires two system provided arguments:
    1) The original input LDASIN file with existing forcing grids.
    2) The supplemental file containing MPE precipitation to be blended in.
    3) The path to the new extended AnA file that will contain the blended MPE.
    :return:
    """
    # Parse out the path to the configuration file.
    parser = argparse.ArgumentParser(description='Utlity program to blend MPE precipitation '
                                                 'into NWM AnA Forcing Files')
    parser.add_argument('orig_ldasin', metavar='orig_ldasin', type=str, nargs='+',
                        help='Original AnA Forcing File to Blend Into')
    parser.add_argument('mpe_path', metavar='mpe_path', type=str, nargs='+',
                        help='Path to the MPE file containing the precipitation')
    parser.add_argument('out_path', metavar='out_path', type=str, nargs='+',
                        help='Path to the final output LDASIN extended AnA file')

    # Process the input arguments into the program.
    args = parser.parse_args()

    if not os.path.isfile(args.orig_ldasin[0]):
        print("Unable to locate input LDASIN file: " + args.orig_ldasin[0])
        sys.exit(-1)

    if not os.path.isfile(args.mpe_path[0]):
        print("Unable to locate input MPE file: " + args.mpe_path[0])
        sys.exit(-1)

    if os.path.isfile(args.out_path[0]):
        print("Output file: " + args.out_path[0] + " already exists")
        sys.exit(-1)

    # Open the MPE NetCDF file.
    try:
        idMpe = Dataset(args.mpe_path[0],'r')
    except:
        print("Unable to open MPE file: " + args.mpe_path[0])
        sys.exit(-1)

    # Read in the MPE grid.
    try:
        mpePrecip = idMpe.variables['varOut'][0,:,:]
    except:
        print("Unable to extract RAINRATE from: " + args.mpe_path[0])
        sys.exit(-1)

    # Extract the fill value for this grid.
    try:
        mpeFill = idMpe.variables['RAINRATE']._FillValue
    except:
        print("Unable to extract fill value from: " + args.mpe_path[0])
        sys.exit(-1)

    # Close the MPE file since we no longer need it.
    try:
        idMpe.close()
    except:
        print("Unable to close: " + args.mpe_path[0])
        sys.exit(-1)

    # Make a copy of the original file to the output path. We will open it up after and
    # blend in the MPE precipitation.
    tmpPath = args.out_path[0] + ".TMP"

    try:
        shutil.copy(args.orig_ldasin[0], args.out_path[0])
    except:
        print("Unable to copy: " + args.orig_ldasin[0] + " to temporary file: " + tmpPath)
        sys.exit(-1)

    # Open the output file to be ammended with MPE precipitation.
    try:
        idOut = Dataset(tmpPath,'a')
    except:
        print("Unable to open output file: " + tmpPath)
        # Remove the temporary file.
        os.remove(tmpPath)
        sys.exit(-1)

    # Extract the existing precipitation rate grid.
    try:
        precipOut = idOut.variables['RAINRATE'][0,:,:]
    except:
        print("Unable to extract RAINRATE from: " + tmpPath)
        # Remove the temporary file.
        os.remove(tmpPath)
        sys.exit(-1)

    # Calculate where we have valid values to blend in.
    try:
        indValid = np.where((mpePrecip != mpeFill) & (precipOut >= 0.0))
    except:
        print("Unable to run numpy query on valid data for MPE and RAINRATE")
        # Remove the temporary file.
        os.remove(tmpPath)
        sys.exit(-1)

    # Blend in the MPE precip.....
    try:
        precipOut[indValid] = mpePrecip[indValid]
    except:
        print("Unable to blend in MPE precip using valid index")
        # Remove the temporary file.
        os.remove(tmpPath)
        sys.exit(-1)

    # Place final grid back into output file.
    try:
        idOut.variables['RAINRATE'][0,:,:] = precipOut[:,:]
    except:
        print("Unable to place final RAINRATE grid into: " + tmpPath)
        # Remove the temporary file.
        os.remove(tmpPath)
        sys.exit(-1)

    # Close the output file.
    try:
        idOut.close()
    except:
        print("Unable to close: " + tmpPath)
        # Remove the temporary file.
        os.remove(tmpPath)
        sys.exit(-1)

    # Move the temporary file into the final place.
    try:
        shutil.move(tmpPath,args.out_path[0])
    except:
        print("Unable to move: " + tmpPath + " to: " + args.out_path[0])
        # Remove the temporary file.
        os.remove(tmpPath)
        sys.exit(-1)

if __name__ == "__main__":
    main()