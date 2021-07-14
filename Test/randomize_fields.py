#!/usr/bin/env python

import argparse
import netCDF4
import numpy as np

#
#Example USAGE: ./randomize_fields.py /glade/scratch/bpetzke/WRF-Hydro/StageIV_Alaska/20210714/nws_precip_last6hours_ak_2021071400.nc observation 
#

def randomize_field(nc_file, field_names):
    print(f"Randomizing {nc_file} fields {field_names}")
    with netCDF4.Dataset(nc_file, 'a') as nc:
        for field_name in field_names:
            nc[field_name][:] = np.random.randn(*nc[field_name].shape)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nc_file',type=str,help='Input/Output .nc file')
    parser.add_argument('field_names',type=str,nargs='+',help='Field names to randomize')
    args = parser.parse_args()    
    randomize_field(args.nc_file,args.field_names)
