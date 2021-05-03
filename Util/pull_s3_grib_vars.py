#!/usr/bin/env python

import argparse
import sys
import re
from io import BytesIO
from io import TextIOWrapper

import boto3
from botocore import UNSIGNED
from botocore.client import Config

"""
Script pulls grib2 data from s3 by variable

Example Usage: ./pull_s3_grib_vars.py s3://noaa-nbm-grib2-pds/blend.20201001/06/qmd/blend.t06z.qmd.f012.ak.grib2 s3://noaa-nbm-grib2-pds/blend.20201001/06/qmd/blend.t06z.qmd.f012.ak.grib2.idx ./blend.t06z.qmd.f012.ak.grib2 "^.*APCP:surface:6-12 hour acc fcst:$"
"""


def file_exists_s3(bucket_name,key,verbose=False):
    """
    Checks if a file exists in hopefully the most performant way possible.
    """
    #client = boto3.client('s3')
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    response_json = client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=key,
    )
    if verbose:
        print(response_json)
    for o in response_json.get('Contents', []):
        if o['Key'] == key:
            return True

    return False


def parse_s3_url(url):
    """
    Break a url into bucket and key
    """
    from urllib.parse import urlparse
    up = urlparse(url)
    bucket = up.netloc
    key = up.path.lstrip('/')
    return bucket, key


def load_txt_stream_s3(bucket_name,key):
    """
    Gets a buffered text stream of the data for bucket_name and key. 
    """
    #client = boto3.client('s3')
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    obj = client.get_object(Bucket=bucket_name, Key=key)
    bytestream = BytesIO(obj['Body'].read())
    txtstream = TextIOWrapper(bytestream)
    return txtstream


def load_byte_stream_s3(bucket_name,key,byte_range=""):
    """
    Gets a buffered byte stream of the data for bucket_name and key.
    """
    #client = boto3.client('s3')
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    obj = client.get_object(Bucket=bucket_name, Key=key, Range=byte_range)
    bytestream = BytesIO(obj['Body'].read())
    return bytestream


def pull_s3_grib_vars(grib_file,grib_index,out_file,var_regex_strs,debug):
    var_regexs = [re.compile(var_regex) for var_regex in var_regex_strs] 
    grib_bucket,grib_key = parse_s3_url(grib_file)
    grib_idx_bucket,grib_idx_key = parse_s3_url(grib_index)
    
    if not file_exists_s3(grib_bucket,grib_key):
        print("Grib file not found")
        return False
    if not file_exists_s3(grib_idx_bucket,grib_idx_key):
        print("Grib index file not found")
        return False

    found_match = False
    with load_txt_stream_s3(grib_idx_bucket,grib_idx_key) as idx_file:
        line = idx_file.readline()
        while line:
            line = line.rstrip()
            if debug:
                print("Index line: %s:" % line)
            for var_regex in var_regexs:
                if var_regex.match(line):
                    found_match = True
                    print("Matched Index line: %s" % line)
                    line_num,beg_offset,ref_date,var_name,var_prime,hour_desc,level = line.split(":")
                    
                    pos = idx_file.tell()
                    next_line = idx_file.readline()
                    idx_file.seek(pos)
                    
                    if next_line:
                        next_line = next_line.rstrip()
                        _,end_offset,_,_,_,_,_ = next_line.split(":")
                        end_offset = str(int(end_offset)-1)
                    else:
                        end_offset = ''

                    byte_range = "bytes=%s-%s" % (beg_offset,end_offset)
                    print("Downloading %s %s" % (grib_file, byte_range))
                    file_part = load_byte_stream_s3(grib_bucket, grib_key, byte_range)
                    with open(out_file,"ab+") as of:
                        of.write(bytes(file_part.read()))
            line = idx_file.readline()

    if not found_match:
        print("Regex patterns not found in the index file")

    return found_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('grib_file',type=str,help='.grib2 file')
    parser.add_argument('grib_index',type=str,help='.grib2.idx file')
    parser.add_argument('out_file',type=str,help='.grib2 file')
    parser.add_argument('var_regex_strs',nargs="+",type=str,help='var_regex1 [var_regex2 ...]')
    parser.add_argument("--debug",action='store_true',help="Debug output flag")
    
    args = parser.parse_args()
    success = pull_s3_grib_vars(args.grib_file,args.grib_index,args.out_file,args.var_regex_strs,args.debug)
    sys.exit(not success)


if __name__ == '__main__':
    main()
