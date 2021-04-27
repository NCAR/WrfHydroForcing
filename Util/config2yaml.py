#!/usr/bin/env python

import argparse
import sys
from pprint import pprint
import configparser

import numpy as np
import yaml

import config
import config_v1
import yaml_comment_template as templ

"""
Script to convert old-style WrfHydroForcing .config files to new-style .yaml

Example Usage: 

export PYTHONPATH=$PYTHONPATH:~/git/WrfHydroForcing/
export PYTHONPATH=$PYTHONPATH:~/git/WrfHydroForcing/core
./config2yaml.py ../Test/template_forcing_engine_AnA_v2.config ../Test/template_forcing_engine_AnA_v2_example.yaml
"""

class SpaceDumper(yaml.SafeDumper):
    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()


def conv_class_2_set(klass):
    """
    Returns a set containing tuples of (key,value) for each attribute in a class
    """
    klass_var_set = set()
    for item in vars(klass).items():
        if isinstance(item[1],(list,np.ndarray)):
            klass_var_set.add((item[0],tuple(item[1])))
        else:
            klass_var_set.add(item)

    return klass_var_set


def compare_old_new(config_file, yaml_file):
    """
    Returns a set containing symmetric difference between config file and yaml file
    """
    config_old = config_v1.ConfigOptions(config_file)
    config_old.read_config()
    #pprint(vars(config_old))

    config_new = config.ConfigOptions(yaml_file)
    config_new.read_config()
    #pprint(vars(config_new))

    config_new_set = conv_class_2_set(config_new)
    config_old_set = conv_class_2_set(config_old)
    #pprint(config_new_set)
    #pprint(config_old_set)
    
    return config_new_set^config_old_set


def convert(config_file, yaml_file):
    print('Converting %s -> %s' % (config_file,yaml_file))

    config_old = config_v1.ConfigOptions(config_file)
    config_old.read_config()
    config_params = vars(config_old)
    #pprint(config_params)

    out_yaml = {'Input':[],'Output':{},'Retrospective':{},'Forecast':{},'Geospatial':{},'Regridding':{}, 'SuppForcing':[]}
    custom_count = 0
    for i in range(len(config_params['input_forcings'])):
        input_dict = {}
        input_dict['Forcing'] = config.ForcingEnum(config_params['input_forcings'][i]).name
        input_dict['Type'] = config_params['input_force_types'][i]
        input_dict['Dir'] = config_params['input_force_dirs'][i]
        input_dict['Mandatory'] = bool(config_params['input_force_mandatory'][i])
        input_dict['Horizon'] = config_params['fcst_input_horizons'][i]
        input_dict['Offset'] = config_params['fcst_input_offsets'][i]
        input_dict['IgnoredBorderWidths'] = config_params['ignored_border_widths'][i]
        input_dict['RegriddingOpt'] = config.RegriddingOptEnum(config_params['regrid_opt'][i]).name
        input_dict['TemporalInterp'] = config.TemporalInterpEnum(config_params['forceTemoralInterp'][i]).name
        if input_dict['Forcing'] == 'CUSTOM_1':
            input_dict['Custom'] = {'input_fcst_freq':config_params['customFcstFreq'][custom_count]}
            custom_count += 1
        input_dict['BiasCorrection'] = {}
        input_dict['BiasCorrection']['Temperature'] = config.BiasCorrTempEnum(config_params['t2BiasCorrectOpt'][i]).name
        input_dict['BiasCorrection']['Pressure'] = config.BiasCorrPressEnum(config_params['psfcBiasCorrectOpt'][i]).name
        input_dict['BiasCorrection']['Humidity'] = config.BiasCorrHumidEnum(config_params['q2BiasCorrectOpt'][i]).name
        input_dict['BiasCorrection']['Wind'] = config.BiasCorrWindEnum(config_params['windBiasCorrect'][i]).name
        input_dict['BiasCorrection']['Shortwave'] = config.BiasCorrSwEnum(config_params['swBiasCorrectOpt'][i]).name
        input_dict['BiasCorrection']['Longwave'] = config.BiasCorrLwEnum(config_params['lwBiasCorrectOpt'][i]).name
        input_dict['BiasCorrection']['Precip'] = config.BiasCorrPrecipEnum(config_params['precipBiasCorrectOpt'][i]).name
        input_dict['Downscaling'] = {}
        input_dict['Downscaling']['Temperature'] = config.DownScaleTempEnum(config_params['t2dDownscaleOpt'][i]).name
        input_dict['Downscaling']['Pressure'] = config.DownScalePressEnum(config_params['psfcDownscaleOpt'][i]).name
        input_dict['Downscaling']['Shortwave'] = config.DownScaleSwEnum(config_params['swDownscaleOpt'][i]).name
        input_dict['Downscaling']['Precip'] = config.DownScalePrecipEnum(config_params['precipDownscaleOpt'][i]).name
        input_dict['Downscaling']['Humidity'] = config.DownScaleHumidEnum(config_params['q2dDownscaleOpt'][i]).name
        input_dict['Downscaling']['ParamDir'] = config_params['dScaleParamDirs'][i]
        if input_dict['Forcing'] == 'CFS_V2':
            input_dict['Ensembles'] = {'cfsEnsNumber':config_params['cfsv2EnsMember']}
        out_yaml['Input'].append(input_dict)

    out_yaml['Output']['Frequency'] = config_params['output_freq']
    out_yaml['Output']['Dir'] = config_params['output_dir']
    out_yaml['Output']['ScratchDir'] = config_params['scratch_dir']
    out_yaml['Output']['CompressOutput'] = bool(config_params['useCompression'])
    out_yaml['Output']['FloatOutput'] = config.OutputFloatEnum(config_params['useFloats']).name

    out_yaml['Retrospective']['Flag'] = bool(config_params['retro_flag'])

    #Due to internal config_v1.py logic that modifies b_date_proc and e_date_proc
    configpsr = configparser.ConfigParser()
    configpsr.read(config_file)

    if out_yaml['Retrospective']['Flag']:
        out_yaml['Retrospective']['BDateProc'] = configparser['Retrospective']['BDateProc']
        out_yaml['Retrospective']['EDateProc'] = configparser['Retrospective']['EDateProc']
    
    out_yaml['Forecast']['AnAFlag'] = bool(config_params['ana_flag'])
    out_yaml['Forecast']['LookBack'] = config_params['look_back']
    out_yaml['Forecast']['RefcstBDateProc'] = configpsr['Forecast']['RefcstBDateProc']
    out_yaml['Forecast']['RefcstEDateProc'] = configpsr['Forecast']['RefcstEDateProc']
    out_yaml['Forecast']['Frequency'] = config_params['fcst_freq']
    out_yaml['Forecast']['Shift'] = config_params['fcst_shift']

    out_yaml['Geospatial']['GeogridIn'] = config_params['geogrid']
    out_yaml['Geospatial']['SpatialMetaIn'] = config_params['spatial_meta']

    out_yaml['Regridding']['WeightsDir'] = config_params['weightsDir']

    for i in range(len(config_params['supp_precip_forcings'])):
        supp_forcing_dict = {}
        supp_forcing_dict['Pcp'] = config.SuppForcingPcpEnum(config_params['supp_precip_forcings'][i]).name
        supp_forcing_dict['PcpType'] = config_params['supp_precip_file_types'][i]
        supp_forcing_dict['PcpDir'] = config_params['supp_precip_dirs'][i]
        supp_forcing_dict['PcpMandatory'] = bool(config_params['supp_precip_mandatory'][i])
        supp_forcing_dict['RegridOptPcp'] = config_params['regrid_opt_supp_pcp'][i]
        supp_forcing_dict['PcpTemporalInterp'] = config_params['suppTemporalInterp'][i]
        supp_forcing_dict['PcpInputOffsets'] = config_params['supp_input_offsets'][i]
        if config_params['rqiMethod']:
            supp_forcing_dict['RqiMethod'] = config.SuppForcingRqiMethodEnum(config_params['rqiMethod']).name
        else:
            supp_forcing_dict['RqiMethod'] = "NONE"
        if config_params['rqiThresh']:
            supp_forcing_dict['RqiThreshold'] = config_params['rqiThresh']
        supp_forcing_dict['PcpParamDir'] = config_params['supp_precip_param_dir']
        out_yaml['SuppForcing'].append(supp_forcing_dict)

    
    #print(yaml.dump(out_yaml,Dumper=SpaceDumper,default_flow_style=False))
    with open(yaml_file,'w') as f:
        yaml.dump(out_yaml,f,Dumper=SpaceDumper,default_flow_style=False)

    
    diff_set = compare_old_new(config_file, yaml_file)
    if tuple(sorted([tup[0] for tup in diff_set])) != ('config_path','config_path','d_program_init','d_program_init'):
        raise Exception("Conversion of %s -> %s failed. diff_set = %s" % (config_file, yaml_file, diff_set))


def add_comments(yaml_file):
    lines = []
    with open(yaml_file) as f:
        lines.append(templ.comments['Header'])
        for line in f.readlines():
            line = line.rstrip()
            if line in {"Input:", "Output:", "Retrospective:", "Forecast:", "Geospatial:", "Regridding:", "SuppForcing:", "Ensembles:"}:
                lines.append(templ.comments[line[:-1]])
            lines.append(line)
    
    #print("\n".join(lines))
    with open(yaml_file, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file',type=str,help='Input old-style .conf file')
    parser.add_argument('yaml_file',type=str,help='Output new-style .yaml file')
    args = parser.parse_args()
    convert(args.config_file,args.yaml_file)
    add_comments(args.yaml_file)


if __name__ == '__main__':
    main()
