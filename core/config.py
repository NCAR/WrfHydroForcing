#!/usr/bin/env python

import datetime
import json
import os
import sys
from enum import IntEnum
import enum
import configparser
import numpy as np
import yaml
from strenum import StrEnum
from core import time_handling
from core import err_handler


class ConfigOptions:
    """
    Configuration abstract class for configuration options read in from the file
    specified by the user.
    """

    def __init__(self, config):
        """
        Initialize the configuration class to empty None attributes
        param config: The user-specified path to the configuration file.
        """
        #self.input_forcings = None
        self.supp_precip_forcings = None
        #self.input_force_dirs = None
        #self.input_force_types = None
        self.supp_precip_dirs = None
        self.supp_precip_file_types = None
        self.supp_precip_param_dir = None
        self.input_force_mandatory = None
        self.supp_precip_mandatory = None
        self.supp_pcp_max_hours = None
        self.number_inputs = None
        self.number_supp_pcp = None
        self.number_custom_inputs = 0
        self.output_freq = None
        self.output_dir = None
        self.scratch_dir = None
        self.useCompression = 0
        self.useFloats = 0
        self.num_output_steps = None
        self.actual_output_steps = None
        self.retro_flag = None
        self.realtime_flag = None
        self.refcst_flag = None
        self.ana_flag = None
        self.ana_out_dir = None
        self.b_date_proc = None
        self.e_date_proc = None
        self.first_fcst_cycle = None
        self.current_fcst_cycle = None
        self.current_output_step = None
        self.cycle_length_minutes = None
        self.prev_output_date = None
        self.current_output_date = None
        self.look_back = None
        self.fcst_freq = None
        self.nFcsts = None
        self.fcst_shift = None
        self.fcst_input_horizons = None
        self.fcst_input_offsets = None
        self.process_window = None
        self.geogrid = None
        self.spatial_meta = None
        self.grid_meta = None
        self.ignored_border_widths = None
        self.regrid_opt = None
        self.weightsDir = None
        self.regrid_opt_supp_pcp = None
        self.config_path = config
        self.errMsg = None
        self.statusMsg = None
        self.logFile = None
        self.logHandle = None
        self.dScaleParamDirs = None
        self.paramFlagArray = None
        self.forceTemoralInterp = None
        self.suppTemporalInterp = None
        self.t2dDownscaleOpt = None
        self.swDownscaleOpt = None
        self.psfcDownscaleOpt = None
        self.precipDownscaleOpt = None
        self.q2dDownscaleOpt = None
        self.t2BiasCorrectOpt = None
        self.psfcBiasCorrectOpt = None
        self.q2BiasCorrectOpt = None
        self.windBiasCorrect = None
        self.swBiasCorrectOpt = None
        self.lwBiasCorrectOpt = None
        self.precipBiasCorrectOpt = None
        self.runCfsNldasBiasCorrect = False
        self.cfsv2EnsMember = None
        self.customFcstFreq = None
        self.rqiMethod = None
        self.rqiThresh = 1.0
        self.globalNdv = -999999.0
        self.d_program_init = datetime.datetime.utcnow()
        self.errFlag = 0
        self.nwmVersion = None
        self.nwmConfig = None
        self.include_lqfraq = False
        self.forcingInputModYaml  = None
        self.suppPrecipModYaml = None
        self.outputVarAttrYaml = None
        self.ForcingEnum = None
        self.SuppForcingPcpEnum = None
        self.OutputEnum = None

    def read_config(self):
        """
        Read in options from the configuration file and check that proper options
        were provided.
        """
        # Read in the configuration file
        yaml_stream = None
        try:
            yaml_stream = open(self.config_path)
            config = yaml.safe_load(yaml_stream)
        except yaml.YAMLError as yaml_exc:
            err_handler.err_out_screen('Error parsing the configuration file: %s\n%s' % (self.config_path,yaml_exc))
        except IOError:
            err_handler.err_out_screen('Unable to open the configuration file: %s' % self.config_path)
        finally:
            if yaml_stream:
                yaml_stream.close()
        try:
            inputs = config['Input']
        except KeyError:
            raise KeyError("Unable to locate Input map in configuration file.")
            err_handler.err_out_screen('Unable to locate Input map in configuration file.')

        if len(inputs) == 0:
            err_handler.err_out_screen('Please choose at least one Forcings dataset to process')

        self.number_inputs = len(inputs)
       
        # Create Enums dynamically from the yaml files
        try:
            yaml_config = config['YamlConfig']
        except KeyError:
            raise KeyError("Unable to locate YamlConfig in configuration file.")
            err_handler.err_out_screen('Unable to locate YamlConfig in configuration file.')

        try:
            self.forcingInputModYaml = yaml_config['forcingInputModYaml']
        except KeyError:
            raise KeyError("Unable to locate forcingInputModYaml in configuration file.")
            err_handler.err_out_screen('Unable to locate forcingInputModYaml in configuration file.')

        forcing_yaml_stream = open(self.forcingInputModYaml)
        forcingConfig = yaml.safe_load(forcing_yaml_stream)
        dynamicForcing = {}
        for k in forcingConfig.keys():
            dynamicForcing[k] = k
        self.ForcingEnum = enum.Enum('ForcingEnum', dynamicForcing)

        try:
            self.suppPrecipModYaml   = yaml_config['suppPrecipModYaml']
        except KeyError:
            raise KeyError("Unable to locate suppPrecipModYaml in configuration file.")
            err_handler.err_out_screen('Unable to locate suppPrecipModYaml in configuration file.')

        supp_yaml_stream = open(self.suppPrecipModYaml)
        suppConfig = yaml.safe_load(supp_yaml_stream)
        dynamicSupp = {}
        for k in suppConfig.keys():
            dynamicSupp[k] = k
        self.SuppForcingPcpEnum = enum.Enum('SuppForcingPcpEnum', dynamicSupp)
        try:
            self.outputVarAttrYaml  = yaml_config['outputVarAttrYaml']
        except KeyError:
            raise KeyError("Unable to locate outputVarAttrYaml in configuration file.")
            err_handler.err_out_screen('Unable to locate outputVarAttrYaml in configuration file.')

        out_yaml_stream = open(self.outputVarAttrYaml)
        outConfig = yaml.safe_load(out_yaml_stream)
        dynamicOut = {}
        for k in outConfig.keys():
            dynamicOut[k] = k
        self.OutputEnum = enum.Enum('OutputEnum', dynamicOut)

        # Read in the base input forcing options as an array of values to map.
        try:
            self.input_forcings = [input['Forcing'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'Forcing\'] from options: %s' % [str(item) for item in self.ForcingEnum])

        for forceOpt in self.input_forcings:
            # Keep tabs on how many custom input forcings we have.
            if "CUSTOM" in forceOpt:
                self.number_custom_inputs = self.number_custom_inputs + 1

        # Read in the input forcings types (GRIB[1|2], NETCDF)
        try:
            self.input_force_types = [input['Type'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'Type\'] from options: %s' % [str(item) for item in ForcingTypeEnum])
        if len(self.input_force_types) != self.number_inputs:
            err_handler.err_out_screen('Number of Forcing Types must match the number '
                                       'of Forcings in the configuration file.')
        # Read in the input directories for each forcing option.
        try:
            self.input_force_dirs = [input['Dir'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Unable to locate Input[i][\'Dir\'] in Input map in the configuration file.')
        if len(self.input_force_dirs) != self.number_inputs:
            err_handler.err_out_screen('Number of Input Directories must match the number '
                                       'of Forcings in the configuration file.')
        # Loop through and ensure all input directories exist. Also strip out any whitespace
        # or new line characters.
        for dirTmp in range(0, len(self.input_force_dirs)):
            self.input_force_dirs[dirTmp] = self.input_force_dirs[dirTmp].strip()
            if not os.path.isdir(self.input_force_dirs[dirTmp]):
                err_handler.err_out_screen('Unable to locate forcing directory: ' +
                                           self.input_force_dirs[dirTmp])

        # Read in the mandatory enforcement options for input forcings.
        try:
            self.input_force_mandatory = [int(input['Mandatory']) for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Missing Input[i][\'Mandatory\'] in Input map in configuration file.')
        except ValueError:
            err_handler.err_out_screen('Invalid Input[i][\'Mandatory\'] value in Input map in configuration file.')

        if len(self.input_force_mandatory) != self.number_inputs:
            err_handler.err_out_screen('Please specify InputMandatory values for each corresponding input '
                                       'forcings in the configuration file.')
        # Check to make sure enforcement options makes sense.
        for enforceOpt in self.input_force_mandatory:
            if enforceOpt < 0 or enforceOpt > 1:
                err_handler.err_out_screen('Invalid InputMandatory chosen in the configuration file. Please'
                                           ' choose a value of 0 or 1 for each corresponding input forcing.')

        # Read in the ForecastInputHorizons options.
        try:
            self.fcst_input_horizons = [input['Horizon'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                       'configuration file.')
        if len(self.fcst_input_horizons) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[\'Horizon\'] values for '
                                       'each corresponding input forcings for forecasts.')
        # Check to make sure the horizons options make sense. There will be additional
        # checking later when input choices are mapped to input products.
        for horizonOpt in self.fcst_input_horizons:
            if horizonOpt <= 0:
                err_handler.err_out_screen('Please specify ForecastInputHorizon values greater '
                                           'than zero.')

        # Read in the ForecastInputOffsets options.
        try:
            self.fcst_input_offsets = [input['Offset'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Unable to locate ForecastInputOffsets under Forecast '
                                       'section in the configuration file.')
        if len(self.fcst_input_offsets) != self.number_inputs:
            err_handler.err_out_screen('Please specify ForecastInputOffset values for each '
                                           'corresponding input forcings for forecasts.')
        # Check to make sure the input offset options make sense. There will be additional
        # checking later when input choices are mapped to input products.
        for inputOffset in self.fcst_input_offsets:
            if inputOffset < 0:
                err_handler.err_out_screen(
                    'Please specify ForecastInputOffset values greater than or equal to zero.')
            
        # Check for the IgnoredBorderWidths
        try:
            self.ignored_border_widths = [input['IgnoredBorderWidths'] for input in inputs]
        except KeyError:
            # if didn't specify, no worries, just set to 0
                self.ignored_border_widths = [0.0]*self.number_inputs
        if len(self.ignored_border_widths) != self.number_inputs:
            err_handler.err_out_screen('Please specify IgnoredBorderWidths values for each '
                                       'corresponding input forcings for SuppForcing.'
                                       '({} was supplied'.format(self.ignored_border_widths))
        if any(map(lambda x: x < 0, self.ignored_border_widths)):
            err_handler.err_out_screen('Please specify IgnoredBorderWidths values greater than or equal to zero:'
                                       '({} was supplied'.format(self.ignored_border_widths))
        # Process regridding options.
        try:
            self.regrid_opt = [input['RegriddingOpt'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'RegriddingOpt\'] from options: %s' % [str(item) for item in RegriddingOptEnum])
        if len(self.regrid_opt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'RegriddingOpt\'] values for each corresponding input '
                                           'forcings in the configuration file.')

        # Read in temporal interpolation options.
        try:
            self.forceTemoralInterp = [input['TemporalInterp'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'TemporalInterp\'] from options: %s' % [str(item) for item in TemporalInterpEnum])
        if len(self.forceTemoralInterp) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'TemporalInterp\'] values for each corresponding input forcings in the configuration file.')

        # Read in information for the custom input NetCDF files that are to be processed.
        # Read in the ForecastInputHorizons options.
        self.customFcstFreq = [input['Custom']['input_fcst_freq'] for input in inputs if 'Custom' in input and 'input_fcst_freq' in input['Custom']]
        if len(self.customFcstFreq) != self.number_custom_inputs:
            err_handler.err_out_screen('Improper custom_input fcst_freq specified. This number must '
                                       'match the frequency of custom input forcings selected.')


        # * Bias Correction Options *
        try:
            forecast = config['Forecast']
        except KeyError:
            raise KeyError("Forecast not found in configuration file")
            err_handler.err_out_screen('Unable to locate Forecast map in configuration file.')
 
        # Read AnA flag option
        try:
            self.ana_flag = int(forecast['AnAFlag'])
        except KeyError:
            err_handler.err_out_screen('Unable to locate Forecast[\'AnAFlag\'] in the configuration file.')
        except ValueError:
            err_handler.err_out_screen('Improper Forecast[\'AnAFlag\'] value ')
        if self.ana_flag < 0 or self.ana_flag > 1:
            err_handler.err_out_screen('Please choose a Forecast[\'AnAFlag\'] value of 0 or 1.')

        # Read in temperature bias correction options
        try:
            self.t2BiasCorrectOpt = [input['BiasCorrection']['Temperature'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'BiasCorrection\'][\'Temperature\'] from options: %s' % [str(item) for item in BiasCorrTempEnum])
        if len(self.t2BiasCorrectOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'BiasCorrection\'][\'Temperature\'] values for each corresponding '
                                       'input forcings in the configuration file.')
        
        # Read in surface pressure bias correction options.
        try:
            self.psfcBiasCorrectOpt = [input['BiasCorrection']['Pressure'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'BiasCorrection\'][\'Pressure\'] from options: %s' % [str(item) for item in BiasCorrPressEnum])
        if len(self.psfcBiasCorrectOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'BiasCorrection\'][\'Pressure\'] values for each corresponding '
                                       'input forcings in the configuration file.')

        # Ensure the bias correction options chosen make sense.
        for optTmp in self.psfcBiasCorrectOpt:
            if optTmp == 'CFS_V2':
                # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                self.runCfsNldasBiasCorrect = True

        # Read in humidity bias correction options.
        try:
            self.q2BiasCorrectOpt = [input['BiasCorrection']['Humidity'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'BiasCorrection\'][\'Humidity\'] from options: %s' % [str(item) for item in BiasCorrHumidEnum])
        if len(self.q2BiasCorrectOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'BiasCorrection\'][\'Humidity\'] values for each corresponding '
                                       'input forcings in the configuration file.')
        # Ensure the bias correction options chosen make sense.
        for optTmp in self.q2BiasCorrectOpt:
            if optTmp == 'CFS_V2':
                # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                self.runCfsNldasBiasCorrect = True 

        # Read in wind bias correction options.
        try:
            self.windBiasCorrect = [input['BiasCorrection']['Wind'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'BiasCorrection\'][\'Wind\'] from options: %s' % [str(item) for item in BiasCorrWindEnum])
        if len(self.windBiasCorrect) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'BiasCorrection\'][\'Wind\'] values for each corresponding '
                                       'input forcings in the configuration file.')
        # Ensure the bias correction options chosen make sense.
        for optTmp in self.windBiasCorrect:
            if optTmp == 'CFS_V2':
                # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                self.runCfsNldasBiasCorrect = True 
 
        # Read in shortwave radiation bias correction options.
        try:
            self.swBiasCorrectOpt = [input['BiasCorrection']['Shortwave'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'BiasCorrection\'][\'Shortwave\'] from options: %s' % [str(item) for item in BiasCorrSwEnum])
        if len(self.swBiasCorrectOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'BiasCorrection\'][\'Shortwave\'] values for each corresponding '
                                       'input forcings in the configuration file.')
        # Ensure the bias correction options chosen make sense.
        for optTmp in self.swBiasCorrectOpt:
            if optTmp == 'CFS_V2':
                # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                self.runCfsNldasBiasCorrect = True

        # Read in longwave radiation bias correction options.
        try:
            self.lwBiasCorrectOpt = [input['BiasCorrection']['Longwave'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'BiasCorrection\'][\'Longwave\'] from options: %s' % [str(item) for item in BiasCorrLwEnum])
        if len(self.lwBiasCorrectOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'BiasCorrection\'][\'Longwave\'] values for each corresponding '
                                       'input forcings in the configuration file.')

        # Ensure the bias correction options chosen make sense.
        for optTmp in self.lwBiasCorrectOpt:
            if optTmp == 'CFS_V2':
                # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                self.runCfsNldasBiasCorrect = True

        # Read in precipitation bias correction options.
        try:
            self.precipBiasCorrectOpt = [input['BiasCorrection']['Precip'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'BiasCorrection\'][\'Precip\'] from options: %s' % [str(item) for item in BiasCorrPrecipEnum])
        if len(self.precipBiasCorrectOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'BiasCorrection\'][\'Precip\'] values for each corresponding '
                                       'input forcings in the configuration file.')
        # Ensure the bias correction options chosen make sense.
        for optTmp in self.precipBiasCorrectOpt:
            if optTmp == 'CFS_V2':
                # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                self.runCfsNldasBiasCorrect = True

        # Putting a constraint here that CFSv2-NLDAS bias correction (NWM only) is chosen, it must be turned on
        # for ALL variables.
        if self.runCfsNldasBiasCorrect:
            if self.precipBiasCorrectOpt != ['CFS_V2']:
                err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                           'Precipitation under this configuration.')
            if self.lwBiasCorrectOpt != ['CFS_V2']:
                err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                           'long-wave radiation under this configuration.')
            if self.swBiasCorrectOpt != ['CFS_V2']:
                err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                           'short-wave radiation under this configuration.')
            if self.t2BiasCorrectOpt != ['CFS_V2']:
                err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                           'surface temperature under this configuration.')
            if self.windBiasCorrect != ['CFS_V2']:
                err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                           'wind forcings under this configuration.')
            if self.q2BiasCorrectOpt != ['CFS_V2']:
                err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                           'specific humidity under this configuration.')
            if self.psfcBiasCorrectOpt != ['CFS_V2']:
                err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                           'surface pressure under this configuration.')
            # Make sure we don't have any other forcings activated. This can only be ran for CFSv2.
            for optTmp in self.input_forcings:
                if optTmp != 'CFS_V2':
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction can only be used in '
                                               'CFSv2-only configurations')

        # Read in the temperature downscaling options.
        # Create temporary array to hold flags of if we need input parameter files.
        param_flag = np.empty([len(self.input_forcings)], int)
        param_flag[:] = 0
        try:
            self.t2dDownscaleOpt = [input['Downscaling']['Temperature'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'Downscaling\'][\'Temperature\'] from options: %s' % [str(item) for item in DownScaleTempEnum])
        if len(self.t2dDownscaleOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'Downscaling\'][\'Temperature\'] value for each corresponding '
                                       'input forcings in the configuration file.')
        # Ensure the downscaling options chosen make sense.
        count_tmp = 0
        for optTmp in self.t2dDownscaleOpt:
            if optTmp == 'LAPSE_PRE_CALC':
                param_flag[count_tmp] = 1
            count_tmp = count_tmp + 1

        # Read in the pressure downscaling options.
        try:
            self.psfcDownscaleOpt = [input['Downscaling']['Pressure'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'Downscaling\'][\'Pressure\'] from options: %s' % [str(item) for item in DownScalePressEnum])
        if len(self.psfcDownscaleOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'Downscaling\'][\'Pressure\'] value for each corresponding '
                                       'input forcings in the configuration file.')

        # Read in the shortwave downscaling options
        try:
            self.swDownscaleOpt = [input['Downscaling']['Shortwave'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'Downscaling\'][\'Shortwave\'] from options: %s' % [str(item) for item in DownScaleSwEnum])
        if len(self.swDownscaleOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'Downscaling\'][\'Shortwave\'] value for each corresponding '
                                       'input forcings in the configuration file.')

        # Read in the precipitation downscaling options
        try:
            self.precipDownscaleOpt =  [input['Downscaling']['Precip'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'Downscaling\'][\'Precip\'] from options: %s' % [str(item) for item in DownScalePrecipEnum])
        if len(self.precipDownscaleOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'Downscaling\'][\'Precip\'] value for each corresponding '
                                       'input forcings in the configuration file.')
        # Ensure the downscaling options chosen make sense.
        count_tmp = 0
        for optTmp in self.precipDownscaleOpt:
            if optTmp == 'NWM_MM':
                param_flag[count_tmp] = 1
            count_tmp = count_tmp + 1

        # Read in humidity downscaling options.
        try:
            self.q2dDownscaleOpt = [input['Downscaling']['Humidity'] for input in inputs]
        except KeyError:
            err_handler.err_out_screen('Please pick Input[i][\'Downscaling\'][\'Humidity\'] from options: %s' % [str(item) for item in DownScaleHumidEnum])
        if len(self.q2dDownscaleOpt) != self.number_inputs:
            err_handler.err_out_screen('Please specify Input[i][\'Downscaling\'][\'Humidity\'] value for each corresponding '
                                       'input forcings in the configuration file.')

        # Read in the downscaling parameter directory.
        self.paramFlagArray = param_flag
        tmp_scale_param_dirs = []
        if param_flag.sum() > 0:
            tmp_scale_param_dirs = [input['Downscaling']['ParamDir'] for input in inputs if 'ParamDir' in input['Downscaling']]
            if len(tmp_scale_param_dirs) < param_flag.sum():
                err_handler.err_out_screen('Please specify a Input[i][\'Downscaling\'][\'ParamDir\'] for each '
                                           'corresponding downscaling option that requires one.')
            # Loop through each downscaling parameter directory and make sure they exist.
            for dirTmp in range(0, len(tmp_scale_param_dirs)):
                tmp_scale_param_dirs[dirTmp] = tmp_scale_param_dirs[dirTmp].strip()
                if not os.path.isdir(tmp_scale_param_dirs[dirTmp]):
                    err_handler.err_out_screen('Unable to locate parameter directory: ' + tmp_scale_param_dirs[dirTmp])

        # Create a list of downscaling parameter directories for each corresponding
        # input forcing. If no directory is needed, or specified, we will set the value to NONE
        self.dScaleParamDirs = []
        for count_tmp, _ in enumerate(self.input_forcings):
            if param_flag[count_tmp] == 0:
                self.dScaleParamDirs.append('NONE')
            if param_flag[count_tmp] == 1:
                self.dScaleParamDirs.append(tmp_scale_param_dirs[count_tmp])

        # if the directory was specified but not downscaling, set it anyway for bias correction etc.
        if param_flag.sum() == 0 and len([input['Downscaling']['ParamDir'] for input in inputs if 'ParamDir' in input['Downscaling']]) >= 1:
            self.dScaleParamDirs = [input['Downscaling']['ParamDir'] for input in inputs if 'ParamDir' in input['Downscaling']]

        try:
            output = config['Output']
        except KeyError:
            raise KeyError("Output not found in configuration file")
            err_handler.err_out_screen('Unable to locate Output map in configuration file.')

        # Read in the output frequency
        try:
            self.output_freq = int(output['Frequency'])
        except ValueError:
            err_handler.err_out_screen('Improper OutputFrequency value specified  in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate OutputFrequency in the configuration file.')
        if self.output_freq <= 0:
            err_handler.err_out_screen('Please specify an OutputFrequency that is greater than zero minutes.')

        # Read in the output directory
        try:
            self.output_dir = output['Dir']
        except ValueError:
            err_handler.err_out_screen('Improper OutDir specified in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate OutDir in the configuration file.')
        if not os.path.isdir(self.output_dir):
            err_handler.err_out_screen('Specified output directory: ' + self.output_dir + ' not found.')

        # Read in the scratch temporary directory.
        try:
            self.scratch_dir = output['ScratchDir']
        except ValueError:
            err_handler.err_out_screen('Improper ScratchDir specified in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate ScratchDir in the configuration file.')
        if not os.path.isdir(self.scratch_dir):
            err_handler.err_out_screen('Specified output directory: ' + self.scratch_dir + ' not found')

        # Read in compression option
        try:
            self.useCompression = int(output['CompressOutput'])
        except KeyError:
            err_handler.err_out_screen('Unable to locate compressOut in the configuration file.')
        except ValueError:
            err_handler.err_out_screen('Improper compressOut value.')
        if self.useCompression < 0 or self.useCompression > 1:
            err_handler.err_out_screen('Please choose a compressOut value of 0 or 1.')

        # Read in floating-point option
        try:
            self.useFloats = output['FloatOutput']
        except KeyError:
            # err_handler.err_out_screen('Unable to locate Output[\'FloatOutput\'] in the configuration file.')
            self.useFloats = 0
            err_handler.err_out_screen('Please pick output[\'FloatOutput\'] from options: %s' % [str(item) for item in OutputFloatEnum])
        except ValueError:
            err_handler.err_out_screen('Improper floatOutput value: {}'.format(config['Output']['floatOutput']))
       
        try:
            retrospective = config['Retrospective']
        except KeyError:
            err_handler.err_out_screen('Unable to locate Retrospective map in configuration file.') 

        # Read in lqfrac option
        try:
            self.include_lqfraq = int(config['Output'].get('includeLQFraq', 0))
        except KeyError:
            # err_handler.err_out_screen('Unable to locate includeLQFraq in the configuration file.')
            self.include_lqfraq = 0
        except configparser.NoOptionError:
            # err_handler.err_out_screen('Unable to locate includeLQFraq in the configuration file.')
            self.useFinclude_lqfraqloats = 0
        except ValueError:
            err_handler.err_out_screen('Improper includeLQFraq value: {}'.format(config['Output']['includeLQFraq']))
        if self.include_lqfraq < 0 or self.include_lqfraq > 1:
            err_handler.err_out_screen('Please choose an includeLQFraq value of 0 or 1.')

        # Read in retrospective options
        try:
            self.retro_flag = retrospective['Flag']
        except KeyError:
            err_handler.err_out_screen('Unable to locate RetroFlag in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate RetroFlag in the configuration file.')
        except ValueError:
            err_handler.err_out_screen('Improper RetroFlag value ')
        if self.retro_flag < 0 or self.retro_flag > 1:
            err_handler.err_out_screen('Please choose a RetroFlag value of 0 or 1.')

        # Process the beginning date of forcings to process.
        if self.retro_flag == 1:
            self.realtime_flag = False
            self.refcst_flag = False
            try:
                beg_date_tmp = str(restrospective['BDateProc'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate BDateProc under Logistics section in '
                                           'configuration file.')
                beg_date_tmp = None
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate BDateProc under Logistics section in '
                                           'configuration file.')
                beg_date_tmp = None
            if beg_date_tmp != '-9999':
                if len(beg_date_tmp) != 12:
                    err_handler.err_out_screen('Improper BDateProc length entered into the '
                                               'configuration file. Please check your entry.')
                try:
                    self.b_date_proc = datetime.datetime.strptime(beg_date_tmp, '%Y%m%d%H%M')
                except ValueError:
                    err_handler.err_out_screen('Improper BDateProc value entered into the '
                                               'configuration file. Please check your entry.')
            else:
                self.b_date_proc = -9999

            # Process the ending date of retrospective forcings to process
            try:
                end_date_tmp = str(retrospective['EDateProc'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate EDateProc under Logistics section in '
                                           'configuration file.')
                end_date_tmp = None
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate EDateProc under Logistics section in '
                                           'configuration file.')
                end_date_tmp = None
            if end_date_tmp != '-9999':
                if len(end_date_tmp) != 12:
                    err_handler.err_out_screen('Improper EDateProc length entered into the '
                                               'configuration file. Please check your entry.')
                try:
                    self.e_date_proc = datetime.datetime.strptime(end_date_tmp, '%Y%m%d%H%M')
                except ValueError:
                    err_handler.err_out_screen('Improper EDateProc value entered into the '
                                               'configuration file. Please check your entry.')
                if self.b_date_proc == -9999 and self.e_date_proc != -9999:
                    err_handler.err_out_screen('If choosing retrospective forecasting, dates must not be -9999')
                if self.e_date_proc <= self.b_date_proc:
                    err_handler.err_out_screen('Please choose an ending EDateProc that is greater than BDateProc.')
            else:
                self.e_date_proc = -9999
            if self.e_date_proc == -9999 and self.b_date_proc != -9999:
                err_handler.err_out_screen('If choosing retrospective forcings, dates must not be -9999')

            # Calculate the number of output time steps
            dt_tmp = self.e_date_proc - self.b_date_proc
            self.num_output_steps = int((dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) / self.output_freq)
            if self.ana_flag:
                self.actual_output_steps = np.int32(self.nFcsts)
            else:
                self.actual_output_steps = np.int32(self.num_output_steps)

        # Process realtime or reforecasting options.
        if self.retro_flag == 0:
            # If the retro flag is off, we are assuming a realtime or reforecast simulation.
            try:
                self.look_back = int(forecast['LookBack'])
                if self.look_back <= 0 and self.look_back != -9999:
                    err_handler.err_out_screen('Please specify a positive LookBack or -9999 for realtime.')
            except ValueError:
                raise ValueError("Improper value")
                err_handler.err_out_screen('Improper LookBack value entered into the '
                                           'configuration file. Please check your entry.')
                
            except KeyError:
                err_handler.err_out_screen('Unable to locate LookBack in the configuration '
                                           'file. Please verify entries exist.')

            # Process the beginning date of reforecast forcings to process
            try:
                beg_date_tmp = str(forecast['RefcstBDateProc'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate RefcstBDateProc under Logistics section in '
                                           'configuration file.')
                beg_date_tmp = None
            if beg_date_tmp != '-9999':
                if len(beg_date_tmp) != 12:
                    err_handler.err_out_screen('Improper RefcstBDateProc length entered into the '
                                               'configuration file. Please check your entry.')
                try:
                    self.b_date_proc = datetime.datetime.strptime(beg_date_tmp, '%Y%m%d%H%M')
                except ValueError:
                    raise ValueError("Improper value")
                    err_handler.err_out_screen('Improper RefcstBDateProc value entered into the '
                                               'configuration file. Please check your entry.')
            else:
                self.b_date_proc = -9999

            # Process the ending date of reforecast forcings to process
            try:
                end_date_tmp = str(forecast['RefcstEDateProc'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate RefcstEDateProc under Logistics section in '
                                           'configuration file.')
                end_date_tmp = None
            if end_date_tmp != '-9999':
                if len(end_date_tmp) != 12:
                    err_handler.err_out_screen('Improper RefcstEDateProc length entered into the'
                                               'configuration file. Please check your entry.')
                try:
                    self.e_date_proc = datetime.datetime.strptime(end_date_tmp, '%Y%m%d%H%M')
                except ValueError:
                    err_handler.err_out_screen('Improper RefcstEDateProc value entered into the '
                                               'configuration file. Please check your entry.')
            else:
                self.e_date_proc = -9999

            if self.e_date_proc != -9999 and self.e_date_proc <= self.b_date_proc:
                err_handler.err_out_screen('Please choose an ending RefcstEDateProc that is greater '
                                           'than RefcstBDateProc.')

            # If the Retro flag is off, and lookback is off, then we assume we are
            # running a reforecast.
            if self.look_back == -9999:
                self.realtime_flag = False
                self.refcst_flag = True

            elif self.b_date_proc == -9999 and self.e_date_proc == -9999:
                self.realtime_flag = True
                self.refcst_flag = True

            else:
                # The processing window will be calculated based on current time and the
                # lookback option since this is a realtime instance.
                self.realtime_flag = False
                self.refcst_flag = False
                # self.b_date_proc = -9999
                # self.e_date_proc = -9999

            # Calculate the delta time between the beginning and ending time of processing.
            # self.process_window = self.e_date_proc - self.b_date_proc

            # Read in the ForecastFrequency option.
            try:
                self.fcst_freq = int(forecast['Frequency'])
            except ValueError:
                err_handler.err_out_screen('Improper ForecastFrequency value entered into '
                                           'the configuration file. Please check your entry.')
            except KeyError:
                err_handler.err_out_screen('Unable to locate ForecastFrequency in the configuration '
                                           'file. Please verify entries exist.')
            if self.fcst_freq <= 0:
                err_handler.err_out_screen('Please specify a ForecastFrequency in the configuration '
                                           'file greater than zero.')
            # Currently, we only support daily or sub-daily forecasts. Any other iterations should
            # be done using custom config files for each forecast cycle.
            if self.fcst_freq > 1440:
                err_handler.err_out_screen('Only forecast cycles of daily or sub-daily are supported '
                                           'at this time')

            # Read in the ForecastShift option. This is ONLY done for the realtime instance as
            # it's used to calculate the beginning of the processing window.
            if True: # was: self.realtime_flag:
                try:
                    self.fcst_shift = int(config['Forecast']['Shift'])
                except ValueError:
                    err_handler.err_out_screen('Improper ForecastShift value entered into the '
                                               'configuration file. Please check your entry.')
                except KeyError:
                    err_handler.err_out_screen('Unable to locate ForecastShift in the configuration '
                                               'file. Please verify entries exist.')
                if self.fcst_shift < 0:
                    err_handler.err_out_screen('Please specify a ForecastShift in the configuration '
                                               'file greater than or equal to zero.')

                # Calculate the beginning/ending processing dates if we are running realtime
                if self.realtime_flag:
                    time_handling.calculate_lookback_window(self)

            if self.refcst_flag:
                # Calculate the number of forecasts to issue, and verify the user has chosen a
                # correct divider based on the dates
                dt_tmp = self.e_date_proc - self.b_date_proc
                if (dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) % self.fcst_freq != 0:
                    err_handler.err_out_screen('Please choose an equal divider forecast frequency for your '
                                               'specified reforecast range.')
                self.nFcsts = int((dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) / self.fcst_freq)

            if self.look_back != -9999:
                time_handling.calculate_lookback_window(self)

            # Calculate the length of the forecast cycle, based on the maximum
            # length of the input forcing length chosen by the user.
            self.cycle_length_minutes = max(self.fcst_input_horizons)

            # Ensure the number maximum cycle length is an equal divider of the output
            # time step specified by the user.
            if self.cycle_length_minutes % self.output_freq != 0:
                err_handler.err_out_screen('Please specify an output time step that is an equal divider of the '
                                           'maximum of the forecast time horizons specified.')
            # Calculate the number of output time steps per forecast cycle.
            self.num_output_steps = int(self.cycle_length_minutes / self.output_freq)
            if self.ana_flag:
                self.actual_output_steps = np.int32(self.nFcsts)
            else:
                self.actual_output_steps = np.int32(self.num_output_steps)

        # Process geospatial information

        try:
            geospatial = config['Geospatial']
        except KeyError:
            err_handler.err_out_screen('Unable to locate Geospatial map in configuration file.')

        try:
            self.geogrid = geospatial['GeogridIn']
        except KeyError:
            err_handler.err_out_screen('Unable to locate GeogridIn in the configuration file.')
        if not os.path.isfile(self.geogrid):
            err_handler.err_out_screen('Unable to locate necessary geogrid file: ' + self.geogrid)

        # Check for the optional geospatial land metadata file.
        try:
            self.spatial_meta = geospatial['SpatialMetaIn']
        except KeyError:
            err_handler.err_out_screen('Unable to locate SpatialMetaIn in the configuration file.')
        if len(self.spatial_meta) == 0:
            # No spatial metadata file found.
            self.spatial_meta = None
        else:
            if not os.path.isfile(self.spatial_meta):
                err_handler.err_out_screen('Unable to locate optional spatial metadata file: ' +
                                           self.spatial_meta)
        # Check for the optional grid metadata file.
        try:
            self.grid_meta = config['Geospatial'].get('GridMeta', '')
        except KeyError:
            err_handler.err_out_screen('Unable to locate Geospatial section  in the configuration file.')
        if len(self.grid_meta) == 0:
            # No spatial metadata file found.
            self.grid_meta = None
        else:
            if not os.path.isfile(self.grid_meta):
                err_handler.err_out_screen('Unable to locate optional grid metadata file: ' + self.grid_meta)

        try:
            regridding = config.get('Regridding')
        except KeyError:
            err_handler.err_out_screen('Unable to locate Regridding map in configuration file.')

        # Read weight file directory (optional)
        self.weightsDir = regridding.get('WeightsDir')
        if self.weightsDir is not None:
            # if we do have one specified, make sure it exists
            if not os.path.exists(self.weightsDir):
                err_handler.err_out_screen('ESMF Weights file directory specifed ({}) but does not exist').format(
                    self.weightsDir)

        # Calculate the beginning/ending processing dates if we are running realtime
        if self.realtime_flag:
            time_handling.calculate_lookback_window(self)

        try:
            suppforcings = config['SuppForcing']
        except KeyError:
            err_handler.err_out_screen('Unable to locate SuppForcing map in configuration file.')


        # Read in supplemental precipitation options as an array of values to map.
        try:
            self.supp_precip_forcings = [suppforcing['Pcp'] for suppforcing in suppforcings]
        except KeyError:
            err_handler.err_out_screen('Please pick SuppForcing[i][\'Pcp\'] from options: %s' % [str(item) for item in self.SuppForcingPcpEnum])
        self.number_supp_pcp = len(self.supp_precip_forcings)

        # Read in the supp pcp types (GRIB[1|2], NETCDF)
        try:
            self.supp_precip_file_types = [suppforcing['PcpType'] for suppforcing in suppforcings]
            self.supp_precip_file_types = [stype.strip() for stype in self.supp_precip_file_types]
            if self.supp_precip_file_types == ['']:
                self.supp_precip_file_types = []
        except KeyError:
            err_handler.err_out_screen('Unable to locate SuppPcpForcingTypes in SuppForcing section '
                                       'in the configuration file.')
        if len(self.supp_precip_file_types) != self.number_supp_pcp:
            err_handler.err_out_screen('Number of SuppPcpForcingTypes ({}) must match the number '
                                       'of SuppPcp inputs ({}) in the configuration file.'.format(len(self.supp_precip_file_types), self.number_supp_pcp))
        for fileType in self.supp_precip_file_types:
            if fileType not in ['GRIB1', 'GRIB2', 'NETCDF']:
                err_handler.err_out_screen('Invalid SuppForcing file type "{}" specified. '
                                   'Only GRIB1, GRIB2, and NETCDF are supported'.format(fileType))

        if self.number_supp_pcp > 0:
            # Check to make sure supplemental precip options make sense. Also read in the RQI threshold
            # if any radar products where chosen.
            for suppOpt in self.supp_precip_forcings:
                # Read in RQI threshold to apply to radar products.
                if suppOpt in ('MRMS','MRMS_GAGE','MRMS_SBCV2','AK_MRMS','AK_NWS_IV'):
                    try:
                        self.rqiMethod = [suppforcing['RqiMethod'] for suppforcing in suppforcings]
                    except KeyError:
                        err_handler.err_out_screen('Please pick SuppForcing[i][\'RqiMethod\'] from options: %s' % [str(item) for item in SuppForcingRqiMethodEnum])
                    # Check that if we have more than one RqiMethod, it's the correct number
                    if type(self.rqiMethod) is list:
                        if len(self.rqiMethod) != self.number_supp_pcp:
                            err_handler.err_out_screen('Number of RqiMethods ({}) must match the number '
                                                       'of SuppPcp inputs ({}) in the configuration file, or '
                                                       'supply a single method for all inputs'.format(
                                                        len(self.rqiMethod), self.number_supp_pcp))
                    elif type(self.rqiMethod) is str:
                        # Support 'classic' mode of single method
                        self.rqiMethod = [self.rqiMethod] * self.number_supp_pcp

                    try:
                        self.rqiThresh = [suppforcing['RqiThreshold'] for suppforcing in suppforcings]
                    except KeyError:
                        err_handler.err_out_screen('Unable to locate RqiThreshold under '
                                                   'SuppForcing section in thelf.supp_precip_param_dir configuration file.')

                    # Check that if we have more than one RqiThreshold, it's the correct number
                    if type(self.rqiThresh) is list:
                        if len(self.rqiThresh) != self.number_supp_pcp:
                            err_handler.err_out_screen('Number of RqiThresholds ({}) must match the number '
                                                       'of SuppPcp inputs ({}) in the configuration file, or '
                                                       'supply a single threshold for all inputs'.format(
                                                        len(self.rqiThresh), self.number_supp_pcp))
                    elif type(self.rqiThresh) is float:
                        # Support 'classic' mode of single threshold
                        self.rqiThresh = [self.rqiThresh] * self.number_supp_pcp

                    # Make sure the RQI threshold makes sense.
                    for threshold in self.rqiThresh:
                        if threshold < 0.0 or threshold > 1.0:
                            err_handler.err_out_screen('Please specify RqiThresholds between 0.0 and 1.0.')

            # Read in the input directories for each supplemental precipitation product.
            try:
                self.supp_precip_dirs = [suppforcing['PcpDir'] for suppforcing in suppforcings]
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpDirectories in SuppForcing section '
                                           'in the configuration file.')
            # Loop through and ensure all supp pcp directories exist. Also strip out any whitespace
            # or new line characters.
            for dirTmp in range(0, len(self.supp_precip_dirs)):
                self.supp_precip_dirs[dirTmp] = self.supp_precip_dirs[dirTmp].strip()
                if not os.path.isdir(self.supp_precip_dirs[dirTmp]):
                    err_handler.err_out_screen('Unable to locate supp pcp directory: ' + self.supp_precip_dirs[dirTmp])

            #Special case for ExtAnA where we treat comma separated stage IV, MRMS data as one SuppPcp input 
            if 'AK_NWS_IV' in self.supp_precip_forcings:
                if len(self.supp_precip_forcings) != 1:
                    err_handler.err_out_screen('Alaska Stage IV/MRMS SuppPcp option is only supported as a standalone option')
                self.supp_precip_dirs = [",".join(self.supp_precip_dirs)]

            if len(self.supp_precip_dirs) != self.number_supp_pcp:
                err_handler.err_out_screen('Number of SuppPcpDirectories must match the number '
                                           'of SuppForcing in the configuration file.')

            # Process supplemental precipitation enforcement options
            try:
                self.supp_precip_mandatory = [int(suppforcing['PcpMandatory']) for suppforcing in suppforcings]
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpMandatory under the SuppForcing section '
                                           'in the configuration file.')
            if len(self.supp_precip_mandatory) != self.number_supp_pcp:
                err_handler.err_out_screen('Please specify SuppPcpMandatory values for each corresponding '
                                           'supplemental precipitation options in the configuration file.')
            # Check to make sure enforcement options makes sense.
            for enforceOpt in self.supp_precip_mandatory:
                if enforceOpt < 0 or enforceOpt > 1:
                    err_handler.err_out_screen('Invalid SuppPcpMandatory chosen in the configuration file. '
                                               'Please choose a value of 0 or 1 for each corresponding '
                                               'supplemental precipitation product.')

            # Read in the regridding options.
            try:
                self.regrid_opt_supp_pcp = [suppforcing['RegridOptPcp'] for suppforcing in suppforcings]
            except KeyError:
                err_handler.err_out_screen('Unable to locate RegridOptSuppPcp under the SuppForcing section '
                                           'in the configuration file.')
            if len(self.regrid_opt_supp_pcp) != self.number_supp_pcp:
                err_handler.err_out_screen('Please specify RegridOptSuppPcp values for each corresponding supplemental '
                                           'precipitation product in the configuration file.')
            # Check to make sure regridding options makes sense.
            for regridOpt in self.regrid_opt_supp_pcp:
                if regridOpt < 1 or regridOpt > 3:
                    err_handler.err_out_screen('Invalid RegridOptSuppPcp chosen in the configuration file. '
                                               'Please choose a value of 1-3 for each corresponding '
                                               'supplemental precipitation product.')

            # Read in temporal interpolation options.
            try:
                self.suppTemporalInterp = [suppforcing['PcpTemporalInterp'] for suppforcing in suppforcings]
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpTemporalInterpolation under the SuppForcing '
                                           'section in the configuration file.')
            if len(self.suppTemporalInterp) != self.number_supp_pcp:
                err_handler.err_out_screen('Please specify SuppPcpTemporalInterpolation values for each '
                                           'corresponding supplemental precip products in the configuration file.')
            # Ensure the SuppPcpTemporalInterpolation values make sense.
            for temporalInterpOpt in self.suppTemporalInterp:
                if temporalInterpOpt < 0 or temporalInterpOpt > 2:
                    err_handler.err_out_screen('Invalid SuppPcpTemporalInterpolation chosen in the configuration file. '
                                               'Please choose a value of 0-2 for each corresponding input forcing')

            # Read in max time option
            try:
                self.supp_pcp_max_hours = [suppforcing['PcpMaxHours'] for suppforcing in suppforcings]
            except (KeyError):
                self.supp_pcp_max_hours = None      # if missing, don't care, just assume all time


            if type(self.supp_pcp_max_hours) is list:
                if len(self.supp_pcp_max_hours) != self.number_supp_pcp:
                    err_handler.err_out_screen('Number of SuppPcpMaxHours ({}) must match the number '
                                               'of SuppPcp inputs ({}) in the configuration file, or '
                                               'supply a single threshold for all inputs'.format(
                            len(self.supp_pcp_max_hours), self.number_supp_pcp))
            elif type(self.supp_pcp_max_hours) is float:
                # Support 'classic' mode of single threshold
                self.supp_pcp_max_hours = [self.supp_pcp_max_hours] * self.number_supp_pcp

            # Read in the SuppPcpInputOffsets options.
            try:
                self.supp_input_offsets = [suppforcing['PcpInputOffsets'] for suppforcing in suppforcings]
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpInputOffsets under SuppForcing '
                                           'section in the configuration file.')
            if len(self.supp_input_offsets) != self.number_supp_pcp:
                err_handler.err_out_screen('Please specify SuppPcpInputOffsets values for each '
                                           'corresponding input forcings for SuppForcing.')
            # Check to make sure the input offset options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for inputOffset in self.supp_input_offsets:
                if inputOffset < 0:
                    err_handler.err_out_screen(
                            'Please specify SuppPcpInputOffsets values greater than or equal to zero.')

            # Read in the optional parameter directory for supplemental precipitation.
            try:
                self.supp_precip_param_dir = [suppforcing['PcpParamDir'] for suppforcing in suppforcings]
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpParamDir under the SuppForcing section '
                                           'in the configuration file.')
            for dir in self.supp_precip_param_dir:
                if not os.path.isdir(dir):
                    err_handler.err_out_screen('Unable to locate SuppForcing[i][\'PcpParamDir\']: ' + dir)
            #For compatability only keep the first PcpParamDir
            self.supp_precip_param_dir = self.supp_precip_param_dir[0]
        # Read in Ensemble information
        # Read in CFS ensemble member information IF we have chosen CFSv2 as an input
        # forcing.
        for optTmp in self.input_forcings:
            if optTmp == 'CFS_V2':
                try:
                    self.cfsv2EnsMember = [input['Ensembles']['cfsEnsNumber'] for input in inputs if 'cfsEnsNumber' in input['Ensembles']][0]
                except KeyError:
                    err_handler.err_out_screen('Unable to locate cfsEnsNumber under the Ensembles '
                                               'section of the configuration file')
                if self.cfsv2EnsMember < 1 or self.cfsv2EnsMember > 4:
                    err_handler.err_out_screen('Please chose an cfsEnsNumber value of 1,2,3 or 4.')

    @property
    def use_data_at_current_time(self):
        if self.supp_pcp_max_hours is not None:
            hrs_since_start = self.current_output_date - self.current_fcst_cycle
            return hrs_since_start <= datetime.timedelta(hours = self.supp_pcp_max_hours)
        else:
            return True
