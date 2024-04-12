import configparser
import datetime
import json
import os

import numpy as np

from core import time_handling
from core import err_handler

FORCE_COUNT = 23

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
        self.input_forcings = None
        self.precip_only_flag = False
        self.supp_precip_forcings = None
        self.input_force_dirs = None
        self.input_force_types = None
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
        self.sub_output_hour = None
        self.sub_output_freq = None
        self.output_dir = None
        self.scratch_dir = None
        self.useCompression = 0
        self.useFloats = 0
        self.num_output_steps = None
        self.num_supp_output_steps = None
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
        self.customSuppPcpFreq = None
        self.customFcstFreq = None
        self.rqiMethod = None
        self.rqiThresh = 1.0
        self.globalNdv = -999999.0
        self.d_program_init = datetime.datetime.utcnow()
        self.errFlag = 0
        self.nwmVersion = None
        self.nwmConfig = None
        self.include_lqfrac = False

    def read_config(self):
        """
        Read in options from the configuration file and check that proper options
        were provided.
        """
        # Read in the configuration file
        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
        except KeyError:
            err_handler.err_out_screen('Unable to open the configuration file: ' + self.config_path)


        # Read in supplemental precipitation options as an array of values to map.
        try:
            self.supp_precip_forcings = json.loads(config['SuppForcing']['SuppPcp'])
        except KeyError:
            err_handler.err_out_screen('Unable to locate SuppPcp under SuppForcing section in configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SuppPcp under SuppForcing section in configuration file.')
        except json.decoder.JSONDecodeError:
            err_handler.err_out_screen('Improper SuppPcp option specified in configuration file')
        self.number_supp_pcp = len(self.supp_precip_forcings)

        if self.number_supp_pcp == 1:
            if int(self.supp_precip_forcings[0]) == 13:
                self.precip_only_flag = True

        if self.precip_only_flag == False:
            # Read in the base input forcing options as an array of values to map.
            try:
                self.input_forcings = json.loads(config['Input']['InputForcings'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputForcings under Input section in configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputForcings under Input section in configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper InputForcings option specified in configuration file')
            if len(self.input_forcings) == 0:
                err_handler.err_out_screen('Please choose at least one InputForcings dataset to process')
            self.number_inputs = len(self.input_forcings)

            # Check to make sure forcing options make sense
            for forceOpt in self.input_forcings:
                if forceOpt < 0 or forceOpt > FORCE_COUNT:
                    err_handler.err_out_screen(f'Please specify InputForcings values between 1 and {FORCE_COUNT}.')

            # Keep tabs on how many custom input forcings we have.
                if forceOpt == 10:
                    self.number_custom_inputs = self.number_custom_inputs + 1

            # Read in the input forcings types (GRIB[1|2], NETCDF)
            try:
                self.input_force_types = config.get('Input', 'InputForcingTypes').strip("[]").split(',')
                self.input_force_types = [ftype.strip() for ftype in self.input_force_types]
                if self.input_force_types == ['']:
                    self.input_force_types = []
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputForcingTypes in Input section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputForcingTypes in Input section '
                                           'in the configuration file.')
            if len(self.input_force_types) != self.number_inputs:
                err_handler.err_out_screen('Number of InputForcingTypes must match the number '
                                           'of InputForcings in the configuration file.')
            for fileType in self.input_force_types:
                if fileType not in ['GRIB1', 'GRIB2', 'NETCDF', 'NETCDF4']:
                    err_handler.err_out_screen('Invalid forcing file type "{}" specified. '
                                               'Only GRIB1, GRIB2, and NETCDF are supported'.format(fileType))


            # Read in the input directories for each forcing option.
            try:
                self.input_force_dirs = config.get('Input', 'InputForcingDirectories').split(',')
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputForcingDirectories in Input section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputForcingDirectories in Input section '
                                           'in the configuration file.')
            if len(self.input_force_dirs) != self.number_inputs:
                err_handler.err_out_screen('Number of InputForcingDirectories must match the number '
                                           'of InputForcings in the configuration file.')
            # Loop through and ensure all input directories exist. Also strip out any whitespace
            # or new line characters.
            for dirTmp in range(0, len(self.input_force_dirs)):
                self.input_force_dirs[dirTmp] = self.input_force_dirs[dirTmp].strip()
                if not os.path.isdir(self.input_force_dirs[dirTmp]):
                    err_handler.err_out_screen('Unable to locate forcing directory: ' +
                                               self.input_force_dirs[dirTmp])

            # Read in the mandatory enforcement options for input forcings.
            try:
                self.input_force_mandatory = json.loads(config['Input']['InputMandatory'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputMandatory under Input section in configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputMandatory under Input section in configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper InputMandatory option specified in configuration file')

            # Process input forcing enforcement options
            try:
                self.input_force_mandatory = json.loads(config['Input']['InputMandatory'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputMandatory under the Input section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputMandatory under the Input section '
                                           'in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper InputMandatory options specified in the configuration file.')
            if len(self.input_force_mandatory) != self.number_inputs:
                err_handler.err_out_screen('Please specify InputMandatory values for each corresponding input '
                                           'forcings in the configuration file.')
            # Check to make sure enforcement options makes sense.
            for enforceOpt in self.input_force_mandatory:
                if enforceOpt < 0 or enforceOpt > 1:
                    err_handler.err_out_screen('Invalid InputMandatory chosen in the configuration file. Please'
                                               ' choose a value of 0 or 1 for each corresponding input forcing.')

        # Read in the output frequency
        try:
            self.output_freq = int(config['Output']['OutputFrequency'])
        except ValueError:
            err_handler.err_out_screen('Improper OutputFrequency value specified  in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate OutputFrequency in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate OutputFrequency in the configuration file.')
        if self.output_freq <= 0:
            err_handler.err_out_screen('Please specify an OutputFrequency that is greater than zero minutes.')

        if self.precip_only_flag == True:
            # Read in the custom supp output frequency
            try:
                self.customSuppPcpFreq = int(config['SuppForcing']['customSuppPcpFreq'])
            except ValueError:
                err_handler.err_out_screen('Improper customSuppPcpFreq value specified  in the configuration file.')
            except KeyError:
                err_handler.err_out_screen('Unable to locate customSuppPcpFreq in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate customSuppPcpFreq in the configuration file.')
            if self.output_freq <= 0:
                err_handler.err_out_screen('Please specify an customSuppPcpFreq that is greater than zero minutes.')

        # Read in the sub output hour
        try:
            self.sub_output_hour = int(config['Output']['SubOutputHour'])
        except ValueError:
            err_handler.err_out_screen('Improper SubOutputHour value specified  in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate SubOutputHour in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SubOutputHour in the configuration file.')
        if self.sub_output_hour < 0:
            err_handler.err_out_screen('Please specify an SubOutputHour that is greater than zero minutes.')
        if self.sub_output_hour == 0:
            self.sub_output_hour = None
        # Read in the output frequency
        try:
            self.sub_output_freq = int(config['Output']['SubOutFreq'])
        except ValueError:
            err_handler.err_out_screen('Improper SubOutFreq value specified  in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate SubOutFreq in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SubOutFreq in the configuration file.')
        if self.sub_output_freq < 0:
            err_handler.err_out_screen('Please specify an SubOutFreq that is greater than zero minutes.')
        if self.sub_output_freq == 0:
            self.sub_output_freq = None

        # Read in the output directory
        try:
            self.output_dir = config['Output']['OutDir']
        except ValueError:
            err_handler.err_out_screen('Improper OutDir specified in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate OutDir in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate OutDir in the configuration file.')
        if not os.path.isdir(self.output_dir):
            err_handler.err_out_screen('Specified output directory: ' + self.output_dir + ' not found.')

        # Read in the scratch temporary directory.
        try:
            self.scratch_dir = config['Output']['ScratchDir']
        except ValueError:
            err_handler.err_out_screen('Improper ScratchDir specified in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate ScratchDir in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate ScratchDir in the configuration file.')
        if not os.path.isdir(self.scratch_dir):
            err_handler.err_out_screen('Specified output directory: ' + self.scratch_dir + ' not found')

        # Read in compression option
        try:
            self.useCompression = int(config['Output']['compressOutput'])
        except KeyError:
            err_handler.err_out_screen('Unable to locate compressOut in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate compressOut in the configuration file.')
        except ValueError:
            err_handler.err_out_screen('Improper compressOut value.')
        if self.useCompression < 0 or self.useCompression > 1:
            err_handler.err_out_screen('Please choose a compressOut value of 0 or 1.')

        # Read in floating-point option
        try:
            self.useFloats = int(config['Output']['floatOutput'])
        except KeyError:
            # err_handler.err_out_screen('Unable to locate floatOutput in the configuration file.')
            self.useFloats = 0
        except configparser.NoOptionError:
            # err_handler.err_out_screen('Unable to locate floatOutput in the configuration file.')
            self.useFloats = 0
        except ValueError:
            err_handler.err_out_screen('Improper floatOutput value: {}'.format(config['Output']['floatOutput']))
        if self.useFloats < 0 or self.useFloats > 1:
            err_handler.err_out_screen('Please choose a floatOutput value of 0 or 1.')

        # Read in lqfrac option
        try:
            self.include_lqfrac = int(config['Output'].get('includeLQFrac', 0))
        except KeyError:
            # err_handler.err_out_screen('Unable to locate includeLQFrac in the configuration file.')
            self.include_lqfrac = 0
        except configparser.NoOptionError:
            # err_handler.err_out_screen('Unable to locate includeLQFrac in the configuration file.')
            self.include_lqfrac = 0
        except ValueError:
            err_handler.err_out_screen('Improper includeLQFrac value: {}'.format(config['Output']['includeLQFrac']))
        if self.include_lqfrac < 0 or self.include_lqfrac > 1:
            err_handler.err_out_screen('Please choose an includeLQFrac value of 0 or 1.')

        # Read AnA flag option
        try:
            # check both the Forecast section and if it's not there, the old BiasCorrection location
            self.ana_flag = config['Forecast'].get('AnAFlag', config['BiasCorrection'].get('AnAFlag'))
            if self.ana_flag is None:
                raise KeyError
            else:
                self.ana_flag = int(self.ana_flag)
        except KeyError:
            err_handler.err_out_screen('Unable to locate AnAFlag in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate AnAFlag in the configuration file.')
        except ValueError:
            err_handler.err_out_screen('Improper AnAFlag value ')
        if self.ana_flag < 0 or self.ana_flag > 1:
            err_handler.err_out_screen('Please choose a AnAFlag value of 0 or 1.')

        # Read in retrospective options
        try:
            self.retro_flag = int(config['Retrospective']['RetroFlag'])
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
                beg_date_tmp = config['Retrospective']['BDateProc']
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
                end_date_tmp = config['Retrospective']['EDateProc']
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
            if self.sub_output_hour == None:
                dt_tmp = self.e_date_proc - self.b_date_proc
                self.num_output_steps = int((dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) / self.output_freq)
                if self.precip_only_flag == True:
                    self.num_supp_output_steps = int((dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) / self.customSuppPcpFreq)
                if self.ana_flag:
                    self.actual_output_steps = np.int32(self.nFcsts)
                else:
                    self.actual_output_steps = np.int32(self.num_output_steps)
            else:
                dt_tmp1 = (self.b_date_proc + datetime.timedelta(hours=self.sub_output_hour)) - self.b_date_proc
                dt_tmp2 = self.e_date_proc - (self.b_date_proc + datetime.timedelta(hours=self.sub_output_hour))
                self.num_output_steps = int((dt_tmp1.days * 1440 + dt_tmp1.seconds / 60.0) / self.output_freq) + int((dt_tmp2.days * 1440 + dt_tmp2.seconds / 60.0) / self.sub_output_freq)
                if self.precip_only_flag == True:
                    self.num_supp_output_steps = int((dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) / self.customSuppPcpFreq)
                if self.ana_flag:
                    self.actual_output_steps = np.int32(self.nFcsts)
                else:
                    self.actual_output_steps = np.int32(self.num_output_steps)
        # Process realtime or reforecasting options.
        if self.retro_flag == 0:
            # If the retro flag is off, we are assuming a realtime or reforecast simulation.
            try:
                self.look_back = int(config['Forecast']['LookBack'])
                if self.look_back <= 0 and self.look_back != -9999:
                    err_handler.err_out_screen('Please specify a positive LookBack or -9999 for realtime.')
            except ValueError:
                err_handler.err_out_screen('Improper LookBack value entered into the '
                                           'configuration file. Please check your entry.')
            except KeyError:
                err_handler.err_out_screen('Unable to locate LookBack in the configuration '
                                           'file. Please verify entries exist.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate LookBack in the configuration '
                                           'file. Please verify entries exist.')

            # Process the beginning date of reforecast forcings to process
            try:
                beg_date_tmp = config['Forecast']['RefcstBDateProc']
            except KeyError:
                err_handler.err_out_screen('Unable to locate RefcstBDateProc under Logistics section in '
                                           'configuration file.')
                beg_date_tmp = None
            except configparser.NoOptionError:
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
                    err_handler.err_out_screen('Improper RefcstBDateProc value entered into the '
                                               'configuration file. Please check your entry.')
            else:
                self.b_date_proc = -9999

            # Process the ending date of reforecast forcings to process
            try:
                end_date_tmp = config['Forecast']['RefcstEDateProc']
            except KeyError:
                err_handler.err_out_screen('Unable to locate RefcstEDateProc under Logistics section in '
                                           'configuration file.')
                end_date_tmp = None
            except configparser.NoOptionError:
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
                self.fcst_freq = int(config['Forecast']['ForecastFrequency'])
            except ValueError:
                err_handler.err_out_screen('Improper ForecastFrequency value entered into '
                                           'the configuration file. Please check your entry.')
            except KeyError:
                err_handler.err_out_screen('Unable to locate ForecastFrequency in the configuration '
                                           'file. Please verify entries exist.')
            except configparser.NoOptionError:
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
                    self.fcst_shift = int(config['Forecast']['ForecastShift'])
                except ValueError:
                    err_handler.err_out_screen('Improper ForecastShift value entered into the '
                                               'configuration file. Please check your entry.')
                except KeyError:
                    err_handler.err_out_screen('Unable to locate ForecastShift in the configuration '
                                               'file. Please verify entries exist.')
                except configparser.NoOptionError:
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

            if self.precip_only_flag == False:
                # Read in the ForecastInputHorizons options.
                try:
                    self.fcst_input_horizons = json.loads(config['Forecast']['ForecastInputHorizons'])
                except KeyError:
                    err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                               'configuration file.')
                except configparser.NoOptionError:
                    err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                               'configuration file.')
                except json.decoder.JSONDecodeError:
                    err_handler.err_out_screen('Improper ForecastInputHorizons option specified in '
                                               'configuration file')
                if len(self.fcst_input_horizons) != self.number_inputs:
                    err_handler.err_out_screen('Please specify ForecastInputHorizon values for '
                                               'each corresponding input forcings for forecasts.')

                # Check to make sure the horizons options make sense. There will be additional
                # checking later when input choices are mapped to input products.
                for horizonOpt in self.fcst_input_horizons:
                    if horizonOpt <= 0:
                        err_handler.err_out_screen('Please specify ForecastInputHorizon values greater '
                                                   'than zero.')

            else:
                # Read in the ForecastInputHorizons options.
                try:
                    self.fcst_input_horizons = json.loads(config['Forecast']['ForecastInputHorizons'])
                except KeyError:
                    err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                               'configuration file.')
                except configparser.NoOptionError:
                    err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                               'configuration file.')
                except json.decoder.JSONDecodeError:
                    err_handler.err_out_screen('Improper ForecastInputHorizons option specified in '
                                               'configuration file')
                if len(self.fcst_input_horizons) != 1:
                    err_handler.err_out_screen('Please specify ForecastInputHorizon values for '
                                               'each corresponding input forcings for forecasts.')

            if self.precip_only_flag == False:
                # Read in the ForecastInputOffsets options.
                try:
                    self.fcst_input_offsets = json.loads(config['Forecast']['ForecastInputOffsets'])
                except KeyError:
                    err_handler.err_out_screen('Unable to locate ForecastInputOffsets under Forecast '
                                               'section in the configuration file.')
                except configparser.NoOptionError:
                    err_handler.err_out_screen('Unable to locate ForecastInputOffsets under Forecast '
                                               'section in the configuration file.')
                except json.decoder.JSONDecodeError:
                    err_handler.err_out_screen('Improper ForecastInputOffsets option specified in '
                                               'the configuration file.')
                if len(self.fcst_input_offsets) != self.number_inputs:
                    err_handler.err_out_screen('Please specify ForecastInputOffset values for each '
                                               'corresponding input forcings for forecasts.')
                # Check to make sure the input offset options make sense. There will be additional
                # checking later when input choices are mapped to input products.
                for inputOffset in self.fcst_input_offsets:
                    if inputOffset < 0:
                        err_handler.err_out_screen(
                            'Please specify ForecastInputOffset values greater than or equal to zero.')

            # Calculate the length of the forecast cycle, based on the maximum
            # length of the input forcing length chosen by the user.
            self.cycle_length_minutes = max(self.fcst_input_horizons)

            # Ensure the number maximum cycle length is an equal divider of the output
            # time step specified by the user.
            if self.cycle_length_minutes % self.output_freq != 0:
                err_handler.err_out_screen('Please specify an output time step that is an equal divider of the '
                                           'maximum of the forecast time horizons specified.')
            # Calculate the number of output time steps per forecast cycle.
            if self.sub_output_hour == None:
                self.num_output_steps = int(self.cycle_length_minutes / self.output_freq)
                if self.precip_only_flag == True:
                    self.num_supp_output_steps = int(self.cycle_length_minutes / self.customSuppPcpFreq)
            else:
                self.num_output_steps = int((self.cycle_length_minutes - (self.sub_output_hour*60))/ self.sub_output_freq) + int((self.sub_output_hour*60)/ self.output_freq) - 1
                if self.precip_only_flag == True:
                    self.num_supp_output_steps = int(self.cycle_length_minutes / self.customSuppPcpFreq)

            if self.ana_flag:
                self.actual_output_steps = np.int32(self.nFcsts)
            else:
                self.actual_output_steps = np.int32(self.num_output_steps)

        # Process geospatial information
        try:
            self.geogrid = config['Geospatial']['GeogridIn']
        except KeyError:
            err_handler.err_out_screen('Unable to locate GeogridIn in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate GeogridIn in the configuration file.')
        if not os.path.isfile(self.geogrid):
            err_handler.err_out_screen('Unable to locate necessary geogrid file: ' + self.geogrid)

        # Check for the optional geospatial land metadata file.
        try:
            self.spatial_meta = config['Geospatial']['SpatialMetaIn']
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

        if self.precip_only_flag == False:
            # Check for the IgnoredBorderWidths
            try:
                self.ignored_border_widths = json.loads(config['Geospatial']['IgnoredBorderWidths'])
            except (KeyError, configparser.NoOptionError):
                # if didn't specify, no worries, just set to 0
                self.ignored_border_widths = [0.0]*self.number_inputs
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper IgnoredBorderWidths option specified in the configuration file.'
                                           '({} was supplied'.format(config['Geospatial']['IgnoredBorderWidths']))
            if len(self.ignored_border_widths) != self.number_inputs:
                err_handler.err_out_screen('Please specify IgnoredBorderWidths values for each '
                                           'corresponding input forcings for SuppForcing.'
                                           '({} was supplied'.format(self.ignored_border_widths))
            if any(map(lambda x: x < 0, self.ignored_border_widths)):
                err_handler.err_out_screen('Please specify IgnoredBorderWidths values greater than or equal to zero:'
                                           '({} was supplied'.format(self.ignored_border_widths))

            # Process regridding options.
            try:
                self.regrid_opt = json.loads(config['Regridding']['RegridOpt'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate RegridOpt under the Regridding section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate RegridOpt under the Regridding section '
                                           'in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper RegridOpt options specified in the configuration file.')
            if len(self.regrid_opt) != self.number_inputs:
                err_handler.err_out_screen('Please specify RegridOpt values for each corresponding input '
                                           'forcings in the configuration file.')
            # Check to make sure regridding options makes sense.
            for regridOpt in self.regrid_opt:
                if regridOpt < 1 or regridOpt > 3:
                    err_handler.err_out_screen('Invalid RegridOpt chosen in the configuration file. Please choose a '
                                               'value of 1-3 for each corresponding input forcing.')

            # Read weight file directory (optional)
            self.weightsDir = config['Regridding'].get('RegridWeightsDir')
            if self.weightsDir is not None:
                # if we do have one specified, make sure it exists
                if not os.path.exists(self.weightsDir):
                    err_handler.err_out_screen('ESMF Weights file directory specifed ({}) but does not exist').format(
                        self.weightsDir)

        # Calculate the beginning/ending processing dates if we are running realtime
        if self.realtime_flag:
            time_handling.calculate_lookback_window(self)

        if self.precip_only_flag == False:
            # Read in temporal interpolation options.
            try:
                self.forceTemoralInterp = json.loads(config['Interpolation']['ForcingTemporalInterpolation'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate ForcingTemporalInterpolation under the Interpolation '
                                           'section in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ForcingTemporalInterpolation under the Interpolation '
                                           'section in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper ForcingTemporalInterpolation options specified in the '
                                           'configuration file.')
            if len(self.forceTemoralInterp) != self.number_inputs:
                err_handler.err_out_screen('Please specify ForcingTemporalInterpolation values for each '
                                           'corresponding input forcings in the configuration file.')
            # Ensure the forcingTemporalInterpolation values make sense.
            for temporalInterpOpt in self.forceTemoralInterp:
                if temporalInterpOpt < 0 or temporalInterpOpt > 2:
                    err_handler.err_out_screen('Invalid ForcingTemporalInterpolation chosen in the configuration file. '
                                               'Please choose a value of 0-2 for each corresponding input forcing.')

            # Read in the temperature downscaling options.
            # Create temporary array to hold flags of if we need input parameter files.
            param_flag = np.empty([len(self.input_forcings)], dtype = int)
            param_flag[:] = 0
            try:
                self.t2dDownscaleOpt = json.loads(config['Downscaling']['TemperatureDownscaling'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate TemperatureDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate TemperatureDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper TemperatureDownscaling options specified in the configuration file.')
            if len(self.t2dDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify TemperatureDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the downscaling options chosen make sense.
            count_tmp = 0
            for optTmp in self.t2dDownscaleOpt:
                if optTmp < 0 or optTmp > 3:
                    err_handler.err_out_screen('Invalid TemperatureDownscaling options specified in '
                                               'the configuration file.')
                if optTmp == 2:
                    param_flag[count_tmp] = 1
                count_tmp = count_tmp + 1

            # Read in the pressure downscaling options.
            try:
                self.psfcDownscaleOpt = json.loads(config['Downscaling']['PressureDownscaling'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate PressureDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate PressureDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper PressureDownscaling options specified in the configuration file.')
            if len(self.psfcDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify PressureDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.psfcDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                    err_handler.err_out_screen('Invalid PressureDownscaling options specified in the configuration file.')

            # Read in the shortwave downscaling options
            try:
                self.swDownscaleOpt = json.loads(config['Downscaling']['ShortwaveDownscaling'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate ShortwaveDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ShortwaveDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper ShortwaveDownscaling options specified in the configuration file.')
            if len(self.swDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify ShortwaveDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.swDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                    err_handler.err_out_screen('Invalid ShortwaveDownscaling options specified in the configuration file.')

            # Read in humidity downscaling options.
            try:
                self.q2dDownscaleOpt = json.loads(config['Downscaling']['HumidityDownscaling'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate HumidityDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate HumidityDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper HumidityDownscaling options specified in the configuration file.')
            if len(self.q2dDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify HumidityDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.q2dDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                    err_handler.err_out_screen('Invalid HumidityDownscaling options specified in the configuration file.')


            # Read in the downscaling parameter directory.
            self.paramFlagArray = param_flag
            tmp_scale_param_dirs = []
            if param_flag.sum() > 0:
                self.paramFlagArray = param_flag
                try:
                    tmp_scale_param_dirs = config.get('Downscaling', 'DownscalingParamDirs').split(',')
                except KeyError:
                    err_handler.err_out_screen('Unable to locate DownscalingParamDirs in the configuration file.')
                except configparser.NoOptionError:
                    err_handler.err_out_screen('Unable to locate DownscalingParamDirs in the configuration file.')
                if len(tmp_scale_param_dirs) == 1:
                    # single directory for all params
                    tmp_scale_param_dirs *= param_flag.sum()
                elif len(tmp_scale_param_dirs) != param_flag.sum():
                    err_handler.err_out_screen('Please specify a downscaling parameter directory for each '
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
            try:
                if param_flag.sum() == 0 and len(config.get('Downscaling', 'DownscalingParamDirs').split(',')) == 1:
                    self.dScaleParamDirs = [config.get('Downscaling', 'DownscalingParamDirs').split(',')[0]] * len(self.input_forcings)
            except KeyError:
                pass    # TODO: this should not be `pass` if we have a parameter-based Bias Correction scheme selected


        # Read in the precipitation downscaling options
        try:
            self.precipDownscaleOpt = json.loads(config['Downscaling']['PrecipDownscaling'])
        except KeyError:
            err_handler.err_out_screen('Unable to locate PrecipDownscaling under the Downscaling '
                                       'section of the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate PrecipDownscaling under the Downscaling '
                                       'section of the configuration file.')
        except json.decoder.JSONDecodeError:
            err_handler.err_out_screen('Improper PrecipDownscaling options specified in the configuration file.')
        if self.precip_only_flag == False:
            if len(self.precipDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify PrecipDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')


        #   * Bias Correction Options *

        if self.precip_only_flag == False:
            # Read in temperature bias correction options
            try:
                self.t2BiasCorrectOpt = json.loads(config['BiasCorrection']['TemperatureBiasCorrection'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate TemperatureBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:

                err_handler.err_out_screen('Unable to locate TemperatureBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper TemperatureBiasCorrection options specified in '
                                           'the configuration file.')
            if len(self.t2BiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify TemperatureBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.t2BiasCorrectOpt:
                if optTmp < 0 or optTmp > 5:
                    err_handler.err_out_screen('Invalid TemperatureBiasCorrection options specified in the '
                                               'configuration file.')

            # Read in surface pressure bias correction options.
            try:
                self.psfcBiasCorrectOpt = json.loads(config['BiasCorrection']['PressureBiasCorrection'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate PressureBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate PressureBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper PressureBiasCorrection options specified in the configuration file.')
            if len(self.psfcDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify PressureBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.psfcBiasCorrectOpt:
                if optTmp < 0 or optTmp > 1:
                    err_handler.err_out_screen('Invalid PressureBiasCorrection options specified in the '
                                               'configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in humidity bias correction options.
            try:
                self.q2BiasCorrectOpt = json.loads(config['BiasCorrection']['HumidityBiasCorrection'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate HumidityBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate HumidityBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper HumdityBiasCorrection options specified in the configuration file.')
            if len(self.q2BiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify HumidityBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.q2BiasCorrectOpt:
                if optTmp < 0 or optTmp > 2:
                    err_handler.err_out_screen('Invalid HumidityBiasCorrection options specified in the '
                                               'configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in wind bias correction options.
            try:
                self.windBiasCorrect = json.loads(config['BiasCorrection']['WindBiasCorrection'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate WindBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate WindBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper WindBiasCorrection options specified in the configuration file.')
            if len(self.windBiasCorrect) != self.number_inputs:
                err_handler.err_out_screen('Please specify WindBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.windBiasCorrect:
                if optTmp < 0 or optTmp > 4:
                    err_handler.err_out_screen('Invalid WindBiasCorrection options specified in the configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in shortwave radiation bias correction options.
            try:
                self.swBiasCorrectOpt = json.loads(config['BiasCorrection']['SwBiasCorrection'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate SwBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SwBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper SwBiasCorrection options specified in the configuration file.')
            if len(self.swBiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify SwBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.swBiasCorrectOpt:
                if optTmp < 0 or optTmp > 2:
                    err_handler.err_out_screen('Invalid SwBiasCorrection options specified in the configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in longwave radiation bias correction options.
            try:
                self.lwBiasCorrectOpt = json.loads(config['BiasCorrection']['LwBiasCorrection'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate LwBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate LwBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper LwBiasCorrection options specified in '
                                           'the configuration file.')
            if len(self.lwBiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify LwBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.lwBiasCorrectOpt:
                if optTmp < 0 or optTmp > 4:
                    err_handler.err_out_screen('Invalid LwBiasCorrection options specified in the configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in precipitation bias correction options.
            try:
                self.precipBiasCorrectOpt = json.loads(config['BiasCorrection']['PrecipBiasCorrection'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate PrecipBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate PrecipBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper PrecipBiasCorrection options specified in the configuration file.')
            if self.precip_only_flag == False:
                if len(self.precipBiasCorrectOpt) != self.number_inputs:
                    err_handler.err_out_screen('Please specify PrecipBiasCorrection values for each corresponding '
                                               'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.precipBiasCorrectOpt:
                if optTmp < 0 or optTmp > 2:
                    err_handler.err_out_screen('Invalid PrecipBiasCorrection options specified in the configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Putting a constraint here that CFSv2-NLDAS bias correction (NWM only) is chosen, it must be turned on
            # for ALL variables.
            if self.runCfsNldasBiasCorrect:
                if min(self.precipBiasCorrectOpt) != 1 and max(self.precipBiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'Precipitation under this configuration.')
                if min(self.lwBiasCorrectOpt) != 1 and max(self.lwBiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'long-wave radiation under this configuration.')
                if min(self.swBiasCorrectOpt) != 1 and max(self.swBiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'short-wave radiation under this configuration.')
                if min(self.t2BiasCorrectOpt) != 1 and max(self.t2BiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'surface temperature under this configuration.')
                if min(self.windBiasCorrect) != 1 and max(self.windBiasCorrect) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'wind forcings under this configuration.')
                if min(self.q2BiasCorrectOpt) != 1 and max(self.q2BiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'specific humidity under this configuration.')
                if min(self.psfcBiasCorrectOpt) != 1 and max(self.psfcBiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'surface pressure under this configuration.')
                # Make sure we don't have any other forcings activated. This can only be ran for CFSv2.
                for optTmp in self.input_forcings:
                    if optTmp != 7:
                        err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction can only be used in '
                                                   'CFSv2-only configurations')


        # Read in precipitation bias correction options.
        try:
            self.precipBiasCorrectOpt = json.loads(config['BiasCorrection']['PrecipBiasCorrection'])
        except KeyError:
            err_handler.err_out_screen('Unable to locate PrecipBiasCorrection under the '
                                       'BiasCorrection section of the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate PrecipBiasCorrection under the '
                                       'BiasCorrection section of the configuration file.')
        except json.JSONDecodeError:
            err_handler.err_out_screen('Improper PrecipBiasCorrection options specified in the configuration file.')
        if self.precip_only_flag == False:
            if len(self.precipBiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify PrecipBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
        # Ensure the bias correction options chosen make sense.
        for optTmp in self.precipBiasCorrectOpt:
            if optTmp < 0 or optTmp > 1:
                err_handler.err_out_screen('Invalid PrecipBiasCorrection options specified in the configuration file.')
            if optTmp == 1:
            # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                self.runCfsNldasBiasCorrect = True
        # Putting a constraint here that CFSv2-NLDAS bias correction (NWM only) is chosen, it must be turned on
        # for ALL variables.
        if self.runCfsNldasBiasCorrect:
            if min(self.precipBiasCorrectOpt) != 1 and max(self.precipBiasCorrectOpt) != 1:
                err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                           'Precipitation under this configuration.')

        # Read in supplemental precipitation options as an array of values to map.
        try:
            self.supp_precip_forcings = json.loads(config['SuppForcing']['SuppPcp'])
        except KeyError:
            err_handler.err_out_screen('Unable to locate SuppPcp under SuppForcing section in configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SuppPcp under SuppForcing section in configuration file.')
        except json.decoder.JSONDecodeError:
            err_handler.err_out_screen('Improper SuppPcp option specified in configuration file')
        self.number_supp_pcp = len(self.supp_precip_forcings)


        # Read in the supp pcp types (GRIB[1|2], NETCDF)
        try:
            self.supp_precip_file_types = config.get('SuppForcing', 'SuppPcpForcingTypes').strip("[]").split(',')
            self.supp_precip_file_types = [stype.strip() for stype in self.supp_precip_file_types]
            if self.supp_precip_file_types == ['']:
                self.supp_precip_file_types = []
        except KeyError:
            err_handler.err_out_screen('Unable to locate SuppPcpForcingTypes in SuppForcing section '
                                       'in the configuration file.')
        except configparser.NoOptionError:
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
                if suppOpt < 0 or suppOpt > 13:
                    err_handler.err_out_screen('Please specify SuppForcing values between 1 and 13')

                # Read in RQI threshold to apply to radar products.
                if suppOpt in (1,2,7,10,11):
                    try:
                        self.rqiMethod = json.loads(config['SuppForcing']['RqiMethod'])
                    except KeyError:
                        err_handler.err_out_screen('Unable to locate RqiMethod under SuppForcing '
                                                   'section in the configuration file.')
                    except configparser.NoOptionError:
                        err_handler.err_out_screen('Unable to locate RqiMethod under SuppForcing '
                                                   'section in the configuration file.')
                    except json.decoder.JSONDecodeError:
                        err_handler.err_out_screen('Improper RqiMethod option in the configuration file.')

                    # Check that if we have more than one RqiMethod, it's the correct number
                    if type(self.rqiMethod) is list:
                        if len(self.rqiMethod) != self.number_supp_pcp:
                            err_handler.err_out_screen('Number of RqiMethods ({}) must match the number '
                                                       'of SuppPcp inputs ({}) in the configuration file, or '
                                                       'supply a single method for all inputs'.format(
                                                        len(self.rqiMethod), self.number_supp_pcp))
                    elif type(self.rqiMethod) is int:
                        # Support 'classic' mode of single method
                        self.rqiMethod = [self.rqiMethod] * self.number_supp_pcp

                    # Make sure the RqiMethod(s) makes sense.
                    for method in self.rqiMethod:
                        if method < 0 or method > 2:
                            err_handler.err_out_screen('Please specify RqiMethods of either 0, 1, or 2.')

                    try:
                        self.rqiThresh = json.loads(config['SuppForcing']['RqiThreshold'])
                    except KeyError:
                        err_handler.err_out_screen('Unable to locate RqiThreshold under '
                                                   'SuppForcing section in the configuration file.')
                    except configparser.NoOptionError:
                        err_handler.err_out_screen('Unable to locate RqiThreshold under '
                                                   'SuppForcing section in the configuration file.')
                    except json.decoder.JSONDecodeError:
                        err_handler.err_out_screen('Improper RqiThreshold option in the configuration file.')

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
                self.supp_precip_dirs = config.get('SuppForcing', 'SuppPcpDirectories').split(',')
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpDirectories in SuppForcing section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpDirectories in SuppForcing section '
                                           'in the configuration file.')

            # Loop through and ensure all supp pcp directories exist. Also strip out any whitespace
            # or new line characters.
            for dirTmp in range(0, len(self.supp_precip_dirs)):
                self.supp_precip_dirs[dirTmp] = self.supp_precip_dirs[dirTmp].strip()
                if not os.path.isdir(self.supp_precip_dirs[dirTmp]):
                    err_handler.err_out_screen('Unable to locate supp pcp directory: ' + self.supp_precip_dirs[dirTmp])

            #Special case for ExtAnA where we treat comma separated stage IV, MRMS data as one SuppPcp input
            if 11 in self.supp_precip_forcings:
                if len(self.supp_precip_forcings) != 1:
                    err_handler.err_out_screen('Alaska Stage IV/MRMS SuppPcp option is only supported as a standalone option')
                self.supp_precip_dirs = [",".join(self.supp_precip_dirs)]

            if len(self.supp_precip_dirs) != self.number_supp_pcp:
                err_handler.err_out_screen('Number of SuppPcpDirectories must match the number '
                                           'of SuppForcing in the configuration file.')

            # Process supplemental precipitation enforcement options
            try:
                self.supp_precip_mandatory = json.loads(config['SuppForcing']['SuppPcpMandatory'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpMandatory under the SuppForcing section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpMandatory under the SuppForcing section '
                                           'in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper SuppPcpMandatory options specified in the configuration file.')
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
                self.regrid_opt_supp_pcp = json.loads(config['SuppForcing']['RegridOptSuppPcp'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate RegridOptSuppPcp under the SuppForcing section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate RegridOptSuppPcp under the SuppForcing section '
                                           'in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper RegridOptSuppPcp options specified in the configuration file.')
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
                self.suppTemporalInterp = json.loads(config['SuppForcing']['SuppPcpTemporalInterpolation'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpTemporalInterpolation under the SuppForcing '
                                           'section in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpTemporalInterpolation under the SuppForcing '
                                           'section in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper SuppPcpTemporalInterpolation options specified in the '
                                           'configuration file.')
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
                self.supp_pcp_max_hours = json.loads(config['SuppForcing']['SuppPcpMaxHours'])
            except (KeyError, configparser.NoOptionError):
                self.supp_pcp_max_hours = None      # if missing, don't care, just assume all time

            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper SuppPcpMaxHours options specified in the '
                                           'configuration file.')

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
                self.supp_input_offsets = json.loads(config['SuppForcing']['SuppPcpInputOffsets'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpInputOffsets under SuppForcing '
                                           'section in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpInputOffsets under SuppForcing '
                                           'section in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper SuppPcpInputOffsets option specified in '
                                           'the configuration file.')
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
                self.supp_precip_param_dir = config['SuppForcing']['SuppPcpParamDir']
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpParamDir under the SuppForcing section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpParamDir under the SuppForcing section '
                                           'in the configuration file.')
            except ValueError:
                err_handler.err_out_screen('Improper SuppPcpParamDir option specified in the configuration file.')
            if not os.path.isdir(self.supp_precip_param_dir):
                err_handler.err_out_screen('Unable to locate SuppPcpParamDir: ' + self.supp_precip_param_dir)


        if self.precip_only_flag == False:
            # Read in Ensemble information
            # Read in CFS ensemble member information IF we have chosen CFSv2 as an input
            # forcing.
            for optTmp in self.input_forcings:
                if optTmp == 7:
                    try:
                        self.cfsv2EnsMember = json.loads(config['Ensembles']['cfsEnsNumber'])
                    except KeyError:
                        err_handler.err_out_screen('Unable to locate cfsEnsNumber under the Ensembles '
                                                   'section of the configuration file')
                    except configparser.NoOptionError:
                        err_handler.err_out_screen('Unable to locate cfsEnsNumber under the Ensembles '
                                                   'section of the configuration file')
                    except json.JSONDecodeError:
                        err_handler.err_out_screen('Improper cfsEnsNumber options specified in the configuration file')
                    if self.cfsv2EnsMember < 1 or self.cfsv2EnsMember > 4:
                        err_handler.err_out_screen('Please chose an cfsEnsNumber value of 1,2,3 or 4.')

            # Read in information for the custom input NetCDF files that are to be processed.
            # Read in the ForecastInputHorizons options.
            try:
                self.customFcstFreq = json.loads(config['Custom']['custom_input_fcst_freq'])
            except KeyError:
                err_handler.err_out_screen('Unable to locate custom_input_fcst_freq under Custom section in '
                                           'configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate custom_input_fcst_freq under Custom section in '
                                           'configuration file.')
            except json.decoder.JSONDecodeError as je:
                err_handler.err_out_screen('Improper custom_input_fcst_freq  option specified in '
                                           'configuration file: ' + str(je))
            if len(self.customFcstFreq) != self.number_custom_inputs:
                err_handler.err_out_screen(f'Improper custom_input fcst_freq specified. '
                                           f'This number ({len(self.customFcstFreq)}) must '
                                           f'match the frequency of custom input forcings selected '
                                           f'({self.number_custom_inputs}).')

    @property
    def use_data_at_current_time(self):
        if self.supp_pcp_max_hours is not None:
            hrs_since_start = self.current_output_date - self.current_fcst_cycle
            return hrs_since_start <= datetime.timedelta(hours = self.supp_pcp_max_hours)
        else:
            return True
