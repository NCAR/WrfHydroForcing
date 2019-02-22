import configparser
from core import errmod
import json
import datetime
import os
from core import dateMod


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
        self.number_inputs = None
        self.output_freq = None
        self.output_dir = None
        self.retro_flag = None
        self.realtime_flag = None
        self.refcst_flag = None
        self.b_date_proc = None
        self.e_date_proc = None
        self.look_back = None
        self.fcst_freq = None
        self.nFcsts = None
        self.fcst_shift = None
        self.fcst_input_horizons = None
        self.fcst_input_offsets = None
        self.process_window = None
        self.geogrid = None
        self.regrid_opt = None
        self.config_path = config
        self.errMsg = None

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
            errmod.err_out_screen('Unable to open the configuration file: ' + self.config_path)

        # Read in the base input forcing options as an array of values to map.
        try:
            self.input_forcings = json.loads(config['Input']['InputForcings'])
        except KeyError:
            errmod.err_out_screen('Unable to locate InputForcings under Input section in'
                                  'configuration file.')
        except json.decoder.JSONDecodeError:
            errmod.err_out_screen('Improper InputForcings option specified in '
                                  'configuration file')
        if len(self.input_forcings) == 0:
            errmod.err_out_screen('Please choose at least one InputForcings dataset'
                                  ' to process')
        self.number_inputs = len(self.input_forcings)

        # Check to make sure forcing options make sense
        for forceOpt in self.input_forcings:
            if forceOpt < 0 or forceOpt > 10:
                errmod.err_out_screen('Please specify InputForcings values between '
                                      '1 and 10.')

        # Read in the output frequency
        try:
            self.output_freq = int(config['Output']['OutputFrequency'])
        except ValueError:
            errmod.err_out_screen('Improper OutputFrequency value specified'
                                  ' in the configuration file.')
        except KeyError:
            errmod.err_out_screen('Unable to locate OutputFrequency in '
                                  'the configuration file.')
        if self.output_freq <= 0:
            errmod.err_out_screen('Please specify an OutputFrequency that'
                                  ' is greater than zero minutes.')

        # Read in the output directory
        try:
            self.output_dir = config['Output']['OutDir']
        except ValueError:
            errmod.err_out_screen('Improper OutDir specified in the '
                                  'configuration file.')
        except KeyError:
            errmod.err_out_screen('Unable to locate OutDir in the '
                                  'configuration file.')
        if not os.path.isdir(self.output_dir):
            errmod.err_out_screen('Specified output directory: ' + \
                                  self.output_dir + ' not found.')

        # Read in retrospective options
        try:
            self.retro_flag = int(config['Retrospective']['RetroFlag'])
        except KeyError:
            errmod.err_out_screen('Unable to locate RetroFlag in the'
                                  'configuration file.')
        except ValueError:
            errmod.err_out_screen('Improper RetroFlag value ')
        if self.retro_flag < 0 or self.retro_flag > 1:
            errmod.err_out_screen('Please choose a RetroFlag value of '
                                  '0 or 1.')
            
        # Process the beginning date of forcings to process.
        if self.retro_flag == 1:
            self.realtime_flag = False
            self.refcst_flag = False
            print("We are running retrospective mode")
            try:
                beg_date_tmp = config['Retrospective']['BDateProc']
            except KeyError:
                errmod.err_out_screen('Unable to locate BDateProc under Logistics section in'
                                      'configuration file.')
            if beg_date_tmp != '-9999':
                if len(beg_date_tmp) != 12:
                    errmod.err_out_screen('Improper BDateProc length entered into the '
                                          'configuration file. Please check your entry.')
                try:
                    self.b_date_proc = datetime.datetime.strptime(beg_date_tmp, '%Y%m%d%H%M')
                except ValueError:
                    errmod.err_out_screen('Improper BDateProc value entered into the'
                                          ' configuration file. Please check your entry.')
            else:
                self.b_date_proc = -9999

            # Process the ending date of retrospective forcings to process
            try:
                end_date_tmp = config['Retrospective']['EDateProc']
            except KeyError:
                errmod.err_out_screen('Unable to locate EDateProc under Logistics section in'
                                      'configuration file.')
            if end_date_tmp != '-9999':
                if len(end_date_tmp) != 12:
                    errmod.err_out_screen('Improper EDateProc length entered into the'
                                          'configuration file. Please check your entry.')
                try:
                    self.e_date_proc = datetime.datetime.strptime(end_date_tmp, '%Y%m%d%H%M')
                except ValueError:
                    errmod.err_out_screen('Improper EDateProc value entered into the'
                                          ' configuration file. Please check your entry.')
                if self.b_date_proc == -9999 and self.e_date_proc != -9999:
                    errmod.err_out_screen('If choosing retrospective forecasting, dates must not be -9999')
                if self.e_date_proc <= self.b_date_proc:
                    errmod.err_out_screen('Please choose an ending EDateProc that is greater'
                                          ' than BDateProc.')
            else:
                self.e_date_proc = -9999
            if self.e_date_proc == -9999 and self.b_date_proc != -9999:
                errmod.err_out_screen('If choosing retrospective forcings, dates must not be -9999')

        # Process realtime or reforecasting options.
        if self.retro_flag == 0:
            # If the retro flag is off, we are assuming a realtime or reforecast simulation.
            try:
                self.look_back = int(config['Forecast']['LookBack'])
                if self.look_back <= 0 and self.look_back != -9999:
                    errmod.err_out_screen('Please specify a positive LookBack or -9999 for realtime.')
            except ValueError:
                errmod.err_out_screen('Improper LookBack value entered into the '
                                      'configuration file. Please check your entry.')
            except KeyError:
                errmod.err_out_screen('Unable to locate LookBack in the configuration '
                                      'file. Please verify entries exist.')

            # If the Retro flag is off, and lookback is off, then we assume we are
            # running a reforecast.
            if self.look_back == -9999:
                self.realtime_flag = False
                self.refcst_flag = True
                print("We are running re-forecast mode")
                try:
                    beg_date_tmp = config['Forecast']['RefcstBDateProc']
                except KeyError:
                    errmod.err_out_screen('Unable to locate RefcstBDateProc under Logistics section in'
                                          'configuration file.')
                if beg_date_tmp != '-9999':
                    if len(beg_date_tmp) != 12:
                        errmod.err_out_screen('Improper RefcstBDateProc length entered into the '
                                              'configuration file. Please check your entry.')
                    try:
                        self.b_date_proc = datetime.datetime.strptime(beg_date_tmp, '%Y%m%d%H%M')
                    except ValueError:
                        errmod.err_out_screen('Improper RefcstBDateProc value entered into the'
                                              ' configuration file. Please check your entry.')
                else:
                    # This is an error. The user MUST specify a date range if reforecasts.
                    errmod.err_out_screen('Please either specify a reforecast range, or change '
                                          'the configuration to process refrospective or realtime.')

                # Process the ending date of reforecast forcings to process
                try:
                    end_date_tmp = config['Forecast']['RefcstEDateProc']
                except KeyError:
                    errmod.err_out_screen('Unable to locate RefcstEDateProc under Logistics section in'
                                          'configuration file.')
                if end_date_tmp != '-9999':
                    if len(end_date_tmp) != 12:
                        errmod.err_out_screen('Improper RefcstEDateProc length entered into the'
                                              'configuration file. Please check your entry.')
                    try:
                        self.e_date_proc = datetime.datetime.strptime(end_date_tmp, '%Y%m%d%H%M')
                    except ValueError:
                        errmod.err_out_screen('Improper RefcstEDateProc value entered into the'
                                              ' configuration file. Please check your entry.')
                else:
                    # This is an error. The user MUST specify a date range if reforecasts.
                    errmod.err_out_screen('Please either specify a reforecast range, or change '
                                          'the configuration to process refrospective or realtime.')
                if self.e_date_proc <= self.b_date_proc:
                    errmod.err_out_screen('Please choose an ending RefcstEDateProc that is greater'
                                          ' than RefcstBDateProc.')

                # Calculate the number of forecasts to issue, and verify the user has chosen a
                # correct divider based on the dates
                #dtTmp = self.e_date_proc - self.b_date_proc
                #print(self.fcst_freq)
                #if (dtTmp.days*1440+dtTmp.seconds/60.0)%self.fcst_freq != 0:
                #    errmod.err_out_screen('Please choose an equal divider forecast frequency for your'
                #                          ' specified reforecast range.')
            else:
                # The processing window will be calculated based on current time and the
                # lookback option since this is a realtime instance.
                self.realtime_flag = True
                self.refcst_flag = False
                print("We are running realtime mode")
                self.b_date_proc = -9999
                self.e_date_proc = -9999

            # Calculate the delta time between the beginning and ending time of processing.
            #self.process_window = self.e_date_proc - self.b_date_proc

            # Read in the ForecastFrequency option.
            try:
                self.fcst_freq = int(config['Forecast']['ForecastFrequency'])
            except ValueError:
                errmod.err_out_screen('Improper ForecastFrequency value entered into '
                                      'the configuration file. Please check your entry.')
            except KeyError:
                errmod.err_out_screen('Unable to locate ForecastFrequency in the configuration '
                                      'file. Please verify entries exist.')
            if self.fcst_freq <= 0:
                errmod.err_out_screen('Please specify a ForecastFrequency in the configuration '
                                      'file greater than zero.')
            # Currently, we only support daily or sub-daily forecasts. Any other iterations should
            # be done using custom config files for each forecast cycle.
            if self.fcst_freq > 1440:
                errmod.err_out_screen('Only forecast cycles of daily or sub-daily are supported '
                                      'at this time')

            # Read in the ForecastShift option. This is ONLY done for the realtime instance as
            # it's used to calculate the beginning of the processing window.
            if self.realtime_flag:
                print("Reading in shifts")
                try:
                    self.fcst_shift = int(config['Forecast']['ForecastShift'])
                except ValueError:
                    errmod.err_out_screen('Improper ForecastShift value entered into the '
                                          'configuration file. Please check your entry.')
                except KeyError:
                    errmod.err_out_screen('Unable to locate ForecastShift in the configuration '
                                           'file. Please verify entries exist.')
                if self.fcst_shift < 0:
                    errmod.err_out_screen('Please specify a ForecastShift in the configuration '
                                          'file greater than or equal to zero.')

                # Calculate the beginning/ending processing dates if we are running realtime
                if self.realtime_flag:
                    dateMod.calculate_lookback_window(self)

            if self.refcst_flag:
                # Calculate the number of forecasts to issue, and verify the user has chosen a
                # correct divider based on the dates
                dtTmp = self.e_date_proc - self.b_date_proc
                if (dtTmp.days*1440+dtTmp.seconds/60.0)%self.fcst_freq != 0:
                    errmod.err_out_screen('Please choose an equal divider forecast frequency for your'
                                          ' specified reforecast range.')
                self.nFcsts = int((dtTmp.days*1440+dtTmp.seconds/60.0)/self.fcst_freq)

            # Read in the ForecastInputHorizons options.
            try:
                self.fcst_input_horizons = json.loads(config['Forecast']['ForecastInputHorizons'])
            except KeyError:
                errmod.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in'
                                      'configuration file.')
            except json.decoder.JSONDecodeError:
                errmod.err_out_screen('Improper ForecastInputHorizons option specified in '
                                      'configuration file')
            if len(self.fcst_input_horizons) != self.number_inputs:
                errmod.err_out_screen('Please specify ForecastInputHorizon values for'
                                      ' each corresponding input forcings for forecasts.')

            # Check to make sure the horizons options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for horizonOpt in self.fcst_input_horizons:
                if horizonOpt <= 0:
                    errmod.err_out_screen('Please specify ForecastInputHorizon values greater '
                                          'than zero.')

            # Read in the ForecastInputOffsets options.
            try:
                self.fcst_input_offsets = json.loads(config['Forecast']['ForecastInputOffsets'])
            except KeyError:
                errmod.err_out_screen('Unable to locate ForecastInputOffsets under Forecast'
                                      ' section in the configuration file.')
            except json.decoder.JSONDecodeError:
                errmod.err_out_screen('Improper ForecastInputOffsets option specified in '
                                      'the configuration file.')
            if len(self.fcst_input_offsets) != self.number_inputs:
                errmod.err_out_screen('Please specify ForecastInputOffset values for each'
                                      ' corresponding input forcings for forecasts.')
            # Check to make sure the input offset options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for inputOffset in self.fcst_input_offsets:
                if inputOffset < 0:
                    errmod.err_out_screen('Please specify ForecastInputOffset values greater '
                                          'than or equal to zero.')

            # CALCULATE B/E DATE PROC STUFF

        # Process geospatial information
        try:
            self.geogrid = config['Geospatial']['GeogridIn']
        except KeyError:
            errmod.err_out_screen('Unable to locate GeogridIn in the configuration file.')
        if not os.path.isfile(self.geogrid):
            errmod.err_out_screen('Unable to locate necessary geogrid file: ' + self.geogrid)

        # Process regridding options.
        try:
            self.regrid_opt = json.loads(config['Regridding']['RegridOpt'])
        except KeyError:
            errmod.err_out_screen('Unable to locate RegridOpt under the Regridding section '
                                  'in the configuration file.')
        except json.decoder.JSONDecodeError:
            errmod.err_out_screen('Improper RegridOpt options specified in the configuration file.')
        if len(self.regrid_opt) != self.number_inputs:
            errmod.err_out_screen('Please specify RegridOpt values for each corresponding input '
                                  'forcings in the configuration file.')
        # Check to make sure regridding options makes sense.
        for regridOpt in self.regrid_opt:
            if regridOpt < 1 or regridOpt > 3:
                errmod.err_out_screen('Invalid RegridOpt chosen in the configuration file. Please'
                                      ' choose a value of 1-3 for each corresponding input forcing.')

        # Calculate the beginning/ending processing dates if we are running realtime
        if self.realtime_flag:
            dateMod.calculate_lookback_window(self)

        print("Beginning of Processing Window: " + self.b_date_proc.strftime('%Y-%m-%d %H:%M'))
        print("Ending of Processing Window: " + self.e_date_proc.strftime('%Y-%m-%d %H:%M'))
        if not self.retro_flag:
            print("Number of forecast cycles to process: " + str(self.nFcsts))

        # PLUG FOR READING IN DOWNSCALING OPTIONS

        # PLUG FOR READING IN BIAS CORRECTION OPTIONS

        # PLUG FOR READING IN SUPPLEMENTAL PRECIP PRODUCTS

        # PLUG FOR READING IN ENSEMBLE INFORMATION
