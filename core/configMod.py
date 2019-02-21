import configparser
from core import errmod
import json
import datetime
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
        self.b_date_proc = None
        self.e_date_proc = None
        self.process_window = None
        self.look_back = None
        self.fcst_freq = None
        self.fcst_intervals = None
        self.geogrid = None
        self.regrid_opt = None
        self.downscale_param_dir = None
        self.supp_pcp_forcings = None
        self.config_path = config
        self.realtime = False
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

        # Process the beginning date of forcings to process.
        try:
            beg_date_tmp = config['Logistics']['BDateProc']
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

        # Process the ending date of forcings to process
        try:
            end_date_tmp = config['Logistics']['EDateProc']
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
                errmod.err_out_screen('If choosing NWM configuration, all lookup dates must be -9999')
            if self.e_date_proc <= self.b_date_proc:
                errmod.err_out_screen('Please choose an ending EDateProc that is greater'
                                      ' than BDateProc.')
            # Calculate the delta time between the beginning and ending time of processing.
            self.process_window = self.e_date_proc - self.b_date_proc
        else:
            self.e_date_proc = -9999
        if self.e_date_proc == -9999 and self.b_date_proc != -9999:
            errmod.err_out_screen('If choosing NWM configuration, all lookup dates must be -9999')

        # Process the optional lookback hours for forecast configurations. If
        # the user specifies a lookback period, this will override the beginning
        # and ending dates.
        try:
            self.look_back = int(config['Logistics']['LookBack'])
            if self.b_date_proc == -9999 and self.e_date_proc == -9999 and self.look_back_hours == -9999:
                errmod.err_out_screen('Please specify either a date range or a set of lookup'
                                      ' minutes for realtime forecasts.')
            if self.look_back <= 0 and self.look_back != -9999:
                errmod.err_out_screen('Please specify a positive LookBack or -9999 if '
                                      'using a date range.')
        except ValueError:
            errmod.err_out_screen('Improper LookBack value entered into the '
                                  'configuration file. Please check your entry.')
        except KeyError:
            errmod.err_out_screen('Unable to locate LookBack in the configuration '
                                  'file. Please verify entries exist.')

        # If we have specified lookup hours, we are running in a realtime forecasting mode.
        if self.look_back > 0:
            self.realtime = True

        # Read in the ForecastFrequency option.
        try:
            self.fcst_freq = int(config['Logistics']['ForecastFrequency'])
        except ValueError:
            errmod.err_out_screen('Improper ForecastFrequency value entered into '
                                  'the configuration file. Please check your entry.')
        except KeyError:
            errmod.err_out_screen('Unable to locate ForecastFrequency in the configuration '
                                  'file. Please verify entries exist.')
        if self.fcst_freq <= 0:
            errmod.err_out_screen('Please specify a ForecastFrequency in the configuration '
                                  'file greater than zero.')
        if self.look_back != -9999:
            # Make sure ForecastFrequency is an even divider of LookBackHours.
            if self.look_back%self.fcst_freq != 0:
                errmod.err_out_screen('Please choose a ForecastFrequency that is an equal divider '
                                      'of LookBack')
            try:
                dateMod.calculate_lookback_window(ConfigOptions)
            except:
                errmod.err_out_screen(self.errMsg)

        # Calculate the number of forecast cycles to loop through.
        if self.fcst_freq > 0:
            self.fcst_intervals = int(self.look_back/self.fcst_freq)
        else:
            self.fcst_intervals = int(se)
        # Read in the ForecastIntervals
