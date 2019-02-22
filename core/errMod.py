import sys

def err_out_screen(err_msg):
    """
    Generic routine to exit the program gracefully. This specific error function does not log
    error messages to the log file, but simply prints them out to the screen. This is function
    is designed specifically for early in the program execution where a log file hasn't been
    established yet.
    Logan Karsten - National Center for Atmospheric Research, karsten@ucar.edu
    """

    err_msg_out = 'ERROR: ' + err_msg
    print(err_msg_out)
    sys.exit(-1)
