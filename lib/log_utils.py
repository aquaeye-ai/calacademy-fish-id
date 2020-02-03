"""
Class for handling logging tasks
"""

# standard libs
import time
import logging
import functools

# personal libs
import lib.globals as globals

logger = logging.getLogger(__name__)


QUIET = False


def init_logging(file_name="train.log"):
    """
    Initializes the logger and associated logging directory.  Requires func: make_hparam_string to be called prior to
    invocation.

    :param classifier:  str, name of classifier being used, e.g. 'any_boundary_classifier'
    :return: None
    """

    # create the file to log to
    file_path = globals.log_dir + '/' + file_name

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(name)s %(filename)s:%(lineno)d - %(funcName)s] %(levelname)s %(message)s',
                        filename=file_path,
                        filemode='w')

    # define a console handler which writes INFO messages or higher to the sys.stderr
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # set a console formatter
    cf = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    # add the handler to the logger
    logging.getLogger('').addHandler(ch)

    # assign the console formatter to the console
    ch.setFormatter(cf)


def timeit(func):
    """
    Timing decorator to time functions

    :param func:    function to time
    :return:        func parameter wrapped in a timing function
    """

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        res = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        if not QUIET:
            logger.info('function [{}] finished in {} ms'.format(func.__name__, int(elapsedTime * 1000)))
        return res

    return newfunc


