"""
A module defining important utility functions that are rather
uninteresting.

.. author:: Malcolm C. A. White
.. date:: 2020-04-17
"""

import argparse
import configparser
import logging
import signal


def configure_logger(name, logfile, processor_name, rank, verbose=False):
    """
    A utility function to configure logging. Return True on successful
    execution.
    """

    # Define the date format for logging.
    datefmt ="%Y%jT%H:%M:%S"

    if verbose is True:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if level == logging.DEBUG:
        fmt = f"%(asctime)s::%(levelname)s::{name}.%(funcName)s()::%(lineno)d::"\
              f"{processor_name}::{rank:04d}:: %(message)s"
    else:
        fmt = f"%(asctime)s::%(levelname)s:: %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return (True)


def get_logger(name):
    """
    Return the logger for *name*.
    """

    return (logging.getLogger(name))


def log_errors(logger):
    """
    A decorator to for error logging.
    """

    def _decorate_func(func):
        """
        An hidden decorator to permit the logger to be passed in as a
        decorator argument.
        """

        def _decorated_func(*args, **kwargs):
            """
            The decorated function.
            """
            try:
                return (func(*args, **kwargs))
            except Exception as exc:
                logger.error(
                    f"{func.__name__}() raised {type(exc)}: {exc}"
                )
                raise (exc)

        return (_decorated_func)

    return (_decorate_func)


def parse_args():
    """
    Parse and return command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "events",
        type=str,
        help="Input event (origins and phases) data file in HDF5 format."
    )
    parser.add_argument(
        "network",
        type=str,
        help="Input network geometry file in HDF5 format."
    )
    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        default="vorotomo.log",
        help="Log file."
    )
    parser.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default="vorotomo.cfg",
        help="Configuration file."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Be verbose."
    )

    return (parser.parse_args())


def parse_cfg(configuration_file):
    """
    Parse and return contents of the configuration file.
    """

    cfg = dict()
    parser = configparser.ConfigParser()
    parser.read(configuration_file)
    _cfg = dict()
    _cfg["niter"] = parser.getint(
        "algorithm",
        "niter",
        fallback=1
    )
    _cfg["nreal"] = parser.getint(
        "algorithm",
        "nreal"
    )
    _cfg["nvoronoi"] = parser.getint(
        "algorithm",
        "nvoronoi"
    )
    _cfg["nphase"] = parser.getint(
        "algorithm",
        "nphase"
    )
    cfg["algorithm"] = _cfg

    _cfg = dict()
    _cfg["lat_min"] = parser.getfloat(
        "model",
        "lat_min"
    )
    _cfg["lon_min"] = parser.getfloat(
        "model",
        "lon_min"
    )
    _cfg["depth_min"] = parser.getfloat(
        "model",
        "depth_min"
    )
    _cfg["nlat"] = parser.getint(
        "model",
        "nlat"
    )
    _cfg["nlon"] = parser.getint(
        "model",
        "nlon"
    )
    _cfg["ndepth"] = parser.getint(
        "model",
        "ndepth"
    )
    _cfg["dlat"] = parser.getfloat(
        "model",
        "dlat"
    )
    _cfg["dlon"] = parser.getfloat(
        "model",
        "dlon"
    )
    _cfg["ddepth"] = parser.getfloat(
        "model",
        "ddepth"
    )
    _cfg["initial_pwave_path"] = parser.get(
        "model",
        "initial_pwave_path"
    )
    _cfg["initial_swave_path"] = parser.get(
        "model",
        "initial_swave_path"
    )
    cfg["model"] = _cfg

    return (cfg)



@log_errors
def signal_handler(sig, frame):
    """
    A utility function to to handle interrupting signals.
    """

    raise (SystemError("Interrupting signal received... aborting"))
