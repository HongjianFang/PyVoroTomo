"""
A module defining important utility functions that are rather
uninteresting.

.. author:: Malcolm C. A. White
.. date:: 2020-04-17
"""

import argparse
import configparser
import logging
import mpi4py.MPI as MPI
import signal
import time

import _constants

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def configure_logger(name, logfile, verbose=False):
    """
    A utility function to configure logging. Return True on successful
    execution.
    """

    # Define the date format for logging.
    datefmt        ="%Y%jT%H:%M:%S"
    processor_name = MPI.Get_processor_name()
    rank           = MPI.COMM_WORLD.Get_rank()

    if verbose is True:
        level = logging.DEBUG
    else:
        level = logging.INFO if rank == _constants.ROOT_RANK else logging.WARNING
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if level == logging.DEBUG:
        fmt = f"%(asctime)s::%(levelname)s::{name}.%(funcName)s()::%(lineno)d::"\
              f"{processor_name}::{rank:04d}:: %(message)s"
    else:
        fmt = f"%(asctime)s::%(levelname)s::{rank:04d}:: %(message)s"
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
        "-c",
        "--configuration_file",
        type=str,
        default="vorotomo.cfg",
        help="Configuration file."
    )
    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        default="vorotomo.log",
        help="Log file."
    )
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"output_{stamp}",
        help="Output directory."
    )
    parser.add_argument(
        "-s",
        "--scratch_dir",
        type=str,
        help="Scratch directory."
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
    _cfg["adaptive_voronoi_cells"] = parser.getboolean(
        "algorithm",
        "adaptive_voronoi_cells",
        fallback=True
    )
    _cfg["narrival"] = parser.getint(
        "algorithm",
        "narrival"
    )
    _cfg["homogenize_raypaths"] = parser.getboolean(
        "algorithm",
        "homogenize_raypaths",
        fallback=True
    )
    _cfg["n_raypath_bins"] = parser.getint(
        "algorithm",
        "n_raypath_bins"
    )
    _cfg["outlier_removal_factor"] = parser.getfloat(
        "algorithm",
        "outlier_removal_factor"
    )
    _cfg["damp"] = parser.getfloat(
        "algorithm",
        "damp"
    )
    _cfg["atol"] = parser.getfloat(
        "algorithm",
        "atol"
    )
    _cfg["btol"] = parser.getfloat(
        "algorithm",
        "btol"
    )
    _cfg["conlim"] = parser.getfloat(
        "algorithm",
        "conlim"
    )
    _cfg["maxiter"] = parser.getfloat(
        "algorithm",
        "maxiter"
    )
    cfg["algorithm"] = _cfg

    _cfg = dict()
    _cfg["initial_pwave_path"] = parser.get(
        "model",
        "initial_pwave_path"
    )
    _cfg["initial_swave_path"] = parser.get(
        "model",
        "initial_swave_path"
    )
    cfg["model"] = _cfg

    _cfg = dict()
    _cfg["dlat"] = parser.getfloat(
        "locate",
        "dlat"
    )
    _cfg["dlon"] = parser.getfloat(
        "locate",
        "dlon"
    )
    _cfg["ddepth"] = parser.getfloat(
        "locate",
        "ddepth"
    )
    _cfg["dtime"] = parser.getfloat(
        "locate",
        "dtime"
    )
    cfg["locate"] = _cfg

    return (cfg)


def root_only(rank, default=True, barrier=True):
    """
    A decorator for functions and methods that only the root rank should
    execute.
    """

    def _decorate_func(func):
        """
        An hidden decorator to permit the rank to be passed in as a
        decorator argument.
        """

        def _decorated_func(*args, **kwargs):
            """
            The decorated function.
            """
            if rank == _constants.ROOT_RANK:
                value = func(*args, **kwargs)
                if barrier is True:
                    COMM.barrier()
                return (value)
            else:
                if barrier is True:
                    COMM.barrier()
                return (default)

        return (_decorated_func)

    return (_decorate_func)


@log_errors
def signal_handler(sig, frame):
    """
    A utility function to to handle interrupting signals.
    """

    raise (SystemError("Interrupting signal received... aborting"))
