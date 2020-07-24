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
import os
import signal
import time

from . import _constants

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def abort():
    COMM.Abort()


def configure_logger(name, log_file, verbose=False):
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
        fmt = f"%(asctime)s::%(levelname)s::%(funcName)s()::"\
              f"{processor_name}::{rank:04d}:: %(message)s"
    else:
        fmt = f"%(asctime)s::%(levelname)s::{rank:04d}:: %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
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


class ArgumentParser(argparse.ArgumentParser):
    """
    A simple subclass to abort all threads if argument parsing fails.
    """

    def exit(self, status=0, message=None):

        self.print_usage()

        if message is not None:
            print(message)

        abort()


def parse_args():
    """
    Parse and return command line arguments.
    """

    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    parser = ArgumentParser()
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
        default=f"{parser.prog}.cfg",
        help="Configuration file."
    )
    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        help="Log file."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"output_{stamp}",
        help="Output directory."
    )
    parser.add_argument(
        "-r",
        "--relocate_first",
        action="store_true",
        help="Relocate events before first model update."
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
        help="Verbose logging."
    )
    parser.add_argument(
        "-x",
        "--output_realizations",
        action="store_true",
        help="Save realizations to disk."
    )


    args = parser.parse_args()

    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, f"{parser.prog}.log")

    for attr in (
        "events",
        "network",
        "configuration_file",
        "log_file",
        "output_dir"
    ):
        _attr = getattr(args, attr)
        _attr = os.path.abspath(_attr)
        setattr(args, attr, _attr)


    if RANK == _constants.ROOT_RANK:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.scratch_dir is not None:
        args.scratch_dir = os.path.abspath(args.scratch_dir)
        if RANK == _constants.ROOT_RANK:
            os.makedirs(args.scratch_dir, exist_ok=True)

    COMM.barrier()

    return (args)


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
    _cfg["kvoronoi"] = parser.getint(
        "algorithm",
        "kvoronoi"
    )
    _cfg["nvoronoi"] = parser.getint(
        "algorithm",
        "nvoronoi"
    )
    _cfg["hvr"] = [int(v) for v in parser.get(
            "algorithm",
            "hvr"
        ).split(",")
    ]
    _cfg["nreal"] = parser.getint(
        "algorithm",
        "nreal"
    )
    _cfg["k_medians_npts"] = parser.getint(
        "algorithm",
        "k_medians_npts"
    )
    _cfg["min_narrival"] = parser.getint(
        "algorithm",
        "min_narrival"
    )
    _cfg["narrival"] = parser.getint(
        "algorithm",
        "narrival"
    )
    _cfg["nevent"] = parser.getint(
        "algorithm",
        "nevent"
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

    initial_pwave_path = parser.get(
        "model",
        "initial_pwave_path"
    )
    initial_pwave_path = os.path.abspath(initial_pwave_path)
    _cfg["initial_pwave_path"] = initial_pwave_path

    initial_swave_path = parser.get(
        "model",
        "initial_swave_path"
    )
    initial_swave_path = os.path.abspath(initial_swave_path)
    _cfg["initial_swave_path"] = initial_swave_path
    cfg["model"] = _cfg

    _cfg = dict()
    _cfg["method"] = parser.get(
        "relocate",
        "method"
    ).upper()
    if _cfg["method"] == "LINEAR":
        # Parse parameters for linearized relocation.
        _cfg["atol"] = parser.getfloat(
            "linearized_relocation",
            "atol"
        )
        _cfg["btol"] = parser.getfloat(
            "linearized_relocation",
            "btol"
        )
        _cfg["maxiter"] = parser.getint(
            "linearized_relocation",
            "maxiter"
        )
        _cfg["conlim"] = parser.getint(
            "linearized_relocation",
            "conlim"
        )
        _cfg["damp"] = parser.getfloat(
            "linearized_relocation",
            "damp"
        )
    elif _cfg["method"].upper() == "DE":
        # Parse parameters for DE relocation.
        _cfg["depth_min"] = parser.getfloat(
            "de_relocation",
            "depth_min"
        )
        _cfg["dlat"] = parser.getfloat(
            "de_relocation",
            "dlat"
        )
        _cfg["dlon"] = parser.getfloat(
            "de_relocation",
            "dlon"
        )
        _cfg["ddepth"] = parser.getfloat(
            "de_relocation",
            "ddepth"
        )
        _cfg["dtime"] = parser.getfloat(
            "de_relocation",
            "dtime"
        )
    else:
        raise (
            ValueError(
                "Relocation method must be either \"linear\" or \"DE\"."
            )
        )
    cfg["relocate"] = _cfg

    return (cfg)


def signal_handler(sig, frame):
    """
    A utility function to to handle interrupting signals.
    """

    raise (SystemError("Interrupting signal received... aborting"))


def write_cfg(argc, cfg):
    """
    Write the execution configuration to disk for later reference.
    """

    output_dir = argc.output_dir

    parser = configparser.ConfigParser()
    argc = vars(argc)
    argc = {key: str(argc[key]) for key in argc}
    cfg["argc"] = argc
    parser.read_dict(cfg)
    path = os.path.join(output_dir, "pyvorotomo.cfg")
    with open(path, "w") as configuration_file:
        parser.write(configuration_file)

    return (True)
