import argparse
import configparser
import logging
import mpi4py.MPI
import numpy as np
import os
import pandas as pd
import pykonal
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import signal
import shutil
import tempfile
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

PROCESSOR_NAME = mpi4py.MPI.Get_processor_name()
COMM = mpi4py.MPI.COMM_WORLD
WORLD_SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
ROOT_RANK: int = 0
EARTH_RADIUS: float = 6371.
DTYPE_REAL = np.float64


class VelocityModel(object):
    """
    A simple container class.
    """

    def __init__(self, grid, velocity):
        self.grid = grid
        self.velocity = velocity

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'event_file',
        type=str,
        help='Event and arrival data in pandas.HDFStore format.'
    )
    parser.add_argument(
        'network_file',
        type=str,
        help='Network geometry in pandas.HDFStore format.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        '-w',
        '--working_dir',
        default=os.path.abspath('.'),
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        '-c',
        '--config_file',
        default='pyvorotomo.cfg',
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        "-l",
        "--logfile",
        type=str,
        help="log file"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose"
    )
    return (parser.parse_args())


def load_params(argc):
    """
    Parse and return parameter file contents.
    """
    params = dict()
    parser = configparser.RawConfigParser()
    with open(argc.config_file) as inf:
        parser.read_file(inf)
    params['latmin'] = parser.getfloat('Model Parameters', 'latmin')
    params['lonmin'] = parser.getfloat('Model Parameters', 'lonmin')
    params['depmin'] = parser.getfloat('Model Parameters', 'depmin')
    params['nlat'] = parser.getint('Model Parameters', 'nlat')
    params['nlon'] = parser.getint('Model Parameters', 'nlon')
    params['nrad'] = parser.getint('Model Parameters', 'nrad')
    params['dlat'] = parser.getfloat('Model Parameters', 'dlat')
    params['dlon'] = parser.getfloat('Model Parameters', 'dlon')
    params['drad'] = parser.getfloat('Model Parameters', 'drad')
    params['initial_vmodel_p'] = parser.get('Model Parameters', 'initial_vmodel_p')
    params['initial_vmodel_s'] = parser.get('Model Parameters', 'initial_vmodel_s')
    params['nsamp'] = parser.getint('Data Sampling Parameters', 'nsamp')
    params['ncell'] = parser.getint('Data Sampling Parameters', 'ncell')
    params['nreal'] = parser.getint('Data Sampling Parameters', 'nreal')
    params['niter'] = parser.getint('Data Sampling Parameters', 'niter')
    params['atol'] = parser.getfloat('Convergence Parameters', 'atol')
    params['btol'] = parser.getfloat('Convergence Parameters', 'btol')
    params['maxiter'] = parser.getint('Convergence Parameters', 'maxiter')
    params['conlim'] = parser.getint('Convergence Parameters', 'conlim')
    params['damp'] = parser.getfloat('Convergence Parameters', 'damp')
    return (params)


###############################################################################
# Geometry convenience functions.

def geo2sph(arr):
    """
    Map Geographical coordinates to spherical coordinates.
    """
    geo = np.array(arr, dtype=DTYPE_REAL)
    sph = np.empty_like(geo)
    sph[..., 0] = EARTH_RADIUS - geo[..., 2]
    sph[..., 1] = np.pi / 2 - np.radians(geo[..., 0])
    sph[..., 2] = np.radians(geo[..., 1])
    return (sph)


def sph2geo(arr):
    """
    Map spherical coordinates to geographic coordinates.
    """
    sph = np.array(arr, dtype=DTYPE_REAL)
    geo = np.empty_like(sph)
    geo[..., 0] = np.degrees(np.pi / 2 - sph[..., 1])
    geo[..., 1] = np.degrees(sph[..., 2])
    geo[..., 2] = EARTH_RADIUS - sph[..., 0]
    return (geo)


def sph2xyz(arr):
    """
    Map spherical coordinates to Cartesian coordinates.
    """
    sph = np.array(arr, dtype=DTYPE_REAL)
    xyz = np.empty_like(sph)
    xyz[..., 0] = sph[..., 0] * np.sin(sph[..., 1]) * np.cos(sph[..., 2])
    xyz[..., 1] = sph[..., 0] * np.sin(sph[..., 1]) * np.sin(sph[..., 2])
    xyz[..., 2] = sph[..., 0] * np.cos(sph[..., 1])
    return (xyz)


###############################################################################


def configure_logging(verbose, logfile):
    """
    A utility function to configure logging.
    """
    if verbose is True:
        level = logging.DEBUG
    else:
        level = logging.INFO
    for name in (__name__,):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if level == logging.DEBUG:
            formatter = logging.Formatter(fmt="%(asctime)s::%(levelname)s::"\
                    "%(funcName)s()::%(lineno)d::{:s}::{:04d}:: "\
                    "%(message)s".format(PROCESSOR_NAME, RANK),
                    datefmt="%Y%jT%H:%M:%S")
        else:
            formatter = logging.Formatter(fmt="%(asctime)s::%(levelname)s::"\
                    "{:s}::{:04d}:: %(message)s".format(PROCESSOR_NAME, RANK),
                    datefmt="%Y%jT%H:%M:%S")
        if logfile is not None:
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


###############################################################################
def compute_traveltime_lookup_table(station, vmodel, wdir, tag=None):
    """
    Create and return the solved EikonalSolver for *station*.

    :param station:
    :param vmodel:
    :param wdir:
    :param tag:
    :return:
    """
    logger.debug(f"Computing traveltime-lookup table for station {station['station_id']}")
    try:
        _compute_traveltime_lookup_table(station, vmodel, wdir, tag=None)
    except Exception as exc:
        logger.error(exc)
        raise

def _compute_traveltime_lookup_table(station, vmodel, wdir, tag=None):
    rho0, theta0, phi0 = geo2sph(station[['lat', 'lon', 'depth']].values)
    far_field = init_farfield(vmodel)
    near_field = init_nearfield(far_field, (rho0, theta0, phi0))
    near_field.solve()
    far_field.transfer_travel_times_from(near_field, (-rho0, theta0, phi0), set_alive=True)
    far_field.solve()
    station_id = station['station_id']
    np.savez_compressed(
        os.path.join(
            wdir,
            f'{station_id}.npz' if tag is None else f'{station_id}.{tag}.npz'
        ),
        uu=far_field.uu,
        min_coords=far_field.pgrid.min_coords,
        node_intervals=far_field.pgrid.node_intervals,
        npts=far_field.pgrid.npts
    )
    return (far_field)


def compute_traveltime_lookup_tables(payload, phase, wdir):
    """
    Compute and store traveltime lookup tables.

    :param payload: Data payload containing df_events, df_arrivals,
        df_stations, vmodel_p, vmodel_s.
    :param phase: Phase to update velocity model for.
    :param wdir: Temporary working directory.
    :type wdir: tempfile.TemporaryDirectory
    """
    if RANK == ROOT_RANK:
        logger.debug(f'Computing {phase}-wave traveltime-lookup tables.')
    try:
        _compute_traveltime_lookup_tables(payload, phase, wdir)
    except Exception as exc:
        logger.error(exc)
        raise


def _compute_traveltime_lookup_tables(payload, phase, wdir):
    df_stations = COMM.scatter(
        None if RANK is not ROOT_RANK else np.array_split(
            payload['df_stations'].sample(n=len(payload['df_stations'])),
            WORLD_SIZE
        ),
        root=ROOT_RANK
    )
    vmodel = payload['vmodel_p'] if phase == 'P' else payload['vmodel_s']
    for idx, station in df_stations.iterrows():
        compute_traveltime_lookup_table(station, vmodel, wdir)


def init_farfield(vmodel):
    """
    Initialize the far-field EikonalSolver with the given velocity model.
    """
    far_field = pykonal.EikonalSolver(coord_sys='spherical')
    far_field.vgrid.min_coords = vmodel.grid.min_coords
    far_field.vgrid.node_intervals = vmodel.grid.node_intervals
    far_field.vgrid.npts = vmodel.grid.npts
    far_field.vv = vmodel.velocity
    return (far_field)


def init_nearfield(far_field, origin):
    """
    Initialize the near-field EikonalSolver.

    :param origin: Station location in spherical coordinates.
    :type origin: (float, float, float)

    :return: Near-field EikonalSolver
    :rtype: pykonal.EikonalSolver
    """
    drho = far_field.vgrid.node_intervals[0] / 5
    near_field = pykonal.EikonalSolver(coord_sys='spherical')
    near_field.vgrid.min_coords = drho, 0, 0
    near_field.vgrid.node_intervals = drho, np.pi / 20, np.pi / 20
    near_field.vgrid.npts = 100, 21, 40
    near_field.transfer_velocity_from(far_field, origin)
    vvi = pykonal.LinearInterpolator3D(near_field.vgrid, near_field.vv)

    for it in range(near_field.pgrid.npts[1]):
        for ip in range(near_field.pgrid.npts[2]):
            idx = (0, it, ip)
            near_field.uu[idx] = near_field.pgrid[idx + (0,)] / vvi(near_field.pgrid[idx])
            near_field.is_far[idx] = False
            near_field.close.push(*idx)
    return (near_field)


def main(argc, params):
    """
    Main loop.

    :param argc: Command line arguments.
    :param params: Configuration file parameters.
    """
    # Load payload in root thread.
    if RANK == ROOT_RANK:
        payload = load_payload(argc, params)
    # Broadcast payload to all threads.
    payload = COMM.bcast(
        None if RANK is not ROOT_RANK else payload,
        root=ROOT_RANK
    )
    # Iterate the inversion process.
    for iiter in range(params['niter']):
        iterate_inversion(payload, argc, params, iiter)


def iterate_inversion(payload, argc, params, iiter):
    """
    Iterate the entire inversion process.

    :param payload: Data payload containing df_events, df_arrivals,
        df_stations, vmodel_p, vmodel_s.
    :param argc: Command-line arguments.
    :param params: Configuration-file parameters.
    :param iiter: Iteration counter.
    """
    # Update P-wave velocity model.
    wdir = COMM.bcast(
        None if RANK is not ROOT_RANK else tempfile.mkdtemp(dir=argc.working_dir),
        root=ROOT_RANK
    )
    payload['vmodel_p'] = update_velocity_model(payload, 'P', wdir)
    if RANK == ROOT_RANK:
        logger.debug(f'Removing working directory {wdir}')
        shutil.rmtree(wdir)
    # TODO:: Update S-wave velocity model.
    # TODO:: Update event locations.


def load_event_data(argc, params):
    """
    Read and return *events* and *arrivals* tables from pandas.HDFStore.

    :param argc: Command-line arguments.
    :param params: Configuration-file parameters.
    """
    with pd.HDFStore(argc.event_file) as store:
        df_events = store['events']
        df_arrivals = store['arrivals']
    return (df_events, df_arrivals)


def load_initial_velocity_model(params, phase):
    """
    Load the initial velocity model.

    :param params: Configuration-file parametesr.
    :param phase: Phase to load initial velocity model for.
    :return:
    """
    grid = pykonal.Grid3D(coord_sys='spherical')
    grid.min_coords = geo2sph(
        (
            params['latmin'] + (params['nlat'] - 1) * params['dlat'],
            params['lonmin'],
            params['depmin'] + (params['nrad'] - 1) * params['drad']
        )
    )
    grid.node_intervals = (
        params['drad'], np.radians(params['dlat']), np.radians(params['dlon'])
    )
    grid.npts = params['nrad'], params['nlat'], params['nlon']
    if phase == 'P':
        if params['initial_vmodel_p'] in ('None', ''):
            velocity = 6. * np.ones(grid.npts)
            vmodel = VelocityModel(grid, velocity)
        else:
            vmodel = load_velocity_from_file(params['initial_vmodel_p'])
    elif phase == 'S':
        if params['initial_vmodel_s'] in ('None', ''):
            velocity = 3.74 * np.ones(grid.npts)
            vmodel = VelocityModel(grid, velocity)
        else:
            vmodel = load_velocity_from_file(params['initial_vmodel_s'])
    else:
        raise (NotImplementedError(f'Did not understand phase type: {phase}'))
    return (vmodel)


def load_network_data(argc):
    """
    Read and return *stations* table from pandas.HDFStore.

    :param argc: Command-line arguments.
    """
    with pd.HDFStore(argc.network_file) as store:
        df_stations = store['stations']
    df_stations['depth'] = df_stations['elev'] * -1
    return (df_stations)


def load_payload(argc, params):
    """
    Return payload containing df_events, df_arrivals, df_stations, vmodel_p, vmodel_s.

    :param argc: Command-line arguments.
    :params params: Configuration file parameters.
    """
    # Load event and network data.
    df_events, df_arrivals = load_event_data(argc, params)
    df_stations = load_network_data(argc)
    df_arrivals = sanitize_arrivals(df_arrivals, df_stations)
    # Load initial velocity model.
    vmodel_p = load_initial_velocity_model(params, 'P')
    vmodel_s = load_initial_velocity_model(params, 'S')
    payload = dict(
        df_events=df_events,
        df_arrivals=df_arrivals,
        df_stations=df_stations,
        vmodel_p=vmodel_p,
        vmodel_s=vmodel_s
    )
    return (payload)


def load_velocity_from_file(fname):
    """
    Load the velocity model in file *fname*.

    :param fname: The file name of the velocity model to load.
    :type fname: str
    :return: Velocity model saved in file *fname*.
    :rtype: VelocityModel
    """
    grid = pykonal.Grid3D(coord_sys='spherical')
    with np.load(fname) as infile:
        grid.min_coords     = infile['min_coords']
        grid.node_intervals = infile['node_intervals']
        grid.npts           = infile['npts']
        vmodel              = VelocityModel(grid=grid, velocity=infile['vv'])
    return (vmodel)


def sanitize_arrivals(df_arrivals, df_stations):
    """
    Remove arrivals at stations without metadata.

    :return: Arrivals at stations with metadata.
    :rtype: pandas.DataFrame
    """
    return (
        df_arrivals[
            df_arrivals['station_id'].isin(df_stations['station_id'].unique())
        ]
    )


def update_velocity_model(payload, phase, wdir):
    """
    Return update velocity model.

    :param payload: Data payload containing df_events, df_arrivals,
        df_stations, vmodel_p, vmodel_s.
    :param phase: Phase to update velocity model for.
    :param wdir: Temporary working directory.
    :type wdir: tempfile.TemporaryDirectory
    """
    if RANK == ROOT_RANK:
        logger.debug(f'Updating {phase}-wave velocity model')
    # Compute traveltime-lookup tables
    compute_traveltime_lookup_tables(payload, phase, wdir)

###############################################################################


def signal_handler(sig, frame):
    raise(SystemError("Interrupting signal received... aborting"))


if __name__ == '__main__':
    # Add some signal handlers to abort all threads.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGCONT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Parse command line arguments.
    argc = parse_args()
    # Load parameters for parameter file.
    params = load_params(argc)
    # Configure logging.
    configure_logging(argc.verbose, argc.logfile)
    logger = logging.getLogger(__name__)
    # Start the  main loop.
    main(argc, params)
