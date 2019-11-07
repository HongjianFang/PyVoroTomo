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
import sys
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
DTYPE_INT = np.int32


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
    params['maxres'] = parser.getfloat('Data Sampling Parameters', 'maxres')
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
        return (_compute_traveltime_lookup_table(station, vmodel, wdir, tag=tag))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)

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
        return (_compute_traveltime_lookup_tables(payload, phase, wdir))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


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
        compute_traveltime_lookup_table(station, vmodel, wdir, tag=phase)


def cost_function(loc, *args):
    """
    A cost function for locating events.

    :param loc:
    :type loc:
    :param *args:
    :type *args:

    :return:
    :rtype:
    """
    lat, lon, depth, time = loc
    df_arrivals, tti = args
    try:
        residuals = np.array([
            tti[arrival['station_id']][arrival['phase']](geo2sph((lat, lon, depth)))
            + time
            - arrival['time']
            for idx, arrival in df_arrivals.iterrows()
        ])
    except pykonal.OutOfBoundsError:
        return (np.inf)
    return (np.median(np.abs(residuals)))


def find_ray_idx(ray, vcells):
    """
    Determine the index of the Voronoi cell hosting each point on
    the ray path.
    """
    try:
        return (_find_ray_idx(ray, vcells))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _find_ray_idx(ray, vcells):
    dist = scipy.spatial.distance.cdist(sph2xyz(ray), sph2xyz(vcells))
    argmin = np.argmin(dist, axis=1)
    idxs, counts = np.unique(argmin, return_counts=True)
    return (idxs, counts)

def generate_inversion_matrix(payload, params, phase, wdir, df_sample, vcells, G_proj):
    """
    Return the G matrix and data vector for the inverse problem: Gm = d.

    :param payload: Data payload containing df_events, df_arrivals, df_stations, vmodel_p, vmodel_s.
    :param params: Configuration-file parameters.
    :param phase: Phase being inverted.
    :param wdir: Temporary working directory.
    :param df_sample: Subset of arrivals to process.
    :param vcells: Voronoi-cell centers in spherical coordinates.
    :param G_proj: Projection matrix.
    :return: G matrix and data vector.
    """
    try:
        return (_generate_inversion_matrix(payload, params, phase, wdir, df_sample, vcells, G_proj))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)

def _generate_inversion_matrix(payload, params, phase, wdir, df_sample, vcells, G_proj):
    col_idx_proj = np.array([], dtype=DTYPE_INT)
    nsegs = np.array([], dtype=DTYPE_INT)
    residuals = np.array([], dtype=DTYPE_REAL)
    nonzero_proj = np.array([], dtype=DTYPE_REAL)
    # Scatter arrivals
    df_sample = COMM.scatter(np.array_split(df_sample, WORLD_SIZE), root=ROOT_RANK)
    df_events = payload['df_events'].set_index('event_id')
    for station_id in df_sample.index.unique(level='station_id'):
        fname = os.path.join(wdir, f'{station_id}.{phase}.npz')
        solver = load_solver_from_disk(fname)
        for event_id in df_sample.loc[station_id].index.values:
            logger.debug(f'{station_id}, {event_id} {df_sample.loc[(station_id, event_id), "phase"]}')
            ret = trace_ray(
                df_events.loc[event_id],
                df_sample.loc[(station_id, event_id)],
                solver,
                vcells
            )
            if ret is None:
                continue
            residual, _col_idx_proj, _nsegs, _nonzero_proj = ret
            if np.abs(residual) > params['maxres']:
                continue
            col_idx_proj = np.append(col_idx_proj, _col_idx_proj)
            nsegs = np.append(nsegs, _nsegs)
            nonzero_proj = np.append(nonzero_proj, _nonzero_proj)
            residuals = np.append(residuals, residual)
    col_idx_proj = COMM.gather(col_idx_proj, root=ROOT_RANK)
    residuals = COMM.gather(residuals, root=ROOT_RANK)
    nsegs = COMM.gather(nsegs, root=ROOT_RANK)
    nonzero_proj = COMM.gather(nonzero_proj, root=ROOT_RANK)
    if RANK == ROOT_RANK:
        col_idx_proj = np.concatenate(col_idx_proj)
        residuals = np.concatenate(residuals)
        nsegs = np.concatenate(nsegs)
        nonzero_proj = np.concatenate(nonzero_proj)
        row_idx_proj = np.array([i for i in range(len(nsegs)) for j in range(nsegs[i])])
        G = scipy.sparse.coo_matrix(
            (nonzero_proj, (row_idx_proj, col_idx_proj)),
            shape=(len(nsegs), params['ncell'])
        )
    residuals = COMM.bcast(
        None if RANK is not ROOT_RANK else residuals,
        root=ROOT_RANK
    )
    G = COMM.bcast(
        None if RANK is not ROOT_RANK else G,
        root=ROOT_RANK
    )
    return (G, residuals)


def generate_projection_matrix(grid, ncell=300):
    """
    Generate the matrix to project each rectilinear grid node to its
    host Voronoi cell.

    :param grid: M x N x P x 3 array of spherical coordinates of grid nodes.
    :type grid: numpy.ndarray
    :param ncell: Number of Voronoi cells to generate.
    :type ncell: int
    :return: ncell x 3 array of spherical coordinates of Voronoi cell centers and M*N*P x ncell projection matrix.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    try:
        return (_generate_projection_matrix(grid, ncell=ncell))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)

def _generate_projection_matrix(grid, ncell=300):
    vcells = generate_voronoi_cells(grid, ncell)
    dist = scipy.spatial.distance.cdist(
        sph2xyz(grid.nodes.reshape(-1, 3)),
        sph2xyz(vcells)
    )
    colid = np.argmin(dist, axis=1)
    rowid = np.arange(np.prod(grid.nodes.shape[:-1]))

    G_proj = scipy.sparse.coo_matrix(
        (np.ones(np.prod(grid.nodes.shape[:-1]), ), (rowid, colid)),
        shape=(np.prod(grid.nodes.shape[:-1]), ncell)
    )
    return (vcells, G_proj)


def generate_voronoi_cells(grid, ncell):
    """
    Generate a random set of points representing the centers of Voronoi cells.

    :param grid: M x N x P x 3 array of spherical coordinates of grid nodes.
    :type grid: numpy.ndarray
    :param ncell: Number of Voronoi cells to generate.
    :type ncell: int
    :return: ncell x 3 array of spherical coordinates of Voronoi cell centers and M*N*P x ncell projection matrix.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    try:
        return (_generate_voronoi_cells(grid, ncell))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)

def _generate_voronoi_cells(grid, ncell):
    delta = (grid.max_coords - grid.min_coords)
    return (np.random.rand(ncell, 3) * delta + grid.min_coords)


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
        payload = iterate_inversion(payload, argc, params, iiter)


def iterate_inversion(payload, argc, params, iiter):
    """
    Iterate the entire inversion process.

    :param payload: Data payload containing df_events, df_arrivals,
        df_stations, vmodel_p, vmodel_s.
    :param argc: Command-line arguments.
    :param params: Configuration-file parameters.
    :param iiter: Iteration counter.
    """
    try:
        return (_iterate_inversion(payload, argc, params, iiter))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _iterate_inversion(payload, argc, params, iiter):
    # Update P-wave velocity model.
    payload['vmodel_p'] = update_velocity_model(payload, params, 'P')
    # Save P-wave velocity model to disk.
    if RANK == ROOT_RANK:
        write_vmodel_to_disk(payload['vmodel_p'], 'P', params, argc, iiter)

    # Update S-wave velocity model.
    payload['vmodel_s'] = update_velocity_model(payload, params, 'S')
    # Save S-wave velocity model to disk.
    if RANK == ROOT_RANK:
        write_vmodel_to_disk(payload['vmodel_s'], 'S', params, argc, iiter)

    # Update event locations.
    payload['df_events'] = update_event_locations(payload, argc, params, iiter)
    if RANK == ROOT_RANK:
        write_events_to_disk(payload['df_events'], params, argc, iiter)

    return (payload)


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
    df_stations = sanitize_stations(df_arrivals, df_stations)
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

def load_solver_from_disk(fname):
    """
    Load the EikonalSolver saved to disk as fname.

    :param fname: Name of file to load.
    :return: EikonalSolver saved to disk.
    :rtype: pykonal.EikonalSolver
    """
    solver = pykonal.EikonalSolver(coord_sys='spherical')
    with np.load(fname) as npz:
        solver.vgrid.min_coords = npz['min_coords']
        solver.vgrid.node_intervals = npz['node_intervals']
        solver.vgrid.npts = npz['npts']
        solver.uu[...] = npz['uu']
    return (solver)


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


def locate_event(df_arrivals, bounds, wdir):
    """
    Locate event.

    :param df_arrivals:
    :param bounds:
    :param wdir:
    :return:
    """
    try:
        return (_locate_event(df_arrivals, bounds, wdir))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _locate_event(df_arrivals, bounds, wdir):
    tti = dict()
    for arrival_idx, arrival in df_arrivals.iterrows():
        station_id, phase = arrival[['station_id', 'phase']]
        fname = os.path.join(wdir, f'{station_id}.{phase}.npz')
        solver = load_solver_from_disk(fname)
        if station_id not in tti:
            tti[station_id] = dict()
        tti[station_id][phase] = pykonal.LinearInterpolator3D(solver.pgrid, solver.uu)
    soln = scipy.optimize.differential_evolution(
        cost_function,
        bounds,
        args=(df_arrivals, tti),
        polish=True
    )
    return (
        pd.DataFrame([np.append(soln.x, soln.fun)], columns=['lat', 'lon', 'depth', 'time', 'res'])
    )


def locate_events(df_arrivals, bounds, wdir):
    """
    Locate events.
    :param df_arrivals:
    :param bounds:
    :param wdir:
    :return:
    """
    try:
        return (_locate_events(df_arrivals, bounds, wdir))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _locate_events(df_arrivals, bounds, wdir):
    event_ids = df_arrivals['event_id'].unique()
    nev = len(event_ids)
    df_arrivals = df_arrivals.sort_values('event_id').set_index('event_id')
    df_events = pd.DataFrame()
    ievent = 0
    for event_id in event_ids:
        ievent += 1
        logger.debug(f"Locating event #{ievent} of {nev} (id #{event_id})")
        event = locate_event(
            df_arrivals.loc[event_id],
            bounds,
            wdir
        )
        df_events = df_events.append(event, ignore_index=True)
    return (df_events)


def realize_random_trial(payload, params, phase, wdir):
    """
    Update the velocity model for one random realization.

    :param payload: Data payload containing df_events, df_arrivals,
        df_stations, vmodel_p, vmodel_s.
    :param params: Configuration-file parameters.
    :param phase: Phase to update velocity model for.
    :param wdir: Temporary working directory.
    :type wdir: tempfile.TemporaryDirectory
    :return: Updated velocity model for one random realization.
    :rtype: VelocityModel
    """
    try:
        return (_realize_random_trial(payload, params, phase, wdir))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _realize_random_trial(payload, params, phase, wdir):
    vmodel = payload['vmodel_p'] if phase == 'P' else payload['vmodel_s']
    if RANK == ROOT_RANK:
        # Generate Voronoi cells and projection matrix
        vcells, G_proj = generate_projection_matrix(vmodel.grid, params['ncell'])
        # Generate random sample of data
        df_sample = sample_observed_data(params, payload, phase)
    vcells = COMM.bcast(None if RANK is not ROOT_RANK else vcells, root=ROOT_RANK)
    G_proj = COMM.bcast(None if RANK is not ROOT_RANK else G_proj, root=ROOT_RANK)
    df_sample = COMM.bcast(None if RANK is not ROOT_RANK else df_sample, root=ROOT_RANK)
    G, residuals = generate_inversion_matrix(
        payload,
        params,
        phase,
        wdir,
        df_sample,
        vcells,
        G_proj
    )
    # Solve matrix
    dslo_proj = scipy.sparse.linalg.lsmr(
        G,
        residuals,
        params['damp'],
        params['atol'],
        params['btol'],
        params['conlim'],
        params['maxiter'],
        show=False
    )[0]
    dslo = (G_proj * dslo_proj).reshape(vmodel.velocity.shape)
    slo = np.power(vmodel.velocity, -1) + dslo
    vmodel = VelocityModel(vmodel.grid, np.power(slo, -1))
    return (vmodel)


def sample_observed_data(params, payload, phase, homo_sample=False, ncell=500):
    """
    Return a random sample of arrivals from the observed data set.
    nsamp now represents how many events are used in the inversion,
    rather than how many traveltime data.

    :type homo_sample:
    :param params:
    :param payload:
    :return:
    """
    vmodel = payload['vmodel_p'] if phase is 'P' else payload['vmodel_s']
    grid = vmodel.grid
    df_events = payload['df_events']
    df_arrivals = payload['df_arrivals']
    df_arrivals = df_arrivals[df_arrivals['phase'] == phase]
    df_arrivals = df_arrivals.merge(
        payload['df_events'][
            ['lat', 'lon', 'depth', 'time', 'event_id']
        ],
        on='event_id',
        suffixes=('_arrival', '_origin')
    ).merge(
        payload['df_stations'][
            ['lat', 'lon', 'depth', 'station_id']
        ],
        on='station_id',
        suffixes=('_origin', '_station')
    )
    origin_xyz = sph2xyz(geo2sph(df_arrivals[['lat_origin', 'lon_origin', 'depth_origin']]))
    df_arrivals['origin_x'] = origin_xyz[:, 0]
    df_arrivals['origin_y'] = origin_xyz[:, 1]
    df_arrivals['origin_z'] = origin_xyz[:, 2]
    station_xyz = sph2xyz(geo2sph(df_arrivals[['lat_station', 'lon_station', 'depth_station']]))
    df_arrivals['station_x'] = station_xyz[:, 0]
    df_arrivals['station_y'] = station_xyz[:, 1]
    df_arrivals['station_z'] = station_xyz[:, 2]
    vcells = sph2xyz(generate_voronoi_cells(grid, ncell=ncell))
    dist = scipy.spatial.distance.cdist(
        df_arrivals[['origin_x', 'origin_y', 'origin_z']],
        vcells
    )
    df_arrivals['origin_cell_id'] = np.argmin(dist, axis=1)
    dist = scipy.spatial.distance.cdist(
        df_arrivals[['station_x', 'station_y', 'station_z']],
        vcells
    )
    df_arrivals['station_cell_id'] = np.argmin(dist, axis=1)
    df_arrivals['path_id'] = list(zip(df_arrivals['origin_cell_id'], df_arrivals['station_cell_id']))
    value_counts = df_arrivals['path_id'].value_counts()
    df_arrivals['weight'] = (1 / value_counts.loc[df_arrivals['path_id']]).values
    df_sample = df_arrivals.sample(weights='weight', n=params['nsamp'])
    df_sample['travel_time'] = df_sample['time_arrival'] - df_sample['time_origin']
    df_sample = df_sample[['station_id', 'event_id', 'phase', 'travel_time']]
    df_sample = df_sample.sort_values(
        ['station_id', 'event_id']
    ).set_index(
        ['station_id', 'event_id']
    )
    return (df_sample)


def sanitize_arrivals(df_arrivals, df_stations):
    """
    Remove arrivals at stations without metadata.

    :return: Arrivals at stations with metadata.
    :rtype: pandas.DataFrame
    """
    try:
        return (_sanitize_arrivals(df_arrivals, df_stations))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _sanitize_arrivals(df_arrivals, df_stations):
    return (
        df_arrivals[
            df_arrivals['station_id'].isin(df_stations['station_id'].unique())
        ]
    )


def sanitize_stations(df_arrivals, df_stations):
    """
    Remove stations without arrivals.

    :return: Stations with arrivals.
    :rtype: pandas.DataFrame
    """
    try:
        return (_sanitize_stations(df_arrivals, df_stations))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _sanitize_stations(df_arrivals, df_stations):
    return (
        df_stations[
            df_stations['station_id'].isin(df_arrivals['station_id'].unique())
        ]
    )


def trace_ray(event, arrival, solver, vcells):
    """
    Return matrix entries for a single ray.
    :param event:
    :param solver:
    :param vcells:
    :return:
    """
    try:
        return (_trace_ray(event, arrival, solver, vcells))
    except pykonal.OutOfBoundsError as exc:
        logger.warning(exc)
        return (None)
    except Exception as exc:
        logger.error(exc)
        raise

def _trace_ray(event, arrival, solver, vcells):
    nonzerop = np.array([], dtype=DTYPE_REAL)
    tti = pykonal.LinearInterpolator3D(solver.pgrid, solver.uu)
    rho_src, theta_src, phi_src = geo2sph(event[['lat', 'lon', 'depth']])
    synthetic = tti((rho_src, theta_src, phi_src))
    residual = arrival['travel_time'] - synthetic
    ray = solver.trace_ray((rho_src, theta_src, phi_src))
    idxs, counts = find_ray_idx(ray, vcells)
    nseg = len(idxs)
    nonzerop = counts * solver._get_step_size()
    return (residual, idxs, nseg, nonzerop)


def update_event_locations(payload, argc, params, iiter):
    """
    Update event locations using differential-evolution optimization and 3D velocity models.

    :param payload:
    :param argc:
    :param params:
    :param iiter:
    :return: Updated event locations.
    """
    try:
        return (_update_event_locations(payload, argc, params, iiter))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _update_event_locations(payload, argc, params, iiter):
    try:
        wdir = COMM.bcast(
            None if RANK is not ROOT_RANK else tempfile.mkdtemp(dir=argc.working_dir),
            root=ROOT_RANK
        )
        if RANK == ROOT_RANK:
            logger.info(f'Updating event locations.')
        # Compute traveltime-lookup tables
        if RANK == ROOT_RANK:
            logger.info("Computing P-wave traveltime-lookup tables.")
        compute_traveltime_lookup_tables(payload, 'P', wdir)
        if RANK == ROOT_RANK:
            logger.info("Computing S-wave traveltime-lookup tables.")
        compute_traveltime_lookup_tables(payload, 'S', wdir)
        COMM.barrier()
        event_ids = COMM.scatter(
            np.array_split(payload['df_events']['event_id'].values, WORLD_SIZE),
            root=ROOT_RANK
        )
        logger.debug(f"Locating {len(event_ids)} events")
        df_arrivals = payload['df_arrivals']
        df_arrivals = df_arrivals[df_arrivals['event_id'].isin(event_ids)]
        latmax, lonmin, dmax = sph2geo(payload['vmodel_p'].grid.min_coords)
        latmin, lonmax, dmin = sph2geo(payload['vmodel_p'].grid.max_coords)
        bounds = (
            (latmin, latmax),
            (lonmin, lonmax),
            (dmin, dmax),
            (0, (pd.to_datetime('now')-pd.to_datetime(0)).total_seconds())
        )
        df_events = locate_events(df_arrivals, bounds, wdir)
        logger.debug(f"Finished locating {len(df_events)} events.")
        df_events = COMM.gather(df_events, root=ROOT_RANK)
        if RANK == ROOT_RANK:
            df_events = pd.concat(df_events, ignore_index=True)
            logger.debug(f"{len(df_events)} total events located.")
        df_events = COMM.bcast(
            None if RANK is not ROOT_RANK else df_events,
            root=ROOT_RANK
        )
    finally:
        if RANK == ROOT_RANK:
            logger.debug(f'Removing working directory {wdir}')
            shutil.rmtree(wdir)
    return (df_events)


def update_velocity_model(payload, params, phase):
    """
    Return updated velocity model.

    :param payload: Data payload containing df_events, df_arrivals,
        df_stations, vmodel_p, vmodel_s.
    :param params: Configuration-file parameters.
    :param phase: Phase to update velocity model for.
    :param wdir: Temporary working directory.
    :type wdir: tempfile.TemporaryDirectory
    """
    try:
        return (_update_velocity_model(payload, params, phase))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)

def _update_velocity_model(payload, params, phase):
    vmodel = payload['vmodel_p'] if phase == 'P' else payload['vmodel_s']
    try:
        wdir = COMM.bcast(
            None if RANK is not ROOT_RANK else tempfile.mkdtemp(dir=argc.working_dir),
            root=ROOT_RANK
        )
        if RANK == ROOT_RANK:
            logger.debug(f'Updating {phase}-wave velocity model')
        # Compute traveltime-lookup tables
        logger.info(f"Computing {phase}-wave traveltime-lookup tables.")
        compute_traveltime_lookup_tables(payload, phase, wdir)
        COMM.barrier()
        vmodels = []
        for ireal in range(params['nreal']):
            if RANK == ROOT_RANK:
                logger.info(f"Realizing random trial #{ireal} of {params['nreal']}")
            vmodels.append(realize_random_trial(payload, params, phase, wdir))
        COMM.barrier()
        # Update velocity model
        velocity = np.stack([vmodel.velocity for vmodel in vmodels])
        velocity = np.mean(velocity, axis=0)
        vmodel = VelocityModel(vmodel.grid, velocity)
    finally:
        if RANK == ROOT_RANK:
            logger.debug(f'Removing working directory {wdir}')
            shutil.rmtree(wdir)
    return (vmodel)


def write_events_to_disk(df_events, params, argc, iiter):
    """
    Write event locations to disk.

    :param df_events:
    :param params:
    :param argc:
    :return:
    """
    try:
        return (_write_events_to_disk(df_events, params, argc, iiter))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _write_events_to_disk(df_events, params, argc, iiter):
    nreal = params['nreal']
    nsamp = params['nsamp']
    ncell = params['ncell']
    fname = os.path.join(
        argc.output_dir,
        f'events.{iiter + 1:03d}.{nreal}.{nsamp}.{ncell}.h5',
    )
    logger.info(f"Saving events to disk: {fname}")
    if not os.path.isdir(argc.output_dir):
        os.makedirs(argc.output_dir)
    # Save the updated event locations to disk.
    with pd.HDFStore(fname) as store:
        store['events'] = df_events


def write_vmodel_to_disk(vmodel, phase, params, argc, iiter):
    """
    Write velocity model to disk.

    :param vmodel:
    :param phase:
    :param params:
    :param argc:
    :return:
    """
    try:
        return (_write_vmodel_to_disk(vmodel, phase, params, argc, iiter))
    except Exception as exc:
        logger.error(exc)
        sys.exit(-1)


def _write_vmodel_to_disk(vmodel, phase, params, argc, iiter):
    nreal = params['nreal']
    nsamp = params['nsamp']
    ncell = params['ncell']
    fname = os.path.join(
        argc.output_dir,
        f'model.{phase}.{iiter + 1:03d}.{nreal}.{nsamp}.{ncell}.npz',
    )
    logger.info(f"Saving velocity model to disk: {fname}")
    if not os.path.isdir(argc.output_dir):
        os.makedirs(argc.output_dir)
    # Save the update velocity model to disk.
    np.savez_compressed(
        fname,
        min_coords=vmodel.grid.min_coords,
        node_intervals=vmodel.grid.node_intervals,
        npts=vmodel.grid.npts,
        vv=vmodel.velocity
    )

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
