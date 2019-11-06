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
# Class definitions

class VelocityModel(object):
    """
    A simple container class.
    """

    def __init__(self, grid, velocity):
        self.grid = grid
        self.velocity = velocity


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


###############################################################################
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
    dist = scipy.spatial.distance.cdist(sph2xyz(ray), sph2xyz(vcells))
    argmin = np.argmin(dist, axis=1)
    idxs, counts = np.unique(argmin, return_counts=True)
    return (idxs, counts)


def generate_projection_matrix(grid, ncell=300):
    """
    Generate the matrix to project each rectilinear grid node to its
    host Voronoi cell.
    """
    vcells = generate_voronoi_cells(grid, ncell)
    dist = scipy.spatial.distance.cdist(
        sph2xyz(grid.nodes.reshape(-1, 3)),
        sph2xyz(vcells)
    )
    colid = np.argmin(dist, axis=1)
    rowid = np.arange(np.prod(grid.nodes.shape[:-1]))

    Gp = scipy.sparse.coo_matrix(
        (np.ones(np.prod(grid.nodes.shape[:-1]), ), (rowid, colid)),
        shape=(np.prod(grid.nodes.shape[:-1]), ncell)
    )
    return (vcells, Gp)


def generate_voronoi_cells(grid, ncell):
    """
    Generate a random set of points representing the centers of Voronoi cells.
    """
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


def iterate_inversion(payload, params, argc, iiter):
    """
    Iterate the entire inversion process to update the velocity model
    and event locations.

    :param payload: Data payload containing df_events, df_arrivals, df_stations, vmodel_p, vmodel_s
    :param params: Inversion parameters
    """
    # Broadcast parameters.
    COMM.bcast(params, root=ROOT_RANK)
    # Update velocity models.
    # Create a temporary working directory:
    with tempfile.TemporaryDirectory(dir=argc.output_dir) as temp_dir:
        payload['temp_dir'] = temp_dir
        # Make sub-directories for P- and S-wave data.
        os.mkdir(os.path.join(temp_dir, 'P'))
        os.mkdir(os.path.join(temp_dir, 'S'))
        # Update P-wave velocity model
        params['phase'] = 'P'
        payload['vmodel_p'] = update_velocity_model(payload, params)
        write_vmodel_to_disk(payload['vmodel_p'], params, argc, iiter)
        # Update S-wave velocity model
        params['phase'] = 'S'
        payload['vmodel_s'] = update_velocity_model(payload, params)
        write_vmodel_to_disk(payload['vmodel_s'], params, argc, iiter)
    # Update locations.
    # Create a temporary working directory:
    with tempfile.TemporaryDirectory(dir=argc.output_dir) as temp_dir:
        payload['temp_dir'] = temp_dir
        # Make sub-directories for P- and S-wave data.
        os.mkdir(os.path.join(temp_dir, 'P'))
        os.mkdir(os.path.join(temp_dir, 'S'))
        # Locate earthquakes.
        payload['df_events'] = update_event_locations(payload, params)
        write_events_to_disk(payload['df_events'], params, argc, iiter)


def update_velocity_model(payload, params):
    """
    Update the velocity model.

    :param payload: Data payload containing df_events, df_arrivals, df_stations, vmodel_p, vmodel_s
    :type payload: dict
    :param params: Inversion parameters.
    :type params: dict
    :param phase: The phase to update model for ('P' or 'S')
    :type phase: str

    :return: Update VelocityModel object.
    :rtype: VelocityModel
    """
    logger.info(f'Updating the {params["phase"]}-wave velocity model.')
    logger.debug(f'Temporary directory is {payload["temp_dir"]}')
    if params['phase'] == 'P':
        payload['vmodel'] = payload['vmodel_p']
    elif params['phase'] == 'S':
        payload['vmodel'] = payload['vmodel_s']
    else:
        raise (NotImplementedError(f'Did not recognize phase type {params["phase"]}'))
    # Subset arrivals for the correct phase
    df_arrivals = payload['df_arrivals']
    payload['df_arrivals'] = df_arrivals[df_arrivals['phase'] == params['phase']]
    dslo = []
    # Iterate over different realizations of the random sampling.
    for ireal in range(params['nreal']):
        logger.info(f'Realizing random trial #{ireal+1} of {params["nreal"]}.')
        dslo.append(realize_random_trial(payload, params))
    # Replace the full df_arrivals data set in the payload.
    payload['df_arrivals'] = df_arrivals
    # Update the velocity model
    vmodel = payload['vmodel']
    dslo = np.mean(dslo, axis=0).reshape(vmodel.grid.npts)
    vmodel.velocity = np.power((np.power(vmodel.velocity, -1) + dslo), -1)
    return (vmodel)


def load_event_data(argc, params):
    """
    Read and return *events* and *arrivals* tables from pandas.HDFStore.
    """
    with pd.HDFStore(argc.event_file) as store:
        df_events = store['events']
        df_arrivals = store['arrivals']
    return (df_events, df_arrivals)


def load_initial_velocity_model(params, phase):
    """
    Load the initial velocity model.

    :param params:
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


def load_network_data(argc):
    """
    Read and return *stations* table from pandas.HDFStore.
    """
    with pd.HDFStore(argc.network_file) as store:
        df_stations = store['stations']
    df_stations['depth'] = df_stations['elev'] * -1
    return (df_stations)


def load_solver_from_disk(fname):
    """
    Load the EikonalSolver saved to disk as name.

    :param fname:
    :return:
    """
    solver = pykonal.EikonalSolver(coord_sys='spherical')
    with np.load(fname) as npz:
        solver.vgrid.min_coords = npz['min_coords']
        solver.vgrid.node_intervals = npz['node_intervals']
        solver.vgrid.npts = npz['npts']
        solver.uu[...] = npz['uu']
    return (solver)


def load_solver_from_scratch(station, vmodel, temp_dir, tag=None):
    """
    Create and return the solved EikonalSolver for *station*.

    :param station:
    :param vmodel:
    :param temp_dir:
    :param tag:
    :return:
    """
    rho0, theta0, phi0 = geo2sph(station[['lat', 'lon', 'depth']].values)
    far_field = init_farfield(vmodel)
    near_field = init_nearfield(far_field, (rho0, theta0, phi0))
    near_field.solve()
    far_field.transfer_travel_times_from(near_field, (-rho0, theta0, phi0), set_alive=True)
    far_field.solve()
    station_id = station['station_id']
    np.savez_compressed(
        os.path.join(
            temp_dir,
            f'{station_id}.npz' if tag is None else f'{station_id}.{tag}.npz'
        ),
        uu=far_field.uu,
        min_coords=far_field.pgrid.min_coords,
        node_intervals=far_field.pgrid.node_intervals,
        npts=far_field.pgrid.npts
    )
    return (far_field)


def update_event_locations(payload, params):
    """
    Update event locations.

    :param payload: Data payload containing df_events, df_arrivals, df_stations, vmodel_p, vmodel_s
    :type payload: dict
    :param params: Inversion parameters.
    :type params: dict

    :return: Event locations.
    :rtype: pandas.DataFrame
    """
    COMM.bcast(payload, root=ROOT_RANK)
    logger.debug(f'Scattering {len(payload["df_events"])} events.')
    df_events = COMM.scatter(
        np.array_split(payload['df_events'], WORLD_SIZE),
        root=ROOT_RANK
    )
    logger.debug(f'Received {len(df_events)} scattered events.')
    df_arrivals = payload['df_arrivals']
    df_arrivals = df_arrivals[df_arrivals['event_id'].isin(df_events['event_id'])]
    df_events = locate_events(
        df_arrivals,
        payload['df_stations'],
        payload['vmodel_p'],
        payload['vmodel_s']
    )
    df_events = pd.DataFrame()
    for df in COMM.gather(df_events, root=ROOT_RANK):
        df_events = df_events.append(df, ignore_index=True)
    return (df_events)

def locate_events(
    df_arrivals,
    df_stations,
    vmodel_p,
    vmodel_s,
):
    '''
    df_arrivals :: pandas.DataFrame :: Arrival data. Must include
        (event_id, station_id, phase, time) fields with (int, str, str,
        float) dtypes, respectively.
    df_stations :: pandas.DataFrame :: Network geometry data. Must
        include (station_id, lat, lon, depth) fields with (str, float,
        float, float) dtypes, respectively.
    vmodel_p :: VelocityModel :: P-wave velocity model.
    vmodel_s :: VelocityModel :: S-wave velocity model.

    return :: pandas.DataFrame :: Event locations. Includes (event_id,
        lat, lon, depth, res) fields with (int, float, float, float,
        float) dtypes, respectively.
    '''
    latmax, lonmin, dmax = sph2geo(vmodel_p.grid.min_coords)
    latmin, lonmax, dmin = sph2geo(vmodel_p.grid.max_coords)
    bounds = (
        (latmin, latmax),
        (lonmin, lonmax),
        (dmin, dmax),
        (0, (pd.to_datetime('now')-pd.to_datetime(0)).total_seconds())
    )
    df_events = pd.DataFrame(columns=['event_id', 'lat', 'lon', 'depth', 'time', 'res'])
    ievent = 0
    event_ids = df_arrivals['event_id'].unique()
    nev = len(event_ids)
    with tempfile.TemporaryDirectory() as temp_dir:
        for event_id in event_ids:
            ievent += 1
            logger.debug(f'Locating event #{ievent} of {nev} (event id: {event_id})')
            location = locate_event(
                df_arrivals.set_index('event_id').loc[event_id],
                df_stations,
                vmodel_p,
                vmodel_s,
                temp_dir,
                bounds
            )
            df_append             = pd.DataFrame(
                [location.x],
                columns=['lat', 'lon', 'depth', 'time']
            )
            df_append['res']      = location.fun
            df_append['event_id'] = event_id
            df_events = df_events.append(
                df_append, ignore_index=True, sort=True
            )
        return (df_events)


def locate_event(df_arrivals, df_stations, vmodel_p, vmodel_s, temp_dir, bounds):
    tti = dict()
    for arrival_idx, arrival in df_arrivals.iterrows():
        station_id, phase = arrival[['station_id', 'phase']]
        fname = os.path.join(temp_dir, f'{station_id}.{phase}.npz')
        if not os.path.isfile(fname):
            station = df_stations.set_index('station_id').loc[station_id]
            station['station_id'] = station.name
            solver = load_solver_from_scratch(
                station,
                vmodel_p if phase == 'P' else vmodel_s,
                temp_dir,
                tag=phase
            )
        else:
            solver = load_solver_from_disk(fname)
        if station_id not in tti:
            tti[station_id] = dict()
        tti[station_id][phase] = pykonal.LinearInterpolator3D(solver.pgrid, solver.uu)
    return (
        scipy.optimize.differential_evolution(
            cost_function,
            bounds,
            args=(df_arrivals, tti), polish=True)
    )


def process_sample(df_stations, sample_payload):
    """
    Determine the matrix components for arrivals at a subset of stations
    using sampled data.

    :param df_stations:
    :param sample_payload:
    :return:
    """
    dobs, colidp, nsegs, nonzerop = [], [], [], []
    for idx, station in df_stations.iterrows():
        station_id = station['station_id']
        station_payload = dict(
            df_obs=sample_payload['df_sample'].loc[station_id],
            df_events=sample_payload['df_events'],
            vcells=sample_payload['vcells'],
            temp_dir=sample_payload['temp_dir'],
            vmodel=sample_payload['vmodel']
        )
        res = process_station(station, station_payload)
        dobs += res[0]
        colidp += res[1]
        nsegs += res[2]
        nonzerop += res[3]
    return (dobs, colidp, nsegs, nonzerop)


def process_station(station, station_payload):
    """
    Determine the matrix components for arrivals at *station*.

    :param args:
    :return:
    """
    station_id, lat, lon, depth = station[['station_id', 'lat', 'lon', 'depth']]
    logger.debug(f'Processing station {station_id}')
    df_obs = station_payload['df_obs']
    df_events = station_payload['df_events'].set_index('event_id')
    vcells = station_payload['vcells']
    colidp = []
    nsegs = []
    nonzerop = []
    dobs = []
    fname = os.path.join(station_payload['temp_dir'], f'{station_id}.npz')
    if os.path.isfile(fname):
        solver = load_solver_from_disk(fname)
    else:
        solver = load_solver_from_scratch(station, station_payload['vmodel'], station_payload['temp_dir'])
    tti = pykonal.LinearInterpolator3D(solver.pgrid, solver.uu)
    for event_id in df_obs.index.unique():
        event = df_events.loc[event_id]
        try:
            rho_src, theta_src, phi_src = geo2sph(event[['lat', 'lon', 'depth']].values)
            synthetic = tti((rho_src, theta_src, phi_src))
            residual = df_obs.loc[event_id, 'travel_time'] - synthetic
            if abs(residual) > 3.0:
                continue
            ray = solver.trace_ray((rho_src, theta_src, phi_src))
            idxs, counts = find_ray_idx(ray, vcells)
            nseg = len(idxs)
            nsegs.append(nseg)
            dobs.append(residual)
            for iseg in range(nseg):
                colidp.append(idxs[iseg])
                nonzerop.append(solver._get_step_size() * counts[iseg])
        except pykonal.OutOfBoundsError as err:
            continue
    return (dobs, colidp, nsegs, nonzerop)


def realize_random_trial(payload, params):
    """

    :param payload:
    :param params:
    :return:
    """
    vcells, G_proj = generate_projection_matrix(payload['vmodel'].grid, params['ncell'])
    df_sample = sample_observed_data(params, payload)
    # Broadcast payload.
    logger.debug('Broadcasting sample payload')
    sample_payload = COMM.bcast(
        dict(
            df_sample=df_sample,
            df_events=payload['df_events'],
            vmodel=payload['vmodel'],
            vcells=vcells,
            temp_dir=os.path.join(payload['temp_dir'], params['phase'])
        ),
        root=ROOT_RANK
    )
    logger.debug('Sample payload broadcasted')
    logger.debug('Scattering stations')
    df_stations = COMM.scatter(
        np.array_split(
            payload['df_stations'][
                payload['df_stations'][
                    'station_id'
                ].isin(
                    df_sample.index.unique(level='station_id')
                )
            ],
            WORLD_SIZE
        ),
        root=ROOT_RANK
    )
    logger.debug('Stations scattered')
    dobs, colidp, nsegs, nonzerop = process_sample(df_stations, sample_payload)
    dobs = np.array(sum(COMM.gather(dobs, root=ROOT_RANK), []))
    col_idx_proj = np.array(sum(COMM.gather(colidp, root=ROOT_RANK), []))
    nsegs = np.array(sum(COMM.gather(nsegs, root=ROOT_RANK), []))
    row_idx_proj = np.array([i for i in range(len(nsegs)) for j in range(nsegs[i])])
    nonzero_proj = np.array(sum(COMM.gather(nonzerop, root=ROOT_RANK), []))
    # Invert data for velocity model update.

    G = scipy.sparse.coo_matrix(
        (nonzero_proj, (row_idx_proj, col_idx_proj)),
        shape=(len(nsegs), params['ncell'])
    )

    x = scipy.sparse.linalg.lsmr(
        G,
        dobs,
        params['damp'],
        params['atol'],
        params['btol'],
        params['conlim'],
        params['maxiter'],
        show=False
    )[0]
    dslo = G_proj * x
    return (dslo)


def sample_observed_data(params, payload, homo_sample = False, ncell = 500):
    """
    Return a random sample of arrivals from the observed data set.
    nsamp now represents how many events are used in the inversion, 
    rather than how many travel time data.

    :param params:
    :param payload:
    :return:
    """
    events = payload['df_events']
    if homo_sample:
        lat_s = events['lat'].min()
        lat_n = events['lat'].max()
        lon_w = events['lon'].min()
        lon_e = events['lon'].max()
        dep_u = events['depth'].min()
        dep_d = events['depth'].max()

        rlat = np.random.rand(ncell,)
        rlat = lat_s+(lat_n-lat_s)*rlat
        rlon = np.random.rand(ncell,)
        rlon = lon_w+(lon_e-lon_w)*rlon
        rdep = np.random.rand(ncell,)
        rdep = dep_u+(dep_d-dep_u)*rdep

        cell3d = np.vstack([rdep,rlat,rlon]).T
        cell3d = geo2sph(cell3d)
        cell3d = sph2xyz(cell3d)
        
        evxyz = events[['depth','lat','lon']]
        evxyz = geo2sph(evxyz)
        evxyz = sph2xyz(evxyz)

        dist = scipy.spatial.distance.cdist(evxyz,cell3d)
        eveidx = np.argmin(dist,axis = 1)
        
        events['cellid'] = eveidx
        evnum = events.groupby('cellid').count()

        evnum = evnum.rename(columns={'time':'evnum'})
        evnum = evnum[['evnum']]
        events = events.merge(evnum,on = 'cellid')
        events['weight'] =  1.0/events['evnum']
        eventsused = events.sample(n=params['nsamp'],weights=weight)
    else:
        eventsused = events.sample(
                    n=params['nsamp']
                )

    df = payload['df_arrivals'].merge(
#        payload['df_events'][['time', 'event_id']],
        eventsused[['time', 'event_id']],
        on='event_id',
        suffixes=('_arrival', '_origin')
    )
    df['travel_time'] = (df['time_arrival'] - df['time_origin'])
    # Remove any arrivals at stations without metadata
    df = df[df['station_id'].isin(payload['df_stations']['station_id'].unique())]
    df = df.sort_values(
                'station_id'
                ).drop_duplicates( # There shouldn't be any duplicates in a clean data set.
                    subset=['station_id', 'event_id'],
                    keep=False
                ).set_index(
                    ['station_id', 'event_id']
                ).drop(
                    columns=['time_arrival', 'time_origin']
                )
    return df


def sanitize_arrivals(df_arrivals, df_stations):
    return (
        df_arrivals[
            df_arrivals['station_id'].isin(df_stations['station_id'].unique())
        ]
    )


def write_events_to_disk(df_events, params, argc, iiter):
    """
    Write event locations to disk.

    :param df_events:
    :param params:
    :param argc:
    :return:
    """
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


def write_vmodel_to_disk(vmodel, params, argc, iiter):
    """
    Write velocity model to disk.

    :param vmodel:
    :param params:
    :param argc:
    :return:
    """
    nreal = params['nreal']
    nsamp = params['nsamp']
    ncell = params['ncell']
    fname = os.path.join(
        argc.output_dir,
        f'model.{params["phase"]}.{iiter + 1:03d}.{nreal}.{nsamp}.{ncell}.npz',
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

def root_main(argc, params):
    """
    The main control loop for the root process.
    :return:
    """
    logger.info("Stating root thread.")
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
    # Iterate the entire inversion process.
    for iiter in range(params['niter']):
        logger.info(f'Stating iteration #{iiter+1} of {params["niter"]}.')
        payload = iterate_inversion(payload, params, argc, iiter)


def worker_main():
    """
    The main control loop for worker processes.
    :return:
    """
    logger.info("Stating worker thread.")
    # Receive parameters.
    params = COMM.bcast(None, root=ROOT_RANK)
    # Iterate over different realizations of the random sampling.
    for iiter in range(params['niter']):
        for phase in ('P', 'S'):
            for ireal in range(params['nreal']):
                # Receive payload.
                sample_payload = COMM.bcast(None, root=ROOT_RANK)
                # Receive list of stations to process
                df_stations = COMM.scatter(None, root=ROOT_RANK)
                dobs, colidp, nsegs, nonzerop = process_sample(df_stations, sample_payload)
                COMM.gather(dobs, root=ROOT_RANK)
                COMM.gather(colidp, root=ROOT_RANK)
                COMM.gather(nsegs, root=ROOT_RANK)
                COMM.gather(nonzerop, root=ROOT_RANK)
        payload = COMM.bcast(None, root=ROOT_RANK)
        df_events = COMM.scatter(None, root=ROOT_RANK)
        df_arrivals = payload['df_arrivals']
        df_arrivals = df_arrivals[df_arrivals['event_id'].isin(df_events['event_id'])]
        df_events = locate_events(
            df_arrivals,
            payload['df_stations'],
            payload['vmodel_p'],
            payload['vmodel_s']
        )
        COMM.gather(df_events, root=ROOT_RANK)


def signal_handler(sig, frame):
    raise(SystemError("Interrupting signal received... aborting"))


if __name__ == '__main__':
    # Add some signal handlers to abort all threads.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGCONT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Parse command line arguments.
    argc = parse_args()
    # Configure logging.
    configure_logging(argc.verbose, argc.logfile)
    logger = logging.getLogger(__name__)
    if RANK == ROOT_RANK:
        # Load parameters for parameter file.
        params = load_params(argc)
        # Start the root thread's main loop.
        root_main(argc, params)
    else:
        # Start the worker threads main loop.
        worker_main()
