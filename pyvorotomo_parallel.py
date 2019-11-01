import argparse
import configparser
import numpy as np
import os
import pandas as pd
import pykonal
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import tempfile
from mpi4py import MPI

COMM = MPI.COMM_WORLD
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
    params['phase'] = parser.get('Data Sampling Parameters', 'phase')
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
# Class definitions

class VelocityModel(object):
    """
    A simple container class.
    """
    def __init__(self, grid, velocity):
        self.grid     = grid
        self.velocity = velocity

###############################################################################
# Geometry convenience functions.

def geo2sph(arr):
    '''
    Map Geographical coordinates to spherical coordinates.
    '''
    geo = np.array(arr, dtype=DTYPE_REAL)
    sph = np.empty_like(geo)
    sph[...,0] = EARTH_RADIUS - geo[...,2]
    sph[...,1] = np.pi/2 - np.radians(geo[...,0])
    sph[...,2] = np.radians(geo[...,1])
    return (sph)


def sph2geo(arr):
    '''
    Map spherical coordinates to geographic coordinates.
    '''
    sph = np.array(arr, dtype=DTYPE_REAL)
    geo = np.empty_like(sph)
    geo[...,0] = np.degrees(np.pi/2 - sph[...,1])
    geo[...,1] = np.degrees(sph[...,2])
    geo[...,2] = EARTH_RADIUS - sph[...,0]
    return (geo)


def sph2xyz(arr):
    '''
    Map spherical coordinates to Cartesian coordinates.
    '''
    sph = np.array(arr, dtype=DTYPE_REAL)
    xyz = np.empty_like(sph)
    xyz[...,0] = sph[...,0] * np.sin(sph[...,1]) * np.cos(sph[...,2])
    xyz[...,1] = sph[...,0] * np.sin(sph[...,1]) * np.sin(sph[...,2])
    xyz[...,2] = sph[...,0] * np.cos(sph[...,1])
    return (xyz)
###############################################################################


###############################################################################
def find_ray_idx(ray, vcells):
    '''
    Determine the index of the Voronoi cell hosting each point on
    the ray path.
    '''
    dist = scipy.spatial.distance.cdist(sph2xyz(ray), sph2xyz(vcells))
    argmin = np.argmin(dist, axis=1)
    idxs, counts = np.unique(argmin, return_counts=True)
    return (idxs, counts)


def generate_projection_matrix(grid, ncell=300):
    '''
    Generate the matrix to project each rectilinear grid node to its
    host Voronoi cell.
    '''
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
    '''
    Generate a random set of points representing the centers of Voronoi cells.
    '''
    delta = (grid.max_coords - grid.min_coords)
    return (np.random.rand(ncell, 3) * delta + grid.min_coords)


def init_farfield(vmodel):
    '''
    Initialize the far-field EikonalSolver with the given velocity model.
    '''
    far_field                      = pykonal.EikonalSolver(coord_sys='spherical')
    far_field.vgrid.min_coords     = vmodel.grid.min_coords
    far_field.vgrid.node_intervals = vmodel.grid.node_intervals
    far_field.vgrid.npts           = vmodel.grid.npts
    far_field.vv                   = vmodel.velocity
    return (far_field)


def init_nearfield(far_field, origin):
    '''
    Initialize the near-field EikonalSolver.

    :param origin: Station location in spherical coordinates.
    :type origin: (float, float, float)

    :return: Near-field EikonalSolver
    :rtype: pykonal.EikonalSolver
    '''
    drho                            = far_field.vgrid.node_intervals[0] / 5
    near_field                      = pykonal.EikonalSolver(coord_sys='spherical')
    near_field.vgrid.min_coords     = drho, 0, 0
    near_field.vgrid.node_intervals = drho, np.pi/20, np.pi/20
    near_field.vgrid.npts           = 100, 21, 40
    near_field.transfer_velocity_from(far_field, origin)
    vvi = pykonal.LinearInterpolator3D(near_field.vgrid, near_field.vv)

    for it in range(near_field.pgrid.npts[1]):
        for ip in range(near_field.pgrid.npts[2]):
            idx = (0, it, ip)
            near_field.uu[idx]     = near_field.pgrid[idx + (0,)] / vvi(near_field.pgrid[idx])
            near_field.is_far[idx] = False
            near_field.close.push(*idx)
    return (near_field)


def iterate_inversion(payload, params):
    """
    Iterate the entire inversion process to update the velocity model
    and event locations.

    :param payload:
    :param params:
    :return:
    """
    dslo = []
    # Broadcast parameters.
    COMM.bcast(params, root=ROOT_RANK)
    # Iterate over different realizations of the random sampling.
    for i in range(params['nreal']):
        dslo.append(realize_random_trial(payload, params))
    # Update the velocity model
    vmodel = payload['vmodel']
    dslo = np.mean(dslo, axis=0).reshape(vmodel.grid.npts)
    vmodel.velocity = np.power((np.power(vmodel.velocity, -1) + dslo), -1)
    payload['vmodel'] = vmodel

    # Update event locations here.

    return (payload)


def load_event_data(argc, params):
    """
    Read and return *events* and *arrivals* tables from pandas.HDFStore.
    """
    with pd.HDFStore(argc.event_file) as store:
        df_events = store['events']
        df_arrivals = store['arrivals']
    return (df_events, df_arrivals)


def load_initial_velocity_model(params):
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
    if params['phase'] == 'P':
        velocity = 6. * np.ones(grid.npts)
    else:
        velocity = 3.74 * np.ones(grid.npts)
    vmodel = VelocityModel(grid, velocity)
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
        solver.vgrid.min_coords     = npz['min_coords']
        solver.vgrid.node_intervals = npz['node_intervals']
        solver.vgrid.npts           = npz['npts']
        solver.uu[...]              = npz['uu']
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
    far_field  = init_farfield(vmodel)
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
    sample_payload = COMM.bcast(
        dict(
            df_sample=df_sample,
            df_events=payload['df_events'],
            vmodel=payload['vmodel'],
            vcells=vcells,
            temp_dir=payload['temp_dir']
        ),
        root=ROOT_RANK
    )
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


def sample_observed_data(params, payload):
    """
    Return a random sample of arrivals from the observed data set.

    :param params:
    :param payload:
    :return:
    """
    df = payload['df_arrivals'].merge(
        payload['df_events'][['time', 'event_id']],
        on='event_id',
        suffixes=('_arrival','_origin')
    )
    df['travel_time'] = (df['time_arrival'] - df['time_origin'])
    # Remove any arrivals at stations without metadata
    df = df[df['station_id'].isin(payload['df_stations']['station_id'].unique())]
    return (
        df.sort_values(
            'station_id'
        ).drop_duplicates( # There shouldn't be any duplicates in a clean data set.
            subset=['station_id', 'event_id'],
            keep=False
        ).set_index(
            ['station_id', 'event_id']
        ).drop(
            columns=['time_arrival', 'time_origin']
        ).sample(
            n=params['nsamp']
        )
    )


def sanitize_arrivals(df_arrivals, df_stations):
    return (
        df_arrivals[
            df_arrivals['station_id'].isin(df_stations['station_id'].unique())
        ]
    )


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

def root_main():
    """
    The main control loop for the root process.
    :return:
    """
    # Parse command line arguments.
    argc = parse_args()
    # Load parameters for parameter file.
    params = load_params(argc)
    # Load event and network data.
    df_events, df_arrivals = load_event_data(argc, params)
    df_stations = load_network_data(argc)
    df_arrivals = sanitize_arrivals(df_arrivals, df_stations)
    # Subset arrivals for the correct phase.
    df_arrivals = df_arrivals[df_arrivals['phase'] == params['phase']]
    # Load initial velocity model.
    vmodel = load_initial_velocity_model(params)
    payload = dict(
        df_events=df_events,
        df_arrivals=df_arrivals,
        df_stations=df_stations,
        vmodel=vmodel
    )
    # Create a temporary working directory:
    with tempfile.TemporaryDirectory() as temp_dir:
        payload['temp_dir'] = temp_dir
        # Iterate the entire inversion process.
        for i in range(params['niter']):
            payload = iterate_inversion(payload, params)
            write_vmodel_to_disk(vmodel, params, argc, i)


def worker_main():
    """
    The main control loop for worker processes.
    :return:
    """
    # Receive parameters.
    params = COMM.bcast(None, root=ROOT_RANK)
    # Iterate over different realizations of the random sampling.
    for i in range(params['nreal']):
        # Receive payload.
        sample_payload = COMM.bcast(None, root=ROOT_RANK)
        # Receive list of stations to process
        df_stations = COMM.scatter(None, root=ROOT_RANK)
        dobs, colidp, nsegs, nonzerop = process_sample(df_stations, sample_payload)
        COMM.gather(dobs, root=ROOT_RANK)
        COMM.gather(colidp, root=ROOT_RANK)
        COMM.gather(nsegs, root=ROOT_RANK)
        COMM.gather(nonzerop, root=ROOT_RANK)



if __name__ == '__main__':
    if RANK == ROOT_RANK:
        root_main()
    else:
        worker_main()
