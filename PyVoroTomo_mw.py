import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pykonal
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import sys
import tempfile

EARTH_RADIUS = 6371.
DTYPE_REAL   = np.float64

class Arguments(object):
    def __init__(self):
        pass
    
class VelocityModel(object):
    def __init__(self, grid, velocity):
        self.grid     = grid
        self.velocity = velocity


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


def main():
    '''
    Main.
    '''
    argc                   = load_argc()
    params                 = load_params()
    df_events, df_arrivals = load_event_data(argc, params)
    df_stations            = load_network_data(argc)
    vmodel                 = load_initial_velocity_model(params)
    df_arrivals            = sanitize_arrivals(df_arrivals, df_stations)
    payload                = dict(
        events   = df_events.sort_values('event_id').set_index('event_id'),
        arrivals = df_arrivals.set_index('phase').loc[params['phase']],
        stations = df_stations.set_index('station_id'),
        vmodel   = vmodel
    )
    # Iteratively invert data
    for i in range(params['niter']):
        # Create a temporary directory to work in.
        with tempfile.TemporaryDirectory() as temp_dir:
            payload['temp_dir'] = temp_dir
            # Update the velocity model.
            vmodel = iterate_inversion(params, payload)
        # Create the output directory if it doesn't exist.
        if not os.path.isdir(argc.output_dir):
            os.makedirs(argc.output_dir)
        # Save the update velocity model to disk.
        np.savez_compressed(
            os.path.join(argc.output_dir, f'model_{params["phase"]}.{i+1:03d}.npz'),
            min_coords=vmodel.grid.min_coords,
            node_intervals=vmodel.grid.node_intervals,
            npts=vmodel.grid.npts,
            vv=vmodel.velocity
        )
        # Update the payload with the new velocity model.
        payload['vmodel'] = vmodel


def generate_projection_matrix(grid, ncell=300):
    '''
    Generate the matrix to project each rectilinear grid node to its
    host Voronoi cell.
    '''
    vcells = generate_voronoi_cells(grid, ncell)
    dist   = scipy.spatial.distance.cdist(
        sph2xyz(grid.nodes.reshape(-1, 3)), 
        sph2xyz(vcells)
    )
    colid = np.argmin(dist, axis=1)
    rowid = np.arange(np.prod(grid.nodes.shape[:-1]))
    
    Gp = scipy.sparse.coo_matrix(
        (np.ones(np.prod(grid.nodes.shape[:-1]),), (rowid, colid)),
        shape=(np.prod(grid.nodes.shape[:-1]), ncell)
    )
    return (vcells, Gp)


def generate_voronoi_cells(grid, ncell):
    '''
    Generate a random set of points representing the centers of Voronoi cells.
    '''
    delta = (grid.max_coords - grid.min_coords)
    return (np.random.rand(ncell, 3) * delta + grid.min_coords)


def find_ray_idx(ray, vcells):
    '''
    Determine the index of the Voronoi cell hosting each point on
    the ray path.
    '''
    dist = scipy.spatial.distance.cdist(sph2xyz(ray), sph2xyz(vcells))
    argmin = np.argmin(dist, axis=1)
    idxs, counts = np.unique(argmin, return_counts=True)
    return (idxs, counts)


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


def station_generator(payload):
    for station_id in payload['df_obs'].index.unique(level='station_id'):
        yield (station_id, payload)
    
    
def iterate_inversion(params, payload):
    '''
    Invert data and update *vmodel*.
    '''
    ds = [] 
    # Iterate over number of realizations.
    df_stations = payload['stations']
    for i in range(params['nreal']):
        print(f'Realization #{i+1}')
        vcells, G_proj  = generate_projection_matrix(payload['vmodel'].grid, params['ncell'])
        df_obs          = sample_observed_data(params, payload)
        station_payload = dict(
            temp_dir    = payload['temp_dir'],
            df_obs      = df_obs,
            df_stations = df_stations,
            vmodel      = payload['vmodel'],
            events      = payload['events'],
            vcells      = vcells
        )
        with mp.Pool(processes=params['nthreads']) as pool:
            pool_output = pool.map(process_station, station_generator(station_payload))

        dobs         = np.array(sum([out[0] for out in pool_output], []))
        col_idx_proj = np.array(sum([out[1] for out in pool_output], []))
        nseg         = np.array(sum([out[2] for out in pool_output], []))
        row_idx_proj = np.array([i for i in range(len(nseg)) for j in range(nseg[i])])
        nonzero_proj = np.array(sum([out[3] for out in pool_output], []))

        G = scipy.sparse.coo_matrix(
            (nonzero_proj, (row_idx_proj, col_idx_proj)), 
            shape=(len(nseg), params['ncell'])
        )
        

        atol    = 1e-3
        btol    = 1e-4
        maxiter = 100
        conlim  = 50
        damp    = 1.0

        x       = scipy.sparse.linalg.lsmr(G, dobs, damp, atol, btol, conlim, maxiter, show=False)[0]
        ds.append(G_proj * x)
    vmodel = payload['vmodel']
    ds = np.mean(ds, axis=0).reshape(vmodel.grid.npts)
    vmodel.velocity = np.power((np.power(vmodel.velocity, -1) + ds), -1)
    return (vmodel)


def load_solver_from_disk(fname):
    solver = pykonal.EikonalSolver(coord_sys='spherical')
    with np.load(fname) as npz:
        solver.vgrid.min_coords     = npz['min_coords']
        solver.vgrid.node_intervals = npz['node_intervals']
        solver.vgrid.npts           = npz['npts']
        solver.uu[...]              = npz['uu']
    return (solver)


def load_solver_from_scratch(payload, station_id, tag=None):
    df_stations = payload['df_stations']
    rho0, theta0, phi0 = geo2sph(
        df_stations.loc[station_id, ['lat', 'lon', 'depth']].values
    )
    far_field  = init_farfield(payload['vmodel'])
    near_field = init_nearfield(far_field, (rho0, theta0, phi0))
    near_field.solve()
    far_field.transfer_travel_times_from(near_field, (-rho0, theta0, phi0), set_alive=True)
    far_field.solve()
    np.savez_compressed(
        os.path.join(
            payload['temp_dir'],
            f'{station_id}.npz' if tag is None else f'{station_id}.{tag}.npz'
        ),
        uu=far_field.uu,
        min_coords=far_field.pgrid.min_coords,
        node_intervals=far_field.pgrid.node_intervals,
        npts=far_field.pgrid.npts
    )
    return (far_field)


def process_station(args):
        station_id, payload = args
        df_obs      = payload['df_obs']
        df_stations = payload['df_stations']
        df_events   = payload['events']
        vmodel      = payload['vmodel']
        events      = payload['events']
        vcells      = payload['vcells']
        colidp      = []
        nsegs       = []
        nonzerop    = []
        dobs        = []
        fname       = os.path.join(payload['temp_dir'], f'{station_id}.npz')
        if os.path.isfile(fname):
            solver = load_solver_from_disk(fname)
        else:
            solver = load_solver_from_scratch(payload, station_id)
        tti = pykonal.LinearInterpolator3D(solver.pgrid, solver.uu)
        for event_id in df_obs.loc[station_id].index.unique():
            event = df_events.loc[event_id]
            try:
                rho_src, theta_src, phi_src = geo2sph(event[['lat', 'lon', 'depth']].values)
                synthetic    = tti((rho_src, theta_src, phi_src))
                residual     = df_obs.loc[(station_id, event_id), 'travel_time'] - synthetic
                if abs(residual) > 3.0:
                    continue
                ray          = solver.trace_ray((rho_src, theta_src, phi_src))
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
        

def load_argc():
    '''
    Return command line arguments.
    '''
    argc              = Arguments()
    argc.event_file   = 'events.h5'
    argc.network_file = 'network.h5'
    argc.output_dir   = os.path.join(os.path.abspath('.'), 'output')
    return (argc)


def load_params():
    '''
    Return parameter file parameters.
    '''
    params = dict(
        latmin   = 32.4,
        lonmin   = -120.1,
        depmin   = -3,
        nlat     = 105,
        nlon     = 127,
        nrad     = 34,
        dlat     = 0.04,
        dlon     = 0.04,
        drad     = 1.0,
        damp     = 0.0,
        datafile = 'scecdc2018.nc',
        phase    = 'P',
        nsamp    = 5000, # Number of observations (arrivals) to sample.
        ncell    = 600,  # Number of Voronoi cells per realization.
        nreal    = 100,    # Number of realizations per iteration.
        niter    = 1,    # Number of iterations.
        nthreads = 24
    )
    return (params)


def load_event_data(argc, params):
    '''
    Read and return *events* and *arrivals* tables from pandas.HDFStore.
    '''
    with pd.HDFStore(argc.event_file) as store:
        df_events   = store['events']
        df_arrivals = store['arrivals']
    df_arrivals['station_id'] = df_arrivals['net'] + '.' + df_arrivals['sta']
    df_arrivals               = df_arrivals = df_arrivals.drop(
        columns=['net', 'sta']
    )
    return (df_events, df_arrivals)


def sanitize_arrivals(df_arrivals, df_stations):
    return (
        df_arrivals[
            df_arrivals['station_id'].isin(df_stations['station_id'].unique())
        ]
    )


def load_network_data(argc):
    '''
    Read and return *stations* table from pandas.HDFStore.
    '''
    with pd.HDFStore(argc.network_file) as store:
        df_stations = store['stations']
    df_stations['station_id'] = df_stations['net'] + '.' + df_stations['sta']
    df_stations               = df_stations.drop(columns=['net', 'sta'])
    df_stations['depth']      = df_stations['elev'] * -1
    return (df_stations)
    

def load_initial_velocity_model(params):
    grid = pykonal.Grid3D(coord_sys='spherical')
    grid.min_coords     = geo2sph(
        (
            params['latmin'] + (params['nlat']-1)*params['dlat'], 
            params['lonmin'], 
            params['depmin'] + (params['nrad']-1)*params['drad']
        )
    )
    grid.node_intervals = (
        params['drad'], np.radians(params['dlat']), np.radians(params['dlon'])
    )
    grid.npts           = params['nrad'], params['nlat'], params['nlon']
    velocity            = 6. * np.ones(grid.npts)
    vmodel              = VelocityModel(grid, velocity)
    return (vmodel)


def sample_observed_data(params, payload):
    df = payload['arrivals'].merge(
        payload['events']['time'],
        left_on='event_id',
        right_index=True,
        suffixes=('_arrival','_origin')
    )
    df['travel_time'] = (df['time_arrival'] - df['time_origin']).dt.total_seconds()
    # Remove any arrivals at stations without metadata
    df = df[df['station_id'].isin(payload['stations'].index.unique())]
    return (
        df.sort_values(
            'station_id'
        ).drop_duplicates( # There shouldn't be any duplicates in a clean data set.
            subset=['station_id', 'event_id'], 
            keep=False
        ).set_index(
            ['station_id', 'event_id']
        ).drop(
            columns=['chan', 'time_arrival', 'time_origin']
        ).sample(
            n=params['nsamp']
        )
    )


def load_velocity_from_file(fname):
    grid = pykonal.Grid3D(coord_sys='spherical')
    with np.load(fname) as infile:
        grid.min_coords     = infile['min_coords']
        grid.node_intervals = infile['node_intervals']
        grid.npts           = infile['npts']
        vmodel              = VelocityModel(grid=grid, velocity=infile['vv'])
    return (vmodel)


if __name__ == '__main__':
    main()