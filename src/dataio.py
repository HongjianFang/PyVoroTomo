"""
A module defining basic I/O functions.

.. author:: Malcolm C. A. White
.. date:: 2020-04-17
"""

import numpy as np
import pandas as pd

import _constants
import _picklable

def parse_event_data(argc):
    """
    Parse and return event data (origins and phases) specified on the
    command line.

    Data are returned as a two-tuple of pandas.DataFrame objects. The
    first entry is the origin data and the second is the phase data.

    The input file is expected to be a HDF5 file readable using
    pandas.HDFStore. The input file should have two tables: "events"
    and "arrivals". The "events" table needs to have "latitude",
    "longitude", "depth", "time", and "event_id" columns. The "arrivals"
    table needs to have "network", "station", "phase", "time", and
    "event_id" columns.
    """
    events = pd.read_hdf(argc.events, key="events")
    arrivals = pd.read_hdf(argc.events, key="arrivals")

    return (events, arrivals)


def parse_network_geometry(argc):
    """
    Parse and return network-geometry file specified on the
    command line.

    Data are returned as a pandas.DataFrame object.

    The input file is expected to be a HDF5 file readable using
    pandas.HDFStore. The input file needs to have one table: "stations"."
    The "stations" table needs to have "network", "station", "latitude",
    "longitude", and "elevation" fields. "latitude" and "longitude" are
    in degrees and "elevation" is in kilometers.
    """
    network = pd.read_hdf(argc.network, key="stations")

    return (network)


def parse_velocity_models(cfg):
    """
    Parse and return velocity models specified in configuration.

    Velocity models are returned as a two-tuple of
    _picklable.ScalarField3D objects. The first entry is the P-wave and
    the second is the S-wave model.
    """

    lat_min = cfg["model"]["lat_min"]
    lon_min = cfg["model"]["lon_min"]
    depth_min = cfg["model"]["depth_min"]
    nlat = cfg["model"]["nlat"]
    nlon = cfg["model"]["nlon"]
    ndepth = cfg["model"]["ndepth"]
    dlat = cfg["model"]["dlat"]
    dlon = cfg["model"]["dlon"]
    ddepth = cfg["model"]["ddepth"]

    lat_max = lat_min  +  (nlat - 1) * dlat
    depth_max = depth_min  +  (ndepth - 1) * ddepth
    theta_min = np.radians(90 - lat_max)
    phi_min = np.radians(lon_min)
    rho_min = _constants.earth_radius - depth_max
    ntheta = nlat
    nphi = nlon
    nrho = ndepth
    dtheta = np.radians(dlat)
    dphi = np.radians(dlon)
    drho = ddepth

    pwave_model = _picklable.ScalarField3D(coord_sys="spherical")
    swave_model = _picklable.ScalarField3D(coord_sys="spherical")

    for model in (pwave_model, swave_model):
        model.min_coords = rho_min, theta_min, phi_min
        model.node_intervals = drho, dtheta, dphi
        model.npts = nrho, ntheta, nphi

    with np.load(cfg["model"]["initial_pwave_path"]) as npz:
        vp = npz["velocity"]
    with np.load(cfg["model"]["initial_swave_path"]) as npz:
        vs = npz["velocity"]

    vp = np.rollaxis(vp,  2)
    vs = np.rollaxis(vs,  2)
    vp = np.flip(vp, axis=(0, 1))
    vs = np.flip(vs, axis=(0, 1))

    pwave_model.values = vp
    swave_model.values = vs

    return (pwave_model, swave_model)
