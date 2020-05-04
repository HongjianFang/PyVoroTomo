"""
A module defining basic I/O functions.

.. author:: Malcolm C. A. White
.. date:: 2020-04-17
"""

import numpy as np
import pandas as pd
import pykonal

from . import _constants
from . import _picklable

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

    for field in _constants.EVENT_FIELDS:
        if field not in events.columns:
            error = ValueError(
                f"Input event data must have the following fields: "
                f"{_constants.EVENT_FIELDS}"
            )
            raise (error)

    for field in _constants.ARRIVAL_FIELDS:
        if field not in arrivals.columns:
            error = ValueError(
                f"Input arrival data must have the following fields: "
                f"{_constants.ARRIVAL_FIELDS}"
            )
            raise (error)

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
    in degrees and "elevation" is in kilometers. The returned DataFrame
    has "network", "station", "latitude", "longitude", and "depth"
    columns.
    """

    network = pd.read_hdf(argc.network, key="stations")
    network["depth"] = -network["elevation"]
    network = network.drop(columns=["elevation"])

    return (network)


def parse_velocity_models(cfg):
    """
    Parse and return velocity models specified in configuration.

    Velocity models are returned as a two-tuple of
    _picklable.ScalarField3D objects. The first entry is the P-wave and
    the second is the S-wave model.
    """

    pwave_model = _picklable.ScalarField3D(coord_sys="spherical")
    swave_model = _picklable.ScalarField3D(coord_sys="spherical")


    path = cfg["model"]["initial_pwave_path"]
    _pwave_model = pykonal.fields.load(path)

    path = cfg["model"]["initial_swave_path"]
    _swave_model = pykonal.fields.load(path)

    models  = (pwave_model, swave_model)
    _models = (_pwave_model, _swave_model)
    for model, _model in zip(models, _models):
        model.min_coords = _model.min_coords
        model.node_intervals = _model.node_intervals
        model.npts = _model.npts
        model.values = _model.values

    return (pwave_model, swave_model)
