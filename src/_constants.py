import numpy as np

EARTH_RADIUS              = 6371.
ROOT_RANK                 = 0
DISPATCH_REQUEST_TAG      = 100
DISPATCH_TRANSMISSION_TAG = 101
DTYPE_INT                 = np.int64
DTYPE_REAL                = np.float64

EVENT_DTYPES = dict(
    event_id=np.int64,
    latitude=np.float64,
    longitude=np.float64,
    depth=np.float64,
    time=np.float64,
    residual=np.float64
)

ARRIVAL_DTYPES = dict(
    event_id=np.int64,
    network=str,
    station=str,
    phase=str,
    time=np.float64,
    residual=np.float64
)
