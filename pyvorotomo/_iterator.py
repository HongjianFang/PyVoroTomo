import h5py as h5
import KDEpy as kp
import mpi4py.MPI as MPI
import numpy as np
import os
import pandas as pd
import pykonal
import scipy.sparse
import scipy.spatial
import shutil
import tempfile

from . import _dataio
from . import _clustering
from . import _constants
from . import _picklable
from . import _utilities

# Get logger handle.
logger = _utilities.get_logger(f"__main__.{__name__}")

# Define aliases.
PointSourceSolver = pykonal.solver.PointSourceSolver
geo2sph = pykonal.transformations.geo2sph
sph2geo = pykonal.transformations.sph2geo
sph2xyz = pykonal.transformations.sph2xyz
xyz2sph = pykonal.transformations.xyz2sph

COMM       = MPI.COMM_WORLD
RANK       = COMM.Get_rank()
WORLD_SIZE = COMM.Get_size()
ROOT_RANK  = _constants.ROOT_RANK


class InversionIterator(object):
    """
    A class providing core functionality for iterating inversion
    procedure.
    """

    def __init__(self, argc):
        self._argc = argc
        self._arrivals = None
        self._cfg = None
        self._events = None
        self._iiter = 0
        self._phases = None
        self._projection_matrix = None
        self._pwave_model = None
        self._swave_model = None
        self._pwave_realization_stack = None
        self._swave_realization_stack = None
        self._pwave_variance = None
        self._swave_variance = None
        self._residuals = None
        self._sensitivity_matrix = None
        self._stations = None
        self._step_size = None
        self._sampled_arrivals = None
        self._voronoi_cells = None
        if RANK == ROOT_RANK:
            scratch_dir = argc.scratch_dir
            self._scratch_dir_obj = tempfile.TemporaryDirectory(dir=scratch_dir)
            self._scratch_dir = self._scratch_dir_obj.name
        self.synchronize(attrs=["scratch_dir"])

    @property
    def argc(self):
        return (self._argc)

    @property
    def arrivals(self):
        return (self._arrivals)

    @arrivals.setter
    def arrivals(self, value):
        self._arrivals = value

    @property
    def cfg(self):
        return (self._cfg)

    @cfg.setter
    def cfg(self, value):
        self._cfg = value

    @property
    def events(self):
        return (self._events)

    @events.setter
    def events(self, value):
        value = value.sort_values("event_id")
        value = value.reset_index(drop=True)
        self._events = value

    @property
    def iiter(self):
        return (self._iiter)

    @iiter.setter
    def iiter(self, value):
        self._iiter = value

    @property
    def phases(self):
        return (self._phases)

    @phases.setter
    def phases(self, value):
        self._phases = value

    @property
    def projection_matrix(self):
        return (self._projection_matrix)

    @projection_matrix.setter
    def projection_matrix(self, value):
        self._projection_matrix = value

    @property
    def pwave_model(self) -> _picklable.ScalarField3D:
        return (self._pwave_model)

    @pwave_model.setter
    def pwave_model(self, value):
        self._pwave_model = value

    @property
    def pwave_realization_stack(self):
        if self._pwave_realization_stack is None:
           self._pwave_realization_stack = []
        return (self._pwave_realization_stack)

    @pwave_realization_stack.setter
    def pwave_realization_stack(self, value):
        self._pwave_realization_stack = value

    @property
    def pwave_variance(self) -> _picklable.ScalarField3D:
        if self._pwave_variance is None:
            field = _picklable.ScalarField3D(coord_sys="spherical")
            field.min_coords = self.pwave_model.min_coords
            field.node_intervals = self.pwave_model.node_intervals
            field.npts = self.pwave_model.npts
            self._pwave_variance = field
        return (self._pwave_variance)

    @pwave_variance.setter
    def pwave_variance(self, value: _picklable.ScalarField3D):
        self._pwave_variance = value

    @property
    def raypath_dir(self):
        return (os.path.join(self.scratch_dir, "raypaths"))

    @property
    def residuals(self):
        return (self._residuals)

    @residuals.setter
    def residuals(self, value):
        self._residuals = value

    @property
    def sampled_arrivals(self):
        return (self._sampled_arrivals)

    @sampled_arrivals.setter
    def sampled_arrivals(self, value):
        self._sampled_arrivals = value

    @property
    def scratch_dir(self):
        return (self._scratch_dir)

    @scratch_dir.setter
    def scratch_dir(self, value):
        self._scratch_dir = value

    @property
    def sensitivity_matrix(self):
        return (self._sensitivity_matrix)

    @sensitivity_matrix.setter
    def sensitivity_matrix(self, value):
        self._sensitivity_matrix = value

    @property
    def stations(self):
        return (self._stations)

    @stations.setter
    def stations(self, value):
        self._stations = value

    @property
    def step_size(self):
        return (self._step_size)

    @step_size.setter
    def step_size(self, value):
        self._step_size = value

    @property
    def swave_model(self):
        return (self._swave_model)

    @swave_model.setter
    def swave_model(self, value):
        self._swave_model = value

    @property
    def swave_realization_stack(self):
        if self._swave_realization_stack is None:
           self._swave_realization_stack = []
        return (self._swave_realization_stack)

    @swave_realization_stack.setter
    def swave_realization_stack(self, value):
        self._swave_realization_stack = value

    @property
    def swave_variance(self) -> _picklable.ScalarField3D:
        if self._swave_variance is None:
            field = _picklable.ScalarField3D(coord_sys="spherical")
            field.min_coords = self.pwave_model.min_coords
            field.node_intervals = self.pwave_model.node_intervals
            field.npts = self.pwave_model.npts
            self._swave_variance = field
        return (self._swave_variance)

    @swave_variance.setter
    def swave_variance(self, value: _picklable.ScalarField3D):
        self._swave_variance = value

    @property
    def traveltime_dir(self):
        return (os.path.join(self.scratch_dir, "traveltimes"))

    @property
    def voronoi_cells(self):
        return (self._voronoi_cells)

    @voronoi_cells.setter
    def voronoi_cells(self, value):
        self._voronoi_cells = value


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def _compute_model_update(self, phase):
        """
        Compute the model update for a single realization and appends
        the results to the realization stack.

        Only the root rank performs this operation.
        """

        logger.info(f"Computing {phase}-wave model update")

        if phase == "P":
            model = self.pwave_model
        elif phase == "S":
            model = self.swave_model
        else:
            raise (ValueError(f"Unrecognized phase ({phase}) supplied."))

        damp = self.cfg["algorithm"]["damp"]
        atol = self.cfg["algorithm"]["atol"]
        btol = self.cfg["algorithm"]["btol"]
        conlim = self.cfg["algorithm"]["conlim"]
        maxiter = self.cfg["algorithm"]["maxiter"]

        result = scipy.sparse.linalg.lsmr(
            self.sensitivity_matrix,
            self.residuals,
            damp,
            atol,
            btol,
            conlim,
            maxiter,
            show=False
        )
        x, istop, itn, normr, normar, norma, conda, normx = result
        delta_slowness = self.projection_matrix * x
        delta_slowness = delta_slowness.reshape(model.npts)
        slowness = np.power(model.values, -1) + delta_slowness
        velocity = np.power(slowness, -1)

        if phase == "P":
            self.pwave_realization_stack.append(velocity)
        else:
            self.swave_realization_stack.append(velocity)

        return (True)


    @_utilities.log_errors(logger)
    def _compute_sensitivity_matrix(self, phase, nvoronoi):
        """
        Compute the sensitivity matrix.
        """

        logger.info(f"Computing {phase}-wave sensitivity matrix")

        raypath_dir = self.raypath_dir

        index_keys = ["network", "station"]
        arrivals = self.sampled_arrivals.set_index(index_keys)

        arrivals = arrivals.sort_index()

        if RANK == ROOT_RANK:
            ids = arrivals.index.unique()
            self._dispatch(ids)

            logger.debug("Compiling sensitivity matrix.")
            column_idxs = COMM.gather(None, root=ROOT_RANK)
            nsegments = COMM.gather(None, root=ROOT_RANK)
            nonzero_values = COMM.gather(None, root=ROOT_RANK)
            residuals = COMM.gather(None, root=ROOT_RANK)

            column_idxs = list(filter(lambda x: x is not None, column_idxs))
            nsegments = list(filter(lambda x: x is not None, nsegments))
            nonzero_values = list(filter(lambda x: x is not None, nonzero_values))
            residuals = list(filter(lambda x: x is not None, residuals))


            column_idxs = np.concatenate(column_idxs)
            nonzero_values = np.concatenate(nonzero_values)
            residuals = np.concatenate(residuals)
            nsegments = np.concatenate(nsegments)

            row_idxs = [
                i for i in range(len(nsegments))
                  for j in range(nsegments[i])
            ]
            row_idxs = np.array(row_idxs)

            matrix = scipy.sparse.coo_matrix(
                (nonzero_values, (row_idxs, column_idxs)),
                shape=(len(nsegments), nvoronoi)
            )

            self.sensitivity_matrix = matrix
            self.residuals = residuals

        else:


            column_idxs = np.array([], dtype=_constants.DTYPE_INT)
            nsegments = np.array([], dtype=_constants.DTYPE_INT)
            nonzero_values = np.array([], dtype=_constants.DTYPE_REAL)
            residuals = np.array([], dtype=_constants.DTYPE_REAL)

            step_size = self.step_size
            events = self.events.set_index("event_id")
            events["idx"] = range(len(events))

            while True:

                item = self._request_dispatch()

                if item is None:
                    logger.debug("Sentinel received. Gathering sensitivity matrix.")

                    column_idxs = COMM.gather(column_idxs, root=ROOT_RANK)
                    nsegments = COMM.gather(nsegments, root=ROOT_RANK)
                    nonzero_values = COMM.gather(nonzero_values, root=ROOT_RANK)
                    residuals = COMM.gather(residuals, root=ROOT_RANK)

                    break

                network, station = item

                # Get the subset of arrivals belonging to this station.
                _arrivals = arrivals.loc[(network, station)]
                _arrivals = _arrivals.set_index("event_id")

                # Open the raypath file.
                filename = f"{network}.{station}.{phase}.h5"
                path = os.path.join(raypath_dir, filename)
                raypath_file = h5.File(path, mode="r")

                for event_id, arrival in _arrivals.iterrows():

                    event = events.loc[event_id]
                    idx = int(event["idx"])

                    raypath = raypath_file[phase][:, idx]
                    raypath = np.stack(raypath).T

                    _column_idxs, counts = self._projected_ray_idxs(raypath)
                    column_idxs = np.append(column_idxs, _column_idxs)
                    nsegments = np.append(nsegments, len(_column_idxs))
                    nonzero_values = np.append(nonzero_values, counts * step_size)
                    residuals = np.append(residuals, arrival["residual"])

                raypath_file.close()

        COMM.barrier()

        return (True)


    @_utilities.log_errors(logger)
    def _dispatch(self, ids, sentinel=None):
        """
        Dispatch ids to hungry workers, then dispatch sentinels.
        """

        logger.debug("Dispatching ids")

        for _id in ids:
            requesting_rank = COMM.recv(
                source=MPI.ANY_SOURCE,
                tag=_constants.DISPATCH_REQUEST_TAG
            )
            COMM.send(
                _id,
                dest=requesting_rank,
                tag=_constants.DISPATCH_TRANSMISSION_TAG
            )
        # Distribute sentinel.
        for irank in range(WORLD_SIZE - 1):
            requesting_rank = COMM.recv(
                source=MPI.ANY_SOURCE,
                tag=_constants.DISPATCH_REQUEST_TAG
            )
            COMM.send(
                sentinel,
                dest=requesting_rank,
                tag=_constants.DISPATCH_TRANSMISSION_TAG
            )

        return (True)


    @_utilities.log_errors(logger)
    def _generate_voronoi_cells(self, phase, nvoronoi):
        """
        Generate Voronoi cells using k-medians clustering of raypaths.
        """

        logger.debug(f"Generating {nvoronoi} Voronoi cells using k-medians clustering.")

        if RANK == ROOT_RANK:

            k_medians_npts = self.cfg["algorithm"]["k_medians_npts"]

            raypaths = []
            raypath_dir = self.raypath_dir

            columns = ["network", "station"]
            arrivals = self.sampled_arrivals.set_index(columns)
            arrivals = arrivals.sort_index()
            index = arrivals.index.unique()

            events = self.events.set_index("event_id")
            events["idx"] = range(len(events))

            for network, station in index:

                _arrivals = arrivals.loc[(network, station)]
                _arrivals = _arrivals.set_index("event_id")

                # Open the raypath file.
                filename = f"{network}.{station}.{phase}.h5"
                path = os.path.join(raypath_dir, filename)
                raypath_file = h5.File(path, mode="r")

                for event_id, arrival in _arrivals.iterrows():

                    event = events.loc[event_id]
                    idx = int(event["idx"])

                    raypath = raypath_file[phase][:, idx]
                    raypath = np.stack(raypath).T

                    raypaths.append(raypath)

            points = np.concatenate(raypaths)

            medians = _clustering.k_medians(
                nvoronoi,
                points,
                npts=k_medians_npts
            )

            self.voronoi_cells = medians

        self.synchronize(attrs=["voronoi_cells"])

        return (True)


    @_utilities.log_errors(logger)
    def _projected_ray_idxs(self, raypath):
        """
        Return the cell IDs (column IDs) of each segment of the given
        raypath and the length of each segment in counts.
        """

        voronoi_cells = sph2xyz(self.voronoi_cells, (0, 0, 0))
        tree = scipy.spatial.cKDTree(voronoi_cells)
        raypath = sph2xyz(raypath, (0, 0, 0))
        _, column_idxs = tree.query(raypath)
        column_idxs, counts = np.unique(column_idxs, return_counts=True)

        return (column_idxs, counts)


    @_utilities.log_errors(logger)
    def _request_dispatch(self):
        """
        Request, receive, and return item from dispatcher.
        """
        COMM.send(
            RANK,
            dest=ROOT_RANK,
            tag=_constants.DISPATCH_REQUEST_TAG
        )
        item = COMM.recv(
            source=ROOT_RANK,
            tag=_constants.DISPATCH_TRANSMISSION_TAG
        )

        return (item)


    @_utilities.log_errors(logger)
    def _sample_arrivals(self, phase):
        """
        Draw a random sample of arrivals and update the
        "sampled_arrivals" attribute.
        """

        if RANK == ROOT_RANK:
            tukey_k = self.cfg["algorithm"]["outlier_removal_factor"]
            narrival = self.cfg["algorithm"]["narrival"]

            # Subset for the appropriate phase.
            arrivals = self.arrivals.set_index("phase")
            arrivals = arrivals.sort_index()
            arrivals = arrivals.loc[phase]

            # Remove outliers.
            arrivals = remove_outliers(arrivals, tukey_k, "residual")

            # Sample arrivals.
            arrivals = arrivals.sample(n=narrival, weights="weight")

            self.sampled_arrivals = arrivals

        self.synchronize(attrs=["sampled_arrivals"])

        return (True)


    @_utilities.log_errors(logger)
    def _trace_rays(self, phase):
        """
        Trace rays for all arrivals in self.sampled_arrivals and store
        in HDF5 file. Only trace non-existent raypaths.
        """

        logger.info("Tracing rays.")

        raypath_dir = self.raypath_dir
        traveltime_dir = self.traveltime_dir
        arrivals = self.sampled_arrivals
        arrivals = arrivals.set_index(["network", "station"])
        arrivals = arrivals.sort_index()

        if RANK == ROOT_RANK:

            os.makedirs(raypath_dir, exist_ok=True)
            index = arrivals.index.unique()
            self._dispatch(index)

        else:

            events = self.events
            events = events.set_index("event_id")
            events["idx"] = range(len(events))

            while True:

                item = self._request_dispatch()

                if item is None:
                    logger.debug("Sentinel received.")
                    break

                network, station = item
                filename = f"{network}.{station}.{phase}"

                path = os.path.join(traveltime_dir, filename + ".npz")
                traveltime = pykonal.fields.load(path)

                path = os.path.join(raypath_dir, filename + ".h5")
                raypath_file = h5.File(path, mode="a")

                if phase not in raypath_file:
                    dtype = h5.vlen_dtype(_constants.DTYPE_REAL)
                    dataset = raypath_file.create_dataset(
                        phase,
                        (3, len(events),),
                        dtype=dtype
                    )
                else:
                    dataset = raypath_file[phase]

                event_ids = arrivals.loc[(network, station), "event_id"].values

                for event_id in event_ids:

                    event = events.loc[event_id]
                    idx = int(event["idx"])

                    if np.stack(dataset[:, idx]).size != 0:
                        continue

                    columns = ["latitude", "longitude", "depth"]
                    coords = event[columns]
                    coords = geo2sph(coords)
                    raypath = traveltime.trace_ray(coords)
                    dataset[:, idx] = raypath.T.copy()

                raypath_file.close()

        COMM.barrier()

        return (True)


    @_utilities.log_errors(logger)
    def _update_arrival_weights(
        self,
        phase: str,
        npts: int=16,
        bandwidth: int=0.1
    ) -> bool:
        """
        Update arrival weights using KDE.
        """

        logger.info("Updating weights for homogeneous raypath sampling.")

        if RANK == ROOT_RANK:
            arrivals = self.arrivals
            arrivals = arrivals[arrivals["phase"] == phase]

            # Merge event data.
            events = self.events.rename(
                columns={
                    "latitude": "event_latitude",
                    "longitude": "event_longitude",
                    "depth": "event_depth"
                }
            )

            merge_columns = [
                "event_latitude",
                "event_longitude",
                "event_depth",
                "event_id"
            ]

            arrivals = arrivals.merge(events[merge_columns], on="event_id")

            # Merge station data.
            stations = self.stations.rename(
                columns={
                    "latitude": "station_latitude",
                    "longitude": "station_longitude"
                }
            )

            merge_columns = [
                "station_latitude",
                "station_longitude",
                "network",
                "station"
            ]
            merge_keys = ["network", "station"]
            arrivals = arrivals.merge(stations[merge_columns], on=merge_keys)

            # Compute station-to-event azimuth and epicentral distance.
            dlat = arrivals["event_latitude"] - arrivals["station_latitude"]
            dlon = arrivals["event_longitude"] - arrivals["station_longitude"]
            arrivals["azimuth"] = np.arctan2(dlat, dlon)
            arrivals["delta"] = np.sqrt(dlat ** 2  +  dlon ** 2)

            # Extract the data for KDE fitting.
            kde_columns = [
                "event_latitude",
                "event_longitude",
                "event_depth",
                "azimuth",
                "delta"
            ]
            ndim = len(kde_columns)
            data = arrivals[kde_columns].values

            # Normalize the data.
            data_min = data.min(axis=0)
            data_max = data.max(axis=0)
            data_range = data_max - data_min
            data_delta = data - data_min
            data = data_delta / data_range

            # Fit and evaluate the KDE.
            kde = kp.FFTKDE(bw=bandwidth).fit(data)
            points, values = kde.evaluate(npts)
            points = [np.unique(points[:,iax]) for iax in range(ndim)]
            values = values.reshape((npts,) * ndim)

            # Initialize an interpolator because FFTKDE is evaluated on a
            # regular grid.
            interpolator = scipy.interpolate.RegularGridInterpolator(points, values)

            # Assign weights to the arrivals.
            arrivals["weight"] = 1 / interpolator(data)

            # Update the self.arrivals attribute with weights.
            index_columns = ["network", "station", "event_id", "phase"]
            arrivals = arrivals.set_index(index_columns)
            _arrivals = self.arrivals.set_index(index_columns)
            _arrivals = _arrivals.sort_index()
            idx = arrivals.index
            _arrivals.loc[idx, "weight"] = arrivals["weight"]
            _arrivals = _arrivals.reset_index()
            self.arrivals = _arrivals

            logger.debug(self.arrivals)

        self.synchronize(attrs=["arrivals"])

        return (True)


    @_utilities.log_errors(logger)
    def _update_projection_matrix(self, nvoronoi):
        """
        Update the projection matrix using the current Voronoi cells.
        """

        logger.info("Updating projection matrix")

        if RANK == ROOT_RANK:
            voronoi_cells = sph2xyz(self.voronoi_cells, origin=(0,0,0))
            tree = scipy.spatial.cKDTree(voronoi_cells)
            nodes = self.pwave_model.nodes.reshape(-1, 3)
            nodes = sph2xyz(nodes, origin=(0,0,0))
            _, column_ids = tree.query(nodes)

            nnodes = np.prod(self.pwave_model.nodes.shape[:-1])
            row_ids = np.arange(nnodes)

            values = np.ones(nnodes,)
            self.projection_matrix = scipy.sparse.coo_matrix(
                (values, (row_ids, column_ids)),
                shape=(nnodes, nvoronoi)
            )

        self.synchronize(attrs=["projection_matrix"])

        return (True)


    @_utilities.log_errors(logger)
    def compute_traveltime_lookup_tables(self):
        """
        Compute traveltime-lookup tables.
        """

        logger.info("Computing traveltime-lookup tables.")

        traveltime_dir = self.traveltime_dir

        logger.debug(f"Working in {traveltime_dir}")

        if RANK == ROOT_RANK:

            os.makedirs(traveltime_dir, exist_ok=True)
            ids = zip(self.stations["network"], self.stations["station"])
            self._dispatch(sorted(ids))

        else:

            geometry = self.stations
            geometry = geometry.set_index(["network", "station"])

            while True:

                # Request an event
                item = self._request_dispatch()

                if item is None:
                    logger.debug("Received sentinel.")

                    break

                network, station = item

                keys = ["latitude", "longitude", "depth"]
                coords = geometry.loc[(network, station), keys]
                coords = geo2sph(coords)

                for phase in self.phases:
                    handle = f"{phase.lower()}wave_model"
                    model = getattr(self, handle)
                    solver = PointSourceSolver(coord_sys="spherical")
                    solver.vv.min_coords = model.min_coords
                    solver.vv.node_intervals = model.node_intervals
                    solver.vv.npts = model.npts
                    solver.vv.values = model.values
                    solver.src_loc = coords
                    solver.solve()
                    path = os.path.join(
                        traveltime_dir,
                        f"{network}.{station}.{phase}.npz"
                    )
                    solver.tt.savez(path)

        COMM.barrier()

        return (True)


    @_utilities.log_errors(logger)
    def iterate(self):
        """
        Execute one iteration the entire inversion procedure including
        updating velocity models, event locations, and arrival residuals.
        """

        output_dir = self.argc.output_dir

        niter = self.cfg["algorithm"]["niter"]
        nfib = self.cfg["algorithm"]["nfib"]
        nrep = self.cfg["algorithm"]["nrep"]
        homogenize_raypaths = self.cfg["algorithm"]["homogenize_raypaths"]

        self.iiter += 1

        logger.info(f"Iteration #{self.iiter} (/{niter}).")

        for phase in self.phases:
            logger.info(f"Updating {phase}-wave model")
            self._update_arrival_weights(phase)
            for ifib in range(nfib):
                nvoronoi = _clustering.fibonacci(ifib + 1)
                for irep in range(nrep):
                    logger.info(f"Repetition #{irep+1} (/{nrep}) for Fibonacci #{ifib+1} (/{nfib})")
                    self._sample_arrivals(phase)
                    self._trace_rays(phase)
                    self._generate_voronoi_cells(
                        phase,
                        nvoronoi
                    )
                    self._update_projection_matrix(nvoronoi)
                    self._compute_sensitivity_matrix(phase, nvoronoi)
                    self._compute_model_update(phase)
            self.update_model(phase)
            self.save_model(phase)
        self.compute_traveltime_lookup_tables()
        self.purge_raypaths()
        self.relocate_events()
        self.update_arrival_residuals()
        self.save_events()


    @_utilities.log_errors(logger)
    def load_cfg(self):
        """
        Parse and store configuration-file parameters.

        ROOT_RANK parses configuration file and broadcasts contents to all
        other processes.
        """

        logger.info("Loading configuration-file parameters.")

        if RANK == ROOT_RANK:

            # Parse configuration-file parameters.
            self.cfg = _utilities.parse_cfg(self.argc.configuration_file)
            _utilities.write_cfg(self.argc, self.cfg)

        self.synchronize(attrs=["cfg"])

        return (True)


    @_utilities.log_errors(logger)
    def load_event_data(self):
        """
        Parse and return event data from file.

        ROOT_RANK parses file and broadcasts contents to all other
        processes.
        """

        logger.info("Loading event data.")

        if RANK == ROOT_RANK:

            # Parse event data.
            data = _dataio.parse_event_data(self.argc)
            self.events, self.arrivals = data

            # Register the available phase types.
            phases = self.arrivals["phase"]
            phases = phases.unique()
            self.phases = sorted(phases)

        self.synchronize(attrs=["events", "arrivals", "phases"])

        return (True)


    @_utilities.log_errors(logger)
    def load_network_geometry(self):
        """
        Parse and return network geometry from file.

        ROOT_RANK parses file and broadcasts contents to all other
        processes.
        """

        logger.info("Loading network geometry")

        if RANK == ROOT_RANK:

            # Parse event data.
            stations = _dataio.parse_network_geometry(self.argc)
            self.stations = stations

        self.synchronize(attrs=["stations"])

        return (True)


    @_utilities.log_errors(logger)
    def load_velocity_models(self):
        """
        Parse and return velocity models from file.

        ROOT_RANK parses file and broadcasts contents to all other
        processes.
        """

        logger.info("Loading velocity models.")

        if RANK == ROOT_RANK:

            # Parse velocity model files.
            velocity_models = _dataio.parse_velocity_models(self.cfg)
            self.pwave_model, self.swave_model = velocity_models
            self.step_size = self.pwave_model.step_size

        self.synchronize(attrs=["pwave_model", "swave_model", "step_size"])

        return (True)


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def purge_raypaths(self):
        """
        Destroys all stored raypaths.
        """

        logger.debug("Purging raypath directory.")

        shutil.rmtree(self.raypath_dir)
        os.makedirs(self.raypath_dir)

        return (True)


    @_utilities.log_errors(logger)
    def relocate_events(self):
        """
        Relocate all events and update the "events" attribute.
        """

        logger.info("Relocating events.")

        traveltime_dir = self.traveltime_dir
        if RANK == ROOT_RANK:
            ids = self.events["event_id"]
            self._dispatch(sorted(ids))

            logger.debug("Dispatch complete. Gathering events.")
            # Gather and concatenate events from all workers.
            events = COMM.gather(None, root=ROOT_RANK)
            events = pd.concat(events, ignore_index=True)
            events = events.convert_dtypes()
            self.events = events

        else:
            # Define columns to output.
            columns = [
                "latitude",
                "longitude",
                "depth",
                "time",
                "residual",
                "event_id"
            ]


            # Initialize EQLocator object.
            locator = pykonal.locate.EQLocator(
                station_dict(self.stations),
                tt_dir=traveltime_dir
            )
            locator.grid.min_coords     = self.pwave_model.min_coords
            locator.grid.node_intervals = self.pwave_model.node_intervals
            locator.grid.npts           = self.pwave_model.npts
            locator.pwave_velocity      = self.pwave_model.values
            locator.swave_velocity      = self.swave_model.values

            # Create some aliases for configuration-file parameters.
            dlat = self.cfg["locate"]["dlat"]
            dlon = self.cfg["locate"]["dlon"]
            dz = self.cfg["locate"]["ddepth"]
            dt = self.cfg["locate"]["dtime"]

            events = pd.DataFrame()

            while True:

                # Request an event
                event_id = self._request_dispatch()

                if event_id is None:
                    logger.debug("Received sentinel, gathering events.")
                    COMM.gather(events, root=ROOT_RANK)

                    break

                logger.debug(f"Received event ID #{event_id}")

                # Clear arrivals from previous event.
                locator.clear_arrivals()
                locator.add_arrivals(arrival_dict(self.arrivals, event_id))
                locator.load_traveltimes()
                loc = locator.locate(dlat=dlat, dlon=dlon, dz=dz, dt=dt)

                # Get residual RMS, reformat result, and append to events
                # DataFrame.
                rms = locator.rms(loc)
                loc[:3] = sph2geo(loc[:3])
                event = pd.DataFrame(
                    [np.concatenate((loc, [rms, event_id]))],
                    columns=columns
                )
                events = events.append(event, ignore_index=True)

        self.synchronize(attrs=["events"])

        return (True)



    @_utilities.log_errors(logger)
    def sanitize_data(self):
        """
        Sanitize input data.
        """

        logger.info("Sanitizing data.")

        if RANK == ROOT_RANK:

            # Drop duplicate stations.
            self.stations = self.stations.drop_duplicates(["network", "station"])

            # Drop stations without arrivals.
            logger.debug("Dropping stations without arrivals.")
            arrivals = self.arrivals.set_index(["network", "station"])
            idx_keep = arrivals.index.unique()
            stations = self.stations.set_index(["network", "station"])
            stations = stations.loc[idx_keep]
            stations = stations.reset_index()
            self.stations = stations

        self.synchronize(attrs=["stations"])

        return (True)


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_events(self):
        """
        Save the current "events", and "arrivals" to and HDF5 file using
        pandas.HDFStore.
        """

        logger.info(f"Saving event data from iteration #{self.iiter}")

        path = os.path.join(self.argc.output_dir, f"{self.iiter:02d}")

        events       = self.events
        EVENT_DTYPES = _constants.EVENT_DTYPES
        for column in EVENT_DTYPES:

            events[column] = events[column].astype(EVENT_DTYPES[column])

        arrivals       = self.arrivals
        ARRIVAL_DTYPES = _constants.ARRIVAL_DTYPES
        for column in ARRIVAL_DTYPES:
            arrivals[column] = arrivals[column].astype(ARRIVAL_DTYPES[column])

        events.to_hdf(f"{path}.events.h5", key="events")
        arrivals.to_hdf(f"{path}.events.h5", key="arrivals")

        return(True)


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_model(self, phase: str) -> bool:

        logger.info(f"Saving {phase}-wave model for iteration #{self.iiter}")

        phase = phase.lower()
        path = os.path.join(self.argc.output_dir, f"{self.iiter:02d}")

        handle = f"{phase}wave_model"
        model = getattr(self, handle)
        model.savez(path + f".{handle}")

        if self.iiter == 0:

            return (True)

        handle = f"{phase}wave_variance"
        model = getattr(self, handle)
        model.savez(path + f".{handle}")

        handle = f"{phase}wave_realization_stack"
        if self.argc.output_realizations is True:
            stack = getattr(self, handle)
            stack = np.stack(stack)
            np.savez(
                path + f".{handle}.npz",
                data=stack
            )
        setattr(self, handle, None)

        return (True)


    @_utilities.log_errors(logger)
    def synchronize(self, attrs="all"):
        """
        Synchronize input data across all processes.

        "attrs" may be an iterable of attribute names to synchronize.
        """


        _all = (
            "arrivals",
            "cfg",
            "events",
            "projection_matrix",
            "pwave_model",
            "swave_model",
            "sampled_arrivals",
            "stations",
            "step_size",
            "voronoi_cells"
        )

        if attrs == "all":
            attrs = _all

        for attr in attrs:
            value = getattr(self, attr) if RANK == ROOT_RANK else None
            value = COMM.bcast(value, root=ROOT_RANK)
            setattr(self, attr, value)

        COMM.barrier()

        return (True)


    @_utilities.log_errors(logger)
    def update_arrival_residuals(self):
        """
        Compute arrival-time residuals based on current event locations
        and velocity models, and update "residual" columns of "arrivals"
        attribute.
        """

        logger.info("Updating arrival residuals.")

        traveltime_dir = self.traveltime_dir
        arrivals = self.arrivals.set_index(["network", "station", "phase"])
        arrivals = arrivals.sort_index()

        if RANK == ROOT_RANK:
            ids = arrivals.index.unique()
            self._dispatch(ids)
            logger.debug("Dispatch complete. Gathering arrivals.")
            arrivals = COMM.gather(None, root=ROOT_RANK)
            arrivals = pd.concat(arrivals, ignore_index=True)
            arrivals = arrivals.convert_dtypes()
            self.arrivals = arrivals

        else:

            events = self.events.set_index("event_id")
            updated_arrivals = pd.DataFrame()

            while True:

                # Request an event
                item = self._request_dispatch()

                if item is None:
                    logger.debug("Received sentinel. Gathering arrivals.")
                    COMM.gather(updated_arrivals, root=ROOT_RANK)

                    break


                network, station, phase = item
                logger.debug(f"Updating {phase}-wave residuals for {network}.{station}.")

                path = os.path.join(traveltime_dir, f"{network}.{station}.{phase}.npz")
                traveltime = pykonal.fields.load(path)

                _arrivals = arrivals.loc[(network, station, phase)]
                _arrivals = _arrivals.set_index("event_id")

                for event_id, arrival in _arrivals.iterrows():
                    arrival_time = arrival["time"]
                    origin_time = events.loc[event_id, "time"]
                    coords = events.loc[event_id, ["latitude", "longitude", "depth"]]
                    coords = geo2sph(coords)
                    residual = arrival_time - (origin_time + traveltime.value(coords))
                    arrival = dict(
                        network=network,
                        station=station,
                        phase=phase,
                        event_id=event_id,
                        time=arrival_time,
                        residual=residual
                    )
                    arrival = pd.DataFrame([arrival])
                    updated_arrivals = updated_arrivals.append(arrival, ignore_index=True)

        self.synchronize(attrs=["arrivals"])

        return (True)

    @_utilities.log_errors(logger)
    def update_model(self, phase):
        """
        Stack random realizations to obtain average model and update
        appropriate attributes.
        """

        phase = phase.lower()

        if RANK == ROOT_RANK:

            handle = f"{phase}wave_realization_stack"
            stack = getattr(self, handle)
            stack = np.stack(stack)
            values = np.mean(stack, axis=0)
            variance = np.var(stack, axis=0)

            handle = f"{phase}wave_model"
            model = getattr(self, handle)
            model.values = values

            handle = f"{phase}wave_variance"
            model = getattr(self, handle)
            model.values = variance

        attrs = [
            f"{phase}wave_model",
            f"{phase}wave_realization_stack",
            f"{phase}wave_variance"
        ]
        self.synchronize(attrs=attrs)

        return (True)



@_utilities.log_errors(logger)
def arrival_dict(dataframe, event_id):
    """
    Return a dictionary with phase-arrival data suitable for passing to
    the EQLocator.add_arrivals() method.

    Returned dictionary has ("station_id", "phase") keys, where
    "station_id" = f"{network}.{station}", and values are
    phase-arrival timestamps.
    """

    dataframe = dataframe.set_index("event_id")
    fields = ["network", "station", "phase", "time"]
    dataframe = dataframe.loc[event_id, fields]

    _arrival_dict = {
        (f"{network}.{station}", phase): timestamp
        for network, station, phase, timestamp in dataframe.values
    }

    return (_arrival_dict)


def remove_outliers(dataframe, tukey_k, column):
    """
    Return DataFrame with outliers removed using Tukey fences.
    """

    q1, q3 = dataframe[column].quantile(q=[0.25, 0.75])
    iqr = q3 - q1
    vmin = q1 - tukey_k * iqr
    vmax = q3 + tukey_k * iqr
    dataframe = dataframe[
         (dataframe[column] > vmin)
        &(dataframe[column] < vmax)
    ]

    return (dataframe)


@_utilities.log_errors(logger)
def station_dict(dataframe):
    """
    Return a dictionary with network geometry suitable for passing to
    the EQLocator constructor.

    Returned dictionary has "station_id" keys, where "station_id" =
    f"{network}.{station}", and values are spherical coordinates of
    station locations.
    """

    if np.any(dataframe[["network", "station"]].duplicated()):
        raise (IOError("Multiple coordinates supplied for single station(s)"))

    dataframe = dataframe.set_index(["network", "station"])

    _station_dict = {
        f"{network}.{station}": geo2sph(
            dataframe.loc[
                (network, station),
                ["latitude", "longitude", "depth"]
            ].values
        ) for network, station in dataframe.index
    }

    return (_station_dict)
