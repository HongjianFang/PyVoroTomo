"""
An MPI script to invert traveltime data for velocity.

.. author:: Malcolm C. A. White
.. date:: 2020-04-17
"""

import mpi4py.MPI as MPI
import signal

# Import local modules.
import dataio
import utilities


PROCESSOR_NAME            = MPI.Get_processor_name()
COMM                      = MPI.COMM_WORLD
WORLD_SIZE                = COMM.Get_size()
RANK                      = COMM.Get_rank()
ROOT_RANK                 = 0
DISPATCH_REQUEST_TAG      = 100
DISPATCH_TRANSMISSION_TAG = 101


logger = utilities.get_logger(__name__)


class InversionIterator(object):
    """
    A class providing core functionality for iterating inversion
    procedure.
    """

    def __init__(self, argc):
        self._rank = RANK
        self._argc = argc
        self._cfg = None
        self._pwave_model = None
        self._swave_model = None
        self._events = None
        self._arrivals = None
        self._stations = None

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
        self._events = value

    @property
    def pwave_model(self):
        return (self._pwave_model)

    @pwave_model.setter
    def pwave_model(self, value):
        self._pwave_model = value

    @property
    def rank(self):
        return (self._rank)

    @property
    def stations(self):
        return (self._stations)

    @stations.setter
    def stations(self, value):
        self._stations = value

    @property
    def swave_model(self):
        return (self._swave_model)

    @swave_model.setter
    def swave_model(self, value):
        self._swave_model = value


    @utilities.log_errors(logger)
    def compute_traveltime_lookup_tables(self):
        """
        Compute traveltime-lookup tables.
        """
        if self.rank == ROOT_RANK:
            ids = zip(self.stations["network"], self.stations["station"])
            self.dispatch(ids)
            return (True)
        while True:
            # Request an event
            COMM.send(self.rank, dest=ROOT_RANK, tag=DISPATCH_REQUEST_TAG)
            item = COMM.recv(
                source=ROOT_RANK,
                tag=DISPATCH_TRANSMISSION_TAG
            )
            if item is None:
                logger.debug("Received sentinel.")
                return (True)
            network, station = item
            logger.debug(f"Received {item}")


    @utilities.log_errors(logger)
    def dispatch(self, ids, sentinel=None):
        """
        Dispatch ids to hungry workers, then dispatch sentinels.
        """

        for _id in ids:
            requesting_rank = COMM.recv(
                source=MPI.ANY_SOURCE,
                tag=DISPATCH_REQUEST_TAG
            )
            COMM.send(
                _id,
                dest=requesting_rank,
                tag=DISPATCH_TRANSMISSION_TAG
            )
        # Distribute sentinel.
        for irank in range(WORLD_SIZE - 1):
            requesting_rank = COMM.recv(
                source=MPI.ANY_SOURCE,
                tag=DISPATCH_REQUEST_TAG
            )
            COMM.send(
                sentinel,
                dest=requesting_rank,
                tag=DISPATCH_TRANSMISSION_TAG
            )

        return (True)


    @utilities.log_errors(logger)
    def load_cfg(self):
        """
        Parse and store configuration-file parameters.

        ROOT_RANK parses configuration file and broadcasts contents to all
        other processes.
        """

        if self.rank != ROOT_RANK:
            return (True)

        logger.debug("Loading configuration file.")

        # Parse configuration-file parameters.
        self.cfg = utilities.parse_cfg(self.argc.configuration_file)

        return (True)


    @utilities.log_errors(logger)
    def load_event_data(self):
        """
        Parse and return event data from file.

        ROOT_RANK parses file and broadcasts contents to all other
        processes.
        """

        if self.rank != ROOT_RANK:
            return (True)

        logger.debug("Loading event data.")

        # Parse event data.
        data = dataio.parse_event_data(self.argc)
        self.events, self.arrivals = data

        return (True)


    @utilities.log_errors(logger)
    def load_network_geometry(self):
        """
        Parse and return network geometry from file.

        ROOT_RANK parses file and broadcasts contents to all other
        processes.
        """

        if self.rank != ROOT_RANK:
            return (True)

        logger.debug("Loading network geometry.")

        # Parse event data.
        stations = dataio.parse_network_geometry(self.argc)
        self.stations = stations

        return (True)


    @utilities.log_errors(logger)
    def load_velocity_models(self):
        """
        Parse and return velocity models from file.

        ROOT_RANK parses file and broadcasts contents to all other
        processes.
        """

        if self.rank != ROOT_RANK:
            return (True)

        logger.debug("Loading velocity models.")

        # Parse velocity model files.
        velocity_models = dataio.parse_velocity_models(self.cfg)
        self.pwave_model, self.swave_model = velocity_models

        return (True)


    @utilities.log_errors(logger)
    def sanitize_data(self):
        """
        Sanitize input data.
        """

        if self.rank != ROOT_RANK:
            return (True)

        logger.debug("Sanitizing data.")

        # Remove stations without arrivals.
        logger.debug("Dropping stations without arrivals.")
        arrivals = self.arrivals.set_index(["network", "station"])
        idx_keep = arrivals.index.unique()
        stations = self.stations.set_index(["network", "station"])
        stations = stations.loc[idx_keep]
        stations = stations.reset_index()
        self.stations = stations

        return (True)


    @utilities.log_errors(logger)
    def synchronize(self):
        """
        Synchronize input data across all processes.
        """

        logger.debug("Synchronizing data.")

        self.arrivals    = COMM.bcast(self.arrivals, root=ROOT_RANK)
        self.cfg         = COMM.bcast(self.cfg, root=ROOT_RANK)
        self.events      = COMM.bcast(self.events, root=ROOT_RANK)
        self.pwave_model = COMM.bcast(self.pwave_model, root=ROOT_RANK)
        self.swave_model = COMM.bcast(self.swave_model, root=ROOT_RANK)
        self.stations    = COMM.bcast(self.stations, root=ROOT_RANK)

        return (True)





@utilities.log_errors(logger)
def main(argc):
    """
    The main control loop.
    """

    logger.debug("Starting thread.")

    # Instantiate an InversionIterator object.
    inversion_iterator = InversionIterator(argc)

    # Load configuration-file parameters.
    inversion_iterator.load_cfg()

    # Load initial velocity models.
    inversion_iterator.load_velocity_models()

    # Load event data.
    inversion_iterator.load_event_data()

    # Load network geometry.
    inversion_iterator.load_network_geometry()

    # Sanitize event data and network metadata.
    inversion_iterator.sanitize_data()

    # Syncronize data across all processes.
    inversion_iterator.synchronize()

    # Compute traveltime-lookup tables.
    inversion_iterator.compute_traveltime_lookup_tables()


    # - Relocate all events.
    # for iiter in range(niter):
    #     for ireal in range(nreal):
    #         - Draw random sample of arrivals.
    #         - Generate Voronoi cells.
    #         - Trace rays and build sensitivity matrix and residuals vector.
    #         - Solve matrix equation for slowness update.
    #     - Stack realizations.
    #     - Relocate events.
    #     - Compute arrival residuals.

    logger.debug("Thread completed without error.")
    return (True)


if __name__ == "__main__":
    # Add some signal handlers to abort all threads.
    signal.signal(signal.SIGINT,  utilities.signal_handler)
    signal.signal(signal.SIGCONT, utilities.signal_handler)
    signal.signal(signal.SIGTERM, utilities.signal_handler)

    # Parse command line arguments.
    argc   = utilities.parse_args()

    # Configure logging.
    utilities.configure_logger(
        __name__,
        argc.log_file,
        PROCESSOR_NAME,
        RANK,
        verbose=argc.verbose
    )

    # Start the  main loop.
    main(argc)
