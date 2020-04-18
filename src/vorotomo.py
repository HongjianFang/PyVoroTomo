"""
An MPI script to invert traveltime data for velocity.

.. author:: Malcolm C. A. White
.. date:: 2020-04-17
"""

import signal

# Import local modules.
import _iterator
import _utilities


logger = _utilities.get_logger(__name__)


@_utilities.log_errors(logger)
def main(argc):
    """
    The main control loop.
    """

    logger.debug("Starting thread.")

    # Instantiate an InversionIterator object.
    inversion_iterator = _iterator.InversionIterator(argc)

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
    inversion_iterator.synchronize(attrs="all")

    # Compute traveltime-lookup tables.
    #inversion_iterator.compute_traveltime_lookup_tables()

    # Relocate all events.
    #inversion_iterator.relocate_events()

    # Update arrival residuals.
    #inversion_iterator.update_arrival_residuals()

    niter = inversion_iterator.cfg["algorithm"]["niter"]
    for iiter in range(niter):
        inversion_iterator.iterate()

    logger.debug("Thread completed without error.")
    return (True)


if __name__ == "__main__":
    # Add some signal handlers to abort all threads.
    signal.signal(signal.SIGINT,  _utilities.signal_handler)
    signal.signal(signal.SIGCONT, _utilities.signal_handler)
    signal.signal(signal.SIGTERM, _utilities.signal_handler)

    # Parse command line arguments.
    argc   = _utilities.parse_args()

    # Configure logging.
    _utilities.configure_logger(
        __name__,
        argc.log_file,
        verbose=argc.verbose
    )

    # Start the  main loop.
    main(argc)
