import numpy as np
import scipy.spatial

from . import _utilities

# Get logger handle.
logger = _utilities.get_logger(f"__main__.{__name__}")


@_utilities.log_errors(logger)
def fibonacci(n):
    """
    Return the n-th number in the Fibonacci sequence.
    """
    if n == 0:
        return (1)
    elif n == 1:
        return (1)
    else:
        return (fibonacci(n - 2)  +  fibonacci(n - 1))


@_utilities.log_errors(logger)
def k_medians(medians, points):
    """
    Return k-medians cluster medians for *points* given initial
    *medians*.
    """

    last_indexes = None

    while True:

        _medians = []
        tree = scipy.spatial.cKDTree(medians)
        _, indexes = tree.query(points)

        if np.all(indexes == last_indexes):

            return (medians)

        last_indexes = indexes

        for index in range(len(medians)):

            _points = points[indexes == index]
            median = np.median(_points, axis=0)
            _medians.append(median)

        medians = np.stack(_medians)

    return (medians)
