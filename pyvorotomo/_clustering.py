import numpy as np
import scipy.spatial
import pykonal

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
def _init_centroids(k, points):
    """
    Use the kmeans++ algorithm to initialize the cluster centroids.

    k - is the number of clusters.
    points - the points to cluster (in spherical coordinates).
    """

    xyz = pykonal.transformations.sph2xyz(points, (0, 0, 0))

    idxs = np.arange(len(xyz))
    idx = np.random.choice(idxs)
    centroids = [xyz[idx]]

    for ik in range(k-1):
        tree = scipy.spatial.cKDTree(centroids)
        dist, _ = tree.query(xyz)
        prob = dist / np.sum(dist)
        idx = np.random.choice(idxs, p=prob)
        centroids.append(xyz[idx])

    centroids = np.stack(centroids)

    return (centroids)


@_utilities.log_errors(logger)
def k_medians(k, points):
    """
    Return k-medians cluster medians for *points*.

    points - Data points to cluster (in spherical coordinates).
    """

    xyz = pykonal.transformations.sph2xyz(points, (0, 0, 0))

    medians = _init_centroids(k, points)

    last_indexes = None

    while True:

        _medians = []
        tree = scipy.spatial.cKDTree(medians)
        _, indexes = tree.query(xyz)

        if np.all(indexes == last_indexes):
            break

        last_indexes = indexes

        for index in range(len(medians)):

            _xyz = xyz[indexes == index]
            median = np.median(_xyz, axis=0)
            _medians.append(median)

        medians = np.stack(_medians)

    medians = pykonal.transformations.xyz2sph(medians, origin=(0, 0, 0))

    return (medians)
