"""
A module for monkey-patching the pickle protocol for Cython classes.

.. author:: Malcolm C. A. White
.. date:: 2020-04-17
"""

import pykonal

class ScalarField3D(pykonal.fields.ScalarField3D):
    def __reduce__(self):
        """
        Magic method to support pickling.
        """

        state = (
            self.coord_sys,
            self.min_coords,
            self.node_intervals,
            self.npts,
            self.values
        )

        return (ScalarField3D._unpickle, state)

    def _unpickle(coord_sys, min_coords, node_intervals, npts, values):
        """
        A method to unpickle a pickled ScalarField3D object.

        Returns unpickled ScalarField3D object.
        """
        field = ScalarField3D(coord_sys=coord_sys)
        field.min_coords = min_coords
        field.node_intervals = node_intervals
        field.npts = npts
        field.values = values

        return (field)
