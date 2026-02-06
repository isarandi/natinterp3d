import os
from typing import Optional

import numpy as np
import scipy.sparse
from natinterp3d.natinterp3d_cython import MeshAndVertices, MeshAndVerticesParallel


def interpolate(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    parallel: bool = True,
    num_threads: Optional[int] = None,
) -> np.ndarray:
    """
    Interpolates function values at query points using the natural neighbor interpolation method
    :footcite:`sibson1981brief,bobach2009natural,hemsley2009interpolation`,
    based on some known values of the function.

    Args:
        queries: The points at which to evaluate the function. Shape (M, 3).
        keys: The points at which the function is known. Shape (N, 3).
        values: The values of the function at the known points. Shape (N, D) or (N,).
        parallel: Whether to use multithreaded parallel processing.
        num_threads: The number of threads to use for parallel processing. If None, the number of
            threads will be determined by the number of cores available. Ignored if ``parallel``
            is False.

    Returns:
        The interpolated values of the function at the query points. Shape (M, D) or (M,),
        depending on the shape of the ``values`` array.
    """
    interpolator = Interpolator(keys, parallel=parallel, num_threads=num_threads)
    return interpolator.interpolate(queries, values)


def get_weights(
    queries: np.ndarray,
    keys: np.ndarray,
    parallel: bool = True,
    num_threads: Optional[int] = None,
) -> scipy.sparse.csr_matrix:
    """Returns the natural interpolation weights (Sibson coordinates) for the query points,
    given the known data points (keys, data sites).

    Args:
        queries: The points for which to compute the interpolation weights.
        keys: The points at which the function is known (sites)
        parallel: Whether to use multithreaded parallel processing.
        num_threads: The number of threads to use for parallel processing. If None, the number
            of threads will be determined by the number of cores available. Ignored if ``parallel``
            is False.

    Returns:
        The interpolation weights for the query points as a sparse matrix.
    """

    interpolator = Interpolator(keys, parallel=parallel, num_threads=num_threads)
    return interpolator.get_weights(queries)


class Interpolator:
    """
    Natural neighbor interpolator for 3D data points.

    If the same data points are used for multiple interpolations, it is more efficient to
    create an Interpolator object and use it for multiple interpolations, rather than
    calling the :func:`interpolate` function multiple times. This is because computing the
    Delaunay tetrahedralization of the data points would be wastefully done multiple times.

    Args:
        data_points: The data points at which the function is known.
        parallel: Whether to use multithreaded parallel processing.
        num_threads: The number of threads to use for parallel processing. If None, the number
            of threads will be determined by the number of cores available. Ignored if ``parallel``
            is False.

    """

    def __init__(
        self,
        data_points: np.ndarray,
        parallel: bool = True,
        num_threads: Optional[int] = None,
    ):

        data_points = np.ascontiguousarray(data_points, np.float64)
        self._centroid = data_points.mean(axis=0)
        centered = data_points - self._centroid
        if parallel:
            # we use os.sched_getaffinity(0) to determine the number of threads to use.
            # os.cpu_count() would be the obvious choice, but that one will give the total
            # count on the machine. But in a Slurm job, for example, we may only be able to use
            # a subset. os.sched_getaffinity(0) will give the number of threads we can use.
            if num_threads is None:
                try:
                    num_threads = len(os.sched_getaffinity(0))
                except AttributeError:
                    num_threads = os.cpu_count() or 1
            self.mesh_and_vertices = MeshAndVerticesParallel(centered, num_threads)
        else:
            self.mesh_and_vertices = MeshAndVertices(centered)

    def get_weights(self, query_points: np.ndarray) -> scipy.sparse.csr_matrix:
        """Compute the natural interpolation weights (Sibson coordinates) for the query points.

        Args:
            query_points: The points for which to compute the interpolation weights.

        Returns:
            The interpolation weights for the query points as a sparse matrix.
        """
        out_dtype = query_points.dtype
        query_points = np.ascontiguousarray(query_points, np.float64) - self._centroid
        result = self.mesh_and_vertices.get_weights(query_points)
        return result.astype(out_dtype, copy=False)

    def interpolate(self, query_points: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Interpolate function values at query points using the natural interpolation method.

        Args:
            query_points: The points at which to evaluate the function. Shape (N, 3).
            values: The values of the function at the known points, which were originally passed
                in the constructor. Shape (M, D) or (M,).

        Returns:
            The interpolated values of the function at the query points. Shape (N, D) or (N,),
            depending on the shape of the ``values`` array.
        """
        weights = self.get_weights(query_points)
        if len(values.shape) == 1:
            values = values[:, np.newaxis]
            interpolated = weights @ values
            return np.squeeze(interpolated, axis=1)
        else:
            return weights @ values
