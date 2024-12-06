import os

import numpy as np

from natinterp3d.natinterp3d_cython import MeshesAndVerticesParallel, MeshAndVertices


class Interpolator:
    def __init__(self, data_points, parallel=True, num_threads=None):
        self.data_points = np.asarray(data_points, np.float64)
        if parallel:
            # we use os.sched_getaffinity(0) to determine the number of threads to use.
            # os.cpu_count() would be the obvious choice, but that one will give the total
            # count on the machine. But in a Slurm job, for example, we may only be able to use
            # a subset. os.sched_getaffinity(0) will give the number of threads we can use.
            num_threads = len(os.sched_getaffinity(0)) if num_threads is None else num_threads
            self.mesh_and_vertices = MeshesAndVerticesParallel(self.data_points, num_threads)
        else:
            self.mesh_and_vertices = MeshAndVertices(self.data_points)

    def get_weights(self, query_points):
        query_points = np.asarray(query_points, np.float64)
        return self.mesh_and_vertices.get_natural_interpolation_weights(
            query_points).astype(query_points.dtype)

    def interpolate(self, query_points, values):
        weights = self.get_weights(query_points)
        if len(values.shape) == 1:
            values = values[:, np.newaxis]
        return weights @ values


def interpolate(queries, keys, values, parallel=True, num_threads=None):
    ni3d = Interpolator(keys, parallel=parallel, num_threads=num_threads)
    return ni3d.interpolate(queries, values)

def get_weights(queries, keys, parallel=True, num_threads=None):
    ni3d = Interpolator(keys, parallel=parallel, num_threads=num_threads)
    return ni3d.get_weights(queries)