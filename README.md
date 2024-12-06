# Natural Neighbor Interpolation in 3D

This is a Python package for 3D [natural neighbor interpolation](https://en.wikipedia.org/wiki/Natural-neighbor_interpolation) (Sibson interpolation).

Natural neighbor interpolation is a form of scattered data interpolation,
where you have a set of sample *values* of a function at arbitrary locations in 3D space (let's call the locations *keys*),
and you want to interpolate the function value at other points (let's call them *queries*).

Specifically, in natural neighbor interpolation, the interpolated value is a weighted average of
the function values of the query point's "natural neighbors".
The natural neighbors of a query point are those vertices which, if we were to add the query point to the data, would
have Voronoi cells sharing a face with the query point's Voronoi cell.
The weights are proportional to the volume "stolen" from the neighbor's Voronoi cell upon the query point's insertion. 

This package uses Cython to wrap my modified version of Ross Hemsley's [interpolate3d](https://code.google.com/archive/p/interpolate3d/).
The modifications are:

* parallelization via OpenMP
* option to extract the natural neighbor weights (Sibson coordinates) directly (the original version only gives the final interpolated value, but not the weights)
* k-d tree for faster search for the containing simplex of an inserted point

## Installation

For now, install via:

```
pip install git+https://github.com/isarandi/natinterp3d.git
```

Publishing to PyPI is planned.

## Usage

Simplest is to call `natinterp3d.interpolate(queries, keys, values)` or `natinterp3d.get_weights(queries, keys)`:

```python
import natinterp3d
import numpy as np

# The positions of the data points where the function values are known
keys = np.array([[x1, y1, z1], [x2, y2, z2], ...])

# The values can also be a 2D array of shape (N, values_dim) with D dimensional vectors as values at each data point
values = np.array([v1, v2, v3, ...])  

# The positions where we want to interpolate the function values
queries = np.array([[qx1, qy1, qz1], [qx2, qy2, qz2], ...])

# Returns either [num_queries] or [num_queries, values_dim], the array of interpolated values
interpolated_values = natinterp3d.interpolate(queries, keys, values)

# or get the interpolation weights as a sparse matrix of size [num_queries, num_keys] (scipy.sparse.csr_matrix)
weights = natinterp3d.get_weights(queries, keys)
```

For more control, e.g., if you want to interpolate queries and/or values on the same key positions, you can use the `natinterp3d.Interpolator` class as:

```python
import natinterp3d

keys = np.array([[x1, y1, z1], [x2, y2, z2], ...])
interpolator = natinterp3d.Interpolator(keys)

values = np.array([v1, v2, v3, ...])  # or a 2D array of shape (N, values_dim)
queries = np.array([[qx1, qy1, qz1], [qx2, qy2, qz2], ...])
interpolated_values = interpolator.interpolate(queries, values)

# or:
weights = interpolator.get_weights(queries)
```

Multithreaded computation is automatically enabled. To customize, use the argument `parallel=True/False` and  `num_threads` in `interpolate` or `get_weights`. With `num_threads=None` (default), the number of threads is automatically determined based on the available CPU cores.

## License

GNU GPL v3
