# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c

import numpy as np
cimport numpy as np
import scipy.sparse

# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memory management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

np.import_array()

cdef extern from "delaunay.h" nogil:
    ctypedef struct mesh:
        pass
    ctypedef struct vertex:
        pass

cdef extern from "natural.h" nogil:
    void buildNewMeshAndVertices(double *dataPoints, int numDataPoints, mesh ** m, vertex ** ps)
    void freeMeshAndVertices(mesh *m, vertex *ps)

cdef extern from "natural_insertionfree.h" nogil:
    int getInsertionFreeWeights(
            double *queryPoints, int numQueryPoints, mesh *m,
            int numDataPoints,
            double ** weightValues, int ** weightColInds, int *weightRowPtrs)
    int getInsertionFreeWeightsParallel(
            double *queryPoints, int numQueryPoints, mesh *m,
            int numThreads, int numDataPoints,
            double ** weightValues, int ** weightColInds, int *weightRowPtrs)

cdef _wrap_csr(int numQueryPoints, int numDataPoints,
               double *weightValues, int *weightColInds,
               np.ndarray[np.int32_t, ndim=1] weightRowPtrs):
    """Wrap C-allocated CSR arrays into a scipy sparse matrix."""
    cdef np.npy_intp shape[1]
    shape[0] = weightRowPtrs[numQueryPoints]

    weightValuesArray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, weightValues)
    weightColIndsArray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, weightColInds)

    PyArray_ENABLEFLAGS(weightValuesArray, np.NPY_ARRAY_OWNDATA)
    PyArray_ENABLEFLAGS(weightColIndsArray, np.NPY_ARRAY_OWNDATA)

    return scipy.sparse.csr_matrix(
        (weightValuesArray, weightColIndsArray, weightRowPtrs),
        shape=(numQueryPoints, numDataPoints))

cdef class MeshAndVertices:
    cdef mesh *m
    cdef vertex *ps
    cdef int numDataPoints

    def __cinit__(self, np.ndarray[np.double_t, ndim=2, mode='c'] dataPoints):
        self.numDataPoints = dataPoints.shape[0]
        if dataPoints.shape[1] != 3:
            raise ValueError("Data points must have shape (num_data_points, 3)")
        cdef double * dataPointsData = <double *> np.PyArray_DATA(dataPoints)
        with nogil:
            buildNewMeshAndVertices(dataPointsData, self.numDataPoints, &self.m, &self.ps)

    def __dealloc__(self):
        if self.m is not NULL:
            freeMeshAndVertices(self.m, self.ps)

    def get_weights(self, np.ndarray[np.double_t, ndim=2, mode='c'] queryPoints):
        cdef int numQueryPoints = queryPoints.shape[0]
        if queryPoints.shape[1] != 3:
            raise ValueError("Query points must have shape (num_query_points, 3)")
        cdef double *queryPointsData = <double *> np.PyArray_DATA(queryPoints)
        cdef double *weightValues
        cdef int *weightColInds
        cdef np.ndarray[np.int32_t, ndim=1] weightRowPtrs = np.empty(
            numQueryPoints + 1, dtype=np.int32)

        cdef int err
        with nogil:
            err = getInsertionFreeWeights(
                queryPointsData, numQueryPoints, self.m, self.numDataPoints,
                &weightValues, &weightColInds, <int *> weightRowPtrs.data)
        if err:
            raise OverflowError(
                "Total number of nonzero weights exceeds int32 limit")

        return _wrap_csr(numQueryPoints, self.numDataPoints,
                         weightValues, weightColInds, weightRowPtrs)

cdef class MeshAndVerticesParallel:
    cdef mesh *m
    cdef int numThreads
    cdef vertex *ps
    cdef int numDataPoints

    def __cinit__(self, np.ndarray[np.double_t, ndim=2, mode='c'] dataPoints, int numThreads):
        self.numThreads = numThreads
        self.numDataPoints = dataPoints.shape[0]
        if dataPoints.shape[1] != 3:
            raise ValueError("Data points must have shape (num_data_points, 3)")

        cdef double *dataPointsData = <double *> np.PyArray_DATA(dataPoints)
        with nogil:
            buildNewMeshAndVertices(dataPointsData, self.numDataPoints, &self.m, &self.ps)

    def __dealloc__(self):
        if self.m is not NULL:
            freeMeshAndVertices(self.m, self.ps)

    def get_weights(self, np.ndarray[np.double_t, ndim=2, mode='c'] queryPoints):
        cdef int numQueryPoints = queryPoints.shape[0]
        if queryPoints.shape[1] != 3:
            raise ValueError("Query points must have shape (num_query_points, 3)")
        cdef double *queryPointsData = <double *> np.PyArray_DATA(queryPoints)
        cdef double *weightValues
        cdef int *weightColInds
        cdef np.ndarray[np.int32_t, ndim=1] weightRowPtrs = np.empty(
            numQueryPoints + 1, dtype=np.int32)

        cdef int err
        with nogil:
            err = getInsertionFreeWeightsParallel(
                queryPointsData, numQueryPoints, self.m, self.numThreads,
                self.numDataPoints,
                &weightValues, &weightColInds, <int *> weightRowPtrs.data)
        if err:
            raise OverflowError(
                "Total number of nonzero weights exceeds int32 limit")

        return _wrap_csr(numQueryPoints, self.numDataPoints,
                         weightValues, weightColInds, weightRowPtrs)
