# cython: language_level=3
# distutils: language = c

import numpy as np
cimport numpy as np
import scipy.sparse

from libc.stdlib cimport malloc, free

# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memory management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

np.import_array()

cdef extern from "delaunay.h":
    ctypedef struct mesh:
        pass
    ctypedef struct vertex:
        pass

    mesh* copyMesh(mesh *m);
    void freeMesh(mesh *m);

cdef extern from "natural.h":
    void getNaturalInterpolationWeights(
        double* queryPoints, int numQueryPoints,
        mesh* mesh, int numDataPoints,
        double** weightValues, int** weightColInds, int* weightRowPtrs)
    void getNaturalInterpolationWeightsParallel(
        double *queryPoints, int numQueryPoints,
        mesh **threadMeshes, int numThreads, int numDataPoints,
        double** weightValues, int** weightColInds, int * weightRowPtrs)
    void buildNewMeshAndVertices(double *dataPoints, int numDataPoints, mesh **m, vertex **ps)
    void freeMeshAndVertices(mesh *m, vertex *ps)

cdef class MeshAndVertices:
    cdef mesh* m
    cdef vertex* ps
    cdef int numDataPoints

    def __cinit__(self, np.ndarray[np.double_t, ndim=2] dataPoints):
        self.numDataPoints = dataPoints.shape[0]
        if dataPoints.shape[1] != 3:
            raise ValueError("Data points must have shape (num_data_points, 3)")
        cdef double* dataPointsData = <double*>np.PyArray_DATA(dataPoints)
        buildNewMeshAndVertices(dataPointsData, self.numDataPoints, &self.m, &self.ps)

    def __dealloc__(self):
        freeMeshAndVertices(self.m, self.ps)

    def get_natural_interpolation_weights(self, np.ndarray[np.double_t, ndim=2] queryPoints):
        cdef int numQueryPoints = queryPoints.shape[0]
        if queryPoints.shape[1] != 3:
            raise ValueError("Query points must have shape (num_query_points, 3)")
        cdef double* queryPointsData = <double*>np.PyArray_DATA(queryPoints)

        # The output is in CSR sparse format
        # for the values, we pass a double** to the function, which will be filled in
        # with the pointer to the data, since we don't know the result length ahead of time
        cdef double* weightValues
        cdef int* weightColInds
        # For the row pointers, we know the length ahead of time, so we can allocate it
        cdef np.ndarray[np.int32_t, ndim=1] weightRowPtrs = np.empty(numQueryPoints + 1, dtype=np.int32)

        getNaturalInterpolationWeights(
            queryPointsData, numQueryPoints,
            self.m, self.numDataPoints,
            &weightValues, &weightColInds, <int*> weightRowPtrs.data)
        # Now wrap the output in a sparse matrix:
        cdef np.npy_intp shape[1]
        shape[0] = weightRowPtrs[numQueryPoints]

        weightValuesArray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, weightValues)
        weightColIndsArray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, weightColInds)

        PyArray_ENABLEFLAGS(weightValuesArray, np.NPY_OWNDATA)
        PyArray_ENABLEFLAGS(weightColIndsArray, np.NPY_OWNDATA)

        return scipy.sparse.csr_matrix((weightValuesArray, weightColIndsArray, weightRowPtrs), shape=(numQueryPoints, self.numDataPoints))

cdef class MeshesAndVerticesParallel:
    cdef mesh** ms
    cdef int numThreads
    cdef vertex* ps
    cdef int numDataPoints

    def __cinit__(self, np.ndarray[np.double_t, ndim=2] dataPoints, int numThreads):
        self.numThreads = numThreads
        self.numDataPoints = dataPoints.shape[0]
        if dataPoints.shape[1] != 3:
            raise ValueError("Data points must have shape (num_data_points, 3)")

        cdef double * dataPointsData = <double *> np.PyArray_DATA(dataPoints)

        # Allocate memory for thread-local mesh copies as a contiguous block
        self.ms = <mesh**>malloc(numThreads * sizeof(mesh*))
        if self.ms == NULL:
            raise MemoryError("Failed to allocate memory for thread-local meshes")

        buildNewMeshAndVertices(dataPointsData, self.numDataPoints, &self.ms[0], &self.ps)
        for i in range(1, numThreads):
            self.ms[i] = copyMesh(self.ms[0])

    def __dealloc__(self):
        freeMeshAndVertices(self.ms[0], self.ps)
        for i in range(1, self.numThreads):
           freeMesh(self.ms[i])
        free(self.ms)

    def get_natural_interpolation_weights(self, np.ndarray[np.double_t, ndim=2] queryPoints):
        cdef int numQueryPoints = queryPoints.shape[0]
        if queryPoints.shape[1] != 3:
            raise ValueError("Query points must have shape (num_query_points, 3)")
        cdef double* queryPointsData = <double*>np.PyArray_DATA(queryPoints)

        # The output is in CSR sparse format
        # for the values, we pass a double** to the function, which will be filled in
        # with the pointer to the data, since we don't know the result length ahead of time
        cdef double* weightValues
        cdef int* weightColInds
        # For the row pointers, we know the length ahead of time, so we can allocate it
        cdef np.ndarray[np.int32_t, ndim=1] weightRowPtrs = np.empty(numQueryPoints + 1, dtype=np.int32)

        getNaturalInterpolationWeightsParallel(
            queryPointsData, numQueryPoints,
            self.ms, self.numThreads, self.numDataPoints,
            &weightValues, &weightColInds, <int*> weightRowPtrs.data)
        # Now wrap the output in a sparse matrix:
        cdef np.npy_intp shape[1]
        shape[0] = weightRowPtrs[numQueryPoints]

        weightValuesArray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, weightValues)
        weightColIndsArray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, weightColInds)

        PyArray_ENABLEFLAGS(weightValuesArray, np.NPY_OWNDATA)
        PyArray_ENABLEFLAGS(weightColIndsArray, np.NPY_OWNDATA)

        return scipy.sparse.csr_matrix((weightValuesArray, weightColIndsArray, weightRowPtrs), shape=(numQueryPoints, self.numDataPoints))

