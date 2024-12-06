/*******************************************************************************
*
*  interpolate.c - By Ross Hemsley Aug. 2009 - rh7223@bris.ac.uk.
*  Modifications by Istvan Sarandi, Dec. 2024
*
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include "delaunay.h"
#include "utils.h"
#include "natural.h"


/******************************************************************************/
/* Set this to run tests, and include debugging information.                  */
// #define DEBUG 0

/* set this to disable all asserts.                                           */
// #define NDEBUG 0 

//#define _TEST_
/******************************************************************************/

#define SQR(x)  (x)*(x)
#define ABS(x)  (x) >0 ? x : -(x)
#define MAX(x, y)  x<y ? y : x
#define MIN(x, y)  x>y ? y : x

/******************************************************************************/

vertex *initPoints(double *xyz, int n) {
    vertex *ps = malloc(sizeof(vertex) * n);
    int i;
    for (i = 0; i < n; i++) {
        ps[i].X = xyz[i * 3];
        ps[i].Y = xyz[i * 3 + 1];
        ps[i].Z = xyz[i * 3 + 2];
        ps[i].voronoiVolume = -1;
        ps[i].index = i;
    }
    return ps;
}

/******************************************************************************/

void lastNaturalNeighbours(vertex *v, mesh *m, arrayList *neighbours,
                           arrayList *neighbourSimplicies) {
    int i, j;
    for (i = 0; i < arrayListSize(m->updates); i++) {
        simplex *this = getFromArrayList(m->updates, i);
        for (j = 0; j < 4; j++) {
            if (this->p[j] != v && (!arrayListContains(neighbours, this->p[j]))) {
                if ((!pointOnSimplex(this->p[j], m->super))) {
                    addToArrayList(neighbours, this->p[j]);
                    addToArrayList(neighbourSimplicies, this);
                }
            }
        }
    }
}

/******************************************************************************/

void getUnnormalizedWeightsSingleQuery(
        double x, double y, double z, mesh *m, double *weights, int *neighbor_indices, int *num_neighbors) {
    int i;

    // Set up a temporary vertex to add to this mesh.
    vertex p;
    p.X = x;
    p.Y = y;
    p.Z = z;
    p.index = -1;
    p.voronoiVolume = -1;

    // The vertices which form the natural neighbours of this point.
    arrayList *neighbours;
    // Simplices attached to each neighbour. Speeds up nearest neighbour lookup.
    arrayList *neighbourSimplicies;

    // Volume of each natural neighbor's voronoi cell.
    double *neighbourVolumes;

    // Volume of the cell of the point to be interpolated.
    double pointVolume;

    // Add the point to the Delaunay Mesh - storing the original state.
    addPoint(&p, m);

    // Find the natural neighbours of the inserted point, and also keep
    // a list of an arbitrary neighbouring simplex, this will give us faster
    // neighbour lookup later.
    neighbours = newArrayList();
    neighbourSimplicies = newArrayList();
    lastNaturalNeighbours(&p, m, neighbours, neighbourSimplicies);
    *num_neighbors = arrayListSize(neighbours);

    // Calculate the volumes of the Voronoi Cells of the natural neighbours.
    neighbourVolumes = malloc(arrayListSize(neighbours) * sizeof(double));
    // Calculate the 'before' volumes of each voronoi cell (when the new point is included).
    // (Note: counterintuitively, we first calculate the volumes when the new point is included,
    // then we remove the point again and recalculate the volumes without the point. This is
    // because we don't know who the neighbors are before inserting our point.)

    for (i = 0; i < arrayListSize(neighbours); i++) {
        vertex *neighborVertex = getFromArrayList(neighbours, i);
        simplex *neighborSimplex = getFromArrayList(neighbourSimplicies, i);
        voronoiCell *vc = getVoronoiCell(neighborVertex, neighborSimplex, m);
        if (vc == NULL) {
            goto cleanup;
        }
        neighbourVolumes[i] = voronoiCellVolume(vc, neighborVertex);
        freeVoronoiCell(vc, m);
    }
    // Calculate the volume of the new point's Voronoi Cell.
    // We just need any neighbour simplex to use as an entry point into the
    // mesh.
    simplex *s = getFromArrayList(neighbourSimplicies, 0);
    voronoiCell *pointCell = getVoronoiCell(&p, s, m);
    if (pointCell == NULL) {
        goto cleanup;
    }
    pointVolume = voronoiCellVolume(pointCell, &p);
    freeVoronoiCell(pointCell, m);
    // Remove the last point.
    removePoint(m);
    // Calculate the 'stolen' volume of each neighbouring Voronoi Cell,
    // by calculating the original volumes (ie, now, after removing the new point), and subtracting the volumes
    // given when the point was added.
    for (i = 0; i < arrayListSize(neighbours); i++) {
        vertex *neighborVertex = getFromArrayList(neighbours, i);
        if (neighborVertex->voronoiVolume < 0) {
            simplex *s = findAnyNeighbour(neighborVertex, m->conflicts);
            voronoiCell *vc = getVoronoiCell(neighborVertex, s, m);
            if (vc == NULL) {
                goto cleanup;
            }
            double vol = voronoiCellVolume(vc, neighborVertex);
            double* voronoiVolumePtr = &neighborVertex->voronoiVolume;

            #pragma omp atomic write
            *voronoiVolumePtr = vol;

            freeVoronoiCell(vc, m);
        }
        neighbourVolumes[i] = neighborVertex->voronoiVolume - neighbourVolumes[i];
    }
    // Compute the weights.
    for (i = 0; i < arrayListSize(neighbours); i++) {
        vertex *neighborVertex = getFromArrayList(neighbours, i);
        assert(neighbourVolumes[i] >= -0.001);
        // Compute the weight of this vertex.
        double weight = neighbourVolumes[i] / pointVolume;
        // Fill the corresponding index in the output weights array.
        neighbor_indices[i] = neighborVertex->index;
        weights[i] = weight;
    }

    cleanup:
    // Put the dead simplicies in the memory pool.
    for (i = 0; i < arrayListSize(m->updates); i++) {
        push(m->deadSimplicies, getFromArrayList(m->updates, i));
    }
    // Free all the memory that we allocated whilst interpolating this point.
    emptyArrayList(m->conflicts);
    emptyArrayList(m->updates);

    // Free memory associated with adding this point.
    freeArrayList(neighbours, NULL);
    freeArrayList(neighbourSimplicies, NULL);
    free(neighbourVolumes);
}

void buildNewMeshAndVertices(double *dataPoints, int numDataPoints, mesh **m, vertex **ps) {
    *ps = initPoints(dataPoints, numDataPoints);
    *m = newMesh();
    buildMesh(*ps, numDataPoints, *m);
}

void freeMeshAndVertices(mesh *m, vertex *ps) {
    freeMesh(m);
    free(ps);
}

void getNaturalInterpolationWeights(
        double *queryPoints, int numQueryPoints, mesh *mesh, int numDataPoints,
        double** weightValues, int** weightColInds, int* weightRowPtrs) {
    double sum;

    // we will store the output as a sparse CSR matrix
    // for each row (query) we will have a double* for its weights. We first allocate an array to hold these pointers:
    double** weightValuesPerQuery = malloc(numQueryPoints * sizeof(double*));
    // we also need the column indices for each weight. We allocate an array to hold these:
    int** weightColIndsPerQuery = malloc(numQueryPoints * sizeof(int*));
    // and we need the number of neighbors for each query point. We allocate an array to hold these:
    int* numNeighborsPerQuery = malloc(numQueryPoints * sizeof(int));

    // we will also keep a double array of size numDataPoints to use as scratchpad for the per-query weight computation
    double* weightsTemp = malloc(numDataPoints * sizeof(double));
    // and similarly for the neighbor indices
    int* neighborIndicesTemp = malloc(numDataPoints * sizeof(int));

    for (int i = 0; i < numQueryPoints; ++i) {
        getUnnormalizedWeightsSingleQuery(
                queryPoints[i * 3],
                queryPoints[i * 3 + 1],
                queryPoints[i * 3 + 2],
                mesh,
                weightsTemp,
                neighborIndicesTemp,
                &numNeighborsPerQuery[i]);
        // the following code works, even if the number of found neighbors is 0. In that case, the weights will be 0.
        sum = 0;
        for (int j = 0; j < numNeighborsPerQuery[i]; ++j) {
            sum += weightsTemp[j];
        }
        for (int j = 0; j < numNeighborsPerQuery[i]; ++j) {
            weightsTemp[j] /= sum;
        }
        // now we copy the weights for this query into the array of pointers
        weightValuesPerQuery[i] = malloc(numNeighborsPerQuery[i] * sizeof(double));
        memcpy(weightValuesPerQuery[i], weightsTemp, numNeighborsPerQuery[i] * sizeof(double));
        // and we copy the neighbor indices for this query into the array of pointers
        weightColIndsPerQuery[i] = malloc(numNeighborsPerQuery[i] * sizeof(int));
        memcpy(weightColIndsPerQuery[i], neighborIndicesTemp, numNeighborsPerQuery[i] * sizeof(int));
    }
    // Now we build the CSR matrix
    int weightValuesSize = 0;
    for (int i = 0; i < numQueryPoints; ++i) {
        weightRowPtrs[i] = weightValuesSize;
        weightValuesSize += numNeighborsPerQuery[i];
    }
    weightRowPtrs[numQueryPoints] = weightValuesSize;
    *weightValues = malloc(weightValuesSize * sizeof(double));
    *weightColInds = malloc(weightValuesSize * sizeof(int));

    for (int i = 0; i < numQueryPoints; ++i) {
        memcpy(&(*weightValues)[weightRowPtrs[i]], weightValuesPerQuery[i], numNeighborsPerQuery[i] * sizeof(double));
        memcpy(&(*weightColInds)[weightRowPtrs[i]], weightColIndsPerQuery[i], numNeighborsPerQuery[i] * sizeof(int));
    }

    free(weightsTemp);
    free(neighborIndicesTemp);
    for (int i = 0; i < numQueryPoints; ++i) {
        free(weightValuesPerQuery[i]);
        free(weightColIndsPerQuery[i]);
    }
    free(weightValuesPerQuery);
    free(weightColIndsPerQuery);
    free(numNeighborsPerQuery);
}

void getNaturalInterpolationWeightsParallel(
        double *queryPoints, int numQueryPoints, mesh **threadMeshes, int numThreads, int numDataPoints,
        double** weightValues, int** weightColInds, int* weightRowPtrs) {
    // we will store the output as a sparse CSR matrix
    // for each row (query) we will have a double* for its weights. We first allocate an array to hold these pointers:
    double** weightValuesPerQuery = malloc(numQueryPoints * sizeof(double*));
    // we also need the column indices for each weight. We allocate an array to hold these:
    int** weightColIndsPerQuery = malloc(numQueryPoints * sizeof(int*));
    // and we need the number of neighbors for each query point. We allocate an array to hold these:
    int* numNeighborsPerQuery = malloc(numQueryPoints * sizeof(int));
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numQueryPoints; ++i) {
        double* localWeightsTemp = malloc(numDataPoints * sizeof(double));
        int* localNeighborIndicesTemp = malloc(numDataPoints * sizeof(int));
        getUnnormalizedWeightsSingleQuery(
                queryPoints[i * 3],
                queryPoints[i * 3 + 1],
                queryPoints[i * 3 + 2],
                threadMeshes[omp_get_thread_num()],
                localWeightsTemp,
                localNeighborIndicesTemp,
                &numNeighborsPerQuery[i]);

        // Normalize weights
        double sum = 0.0;
        for (int j = 0; j < numNeighborsPerQuery[i]; ++j) {
            sum += localWeightsTemp[j];
        }
        for (int j = 0; j < numNeighborsPerQuery[i]; ++j) {
            localWeightsTemp[j] /= sum;
        }

        // Allocate and copy weights and neighbor indices for this query
        weightValuesPerQuery[i] = malloc(numNeighborsPerQuery[i] * sizeof(double));
        memcpy(weightValuesPerQuery[i], localWeightsTemp, numNeighborsPerQuery[i] * sizeof(double));

        weightColIndsPerQuery[i] = malloc(numNeighborsPerQuery[i] * sizeof(int));
        memcpy(weightColIndsPerQuery[i], localNeighborIndicesTemp, numNeighborsPerQuery[i] * sizeof(int));

        free(localWeightsTemp);
        free(localNeighborIndicesTemp);
    }

    // Now we build the CSR matrix
    int weightValuesSize = 0;
    for (int i = 0; i < numQueryPoints; ++i) {
        weightRowPtrs[i] = weightValuesSize;
        weightValuesSize += numNeighborsPerQuery[i];
    }
    weightRowPtrs[numQueryPoints] = weightValuesSize;
    *weightValues = malloc(weightValuesSize * sizeof(double));
    *weightColInds = malloc(weightValuesSize * sizeof(int));

    for (int i = 0; i < numQueryPoints; ++i) {
        memcpy(&(*weightValues)[weightRowPtrs[i]], weightValuesPerQuery[i], numNeighborsPerQuery[i] * sizeof(double));
        memcpy(&(*weightColInds)[weightRowPtrs[i]], weightColIndsPerQuery[i], numNeighborsPerQuery[i] * sizeof(int));
    }

    for (int i = 0; i < numQueryPoints; ++i) {
        free(weightValuesPerQuery[i]);
        free(weightColIndsPerQuery[i]);
    }
    free(weightValuesPerQuery);
    free(weightColIndsPerQuery);
    free(numNeighborsPerQuery);
}


/* Unit testing. */
#ifdef _TEST_

#include <sys/time.h>

/* The number of points to create in our test data. */
#define NUM_TEST_POINTS 1e4

/* How detailed the interpolated output sohuld be: the cube of this value
   is the number of points we create.                                     */
#define INTERPOLATE_DETAIL 20

/* Do we print the output to file? */
#define OUTPUT_TO_FILE

/******************************************************************************/

double getTime()
{
struct timeval tv;
gettimeofday(&tv,NULL);
return tv.tv_sec + tv.tv_usec/1.0e6;
}

/******************************************************************************/
#endif

