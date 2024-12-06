/*******************************************************************************
*
*  interpolate.h - By Ross Hemsley Aug. 2009 - rh7223@bris.ac.uk.
*  Modifications by Istvan Sarandi, Dec. 2024
*  
*  This unit will perform Natural-Neighbour interpolation. To do this, we first
*  need a Mesh of Delaunay Tetrahedrons for our input points. Each point to be
*  interpolated is then inserted into the mesh (remembering the steps that were
*  taken to insert it) and then the volume of the modified Voronoi cells 
*  (easily computed from the Delaunay Mesh) are used to weight the neighbouring
*  points. We can then revert the Delaunay mesh back to the original mesh by 
*  reversing the flips required to insert the point.
*
*******************************************************************************/

#ifndef natural_h
#define natural_h

#include "delaunay.h"

/******************************************************************************/
vertex *initPoints(double *xyz, int n);
//------------------------------------------------------------------------------
void    lastNaturalNeighbours(vertex *v, mesh *m, arrayList *neighbours,
                                                arrayList *neighbourSimplicies);
//------------------------------------------------------------------------------                                               

void buildNewMeshAndVertices(double *dataPoints, int numDataPoints, mesh **m, vertex **ps);

void freeMeshAndVertices(mesh *m, vertex *ps);

void getNaturalInterpolationWeights(
        double * queryPoints, int numQueryPoints, mesh *mesh,
        int numDataPoints, double** weightValues, int** weightColInds, int* weightRowPtrs);

void getNaturalInterpolationWeightsParallel(
        double *queryPoints, int numQueryPoints, mesh **threadMeshes, int numThreads, int numDataPoints,
        double** weightValues, int** weightColInds, int* weightRowPtrs);

/******************************************************************************/
#endif

