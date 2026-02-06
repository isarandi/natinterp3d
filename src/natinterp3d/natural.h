/*******************************************************************************
*
*  natural.h - By Ross Hemsley Aug. 2009 - rh7223@bris.ac.uk.
*  Modifications by Istvan Sarandi, Dec. 2024
*
*******************************************************************************/

#ifndef natural_h
#define natural_h

#include "delaunay.h"

/******************************************************************************/
vertex *initPoints(double *xyz, int n);

void buildNewMeshAndVertices(double *dataPoints, int numDataPoints, mesh **m, vertex **ps);

void freeMeshAndVertices(mesh *m, vertex *ps);

/******************************************************************************/
#endif

