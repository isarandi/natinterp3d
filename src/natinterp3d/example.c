#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"
#include "delaunay.h"
#include "natural.h"


int main(int argc, char **argv)
{
  // the number of points in our point set.
  int n = 3;

  // the points.
  double x[] = {0,1,2};
  double y[] = {3,4,5};
  double z[] = {6,7,8};

  // initialise the points.
  vertex* ps = initPoints(x,y,z, n);
  
  // create and build the mesh.
  mesh *exampleMesh = newMesh(); 
  buildMesh(ps ,n, exampleMesh);
  
  // get interpolation weights.
  getInterpolationWeights(1,4,7, &weights_out, exampleMesh);

  // free the mesh.  
  freeMesh(exampleMesh);
  free(ps);

}
