/*******************************************************************************
 * natural_insertionfree.h
 *
 * Insertion-free Sibson (natural neighbor) interpolation.
 * Computes the same weights as the insert-remove approach but without
 * modifying the Delaunay mesh, enabling a single shared mesh for all threads.
 ******************************************************************************/
#ifndef natural_insertionfree_h
#define natural_insertionfree_h

#include "delaunay.h"

/******************************************************************************/
/* Per-thread scratch space for insertion-free queries.                        */
/******************************************************************************/

/* Interleaved visited hash entry: pointer and generation co-located
 * in the same cache line for fewer cache misses during probing. */
typedef struct {
    simplex *ptr;
    uint32_t gen;
} visited_entry;

/* Face circumcenter cache entry: keyed by sorted triple of vertex pointers. */
typedef struct {
    vertex *k0, *k1, *k2;  /* sorted vertex pointers (key) */
    double cc[3];           /* cached circumcenter (value) */
    uint32_t gen;           /* generation for O(1) reset */
} face_cc_entry;

typedef struct {
    /* Cavity: list of simplex pointers */
    simplex **cavity;
    int cavity_count, cavity_cap;

    /* BFS stack */
    simplex **bfs_stack;
    int bfs_top, bfs_cap;

    /* Visited: open-addressed hash table with interleaved entries.
     * Uses generation counters for O(1) reset. */
    visited_entry *visited;
    uint32_t visited_generation;  /* current generation */
    int visited_size;  /* power of 2, e.g. 2048 */
    int visited_count; /* number of entries in current generation */

    /* Boundary faces: groups of 3 vertex pointers per face */
    vertex **boundary_verts; /* [3*i+0], [3*i+1], [3*i+2] per face */
    int boundary_count, boundary_cap;

    /* Natural neighbor tracking */
    int *neighbor_indices;
    double *stolen_volumes;
    int neighbor_count, neighbor_cap;

    /* Neighbor lookup: maps vertex index -> slot in neighbor arrays.
     * Size = numDataPoints, initialized to -1. Reset per query. */
    int *neighbor_map;
    int neighbor_map_size;

    /* Face circumcenter cache: avoids recomputing shared face CCs
     * between adjacent cavity tets. Generation-based O(1) reset. */
    face_cc_entry *face_cc;
    int face_cc_size;          /* power of 2 */
    uint32_t face_cc_gen;
} if_scratch;

/******************************************************************************/

if_scratch *newIfScratch(int numDataPoints);
void freeIfScratch(if_scratch *s);
void resetIfScratch(if_scratch *s);

void getWeightsSingleQueryIF(double *query, mesh *m, if_scratch *scratch);

int getInsertionFreeWeights(
    double *queryPoints, int numQueryPoints, mesh *m,
    int numDataPoints,
    double **weightValues, int **weightColInds, int *weightRowPtrs);

int getInsertionFreeWeightsParallel(
    double *queryPoints, int numQueryPoints, mesh *m,
    int numThreads, int numDataPoints,
    double **weightValues, int **weightColInds, int *weightRowPtrs);

/******************************************************************************/
#endif
