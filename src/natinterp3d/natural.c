/*******************************************************************************
*
*  natural.c - By Ross Hemsley Aug. 2009 - rh7223@bris.ac.uk.
*  Modifications by Istvan Sarandi, Dec. 2024
*
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "delaunay.h"
#include "utils.h"
#include "natural.h"

/******************************************************************************/
// Morton (Z-order) code helpers for spatial sorting of query points

static inline uint64_t expand_bits_21(uint64_t v) {
    v &= 0x1fffff;
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v << 8))  & 0x100f00f00f00f00fULL;
    v = (v | (v << 4))  & 0x10c30c30c30c30c3ULL;
    v = (v | (v << 2))  & 0x1249249249249249ULL;
    return v;
}

static inline uint64_t morton3d(double x, double y, double z,
                                double minx, double miny, double minz,
                                double inv_rx, double inv_ry, double inv_rz) {
    uint64_t ix = (uint64_t)((x - minx) * inv_rx * 2097151.0);
    uint64_t iy = (uint64_t)((y - miny) * inv_ry * 2097151.0);
    uint64_t iz = (uint64_t)((z - minz) * inv_rz * 2097151.0);
    if (ix > 2097151) ix = 2097151;
    if (iy > 2097151) iy = 2097151;
    if (iz > 2097151) iz = 2097151;
    return expand_bits_21(ix) | (expand_bits_21(iy) << 1) | (expand_bits_21(iz) << 2);
}

typedef struct {
    uint64_t code;
    int original_index;
} sort_entry;

/* 8-pass LSB radix sort on 64-bit Morton codes (O(n) vs O(n log n) for qsort). */
static void radix_sort_morton(sort_entry *entries, int n) {
    sort_entry *tmp = malloc(n * sizeof(sort_entry));
    for (int pass = 0; pass < 8; pass++) {
        int shift = pass * 8;
        int count[256];
        memset(count, 0, sizeof(count));
        for (int i = 0; i < n; i++)
            count[(entries[i].code >> shift) & 0xFF]++;
        int prefix[256];
        prefix[0] = 0;
        for (int i = 1; i < 256; i++)
            prefix[i] = prefix[i - 1] + count[i - 1];
        for (int i = 0; i < n; i++) {
            int bucket = (entries[i].code >> shift) & 0xFF;
            tmp[prefix[bucket]++] = entries[i];
        }
        sort_entry *swap = entries;
        entries = tmp;
        tmp = swap;
    }
    /* After 8 (even) passes, result is in original entries buffer */
    free(tmp);
}

static sort_entry* compute_morton_order(double *queryPoints, int numQueryPoints) {
    if (numQueryPoints == 0) {
        sort_entry *entries = malloc(0);
        return entries;
    }
    double minx, miny, minz, maxx, maxy, maxz;
    minx = maxx = queryPoints[0];
    miny = maxy = queryPoints[1];
    minz = maxz = queryPoints[2];
    for (int i = 1; i < numQueryPoints; i++) {
        double x = queryPoints[i*3], y = queryPoints[i*3+1], z = queryPoints[i*3+2];
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
        if (z < minz) minz = z; if (z > maxz) maxz = z;
    }
    double rx = maxx - minx, ry = maxy - miny, rz = maxz - minz;
    double inv_rx = rx > 0 ? 1.0 / rx : 1.0;
    double inv_ry = ry > 0 ? 1.0 / ry : 1.0;
    double inv_rz = rz > 0 ? 1.0 / rz : 1.0;
    sort_entry *entries = malloc(numQueryPoints * sizeof(sort_entry));
    for (int i = 0; i < numQueryPoints; i++) {
        entries[i].code = morton3d(queryPoints[i*3], queryPoints[i*3+1], queryPoints[i*3+2], minx, miny, minz, inv_rx, inv_ry, inv_rz);
        entries[i].original_index = i;
    }
    radix_sort_morton(entries, numQueryPoints);
    return entries;
}

/******************************************************************************/

vertex *initPoints(double *xyz, int n) {
    vertex *ps = malloc(sizeof(vertex) * n);
    int i;
    for (i = 0; i < n; i++) {
        ps[i].X = xyz[i * 3];
        ps[i].Y = xyz[i * 3 + 1];
        ps[i].Z = xyz[i * 3 + 2];
        ps[i].index = i;
    }
    return ps;
}

/******************************************************************************/

void buildNewMeshAndVertices(double *dataPoints, int numDataPoints, mesh **m, vertex **ps) {
    *ps = initPoints(dataPoints, numDataPoints);
    *m = newMesh();
    buildMesh(*ps, numDataPoints, *m);
}

void freeMeshAndVertices(mesh *m, vertex *ps) {
    freeMesh(m);
    free(ps);
}

