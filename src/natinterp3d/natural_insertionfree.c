/*******************************************************************************
 * natural_insertionfree.c
 *
 * Insertion-free Sibson (natural neighbor) interpolation in 3D.
 * Computes Sibson weights by finding the Bowyer-Watson cavity (read-only BFS)
 * and computing stolen Voronoi volumes geometrically, without modifying the mesh.
 *
 * This file is included via unity.c and relies on functions from delaunay.c,
 * natural.c, and predicates.c being in the same translation unit.
 *
 * Istvan Sarandi, 2025
 ******************************************************************************/
#include "natural_insertionfree.h"
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/******************************************************************************/
/* Circumcenter of a triangle in 3D from raw coordinates.                     */
/******************************************************************************/
static void circumCenterTri3D(double *a, double *b, double *c, double *out) {
    double ab[3], ac[3], n[3];
    vertexSub(b, a, ab);
    vertexSub(c, a, ac);
    crossProduct(ab, ac, n);
    double n2 = scalarProduct(n, n);
    if (n2 < 1e-30) {
        /* Degenerate: return centroid as fallback */
        out[0] = (a[0] + b[0] + c[0]) / 3.0;
        out[1] = (a[1] + b[1] + c[1]) / 3.0;
        out[2] = (a[2] + b[2] + c[2]) / 3.0;
        return;
    }
    double ab2 = scalarProduct(ab, ab);
    double ac2 = scalarProduct(ac, ac);
    /* cc = a + cross(ab2*ac - ac2*ab, n) / (2*n2) */
    double tmp[3], cross_out[3];
    tmp[0] = ab2 * ac[0] - ac2 * ab[0];
    tmp[1] = ab2 * ac[1] - ac2 * ab[1];
    tmp[2] = ab2 * ac[2] - ac2 * ab[2];
    crossProduct(tmp, n, cross_out);
    double inv = 1.0 / (2.0 * n2);
    out[0] = a[0] + cross_out[0] * inv;
    out[1] = a[1] + cross_out[1] * inv;
    out[2] = a[2] + cross_out[2] * inv;
}

/******************************************************************************/
/* Circumcenter of a tetrahedron from raw double* coordinates.                */
/******************************************************************************/
static void circumCenterTet(double *a, double *b, double *c, double *d, double *out) {
    double b_a[3], c_a[3], d_a[3];
    double cross1[3], cross2[3], cross3[3];
    double mult1[3], mult2[3], mult3[3], sum[3];

    vertexSub(b, a, b_a);
    vertexSub(c, a, c_a);
    vertexSub(d, a, d_a);

    crossProduct(b_a, c_a, cross1);
    crossProduct(d_a, b_a, cross2);
    crossProduct(c_a, d_a, cross3);

    vertexByScalar(cross1, squaredDistance(d_a), mult1);
    vertexByScalar(cross2, squaredDistance(c_a), mult2);
    vertexByScalar(cross3, squaredDistance(b_a), mult3);

    vertexAdd(mult1, mult2, sum);
    vertexAdd(mult3, sum, sum);

    double denom = 2.0 * scalarProduct(b_a, cross3);
    if (fabs(denom) < 1e-30) {
        /* Degenerate (coplanar): return centroid as fallback */
        out[0] = (a[0] + b[0] + c[0] + d[0]) * 0.25;
        out[1] = (a[1] + b[1] + c[1] + d[1]) * 0.25;
        out[2] = (a[2] + b[2] + c[2] + d[2]) * 0.25;
        return;
    }
    vertexByScalar(sum, 1.0 / denom, out);
    vertexAdd(out, a, out);
}

/******************************************************************************/
/* Circumcenter of a tetrahedron plus a conservative bound on its rounding    */
/* error (distance between the computed and the true circumcenter). The       */
/* bound is a standard forward estimate: eps times the magnitude sum of the   */
/* numerator terms over the denominator. For sliver tets the denominator      */
/* (6 times the volume) is tiny and the bound becomes large, correctly        */
/* flagging the circumcenter as unreliable. Used at precompute time only.     */
/******************************************************************************/
static void circumCenterTetErr(double *a, double *b, double *c, double *d,
                               double *out, double *err) {
    double b_a[3], c_a[3], d_a[3];
    double cross1[3], cross2[3], cross3[3];
    double mult1[3], mult2[3], mult3[3], sum[3];

    vertexSub(b, a, b_a);
    vertexSub(c, a, c_a);
    vertexSub(d, a, d_a);

    crossProduct(b_a, c_a, cross1);
    crossProduct(d_a, b_a, cross2);
    crossProduct(c_a, d_a, cross3);

    vertexByScalar(cross1, squaredDistance(d_a), mult1);
    vertexByScalar(cross2, squaredDistance(c_a), mult2);
    vertexByScalar(cross3, squaredDistance(b_a), mult3);

    vertexAdd(mult1, mult2, sum);
    vertexAdd(mult3, sum, sum);

    double denom = 2.0 * scalarProduct(b_a, cross3);
    if (fabs(denom) < 1e-30) {
        /* Degenerate (coplanar): centroid fallback, error unbounded */
        out[0] = (a[0] + b[0] + c[0] + d[0]) * 0.25;
        out[1] = (a[1] + b[1] + c[1] + d[1]) * 0.25;
        out[2] = (a[2] + b[2] + c[2] + d[2]) * 0.25;
        *err = HUGE_VAL;
        return;
    }
    vertexByScalar(sum, 1.0 / denom, out);
    vertexAdd(out, a, out);

    double mag = 0.0;
    for (int i = 0; i < 3; i++) {
        mag += fabs(mult1[i]) + fabs(mult2[i]) + fabs(mult3[i]);
    }
    *err = 32.0 * DBL_EPSILON * mag / fabs(denom);
}

/******************************************************************************/
/* Volume of the Voronoi subcell of vertex pk within tet (pk, pa, pb, pc).   */
/*                                                                            */
/* Uses algebraic simplification: the 6 signed sub-tetrahedra reduce to      */
/* 3 cross products and 3 dot products (instead of 6 of each).               */
/******************************************************************************/
static inline double voronoiSubcellVolumeOriented(
    double *pk, double *pa, double *pb, double *pc,
    double *c_tet, double *c_fab, double *c_fac, double *c_fbc,
    int swap)
{
    double *a = pa, *b = pb;
    double *fab = c_fab, *fac = c_fac, *fbc = c_fbc;
    if (swap) {
        a = pb; b = pa;
        fac = c_fbc; fbc = c_fac;
    }

    /* Differences from pk for circumcenters */
    double df[3], dg[3], dh[3], dt[3];
    df[0] = fab[0]-pk[0]; df[1] = fab[1]-pk[1]; df[2] = fab[2]-pk[2];
    dg[0] = fac[0]-pk[0]; dg[1] = fac[1]-pk[1]; dg[2] = fac[2]-pk[2];
    dh[0] = fbc[0]-pk[0]; dh[1] = fbc[1]-pk[1]; dh[2] = fbc[2]-pk[2];
    dt[0] = c_tet[0]-pk[0]; dt[1] = c_tet[1]-pk[1]; dt[2] = c_tet[2]-pk[2];

    /* 3 cross products (halved from 6 by algebraic cancellation):
     *   C1 = cross(d_fab, d_ct)   used for bisector(pk,a) and bisector(pk,b)
     *   C2 = cross(d_ct, d_fac)   used for bisector(pk,a) and bisector(pk,c)
     *   C3 = cross(d_fbc, d_ct)   used for bisector(pk,b) and bisector(pk,c) */
    double C1[3], C2[3], C3[3];
    C1[0] = df[1]*dt[2] - df[2]*dt[1];
    C1[1] = df[2]*dt[0] - df[0]*dt[2];
    C1[2] = df[0]*dt[1] - df[1]*dt[0];
    C2[0] = dt[1]*dg[2] - dt[2]*dg[1];
    C2[1] = dt[2]*dg[0] - dt[0]*dg[2];
    C2[2] = dt[0]*dg[1] - dt[1]*dg[0];
    C3[0] = dh[1]*dt[2] - dh[2]*dt[1];
    C3[1] = dh[2]*dt[0] - dh[0]*dt[2];
    C3[2] = dh[0]*dt[1] - dh[1]*dt[0];

    /* 3 dot products with vertex differences (halved from 6):
     * vol = (1/12) * [dot(a-b, C1) + dot(a-pc, C2) + dot(b-pc, C3)] */
    double dab[3], dac[3], dbc[3];
    dab[0] = a[0]-b[0];  dab[1] = a[1]-b[1];  dab[2] = a[2]-b[2];
    dac[0] = a[0]-pc[0]; dac[1] = a[1]-pc[1]; dac[2] = a[2]-pc[2];
    dbc[0] = b[0]-pc[0]; dbc[1] = b[1]-pc[1]; dbc[2] = b[2]-pc[2];

    double vol = dab[0]*C1[0] + dab[1]*C1[1] + dab[2]*C1[2]
               + dac[0]*C2[0] + dac[1]*C2[1] + dac[2]*C2[2]
               + dbc[0]*C3[0] + dbc[1]*C3[1] + dbc[2]*C3[2];
    return vol / 12.0;
}

static double voronoiSubcellVolume(
    double *pk, double *pa, double *pb, double *pc,
    double *c_tet, double *c_fab, double *c_fac, double *c_fbc)
{
    /* Orientation check (inline to avoid redundant vertex subtractions) */
    double ta[3], tb[3], tc[3];
    ta[0] = pa[0]-pk[0]; ta[1] = pa[1]-pk[1]; ta[2] = pa[2]-pk[2];
    tb[0] = pb[0]-pk[0]; tb[1] = pb[1]-pk[1]; tb[2] = pb[2]-pk[2];
    tc[0] = pc[0]-pk[0]; tc[1] = pc[1]-pk[1]; tc[2] = pc[2]-pk[2];
    double ox = tb[1]*tc[2] - tb[2]*tc[1];
    double oy = tb[2]*tc[0] - tb[0]*tc[2];
    double oz = tb[0]*tc[1] - tb[1]*tc[0];
    double sv = ta[0]*ox + ta[1]*oy + ta[2]*oz;

    return voronoiSubcellVolumeOriented(pk, pa, pb, pc,
                                        c_tet, c_fab, c_fac, c_fbc, sv < 0);
}

/******************************************************************************/
/* Hash set operations for simplex pointer tracking (with generation counter) */
/******************************************************************************/

static inline unsigned int hashPtr(simplex *p, int size) {
    uintptr_t v = (uintptr_t)p;
    v = ((v >> 4) ^ (v >> 16)) * 0x45d9f3b;
    return (unsigned int)(v & (size - 1));
}

/* Probe for p. Returns the slot where p lives (found=1) or where it would
 * be inserted (found=0). */
static inline unsigned int hashProbe(visited_entry *table, int size,
                                     uint32_t gen, simplex *p, int *found) {
    int mask = size - 1;
    unsigned int h = hashPtr(p, size);
    while (1) {
        if (table[h].gen != gen) { *found = 0; return h; }
        if (table[h].ptr == p)   { *found = 1; return h; }
        h = (h + 1) & mask;
    }
}

/* Grow visited hash table to double its size and rehash all entries. */
static void visitedGrow(if_scratch *s) {
    int old_size = s->visited_size;
    visited_entry *old_table = s->visited;
    uint32_t gen = s->visited_generation;
    int new_size = old_size * 2;
    visited_entry *new_table = calloc(new_size, sizeof(visited_entry));
    for (int i = 0; i < old_size; i++) {
        if (old_table[i].gen == gen) {
            int found;
            unsigned int slot = hashProbe(new_table, new_size, gen,
                                          old_table[i].ptr, &found);
            new_table[slot] = old_table[i];
        }
    }
    free(old_table);
    s->visited = new_table;
    s->visited_size = new_size;
}

/******************************************************************************/
/* Scratch buffer management                                                  */
/******************************************************************************/

if_scratch *newIfScratch(int numDataPoints) {
    if_scratch *s = malloc(sizeof(if_scratch));

    s->cavity_cap = 256;
    s->cavity = malloc(s->cavity_cap * sizeof(simplex *));
    s->cavity_count = 0;

    s->bfs_cap = 256;
    s->bfs_stack = malloc(s->bfs_cap * sizeof(simplex *));
    s->bfs_top = 0;

    /* Visited hash table: power-of-2 size, generation-based reset */
    s->visited_size = 2048;
    s->visited = calloc(s->visited_size, sizeof(visited_entry));
    s->visited_generation = 1; /* start at 1 so calloc'd 0s don't match */
    s->visited_count = 0;

    s->boundary_cap = 256;
    s->boundary_verts = malloc(s->boundary_cap * 3 * sizeof(vertex *));
    s->boundary_count = 0;

    s->neighbor_cap = 256;
    s->neighbor_indices = malloc(s->neighbor_cap * sizeof(int));
    s->stolen_volumes = malloc(s->neighbor_cap * sizeof(double));
    s->neighbor_count = 0;

    s->neighbor_map_size = numDataPoints;
    s->neighbor_map = malloc(numDataPoints * sizeof(int));
    memset(s->neighbor_map, 0xff, numDataPoints * sizeof(int)); /* all -1 */

    s->boundary_fcc = malloc(s->boundary_cap * sizeof(double *));

    return s;
}

void freeIfScratch(if_scratch *s) {
    free(s->cavity);
    free(s->bfs_stack);
    free(s->visited);
    free(s->boundary_verts);
    free(s->neighbor_indices);
    free(s->stolen_volumes);
    free(s->neighbor_map);
    free(s->boundary_fcc);
    free(s);
}

void resetIfScratch(if_scratch *s) {
    s->cavity_count = 0;
    s->bfs_top = 0;
    s->boundary_count = 0;

    /* Reset visited hash: O(1) generation increment */
    s->visited_generation++;
    s->visited_count = 0;
    if (s->visited_generation == 0) {
        memset(s->visited, 0, s->visited_size * sizeof(visited_entry));
        s->visited_generation = 1;
    }

    /* Reset neighbor_map only for entries we used (O(neighbors) not O(numDataPoints)) */
    for (int i = 0; i < s->neighbor_count; i++) {
        s->neighbor_map[s->neighbor_indices[i]] = -1;
    }
    s->neighbor_count = 0;
}

/******************************************************************************/
/* Ensure dynamic arrays have enough capacity                                 */
/******************************************************************************/

static inline void ensureCavityCap(if_scratch *s, int needed) {
    if (needed > s->cavity_cap) {
        s->cavity_cap = needed * 2;
        s->cavity = realloc(s->cavity, s->cavity_cap * sizeof(simplex *));
    }
}

static inline void ensureBfsCap(if_scratch *s, int needed) {
    if (needed > s->bfs_cap) {
        s->bfs_cap = needed * 2;
        s->bfs_stack = realloc(s->bfs_stack, s->bfs_cap * sizeof(simplex *));
    }
}

static inline void ensureBoundaryCap(if_scratch *s, int needed) {
    if (needed > s->boundary_cap) {
        s->boundary_cap = needed * 2;
        s->boundary_verts = realloc(s->boundary_verts, s->boundary_cap * 3 * sizeof(vertex *));
        s->boundary_fcc = realloc(s->boundary_fcc, s->boundary_cap * sizeof(double *));
    }
}

static inline void ensureNeighborCap(if_scratch *s, int needed) {
    if (needed > s->neighbor_cap) {
        s->neighbor_cap = needed * 2;
        s->neighbor_indices = realloc(s->neighbor_indices, s->neighbor_cap * sizeof(int));
        s->stolen_volumes = realloc(s->stolen_volumes, s->neighbor_cap * sizeof(double));
    }
}

/******************************************************************************/
/* Add a natural neighbor (or look up existing slot).                          */
/* Returns the slot index in neighbor arrays.                                 */
/******************************************************************************/
static inline int addOrGetNeighbor(if_scratch *s, int vertexIndex) {
    int slot = s->neighbor_map[vertexIndex];
    if (slot >= 0) return slot;
    slot = s->neighbor_count++;
    ensureNeighborCap(s, s->neighbor_count);
    s->neighbor_indices[slot] = vertexIndex;
    s->stolen_volumes[slot] = 0.0;
    s->neighbor_map[vertexIndex] = slot;
    return slot;
}

/******************************************************************************/
/* Pack mesh simplices into a contiguous array for cache-friendly BFS.        */
/* Called once; remaps neighbor pointers and kd-tree data.                     */
/******************************************************************************/
static simplex *remapLookup(simplex *old, simplex **map_keys, int *map_vals,
                            simplex *packed, int map_mask) {
    uintptr_t h = ((uintptr_t)old >> 4) * 0x9e3779b97f4a7c15ULL;
    unsigned int slot = (unsigned int)(h & map_mask);
    while (map_keys[slot] != old)
        slot = (slot + 1) & map_mask;
    return &packed[map_vals[slot]];
}

static void packMeshSimplices(mesh *m) {
    if (m->packed_simplices) return; /* already packed */

    int n = getNumSimplicies(m);
    if (n == 0) return;

    simplex *packed = malloc(n * sizeof(simplex));

    /* Hash map for pointer remapping: old simplex ptr -> index in packed */
    int map_size = 1;
    while (map_size < n * 2) map_size *= 2; /* <50% load */
    int map_mask = map_size - 1;
    simplex **map_keys = calloc(map_size, sizeof(simplex *));
    int *map_vals = malloc(map_size * sizeof(int));

    /* Phase 1: Copy simplices and build remap table */
    listNode *iter = topOfLinkedList(m->tets);
    simplex *s;
    int idx = 0;
    while ((s = nextElement(m->tets, &iter))) {
        packed[idx] = *s; /* struct copy */
        /* Insert into hash: key=old ptr, value=index */
        uintptr_t h = ((uintptr_t)s >> 4) * 0x9e3779b97f4a7c15ULL;
        unsigned int slot = (unsigned int)(h & map_mask);
        while (map_keys[slot]) slot = (slot + 1) & map_mask;
        map_keys[slot] = s;
        map_vals[slot] = idx;
        idx++;
    }

    /* Phase 2: Remap neighbor pointers */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 4; j++) {
            if (packed[i].s[j]) {
                packed[i].s[j] = remapLookup(packed[i].s[j], map_keys, map_vals,
                                             packed, map_mask);
            }
        }
    }

    /* Phase 3: Remap kd-tree data */
    for (int i = 0; i < n; i++) {
        if (m->simplicies_kd[i]) {
            m->simplicies_kd[i] = remapLookup(m->simplicies_kd[i], map_keys, map_vals,
                                              packed, map_mask);
        }
    }

    free(map_keys);
    free(map_vals);

    m->packed_simplices = packed;
    m->num_packed = n;

    /* Precompute circumsphere data: the mesh is static during queries, so
     * every tet circumcenter and face circumcenter is computed exactly once
     * here instead of per query.
     *
     * The in-sphere BFS test compares dist2(q, cc) against two thresholds:
     * below t_in the query is certainly inside the true circumsphere, above
     * t_out certainly outside, accounting for the rounding error of the
     * computed circumcenter (which is unbounded for sliver tets). In the
     * ambiguous band the caller falls back to the determinant test
     * (inspherefast), whose error stays confined to a narrow band around
     * the true sphere. For well-shaped tets the band is a few ulps wide,
     * so the fallback is rare. */
    m->packed_ccr = malloc((size_t)n * 8 * sizeof(double));
    m->packed_fcc = malloc((size_t)n * 12 * sizeof(double));
    for (int i = 0; i < n; i++) {
        simplex *s2 = &packed[i];
        double *ccr = &m->packed_ccr[(size_t)i * 8];
        double err;
        circumCenterTetErr(s2->p[0]->v, s2->p[1]->v, s2->p[2]->v, s2->p[3]->v,
                           ccr, &err);
        double r2min = -1.0, r2max = -1.0;
        for (int j = 0; j < 4; j++) {
            double dx = s2->p[j]->v[0] - ccr[0];
            double dy = s2->p[j]->v[1] - ccr[1];
            double dz = s2->p[j]->v[2] - ccr[2];
            double d2 = dx * dx + dy * dy + dz * dz;
            if (r2min < 0 || d2 < r2min) r2min = d2;
            if (d2 > r2max) r2max = d2;
        }
        /* True center is within err of cc, so the true radius lies in
         * [sqrt(r2min) - err, sqrt(r2max) + err]; a query is certainly
         * inside if dist(q, cc) < sqrt(r2min) - 2*err and certainly
         * outside if dist(q, cc) > sqrt(r2max) + 2*err. */
        double rin = sqrt(r2min) - 2.0 * err;
        ccr[3] = rin > 0 ? rin * rin : -1.0;  /* t_in (< 0: never certain) */
        double rout = sqrt(r2max) + 2.0 * err;
        ccr[4] = rout * rout;                  /* t_out */

        /* Orientation sign of (p0, p1, p2, p3): the subcell formula needs
         * to know it for each (vertex, other three) permutation, where it
         * equals this sign flipped by the permutation's parity. */
        {
            double *q0 = s2->p[0]->v, *q1 = s2->p[1]->v;
            double *q2 = s2->p[2]->v, *q3 = s2->p[3]->v;
            double ta[3] = {q1[0]-q0[0], q1[1]-q0[1], q1[2]-q0[2]};
            double tb[3] = {q2[0]-q0[0], q2[1]-q0[1], q2[2]-q0[2]};
            double tc[3] = {q3[0]-q0[0], q3[1]-q0[1], q3[2]-q0[2]};
            double sv = ta[0]*(tb[1]*tc[2] - tb[2]*tc[1])
                      + ta[1]*(tb[2]*tc[0] - tb[0]*tc[2])
                      + ta[2]*(tb[0]*tc[1] - tb[1]*tc[0]);
            ccr[5] = sv < 0 ? -1.0 : 1.0;
        }
        ccr[6] = ccr[7] = 0.0;

        /* Face circumcenters: face opposite p[j] at offset 3*j */
        double *fcc = &m->packed_fcc[(size_t)i * 12];
        circumCenterTri3D(s2->p[1]->v, s2->p[2]->v, s2->p[3]->v, &fcc[0]);
        circumCenterTri3D(s2->p[0]->v, s2->p[2]->v, s2->p[3]->v, &fcc[3]);
        circumCenterTri3D(s2->p[0]->v, s2->p[1]->v, s2->p[3]->v, &fcc[6]);
        circumCenterTri3D(s2->p[0]->v, s2->p[1]->v, s2->p[2]->v, &fcc[9]);
    }
}

/******************************************************************************/
/* In-circumsphere test via precomputed circumcenter: certainly inside below  */
/* t_in, certainly outside above t_out; in the (typically few ulps wide)      */
/* ambiguous band, fall back to the determinant test.                         */
/******************************************************************************/
static inline int inCircumsphere(mesh *m, simplex *cur, double *query) {
    double *ccr = &m->packed_ccr[(cur - m->packed_simplices) * 8];
    double dx = query[0] - ccr[0];
    double dy = query[1] - ccr[1];
    double dz = query[2] - ccr[2];
    double d2 = dx * dx + dy * dy + dz * dz;

    if (d2 < ccr[3]) return 1;
    if (d2 > ccr[4]) return 0;
    double o = orient3dfast(cur->p[0]->v, cur->p[1]->v, cur->p[2]->v,
                            cur->p[3]->v);
    double is = inspherefast(cur->p[0]->v, cur->p[1]->v, cur->p[2]->v,
                             cur->p[3]->v, query);
    return o > 0 ? is > 0 : is < 0;
}

/******************************************************************************/
/* Find the Bowyer-Watson cavity (all tets whose circumsphere contains the    */
/* query point) by BFS from the containing simplex, extracting the boundary   */
/* faces and natural neighbors in the same pass: when a cavity tet looks at   */
/* a neighbor, the neighbor's verdict (from the visited hash, or tested on    */
/* first encounter) immediately decides whether the shared face is boundary.  */
/* Each tet is tested exactly once and each boundary face recorded exactly    */
/* once, from its unique cavity side.                                         */
/******************************************************************************/
static void findCavity(double *query, mesh *m, if_scratch *scratch) {
    vertex qv;
    qv.v[0] = query[0]; qv.v[1] = query[1]; qv.v[2] = query[2];
    qv.index = -1;

    simplex *start = findContainingSimplex(m, &qv);
    if (!start) return;

    /* Empty cavity if the query is not inside the start's circumsphere
     * (e.g. it exactly coincides with a mesh vertex). */
    if (!inCircumsphere(m, start, query)) return;

    {
        int found;
        unsigned int slot = hashProbe(scratch->visited, scratch->visited_size,
                                      scratch->visited_generation, start, &found);
        scratch->visited[slot].ptr = start;
        scratch->visited[slot].gen = scratch->visited_generation;
        scratch->visited[slot].in_cavity = 1;
        scratch->visited_count++;
    }
    ensureCavityCap(scratch, 1);
    scratch->cavity[scratch->cavity_count++] = start;
    ensureBfsCap(scratch, 1);
    scratch->bfs_stack[scratch->bfs_top++] = start;

    while (scratch->bfs_top > 0) {
        simplex *cur = scratch->bfs_stack[--scratch->bfs_top];
        double *fcc = &m->packed_fcc[(cur - m->packed_simplices) * 12];

        for (int fi = 0; fi < 4; fi++) {
            simplex *nbr = cur->s[fi];
            int boundary;
            if (nbr == NULL) {
                boundary = 1;
            } else {
                if ((scratch->visited_count + 1) * 4 >= scratch->visited_size * 3)
                    visitedGrow(scratch); /* keep load < 75% */
                int found;
                unsigned int slot = hashProbe(scratch->visited,
                                              scratch->visited_size,
                                              scratch->visited_generation,
                                              nbr, &found);
                if (found) {
                    boundary = !scratch->visited[slot].in_cavity;
                } else {
                    int in = inCircumsphere(m, nbr, query);
                    scratch->visited[slot].ptr = nbr;
                    scratch->visited[slot].gen = scratch->visited_generation;
                    scratch->visited[slot].in_cavity = (uint32_t)in;
                    scratch->visited_count++;
                    if (in) {
                        ensureCavityCap(scratch, scratch->cavity_count + 1);
                        scratch->cavity[scratch->cavity_count++] = nbr;
                        ensureBfsCap(scratch, scratch->bfs_top + 1);
                        scratch->bfs_stack[scratch->bfs_top++] = nbr;
                    }
                    boundary = !in;
                }
            }

            if (boundary) {
                /* Boundary face: get the 3 vertices of face fi */
                vertex *v1, *v2, *v3;
                getFaceVerticies3(cur, fi, &v1, &v2, &v3);

                ensureBoundaryCap(scratch, scratch->boundary_count + 1);
                int bi = scratch->boundary_count * 3;
                scratch->boundary_verts[bi + 0] = v1;
                scratch->boundary_verts[bi + 1] = v2;
                scratch->boundary_verts[bi + 2] = v3;
                /* Precomputed circumcenter of this face: getFaceVerticies3
                 * returns for face index fi the face opposite p[3-fi], and
                 * packed_fcc stores the face opposite p[j] at offset 3*j. */
                scratch->boundary_fcc[scratch->boundary_count] = &fcc[(3 - fi) * 3];
                scratch->boundary_count++;

                /* Register natural neighbors (skip super vertices with index < 0) */
                if (v1->index >= 0) addOrGetNeighbor(scratch, v1->index);
                if (v2->index >= 0) addOrGetNeighbor(scratch, v2->index);
                if (v3->index >= 0) addOrGetNeighbor(scratch, v3->index);
            }
        }
    }
}

/******************************************************************************/
/* Compute insertion-free Sibson weights for a single query point.            */
/* Results stored in scratch->neighbor_indices, scratch->stolen_volumes,      */
/* and scratch->neighbor_count.                                               */
/******************************************************************************/
void getWeightsSingleQueryIF(double *query, mesh *m, if_scratch *scratch) {
    resetIfScratch(scratch);

    /* Steps 1+2: Find cavity; boundary faces and natural neighbors are
     * extracted in the same pass. */
    findCavity(query, m, scratch);
    if (scratch->cavity_count == 0) {
        /* Empty cavity: the query may exactly coincide with a mesh vertex.
         * Such a query is never certainly-inside (its distance to the
         * circumcenter is at least the smallest vertex distance), and the
         * ambiguous-band determinant is exactly zero for tets incident to
         * that vertex, so they are excluded. Find and assign weight 1. */
        vertex qv;
        qv.v[0] = query[0]; qv.v[1] = query[1]; qv.v[2] = query[2];
        qv.index = -1;
        simplex *start = findContainingSimplex(m, &qv);
        if (!start) return;
        int n_coin = 0;
        int coin_idx[4];
        for (int i = 0; i < 4; i++) {
            vertex *p = start->p[i];
            if (p->index >= 0 &&
                p->v[0] == query[0] && p->v[1] == query[1] && p->v[2] == query[2]) {
                coin_idx[n_coin++] = p->index;
            }
        }
        if (n_coin > 0) {
            double w = 1.0 / n_coin;
            for (int i = 0; i < n_coin; i++) {
                int slot = addOrGetNeighbor(scratch, coin_idx[i]);
                scratch->stolen_volumes[slot] = w;
            }
        }
        return;
    }

    if (scratch->neighbor_count == 0) return;

    /* Step 3: Old contributions -- cavity tets.
     * Tet and face circumcenters are precomputed (see packMeshSimplices):
     * c_tet at packed_ccr[8*idx], fcc of face opposite p[i] at
     * packed_fcc[12*idx + 3*i]. */
    for (int ci = 0; ci < scratch->cavity_count; ci++) {
        simplex *s = scratch->cavity[ci];
        double *ccr = &m->packed_ccr[(s - m->packed_simplices) * 8];
        double *fcc = &m->packed_fcc[(s - m->packed_simplices) * 12];
        int tet_neg = ccr[5] < 0.0;

        for (int vi = 0; vi < 4; vi++) {
            vertex *pk = s->p[vi];
            if (pk->index < 0) continue; /* skip super vertices */
            int slot = scratch->neighbor_map[pk->index];
            if (slot < 0) continue; /* not a natural neighbor */

            /* Other three vertices of this tet */
            int o0 = (vi + 1) & 3, o1 = (vi + 2) & 3, o2 = (vi + 3) & 3;
            if (o2 < o0) { int tmp = o0; o0 = o2; o2 = tmp; }
            if (o1 > o2) { int tmp = o1; o1 = o2; o2 = tmp; }
            if (o0 > o1) { int tmp = o0; o0 = o1; o1 = tmp; }

            /* Face CCs for faces containing pk:
             *   c_fab = CC(pk, p[o0], p[o1]) = face opposite p[o2]
             *   c_fac = CC(pk, p[o0], p[o2]) = face opposite p[o1]
             *   c_fbc = CC(pk, p[o1], p[o2]) = face opposite p[o0]
             * The orientation of (pk, p[o0], p[o1], p[o2]) is the tet's
             * orientation flipped by the permutation parity, which for the
             * sorted (o0, o1, o2) is odd exactly for odd vi. */
            double vol = voronoiSubcellVolumeOriented(
                pk->v, s->p[o0]->v, s->p[o1]->v, s->p[o2]->v,
                ccr, &fcc[o2 * 3], &fcc[o1 * 3], &fcc[o0 * 3],
                tet_neg ^ (vi & 1));

            scratch->stolen_volumes[slot] += vol;
        }
    }

    /* Step 4: New contributions -- virtual tets (query + each boundary face).
     * For each boundary face, compute the virtual tet circumcenter and the
     * three virtual-face circumcenters (triangles through the query and one
     * face edge; each is shared by two of the face's vertices), then for
     * each data vertex, subtract the virtual tet subcell volume. */
    for (int bi = 0; bi < scratch->boundary_count; bi++) {
        vertex *fa = scratch->boundary_verts[bi * 3 + 0];
        vertex *fb = scratch->boundary_verts[bi * 3 + 1];
        vertex *fc = scratch->boundary_verts[bi * 3 + 2];

        int slot_a = fa->index >= 0 ? scratch->neighbor_map[fa->index] : -1;
        int slot_b = fb->index >= 0 ? scratch->neighbor_map[fb->index] : -1;
        int slot_c = fc->index >= 0 ? scratch->neighbor_map[fc->index] : -1;
        if (slot_a < 0 && slot_b < 0 && slot_c < 0) continue;

        /* Virtual tet circumcenter: (query, fa, fb, fc) */
        double c_vtet[3];
        circumCenterTet(query, fa->v, fb->v, fc->v, c_vtet);

        /* Boundary face CC: precomputed, recorded during boundary extraction */
        double *c_face = scratch->boundary_fcc[bi];

        /* Virtual-face circumcenters, one per face edge */
        double cc_ab[3], cc_ac[3], cc_bc[3];
        circumCenterTri3D(fa->v, query, fb->v, cc_ab);
        circumCenterTri3D(fa->v, query, fc->v, cc_ac);
        circumCenterTri3D(fb->v, query, fc->v, cc_bc);

        /* Virtual tet is (pk, query, pa, pb); its faces containing pk are
         * (pk, query, pa), (pk, query, pb) and (pk, pa, pb) = boundary face */
        if (slot_a >= 0) {
            scratch->stolen_volumes[slot_a] -= voronoiSubcellVolume(
                fa->v, query, fb->v, fc->v, c_vtet, cc_ab, cc_ac, c_face);
        }
        if (slot_b >= 0) {
            scratch->stolen_volumes[slot_b] -= voronoiSubcellVolume(
                fb->v, query, fc->v, fa->v, c_vtet, cc_bc, cc_ab, c_face);
        }
        if (slot_c >= 0) {
            scratch->stolen_volumes[slot_c] -= voronoiSubcellVolume(
                fc->v, query, fa->v, fb->v, c_vtet, cc_ac, cc_bc, c_face);
        }
    }

    /* Step 5: Normalize weights */
    double total = 0.0;
    for (int i = 0; i < scratch->neighbor_count; i++) {
        total += scratch->stolen_volumes[i];
    }
    if (total > 1e-30) {
        double inv_total = 1.0 / total;
        for (int i = 0; i < scratch->neighbor_count; i++) {
            scratch->stolen_volumes[i] *= inv_total;
        }
    } else {
        /* Degenerate case (near convex hull boundary): zero out weights */
        for (int i = 0; i < scratch->neighbor_count; i++) {
            scratch->stolen_volumes[i] = 0.0;
        }
        scratch->neighbor_count = 0;
    }
}

/******************************************************************************/
/* Per-thread result arena for accumulating query results without per-query   */
/* malloc. Each thread appends results to its own growing buffer.             */
/******************************************************************************/
typedef struct {
    double *values;
    int *indices;
    int used, cap;
} result_arena;

static inline void arenaEnsure(result_arena *a, int extra) {
    while (a->used + extra > a->cap) {
        a->cap *= 2;
        a->values = realloc(a->values, a->cap * sizeof(double));
        a->indices = realloc(a->indices, a->cap * sizeof(int));
    }
}

/******************************************************************************/
/* Single-threaded CSR output                                                 */
/******************************************************************************/
int getInsertionFreeWeights(
    double *queryPoints, int numQueryPoints, mesh *m,
    int numDataPoints,
    double **weightValues, int **weightColInds, int *weightRowPtrs)
{
    int *queryOffset = malloc(numQueryPoints * sizeof(int));
    int *queryCount = malloc(numQueryPoints * sizeof(int));

    result_arena arena;
    arena.cap = 4096;
    arena.values = malloc(arena.cap * sizeof(double));
    arena.indices = malloc(arena.cap * sizeof(int));
    arena.used = 0;

    if_scratch *scratch = newIfScratch(numDataPoints);

    /* Spatial sorting for cache locality */
    sort_entry *order = compute_morton_order(queryPoints, numQueryPoints);

    /* Pack simplices for cache-friendly BFS + precompute circumsphere data */
    packMeshSimplices(m);

    for (int si = 0; si < numQueryPoints; si++) {
        int i = order[si].original_index;
        getWeightsSingleQueryIF(&queryPoints[i * 3], m, scratch);

        int nc = scratch->neighbor_count;
        arenaEnsure(&arena, nc);
        queryOffset[i] = arena.used;
        queryCount[i] = nc;
        memcpy(&arena.values[arena.used], scratch->stolen_volumes, nc * sizeof(double));
        memcpy(&arena.indices[arena.used], scratch->neighbor_indices, nc * sizeof(int));
        arena.used += nc;
    }

    freeIfScratch(scratch);

    /* Build CSR matrix */
    int64_t totalNnz = 0;
    for (int i = 0; i < numQueryPoints; i++) {
        weightRowPtrs[i] = (int)totalNnz;
        totalNnz += queryCount[i];
    }
    if (totalNnz > INT32_MAX) {
        free(arena.values);
        free(arena.indices);
        free(queryOffset);
        free(queryCount);
        free(order);
        return -1;
    }
    weightRowPtrs[numQueryPoints] = (int)totalNnz;

    *weightValues = malloc(totalNnz * sizeof(double));
    *weightColInds = malloc(totalNnz * sizeof(int));

    for (int i = 0; i < numQueryPoints; i++) {
        memcpy(&(*weightValues)[weightRowPtrs[i]], &arena.values[queryOffset[i]],
               queryCount[i] * sizeof(double));
        memcpy(&(*weightColInds)[weightRowPtrs[i]], &arena.indices[queryOffset[i]],
               queryCount[i] * sizeof(int));
    }

    free(arena.values);
    free(arena.indices);
    free(queryOffset);
    free(queryCount);
    free(order);
    return 0;
}

/******************************************************************************/
/* Multi-threaded CSR output: one shared mesh, per-thread scratch + arena     */
/******************************************************************************/
int getInsertionFreeWeightsParallel(
    double *queryPoints, int numQueryPoints, mesh *m,
    int numThreads, int numDataPoints,
    double **weightValues, int **weightColInds, int *weightRowPtrs)
{
    int *queryCount = malloc(numQueryPoints * sizeof(int));
    int *queryTid = malloc(numQueryPoints * sizeof(int));
    int *queryOffset = malloc(numQueryPoints * sizeof(int));

    /* Per-thread scratch buffers and result arenas */
    if_scratch **scratches = malloc(numThreads * sizeof(if_scratch *));
    result_arena *arenas = malloc(numThreads * sizeof(result_arena));
    for (int t = 0; t < numThreads; t++) {
        scratches[t] = newIfScratch(numDataPoints);
        arenas[t].cap = 4096;
        arenas[t].values = malloc(4096 * sizeof(double));
        arenas[t].indices = malloc(4096 * sizeof(int));
        arenas[t].used = 0;
    }

    /* Spatial sorting for cache locality */
    sort_entry *order = compute_morton_order(queryPoints, numQueryPoints);

    /* Pack simplices for cache-friendly BFS + precompute circumsphere data */
    packMeshSimplices(m);

#ifdef _OPENMP
    omp_set_num_threads(numThreads);
#endif
    int si;
    #pragma omp parallel for schedule(dynamic, 64)
    for (si = 0; si < numQueryPoints; si++) {
        int i = order[si].original_index;
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        if_scratch *scratch = scratches[tid];
        result_arena *arena = &arenas[tid];

        getWeightsSingleQueryIF(&queryPoints[i * 3], m, scratch);

        int nc = scratch->neighbor_count;
        arenaEnsure(arena, nc);
        queryCount[i] = nc;
        queryTid[i] = tid;
        queryOffset[i] = arena->used;
        memcpy(&arena->values[arena->used], scratch->stolen_volumes, nc * sizeof(double));
        memcpy(&arena->indices[arena->used], scratch->neighbor_indices, nc * sizeof(int));
        arena->used += nc;
    }

    for (int t = 0; t < numThreads; t++) {
        freeIfScratch(scratches[t]);
    }
    free(scratches);

    /* Build CSR matrix */
    int64_t totalNnz = 0;
    for (int i = 0; i < numQueryPoints; i++) {
        weightRowPtrs[i] = (int)totalNnz;
        totalNnz += queryCount[i];
    }
    if (totalNnz > INT32_MAX) {
        for (int t = 0; t < numThreads; t++) {
            free(arenas[t].values);
            free(arenas[t].indices);
        }
        free(arenas);
        free(queryCount);
        free(queryTid);
        free(queryOffset);
        free(order);
        return -1;
    }
    weightRowPtrs[numQueryPoints] = (int)totalNnz;

    *weightValues = malloc(totalNnz * sizeof(double));
    *weightColInds = malloc(totalNnz * sizeof(int));

    double *wv = *weightValues;
    int *wci = *weightColInds;
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < numQueryPoints; i++) {
        int t = queryTid[i];
        int off = queryOffset[i];
        int nc = queryCount[i];
        memcpy(&wv[weightRowPtrs[i]], &arenas[t].values[off], nc * sizeof(double));
        memcpy(&wci[weightRowPtrs[i]], &arenas[t].indices[off], nc * sizeof(int));
    }

    for (int t = 0; t < numThreads; t++) {
        free(arenas[t].values);
        free(arenas[t].indices);
    }
    free(arenas);
    free(queryCount);
    free(queryTid);
    free(queryOffset);
    free(order);
    return 0;
}
