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
#ifdef _OPENMP
#include <omp.h>
#endif

/* High bit of simplex mark field stores precomputed orient3d sign */
#define ORIENT_POS_BIT ((uint64_t)1 << 63)

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
/* Volume of the Voronoi subcell of vertex pk within tet (pk, pa, pb, pc).   */
/*                                                                            */
/* Uses algebraic simplification: the 6 signed sub-tetrahedra reduce to      */
/* 3 cross products and 3 dot products (instead of 6 of each).               */
/******************************************************************************/
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

    double *a = pa, *b = pb;
    double *fab = c_fab, *fac = c_fac, *fbc = c_fbc;
    if (sv < 0) {
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

/******************************************************************************/
/* Hash set operations for simplex pointer tracking (with generation counter) */
/******************************************************************************/

static inline unsigned int hashPtr(simplex *p, int size) {
    uintptr_t v = (uintptr_t)p;
    v = ((v >> 4) ^ (v >> 16)) * 0x45d9f3b;
    return (unsigned int)(v & (size - 1));
}

/* Low-level insert into interleaved table (no resize check). */
static inline int hashSetInsertRaw(visited_entry *table, int size,
                                    uint32_t gen, simplex *p) {
    int mask = size - 1;
    unsigned int h = hashPtr(p, size);
    while (1) {
        if (table[h].gen != gen) {
            table[h].ptr = p;
            table[h].gen = gen;
            return 0; /* newly inserted */
        }
        if (table[h].ptr == p) {
            return 1; /* already present */
        }
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
        if (old_table[i].gen == gen)
            hashSetInsertRaw(new_table, new_size, gen, old_table[i].ptr);
    }
    free(old_table);
    s->visited = new_table;
    s->visited_size = new_size;
}

/* Returns 1 if already present, 0 if newly inserted. Grows table if needed. */
static inline int hashSetInsert(if_scratch *s, simplex *p) {
    if (s->visited_count * 4 >= s->visited_size * 3) /* >75% load */
        visitedGrow(s);
    int ret = hashSetInsertRaw(s->visited, s->visited_size,
                                s->visited_generation, p);
    if (ret == 0) s->visited_count++;
    return ret;
}

static inline int hashSetContains(visited_entry *table, int size,
                                   uint32_t gen, simplex *p) {
    int mask = size - 1;
    unsigned int h = hashPtr(p, size);
    while (1) {
        if (table[h].gen != gen) return 0;
        if (table[h].ptr == p) return 1;
        h = (h + 1) & mask;
    }
}

/******************************************************************************/
/* Face circumcenter cache                                                    */
/******************************************************************************/

static inline void sortThreePointers(vertex *a, vertex *b, vertex *c,
                                      vertex **o0, vertex **o1, vertex **o2) {
    if (a > b) { vertex *t = a; a = b; b = t; }
    if (b > c) { vertex *t = b; b = c; c = t; }
    if (a > b) { vertex *t = a; a = b; b = t; }
    *o0 = a; *o1 = b; *o2 = c;
}

static inline unsigned int hashFaceKey(vertex *a, vertex *b, vertex *c, int size) {
    uintptr_t h = (uintptr_t)a * 0x9e3779b97f4a7c15ULL;
    h ^= (uintptr_t)b * 0x517cc1b727220a95ULL;
    h ^= (uintptr_t)c * 0x6c62272e07bb0142ULL;
    return (unsigned int)(h & (size - 1));
}

/* Look up or compute+insert a face circumcenter. Returns pointer to cached cc. */
static inline double *faceCCLookup(if_scratch *s, vertex *va, vertex *vb, vertex *vc) {
    vertex *k0, *k1, *k2;
    sortThreePointers(va, vb, vc, &k0, &k1, &k2);
    int mask = s->face_cc_size - 1;
    unsigned int h = hashFaceKey(k0, k1, k2, s->face_cc_size);
    uint32_t gen = s->face_cc_gen;
    while (1) {
        face_cc_entry *e = &s->face_cc[h];
        if (e->gen != gen) {
            /* Empty slot: compute and store */
            circumCenterTri3D(va->v, vb->v, vc->v, e->cc);
            e->k0 = k0; e->k1 = k1; e->k2 = k2;
            e->gen = gen;
            return e->cc;
        }
        if (e->k0 == k0 && e->k1 == k1 && e->k2 == k2) {
            return e->cc; /* cache hit */
        }
        h = (h + 1) & mask;
    }
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

    /* Face CC cache: sized for typical cavity (~50 faces) */
    s->face_cc_size = 256;
    s->face_cc = calloc(s->face_cc_size, sizeof(face_cc_entry));
    s->face_cc_gen = 1;

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
    free(s->face_cc);
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

    /* Reset face CC cache: O(1) generation increment */
    s->face_cc_gen++;
    if (s->face_cc_gen == 0) {
        memset(s->face_cc, 0, s->face_cc_size * sizeof(face_cc_entry));
        s->face_cc_gen = 1;
    }
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
}

/******************************************************************************/
/* Precompute orient3d signs and store in high bit of simplex mark field.     */
/* Called once before queries; the mark field is unused during read-only BFS. */
/* Uses packed array if available for better cache behavior.                  */
/******************************************************************************/
static void precomputeOrientSigns(mesh *m) {
    if (m->packed_simplices) {
        for (int i = 0; i < m->num_packed; i++) {
            simplex *s = &m->packed_simplices[i];
            if (orient3dfast(s->p[0]->v, s->p[1]->v, s->p[2]->v, s->p[3]->v) > 0)
                s->mark = ORIENT_POS_BIT;
            else
                s->mark = 0;
        }
    } else {
        listNode *iter = topOfLinkedList(m->tets);
        simplex *s;
        while ((s = nextElement(m->tets, &iter))) {
            if (orient3dfast(s->p[0]->v, s->p[1]->v, s->p[2]->v, s->p[3]->v) > 0)
                s->mark = ORIENT_POS_BIT;
            else
                s->mark = 0;
        }
    }
}

/******************************************************************************/
/* Find Bowyer-Watson cavity: BFS from containing simplex.                    */
/* Cavity = all tets whose circumsphere contains the query point.             */
/* Uses inspherefast for the in-circumsphere test.                            */
/*                                                                            */
/* After this function, scratch->visited contains ONLY cavity members,        */
/* suitable for O(1) cavity membership checks in boundary extraction.         */
/******************************************************************************/
static void findCavity(double *query, mesh *m, if_scratch *scratch) {
    vertex qv;
    qv.v[0] = query[0]; qv.v[1] = query[1]; qv.v[2] = query[2];
    qv.index = -1;

    simplex *start = findContainingSimplex(m, &qv);
    if (!start) return;

    /* We use a two-phase approach:
     * Phase 1: BFS using visited hash for dedup (visited = all checked tets)
     * Phase 2: Rebuild visited to contain only cavity members */

    scratch->bfs_top = 0;
    ensureBfsCap(scratch, 1);
    scratch->bfs_stack[scratch->bfs_top++] = start;

    while (scratch->bfs_top > 0) {
        simplex *cur = scratch->bfs_stack[--scratch->bfs_top];

        /* Skip if already checked */
        if (hashSetInsert(scratch, cur))
            continue;

        /* In-circumsphere test using precomputed orient sign (from mark field) */
        double in_sph;
        if (cur->mark & ORIENT_POS_BIT) {
            in_sph = inspherefast(cur->p[0]->v, cur->p[1]->v, cur->p[2]->v, cur->p[3]->v, query);
        } else {
            in_sph = inspherefast(cur->p[1]->v, cur->p[0]->v, cur->p[2]->v, cur->p[3]->v, query);
        }

        if (in_sph > 0) {
            /* Query inside circumsphere: add to cavity, push neighbors */
            ensureCavityCap(scratch, scratch->cavity_count + 1);
            scratch->cavity[scratch->cavity_count++] = cur;

            for (int i = 0; i < 4; i++) {
                if (cur->s[i] && !hashSetContains(scratch->visited,
                        scratch->visited_size, scratch->visited_generation, cur->s[i])) {
                    ensureBfsCap(scratch, scratch->bfs_top + 1);
                    scratch->bfs_stack[scratch->bfs_top++] = cur->s[i];
                }
            }
        }
    }

    /* Phase 2: Rebuild visited hash to contain ONLY cavity members.
     * Increment generation to logically clear the table. */
    scratch->visited_generation++;
    if (scratch->visited_generation == 0) {
        memset(scratch->visited, 0, scratch->visited_size * sizeof(visited_entry));
        scratch->visited_generation = 1;
    }

    /* Ensure table is large enough for cavity_count at <75% load */
    while (scratch->cavity_count * 4 >= scratch->visited_size * 3) {
        free(scratch->visited);
        scratch->visited_size *= 2;
        scratch->visited = calloc(scratch->visited_size, sizeof(visited_entry));
    }
    scratch->visited_count = scratch->cavity_count;
    for (int i = 0; i < scratch->cavity_count; i++) {
        hashSetInsertRaw(scratch->visited, scratch->visited_size,
                         scratch->visited_generation, scratch->cavity[i]);
    }
}

/******************************************************************************/
/* Extract boundary faces and collect natural neighbor vertex indices.         */
/* A boundary face is a face of a cavity tet whose opposite neighbor          */
/* is NULL or not in the cavity.                                              */
/* After findCavity, scratch->visited contains exactly the cavity members.    */
/******************************************************************************/
static void extractBoundaryAndNeighbors(if_scratch *scratch) {
    for (int ci = 0; ci < scratch->cavity_count; ci++) {
        simplex *s = scratch->cavity[ci];
        for (int fi = 0; fi < 4; fi++) {
            simplex *nbr = s->s[fi];
            if (nbr == NULL || !hashSetContains(scratch->visited,
                    scratch->visited_size, scratch->visited_generation, nbr)) {
                /* Boundary face: get the 3 vertices of face fi */
                vertex *v1, *v2, *v3;
                getFaceVerticies3(s, fi, &v1, &v2, &v3);

                ensureBoundaryCap(scratch, scratch->boundary_count + 1);
                int bi = scratch->boundary_count * 3;
                scratch->boundary_verts[bi + 0] = v1;
                scratch->boundary_verts[bi + 1] = v2;
                scratch->boundary_verts[bi + 2] = v3;
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

    /* Step 1: Find cavity (also sets up visited = cavity for boundary check) */
    findCavity(query, m, scratch);
    if (scratch->cavity_count == 0) {
        /* Empty cavity: the query may exactly coincide with a mesh vertex.
         * inspherefast returns 0 (on-sphere) for tets incident to that vertex,
         * so they're excluded from the cavity. Find and assign weight 1. */
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

    /* Step 2: Extract boundary faces and identify natural neighbors */
    extractBoundaryAndNeighbors(scratch);
    if (scratch->neighbor_count == 0) return;

    /* Ensure face CC cache is large enough (<75% load for 4*cavity_count faces) */
    {
        int needed = scratch->cavity_count * 4;
        while (needed * 4 >= scratch->face_cc_size * 3) {
            free(scratch->face_cc);
            scratch->face_cc_size *= 2;
            scratch->face_cc = calloc(scratch->face_cc_size, sizeof(face_cc_entry));
        }
    }

    /* Step 3: Old contributions -- cavity tets.
     * For each cavity tet, look up face circumcenters from cache (shared faces
     * between adjacent tets are computed once and reused).
     * fcc[i] = circumcenter of face opposite to p[i]. */
    for (int ci = 0; ci < scratch->cavity_count; ci++) {
        simplex *s = scratch->cavity[ci];
        double c_tet[3];
        circumCenterTet(s->p[0]->v, s->p[1]->v, s->p[2]->v, s->p[3]->v, c_tet);

        /* Look up 4 face CCs from cache: fcc[i] = CC of face opposite to p[i] */
        double *fcc[4];
        fcc[0] = faceCCLookup(scratch, s->p[1], s->p[2], s->p[3]);
        fcc[1] = faceCCLookup(scratch, s->p[0], s->p[2], s->p[3]);
        fcc[2] = faceCCLookup(scratch, s->p[0], s->p[1], s->p[3]);
        fcc[3] = faceCCLookup(scratch, s->p[0], s->p[1], s->p[2]);

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
             *   c_fab = CC(pk, p[o0], p[o1]) = face opposite p[o2] = fcc[o2]
             *   c_fac = CC(pk, p[o0], p[o2]) = face opposite p[o1] = fcc[o1]
             *   c_fbc = CC(pk, p[o1], p[o2]) = face opposite p[o0] = fcc[o0] */
            double vol = voronoiSubcellVolume(
                pk->v, s->p[o0]->v, s->p[o1]->v, s->p[o2]->v,
                c_tet, fcc[o2], fcc[o1], fcc[o0]);

            scratch->stolen_volumes[slot] += vol;
        }
    }

    /* Step 4: New contributions -- virtual tets (query + each boundary face).
     * For each boundary face, compute the boundary face CC once, then
     * for each data vertex, subtract the virtual tet subcell volume. */
    for (int bi = 0; bi < scratch->boundary_count; bi++) {
        vertex *fa = scratch->boundary_verts[bi * 3 + 0];
        vertex *fb = scratch->boundary_verts[bi * 3 + 1];
        vertex *fc = scratch->boundary_verts[bi * 3 + 2];

        /* Virtual tet circumcenter: (query, fa, fb, fc) */
        double c_vtet[3];
        circumCenterTet(query, fa->v, fb->v, fc->v, c_vtet);

        /* Boundary face CC: reuse from cache (already computed in Step 3) */
        double *c_face = faceCCLookup(scratch, fa, fb, fc);

        /* For each data vertex pk in this boundary face */
        vertex *face_verts[3] = {fa, fb, fc};
        for (int fvi = 0; fvi < 3; fvi++) {
            vertex *pk = face_verts[fvi];
            if (pk->index < 0) continue;
            int slot = scratch->neighbor_map[pk->index];
            if (slot < 0) continue;

            /* Other two data vertices (not pk) */
            vertex *ppa = face_verts[(fvi + 1) % 3];
            vertex *ppb = face_verts[(fvi + 2) % 3];

            /* Virtual tet is (pk, query, pa, pb).
             * Face circumcenters for faces containing pk:
             *   face(pk, query, pa), face(pk, query, pb): computed fresh
             *   face(pk, pa, pb) = boundary face CC */
            double c_f_q_a[3], c_f_q_b[3];
            circumCenterTri3D(pk->v, query, ppa->v, c_f_q_a);
            circumCenterTri3D(pk->v, query, ppb->v, c_f_q_b);

            double vol = voronoiSubcellVolume(
                pk->v, query, ppa->v, ppb->v,
                c_vtet, c_f_q_a, c_f_q_b, c_face);

            scratch->stolen_volumes[slot] -= vol;
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

    /* Pack simplices for cache-friendly BFS + precompute orient signs */
    packMeshSimplices(m);
    precomputeOrientSigns(m);

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

    /* Pack simplices for cache-friendly BFS + precompute orient signs */
    packMeshSimplices(m);
    precomputeOrientSigns(m);

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
