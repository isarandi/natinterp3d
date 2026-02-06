/*******************************************************************************
*
*        delaunay.c - By Ross Hemsley Aug. 2009 - rh7223@bris.ac.uk.
* Modifications by Istvan Sarandi, Dec. 2024
*
* This file implements Delaunay meshing in 3D, using the edge flipping
* algorithm. To stop degenerecies arising from floating point errors, we use
* the geometical predicates provided in predicates.c - giving adaptive
* floating point arithmetic. We also remove degenerecies present in data
* caused by points which are coplanar, or cospherical. These points are removed
* by gradually adding random peterbations until the degenerecies are removed.
*
* This file has unit testing, which can be done by defining _TEST_ as shown
* seen below. The file can then be compiled by running:
*
*   >gcc -O3 delaunay.c utils.c
*
* The executible created can be run to create a set of random points, which are
* then meshed and checked for Delaunayness.
*
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "utils.h"
#include "delaunay.h"
#include "predicates.c"

// Forward declarations for static inline functions
static inline void getFaceVerticies(simplex *s, int i, vertex **p1, vertex **p2,
                                    vertex **p3, vertex **p4);
static inline void getFaceVerticies3(simplex *s, int i, vertex **p1, vertex **p2,
                                     vertex **p3);

/******************************************************************************/

/* Set this to be lower than the average distance between points. It is the
   amount that we will shift points by when we detect degenerecies. We
   gradually increase the value until the degenercy is removed                */
#define PERTURBATION_VALUE  0.0001

/* Do we show status of meshing? */
//  #define VERBOSE

/* This dictates the amount of debugging information to print. Starting
   at level 0. If not defined, no debugging information is printed.           */
//  #define DEBUG 0

/* This allows us to turn off all error checking. */
//#define NDEBUG

/* Turn on Unit testing. */
// #define _TEST_

/******************************************************************************/
/* - DO NOT EDIT - */
#ifdef _TEST_
#undef NDEBUG
#endif

#ifdef DEBUG
int SIMPLEX_MALLOC = 0;
int VERTEX_MALLOC  = 0;
#endif
/******************************************************************************/
#define MAX(x, y)  ((x) < (y) ? (y) : (x))
#define MIN(x, y)  ((x) > (y) ? (y) : (x))
#define SWAP(x, y)                                                              \
{                                                                              \
  double tmp;                                                                  \
  tmp = x;                                                                     \
  x   = y;                                                                     \
  y   = tmp;                                                                   \
}

/******************************************************************************/

simplex *newSimplex(mesh *m) {
    simplex *s = pop(m->deadSimplicies);

    // Obviously, we aren't going to re-use the super simplex..
    if (s == m->super) { s = 0; }

    if (!s) {
        s = malloc(sizeof(simplex));
#ifdef DEBUG
        SIMPLEX_MALLOC ++;
#endif
    }
    s->s[0] = 0;
    s->s[1] = 0;
    s->s[2] = 0;
    s->s[3] = 0;
    s->mark = 0;

    return s;
}

/******************************************************************************/
// This will take a list of points, and a mesh struct, and create a
// Delaunay Tetrahedralisation.

void buildMesh(vertex *ps, int n, mesh *m) {

    // We have no degenerecies to start with.
    m->coplanar_degenerecies = 0;
    m->cospherical_degenerecies = 0;

    m->kd = NULL;

    // This simplex will contain our entire point-set.
    initSuperSimplex(ps, n, m);
    addSimplexToMesh(m, m->super);
    int i, j;
    // Add each point to the mesh 1-by-1 using Bowyer-Watson cavity insertion.
    for (i = 0; i < n; i++) {
        addPoint(&ps[i], m);

        // Push conflicts to the memory pool.
        for (j = 0; j < arrayListSize(m->conflicts); j++) {
            push(m->deadSimplicies, getFromArrayList(m->conflicts, j));
        }

        // Reset the conflict and update lists.
        emptyArrayList(m->conflicts);
        emptyArrayList(m->updates);

        // Clear out the old neighobur update structs. (We don't use them here).
        resetNeighbourUpdates(m->neighbourUpdates);

#ifdef VERBOSE
        // Show status of meshing.
        printf("Meshing: %d%%.\n%c[1A", (int)((i+1)/(double)n *100),27);
#endif
    }

    m->kd = kd_create(3);
    m->owns_kd = true;
    listNode *iter = topOfLinkedList(m->tets);
    int numsimpl =  getNumSimplicies(m);
    kd_preallocate(m->kd, numsimpl);
    m->simplicies_kd = malloc(sizeof(simplex*) * numsimpl);

    simplex *s;
    i = 0;
    while ((s = nextElement(m->tets, &iter))) {
        kd_insert3(
            m->kd,
            (s->p[0]->v[0] + s->p[1]->v[0] + s->p[2]->v[0] + s->p[3]->v[0]) / 4,
            (s->p[0]->v[1] + s->p[1]->v[1] + s->p[2]->v[1] + s->p[3]->v[1]) / 4,
            (s->p[0]->v[2] + s->p[1]->v[2] + s->p[2]->v[2] + s->p[3]->v[2]) / 4,
            i);
        m->simplicies_kd[i++] = s;
    }

}

/******************************************************************************/
// This will allow us to remove all the simplicies which are connected
// to the super simplex.
void removeExternalSimplicies(mesh *m) {
    listNode *iter = topOfLinkedList(m->tets);
    simplex *s;

    // Remove all simplicies which connect to the super simplex
    while ((s = nextElement(m->tets, &iter))) {
        if (simplexContainsPoint(s, m->super->p[0]) ||
            simplexContainsPoint(s, m->super->p[1]) ||
            simplexContainsPoint(s, m->super->p[2]) ||
            simplexContainsPoint(s, m->super->p[3])) {
            swapSimplexNeighbour(s->s[0], s, NULL);
            swapSimplexNeighbour(s->s[1], s, NULL);
            swapSimplexNeighbour(s->s[2], s, NULL);
            swapSimplexNeighbour(s->s[3], s, NULL);

            removeSimplexFromMesh(m, s);
        }
    }
}

/******************************************************************************/
// return the value that we modified.

simplex **swapSimplexNeighbour(simplex *s, simplex *old, simplex *new) {
    // If this neighbour is on the exterior, we don't need to do anything.
    if (!s) { return NULL; }

    int i;

    // We are going to go through each of the elements children to see which one
    // points to the old simplex. When we find that value, we are going to swap
    // it for the new simplex value.
    for (i = 0; i < 4; i++) {
        if (s->s[i] == old) {
            s->s[i] = new;
            return &s->s[i];
        }
    }

    assert(0 && "swapSimplexNeighbour: old not found");
    return NULL;
}

/******************************************************************************/
// we are going to go through every face of every simplex to see if the
// orientation is consistent.

void orientationTest(linkedList *tets) {
#if DEBUG >= 1
    printf("Running orientation test: ---------------------------------------\n");
#endif

    int i;
    listNode *iter = topOfLinkedList(tets);
    simplex *s;

    while ((s = nextElement(tets, &iter))) {
        vertex *p1, *p2, *p3, *p4;

#if DEBUG >= 1
        printf("Checking orientation of %p\n", s);
#endif

        // Go through every face of this simplex
        for (i = 0; i < 4; i++) {
            getFaceVerticies(s, i, &p1, &p2, &p3, &p4);
            double o = orient3dfast(p1->v, p2->v, p3->v, p4->v);
            assert(o > 0);
        }
    }
}

/******************************************************************************/

int delaunayTest(mesh *m, vertex *ps, int n) {
    listNode *iter = topOfLinkedList(m->tets);
    simplex *s;

    int isDel = 0;
    int notDel = 0;

    while ((s = nextElement(m->tets, &iter))) {
#if DEBUG >= 2
        printf("Checking Delaunayness of %p.\n",s);
#endif

        // we want to see if this simplex is delaunay
        int i, succes = 1;
        for (i = 0; i < n; i++) {
            // if this point is not on the simplex, then it should not be within
            // the circumsphere of this given simplex.
            if (!pointOnSimplex(&ps[i], s)) {
#if DEBUG >= 0
                double orientation = orient3dfast(s->p[0]->v,
                                                  s->p[1]->v,
                                                  s->p[2]->v,
                                                  s->p[3]->v);

                assert(orientation != 0);
                assert(orientation > 0);
#endif

                double inSph = inspherefast(s->p[0]->v,
                                            s->p[1]->v,
                                            s->p[2]->v,
                                            s->p[3]->v,
                                            ps[i].v);
                if (inSph >= 0) {
                    notDel++;
                    succes = 0;
                    break;
                }
            }

        }
        if (succes) { isDel++; }
    }

#if DEBUG >= 2
    printf("Non-Delaunay Simplicies: %d.\n", notDel);
    printf("There are %f%% non-Delaunay Simplicies.\n",
                                           (notDel/(double)(isDel + notDel))*100);
#endif
    return notDel == 0;
}

/******************************************************************************/
// This function is purely to test whether the set of neighbours of each
// simplex is correct - If it is not reliable, then the program behaviour will
// be undeterministic: potentially giving a very difficult bug.
// We only need to run this test when the code is modified.

void faceTest(mesh *m) {
    int j;

#if DEBUG >= 1
    printf("Running face test: ----------------------------------------------\n");
#endif

    // Set our iterator to point to the top of the tet list.
    listNode *iter = topOfLinkedList(m->tets);
    // The pointre to the current simplex that we are considering.
    simplex *s;

    // Go through every simplex in the list.
    while ((s = nextElement(m->tets, &iter))) {
#if DEBUG >= 1
        printf("# Checking simplex: %p.\n", s);
#endif

        // Go through each neighbour of the simplex (this is equivilent
        // to going through every face of the simplex).
        for (j = 0; j < 4; j++) {
            vertex *p1, *p2, *p3, *p4, *t1, *t2, *t3, *t4;;

            // Get the verticies of the face we are considering.
            getFaceVerticies(s, j, &p1, &p2, &p3, &p4);

            // This is the neighbour that should share the given verticies.
            simplex *neighbour = s->s[j];

#if DEBUG >= 1
            printf("  Neighbour: s->[%d]: %p\n", j, neighbour);
#endif

            // This could be an outer-face: in which case, there is no neighbour here.
            if (neighbour != NULL) {
                int x, found = 0;

                // Go through each neighbour and see if it points to us.
                // if it does (which it should) check the points match.
                for (x = 0; x < 4; x++) {
#if DEBUG >= 1
                    printf("  Checking: %p\n", neighbour->s[x]);
#endif
                    if (neighbour && neighbour->s[x] && neighbour->s[x] == s) {
                        found = 1;

                        // Get the verticies of the face that we share with the current
                        // simplex.
                        getFaceVerticies(neighbour, x, &t1, &t2, &t3, &t4);

                        // We want to check that these two simplicies share their first
                        // three verticies.
                        getFaceVerticies(neighbour, x, &t1, &t2, &t3, &t4);

                        assert(vercmp(t1, p1) || vercmp(t2, p1) || vercmp(t3, p1));
                        assert(vercmp(t1, p2) || vercmp(t2, p2) || vercmp(t3, p2));
                        assert(vercmp(t1, p3) || vercmp(t2, p3) || vercmp(t3, p3));
                    }
                }
                // We have a pointer to a neighbour which does not point back to us.
                assert(found);
            }
        }
    }
}

/******************************************************************************/

int vercmp(vertex *v1, vertex *v2) {
    int i;
    for (i = 0; i < 3; i++) {
        if (v1->v[i] != v2->v[i]) { return 0; }
    }
    return 1;
}

/******************************************************************************/
// This is a slightly optimised method to find the containing simplex
// of a point. We go through each simplex, check to see which faces, if any
// face the point we are looking for. The first one we find that does, we
// follow that neighbour. If all the faces are oriented so that the point is
// not in front of them, then we know that we have found the containing simplex.
// It is likely to be provably O(n^1/2).


simplex *findContainingSimplex(mesh *m, vertex *p) {
    // This will arbitrarily get the first simplex to consider.
    // ideally we want to start from the middle, but chosing a random
    // simplex will give us good performance in general.

    simplex *s;
    if (m->kd == NULL) {
        listNode *iter = topOfLinkedList(m->tets);
        s = nextElement(m->tets, &iter);
    } else {
        int kd_id = kd_nearest3_data(m->kd, p->v[0], p->v[1], p->v[2]);
        s = m->simplicies_kd[kd_id];
    }
    vertex *v1, *v2, *v3;

    for (int i = 0; i < 4; i++) {
        getFaceVerticies3(s, i, &v1, &v2, &v3);
        if (orient3dfast(v1->v, v2->v, v3->v, p->v) < 0) {
            if (!s->s[i]) return NULL; // point is outside the mesh
            s = s->s[i];
            i = -1;
        }
    }
    return s;
}

/******************************************************************************/
// Return, as 3 arrays of double, the verticies of the face i of this simplex.
// This function aims to help us ensure consistant orientation.
// The last value is that of the remaining vertex which is left over.

static inline void getFaceVerticies(simplex *s, int i, vertex **p1, vertex **p2,
                                    vertex **p3, vertex **p4) {
    switch (i) {
        case 0:
            *p1 = s->p[0];
            *p2 = s->p[1];
            *p3 = s->p[2];
            *p4 = s->p[3];
            break;
        case 1:
            *p1 = s->p[3];
            *p2 = s->p[1];
            *p3 = s->p[0];
            *p4 = s->p[2];
            break;
        case 2:
            *p1 = s->p[0];
            *p2 = s->p[2];
            *p3 = s->p[3];
            *p4 = s->p[1];
            break;
        case 3:
            *p1 = s->p[3];
            *p2 = s->p[2];
            *p3 = s->p[1];
            *p4 = s->p[0];
            break;
    }
}

static inline void getFaceVerticies3(simplex *s, int i, vertex **p1, vertex **p2, vertex **p3) {
    switch (i) {
        case 0:
            *p1 = s->p[0];
            *p2 = s->p[1];
            *p3 = s->p[2];
            break;
        case 1:
            *p1 = s->p[3];
            *p2 = s->p[1];
            *p3 = s->p[0];
            break;
        case 2:
            *p1 = s->p[0];
            *p2 = s->p[2];
            *p3 = s->p[3];
            break;
        case 3:
            *p1 = s->p[3];
            *p2 = s->p[2];
            *p3 = s->p[1];
            break;
    }
}


/******************************************************************************/
// This routine will tell us whether or not a simplex contains a given point.

int simplexContainsPoint(simplex *s, vertex *p) {
    vertex *p1, *p2, *p3;
    for (int i = 0; i < 4; i++) {
        getFaceVerticies3(s, i, &p1, &p2, &p3);
        if (orient3dfast(p1->v, p2->v, p3->v, p->v) < 0) { return 0; }
    }
    return 1;
}

/******************************************************************************/
// Write out all the tets in the list, except for those ones connected to
// the points on S0: which we can use as the super simplex.

void writeTetsToFile(mesh *m) {
    FILE *f = fopen("./tets.mat", "wt");
    if (!f) {
        fprintf(stderr, "Could not open tet. file for writing.\n");
        exit(1);
    }

    simplex *s;
    listNode *iter = topOfLinkedList(m->tets);

    while ((s = nextElement(m->tets, &iter))) {
        int super = 0;
        for (int i = 0; i < 4; i++) {
            if (pointOnSimplex(s->p[i], m->super)) { super = 1; }
        }
        if (!super) {
            fprintf(f, "%d %d %d %d\n", s->p[0]->index, s->p[1]->index,
                    s->p[2]->index, s->p[3]->index);
        }
    }
    fclose(f);
}

/******************************************************************************/
// Add gradually larger random perturbations to this point, until we can
// get a sphere which is not degenerate.

void randomPerturbation(vertex *v, int attempt) {
    int i;
    for (i = 0; i < 3; i++) {
        // Get a [0,1] distributed random variable.
        double rand01 = (double) rand() / ((double) RAND_MAX + 1);
        // add a random perturbation to each component of this vertex.
        double p = (rand01 - 0.5) * PERTURBATION_VALUE * (attempt + 1);
        v->v[i] += p;
    }
}

/******************************************************************************/
// This routine will return 0 if the simplex is no longer Delaunay with
// the addition of this new point, 1 if this simplex is still Delaunay
// with the addition of this new point, and -1 if this simplex is
// degenerate with the addition of this new point (i.e. if the simplex is
// co-spherical.)

int isDelaunay(simplex *s, vertex *p) {
    double orientation = orient3dfast(s->p[0]->v,
                                      s->p[1]->v,
                                      s->p[2]->v,
                                      s->p[3]->v);


    if (orientation <= 0) {
        printf("orientation error: %p, %lf\n", s, orientation);

        exit(1);
    }

    double inSph = inspherefast(s->p[0]->v,
                                s->p[1]->v,
                                s->p[2]->v,
                                s->p[3]->v, p->v);


    // We have a degenerecy.
    if (inSph == 0) { return -1; }

    return inSph < 0;

}

/******************************************************************************/
// We assume that the list is correct on starting (i.e. contains no
// non-conflicting simplicies).

void updateConflictingSimplicies(vertex *p, mesh *m) {
    int i;
    // Get at least one simplex which contains this point.
    simplex *s0 = findContainingSimplex(m, p);
    simplex *current;

    // Increment generation for stamp-based O(1) contains check
    m->simplex_generation++;
    uint64_t gen = m->simplex_generation;

    // Go through each simplex, if it contains neighbours which are
    // not already present, which are not in the list already,
    // and which are not delaunay, we add them to the list of conflicts
    stack *toCheck = m->reusable_bfs;
    emptyStack(toCheck);
    push(toCheck, s0);
    while (!isEmpty(toCheck)) {
        // pop the next one to check from the stack.
        current = pop(toCheck);

        // Already visited this simplex in this generation
        if (current->mark == gen) { continue; }

        int isDel = isDelaunay(current, p);

        // Check to see whether or not we have a degenerecy
        if (isDel == -1) {
            m->cospherical_degenerecies++;
            int i = 0;
            while (isDel == -1) {
                randomPerturbation(p, i);
                isDel = isDelaunay(current, p);
                i++;
            }

            // Start this function again now that we have moved the point.
            emptyArrayList(m->conflicts);
            updateConflictingSimplicies(p, m);
            return;
        }

        if (!isDel) {
            // Stamp and add this simplex, and check its neighbours.
            current->mark = gen;
            addToArrayList(m->conflicts, current);
            for (i = 0; i < 4; i++) {
                if (current->s[i]) {
                    push(toCheck, current->s[i]);
                }
            }
        }

    }
}

/******************************************************************************/
// Add a point by using the edge flipping algorithm.

void addPoint(vertex *p, mesh *m) {
    // This will set a list of conflicting non-Delaunay simplicies in the mesh
    // structure.
    updateConflictingSimplicies(p, m);

    // We now have a list of simplicies which contain the point p within
    // their circum-sphere.
    uint64_t gen = m->simplex_generation;
    int i, j;
    for (j = 0; j < arrayListSize(m->conflicts); j++) {
        simplex *s = getFromArrayList(m->conflicts, j);

        for (i = 0; i < 4; i++) {
            vertex *v1, *v2, *v3;
            getFaceVerticies3(s, i, &v1, &v2, &v3);

            // Now, check to see whether or not this face is shared with any
            // other simplicies in the list (use stamp check).
            if (s->s[i] == NULL || s->s[i]->mark != gen) {
                // We will create a new simplex connecting this face to our point.
                simplex *new = newSimplex(m);
                new->p[0] = v1;
                new->p[1] = v2;
                new->p[2] = v3;
                new->p[3] = p;

                int attempt = 0;
                // Detect degenerecies resulting from coplanar points.
                double o = orient3dfast(v1->v, v2->v, v3->v, p->v);
                if (o <= 0) {
                    m->coplanar_degenerecies++;
                    while (o <= 0) {
                        randomPerturbation(p, attempt);
                        o = orient3dfast(v1->v, v2->v, v3->v, p->v);
                        attempt++;
                    }
                    // We are going to have to start adding this point again.
                    push(m->deadSimplicies, new);
                    undoNeighbourUpdates(m->neighbourUpdates);
                    int k;
                    for (k = 0; k < arrayListSize(m->updates); k++) {
                        removeSimplexFromMesh(m, getFromArrayList(m->updates, k));
                        push(m->deadSimplicies, getFromArrayList(m->updates, k));
                    }
                    emptyArrayList(m->updates);
                    emptyArrayList(m->conflicts);
                    addPoint(p, m);
                    return;
                }
                new->s[0] = s->s[i];

                // update, storing each neighbour pointer change we make.
                simplex **update = swapSimplexNeighbour(s->s[i], s, new);
                pushNeighbourUpdate(m->neighbourUpdates, update, s);

                addToArrayList(m->updates, new);
                addSimplexToMesh(m, new);
            }
        }
    }

    // Connect up the internal neighbours of all our new simplicies.
    setNeighbours(m->updates, m);

    // Remove the conflicting simplicies.
    for (i = 0; i < arrayListSize(m->conflicts); i++) {
        simplex *s = getFromArrayList(m->conflicts, i);
        removeSimplexFromMesh(m, s);
    }
}

/******************************************************************************/
// Connect internal neighbours of new simplices using edge sorting (O(k log k)).

typedef struct {
    uintptr_t v_lo, v_hi;
    simplex *s;
    int slot;
} edge_entry;

static int edge_compare(const void *a, const void *b) {
    const edge_entry *ea = (const edge_entry *)a;
    const edge_entry *eb = (const edge_entry *)b;
    if (ea->v_lo != eb->v_lo) return (ea->v_lo < eb->v_lo) ? -1 : 1;
    if (ea->v_hi != eb->v_hi) return (ea->v_hi < eb->v_hi) ? -1 : 1;
    return 0;
}

void setNeighbours(arrayList *newTets, mesh *m) {
    int k = arrayListSize(newTets);
    int nEntries = 3 * k;
    /* Reuse mesh-owned edge buffer to avoid malloc/free per insertion */
    if (nEntries > m->edge_buf_cap) {
        m->edge_buf_cap = nEntries * 2;
        free(m->edge_buf);
        m->edge_buf = malloc(m->edge_buf_cap * sizeof(edge_entry));
    }
    edge_entry *entries = (edge_entry *)m->edge_buf;

    for (int j = 0; j < k; j++) {
        simplex *s = getFromArrayList(newTets, j);
        // Outer face vertices: p[0], p[1], p[2]. p[3] is new point.
        // Edge (p[0],p[1]) -> slot 1
        // Edge (p[2],p[0]) -> slot 2
        // Edge (p[1],p[2]) -> slot 3
        vertex *pairs[3][2] = {
            {s->p[0], s->p[1]},
            {s->p[2], s->p[0]},
            {s->p[1], s->p[2]}
        };
        int slots[3] = {1, 2, 3};
        for (int e = 0; e < 3; e++) {
            uintptr_t a = (uintptr_t)pairs[e][0];
            uintptr_t b = (uintptr_t)pairs[e][1];
            entries[3*j + e].v_lo = a < b ? a : b;
            entries[3*j + e].v_hi = a < b ? b : a;
            entries[3*j + e].s = s;
            entries[3*j + e].slot = slots[e];
        }
    }

    qsort(entries, nEntries, sizeof(edge_entry), edge_compare);

    // Adjacent matching pairs share an edge; connect them bidirectionally.
    for (int i = 0; i < nEntries - 1; i++) {
        if (entries[i].v_lo == entries[i+1].v_lo && entries[i].v_hi == entries[i+1].v_hi) {
            entries[i].s->s[entries[i].slot] = entries[i+1].s;
            entries[i+1].s->s[entries[i+1].slot] = entries[i].s;
        }
    }

}

/******************************************************************************/

int shareThreePoints(simplex *s0, int i, simplex *s1) {
    vertex *v1, *v2, *v3, *v4;

    getFaceVerticies(s0, i, &v1, &v2, &v3, &v4);

    return (pointOnSimplex(v1, s1) && pointOnSimplex(v2, s1) &&
            pointOnSimplex(v3, s1));
}

/******************************************************************************/
// Print an edge of a simplex to an output stream.

void printEdge(vertex *v1, vertex *v2, FILE *stream) {
    fprintf(stream, "%lf %lf %lf   %lf %lf %lf\n",
            v1->X, v2->X, v1->Y, v2->Y, v1->Z, v2->Z);
}

/******************************************************************************/
// Does this simplex have the point p?

int pointOnSimplex(vertex *p, simplex *s) {
    if (!s) { return 0; }

    if (p == s->p[0] || p == s->p[1] || p == s->p[2] || p == s->p[3]) {
        return 1;
    }

    return 0;
}

/******************************************************************************/
// This routine tell us the neighbour of a simplex which is _not_ connected
// to the given point.

simplex *findNeighbour(simplex *s, vertex *p) {
    vertex *t1, *t2, *t3, *t4;
    int i, found = 0;
    for (i = 0; i < 4; i++) {
        getFaceVerticies(s, i, &t1, &t2, &t3, &t4);
        if (t4 == p) {
            found = 1;
            break;
        }
    }

    // If this fails then we couldn't find this point on this simplex.
    assert(found);

    return s->s[i];
}

/******************************************************************************/

int isConvex(vertex *v1, vertex *v2, vertex *v3, vertex *t, vertex *b) {
    int i = 0;
    if (orient3dfast(v3->v, t->v, v1->v, b->v) < 0) { i++; }
    if (orient3dfast(v1->v, t->v, v2->v, b->v) < 0) { i++; }
    if (orient3dfast(v2->v, t->v, v3->v, b->v) < 0) { i++; }

    return (i == 0);
}

/******************************************************************************/

void addSimplexToMesh(mesh *m, simplex *s) {
    s->node = addToLinkedList(m->tets, s);
}

/******************************************************************************/

void removeSimplexFromMesh(mesh *m, simplex *s) {
    removeFromLinkedList(m->tets, s->node);
}

/******************************************************************************/
// This will create a 'super simplex' that contains all of our data to form a
// starting point for our triangulation.

void initSuperSimplex(vertex *ps, int n, mesh *m) {
    int i;
    m->super = newSimplex(m);

    // Get the range of our data set.
    vertex min, max, range;
    getRange(ps, n, &min, &max, &range, 1);

    // We will go clockwise around the base, and then do the top.
    m->superVerticies[0].X = min.X + range.X / 2;
    m->superVerticies[0].Y = max.Y + 3 * range.Y;
    m->superVerticies[0].Z = min.Z - range.Z;

    m->superVerticies[1].X = max.X + 2 * range.X;
    m->superVerticies[1].Y = min.Y - 2 * range.Y;
    m->superVerticies[1].Z = min.Z - range.Z;

    m->superVerticies[2].X = min.X - 2 * range.X;
    m->superVerticies[2].Y = min.Y - 2 * range.Y;
    m->superVerticies[2].Z = min.Z - range.Z;

    m->superVerticies[3].X = min.X + range.X / 2;
    m->superVerticies[3].Y = min.Y + range.Y / 2;
    m->superVerticies[3].Z = max.Z + 2 * range.Z;

    // The super-simplex doesn't have any neighbours.
    for (i = 0; i < 4; i++) {
        m->superVerticies[i].index = -1 - i;
        m->super->p[i] = &m->superVerticies[i];
        m->super->s[i] = NULL;
    }
}

/******************************************************************************/

void pushNeighbourUpdate(neighbourUpdate *nu, simplex **ptr, simplex *old) {
    push(nu->ptrs, ptr);
    push(nu->old, old);
}

/******************************************************************************/

void freeNeighbourUpdates(neighbourUpdate *nu) {
    freeStack(nu->ptrs, NULL); /* elements are interior pointers (&s->s[i]) */
    freeStack(nu->old, NULL);  /* elements are simplex* owned elsewhere */
    free(nu);
}

/******************************************************************************/

void undoNeighbourUpdates(neighbourUpdate *nu) {
    simplex **thisPtr;
    simplex *thisSimplex;

    while (!isEmpty(nu->ptrs)) {
        thisPtr = pop(nu->ptrs);
        thisSimplex = pop(nu->old);

        if (thisPtr) {
            *thisPtr = thisSimplex;
        }
    }
}

/******************************************************************************/

void resetNeighbourUpdates(neighbourUpdate *nu) {
    emptyStack(nu->ptrs);
    emptyStack(nu->old);
}

/******************************************************************************/

neighbourUpdate *initNeighbourUpdates() {
    neighbourUpdate *nu = malloc(sizeof(neighbourUpdate));
    nu->ptrs = newStack();
    nu->old = newStack();
    return nu;
}

/******************************************************************************/
// Allocate all the strucutres required to maintain a mesh in memory.

mesh *newMesh() {
    mesh *m = malloc(sizeof(mesh));
    m->super = NULL;
    m->tets = newLinkedList();
    m->deadSimplicies = newStack();
    m->conflicts = newArrayList();
    m->updates = newArrayList();
    m->neighbourUpdates = initNeighbourUpdates();

    m->kd = NULL;
    m->owns_kd = false;
    m->simplicies_kd = NULL;
    m->simplex_generation = 0;
    m->packed_simplices = NULL;
    m->num_packed = 0;
    m->reusable_bfs = newStack();
    m->edge_buf = NULL;
    m->edge_buf_cap = 0;
    return m;
}

/******************************************************************************/

void freeMesh(mesh *m) {
#ifdef DEBUG
    printf("Mallocs for vertex: %d.\n", VERTEX_MALLOC);
    printf("Mallocs for simplex: %d.\n", SIMPLEX_MALLOC);
#endif

    /* m->super is in deadSimplicies (pushed there as a conflict during the first
       addPoint), so don't free it separately — freeStack handles it. */
    free(m->packed_simplices);
    freeStack(m->reusable_bfs, NULL);
    free(m->edge_buf);
    freeStack(m->deadSimplicies, free);
    freeLinkedList(m->tets, free);
    freeArrayList(m->conflicts, NULL); /* empty after normal build; elements owned by deadSimplicies */
    freeArrayList(m->updates, NULL);
    freeNeighbourUpdates(m->neighbourUpdates);
    free(m->simplicies_kd);
    if (m->owns_kd) {
        kd_free(m->kd);
    }
    free(m);
}

/******************************************************************************/
// This will give us the volume of the arbitrary tetrahedron formed by
// v1, v2, v3, v4

double volumeOfTetrahedron(double *a, double *b, double *c, double *d) {
    double a_d[3], b_d[3], c_d[3], cross[3];

    vertexSub(a, d, a_d);
    vertexSub(b, d, b_d);
    vertexSub(c, d, c_d);

    crossProduct(b_d, c_d, cross);
    double v = scalarProduct(a_d, cross) / (double) 6;

    return (v >= 0) ? v : -v;
}

/******************************************************************************/

double squaredDistance(double *a) {
    return scalarProduct(a, a);
}

/******************************************************************************/
// Take the cross product of two verticies and put it in the vertex 'out'.
void crossProduct(double *b, double *c, double *out) {
    out[0] = b[1] * c[2] - b[2] * c[1];
    out[1] = b[2] * c[0] - b[0] * c[2];
    out[2] = b[0] * c[1] - b[1] * c[0];
}

/******************************************************************************/

double scalarProduct(double *a, double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/******************************************************************************/

void vertexSub(double *a, double *b, double *out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

/******************************************************************************/

void vertexAdd(double *a, double *b, double *out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

/******************************************************************************/

void vertexByScalar(double *a, double b, double *out) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
}

/******************************************************************************/
// This function will compute the circumcenter of a given simplex.

void circumCenter(simplex *s, double *out) {
    vertex *a, *b, *c, *d;
    getFaceVerticies(s, 0, &a, &b, &c, &d);

    double b_a[3], c_a[3], d_a[3],
            cross1[3], cross2[3], cross3[3],
            mult1[3], mult2[3], mult3[3],
            sum[3];
    double denominator;

    // Calculate diferences between points.
    vertexSub(b->v, a->v, b_a);
    vertexSub(c->v, a->v, c_a);
    vertexSub(d->v, a->v, d_a);

    // Calculate first cross product.
    crossProduct(b_a, c_a, cross1);

    // Calculate second cross product.
    crossProduct(d_a, b_a, cross2);

    // Calculate third cross product.
    crossProduct(c_a, d_a, cross3);

    vertexByScalar(cross1, squaredDistance(d_a), mult1);
    vertexByScalar(cross2, squaredDistance(c_a), mult2);
    vertexByScalar(cross3, squaredDistance(b_a), mult3);

    // Add up the sum of the numerator.
    vertexAdd(mult1, mult2, sum);
    vertexAdd(mult3, sum, sum);

    // Calculate the denominator.
    denominator = 2 * scalarProduct(b_a, cross3);

    // Do the division, and output to out.
    vertexByScalar(sum, 1 / (double) (denominator), out);

    vertexAdd(out, a->v, out);
}

/******************************************************************************/

int getNumSimplicies(mesh *m) {
    return linkedListSize(m->tets);
}

/******************************************************************************/

int numSphericalDegenerecies(mesh *m) {
    return m->cospherical_degenerecies;
}

/******************************************************************************/

int numPlanarDegenerecies(mesh *m) {
    return m->coplanar_degenerecies;
}

/******************************************************************************/

void getRange(vertex *ps, int n, vertex *min, vertex *max, vertex *range, int r) {
    int i;

    *min = ps[0];
    *max = ps[0];

    for (i = 0; i < n; i++) {
        if (0) {
            ps[i].X += ((double) rand() / ((double) RAND_MAX + 1) - 0.5);
            ps[i].Y += ((double) rand() / ((double) RAND_MAX + 1) - 0.5);
            ps[i].Z += ((double) rand() / ((double) RAND_MAX + 1) - 0.5);
        }

        max->X = MAX(max->X, ps[i].X);
        max->Y = MAX(max->Y, ps[i].Y);
        max->Z = MAX(max->Z, ps[i].Z);

        min->X = MIN(min->X, ps[i].X);
        min->Y = MIN(min->Y, ps[i].Y);
        min->Z = MIN(min->Z, ps[i].Z);
    }

    for (i = 0; i < 3; i++) {
        range->v[i] = max->v[i] - min->v[i];
    }
}

/*******************************************************************************
* Unit testing.                                                                *
*******************************************************************************/

#ifdef _TEST_

#include <sys/time.h>
#undef NDEBUG
#define NUM_TEST_POINTS 1e4

/******************************************************************************/

double getTime()
{
struct timeval tv;
gettimeofday(&tv,NULL);
return tv.tv_sec + tv.tv_usec/1.0e6;
}

/******************************************************************************/

int main(int argc, char **argv)
{
int i;
srand ( time(NULL) );

// Create a random pointset for testing.
vertex *ps = malloc(sizeof(vertex)*NUM_TEST_POINTS);

for (i=0; i<NUM_TEST_POINTS; i++)
{
  ps[i].X = (double)rand() / ((double)RAND_MAX + 1);
  ps[i].Y = (double)rand() / ((double)RAND_MAX + 1);
  ps[i].Z = (double)rand() / ((double)RAND_MAX + 1);
  ps[i].index = i;
}

mesh *delaunayMesh = newMesh();

// Build the mesh, timing how long it takes.
double t1 = getTime();
buildMesh(ps, NUM_TEST_POINTS, delaunayMesh);
double t2 = getTime();

int n = NUM_TEST_POINTS;
printf("\nMeshed %d points using %d simplicies in %lf seconds.\n", n,
                                   getNumSimplicies(delaunayMesh), t2-t1);
printf("Co-planar degenerecies fixed: %d.\n",
                                   numPlanarDegenerecies(delaunayMesh));
printf("Co-spherical degenerecies fixed: %d.\n",
                                   numSphericalDegenerecies(delaunayMesh));

printf("Now testing mesh...\n");

orientationTest(delaunayMesh->tets);
delaunayTest(delaunayMesh, ps, NUM_TEST_POINTS);
faceTest(delaunayMesh);

freeMesh(delaunayMesh);
printf("Testing Complete.\n");
return 0;
}

/******************************************************************************/
#endif

