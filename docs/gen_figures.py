"""Generate SVG illustrations for insertion-free Sibson explanation."""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


OUTPUT_DIR = pathlib.Path(__file__).parent / '_static'


def circumcenter_2d(a, b, c):
    """Circumcenter of triangle (a, b, c) in 2D."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / D
    return np.array([ux, uy])


def in_circumcircle(a, b, c, p):
    """Return True if p is strictly inside the circumcircle of (a, b, c)."""
    cc = circumcenter_2d(a, b, c)
    r2 = np.sum((cc - a)**2)
    return np.sum((cc - p)**2) < r2


def midpoint(a, b):
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Figure 1: Cavity and boundary
# ---------------------------------------------------------------------------

def gen_cavity_boundary():
    rng = np.random.RandomState(42)
    pts = rng.rand(18, 2) * 4 - 0.2
    # Push points away from center to leave room for query
    pts = np.vstack([pts, [[0.5, 3.5], [3.5, 0.5]]])

    tri = Delaunay(pts)
    q = np.array([1.8, 1.7])

    # Find cavity triangles
    cavity = set()
    for i, simplex in enumerate(tri.simplices):
        a, b, c = pts[simplex]
        if in_circumcircle(a, b, c, q):
            cavity.add(i)

    # Cavity vertices = natural neighbors (excluding super-triangle, all real here)
    cavity_verts = set()
    for i in cavity:
        for v in tri.simplices[i]:
            cavity_verts.add(v)

    # Boundary edges: edges of cavity tris where the neighbor is outside cavity or -1
    boundary_edges = []
    for i in cavity:
        for j in range(3):
            nbr = tri.neighbors[i][j]
            if nbr == -1 or nbr not in cavity:
                # The edge opposite vertex j in the simplex
                edge_verts = [tri.simplices[i][k] for k in range(3) if k != j]
                boundary_edges.append(tuple(sorted(edge_verts)))

    boundary_edges = list(set(boundary_edges))

    # Boundary vertex set (natural neighbors)
    boundary_verts = set()
    for e in boundary_edges:
        boundary_verts.update(e)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Draw all triangulation edges (thin gray)
    edges_set = set()
    for simplex in tri.simplices:
        for j in range(3):
            for k in range(j + 1, 3):
                e = tuple(sorted((simplex[j], simplex[k])))
                edges_set.add(e)

    for e in edges_set:
        ax.plot(pts[list(e), 0], pts[list(e), 1], color='#bbbbbb', lw=0.6, zorder=1)

    # Shade cavity triangles
    for i in cavity:
        verts = pts[tri.simplices[i]]
        patch = plt.Polygon(verts, facecolor='#f4cccc', edgecolor='none', zorder=2)
        ax.add_patch(patch)

    # Draw boundary edges (thick dark blue)
    for e in boundary_edges:
        ax.plot(pts[list(e), 0], pts[list(e), 1], color='#1a3a6b', lw=2.2, zorder=4)

    # Draw virtual edges from q to boundary vertices (dashed blue)
    for v in boundary_verts:
        ax.plot([q[0], pts[v, 0]], [q[1], pts[v, 1]],
                color='#4a7ab5', lw=1.2, ls='--', zorder=3)

    # Draw points
    non_neighbor = [i for i in range(len(pts)) if i not in boundary_verts]
    ax.scatter(pts[non_neighbor, 0], pts[non_neighbor, 1],
               c='#999999', s=20, zorder=5, edgecolors='#666666', linewidths=0.5)
    neighbor_list = list(boundary_verts)
    ax.scatter(pts[neighbor_list, 0], pts[neighbor_list, 1],
               c='#3366aa', s=30, zorder=5, edgecolors='#1a3a6b', linewidths=0.7)

    # Query point
    ax.scatter([q[0]], [q[1]], c='#cc3333', s=50, zorder=6, edgecolors='#881111',
               linewidths=0.8, marker='*')
    ax.annotate('q', q, textcoords='offset points', xytext=(6, 6),
                fontsize=11, fontweight='bold', color='#cc3333', zorder=7)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(OUTPUT_DIR / 'cavity-boundary.svg',
                bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.close(fig)
    print('  cavity-boundary.svg')


# ---------------------------------------------------------------------------
# Figure 2: Voronoi subcell R(p_k, T)
# ---------------------------------------------------------------------------

def gen_voronoi_subcell():
    # Acute triangle with circumcenter inside
    pk = np.array([0.0, 0.0])
    pa = np.array([3.0, 0.5])
    pb = np.array([1.0, 2.8])

    cc = circumcenter_2d(pk, pa, pb)
    ma = midpoint(pk, pa)
    mb = midpoint(pk, pb)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    # Draw the triangle
    tri_verts = np.array([pk, pa, pb, pk])
    ax.plot(tri_verts[:, 0], tri_verts[:, 1], color='#444444', lw=1.5, zorder=2)

    # Shade the subcell R(p_k, T) = quadrilateral (pk, ma, cc, mb)
    subcell = plt.Polygon([pk, ma, cc, mb], facecolor='#c6dbef', edgecolor='#2166ac',
                          lw=1.5, zorder=3, alpha=0.7)
    ax.add_patch(subcell)

    # Dashed line from pk to cc (splitting the subcell)
    ax.plot([pk[0], cc[0]], [pk[1], cc[1]], color='#2166ac', lw=1.0, ls='--', zorder=4)

    # Perpendicular bisector segments: from midpoint to circumcenter
    ax.plot([ma[0], cc[0]], [ma[1], cc[1]], color='#888888', lw=1.0, ls=':', zorder=2)
    ax.plot([mb[0], cc[0]], [mb[1], cc[1]], color='#888888', lw=1.0, ls=':', zorder=2)

    # Mark points
    for pt, label, offset, color in [
        (pk, r'$p_k$', (-14, -12), '#000000'),
        (pa, r'$p_a$', (6, -6), '#000000'),
        (pb, r'$p_b$', (-6, 8), '#000000'),
        (ma, r'$m_a$', (6, -8), '#555555'),
        (mb, r'$m_b$', (-18, 4), '#555555'),
        (cc, r'$c_T$', (6, 4), '#2166ac'),
    ]:
        ax.scatter([pt[0]], [pt[1]], c=color, s=25, zorder=6, edgecolors='none')
        ax.annotate(label, pt, textcoords='offset points', xytext=offset,
                    fontsize=11, color=color, zorder=7)

    # Label the subcell
    centroid = (pk + ma + cc + mb) / 4
    ax.annotate(r'$R(p_k, T)$', centroid, textcoords='offset points', xytext=(4, -2),
                fontsize=10, color='#2166ac', fontstyle='italic', zorder=7,
                ha='center', va='center')

    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(OUTPUT_DIR / 'voronoi-subcell.svg',
                bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.close(fig)
    print('  voronoi-subcell.svg')


# ---------------------------------------------------------------------------
# Figure 3: Signed volumes — why they work
# ---------------------------------------------------------------------------

def signed_area(a, b, c):
    """Signed area of triangle (a, b, c)."""
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def gen_signed_volume():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # -- Left panel: acute triangle, circumcenter inside --
    ax = axes[0]
    pk = np.array([0.0, 0.0])
    pa = np.array([3.0, 0.5])
    pb = np.array([1.0, 2.8])

    cc = circumcenter_2d(pk, pa, pb)
    ma = midpoint(pk, pa)
    mb = midpoint(pk, pb)

    # Triangle
    tri_verts = np.array([pk, pa, pb, pk])
    ax.plot(tri_verts[:, 0], tri_verts[:, 1], color='#444444', lw=1.5, zorder=2)

    # Both sub-triangles positive (same blue shade)
    t1 = plt.Polygon([pk, ma, cc], facecolor='#a6c8e0', edgecolor='#2166ac',
                      lw=1.0, zorder=3, alpha=0.7)
    t2 = plt.Polygon([pk, cc, mb], facecolor='#a6c8e0', edgecolor='#2166ac',
                      lw=1.0, zorder=3, alpha=0.7)
    ax.add_patch(t1)
    ax.add_patch(t2)

    # Dashed split line
    ax.plot([pk[0], cc[0]], [pk[1], cc[1]], color='#2166ac', lw=1.0, ls='--', zorder=4)

    # Mark circumcenter
    ax.scatter([cc[0]], [cc[1]], c='#2166ac', s=25, zorder=6)
    ax.annotate(r'$c_T$', cc, textcoords='offset points', xytext=(6, 4),
                fontsize=10, color='#2166ac', zorder=7)

    # Plus signs in both triangles
    c1 = (pk + ma + cc) / 3
    c2 = (pk + cc + mb) / 3
    ax.text(c1[0], c1[1], '+', fontsize=14, color='#2166ac', ha='center', va='center',
            fontweight='bold', zorder=5)
    ax.text(c2[0], c2[1], '+', fontsize=14, color='#2166ac', ha='center', va='center',
            fontweight='bold', zorder=5)

    ax.set_title('Circumcenter inside', fontsize=11, pad=8)
    ax.set_aspect('equal')
    ax.axis('off')

    # -- Right panel: obtuse triangle, circumcenter outside --
    ax = axes[1]
    # Obtuse triangle: angle at pa > 90, so circumcenter falls outside
    pk2 = np.array([0.0, 0.0])
    pa2 = np.array([4.0, 0.8])
    pb2 = np.array([4.2, 1.6])

    cc2 = circumcenter_2d(pk2, pa2, pb2)
    ma2 = midpoint(pk2, pa2)
    mb2 = midpoint(pk2, pb2)

    # Triangle
    tri_verts2 = np.array([pk2, pa2, pb2, pk2])
    ax.plot(tri_verts2[:, 0], tri_verts2[:, 1], color='#444444', lw=1.5, zorder=2)

    # Check sign of sub-triangles to color them
    area1 = signed_area(pk2, ma2, cc2)
    area2 = signed_area(pk2, cc2, mb2)

    color1 = '#b5d8a6' if area1 > 0 else '#f4a0a0'
    color2 = '#b5d8a6' if area2 > 0 else '#f4a0a0'
    label1 = '+' if area1 > 0 else '\u2212'
    label2 = '+' if area2 > 0 else '\u2212'
    tcolor1 = '#2d7a2d' if area1 > 0 else '#aa2222'
    tcolor2 = '#2d7a2d' if area2 > 0 else '#aa2222'

    t1 = plt.Polygon([pk2, ma2, cc2], facecolor=color1, edgecolor='#444444',
                      lw=1.0, zorder=3, alpha=0.6)
    t2 = plt.Polygon([pk2, cc2, mb2], facecolor=color2, edgecolor='#444444',
                      lw=1.0, zorder=3, alpha=0.6)
    ax.add_patch(t1)
    ax.add_patch(t2)

    # Dashed split line
    ax.plot([pk2[0], cc2[0]], [pk2[1], cc2[1]], color='#666666', lw=1.0, ls='--', zorder=4)

    # Mark circumcenter (outside the triangle)
    ax.scatter([cc2[0]], [cc2[1]], c='#cc3333', s=25, zorder=6)
    ax.annotate(r'$c_T$', cc2, textcoords='offset points', xytext=(6, 4),
                fontsize=10, color='#cc3333', zorder=7)

    # Signs in sub-triangles
    c1 = (pk2 + ma2 + cc2) / 3
    c2 = (pk2 + cc2 + mb2) / 3
    ax.text(c1[0], c1[1], label1, fontsize=14, color=tcolor1, ha='center', va='center',
            fontweight='bold', zorder=5)
    ax.text(c2[0], c2[1], label2, fontsize=14, color=tcolor2, ha='center', va='center',
            fontweight='bold', zorder=5)

    ax.set_title('Circumcenter outside', fontsize=11, pad=8)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout(w_pad=2)
    fig.savefig(OUTPUT_DIR / 'signed-volume.svg',
                bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.close(fig)
    print('  signed-volume.svg')


# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Generating figures...')
    gen_cavity_boundary()
    gen_voronoi_subcell()
    gen_signed_volume()
    print('Done.')


if __name__ == '__main__':
    main()
