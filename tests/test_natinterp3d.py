import numpy as np
import pytest
import scipy.sparse

import natinterp3d


# ── Helpers ──────────────────────────────────────────────────────────────────

def random_interior_queries(keys, n_queries, rng, margin=0.8):
    """Generate queries likely to be inside the convex hull of keys."""
    center = keys.mean(axis=0)
    return center + (rng.randn(n_queries, 3)) * keys.std(axis=0) * margin * 0.3


# ── Mathematical properties of Sibson coordinates ────────────────────────────

class TestPartitionOfUnity:
    """Sibson weights for interior points must sum to 1."""

    def test_random_queries(self):
        rng = np.random.RandomState(42)
        keys = rng.randn(200, 3)
        queries = random_interior_queries(keys, 50, rng)
        W = natinterp3d.get_weights(queries, keys)
        sums = np.asarray(W.sum(axis=1)).ravel()
        # Only check queries that got nonzero weights (i.e. are inside convex hull)
        interior = sums > 0.5
        assert interior.sum() > 30, 'Too few queries ended up inside the convex hull'
        np.testing.assert_allclose(sums[interior], 1.0, atol=1e-10)

    def test_grid_data(self):
        """Grid-structured data (non-random)."""
        xs = np.linspace(-1, 1, 6)
        keys = np.array(np.meshgrid(xs, xs, xs)).reshape(3, -1).T
        rng = np.random.RandomState(7)
        queries = rng.uniform(-0.8, 0.8, (30, 3))
        W = natinterp3d.get_weights(queries, keys)
        sums = np.asarray(W.sum(axis=1)).ravel()
        interior = sums > 0.5
        assert interior.sum() > 20
        np.testing.assert_allclose(sums[interior], 1.0, atol=1e-10)


class TestNonNegativity:
    """All Sibson weights must be non-negative."""

    def test_random(self):
        rng = np.random.RandomState(99)
        keys = rng.randn(300, 3)
        queries = random_interior_queries(keys, 80, rng)
        W = natinterp3d.get_weights(queries, keys)
        assert W.min() >= -1e-12, f'Negative weight found: {W.min()}'

    def test_serial(self):
        rng = np.random.RandomState(99)
        keys = rng.randn(300, 3)
        queries = random_interior_queries(keys, 80, rng)
        W = natinterp3d.get_weights(queries, keys, parallel=False)
        assert W.min() >= -1e-12, f'Negative weight found: {W.min()}'


class TestLinearPrecision:
    """Sibson interpolation must reproduce linear functions exactly."""

    @pytest.fixture()
    def setup(self):
        rng = np.random.RandomState(12)
        keys = rng.randn(500, 3)
        queries = random_interior_queries(keys, 60, rng)
        W = natinterp3d.get_weights(queries, keys)
        sums = np.asarray(W.sum(axis=1)).ravel()
        interior = sums > 0.5
        return keys, queries, W, interior

    def test_constant_function(self, setup):
        keys, queries, W, interior = setup
        values = np.full(len(keys), 7.0)
        result = np.asarray(W @ values[:, None]).ravel()
        np.testing.assert_allclose(result[interior], 7.0, atol=1e-10)

    def test_linear_xyz(self, setup):
        keys, queries, W, interior = setup
        # f(x,y,z) = 2x - 3y + 5z + 1
        coeffs = np.array([2.0, -3.0, 5.0])
        bias = 1.0
        values = keys @ coeffs + bias
        expected = queries @ coeffs + bias
        result = np.asarray(W @ values[:, None]).ravel()
        np.testing.assert_allclose(result[interior], expected[interior], atol=1e-8)

    def test_each_coordinate(self, setup):
        """f(x,y,z) = x, f = y, f = z should each be exact."""
        keys, queries, W, interior = setup
        for dim in range(3):
            values = keys[:, dim]
            expected = queries[:, dim]
            result = np.asarray(W @ values[:, None]).ravel()
            np.testing.assert_allclose(result[interior], expected[interior], atol=1e-8)


class TestLocalCoordinateProperty:
    """sum(w_k * p_k) == q for interior queries (implied by linear precision)."""

    def test_weighted_position(self):
        rng = np.random.RandomState(55)
        keys = rng.randn(300, 3)
        queries = random_interior_queries(keys, 40, rng)
        W = natinterp3d.get_weights(queries, keys)
        sums = np.asarray(W.sum(axis=1)).ravel()
        interior = sums > 0.5
        reconstructed = np.asarray(W @ keys)  # shape (n_queries, 3)
        errors = np.linalg.norm(reconstructed[interior] - queries[interior], axis=1)
        np.testing.assert_allclose(errors, 0.0, atol=1e-8)


class TestConvergence:
    """Interpolation error should decrease with increasing data density."""

    def test_smooth_function(self):
        rng = np.random.RandomState(10)

        def func(x):
            return np.sin(x[:, 0]) + np.cos(x[:, 1]) + 0.5 * x[:, 2]

        queries = rng.randn(100, 3) * 0.3

        errors = []
        for n in [200, 2000, 20000]:
            keys = rng.randn(n, 3)
            values = func(keys)
            result = natinterp3d.interpolate(queries, keys, values)
            expected = func(queries)
            errors.append(np.mean(np.abs(result - expected)))

        # Error should decrease as n grows
        assert errors[1] < errors[0], f'{errors[1]:.4e} not < {errors[0]:.4e}'
        assert errors[2] < errors[1], f'{errors[2]:.4e} not < {errors[1]:.4e}'
        # With 20k points, error should be small
        assert errors[2] < 2e-3


# ── Known geometric configurations ──────────────────────────────────────────

class TestSymmetricConfigurations:
    """Queries at symmetric positions should produce symmetric weights."""

    def test_cube_center(self):
        """Query at center of cube → 8 equal weights.

        Tolerance is loose because the Delaunay construction uses random perturbation
        for highly symmetric/degenerate configurations like cube vertices.
        """
        keys = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ], dtype=np.float64)
        queries = np.array([[0.0, 0.0, 0.0]])
        W = natinterp3d.get_weights(queries, keys)
        w = np.asarray(W.todense()).ravel()
        np.testing.assert_allclose(w, 1.0 / 8, atol=1e-4)

    def test_regular_tetrahedron_centroid(self):
        """Query at centroid of regular tet → 4 equal weights."""
        keys = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
        ], dtype=np.float64)
        centroid = keys.mean(axis=0, keepdims=True)
        W = natinterp3d.get_weights(centroid, keys)
        w = np.asarray(W.todense()).ravel()
        np.testing.assert_allclose(w, 0.25, atol=1e-10)

    def test_octahedron_center(self):
        """Query at center of octahedron → 6 equal weights."""
        keys = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ], dtype=np.float64)
        queries = np.array([[0.0, 0.0, 0.0]])
        W = natinterp3d.get_weights(queries, keys)
        w = np.asarray(W.todense()).ravel()
        np.testing.assert_allclose(w, 1.0 / 6, atol=1e-4)


class TestQueryAtDataPoint:
    """Querying exactly at a data point should yield weight 1 for that point."""

    def test_exact_data_point(self):
        rng = np.random.RandomState(33)
        keys = rng.randn(50, 3)
        for idx in [0, 10, 25, 49]:
            query = keys[idx:idx + 1]
            W = natinterp3d.get_weights(query, keys)
            w = np.asarray(W.todense()).ravel()
            assert w[idx] == pytest.approx(1.0, abs=1e-4), (
                f'Weight at data point {idx} is {w[idx]}, expected ~1.0'
            )
            # All other weights should be ~0
            others = np.delete(w, idx)
            np.testing.assert_allclose(others, 0.0, atol=1e-4)

    def test_interpolation_at_data_point(self):
        rng = np.random.RandomState(34)
        keys = rng.randn(50, 3)
        values = rng.randn(50)
        for idx in [0, 25, 49]:
            query = keys[idx:idx + 1]
            result = natinterp3d.interpolate(query, keys, values)
            np.testing.assert_allclose(result[0], values[idx], atol=1e-3)


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestOutsideConvexHull:
    """Queries outside the convex hull should get zero weights."""

    def test_far_away_query(self):
        rng = np.random.RandomState(20)
        keys = rng.randn(100, 3)
        queries = np.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
        W = natinterp3d.get_weights(queries, keys)
        sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(sums, 0.0, atol=1e-15)

    def test_interpolation_outside(self):
        """Exterior queries should not crash and should return finite values.

        Note: queries inside the super simplex but outside the data convex hull
        may get partial extrapolated weights from hull vertices, so we cannot
        assume the result is zero.
        """
        rng = np.random.RandomState(21)
        keys = rng.randn(100, 3)
        values = rng.randn(100) + 10.0
        queries = np.array([[100.0, 100.0, 100.0]])
        result = natinterp3d.interpolate(queries, keys, values)
        assert np.all(np.isfinite(result))


class TestSmallPointSets:
    """Minimum viable configurations."""

    def test_five_points(self):
        """5 points = simplest non-trivial Delaunay."""
        keys = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5],
        ], dtype=np.float64)
        queries = np.array([[0.2, 0.2, 0.2]])
        W = natinterp3d.get_weights(queries, keys)
        w = np.asarray(W.todense()).ravel()
        total = w.sum()
        assert total == pytest.approx(1.0, abs=1e-8)
        assert np.all(w >= -1e-12)

    def test_four_points(self):
        """4 points = single tetrahedron."""
        keys = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float64)
        queries = np.array([[0.1, 0.1, 0.1]])
        W = natinterp3d.get_weights(queries, keys)
        w = np.asarray(W.todense()).ravel()
        total = w.sum()
        assert total == pytest.approx(1.0, abs=1e-8)
        assert np.all(w >= -1e-12)


class TestSingleQuery:
    def test_single_query(self):
        rng = np.random.RandomState(77)
        keys = rng.randn(50, 3)
        queries = np.array([[0.0, 0.0, 0.0]])
        W = natinterp3d.get_weights(queries, keys)
        assert W.shape == (1, 50)
        values = rng.randn(50)
        result = natinterp3d.interpolate(queries, keys, values)
        assert result.shape == (1,)


class TestLargeCoordinates:
    """Data with large coordinate values (tests scale-independence)."""

    def test_large_offset(self):
        rng = np.random.RandomState(88)
        offset = np.array([1e6, -2e6, 3e6])
        keys = rng.randn(200, 3) + offset
        queries = random_interior_queries(keys, 30, rng)
        W = natinterp3d.get_weights(queries, keys)
        sums = np.asarray(W.sum(axis=1)).ravel()
        interior = sums > 0.5
        assert interior.sum() > 15
        np.testing.assert_allclose(sums[interior], 1.0, atol=1e-6)

    def test_large_scale(self):
        rng = np.random.RandomState(89)
        keys = rng.randn(200, 3) * 1e4
        queries = random_interior_queries(keys, 30, rng)
        W = natinterp3d.get_weights(queries, keys)
        sums = np.asarray(W.sum(axis=1)).ravel()
        interior = sums > 0.5
        assert interior.sum() > 15
        np.testing.assert_allclose(sums[interior], 1.0, atol=1e-6)

    def test_small_scale(self):
        rng = np.random.RandomState(90)
        keys = rng.randn(200, 3) * 1e-4
        queries = random_interior_queries(keys, 30, rng)
        W = natinterp3d.get_weights(queries, keys)
        sums = np.asarray(W.sum(axis=1)).ravel()
        interior = sums > 0.5
        assert interior.sum() > 15
        np.testing.assert_allclose(sums[interior], 1.0, atol=1e-6)


# ── API correctness ─────────────────────────────────────────────────────────

class TestOutputShapes:
    @pytest.fixture()
    def data(self):
        rng = np.random.RandomState(1)
        keys = rng.randn(100, 3)
        queries = rng.randn(50, 3) * 0.3
        return keys, queries

    def test_1d_values(self, data):
        keys, queries = data
        values = np.ones(100)
        result = natinterp3d.interpolate(queries, keys, values)
        assert result.shape == (50,)

    def test_2d_values(self, data):
        keys, queries = data
        values = np.ones((100, 5))
        result = natinterp3d.interpolate(queries, keys, values)
        assert result.shape == (50, 5)

    def test_weight_matrix_shape(self, data):
        keys, queries = data
        W = natinterp3d.get_weights(queries, keys)
        assert W.shape == (50, 100)

    def test_weight_matrix_is_csr(self, data):
        keys, queries = data
        W = natinterp3d.get_weights(queries, keys)
        assert isinstance(W, scipy.sparse.csr_matrix)


class TestParallelSerialConsistency:
    def test_weights_match(self):
        rng = np.random.RandomState(2)
        keys = rng.randn(200, 3)
        queries = rng.randn(80, 3) * 0.3
        W_par = natinterp3d.get_weights(queries, keys, parallel=True)
        W_ser = natinterp3d.get_weights(queries, keys, parallel=False)
        np.testing.assert_allclose(W_par.toarray(), W_ser.toarray(), atol=1e-12)

    def test_interpolation_match(self):
        rng = np.random.RandomState(3)
        keys = rng.randn(200, 3)
        queries = rng.randn(80, 3) * 0.3
        values = rng.randn(200)
        r_par = natinterp3d.interpolate(queries, keys, values, parallel=True)
        r_ser = natinterp3d.interpolate(queries, keys, values, parallel=False)
        np.testing.assert_allclose(r_par, r_ser, atol=1e-12)

    def test_explicit_thread_count(self):
        rng = np.random.RandomState(4)
        keys = rng.randn(100, 3)
        queries = rng.randn(30, 3) * 0.3
        W1 = natinterp3d.get_weights(queries, keys, parallel=True, num_threads=1)
        W2 = natinterp3d.get_weights(queries, keys, parallel=True, num_threads=4)
        W_ser = natinterp3d.get_weights(queries, keys, parallel=False)
        np.testing.assert_allclose(W1.toarray(), W_ser.toarray(), atol=1e-12)
        np.testing.assert_allclose(W2.toarray(), W_ser.toarray(), atol=1e-12)


class TestInterpolatorReuse:
    """The same Interpolator should give consistent results across multiple calls."""

    def test_multiple_get_weights(self):
        rng = np.random.RandomState(5)
        keys = rng.randn(100, 3)
        interp = natinterp3d.Interpolator(keys)
        q1 = rng.randn(20, 3) * 0.3
        q2 = rng.randn(30, 3) * 0.3
        W1a = interp.get_weights(q1)
        W2 = interp.get_weights(q2)
        W1b = interp.get_weights(q1)
        np.testing.assert_allclose(W1a.toarray(), W1b.toarray(), atol=1e-15)
        assert W2.shape == (30, 100)

    def test_multiple_interpolate(self):
        rng = np.random.RandomState(6)
        keys = rng.randn(100, 3)
        values = rng.randn(100)
        interp = natinterp3d.Interpolator(keys)
        queries = rng.randn(20, 3) * 0.3
        r1 = interp.interpolate(queries, values)
        r2 = interp.interpolate(queries, values)
        np.testing.assert_allclose(r1, r2, atol=1e-15)


class TestDtypeHandling:
    def test_float32_input(self):
        rng = np.random.RandomState(50)
        keys = rng.randn(100, 3).astype(np.float32)
        queries = (rng.randn(20, 3) * 0.3).astype(np.float32)
        values = rng.randn(100).astype(np.float32)
        result = natinterp3d.interpolate(queries, keys, values)
        assert result.dtype == np.float64 or result.dtype == np.float32
        assert result.shape == (20,)

    def test_float32_weights(self):
        rng = np.random.RandomState(51)
        keys = rng.randn(100, 3).astype(np.float32)
        queries = (rng.randn(20, 3) * 0.3).astype(np.float32)
        W = natinterp3d.get_weights(queries, keys)
        assert W.shape == (20, 100)
        sums = np.asarray(W.sum(axis=1)).ravel()
        interior = sums > 0.5
        # Should still be correct despite float32 input
        np.testing.assert_allclose(sums[interior], 1.0, atol=1e-6)


class TestConvenienceFunctions:
    """Test the module-level functions match the Interpolator class."""

    def test_interpolate_matches_class(self):
        rng = np.random.RandomState(60)
        keys = rng.randn(100, 3)
        queries = rng.randn(20, 3) * 0.3
        values = rng.randn(100)
        r1 = natinterp3d.interpolate(queries, keys, values, parallel=False)
        interp = natinterp3d.Interpolator(keys, parallel=False)
        r2 = interp.interpolate(queries, values)
        np.testing.assert_allclose(r1, r2, atol=1e-15)

    def test_get_weights_matches_class(self):
        rng = np.random.RandomState(61)
        keys = rng.randn(100, 3)
        queries = rng.randn(20, 3) * 0.3
        W1 = natinterp3d.get_weights(queries, keys, parallel=False)
        interp = natinterp3d.Interpolator(keys, parallel=False)
        W2 = interp.get_weights(queries)
        np.testing.assert_allclose(W1.toarray(), W2.toarray(), atol=1e-15)


# ── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:
    """Repeated calls with the same input must produce identical output."""

    def test_weights_deterministic(self):
        rng = np.random.RandomState(70)
        keys = rng.randn(200, 3)
        queries = rng.randn(50, 3) * 0.3
        W1 = natinterp3d.get_weights(queries, keys, parallel=True)
        W2 = natinterp3d.get_weights(queries, keys, parallel=True)
        np.testing.assert_array_equal(W1.toarray(), W2.toarray())

    def test_serial_deterministic(self):
        rng = np.random.RandomState(71)
        keys = rng.randn(200, 3)
        queries = rng.randn(50, 3) * 0.3
        W1 = natinterp3d.get_weights(queries, keys, parallel=False)
        W2 = natinterp3d.get_weights(queries, keys, parallel=False)
        np.testing.assert_array_equal(W1.toarray(), W2.toarray())


# ── Sparsity ─────────────────────────────────────────────────────────────────

class TestSparsity:
    """Weight matrices should be sparse (not all keys are natural neighbors)."""

    def test_few_nonzeros_per_row(self):
        rng = np.random.RandomState(80)
        keys = rng.randn(1000, 3)
        queries = rng.randn(50, 3) * 0.3
        W = natinterp3d.get_weights(queries, keys)
        # Each interior query should have a modest number of neighbors,
        # certainly far fewer than 1000
        for i in range(W.shape[0]):
            nnz = W[i].nnz
            if nnz > 0:
                assert nnz < 100, f'Query {i} has {nnz} non-zeros, expected << 1000'


# ── Regression: original test (kept for continuity) ─────────────────────────

class TestOriginal:
    """The original test from the first version of the test suite."""

    def test_natinterp3d(self):
        rng = np.random.RandomState(10)
        keys = rng.randn(10000, 3)

        def func(x):
            x = x * 0.1
            return np.sin(x[:, 0]) + np.cos(x[:, 1]) + np.tan(x[:, 2])

        values = func(keys)
        queries = rng.randn(100, 3)
        interp_values = natinterp3d.interpolate(queries, keys, values)
        actual_values = func(queries)
        assert np.mean(np.abs(interp_values - actual_values)) < 1e-3