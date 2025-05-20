"""
Tests for the LinearSearchCosine vector index.
"""

import threading
import time

import numpy as np
import pytest

from app.indexing.linear_search import DuplicateVectorError, LinearSearchCosine


class TestLinearSearchCosine:
    """Test suite for LinearSearchCosine."""

    def test_init(self):
        """Test initialization with dimension."""
        index = LinearSearchCosine[str](dim=3)
        assert index.size() == 0

    def test_dimension_validation(self):
        """Test dimension validation."""
        index = LinearSearchCosine[str](dim=3)

        # Valid dimension
        index.add("vec1", [1.0, 0.0, 0.0])

        # Invalid dimension
        with pytest.raises(ValueError):
            index.add("vec2", [1.0, 0.0])

        with pytest.raises(ValueError):
            index.add("vec3", [1.0, 0.0, 0.0, 0.0])

        # Invalid dimension in build
        with pytest.raises(ValueError):
            index.build(
                [[1.0, 0.0, 0.0], [0.0, 1.0]], ["vec4", "vec5"]  # Wrong dimension
            )

    def test_duplicate_id_handling(self):
        """Test duplicate ID handling."""
        index = LinearSearchCosine[str](dim=3)

        # Add a vector
        index.add("vec1", [1.0, 0.0, 0.0])

        # Try to add a vector with the same ID
        with pytest.raises(DuplicateVectorError):
            index.add("vec1", [0.0, 1.0, 0.0])

        # Add with overwrite=True
        index.add("vec1", [0.0, 1.0, 0.0], overwrite=True)

        # Query to verify the vector was overwritten
        results = index.query([0.0, 1.0, 0.0])
        assert len(results) == 1
        assert results[0][0] == "vec1"
        assert results[0][1] > 0.99  # Close to 1.0

    def test_vectorized_query(self):
        """Test vectorized query implementation."""
        # Create test vectors
        vectors = [
            [1.0, 0.0, 0.0],  # x-axis
            [0.0, 1.0, 0.0],  # y-axis
            [0.0, 0.0, 1.0],  # z-axis
            [0.7, 0.7, 0.0],  # xy-plane
            [0.7, 0.0, 0.7],  # xz-plane
        ]
        ids = ["x", "y", "z", "xy", "xz"]

        # Create index
        index = LinearSearchCosine[str](dim=3)
        index.build(vectors, ids)

        # Query along x-axis
        results = index.query([1.0, 0.0, 0.0])
        assert len(results) == 5
        assert results[0][0] == "x"
        assert results[1][0] in ["xy", "xz"]
        assert results[2][0] in ["xy", "xz"]

        # Query along y-axis
        results = index.query([0.0, 1.0, 0.0])
        assert len(results) == 5
        assert results[0][0] == "y"
        assert results[1][0] == "xy"

        # Query with k=2
        results = index.query([1.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0][0] == "x"



    def test_thread_safety(self):
        """Test thread safety."""
        index = LinearSearchCosine[int](dim=3)
        n_threads, n_ops = 10, 100
        errors = []

        def worker(thread_id):
            try:
                for i in range(n_ops):
                    vec_id = thread_id * n_ops + i
                    vector = [float(thread_id), float(i), 0.0]
                    index.add(vec_id, vector)
                    index.query([float(thread_id), float(i), 0.0], k=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        assert not errors, f"Errors occurred during concurrent execution: {errors}"
        assert index.size() == n_threads * n_ops

    def test_metrics_callback(self):
        """Test metrics callback."""
        metrics = {}
        def observer(op: str, duration_ms: float) -> None:
            metrics.setdefault(op, []).append(duration_ms)

        index = LinearSearchCosine[str](dim=3, observer=observer)
        index.add("vec1", [1.0, 0.0, 0.0])
        index.build([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], ["vec2", "vec3"])
        index.query([1.0, 0.0, 0.0])

        for op in ["add", "build", "query"]:
            assert op in metrics and len(metrics[op]) == 1

    def test_bulk_build_optimization(self):
        """Test bulk build optimization."""
        n_vectors, dim = 1000, 10
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dim).tolist()
        ids = [f"vec{i}" for i in range(n_vectors)]

        # Time individual adds
        start_time = time.time()
        index1 = LinearSearchCosine[str](dim=dim)
        for id_val, vector in zip(ids, vectors):
            index1.add(id_val, vector)
        individual_time = time.time() - start_time

        # Time bulk build
        start_time = time.time()
        index2 = LinearSearchCosine[str](dim=dim)
        index2.build(vectors, ids)
        bulk_time = time.time() - start_time

        # Verify performance and correctness
        assert bulk_time < individual_time
        assert index1.size() == index2.size()
        
        # Check similarity in results
        query = np.random.randn(dim).tolist()
        results1 = index1.query(query, k=10)
        results2 = index2.query(query, k=10)
        top_ids1 = set(id_val for id_val, _ in results1[:5])
        top_ids2 = set(id_val for id_val, _ in results2[:5])
        assert len(top_ids1.intersection(top_ids2)) >= 3
