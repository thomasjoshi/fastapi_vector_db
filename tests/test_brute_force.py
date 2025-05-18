"""
Tests for the BruteForceCosine vector index.
"""

import threading
import time

import numpy as np
import pytest

from app.indexing.brute import BruteForceCosine, DuplicateVectorError


class TestBruteForceCosine:
    """Test suite for BruteForceCosine."""

    def test_init(self):
        """Test initialization with dimension."""
        index = BruteForceCosine[str](dim=3)
        assert index.size() == 0

    def test_dimension_validation(self):
        """Test dimension validation."""
        index = BruteForceCosine[str](dim=3)

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
        index = BruteForceCosine[str](dim=3)

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
        index = BruteForceCosine[str](dim=3)
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

    def test_persistence(self):
        """Test serialization and deserialization."""
        # Create test vectors
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        ids = ["x", "y", "z"]

        # Create and populate index
        index = BruteForceCosine[str](dim=3)
        index.build(vectors, ids)

        # Serialize
        data = index.to_bytes()

        # Deserialize
        loaded_index = BruteForceCosine.from_bytes(data)

        # Verify dimensions
        assert loaded_index._dim == 3
        assert loaded_index.size() == 3

        # Verify query results
        original_results = index.query([1.0, 0.0, 0.0])
        loaded_results = loaded_index.query([1.0, 0.0, 0.0])

        assert len(original_results) == len(loaded_results)
        for (id1, sim1), (id2, sim2) in zip(original_results, loaded_results):
            assert id1 == id2
            assert abs(sim1 - sim2) < 1e-6

    def test_thread_safety(self):
        """Test thread safety."""
        index = BruteForceCosine[int](dim=3)

        # Number of threads and operations
        n_threads = 10
        n_ops = 100

        # Shared state for tracking errors
        errors = []

        def worker(thread_id):
            try:
                for i in range(n_ops):
                    # Add a vector
                    vec_id = thread_id * n_ops + i
                    vector = [float(thread_id), float(i), 0.0]
                    index.add(vec_id, vector)

                    # Query the index
                    index.query([float(thread_id), float(i), 0.0], k=5)
            except Exception as e:
                errors.append(e)

        # Create and start threads
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Check for errors
        assert not errors, f"Errors occurred during concurrent execution: {errors}"

        # Verify the index size
        assert index.size() == n_threads * n_ops

    def test_metrics_callback(self):
        """Test metrics callback."""
        # Create a callback to collect metrics
        metrics = {}

        def observer(op: str, duration_ms: float) -> None:
            if op not in metrics:
                metrics[op] = []
            metrics[op].append(duration_ms)

        # Create index with observer
        index = BruteForceCosine[str](dim=3, observer=observer)

        # Perform operations
        index.add("vec1", [1.0, 0.0, 0.0])
        index.build([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], ["vec2", "vec3"])
        index.query([1.0, 0.0, 0.0])

        # Verify metrics were collected
        assert "add" in metrics
        assert "build" in metrics
        assert "query" in metrics
        assert len(metrics["add"]) == 1
        assert len(metrics["build"]) == 1
        assert len(metrics["query"]) == 1

    def test_bulk_build_optimization(self):
        """Test bulk build optimization."""
        # Create a large number of vectors
        n_vectors = 1000
        dim = 10

        # Generate random vectors
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dim).tolist()
        ids = [f"vec{i}" for i in range(n_vectors)]

        # Measure time for individual adds
        start_time = time.time()
        index1 = BruteForceCosine[str](dim=dim)
        for id_val, vector in zip(ids, vectors):
            index1.add(id_val, vector)
        individual_time = time.time() - start_time

        # Measure time for bulk build
        start_time = time.time()
        index2 = BruteForceCosine[str](dim=dim)
        index2.build(vectors, ids)
        bulk_time = time.time() - start_time

        # Verify bulk build is faster
        assert bulk_time < individual_time

        # Verify both indices have the same content
        assert index1.size() == index2.size()

        # Query both indices and verify similar results
        query = np.random.randn(dim).tolist()
        results1 = index1.query(query, k=10)
        results2 = index2.query(query, k=10)

        # Check that the top results are similar
        # They might not be identical due to floating point differences
        top_ids1 = set(id_val for id_val, _ in results1[:5])
        top_ids2 = set(id_val for id_val, _ in results2[:5])
        assert len(top_ids1.intersection(top_ids2)) >= 3  # At least 3 in common
