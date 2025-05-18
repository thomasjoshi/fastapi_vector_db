#!/usr/bin/env python
"""
Benchmark script for vector search implementations.

This script compares the performance of different vector search implementations
in terms of build time, query time, and memory usage.
"""

import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from uuid import UUID, uuid4
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.indexing.brute import BruteForceCosine
from app.indexing.ball_tree import BallTreeCosine


def generate_random_vectors(
    count: int, dimensions: int, seed: int = 42
) -> List[List[float]]:
    """Generate random unit vectors."""
    np.random.seed(seed)
    vectors = []
    for _ in range(count):
        # Generate random vector
        vec = np.random.randn(dimensions).astype(np.float32)
        # Normalize to unit length
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec.tolist())
    return vectors


def benchmark_index(
    name: str,
    index_class: Any,
    vectors: List[List[float]],
    query_vectors: List[List[float]],
    k: int = 10,
) -> Dict[str, Any]:
    """Benchmark a vector index implementation."""
    print(f"Benchmarking {name}...")
    
    # Generate IDs
    ids = [uuid4() for _ in range(len(vectors))]
    
    # Measure build time
    start_time = time.time()
    index = index_class()
    index.build(vectors, ids)
    build_time = time.time() - start_time
    print(f"  Build time: {build_time:.4f} seconds")
    
    # Measure query time
    query_times = []
    for query in query_vectors:
        start_time = time.time()
        results = index.query(query, k=k)
        query_time = time.time() - start_time
        query_times.append(query_time)
    
    avg_query_time = sum(query_times) / len(query_times)
    print(f"  Average query time: {avg_query_time:.6f} seconds")
    
    return {
        "name": name,
        "build_time": build_time,
        "avg_query_time": avg_query_time,
    }


def run_benchmarks() -> None:
    """Run benchmarks for different index implementations and vector dimensions."""
    dimensions_list = [10, 50, 100, 300, 768]
    vector_count = 10000
    query_count = 100
    k = 10
    
    results = []
    
    for dimensions in dimensions_list:
        print(f"\nBenchmarking with {dimensions} dimensions:")
        # Generate vectors
        vectors = generate_random_vectors(vector_count, dimensions)
        query_vectors = generate_random_vectors(query_count, dimensions, seed=43)
        
        # Benchmark BruteForceCosine
        brute_results = benchmark_index(
            f"BruteForceCosine ({dimensions}D)",
            BruteForceCosine,
            vectors,
            query_vectors,
            k,
        )
        results.append(brute_results)
        
        # Benchmark BallTreeCosine
        ball_results = benchmark_index(
            f"BallTreeCosine ({dimensions}D)",
            BallTreeCosine,
            vectors,
            query_vectors,
            k,
        )
        results.append(ball_results)
    
    # Print summary
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Index':<30} {'Build Time (s)':<15} {'Query Time (s)':<15}")
    print("-" * 80)
    for result in results:
        print(
            f"{result['name']:<30} {result['build_time']:<15.4f} {result['avg_query_time']:<15.6f}"
        )


if __name__ == "__main__":
    run_benchmarks()
