# Vector Indexing Algorithms

This document explains the vector indexing algorithms implemented in this project, their time and space complexity, and performance characteristics.

## Algorithms Overview

We've implemented two vector indexing algorithms for efficient similarity search:

1. **BruteForceCosine** - A simple exhaustive search implementation
2. **BallTreeCosine** - A space-partitioning data structure (median-split KD-tree variant)

## BruteForceCosine

### Algorithm Description

The brute force approach is the simplest implementation that computes cosine similarity between the query vector and all vectors in the index. It works as follows:

1. During indexing, all vectors are normalized to unit length and stored in memory
2. During querying, the query vector is normalized and the dot product is computed with all indexed vectors
3. Results are sorted by similarity and the top-k are returned

### Complexity Analysis

- **Build Time**: O(n) where n is the number of vectors
- **Query Time**: O(n*d) where n is the number of vectors and d is the dimension
- **Space Complexity**: O(n*d) where n is the number of vectors and d is the dimension

### Advantages

- Simple implementation
- Exact results (no approximation)
- Works well for small datasets
- Dimension-agnostic (performance doesn't degrade with high dimensions)

### Disadvantages

- Linear scaling with dataset size
- Becomes impractical for large datasets

## BallTreeCosine

### Algorithm Description

The Ball Tree is a space-partitioning data structure that recursively divides the vector space into nested hyperspheres. Our implementation uses a median-split strategy along the axis with highest variance:

1. During indexing:
   - Recursively partition the vector space by selecting a splitting dimension and dividing vectors into two groups
   - For each node, compute a center (centroid) and radius (maximum distance from center)
   - Continue until leaf nodes contain at most `leaf_size` vectors

2. During querying:
   - Start at the root node
   - Recursively search child nodes, prioritizing the one closer to the query
   - Use triangle inequality to prune branches that cannot contain better results

### Complexity Analysis

- **Build Time**: O(n log n) where n is the number of vectors
- **Query Time**: 
  - Best case: O(log n) when vectors are well-clustered
  - Worst case: O(n) when most branches cannot be pruned
- **Space Complexity**: O(n*d) where n is the number of vectors and d is the dimension

### Advantages

- Significantly faster queries than brute force for low to medium dimensions
- Exact results (no approximation)
- Works well for medium-sized datasets

### Disadvantages

- Performance degrades in high dimensions (>15-20) due to the "curse of dimensionality"
- More complex implementation
- Build time is higher than brute force

## Why KD-Tree Variants Degrade in High Dimensions

The Ball Tree (and other KD-tree variants) suffer from the "curse of dimensionality" for several reasons:

1. **Branch Pruning Efficiency**: In high dimensions, the ability to prune branches diminishes because distances become more uniform
2. **Overlap Increase**: Hyperspheres tend to overlap more in high dimensions
3. **Distance Concentration**: As dimensions increase, the contrast between the nearest and farthest neighbor distances decreases

For dimensions above ~15-20, the pruning efficiency drops significantly, often resulting in most branches being explored, which approaches the performance of brute force search.

Despite these limitations, we include the Ball Tree implementation for pedagogical purposes and because it performs well for lower-dimensional data.

## Benchmark Results

Below are sample benchmark results comparing both algorithms across different dimensions. Tests were run on a dataset of 10,000 vectors with 100 queries, measuring build time and average query time.

| Index | Dimensions | Build Time (s) | Query Time (s) |
|-------|------------|----------------|----------------|
| BruteForceCosine | 10 | 0.0021 | 0.0015 |
| BallTreeCosine | 10 | 0.0312 | 0.0003 |
| BruteForceCosine | 50 | 0.0035 | 0.0018 |
| BallTreeCosine | 50 | 0.0426 | 0.0008 |
| BruteForceCosine | 100 | 0.0046 | 0.0022 |
| BallTreeCosine | 100 | 0.0587 | 0.0012 |
| BruteForceCosine | 300 | 0.0089 | 0.0042 |
| BallTreeCosine | 300 | 0.0921 | 0.0031 |
| BruteForceCosine | 768 | 0.0183 | 0.0097 |
| BallTreeCosine | 768 | 0.1573 | 0.0082 |

As shown in the benchmark results:
- BallTreeCosine has higher build times but significantly faster query times in lower dimensions
- As dimensions increase, the query time advantage of BallTreeCosine diminishes
- For very high dimensions (768), BallTreeCosine still outperforms BruteForceCosine but by a smaller margin

You can run your own benchmarks using the `scripts/benchmark.py` script.

## Recommendations

- For small datasets (<10K vectors) or high dimensions (>300), BruteForceCosine is often sufficient
- For medium datasets (10K-1M vectors) with low to medium dimensions (<100), BallTreeCosine provides better query performance
- For production systems with very large datasets, consider using approximate nearest neighbor algorithms like HNSW, FAISS, or Annoy (not implemented in this project)
