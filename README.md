# FastAPI Vector Database

A high-performance vector database implementation for semantic similarity search built with FastAPI, async Python, and domain-driven design principles. This project demonstrates how to build efficient vector indexing algorithms from scratch while maintaining production-quality architecture patterns.

## Core Features

- **Async Architecture**: Non-blocking I/O with FastAPI for high throughput and scalability
- **Custom Vector Algorithms**: Implemented LinearSearchCosine and BallTreeCosine from scratch with documented complexity analysis
- **Thread Safety**: Reentrant locks at multiple levels to prevent data races during concurrent operations
- **Cohere API Integration**: Automatic generation of 1024-dimensional embeddings from text
- **Persistence Layer**: Configurable JSON-based storage with atomic writes and automated checkpoint intervals
- **Metadata Filtering**: Post-retrieval filtering for combining semantic and attribute-based queries
- **Domain-Driven Design**: Clear boundaries between API, domain models, services, and repositories
- **Immutable Domain Models**: Frozen Pydantic models to prevent accidental state mutations
- **Docker Containerization**: Production-ready deployment with minimal configuration

## Architecture

This project implements an architecture pattern with domain-driven design principles, creating a maintainable and testable codebase:

```
app/
├── api/              # API layer: Routers, request/response schemas, and dependencies
├── core/             # Core config: Settings, logging, and application-wide utilities
├── domain/           # Domain layer: Immutable entity models with business rules
├── indexing/         # Vector algorithms: Custom implementations with complexity guarantees
├── repos/            # Repository layer: Thread-safe data access with atomic operations
└── services/         # Service layer: Business logic and orchestration between layers
```

This separation allows each layer to evolve independently, with clear contracts between them. Domain models represent the core entities, services implement business processes, repositories manage persistence, and the API layer handles HTTP communication.

## Vector Indexing Algorithms

This section explains the vector indexing algorithms implemented in this project, their time and space complexity, and performance characteristics.

### Algorithms Overview

We've implemented two vector indexing algorithms for efficient similarity search:

1. **LinearSearchCosine** - A simple exhaustive search implementation
2. **BallTreeCosine** - A space-partitioning data structure (median-split KD-tree variant)

### LinearSearchCosine

#### Algorithm Description

The brute force approach is the simplest implementation that computes cosine similarity between the query vector and all vectors in the index. It works as follows:

1. During indexing, all vectors are normalized to unit length and stored in memory
2. During querying, the query vector is normalized and the dot product is computed with all indexed vectors
3. Results are sorted by similarity and the top-k are returned

#### Complexity Analysis

- **Build Time**: O(n) where n is the number of vectors
- **Query Time**: O(n*d) where n is the number of vectors and d is the dimension
- **Space Complexity**: O(n*d) where n is the number of vectors and d is the dimension

#### Advantages

- Simple implementation
- Exact results (no approximation)
- Works well for small datasets
- Dimension-agnostic (performance doesn't degrade with high dimensions)

#### Disadvantages

- Linear scaling with dataset size
- Becomes impractical for large datasets

### BallTreeCosine

#### Algorithm Description

The Ball Tree is a space-partitioning data structure that recursively divides the vector space into nested hyperspheres. Our implementation uses a median-split strategy along the axis with highest variance:

1. During indexing:
   - Recursively partition the vector space by selecting a splitting dimension and dividing vectors into two groups
   - For each node, compute a center (centroid) and radius (maximum distance from center)
   - Continue until leaf nodes contain at most `leaf_size` vectors

2. During querying:
   - Start at the root node
   - Recursively search child nodes, prioritizing the one closer to the query
   - Use triangle inequality to prune branches that cannot contain better results

#### Complexity Analysis

- **Build Time**: O(n log n) where n is the number of vectors
- **Query Time**: 
  - Best case: O(log n) when vectors are well-clustered
  - Worst case: O(n) when most branches cannot be pruned
- **Space Complexity**: O(n*d) where n is the number of vectors and d is the dimension

#### Advantages

- Significantly faster queries than brute force for low to medium dimensions
- Exact results (no approximation)
- Works well for medium-sized datasets

#### Disadvantages

- Theoretically susceptible to the "curse of dimensionality" as dimensions increase
- More complex implementation than linear search
- Higher build time complexity than linear search

### Theoretical Considerations for High-Dimensional Data

In computational geometry and machine learning literature, space-partitioning data structures like Ball Trees may face challenges with very high-dimensional data due to what's known as the "curse of dimensionality":

1. **Branch Pruning**: As dimensionality increases, more branches of the tree may need to be explored during search operations.

2. **Partition Efficiency**: The effectiveness of spatial partitioning can theoretically decrease in higher dimensions.

3. **Distance Metrics**: Distance calculations become more computationally intensive with more dimensions.

For applications using high-dimensional embeddings, performance testing would be recommended to determine the most efficient algorithm for your specific use case.

### Performance Evaluation

When choosing between LinearSearchCosine and BallTreeCosine for your application, consider these theoretical performance characteristics:

- **LinearSearchCosine**: Provides consistent performance across any number of dimensions with predictable scaling.

- **BallTreeCosine**: May offer improved query performance for certain datasets, particularly those with natural clustering properties.


### Algorithm Selection Guidelines

- **For smaller datasets**: LinearSearchCosine offers implementation simplicity and predictable performance
- **For potentially larger datasets**: BallTreeCosine may provide query optimization opportunities
- **For production-scale systems**: Consider implementing approximate nearest neighbor algorithms like HNSW, FAISS, or Annoy in future iterations

## API Endpoints

### Libraries
- `POST /libraries/` - Create library
- `GET/PUT/DELETE /libraries/{library_id}` - Get, update, delete library

### Documents
- `POST /libraries/{library_id}/documents/` - Add document
- `GET/PUT/DELETE /libraries/{library_id}/documents/{document_id}` - Get, update, delete document

### Chunks
- `POST /libraries/{library_id}/documents/{document_id}/chunks` - Add chunk
- `POST /libraries/{library_id}/documents/{document_id}/chunks/embed` - Add with auto-embedding
- `GET/PUT/DELETE /libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}`

### Search
- `POST /libraries/{library_id}/index` - Index library
- `POST /libraries/{library_id}/search` - Search with metadata filtering

## Setup

```bash
# Clone and install
git clone https://github.com/thomasjoshi/fastapi_vector_db.git
cd fastapi_vector_db
poetry install
poetry env activate

# Run server
uvicorn app.main:app --reload
```

## Key Design Patterns & Engineering Decisions

- **Async-First Architecture**: Designed from the ground up with `async/await` patterns to maximize throughput by preventing thread blocking during I/O operations

- **Multi-Level Thread Safety**: Implemented reentrant locks (RLock) at both the repository and index levels, allowing concurrent reads while ensuring write operations are atomic

- **Immutable Domain Models**: Used frozen Pydantic models to prevent accidental state mutation, eliminating an entire class of potential bugs

- **Dependency Injection**: Services and repositories are injected at runtime, enabling easier unit testing and maintaining the Dependency Inversion Principle

- **Custom Exceptions**: Implemented basic error types like NotFoundError and ValidationError with appropriate HTTP status code mapping at the API layer

- **Performance Callbacks**: Added simple timing callbacks to index operations, allowing basic performance monitoring

## Advanced Implementation Features

### Thread-Safe Persistence
- **Atomic Serialization**: Uses atomic file operations to prevent data corruption during persistence operations
- **Configurable Checkpointing**: Auto-save intervals and file paths configurable through environment variables
- **Efficient Serialization**: Custom to/from_bytes methods that maintain the full structure of indices, including tree topologies

### Metadata Filtering
- **Post-Retrieval Filtering**: Simple two-phase approach that performs vector similarity search first, then filters results by metadata
- **Key-Value Matching**: Support for exact matching on chunk metadata fields
- **Efficient Design**: Filtering happens after vector search to maintain query performance


```json
# Example: Create a chunk with automatic embedding generation
POST /libraries/{library_id}/documents/{document_id}/chunks/embed
{
  "text": "Vector search enables efficient similarity-based retrieval of data.",
  "metadata": { "category": "technology", "importance": "high" },
  "generate_embedding": true
}
```

## Future Improvements

The architecture provides a solid foundation for extending functionality:

- **Alternative Storage Backends**: The repository pattern makes it straightforward to implement SQL, Redis, or other persistence mechanisms

- **Advanced Metadata Queries**: The metadata filtering infrastructure could be extended to support regex matching, numeric ranges, and geospatial queries

- **Approximate Nearest Neighbor Algorithms**: Implementing HNSW or LSH algorithms would provide better scaling for very large datasets

- **Vector Compression**: Dimensionality reduction and quantization techniques could reduce memory footprint while maintaining search quality

- **Distributed Architecture**: The stateless service design facilitates horizontal scaling across multiple nodes with shared storage

## License

MIT
