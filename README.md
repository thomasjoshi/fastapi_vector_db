# FastAPI Vector Database

A vector database implementation using FastAPI, Pydantic, and Python's async capabilities. This project provides a high-performance vector database for similarity search with a clean RESTful API and robust architecture.

## Features

- **Fully Asynchronous Architecture**: Built with FastAPI's async features for optimal performance
- **Domain-Driven Design**: Clean separation between domain models, services, and API layers
- **Multiple Indexing Algorithms**: 
  - LinearSearch: Simple and accurate for smaller datasets
  - BallTreeCosine: Efficient space partitioning for larger datasets
- **Thread-Safe Implementation**: Support for concurrent operations with reader-writer locks
- **Immutable Data Model**: Pydantic V2 with frozen models to prevent side effects
- **Persistence Support**: Optional JSON-based persistence with automatic saving
- **Metadata Filtering**: Filter search results by metadata attributes
- **Comprehensive Testing**: Pytest-asyncio based test suite
- **RESTful API**: Following best practices for status codes and headers
- **Docker Support**: Containerized for easy deployment

## Architecture

The project follows a clean architecture approach with the following layers:

```
app/
├── api/              # API layer with routers and schemas
├── core/             # Core settings and utilities
├── domain/           # Domain models (immutable)
├── indexing/         # Indexing algorithms
├── repos/            # Data access layer
└── services/         # Business logic layer
```

### Key Components

- **Domain Models**: Immutable data structures (Chunk, Document, Library)
- **Repository**: Thread-safe in-memory store with reader-writer lock
- **Services**: Business logic separated into Library, Document, Chunk, and Search services
- **Indexing**: Implementations of vector similarity search algorithms
- **API Layer**: FastAPI routers and Pydantic schemas for validation

## Vector Indexing Algorithms

### LinearSearch

**Time Complexity**:
- Build: O(1) - Just stores vectors in memory
- Search: O(n) - Compares query against all vectors sequentially
- Update: O(1) - Direct insertion/replacement
- Delete: O(n) - Need to rebuild the entire index

**Space Complexity**:
- O(n*d) where n is number of vectors and d is dimension

**Algorithm Details**:
- Cosine similarity computation for each vector in the dataset
- Exact search that guarantees finding the true nearest neighbors
- No preprocessing or special data structures

**Use Case**:
- Small to medium datasets (up to ~10K vectors)
- When accuracy is more important than speed
- When frequent updates are needed
- Development and testing environments

### BallTreeCosine

**Time Complexity**:
- Build: O(n log n) - Constructs tree through recursive partitioning
- Search: O(log n) average case, O(n) worst case - Traverses tree with pruning
- Update: O(n log n) - Requires partial or complete rebuild
- Delete: O(n log n) - Requires rebuilding affected branches

**Space Complexity**:
- O(n*d) - Slightly more than linear due to tree structure overhead

**Algorithm Details**:
- Hierarchical tree structure for spatial partitioning
- Uses ball (hypersphere) nodes with centroids and radii
- Cosine distance is used for consistency with embedding similarity
- Pruning strategy eliminates entire subtrees from search

**Use Case**:
- Larger datasets (10K+ vectors)
- When query speed is important for large collections
- When updates are less frequent
- Production environments with stable data

## API Documentation

### Libraries

- `POST /libraries/` - Create a new library
- `GET /libraries/{library_id}` - Get a library by ID
- `PUT /libraries/{library_id}` - Update a library
- `DELETE /libraries/{library_id}` - Delete a library

### Documents

- `POST /libraries/{library_id}/documents/` - Add a document to a library
- `GET /libraries/{library_id}/documents/{document_id}` - Get a document
- `PUT /libraries/{library_id}/documents/{document_id}` - Update a document
- `DELETE /libraries/{library_id}/documents/{document_id}` - Delete a document

### Chunks

- `POST /libraries/{library_id}/documents/{document_id}/chunks` - Add a chunk
- `GET /libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}` - Get a chunk
- `PUT /libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}` - Update a chunk
- `DELETE /libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}` - Delete a chunk

### Search

- `POST /libraries/{library_id}/index` - Index a library for vector search
- `POST /libraries/{library_id}/search` - Search for similar chunks with optional metadata filtering

## Installation and Usage

### Prerequisites

- Python 3.11+
- Poetry (for development)

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/fastapi_vector_db.git
cd fastapi_vector_db

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell

# Run the server
uvicorn app.main:app --reload
```

Access the API documentation at http://localhost:8000/docs

## Design Decisions and Trade-offs

### Async Implementation

The application uses async/await patterns throughout to maximize throughput for I/O-bound operations. This approach allows the server to handle more concurrent requests, especially during search operations which would otherwise block the event loop.

### Thread Safety

Even with async code, we implemented a reader-writer lock pattern to handle concurrent access to shared data. This ensures that multiple readers can access data simultaneously while writers get exclusive access, preventing data corruption and race conditions.

### Immutable Domain Models

All domain models are immutable (frozen=True) to prevent accidental state changes. This makes concurrent code safer and easier to reason about. When updates are needed, we create new instances rather than modifying existing ones.

### Service Layer

The service layer decouples business logic from the API and repository implementations. This makes the code more testable and maintainable, and allows for easier substitution of components (like switching from in-memory to persistent storage).

### Error Handling

Custom exception types (NotFoundError, ValidationError) provide context-specific error information. Global exception handlers translate these into appropriate HTTP responses, ensuring consistent error handling throughout the API.

## Implementation Details

### Persistence Layer

The Vector DB supports optional persistence through a JSON-based file storage mechanism:

- **Auto-save**: Configurable periodic saving of the database state
- **Atomic Writes**: Uses temporary files and atomic rename to prevent data corruption
- **Environment Configuration**: Easily enable/disable via environment variables
- **Docker Volume Mounting**: Persistence across container restarts

### Metadata Filtering

Search queries support metadata filtering to narrow down results:

- **Key-Value Matching**: Filter chunks by exact metadata field matches
- **Combined with Vector Search**: First find similar vectors, then filter by metadata
- **Efficient Implementation**: Uses dictionary-based lookup for performance

## Future Improvements

- **Alternative Persistence**: Add support for more scalable storage options (SQL, Redis)
- **Advanced Metadata Filtering**: Support regex, ranges, and fuzzy matching
- **Bulk Operations**: Add batch processing for documents and chunks
- **Caching**: Implement caching for frequent queries
- **Metrics and Monitoring**: Add comprehensive metrics collection
- **Distributed Architecture**: Support for leader-follower pattern

## License

MIT
