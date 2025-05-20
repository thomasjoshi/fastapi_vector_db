# FastAPI Vector Database

High-performance vector database for similarity search with FastAPI, async Python, and clean architecture.

## Features

- **Async Architecture**: Non-blocking I/O with FastAPI
- **Multiple Indexing Algorithms**: LinearSearch (small datasets) and BallTreeCosine (large datasets)
- **Thread-Safe**: Concurrent operations with reader-writer locks
- **Cohere API Integration**: Automatic embedding generation from text
- **Persistence**: JSON-based storage with auto-saving
- **Metadata Filtering**: Filter search results by attributes
- **RESTful API**: Clean endpoints with proper status codes
- **Docker Ready**: Simple deployment with containers

## Architecture

```
app/
├── api/              # API routers and schemas
├── core/             # Settings and utilities
├── domain/           # Immutable models
├── indexing/         # Vector algorithms
├── repos/            # Data access
└── services/         # Business logic
```

**Key Components**: Domain models, repositories with locks, separated services, and API validation.

## Vector Indexing Algorithms

### LinearSearch
- **Complexity**: Build O(1), Search O(n), Update O(1), Delete O(n)
- **Space**: O(n*d) where n=vectors, d=dimensions
- **Best for**: Small datasets (<10K), frequent updates, accuracy over speed

### BallTreeCosine
- **Complexity**: Build O(n log n), Search O(log n) avg, Update/Delete O(n log n)
- **Space**: O(n*d) with minimal overhead
- **Best for**: Larger datasets (>10K), speed-critical applications, stable data

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
git clone https://github.com/yourusername/fastapi_vector_db.git
cd fastapi_vector_db
poetry install
poetry shell

# Run server
uvicorn app.main:app --reload
```

API docs at http://localhost:8000/docs

## Key Design Patterns

- **Async Implementation**: Non-blocking I/O for high concurrency
- **Thread Safety**: Reader-writer locks for data consistency
- **Immutable Models**: Frozen Pydantic models prevent side effects
- **Service Layer**: Decoupled business logic for testability
- **Custom Exceptions**: Type-specific error handling

## Implementation Features

### Persistence
- JSON-based storage with configurable auto-save
- Atomic writes to prevent data corruption
- Environment variable configuration

### Metadata Filtering
- Key-value filtering on search results
- Post-vector search filtering for efficiency

## Cohere API Integration

- Default model: `embed-english-v3.0` (configurable)
- Non-blocking API calls with error handling

```json
POST /libraries/{library_id}/documents/{document_id}/chunks/embed
{
  "text": "Your content here",
  "metadata": { "category": "science" },
  "generate_embedding": true
}
```

## Future Improvements

- Alternative storage backends (SQL, Redis)
- Advanced metadata filtering (regex, ranges)
- Bulk operations and caching
- Distributed architecture support

## License

MIT
