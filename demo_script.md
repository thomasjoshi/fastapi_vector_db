# FastAPI Vector DB Demo Script

## Setup and Installation (30 seconds)

```bash
# Clone the repository
git clone https://github.com/yourusername/fastapi_vector_db.git
cd fastapi_vector_db

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run the server
uvicorn app.main:app --reload
```

## API Demonstration (2-3 minutes)

Open the Swagger UI at http://localhost:8000/docs

### 1. Create a Library (15 seconds)

```json
POST /libraries/
{
  "metadata": {
    "name": "Science Articles",
    "description": "Collection of scientific articles",
    "created_by": "demo_user"
  }
}
```

### 2. Add a Document to the Library (15 seconds)

```json
POST /libraries/{library_id}/documents/
{
  "metadata": {
    "title": "Introduction to Machine Learning",
    "author": "John Smith",
    "date": "2025-01-15"
  }
}
```

### 3. Add Chunks to the Document (30 seconds)

```json
POST /libraries/{library_id}/documents/{document_id}/chunks
{
  "text": "Machine learning is a subfield of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
  "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
  "metadata": {
    "section": "introduction",
    "page": "1"
  }
}
```

Add 2-3 more chunks with different embeddings.

### 4. Index the Library (15 seconds)

```json
POST /libraries/{library_id}/index
```

### 5. Search for Similar Chunks (30 seconds)

```json
POST /libraries/{library_id}/search
{
  "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
  "k": 2
}
```

### 6. Delete Operations (30 seconds)

Demonstrate delete operations for chunks, documents, and libraries.

## Architectural Design (1-2 minutes)

Explain the key design decisions:

1. **Asynchronous Architecture**
   - Show how all services and endpoints use async/await
   - Explain the performance benefits

2. **Thread-Safety**
   - Demonstrate the reader-writer lock implementation
   - Explain how it prevents data races

3. **Indexing Algorithms**
   - Briefly explain both algorithms
   - Show the time/space complexity differences
   - Explain when to use each one

4. **Domain-Driven Design**
   - Show the separation of concerns between layers
   - Demonstrate how services decouple business logic from API

## Testing (30 seconds)

Run the test suite to demonstrate test coverage:

```bash
poetry run pytest -v
```

Explain how tests verify both happy paths and error conditions.

## Conclusion (15 seconds)

Summarize the key features and benefits of the implementation.
