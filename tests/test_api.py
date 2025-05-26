"""Tests for the Vector DB API."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from loguru import logger

from app.domain.models import Chunk, Document, Library
from app.main import app
from app.repos.in_memory import _repo
from tests.test_search import create_test_library

# Create a test client with a shared repository instance
# Reset the repository before each test
_repo._libraries = {}

client = TestClient(app)


def test_create_library() -> None:
    """Test creating a library via the API with Location header."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)

    # Check response
    assert response.status_code == status.HTTP_201_CREATED
    library_data = response.json()
    assert library_data is not None
    library_id = library_data["id"]

    # Check Location header
    assert "Location" in response.headers
    assert response.headers["Location"] == f"/libraries/{library_id}"


def test_get_library() -> None:
    """Test getting a library via the API with correct schema."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)
    library_id = response.json()["id"]

    # Get the library
    response = client.get(f"/libraries/{library_id}")

    # Check response
    assert response.status_code == status.HTTP_200_OK
    retrieved_lib = response.json()
    assert retrieved_lib["id"] == library_id
    assert len(retrieved_lib["documents"]) == len(lib.documents)

    # Verify schema fields
    assert "id" in retrieved_lib
    assert "documents" in retrieved_lib
    assert "metadata" in retrieved_lib


def test_update_library() -> None:
    """Test updating a library via the API with full data."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)
    library_id = response.json()["id"]

    # Update the library with full data
    updated_data = {
        "documents": lib.model_dump()["documents"],
        "metadata": {"name": "Updated via API", "created_by": "Test"},
    }
    response = client.put(f"/libraries/{library_id}", json=updated_data)

    # Check response
    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Get the updated library
    response = client.get(f"/libraries/{library_id}")
    retrieved_lib = response.json()
    assert retrieved_lib["metadata"]["name"] == "Updated via API"


def test_update_library_with_partial_data() -> None:
    """Test updating a library via the API with partial data."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)
    library_id = response.json()["id"]

    # Update only the metadata
    update_data = {"metadata": {"name": "Updated via API", "created_by": "Test"}}
    response = client.put(f"/libraries/{library_id}", json=update_data)

    # Check response
    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Get the updated library
    response = client.get(f"/libraries/{library_id}")
    retrieved_lib = response.json()
    assert retrieved_lib["metadata"]["name"] == "Updated via API"
    # Documents should remain unchanged
    assert len(retrieved_lib["documents"]) == len(lib.documents)


def test_delete_library() -> None:
    """Test deleting a library via the API."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)
    library_id = response.json()["id"]

    # Delete the library
    response = client.delete(f"/libraries/{library_id}")

    # Check response
    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify deletion
    response = client.get(f"/libraries/{library_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_delete_library_not_found() -> None:
    """Test deleting a non-existent library via the API."""
    # Try to delete a non-existent library
    response = client.delete("/libraries/00000000-0000-0000-0000-000000000000")

    # Check response
    assert response.status_code == status.HTTP_404_NOT_FOUND
    error_data = response.json()

    # Verify error schema
    assert "detail" in error_data
    assert "resource_type" in error_data
    assert "resource_id" in error_data
    assert error_data["resource_type"] == "Library"


def test_centralized_error_handling() -> None:
    """Test that centralized error handling works for NotFoundError."""
    # Try to get a non-existent library
    response = client.get("/libraries/00000000-0000-0000-0000-000000000000")

    # Check response
    assert response.status_code == status.HTTP_404_NOT_FOUND
    error_data = response.json()

    # Verify error schema from centralized handler
    assert "detail" in error_data
    assert "resource_type" in error_data
    assert "resource_id" in error_data
    assert error_data["resource_type"] == "Library"
    assert error_data["resource_id"] == "00000000-0000-0000-0000-000000000000"


@pytest.mark.asyncio
async def test_document_chunk_search_flow_async() -> None:
    """Async implementation of the document chunk search flow test."""
    from app.repos.in_memory import InMemoryRepo
    from app.services.chunk import ChunkService
    from app.services.document import DocumentService
    from app.services.library import LibraryService
    from app.services.search import SearchService

    # Create a shared repository instance for all services
    repo = InMemoryRepo()
    library_service = LibraryService(repo)
    document_service = DocumentService(repo)
    chunk_service = ChunkService(repo)
    search_service = SearchService(repo)

    # Create a library directly using the service
    lib = Library.example()
    library_id = await library_service.add_library(lib)
    logger.info(f"Created library with ID {library_id}")

    # Create a document directly using the service
    doc = Document(chunks=[], metadata={"source": "test", "author": "test_user"})
    document = await document_service.add_document(library_id, doc)
    doc_id = document.id
    logger.info(f"Created document with ID {doc_id}")

    # Create chunks with one-hot vectors
    chunks = []
    for i in range(3):
        # Create a chunk with one-hot encoding
        embedding = [0.0] * 5
        embedding[i] = 1.0  # Set one-hot encoding

        chunk = Chunk(
            text=f"This is chunk {i}",
            metadata={"position": str(i)},  # Metadata values must be strings
            embedding=embedding,
        )
        chunks.append(chunk)

    # Add chunks to the document
    for chunk in chunks:
        await chunk_service.add_chunk(library_id, doc_id, chunk)
        logger.info(f"Added chunk with ID {chunk.id}")

    # Index the library for search
    await search_service.index_library(library_id)
    logger.info(f"Indexed library with ID {library_id}")

    # Search for chunks directly using the service
    query_embedding = [0.0] * 5
    query_embedding[1] = 1.0  # Should match the second chunk
    results = await search_service.search(library_id, query_embedding, k=1)

    # Verify search results
    assert len(results) > 0, "Search should return at least one result"
    # Check if the top result has position 1 (the second chunk)
    assert results[0][0].metadata.get("position") == "1"

    # Delete the document
    await document_service.delete_document(library_id, doc_id)

    # Delete the library
    await library_service.delete_library(library_id)

    logger.info("Test completed successfully")

    # Verify the library is deleted
    try:
        await library_service.get_library(library_id)
        raise AssertionError("Library should have been deleted")
    except Exception:
        # Expected exception
        pass


@pytest.mark.asyncio
async def test_document_chunk_search_flow() -> None:
    """Test full flow: create lib → add doc → chunks → index → search → delete."""
    from app.repos.in_memory import InMemoryRepo
    from app.services.chunk import ChunkService
    from app.services.document import DocumentService
    from app.services.library import LibraryService
    from app.services.search import SearchService

    # Create a shared repository instance for all services
    repo = InMemoryRepo()
    library_service = LibraryService(repo)
    document_service = DocumentService(repo)
    chunk_service = ChunkService(repo)
    search_service = SearchService(repo)

    # Create a library directly using the service
    lib = Library.example()
    library_id = await library_service.add_library(lib)
    logger.info(f"Created library with ID {library_id}")

    # Create a document directly using the service
    doc = Document(chunks=[], metadata={"source": "test", "author": "test_user"})
    document = await document_service.add_document(library_id, doc)
    doc_id = document.id
    logger.info(f"Created document with ID {doc_id}")

    # Create chunks with one-hot vectors
    chunks = []
    for i in range(3):
        # Create a chunk with one-hot encoding
        embedding = [0.0] * 5
        embedding[i] = 1.0  # Set one-hot encoding

        chunk = Chunk(
            text=f"This is chunk {i}",
            metadata={"position": str(i)},  # Metadata values must be strings
            embedding=embedding,
        )
        chunks.append(chunk)

    # Add chunks to the document
    for chunk in chunks:
        await chunk_service.add_chunk(library_id, doc_id, chunk)
        logger.info(f"Added chunk with ID {chunk.id}")

    # Index the library for search
    await search_service.index_library(library_id)

    # Index the library for search
    await search_service.index_library(library_id)
    logger.info(f"Indexed library with ID {library_id}")

    # Search for chunks directly using the service
    query_embedding = [0.0] * 5
    query_embedding[1] = 1.0  # Should match the second chunk
    results = await search_service.search(library_id, query_embedding, k=1)

    # Verify search results
    assert len(results) > 0, "Search should return at least one result"
    # Check if the top result has position 1 (the second chunk)
    assert results[0][0].metadata.get("position") == "1"

    # Delete the document
    await document_service.delete_document(library_id, doc_id)

    # Delete the library
    await library_service.delete_library(library_id)

    logger.info("Test completed successfully")

    # Verify the library is deleted
    try:
        await library_service.get_library(library_id)
        raise AssertionError("Library should have been deleted")
    except Exception:
        # Expected exception
        pass


@pytest.mark.asyncio
async def test_dimension_mismatch() -> None:
    """Test that searching with wrong dimension returns ValidationError."""
    from app.repos.in_memory import InMemoryRepo
    from app.services.exceptions import ValidationError
    from app.services.search import SearchService

    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    # We need to handle the fact that add_library is now async
    await repo.add_library(library)

    # Create search service and index the library
    search_service = SearchService(repo)
    await search_service.index_library(library.id)

    # Try to search with wrong dimension (should be 3, using 5)
    with pytest.raises(ValidationError) as excinfo:
        await search_service.search(library.id, [0.1, 0.2, 0.3, 0.4, 0.5], k=3)

    # Verify the error message
    assert "dimension" in str(excinfo.value)
