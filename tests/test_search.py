"""
Tests for search functionality.
"""

import asyncio
from uuid import UUID, uuid4
from typing import List, Tuple

import pytest

from app.domain.models import Chunk, Document, Library
from app.indexing.ball_tree import BallTreeCosine
from app.indexing.linear_search import LinearSearchCosine
from app.repos.in_memory import InMemoryRepo
from app.services.search import SearchService
from app.services.exceptions import ValidationError

def create_test_library() -> Library:
    """Create a test library with documents and chunks."""
    doc_id = uuid4()  # Generate document ID once
    # Create chunks with different embeddings
    chunk1 = Chunk(
        id=uuid4(),
        document_id=doc_id, # Use the generated doc_id
        text="This is the first chunk",
        embedding=[1.0, 0.0, 0.0],  # pointing along x-axis
        metadata={"page": "1"},
    )

    chunk2 = Chunk(
        id=uuid4(),
        document_id=doc_id, # Use the generated doc_id
        text="This is the second chunk",
        embedding=[0.0, 1.0, 0.0],  # pointing along y-axis
        metadata={"page": "2"},
    )

    chunk3 = Chunk(
        id=uuid4(),
        document_id=doc_id, # Use the generated doc_id
        text="This is the third chunk",
        embedding=[0.0, 0.0, 1.0],  # pointing along z-axis
        metadata={"page": "3"},
    )

    # Create a document with these chunks
    document = Document(
        id=doc_id, # Use the generated doc_id
        chunks=[chunk1, chunk2, chunk3],
        metadata={"title": "Test Document"},
    )

    # Create a library with this document
    library = Library(
        id=uuid4(),
        documents=[document],
        metadata={"name": "Test Library"},
    )

    return library


@pytest.mark.asyncio
async def test_brute_force_search() -> None:
    """Test LinearSearchCosine search."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)

    # Create search service with LinearSearchCosine
    search_service: SearchService[UUID] = SearchService(repo, LinearSearchCosine)

    # Index the library
    chunks_indexed = await search_service.index_library(library.id)
    assert chunks_indexed == 3

    # Query for vectors similar to [1.0, 0.1, 0.1]
    # This should be closest to the first chunk
    results = await search_service.query(library.id, [1.0, 0.1, 0.1], k=3)

    # Verify results
    assert len(results) == 3
    assert results[0].text == "This is the first chunk"
    assert results[1].text in ["This is the second chunk", "This is the third chunk"]
    assert results[2].text in ["This is the second chunk", "This is the third chunk"]


@pytest.mark.asyncio
async def test_ball_tree_search() -> None:
    """Test BallTreeCosine search."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)

    # Create search service with BallTreeCosine
    search_service: SearchService[UUID] = SearchService(repo, BallTreeCosine)

    # Index the library
    chunks_indexed = await search_service.index_library(library.id)
    assert chunks_indexed == 3

    # Query for vectors similar to [0.1, 0.0, 1.0]
    # This should be closest to the third chunk
    results = await search_service.query(library.id, [0.1, 0.0, 1.0], k=3)

    # Verify results
    assert len(results) == 3
    assert results[0].text == "This is the third chunk"
    assert results[1].text in ["This is the first chunk", "This is the second chunk"]
    assert results[2].text in ["This is the first chunk", "This is the second chunk"]


@pytest.mark.asyncio
async def test_search_not_indexed() -> None:
    """Test searching a library that hasn't been indexed."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)

    # Create search service
    search_service: SearchService[UUID] = SearchService(repo, LinearSearchCosine)

    # Try to query without indexing first
    with pytest.raises(ValidationError):
        await search_service.query(library.id, [1.0, 0.0, 0.0], k=3)


@pytest.mark.asyncio
async def test_concurrent_index_search() -> None:
    """Test concurrent indexing and searching with asyncio tasks."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)

    # Create search service
    search_service: SearchService[UUID] = SearchService(repo, LinearSearchCosine)

    # Create tasks for concurrent indexing and searching
    async def index_task() -> int:
        return await search_service.index_library(library.id)

    async def search_task() -> List[Chunk]:
        # Small delay to ensure indexing starts first
        await asyncio.sleep(0.01)
        try:
            return await search_service.query(library.id, [1.0, 0.1, 0.1], k=3)
        except KeyError:
            return []
        except Exception as e:
            raise e

    # Run both tasks concurrently
    index_task_obj = asyncio.create_task(index_task())
    search_task_obj = asyncio.create_task(search_task())

    # Wait for both tasks to complete
    index_result = await index_task_obj
    search_results = await search_task_obj

    # Verify results
    assert index_result == 3
    assert isinstance(search_results, list)
    assert len(search_results) == 3
    if search_results:
        assert search_results[0].text == "This is the first chunk"


@pytest.mark.asyncio
async def test_reindex_library() -> None:
    """Test re-indexing a library after changes."""
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)

    search_service: SearchService[UUID] = SearchService(repo, LinearSearchCosine)

    # Initial indexing
    initial_chunks_indexed = await search_service.index_library(library.id)
    assert initial_chunks_indexed == 3

    # Query before re-indexing
    query_embedding = [0.9, 0.1, 0.0]  
    results_before_reindex = await search_service.query(library.id, query_embedding, k=1)
    assert len(results_before_reindex) == 1

    # Add a new chunk to the library that is very similar to the query
    new_chunk_doc_id = library.documents[0].id
    new_chunk = Chunk(
        id=uuid4(),
        document_id=new_chunk_doc_id, # Use the ID of the first document in the library
        text="This is a new very similar chunk",
        embedding=[0.95, 0.05, 0.0],
        metadata={"page": "new"},
    )
    library.documents[0].chunks.append(new_chunk)
    await repo.update_library(library.id, library) 

    # Re-index the library
    reindexed_chunks = await search_service.reindex_library(library.id)
    assert reindexed_chunks == 4 

    # Query after re-indexing
    results_after_reindex = await search_service.query(library.id, query_embedding, k=1)
    assert len(results_after_reindex) == 1
    assert results_after_reindex[0].id == new_chunk.id
