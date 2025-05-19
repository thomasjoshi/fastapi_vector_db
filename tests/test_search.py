"""
Tests for search functionality.
"""

import pytest
import threading
import time
import asyncio
from uuid import UUID, uuid4
from typing import List, Dict, Any

from app.domain.models import Library, Document, Chunk
from app.repos.in_memory import InMemoryRepo
from app.services.search import SearchService
from app.indexing.linear_search import LinearSearchCosine
from app.indexing.ball_tree import BallTreeCosine


def create_test_library() -> Library:
    """Create a test library with documents and chunks."""
    # Create chunks with different embeddings
    chunk1 = Chunk(
        id=uuid4(),
        text="This is the first chunk",
        embedding=[1.0, 0.0, 0.0],  # pointing along x-axis
        metadata={"page": "1"},
    )
    
    chunk2 = Chunk(
        id=uuid4(),
        text="This is the second chunk",
        embedding=[0.0, 1.0, 0.0],  # pointing along y-axis
        metadata={"page": "2"},
    )
    
    chunk3 = Chunk(
        id=uuid4(),
        text="This is the third chunk",
        embedding=[0.0, 0.0, 1.0],  # pointing along z-axis
        metadata={"page": "3"},
    )
    
    # Create a document with these chunks
    document = Document(
        id=uuid4(),
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
async def test_brute_force_search():
    """Test LinearSearchCosine search."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)
    
    # Create search service with LinearSearchCosine
    search_service = SearchService(repo, LinearSearchCosine)
    
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
async def test_ball_tree_search():
    """Test BallTreeCosine search."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)
    
    # Create search service with BallTreeCosine
    search_service = SearchService(repo, BallTreeCosine)
    
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
async def test_search_not_indexed():
    """Test searching a library that hasn't been indexed."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)
    
    # Create search service
    search_service = SearchService(repo, LinearSearchCosine)
    
    # Try to query without indexing first
    with pytest.raises(Exception):
        await search_service.query(library.id, [1.0, 0.0, 0.0], k=3)


@pytest.mark.asyncio
async def test_concurrent_index_search():
    """Test concurrent indexing and searching with asyncio tasks."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    await repo.add_library(library)
    
    # Create search service
    search_service = SearchService(repo, LinearSearchCosine)
    
    # Create tasks for concurrent indexing and searching
    async def index_task():
        return await search_service.index_library(library.id)
    
    async def search_task():
        # Small delay to ensure indexing starts first
        await asyncio.sleep(0.01)
        try:
            return await search_service.query(library.id, [1.0, 0.1, 0.1], k=3)
        except Exception as e:
            return e
    
    # Run both tasks concurrently
    index_task_obj = asyncio.create_task(index_task())
    search_task_obj = asyncio.create_task(search_task())
    
    # Wait for both tasks to complete
    index_result = await index_task_obj
    search_results = await search_task_obj
    
    # Verify results
    assert index_result == 3  # Number of chunks indexed
    assert isinstance(search_results, list)  # Should not be an exception
    assert len(search_results) == 3
    assert search_results[0].text == "This is the first chunk"
