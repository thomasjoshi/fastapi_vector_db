"""
Tests for search functionality.
"""

import pytest
import threading
import time
from uuid import UUID, uuid4
from typing import List, Dict, Any

from app.domain.models import Library, Document, Chunk
from app.repos.in_memory import InMemoryRepo
from app.services.search import SearchService
from app.indexing.brute import BruteForceCosine
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


def test_brute_force_search():
    """Test BruteForceCosine search."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    repo.add_library(library)
    
    # Create search service with BruteForceCosine
    search_service = SearchService(repo, BruteForceCosine)
    
    # Index the library
    chunks_indexed = search_service.index_library(library.id)
    assert chunks_indexed == 3
    
    # Query for vectors similar to [1.0, 0.1, 0.1]
    # This should be closest to the first chunk
    results = search_service.query(library.id, [1.0, 0.1, 0.1], k=3)
    
    # Verify results
    assert len(results) == 3
    assert results[0].text == "This is the first chunk"
    assert results[1].text in ["This is the second chunk", "This is the third chunk"]
    assert results[2].text in ["This is the second chunk", "This is the third chunk"]


def test_ball_tree_search():
    """Test BallTreeCosine search."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    repo.add_library(library)
    
    # Create search service with BallTreeCosine
    search_service = SearchService(repo, BallTreeCosine)
    
    # Index the library
    chunks_indexed = search_service.index_library(library.id)
    assert chunks_indexed == 3
    
    # Query for vectors similar to [0.1, 0.0, 1.0]
    # This should be closest to the third chunk
    results = search_service.query(library.id, [0.1, 0.0, 1.0], k=3)
    
    # Verify results
    assert len(results) == 3
    assert results[0].text == "This is the third chunk"
    assert results[1].text in ["This is the first chunk", "This is the second chunk"]
    assert results[2].text in ["This is the first chunk", "This is the second chunk"]


def test_search_not_indexed():
    """Test searching a library that hasn't been indexed."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    repo.add_library(library)
    
    # Create search service
    search_service = SearchService(repo, BruteForceCosine)
    
    # Try to query without indexing first
    with pytest.raises(Exception):
        search_service.query(library.id, [1.0, 0.0, 0.0], k=3)


def test_concurrent_index_search():
    """Test concurrent indexing and searching."""
    # Create repo and add a library
    repo = InMemoryRepo()
    library = create_test_library()
    repo.add_library(library)
    
    # Create search service
    search_service = SearchService(repo, BruteForceCosine)
    
    # Variables for thread communication
    index_done = threading.Event()
    search_results = []
    search_error = None
    
    def index_thread():
        """Thread function for indexing."""
        search_service.index_library(library.id)
        index_done.set()
    
    def search_thread():
        """Thread function for searching."""
        nonlocal search_results, search_error
        
        # Wait a bit to ensure indexing has started
        time.sleep(0.01)
        
        try:
            # Try to search while indexing might still be in progress
            results = search_service.query(library.id, [1.0, 0.0, 0.0], k=3)
            search_results = results
        except Exception as e:
            search_error = e
    
    # Start the threads
    t1 = threading.Thread(target=index_thread)
    t2 = threading.Thread(target=search_thread)
    
    t1.start()
    t2.start()
    
    # Wait for both threads to finish
    t1.join()
    t2.join()
    
    # Verify results
    if search_error is None:
        # If search succeeded, verify the results
        assert len(search_results) == 3
        assert search_results[0].text == "This is the first chunk"
    else:
        # If search failed, it should be because the library wasn't indexed yet
        assert "not indexed" in str(search_error)
