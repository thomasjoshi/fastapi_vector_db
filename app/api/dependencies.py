"""
Dependency injection for FastAPI.
"""

from fastapi import Depends

from app.repos.in_memory import InMemoryRepo
from app.services.chunk import ChunkService
from app.services.document import DocumentService
from app.services.library import LibraryService, MetricsCallback, noop_metrics_callback
from app.services.search import SearchService

# Singleton instances for the application lifetime
_repo = InMemoryRepo()
_metrics_callback = noop_metrics_callback


def get_repo() -> InMemoryRepo:
    """
    Dependency that provides the repository instance.
    Returns a singleton repository instance for the entire application.

    Returns:
        An instance of InMemoryRepo
    """
    return _repo


def get_metrics_callback() -> MetricsCallback:
    """
    Dependency that provides the metrics callback.

    Returns:
        A metrics callback function
    """
    return _metrics_callback


def get_library_service(repo: InMemoryRepo = Depends(get_repo)) -> LibraryService:
    """
    Get a library service instance.

    Args:
        repo: The repository to use

    Returns:
        An instance of LibraryService
    """
    return LibraryService(repo, _metrics_callback)


def get_document_service(repo: InMemoryRepo = Depends(get_repo)) -> DocumentService:
    """
    Get a document service instance.

    Args:
        repo: The repository to use

    Returns:
        An instance of DocumentService
    """
    return DocumentService(repo, _metrics_callback)


def get_chunk_service(repo: InMemoryRepo = Depends(get_repo)) -> ChunkService:
    """
    Get a chunk service instance.

    Args:
        repo: The repository to use

    Returns:
        An instance of ChunkService
    """
    return ChunkService(repo, _metrics_callback)


def get_search_service(repo: InMemoryRepo = Depends(get_repo)) -> SearchService:
    """
    Get a search service instance.

    Args:
        repo: The repository to use

    Returns:
        An instance of SearchService
    """
    # SearchService doesn't use metrics callback, it needs a vector index observer with a different signature
    return SearchService(repo)
