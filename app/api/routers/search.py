"""
Router for search operations.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_library_service
from app.api.schemas.chunk import ChunkRead, IndexResponse, SearchQuery
from app.indexing.brute import BruteForceCosine
from app.services.errors import NotFoundError
from app.services.library import LibraryService
from app.services.search import SearchService


# Create the search service with BruteForce index by default
# In a real application, this would be configurable
def get_search_service(library_service: LibraryService = None) -> SearchService:
    """Get the search service."""
    # Use dependency injection to get the library service
    if library_service is None:
        library_service = get_library_service()
    # Use the same repo as the library service
    repo = library_service._repo
    # Use BruteForce index by default
    return SearchService(repo, BruteForceCosine)


# Create router
router = APIRouter(prefix="/libraries", tags=["search"])


@router.post("/{library_id}/index", response_model=IndexResponse)
async def index_library(
    library_id: UUID,
    search_service: SearchService = None,
) -> IndexResponse:
    # Use dependency injection
    if search_service is None:
        search_service = get_search_service()
    """
    Index a library for vector search.
    
    This operation builds a vector index for the library,
    enabling efficient similarity search.
    """
    try:
        # Run in threadpool to avoid blocking
        chunks_indexed = await run_in_threadpool(
            search_service.index_library, library_id
        )
        return IndexResponse(chunks_indexed=chunks_indexed)
    except NotFoundError as e:
        # Re-raise as HTTPException
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("/{library_id}/search", response_model=List[ChunkRead])
async def search_library(
    library_id: UUID,
    query: SearchQuery,
    search_service: SearchService = None,
) -> List[ChunkRead]:
    # Use dependency injection
    if search_service is None:
        search_service = get_search_service()
    """
    Search for similar chunks in a library.
    
    Returns a list of chunks sorted by similarity to the query vector.
    """
    try:
        # Run in threadpool to avoid blocking
        chunks = await run_in_threadpool(
            search_service.query, library_id, query.embedding, query.k
        )
        # Convert to ChunkRead schema
        return [
            ChunkRead(
                id=chunk.id,
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
    except NotFoundError as e:
        # Re-raise as HTTPException
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
