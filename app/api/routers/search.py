"""
Router for search operations.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_search_service
from app.api.schemas.chunk import (
    ChunkRead,
    IndexResponse,
    SearchHit,
    SearchQuery,
    SearchResponse,
)
from app.services.exceptions import NotFoundError, ValidationError
from app.services.search import SearchService

# Create router
router = APIRouter(tags=["search"])


@router.post("/libraries/{library_id}/index", response_model=IndexResponse)
async def index_library(
    library_id: UUID,
    service: SearchService = Depends(get_search_service),
) -> IndexResponse:
    """
    Index a library for vector search.

    This operation builds a vector index for the library,
    enabling efficient similarity search.
    """
    try:
        chunks_indexed = await service.index_library(library_id)
        return IndexResponse(chunks_indexed=chunks_indexed)
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e


@router.post("/libraries/{library_id}/search", response_model=SearchResponse)
async def search_library(
    library_id: UUID,
    query: SearchQuery,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Search for similar chunks in a library.

    Returns a list of chunks sorted by similarity to the query vector.
    """
    try:
        results = await service.search(
            library_id, 
            query.embedding, 
            query.k,
            query.metadata_filters if query.metadata_filters else None
        )

        # Convert to SearchResponse schema
        hits = []
        for chunk, score in results:
            chunk_read = ChunkRead(
                id=chunk.id,
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
            )
            hits.append(SearchHit(chunk=chunk_read, score=score))

        return SearchResponse(hits=hits)
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
