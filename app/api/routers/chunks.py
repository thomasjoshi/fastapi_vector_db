"""
Router for chunk operations.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.schemas.chunk import ChunkCreate, ChunkRead
from app.domain.models import Chunk
from app.services.chunk import ChunkService
from app.api.dependencies import get_chunk_service

router = APIRouter(tags=["chunks"])


@router.post(
    "/libraries/{library_id}/documents/{document_id}/chunks",
    response_model=ChunkRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_data: ChunkCreate,
    service: ChunkService = Depends(get_chunk_service),
):
    """Create a new chunk in a document."""
    chunk = Chunk(
        text=chunk_data.text,
        embedding=chunk_data.embedding,
        metadata=chunk_data.metadata,
    )
    result = await service.add_chunk(library_id, document_id, chunk)
    return ChunkRead(
        id=result.id,
        text=result.text,
        embedding=result.embedding,
        metadata=result.metadata,
    )


@router.get(
    "/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}",
    response_model=ChunkRead,
)
async def get_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
):
    """Get a chunk by ID."""
    result = await service.get_chunk(library_id, document_id, chunk_id)
    return ChunkRead(
        id=result.id,
        text=result.text,
        embedding=result.embedding,
        metadata=result.metadata,
    )


@router.put(
    "/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def update_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    chunk_data: ChunkCreate,
    service: ChunkService = Depends(get_chunk_service),
):
    """Update a chunk."""
    chunk = Chunk(
        text=chunk_data.text,
        embedding=chunk_data.embedding,
        metadata=chunk_data.metadata,
    )
    await service.update_chunk(library_id, document_id, chunk_id, chunk)


@router.delete(
    "/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
):
    """Delete a chunk."""
    await service.delete_chunk(library_id, document_id, chunk_id)
