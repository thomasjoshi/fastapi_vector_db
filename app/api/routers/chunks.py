"""
Router for chunk operations.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_chunk_service
from app.api.schemas.chunk import ChunkCreate, ChunkCreateWithEmbedding, ChunkRead
from app.core.embeddings import get_embedding_generator
from app.domain.models import Chunk
from app.services.chunk import ChunkService

router = APIRouter(tags=["chunks"])


# Create dependency at module level to avoid B008 lint issue
chunk_service_dep = Depends(get_chunk_service)

@router.post(
    "/libraries/{library_id}/documents/{document_id}/chunks/embed",
    response_model=ChunkRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_chunk_with_embedding(
    library_id: UUID,
    document_id: UUID,
    chunk_data: ChunkCreateWithEmbedding,
    service: ChunkService = chunk_service_dep,
) -> ChunkRead:
    """Create chunk with auto-generated embedding via Cohere API.
    
    Args:
        library_id: Library ID
        document_id: Document ID
        chunk_data: Chunk data with text and metadata
        service: Chunk service
        
    Returns:
        Created chunk with embedding
        
    Raises:
        HTTPException: On embedding failure or invalid IDs
    """
    # Generate embedding using Cohere API if requested
    embedding = []
    if chunk_data.generate_embedding:
        try:
            embedding_generator = get_embedding_generator()
            embedding = await embedding_generator.generate_embedding(
                chunk_data.text
            )
            
            if not embedding:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to generate embedding. Empty result.",
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {str(e)}",
            ) from e
    
    # Create the chunk with the generated embedding
    chunk = Chunk(
        text=chunk_data.text,
        embedding=embedding,
        metadata=chunk_data.metadata,
    )
    
    try:
        result = await service.add_chunk(library_id, document_id, chunk)
        return ChunkRead(
            id=result.id,
            text=result.text,
            embedding=result.embedding,
            metadata=result.metadata,
        )
    except Exception:
        # Let FastAPI's exception handlers handle service-specific exceptions
        raise


@router.post(
    "/libraries/{library_id}/documents/{document_id}/chunks",
    response_model=ChunkRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_data: ChunkCreate,
    service: ChunkService = chunk_service_dep,
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
    service: ChunkService = chunk_service_dep,
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
    service: ChunkService = chunk_service_dep,
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
    service: ChunkService = chunk_service_dep,
):
    """Delete a chunk."""
    await service.delete_chunk(library_id, document_id, chunk_id)
