"""
Router for document operations.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_document_service
from app.api.schemas.document import DocumentCreate, DocumentRead
from app.domain.models import Document
from app.services.document import DocumentService

router = APIRouter(tags=["documents"])


@router.post(
    "/libraries/{library_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    library_id: UUID,
    document_data: DocumentCreate,
    service: DocumentService = Depends(get_document_service),
):
    """Create a new document in a library."""
    from loguru import logger

    logger.info(
        f"Creating document in library with ID {library_id}, type: {type(library_id)}"
    )

    # Create a document with metadata from the request
    document = Document(chunks=[], metadata=document_data.metadata)
    
    # Add the document to the library
    result = await service.add_document(library_id, document)
    
    # Return the document with consistent ID formatting
    return DocumentRead(id=result.id, metadata=result.metadata)


@router.get(
    "/libraries/{library_id}/documents/{document_id}",
    response_model=DocumentRead,
)
async def get_document(
    library_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
):
    """Get a document by ID."""
    result = await service.get_document(library_id, document_id)
    return DocumentRead(id=result.id, metadata=result.metadata)


@router.put(
    "/libraries/{library_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def update_document(
    library_id: UUID,
    document_id: UUID,
    document_data: DocumentCreate,
    service: DocumentService = Depends(get_document_service),
):
    """Update a document."""
    document = Document(chunks=[], metadata=document_data.metadata)
    await service.update_document(library_id, document_id, document)


@router.delete(
    "/libraries/{library_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    library_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
):
    """Delete a document."""
    await service.delete_document(library_id, document_id)
