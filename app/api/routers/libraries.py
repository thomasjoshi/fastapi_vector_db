from uuid import UUID

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_library_service
from app.api.schemas.library import (
    LibraryCreate,
    LibraryRead,
    LibraryUpdate,
)
from app.domain.models import Library
from app.services.library import LibraryService

# Define dependency to avoid B008 linting errors
get_service = Depends(get_library_service)

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=LibraryRead)
async def create_library(
    library: LibraryCreate,
    response: Response,
    library_service: LibraryService = get_service,
) -> LibraryRead:
    """
    Create a new library.
    Returns the created library with its ID and sets the Location header.
    """
    # Convert from API schema to domain model
    domain_library = Library(
        documents=library.documents,
        metadata=library.metadata,
    )

    # Call the async service method directly
    library_id = await library_service.add_library(domain_library)

    # Set Location header for REST best practices
    response.headers["Location"] = f"/libraries/{library_id}"

    # Get the complete library after creation
    created_library = await library_service.get_library(library_id)

    # Return the library with its ID
    return LibraryRead(
        id=created_library.id,
        documents=created_library.documents,
        metadata=created_library.metadata,
    )


@router.get("/{library_id}", response_model=LibraryRead)
async def get_library(
    library_id: UUID,
    library_service: LibraryService = get_service,
) -> LibraryRead:
    """
    Get a library by ID.
    Raises 404 if the library does not exist.
    """
    # Call the async service method directly
    domain_library = await library_service.get_library(library_id)

    # Convert from domain model to API schema
    return LibraryRead(
        id=domain_library.id,
        documents=domain_library.documents,
        metadata=domain_library.metadata,
    )


@router.put("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_library(
    library_id: UUID,
    library: LibraryUpdate,
    library_service: LibraryService = get_service,
) -> None:
    """
    Update a library.
    Raises 404 if the library does not exist.
    """
    # First get the existing library to ensure it exists
    domain_library = await library_service.get_library(library_id)

    # Update with new values
    documents = (
        library.documents if library.documents is not None else domain_library.documents
    )
    metadata = (
        library.metadata if library.metadata is not None else domain_library.metadata
    )

    updated_library = Library(
        id=domain_library.id,
        documents=documents,
        metadata=metadata,
    )

    # Call the async service method directly
    await library_service.update_library(library_id, updated_library)


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(
    library_id: UUID,
    library_service: LibraryService = get_service,
) -> None:
    """
    Delete a library by ID.
    Raises 404 if the library does not exist.
    """
    # Call the async service method directly
    await library_service.delete_library(library_id)
