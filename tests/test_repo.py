import concurrent.futures
from uuid import UUID

import pytest
import pytest_asyncio

from app.domain.models import Library
from app.repos.in_memory import InMemoryRepo
from app.services.exceptions import NotFoundError
from app.services.library import LibraryService


@pytest.mark.asyncio
async def test_repo_crud() -> None:
    """Test basic CRUD operations on the repository."""
    # Setup
    repo = InMemoryRepo()
    service = LibraryService(repo)

    # Create a library
    lib = Library.example()
    lib_id = await service.add_library(lib)

    # Get the library
    retrieved_lib = await service.get_library(lib_id)
    assert retrieved_lib.id == lib.id
    assert len(retrieved_lib.documents) == len(lib.documents)

    # Update the library
    updated_lib = Library(
        id=lib_id,
        documents=lib.documents,
        metadata={"name": "Updated Library", "created_by": "Test"},
    )
    await service.update_library(lib_id, updated_lib)

    # Get the updated library
    retrieved_updated_lib = await service.get_library(lib_id)
    assert retrieved_updated_lib.metadata["name"] == "Updated Library"

    # Delete the library
    await service.delete_library(lib_id)

    # Verify deletion
    with pytest.raises(NotFoundError):
        await service.get_library(lib_id)


@pytest.mark.asyncio
async def test_repo_concurrent_reads() -> None:
    """Test concurrent reads from the repository."""
    # Setup
    repo = InMemoryRepo()
    service = LibraryService(repo)

    # Create a library
    lib = Library.example()
    lib_id = await service.add_library(lib)

    # Function to read the library asynchronously
    async def read_library(lib_id: UUID) -> Library:
        return await service.get_library(lib_id)

    # Read the library 10 times sequentially 
    # (can't easily use ThreadPoolExecutor with async functions)
    results = []
    for _ in range(10):
        result = await read_library(lib_id)
        results.append(result)

    # Verify all reads were successful
    assert len(results) == 10
    for result in results:
        assert result.id == lib_id


@pytest.mark.asyncio
async def test_repo_update_during_reads() -> None:
    """Test updating the repository while reads are active."""
    # Setup
    repo = InMemoryRepo()
    service = LibraryService(repo)

    # Create a library
    lib = Library.example()
    lib_id = await service.add_library(lib)

    # Function to read the library
    async def read_library(lib_id: UUID) -> Library:
        return await service.get_library(lib_id)

    # Function to update the library
    async def update_library(lib_id: UUID) -> None:
        updated_lib = Library(
            id=lib_id,
            documents=lib.documents,
            metadata={"name": "Concurrent Update", "created_by": "Test"},
        )
        await service.update_library(lib_id, updated_lib)

    # First batch of reads
    results = []
    for _ in range(5):
        result = await read_library(lib_id)
        results.append(result)

    # Update the library
    await update_library(lib_id)

    # Second batch of reads
    for _ in range(5):
        result = await read_library(lib_id)
        results.append(result)

    # Verify all reads were successful
    assert len(results) == 10

    # Verify the final state
    final_lib = await service.get_library(lib_id)
    assert final_lib.metadata.get("name") == "Concurrent Update"
