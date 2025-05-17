import concurrent.futures
from uuid import UUID

import pytest

from app.domain.models import Library
from app.repos.in_memory import InMemoryRepo
from app.services.errors import NotFoundError
from app.services.library import LibraryService


def test_repo_crud() -> None:
    """Test basic CRUD operations on the repository."""
    # Setup
    repo = InMemoryRepo()
    service = LibraryService(repo)

    # Create a library
    lib = Library.example()
    lib_id = service.add_library(lib)

    # Get the library
    retrieved_lib = service.get_library(lib_id)
    assert retrieved_lib.id == lib.id
    assert len(retrieved_lib.documents) == len(lib.documents)

    # Update the library
    updated_lib = Library(
        id=lib_id,
        documents=lib.documents,
        metadata={"name": "Updated Library", "created_by": "Test"},
    )
    service.update_library(lib_id, updated_lib)

    # Get the updated library
    retrieved_updated_lib = service.get_library(lib_id)
    assert retrieved_updated_lib.metadata["name"] == "Updated Library"

    # Delete the library
    service.delete_library(lib_id)

    # Verify deletion
    with pytest.raises(NotFoundError):
        service.get_library(lib_id)


def test_repo_concurrent_reads() -> None:
    """Test concurrent reads from the repository."""
    # Setup
    repo = InMemoryRepo()
    service = LibraryService(repo)

    # Create a library
    lib = Library.example()
    lib_id = service.add_library(lib)

    # Function to read the library
    def read_library(lib_id: UUID) -> Library:
        return service.get_library(lib_id)

    # Read the library concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(read_library, lib_id) for _ in range(10)]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    # Verify all reads were successful
    assert len(results) == 10
    for result in results:
        assert result.id == lib_id


def test_repo_update_during_reads() -> None:
    """Test updating the repository while reads are active."""
    # Setup
    repo = InMemoryRepo()
    service = LibraryService(repo)

    # Create a library
    lib = Library.example()
    lib_id = service.add_library(lib)

    # Function to read the library
    def read_library(lib_id: UUID) -> Library:
        return service.get_library(lib_id)

    # Function to update the library
    def update_library(lib_id: UUID) -> None:
        updated_lib = Library(
            id=lib_id,
            documents=lib.documents,
            metadata={"name": "Concurrent Update", "created_by": "Test"},
        )
        service.update_library(lib_id, updated_lib)

    # Start concurrent reads
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        read_futures = [executor.submit(read_library, lib_id) for _ in range(5)]

        # Update the library while reads are active
        update_future = executor.submit(update_library, lib_id)
        update_future.result()  # Wait for update to complete

        # Continue with more reads
        more_read_futures = [executor.submit(read_library, lib_id) for _ in range(5)]

        # Collect all results
        all_futures = read_futures + more_read_futures
        results = [
            future.result() for future in concurrent.futures.as_completed(all_futures)
        ]

    # Verify all reads were successful
    assert len(results) == 10

    # Verify the final state
    final_lib = service.get_library(lib_id)
    assert final_lib.metadata.get("name") == "Concurrent Update"
