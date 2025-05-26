"""
Unit tests for the LibraryService class.
"""
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import ANY, AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio

from app.domain.models import Library
from app.repos.in_memory import InMemoryRepo
from app.services.exceptions import NotFoundError
from app.services.library import LibraryService


class TestLibraryService:
    """Tests for the LibraryService class."""

    @pytest_asyncio.fixture
    async def mock_repo(self) -> AsyncMock:
        """Create a mock repository."""
        mock = AsyncMock(spec=InMemoryRepo)
        return mock

    @pytest.fixture
    def mock_metrics(self) -> MagicMock:
        """Create a mock metrics callback."""
        return MagicMock()

    @pytest.fixture
    def metrics_calls(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], Callable[..., None]]:
        """Track metrics calls."""
        calls: List[Tuple[str, Dict[str, Any]]] = []

        def metrics_callback(metric_name: str, **kwargs: Any) -> None:
            calls.append((metric_name, kwargs))

        return calls, metrics_callback

    @pytest.mark.asyncio
    async def test_add_library(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test adding a library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib = Library.example()
        mock_repo.add_library.return_value = lib.id

        # Act
        result = await service.add_library(lib)

        # Assert
        assert result == lib.id
        mock_repo.add_library.assert_called_once_with(lib)
        mock_metrics.assert_called_once_with(
            "library.add", library_id=str(lib.id), duration_ms=ANY
        )

    @pytest.mark.asyncio
    async def test_get_library(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test getting a library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib = Library.example()
        lib_id = lib.id
        mock_repo.get_library.return_value = lib

        # Act
        result = await service.get_library(lib_id)

        # Assert
        assert result == lib
        mock_repo.get_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with(
            "library.get", library_id=str(lib_id), duration_ms=ANY
        )

    @pytest.mark.asyncio
    async def test_get_library_not_found(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test getting a non-existent library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib_id = uuid4()
        mock_repo.get_library.side_effect = NotFoundError(
            f"Library with ID {lib_id} not found"
        )

        # Act & Assert
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_library(lib_id)

        # Verify error details
        assert "Library" in str(excinfo.value)
        assert str(lib_id) in str(excinfo.value)
        mock_repo.get_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with(
            "library.get_not_found", library_id=str(lib_id), duration_ms=ANY
        )

    @pytest.mark.asyncio
    async def test_update_library_atomic(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test updating a library with atomic operation."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib = Library.example()
        lib_id = lib.id
        mock_repo.update_library_if_exists.return_value = True

        # Act
        await service.update_library(lib_id, lib)

        # Assert
        mock_repo.update_library_if_exists.assert_called_once_with(lib_id, lib)
        mock_metrics.assert_called_once_with(
            "library.update", library_id=str(lib_id), duration_ms=ANY
        )
        # Ensure we didn't need to fall back to the existence check
        mock_repo.get_library.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_library_not_found(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test updating a non-existent library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib_id = uuid4()
        lib = Library.example(id=lib_id)
        # Simulate that update_library_if_exists returns False (library not found)
        mock_repo.update_library_if_exists.return_value = False
        # Set up get_library to raise NotFoundError
        mock_repo.get_library.side_effect = NotFoundError(
            f"Library with ID {lib_id} not found"
        )

        # Act & Assert
        with pytest.raises(NotFoundError) as excinfo:
            await service.update_library(lib_id, lib)

        # Verify error details
        assert "Library" in str(excinfo.value)
        assert str(lib_id) in str(excinfo.value)
        mock_repo.update_library_if_exists.assert_called_once_with(lib_id, lib)
        mock_repo.get_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with(
            "library.update_not_found", library_id=str(lib_id), duration_ms=ANY
        )

    @pytest.mark.asyncio
    async def test_delete_library(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test deleting a library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib_id = uuid4()
        mock_repo.delete_library.return_value = True

        # Act
        await service.delete_library(lib_id)

        # Assert
        mock_repo.delete_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with(
            "library.delete", library_id=str(lib_id), duration_ms=ANY
        )

    @pytest.mark.asyncio
    async def test_delete_library_not_found(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test deleting a non-existent library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib_id = uuid4()

        # Configure delete_library to raise error
        mock_repo.delete_library.side_effect = NotFoundError(
            f"Library with ID {lib_id} not found"
        )

        # Act & Assert
        with pytest.raises(NotFoundError) as excinfo:
            await service.delete_library(lib_id)

        # Verify error details
        assert "Library" in str(excinfo.value)
        assert str(lib_id) in str(excinfo.value)
        mock_repo.delete_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with(
            "library.delete_not_found", library_id=str(lib_id), duration_ms=ANY
        )

    @pytest.mark.asyncio
    async def test_metrics_callback_integration(
        self,
        metrics_calls: Tuple[List[Tuple[str, Dict[str, Any]]], Callable[..., None]],
    ) -> None:
        """Test that metrics callback is correctly invoked."""
        # Arrange
        calls, callback = metrics_calls
        repo = InMemoryRepo()
        service = LibraryService(repo, callback)
        lib = Library.example()

        # Act
        await service.add_library(lib)

        # Assert
        assert len(calls) == 1
        assert calls[0][0] == "library.add"
        assert calls[0][1]["library_id"] == str(lib.id)

    @pytest.mark.asyncio
    async def test_add_library_duplicate_id(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test adding a library with a duplicate ID."""
        service = LibraryService(mock_repo, mock_metrics)
        lib = Library.example()
        mock_repo.add_library.side_effect = ValueError(
            f"Library {lib.id} already exists"
        )

        with pytest.raises(ValueError, match=f"Library {lib.id} already exists"):
            await service.add_library(lib)

    # @pytest.mark.asyncio
    # async def test_list_libraries(
    #     self, mock_repo: AsyncMock, mock_metrics: MagicMock
    # ) -> None:
    #     """Test listing libraries."""
    #     service = LibraryService(mock_repo, mock_metrics)
    #     libraries_example = [Library.example(), Library.example()]
    #     # Assuming the repo method is list_libraries and service calls it directly
    #     mock_repo.list_libraries.return_value = libraries_example

    #     result = await service.list_libraries(skip=0, limit=10)

    #     assert result == libraries_example
    #     mock_repo.list_libraries.assert_called_once_with(skip=0, limit=10)

    @pytest.mark.asyncio
    async def test_update_library(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test updating an existing library (verifies call, no return value)."""
        service = LibraryService(mock_repo, mock_metrics)
        lib_update = Library.example()
        # LibraryService.update_library calls repo.update_library_if_exists
        mock_repo.update_library_if_exists.return_value = True

        # Simulate successful update
        await service.update_library(lib_update.id, lib_update)

        # Assert that the correct repo method was called
        mock_repo.update_library_if_exists.assert_called_once_with(
            lib_update.id, lib_update
        )
        # Optionally, assert metrics call if that's part of the contract
        # for this specific test focus
        # mock_metrics.assert_any_call("library.update", duration_ms=ANY) # Example

    @pytest.mark.asyncio
    async def test_delete_library_with_documents(
        self, mock_repo: AsyncMock, mock_metrics: MagicMock
    ) -> None:
        """Test deleting a library that contains documents."""
        service = LibraryService(mock_repo, mock_metrics)
        library_id = uuid4()
        # Simulate that the repo's delete_library raises a ValueError if documents exist
        mock_repo.delete_library.side_effect = ValueError(
            f"Cannot delete library {library_id} as it contains documents."
        )

        with pytest.raises(ValueError):
            await service.delete_library(library_id)
